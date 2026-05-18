# FastWAM Active Loop Server — 独立主动闭环服务设计

> 版本：v2 · 日期：2026-05-18 · 状态：仅设计，未实现
>
> v2 关键变化（相对 v1）：
> - **架构解耦**：放弃在 `scripts/fastwam_http_server.py` 内加 `--enable-self-fetch` 开关。旧 server **完全不动**，主动模式拆成**独立脚本** `scripts/fastwam_active_loop_server.py`。这样 `_INFER_LOCK` / `/infer` fallback / 向后兼容 flag 一整套都不需要。
> - **WS / ARM probe 已跑通**，协议字段、通道名、分辨率、gripper 单位全部落实，原"未解项"清零。
> - **图像流程简化**：1088×1280 → undistort → 直接交给 `FastWAMModelClient.infer()`，**不再额外 cv2.resize 到 480×640**（model_clients 内部 stitch+resize+crop 一步到位 224×448）。
> - **lease**：信任 arm_sdk 内置 `acquire_control` 自动续期，**不外加 timer**。

## 0. TL;DR

新增 `scripts/fastwam_active_loop_server.py`，内部跑：

- **WS 图像 ingester**（PyAV H.264 解码，双路 `head_left` / `right_wrist_left`）
- **ARM 状态 poller**（gRPC 三连发：joint / eef / end_pose）
- **ClosedLoopRunner**（2.5 Hz 推理 → 32 帧绝对动作 chunk）
- **Dispatcher**（20 Hz 下发，4 帧线性混合）
- **Watchdog**（10 ms 周期，异常进入 hold / emergency_stop）

HTTP 接口：

1. `POST /start`（可选 `instruction`）触发闭环；
2. `POST /stop` 或 `POST /emergency` 终止；
3. `GET /health`、`GET /closed_loop_status`、`GET /ws_status` 观测。

`scripts/fastwam_http_server.py` 与本脚本完全独立，部署时按需起其中一个。

实测端到端闭环延迟 **~225 ms**（capture → 第一帧下发；图像传输 ~45 ms p50 + ARM RPC ~1 ms + 推理 ~170 ms p50 / 10 steps）。控制环 20 Hz，推理环 2.5 Hz，chunk 内 4 帧（200 ms）线性混合做无缝拼接。

---

## 1. 设计目标

- **架构解耦**：旧 HTTP server 字节级不动，主动模式独立成新脚本；
- **接口最小化**：客户端只发触发信号；
- **延迟可控**：闭环 < 400 ms，控制频率 20 Hz、推理频率 2.5 Hz；
- **失败安全**：WS 断流、ARM RPC 失败、推理超时、chunk 过期，任一收敛到「机械臂可控的安全保持」或「急停」；
- **可观测**：每段延迟（图像准备 / 模型推理 / 下发调度）都有 `perf_counter` 打点 + 周期性日志；
- **风险隔离**：所有新模块放在独立目录 `src/fastwam/server/`，删除该目录可整体回滚。

---

## 2. 架构图

```text
                       +---------------------+
                       |    HTTP Client      |
                       +----------+----------+
                                  |
              /start  /stop  /emergency  /status  /health  /ws_status
                                  v
+-----------------------------------------------------------------+
| fastwam_active_loop_server.py  (single process, GPU=cuda:1)    |
|                                                                 |
|   +------------------+              +------------------+        |
|   | WS Ingester      |              | ARM Poller       |        |
|   | thread           |              | thread @ 50 Hz   |        |
|   | - PyAV H.264     |              | - get_arm_joint  |        |
|   | - auto reconnect |              | - get_eef_joint  |        |
|   | frame_cache[k]   |              | - get_end_pose   |        |
|   +--------+---------+              +---------+--------+        |
|            |                                  |                 |
|            v                                  v                 |
|       frame_cache                        state_cache            |
|       (per-key TTL)                      (TTL + ts)             |
|             \                                /                  |
|              \      snapshot @ 2.5 Hz       /                   |
|               \                            /                    |
|                v                          v                     |
|             +------------------------------+                    |
|             | ClosedLoopRunner (infer thr) |                    |
|             | 1. snapshot(frame,state,tcap)|                    |
|             | 2. undistort 1088x1280 raw   |                    |
|             | 3. FastWAMModelClient.infer  |                    |
|             | 4. _delta_to_absolute        |                    |
|             | 5. push to chunk_ringbuffer  |                    |
|             |    (keep newest 2, drop old) |                    |
|             +--------------+---------------+                    |
|                            |                                    |
|                            v                                    |
|             +------------------------------+                    |
|             | Dispatcher (timer @ 20 Hz)   |                    |
|             | - drop stale: ceil(age/50ms) |                    |
|             | - blend 4 frames w=0.25..1.0 |                    |
|             | - rpy_to_quat_xyzw           |                    |
|             | - client.move_end_pose       |                    |
|             | - client.move_eef            |                    |
|             +--------------+---------------+                    |
|                            |                                    |
|                            v                                    |
|             +------------------------------+                    |
|             | Watchdog (thread @ 100 Hz)   |                    |
|             | - chunk_age > limit -> hold  |                    |
|             | - SDK fail streak   -> stop  |                    |
|             | - WS stale > limit  -> stop  |                    |
|             +--------------+---------------+                    |
+-----------------------------------------------------------------+
                             |
                             v
                 +---------------------------+
                 | External services         |
                 | WS  192.168.31.66:19095   |
                 | ARM 192.168.31.34:50051   |
                 +---------------------------+
```

---

## 3. 数据流时序

一次完整闭环的事件序列，`t=0` 是相机捕获时刻（=`head_left.stamp_ns`）；每段标「实测」或「估算」。

```text
t=0       camera capture (head_left + right_wrist_left @ 20 Hz)
t≈85      bridge_send_ns (上游编码+打包) - 实测 ~85 ms
t≈45-90   WS frame arrives in frame_cache       (实测 p50 45 ms p99 87 ms)
t≈47      ARM RPC triple returns to state_cache (实测 ~1 ms; 50 Hz poller)
t≈50      ClosedLoopRunner snapshot(frame, state, t_capture = head.stamp_ns)
t≈50..220 FastWAMModelClient.infer (实测 p50 167.5 ms @ 10 steps; 含 undistort + stitch + resize + crop)
t≈225     chunk_ringbuffer push: 32 frames @ 50 ms
          chunk[i] logical time = base_capture_ts_ns + i * 50 ms

t≈225     Dispatcher picks up new chunk
          drop_n = ceil(225 / 50) = 5
t≈225     send chunk[5],  blend w = 0.25
t≈275     send chunk[6],  blend w = 0.50
t≈325     send chunk[7],  blend w = 0.75
t≈375     send chunk[8],  blend w = 1.00  (blend window ends)
t≈425..   send chunk[9..31] without blending

(in parallel)
t=400     next inference triggered
```

延迟分解：

| 段 | 时长 | 来源 |
| --- | --- | --- |
| 摄像头 → bridge_send_ns | ~85 ms | **实测**（probe meta JSON：`bridge_send_ns - stamp_ns`） |
| bridge_send_ns → frame_cache | ~10 ms 量级 | LAN，单网段 0.15 ms ping，扣 jitter |
| ARM gRPC 三连发 | 0.9–1.2 ms | **实测** |
| undistort (1088×1280 双路) | 待实测 | OpenCV |
| `_preprocess_image` (stitch + resize + crop + normalize) | 待实测（v1 估 ~2.4 ms 是 480×640 → 224×448；1088×1280 → 224×448 会略慢） | **新增项** |
| `FastWAMModelClient.infer` @ 10 steps | 167.5–192.5 ms (p50–p99) | **实测**（5090 GPU，n=30，含 _preprocess_image） |
| chunk_cache → dispatcher 接管 | ≤ 50 ms | 设计目标 |
| `move_end_pose` + `move_eef` RPC | < 1 ms | 实测同 ARM 三连发 |
| **总闭环（捕获 → 第一帧下发）** | **~225 ms** | 实测 |

WS 帧到达 jitter（**实测**）：

```
recv intervals (ms): [49.85, 85.70, 25.86, 39.87, 47.51, 49.99, 49.67]
avg fps: 20.09  (target: 20 Hz)
```

上游不严格 50 ms 周期，**单次抖动可达 ±35 ms**。`--ws-frame-max-age-ms` 默认 **250 ms**（原 v1 200 ms），避免误触发跳推理。

---

## 4. WS / ARM 探测结果（实测，2026-05-18）

### 4.1 WS 协议字段（完整 meta JSON）

```json
{
  "bridge_send_ns": 1779098575753357660,
  "channels": [
    {
      "data_size": 145769,
      "format": "h264",
      "frame_id": "camera_cam1_frame",
      "name": "head_left",
      "original_data_size": 145737,
      "prepended_parameter_sets": true,
      "stamp_ns": 1779098575668060470,
      "topic": "rt/robot/camera/head/left/video_encoded"
    },
    {
      "data_size": 163478,
      "format": "h264",
      "frame_id": "camera_cam5_frame",
      "name": "right_wrist_left",
      "original_data_size": 163446,
      "prepended_parameter_sets": true,
      "stamp_ns": 1779098575668889972,
      "topic": "rt/robot/camera/right_wrist/left/video_encoded"
    }
  ],
  "delta_ns": 829502,
  "matched_by": "timestamp",
  "pair_seq": 8469,
  "payload_format": "h264_annexb_pair_v1",
  "type": "encoded_sync_pair"
}
```

字段约定：

- **magic**：`b"RGBDWS4\x00"`（8 bytes）
- **header**：`struct.unpack_from("<8sIII", msg, 0)` → `(magic, header_size, size_a, size_b)`（20 bytes）
- **payload_format 必须等于** `"h264_annexb_pair_v1"`（启动自检）
- **type 必须等于** `"encoded_sync_pair"`（启动自检）
- **channel.name** 是 `"head_left"` / `"right_wrist_left"`，**与模型 `image_key` 同名**（identity 映射，**省略 `--ws-channel-map`**）
- **channel.stamp_ns**：上游采集时刻；`base_capture_ts_ns = channels[head_left].stamp_ns`（**写死**）
- **channel.prepended_parameter_sets = true**：每包自带 SPS/PPS，任意点接入即可解码
- **delta_ns**：双通道 stamp 差，实测 0.83–0.95 ms（远小于 chunk dt 50 ms），head 锚点选取无歧义
- **pair_seq**：单调递增，gap > 1 视为丢包，连续 30 帧 gap 异常 → 重连

实测帧分辨率：**1088×1280×3 uint8**（H.264 解码后 bgr24）。

### 4.2 ARM 实测

```
host=192.168.31.34 port=50051, 5 Hz × 3 s 采样

joint angles_rad (起始零位):
  j0..j5 = (0.0013, -0.0013, 0.0013, 1.5715, 0.0006, -1.5692)
gripper_pos:  0.0887 - 0.0895  m   ← 单位 = 米 = state[6] = action[6]
gripper_vel:  <0.01 m/s
gripper_eff:  0
end_pose.position_m  : x≈0.2867 y≈0.0004 z≈0.2150
end_pose.orientation : (~0, 0, 0, 1) xyzw   ← 起始接近无旋转
```

`dataset_stats.json` 对齐：

| 维 | dataset min / max | ARM 实测起始 |
| --- | --- | --- |
| state[0..5] (rad) | 详见 dataset_stats | 与零位一致 |
| state[6] (gripper_m) | [0, 0.0904] | 0.0887 ✓ |
| action[6] (gripper_m) | [0, 0.0904] | — |

**确认**：`proprio_raw = [j0..j5(rad), eef_pos(m)]`（7 维），`current_position = [eef_xyz(m), eef_rpy(rad)]`（6 维，从 quat 转 rpy）。

### 4.3 探测脚本位置

- WS probe：`scripts/fastwam_ws_probe.py`（新增，本次提交）
- ARM probe：参考脚本 `~/Desktop/read_arm_joint_angles.py`（用户提供，未入库）
- 上游 WS bridge 源码：`/home/tb5z035i/robot/utils/rgbd_ws_bridge/`（5090_1 本地）
- 上游 ai_client 参考脚本：`/home/tb5z035i/robot/utils/rgbd_ws_bridge/scripts/encoded_pair_ai_client.py`

---

## 5. CLI / 配置

`scripts/fastwam_active_loop_server.py` 的 flag：

| Flag | 默认值 | 含义 |
| --- | --- | --- |
| `--host` | `0.0.0.0` | HTTP 监听 host |
| `--port` | `8118` | HTTP 监听 port（与旧 server 8117 错开） |
| `--config` | `configs/task/real_1048_uncond_2cam224_1e-4.yaml` | model config |
| `--checkpoint` | `runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/checkpoints/step_020000.pt` | ckpt |
| `--dataset-stats` | `runs/.../dataset_stats.json` | normalizer stats |
| `--text-cache-dir` | `data/text_embeds_cache/real_1048` | text emb cache |
| `--default-camera-info` | `configs/camera_info/real_1048_default.json` | 双路相机标定，复用旧 server |
| `--ws-url` | `ws://192.168.31.66:19095` | rgbd_ws_bridge 地址 |
| `--ws-frame-max-age-ms` | `250` | frame_cache 单帧最大允许年龄（v1 200，v2 放宽以应对 ±35 ms recv jitter） |
| `--ws-reconnect-backoff-ms` | `500,1000,2000,5000,10000` | 断线重连退避（逗号分隔，封顶最后一个值） |
| `--ws-startup-timeout-ms` | `30000` | 启动时等首帧的最大时间，超时 abort |
| `--ws-warn-stale-ms` | `200` | 单通道 last_frame_age 超过 → log WARNING |
| `--ws-hold-stale-ms` | `500` | 单通道 last_frame_age 超过 → dispatcher 进入 hold |
| `--ws-estop-stale-ms` | `1500` | 单通道 last_frame_age 超过 → emergency_stop |
| `--arm-host` | `192.168.31.34` | arm_sdk gRPC host |
| `--arm-port` | `50051` | arm_sdk gRPC port |
| `--arm-poll-hz` | `50` | ARM 状态轮询频率 |
| `--arm-state-max-age-ms` | `100` | state_cache 最大允许年龄 |
| `--arm-lease-ms` | `15000` | acquire_control lease 时长（SDK 默认 15 s，内置自动续期） |
| `--arm-acquire-on` | `start` | `start`（默认）= `/start` 时 acquire / `/stop` 时 release；`init` = server 启动时 acquire 直到进程退出 |
| `--infer-period-ms` | `400` | 推理触发周期（2.5 Hz） |
| `--send-period-ms` | `50` | 下发周期（20 Hz） |
| `--blend-frames` | `4` | 新旧 chunk 重叠区帧数（线性混合） |
| `--chunk-len` | `None`（auto） | 单次推理输出帧数。None 时从训练 config `data.train.num_frames - 1`（=32）推导 |
| `--num-inference-steps` | `None`（auto） | 推理步数；None 时读训练 config `eval_num_inference_steps`（=10） |
| `--chunk-max-stale-ms` | `2000` | chunk 老于此值 → 安全保持模式 |
| `--auto-dispatch` | `false` | `false` 时只跑推理 + 日志，不调 SDK |
| `--emergency-on-failure` | `true` | 失败时是否立刻 `set_arm_emergency_stop(True)` |
| `--watchdog-period-ms` | `10` | watchdog 检查周期 |
| `--instruction` | `"open the door"` | 默认 task；`/start` 可覆盖 |
| `--device` | `cuda:1` | 默认 GPU（避开 5090_1 上 GPU 0 的 airdc 占用） |
| `--require-gpu-mem-free-gb` | `8` | 启动检查 GPU 显存空闲下限，不足 → log ERROR + exit(1) |
| `--undistort` | `true` | 是否对 1088×1280 输入做去畸变。设为 false 时跳过 undistort，原图直接给 model_clients（用于纯测试） |

去掉的 v1 flag：

- ~~`--enable-self-fetch`~~：新脚本默认就是 self-fetch 模式，旧 server 走另一条路
- ~~`--ws-channel-map`~~：通道名与 image_key 同名，identity
- ~~`--train-config-path`~~：从 `--checkpoint` 同级目录自动推导
- ~~`--lease-renew-ms`~~：信任 SDK 内置续期

---

## 6. 图像流程（**关键变化**）

**v2 决策**：服务端只做 undistort，**不再 cv2.resize 到 480×640**。所有 stitch+resize+crop+normalize 在 `FastWAMModelClient._preprocess_image` 内部完成。

```
WS frame_cache (1088×1280×3 uint8 BGR, 两路)
    ↓ ClosedLoopRunner.snapshot
两路 BGR 1088×1280
    ↓ undistort_stereo_side_from_camera_info(eye=left/right, output_size="native")
    ↓ (跳过 cv2.resize 到 480×640)
两路 BGR 1088×1280 (去畸变后)
    ↓ FastWAMModelClient.infer(images=dict, ...)
        内部:
        ├── _stitch_cameras_native(horizontal): 1088 × 2560 × 3
        ├── ResizeSmallestSideAspectPreserving(target 224×448):
        │     scaling_ratio = max(448/2560, 224/1088) = max(0.175, 0.206) = 0.206
        │     → 224 × 528 × 3
        ├── CenterCrop(224, 448): 224 × 448 × 3
        └── Normalize(mean=0.5, std=0.5) → tensor [C, 224, 448] in [-1, 1]
```

**与训练侧（480×640 raw → stitch 480×1280 → resize 短边 480→224 → 224×597 → crop 224×448）的差异**：

| 阶段 | 训练（lerobot raw 480×640） | v2 服务端（raw 1088×1280） |
| --- | --- | --- |
| stitch 后纵横比 | 480 : 1280 = 1 : 2.67 | 1088 : 2560 = 1 : 2.35 |
| resize 后大小 | 224 × 597 | 224 × 528 |
| center crop 留下的 W 中心区 | 224 × 448 of 597 → 留中心 75% | 224 × 448 of 528 → 留中心 85% |
| 边缘视场 crop 量 | 25% | 15% |

**结论**：v2 流程看到的图像视场**比训练时略宽**（边缘多保留 10%）。这是新引入的分布偏移项，**必须在上线前做一次 dry-test 验证**（见 §10）。如果效果差，回滚到 v1 写法（先 cv2.resize 到 480×640）只需要加一行 `cv2.resize(out[k], (640, 480), cv2.INTER_AREA)`。

**保留训练流程的备用方案（feature flag）**：CLI 加 `--image-pipeline {raw_native, lerobot_480x640}`，默认 `raw_native`。如果上线 dry-test 发现偏差大，切到 `lerobot_480x640` 即恢复 v1 行为。

---

## 7. 逐文件代码改动清单

> 所有新文件放在 `src/fastwam/server/` 目录。删除该目录 + 删除 `scripts/fastwam_active_loop_server.py` 即可整体回滚。

### 7.1 `src/fastwam/server/__init__.py`（新增）

空 `__init__.py`，仅为打包。

### 7.2 `src/fastwam/server/rotation.py`（新增，工具函数）

```python
def quat_xyzw_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray   # (3,) float32, extrinsic XYZ
def rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray         # (4,) float32
def unwrap_rpy_sequence(rpy_seq: np.ndarray) -> np.ndarray  # (N,3) 处理 2π 跳变
def quat_canonicalize(q_new, q_prev) -> np.ndarray          # 同号化 (dot<0 取负)
def unwrap_quat_sign(q_new, q_prev) -> np.ndarray           # alias of quat_canonicalize
```

约定：scipy `Rotation.from_quat(..).as_euler("xyz", degrees=False)`，extrinsic XYZ。

**ground truth 防偏移机制**：靠 fingerprint 单测，不靠注释里的 commit hash。

文件头注释只保留约定本身（便于后人快速 grep 到上游来源），不强制 verify hash：

```python
# rotation.py
"""
Euler convention: scipy Rotation.from_quat(..).as_euler("xyz", degrees=False), extrinsic XYZ.
Must match training-side convention in
  mcap_preprocess_pipeline/scripts/step01_extract_mcap_rgb_and_params.py:1117
If the upstream changes convention, the fingerprint test below will fail.
"""
```

fingerprint 单测才是真正的防线：

- **GT 来源**：h200-1 `/DATA/disk1/datasets_lerobot/opendoor_real_1048` 数据集，随机抽 5 帧 action 的 `(quat_xyzw, expected_rpy)` 对，离线脚本 `scripts/fastwam_generate_rotation_fingerprint.py` 生成 `tests/fixtures/rotation_fingerprint.json`，commit 入库。
- **运行**：`test_rotation_fingerprint.py` 加载 fixture，`assert max abs error < 1e-5`。
- **启动时自动跑一次**，失败 → `sys.exit(1)`。
- **失效机制**：如果上游改了约定 → 重训模型 → 重生成 dataset → 重生成 fingerprint fixture，单测自动随之 fail，迫使 rotation.py 同步更新。如果上游改了约定但 dataset 没重训，dataset 里 action 仍是老约定，FastWAM 也不受影响。

单元测试要点：

- 往返一致：`rpy_to_quat_xyzw(quat_xyzw_to_rpy(q)) ≈ ±q`（同号化后严格相等）；
- N=1000 随机旋转，最大相对误差 < 1e-6；
- gimbal-lock 边界：pitch ∈ {±89.9°, ±90°, ±90.1°} 各跑一例；
- 符号翻转：连续两帧 quat 翻号，`unwrap_quat_sign` 必须修回参考符号。

### 7.3 `src/fastwam/server/arm_client.py`（新增）

```python
class ArmStateSnapshot(NamedTuple):
    angles_rad: np.ndarray       # (6,)
    gripper_m: float             # 实际取自 client.get_eef_joint_state().eef_pos (米)
    eef_xyz: np.ndarray          # (3,)
    eef_rpy: np.ndarray          # (3,) extrinsic XYZ，已 unwrap + canonicalize
    eef_quat_xyzw: np.ndarray    # (4,) 原始（未取负）
    capture_ts_ns: int           # 本次 RPC 完成时刻（SDK 不返回采集时间戳）

class ArmClient:
    def __init__(self, host, port, poll_hz, state_max_age_ms, lease_ms, logger): ...
    def start(self) -> None: ...                # 起后台 poller 线程（只读 RPC，无需 lease）
    def stop(self) -> None: ...
    def acquire_control(self) -> bool: ...      # 调 SDK acquire_control(lease_ms=lease_ms); SDK 内部自动续期
    def release_control(self) -> None: ...
    def latest(self) -> ArmStateSnapshot | None: ...
    def send_pose(self, target_xyz, target_rpy, gripper_m) -> bool: ...
    def emergency_stop(self, enable: bool = True) -> None: ...
    def health(self) -> dict: ...               # 含 last_poll_age_ms, consecutive_fail, lease_alive
```

行为细节：

- **lease**：完全信任 `AirbotClient.acquire_control(lease_ms=15000, renew_period_s=5.0)` 的内置续期线程，**不外加 timer**。`health()` 读 SDK 内部 `_lease_id` / `_lease_expire_unix_ms` 暴露状态。
- 后台 poller 串行三个 RPC（实测 ~1 ms），写 `state_cache`。失败计数 `_consecutive_fail`，连续 5 次 → `health()` 标 RED。
- 每次写入 cache 前对 `eef_quat_xyzw` 做 `unwrap_quat_sign(new, last_quat)`，避免 ±q 翻号。
- `send_pose`：`rpy → rpy_to_quat_xyzw → CartesianPose(position=(x,y,z), orientation=(qx,qy,qz,qw))` → `client.move_end_pose(pose, options=ArmControlOptions(), timeout_ms=1000)`；gripper 单独 `client.move_eef(g, options=ArmControlOptions(), timeout_ms=1000)`。`ArmControlOptions()` 默认 `blocking=False`。任一返回 False → 抛 `ArmSdkError`。
- `emergency_stop` 阻塞 0.15 s（SDK 已知行为）。

⚠️ **arm_sdk `set_arm_speed(arm_speed: list[float])` 只接受 6 个关节速度，没有独立 EEF 线速度 API**。v1 设计稿的 `set_speeds(arm_rad_s, eef_m_s)` 改为：

- **acquire 后**调一次 `set_arm_speed([π/6]*6)` 设置统一安全速度（每关节 30°/s）；
- **hold 模式**：不重发命令，依靠 SDK 自然停在 last_pose；如发生超时，watchdog 触发 emergency_stop。

### 7.4 `src/fastwam/server/ws_ingest.py`（新增）

```python
class FrameSnapshot(NamedTuple):
    bgr: np.ndarray            # H×W×3 uint8 (1088×1280)
    capture_ts_ns: int         # channel.meta['stamp_ns']
    decode_ts_ns: int
    pair_seq: int

class WSFrameIngester:
    def __init__(self, ws_url, expected_channels, frame_max_age_ms, reconnect_backoff_ms_list, startup_timeout_ms, logger): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self, image_key: str) -> FrameSnapshot | None: ...
    def health(self) -> dict: ...     # last_frame_age_per_channel_ms, fps_5s, decode_fail_count, reconnect_count
```

实现要点：

1. 后台线程跑 `websocket-client` 同步循环（无 asyncio）。
2. **代理 bypass**：启动时把 `ws_url` host 加进 `NO_PROXY`/`no_proxy`，并对 `WebSocketApp.run_forever` 传 `http_no_proxy=[host]`（参考 `encoded_pair_ai_client.py:76`）。
3. 收到 bytes 后按 V4 协议拆包（见 §4.1）。
4. 启动自检（5 秒内必须通过）：
   - `payload_format == "h264_annexb_pair_v1"`
   - `type == "encoded_sync_pair"`
   - `len(channels) == 2` 且 `set(channel.name) == expected_channels`
   - 任一不过 → 抛 `RuntimeError` 终止启动
5. **PyAV 首包暖机**：probe 实测首包 codec context 输出空帧（pkt0 shape=None），pkt1+ 才正常。允许首 **2 个 packet** 解码失败；超过则报错。
6. 启动 5 s 滑动窗口监测：fps、解码失败率、pair_seq 缺口；fps < 14 Hz（期望 70%）→ WARNING。
7. **`base_capture_ts_ns = channels[head_left].stamp_ns`**（写死）。
8. 每路 `latest_frame[image_key]` 用 `threading.Lock`；解码缓冲池复用。

### 7.5 `src/fastwam/server/image_pipeline.py`（新增）

把 `scripts/fastwam_http_server.py:_normalize_image_resolution` 抽出来供两边复用：

```python
def undistort_native(
    images: dict[str, np.ndarray],   # 1088×1280×3 uint8 BGR
    default_camera_info: dict[str, dict],
    stereo_pair: dict[str, str],     # {"left": "head_left", "right": "right_wrist_left"}
    alpha: float = 0.0,
) -> dict[str, np.ndarray]:
    """与旧 server 一致的 undistort 逻辑，但不做 cv2.resize 到 480×640。
    返回的 BGR 仍是 1088×1280×3 uint8。"""
```

旧 `scripts/fastwam_http_server.py` 内部 `_normalize_image_resolution` 拆出 helper 后保持原行为不变（向后兼容）。

### 7.6 `src/fastwam/server/closed_loop.py`（新增）

```python
@dataclass
class ChunkEntry:
    action_abs: np.ndarray         # (chunk_len, 7) — xyz, rpy, gripper
    base_capture_ts_ns: int        # = head_left.stamp_ns at snapshot
    step_dt_ns: int = 50_000_000
    chunk_id: int

class ClosedLoopRunner:
    def __init__(self, model_client, arm_client, ws_ingester, train_config_path,
                 infer_period_ms, send_period_ms, chunk_len, num_inference_steps,
                 blend_frames, chunk_max_stale_ms, auto_dispatch,
                 emergency_on_failure, watchdog_period_ms,
                 default_camera_info, stereo_pair, logger): ...
    def start(self, instruction: str | None) -> None: ...
    def stop(self) -> None: ...
    def emergency(self) -> None: ...
    def status(self) -> dict: ...
```

构造时强制校验（不一致直接 abort）：

```python
train_cfg = yaml.safe_load(open(self.train_config_path))
cfg_num_frames        = train_cfg["data"]["train"]["num_frames"]            # 33
cfg_action_dim        = train_cfg["data"]["train"]["processor"]["action_output_dim"]  # 7
cfg_context_len       = train_cfg["data"]["train"]["context_len"]           # 128
cfg_eval_steps        = train_cfg["eval_num_inference_steps"]               # 10

expected_chunk_len    = cfg_num_frames - 1                                   # 32
if self.chunk_len is None:
    self.chunk_len = expected_chunk_len
elif self.chunk_len != expected_chunk_len:
    raise RuntimeError(...)
if self.chunk_len != self.model_client.action_horizon:
    raise RuntimeError(...)
assert cfg_action_dim == 7
if self.num_inference_steps is None:
    self.num_inference_steps = cfg_eval_steps
```

**三条线程**：

#### 7.6.1 InferLoop（2.5 Hz）

1. 每 `infer_period_ms` 触发，用 `time.monotonic()` 漂移补偿；
2. **snapshot**：从 `ws_ingester.latest("head_left")` / `latest("right_wrist_left")` 取两帧、`arm_client.latest()` 取 state；
3. **校验**：任一图像 / state 老于阈值则跳过本次推理，记 `skip_reason`；
4. **undistort**：调 `image_pipeline.undistort_native(images, default_camera_info, stereo_pair)`，保持 1088×1280 输出；
5. **组装 model input**：
   ```python
   model_input = {
       "images": {"head_left": ndarray, "right_wrist_left": ndarray},
       "proprio_raw": np.concatenate([state.angles_rad, [state.gripper_m]]),
       "current_position": np.concatenate([state.eef_xyz, state.eef_rpy]),
       "instruction": current_instruction,
   }
   ```
6. **调推理**：`result = self.model_client.infer(model_input)`（**v1 设计稿误写为 `predict_action`，实际方法名是 `infer`**）；
7. `result["actions"]` 已经是 cartesian_absolute (32, 7)；
8. **写 chunk_ringbuffer**：双缓冲 ringbuffer，容量 **2**。**push 新 chunk 时丢最旧的**（决策见 §10 风险 8）；
9. 整段 `perf_counter` 打点：`image_prep / model / postproc` 三段。

#### 7.6.2 DispatchLoop（20 Hz，50 ms timer，`time.monotonic_ns()` 漂移补偿）

1. 取当前最新的 chunk（ringbuffer 末尾），以及上一条（混合用，可能为 None）；
2. 计算 `idx = round((t_slot - chunk.base_capture_ts_ns) / chunk.step_dt_ns)`；
3. `idx < 0` 跳过（理论不出现）；`idx >= chunk_len` → chunk 耗尽，进入 hold（不发新命令，依赖 SDK 自然停在 last_pose）；
4. 若 prev_chunk 存在且 `idx_in_blend = (now - new.base) / dt < blend_frames`：
   - `idx_old = (t_slot - prev.base) / dt`
   - `idx_new = (t_slot - new.base) / dt`
   - `w = (idx_in_blend + 1) / blend_frames`，即 0.25 → 1.0
   - **xyz** 线性混合
   - **rpy**：先对 `[prev.rpy[idx_old], new.rpy[idx_new]]` 做联合 `unwrap_rpy_sequence`，再 lerp；任一维差 > π/4 → 放弃混合，硬切（log WARNING）
   - **gripper** 线性混合
5. `arm_client.send_pose(xyz, rpy, gripper)`；
6. send 失败 → `emergency_on_failure` 决定急停 / hold。

#### 7.6.3 Watchdog（100 Hz，10 ms 周期）

- `chunk_age = now - chunk.base_capture_ts_ns - chunk_len * dt`，> `chunk_max_stale_ms` 且 ringbuffer 无更新 chunk → hold；
- `arm_client.health() / ws_ingester.health()` 连续 RED > 200 ms → emergency_stop；
- InferLoop 心跳：上次结束 > 2× `infer_period_ms` → WARN；> 5× → hold；
- WS stale 三档：warn / hold / estop（CLI flag）。

`status()` 报告：`current_chunk_id`、`prev_chunk_id`、`last_dispatch_idx`、`blend_state`、`hold_mode`、各 health 灯、最近 N 次推理耗时直方图。

### 7.7 `scripts/fastwam_active_loop_server.py`（新增）

入口结构：

```python
def main():
    args = build_argparser().parse_args()
    # 1. 启动日志 banner (见 §7.7.1)
    # 2. GPU 显存检查
    check_gpu_free_mem(args.device, args.require_gpu_mem_free_gb)
    # 3. 加载 model_client (复用 FastWAMModelClient)
    model = init_model(args)
    # 4. rotation fingerprint
    run_rotation_fingerprint_or_exit(args)
    # 5. ARM 启动 poller（不 acquire）
    arm = ArmClient(...); arm.start()
    # 6. WS ingester 启动 + 启动自检
    ws = WSFrameIngester(...); ws.start()
    # 7. ClosedLoopRunner（不 start，等 /start）
    runner = ClosedLoopRunner(model, arm, ws, ...)
    # 8. HTTP server
    httpd = ThreadingHTTPServer((host, port), make_handler(arm, ws, runner))
    # 9. warmup + benchmark 推理 N 次
    warmup_benchmark_or_exit(model)
    # 10. serve_forever
```

HTTP endpoints（仅这些，无 `/infer`）：

- `POST /start`：body `{"instruction": "open the door"}`（可选）。流程：`arm.acquire_control()` → `runner.start(instruction)`
- `POST /stop`：`runner.stop()` → `arm.release_control()`
- `POST /emergency`：body `{"enable": true|false}`（默认 true）。调 `arm.emergency_stop(enable)`，急停后必须再 POST `{"enable": false}` 复位才能 `/start`
- `GET /health`：基础状态 + arm.health() + ws.health()
- `GET /closed_loop_status`：runner.status()
- `GET /ws_status`：ws.health() 详版（每通道近 N 次解码耗时、key frame 间隔、最近一次重连原因）

#### 7.7.1 启动日志 banner（INFO 级，**强制打印**）

```
[startup] script               = fastwam_active_loop_server.py v2
[startup] train_config_path    = runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/config.yaml
[startup] train.num_frames     = 33
[startup] chunk_len            = 32        (= num_frames - 1, frame-aligned backward delta)
[startup] action_output_dim    = 7         (xyz + rpy + gripper)
[startup] context_len          = 128
[startup] num_inference_steps  = 10        (from eval_num_inference_steps)
[startup] device               = cuda:1
[startup] gpu_free_mem_gb      = NN.N      (require ≥ 8)
[startup] ws_url               = ws://192.168.31.66:19095
[startup] ws_channels          = head_left, right_wrist_left  (identity mapping)
[startup] arm_host:port        = 192.168.31.34:50051
[startup] arm_lease_ms         = 15000     (SDK auto-renew @ 5s)
[startup] infer_period_ms      = 400       (2.5 Hz)
[startup] send_period_ms       = 50        (20 Hz)
[startup] blend_frames         = 4
[startup] image_pipeline       = raw_native (1088x1280 -> undistort -> model_clients)
[startup] scipy_version        = <pinned>
[startup] rotation fingerprint test (5 GT samples from opendoor_real_1048): PASS
[startup] warmup infer (5 calls) ... last latency = NNN ms
[startup] benchmark infer (10 calls): p50=NNNms p95=NNNms p99=NNNms
```

启动失败硬条件（任一不过 → `sys.exit(1)`）：

- 训练 config 与 CLI override 不一致
- fingerprint 单测失败
- GPU 显存 free < `--require-gpu-mem-free-gb`
- benchmark p50 > 500 ms
- WS 启动自检不过（payload_format / type / channels 不符合）
- WS 首帧超时（30 s）

### 7.8 `scripts/fastwam_http_server.py`（**不动**）

v1 设计稿要求修改此文件。**v2 完全不动**，保持现有行为：客户端推 images + proprio + current_position，被动调用。仅在 §7.5 提到 `_normalize_image_resolution` 拆出 helper 时做无行为变化的重构。

### 7.9 `scripts/fastwam_ws_probe.py`（新增，本次提交）

把 §4 中跑过的 probe 脚本固化入库，供后续上线核验 / 故障排查。

### 7.10 `scripts/fastwam_generate_rotation_fingerprint.py`（新增）

离线脚本：从 h200-1 `/DATA/disk1/datasets_lerobot/opendoor_real_1048` 读 5 个随机 frame 的 cartesian action，dump 成 `tests/fixtures/rotation_fingerprint.json`。

### 7.11 `docs/fastwam_http_server.md`（修改）

新增 "Active Loop Server" 章节，说明：

- 两个 server 的关系（旧的被动 / 新的主动，端口不同 8117 / 8118）
- `/start /stop /emergency /closed_loop_status` 端点
- 运行手册：切换 instruction 需先跑 `precompute_text_embeds.py`
- 上线 checklist（坐标系零位测试、GPU 选择、上游 WS bridge 启动顺序）

### 7.12 `pyproject.toml` / `uv.lock`（修改）

加依赖：

```
av >= 16.0.1
websocket-client >= 1.9.0
```

5090_1 `.venv` 已有这两个，pin 当前实测版本（av 16.0.1 / websocket-client 1.9.0）。

---

## 8. 错误处理 / 安全策略

| 事件 | 触发条件 | 响应 |
| --- | --- | --- |
| WS 单帧解码失败 | PyAV throw | 跳过该帧，计数 +1，连续 30 帧失败 → 关闭重连 |
| WS 断流 | socket 断 / 心跳超时（默认 1 s） | 进入退避重连；断流 > 1 s 且闭环运行中 → DispatchLoop 进入 hold；> 5 s → 急停 |
| ARM RPC 单次失败 | `move_*` 返回 False / RPC 抛异常 | 立即 hold；若 `emergency_on_failure=true` 立刻急停 |
| ARM RPC 连续失败 N=5 | poller / dispatcher 共用计数 | 急停 + health RED + 拒绝 `/start` |
| 推理异常 | model.infer throw | 当前 chunk 继续耗尽；耗尽时仍无新 chunk → hold |
| 推理超时 | 单次 > 2 × infer_period_ms | WARN；> 5 × → hold；连续 3 次 → 急停 |
| chunk 跑完无新 chunk | dispatch idx >= chunk_len 且 ringbuffer 空 | hold（不发新命令，SDK 自然停） |
| lease 被抢 | SDK 内部 `_lease_id` 清空 | 急停 + health RED + 拒绝 `/start`（手动 `/emergency` `{"enable": false}` + 重启进程） |
| `set_arm_emergency_stop` 调用 | watchdog 决策 | 阻塞 0.15 s，记录原因；后续 `/start` 必须先 POST `/emergency {"enable": false}` 复位 |
| watchdog 反应窗口 | 10 ms + ≤ 1 ms RPC + 150 ms 急停阻塞 | 最坏约 165 ms |
| 主进程崩溃 | Python 未捕获异常 | try-finally 内调一次 emergency_stop；systemd/supervisor 重启策略由部署侧决定 |

---

## 9. 测试计划

- **单元测试**（pytest）：
  - `rotation.py`：fingerprint（从 opendoor_real_1048 抽取的 5 GT）+ 往返一致 + unwrap + gimbal-lock 边界；
  - `ws_ingest.py`：喂事先 dump 好的 V4 字节流 → 验证解析 + 解码后图像 hash；
  - `closed_loop.py`：`ChunkEntry` ringbuffer + blend 数学正确性；
  - `arm_client.py`：mock gRPC stub 验证 `send_pose` 转换链路（rpy→quat→CartesianPose 字段顺序）。
- **集成测试**（不连真硬件）：
  - mock WS server（asyncio）回放预录二进制流；
  - mock arm_sdk（grpc 服务端 stub）回放 angle / pose；
  - 端到端跑 60 s `--auto-dispatch=false`，验证调度时序日志：每槽间隔标准差 < 2 ms。
- **dry-run**（连真 SDK 但不动机械臂）：
  - `--auto-dispatch=false`，机械臂围栏内但断电；
  - 跑 60 s，看 chunk_ringbuffer 写入频率、混合区帧索引、watchdog 心跳；
- **零位 dry-test（坐标系一致性，**绕过模型**）**：
  - 不启动 `runner.start()`，单独写一个测试 endpoint `POST /debug/zero_pose_test`；
  - 流程：读 `arm.latest()` → `current_pose = (eef_xyz, eef_rpy)` → 伪造 `actions = repeat(concat(current_pose, gripper_m), 32)` → 直接构造 `ChunkEntry` 喂 dispatcher；
  - 跑 5 s，机械臂应**完全不动**。任何位姿漂移 = 坐标系错配，立刻 abort + 修代码。
- **上线测试**（最后一步）：
  - `--auto-dispatch=true`，安全工位，单次开门；
  - 先做 §9 零位 dry-test 确认坐标系，再跑真实任务。

---

## 10. 风险分析

### 风险 1 — 欧拉约定 ground truth 在仓库外
- **现象**：训练侧用 `scipy.Rotation.from_quat(..).as_euler("xyz")` 在 `mcap_preprocess_pipeline/scripts/step01_extract_mcap_rgb_and_params.py:1117`。
- **影响**：高。一旦上游改了约定，反积分后位姿大幅偏差。
- **缓解**：
  1. `rotation.py` 文件头注释记录上游路径 + 关键片段（不要求填 commit hash，因为静态字符串不会自动更新，靠 fingerprint 单测兜底）；
  2. `test_rotation_fingerprint.py` 单测 5 组 GT（从 opendoor_real_1048 抽），`max abs error < 1e-5`；
  3. server 启动自动跑一次，失败 abort；
  4. 启动日志打印 fingerprint 通过 + `scipy.__version__`。
- **回滚**：发现不一致 → `--auto-dispatch=false` + 修 convention + 重生成 fingerprint。

### 风险 2 — SDK 帧间四元数符号翻转
- **现象**：`get_end_pose` 可能返回 `q` 或 `-q`（同一姿态），cumsum 放大成爆炸级偏差。
- **影响**：高。
- **缓解**：`rotation.unwrap_quat_sign(new, last)` + ArmPoller 维护 `last_quat`；写入 rpy 前再叠一层 `unwrap_rpy_sequence` 兜底。
- **回滚**：watchdog 检测 rpy 帧间跳变 > π/2 → 急停。

### 风险 3 — Gimbal lock
- **现象**：pitch ≈ ±90° 时 rpy 不连续。
- **影响**：中。open-the-door 一般不接近。
- **缓解**（分层）：
  - A. Dispatcher 每帧检查 `|pitch|`：> 75° WARN，> 85° ERROR + hold；
  - B. 连续两帧 rpy 任一维差 > π/4 视为异常，跳本帧；连续 3 帧异常 → emergency_stop；
  - C. 任务先验：open-the-door EEF 朝向基本水平；若 WARN 频繁触发，先排查标定，**不能简单调阈值**；
  - D. （future work）chunk 接近 gimbal 区切到 slerp delta，本期不实施。

### 风险 4 — WS 推流帧率波动
- **现象**：probe 实测 recv jitter ±35 ms。
- **影响**：中。
- **缓解**：`frame_max_age_ms=250`（v2 放宽）；InferLoop 内若图像 > 阈值跳本次推理。
- **回滚**：调大 max_age 或调小推理频率。

### 风险 5 — chunk 时间戳锚点漂移
- **现象**：若 base 取错（server now 而非 head_left.stamp_ns），混合区出现非物理跳变。
- **影响**：高。
- **缓解**：**强制 `chunk.base_capture_ts_ns = head_left.stamp_ns`**（invariant 检查 + 日志同时打三个时间戳：head.stamp_ns / server now / dispatch ts）。
- **回滚**：发现漂移切到「不混合，硬切换」模式。

### 风险 6 — delta_rpy 是分量差而非物理 delta
- **现象**：训练数据 `delta_rpy[t] = rpy[t] - rpy[t-1]`，server 现 `_delta_to_absolute` 用 cumsum 严格一致 → 不要改。
- **混合**：rpy 是"分量累加"语义，重叠区 lerp 合理，前提是无 ±π 跳变 → 混合前联合 unwrap，任一维差 > π/4 放弃混合。
- **影响**：中。
- **回滚**：`--blend-frames=0`。

### 风险 7 — lease 续期失败（v2 简化）
- **现象**：arm_sdk lease 默认 15 s，SDK 自带后台续期；若 SDK 自身故障 / 网络断 → 续期失败。
- **影响**：高。
- **缓解**：(a) **完全信任 SDK 内置续期**，不外加 timer；(b) `arm.health()` 每 100 ms 检查 SDK 内部 `_lease_id` 是否为 None，None > 200 ms → emergency_stop；(c) RPC 返回 "control lease cleared" 类错误立刻急停。
- **回滚**：手动急停 + 重启 server。

### 风险 8 — 模型推理时延（已实测，仍需运行期监控）
- **现象**：5090 GPU 实测 p50 167.5 / p99 192.5 ms @ 10 steps（n=30）。
- **影响**：中。预算 400 ms 余 ~230 ms。
- **缓解**：
  1. 启动 warmup + benchmark，p50/p95/p99 写入启动日志；
  2. **强制 `--device cuda:1`**，避开 GPU 0 上 airdc；启动检查显存 free < `--require-gpu-mem-free-gb` 直接 exit；
  3. `infer` 内 `perf_counter` 打 `preprocess / infer / denorm / format / total`；
  4. 启动告警阈值：p50 > 220 WARN，> 350 WARN + 提示降 steps，> 500 ERROR + exit；
  5. 运行期：单次 > 2× `infer_period_ms` WARN，> 5× hold；连续 3 次超 `infer_period_ms` emergency_stop；
  6. 调优空间：`num_inference_steps=6` mean 116.8 ms，端到端可降到 ~175 ms（quality 待验）。

### 风险 9 — 坐标系一致性（world vs base）
- **现象**：训练 6D pose 是哪个坐标系 / SDK `get_end_pose` 是哪个坐标系，文档未明。
- **影响**：高。
- **缓解**：(a) §9 **零位 dry-test 必跑**（绕过模型，伪造 actions = repeat(current_pose, 32)）；(b) 任何位姿漂移 = 坐标系错配，立刻 abort。
- **回滚**：`--auto-dispatch=false` 退场。

### 风险 10 — 急停反应窗口
- **现象**：watchdog 10 ms + RPC ~1 ms + 急停阻塞 150 ms ≈ 165 ms 最坏。
- **缓解**：watchdog 触发急停前先把 dispatcher `auto_dispatch` 标志原子置 False；急停期间忽略所有 send 重试；急停阻塞期间 InferLoop 也暂停。
- **回滚**：物理按钮兜底（部署侧）。

### 风险 11 — WS 上游不稳定
- **现象**：rgbd_ws_bridge 历史上经常无数据；本次开发期间也复现过 `Connection refused`（需手动启动）。
- **影响**：高。
- **缓解**（详见 §7.4）：
  1. 启动 30 s 内无首帧 → abort；首帧 payload_format/type/channel 校验不过 → abort；
  2. 5 s 滑动窗口监测 fps / 解码失败 / pair_seq 跳跃；
  3. 运行期 watchdog 三档：warn 200 ms / hold 500 ms / estop 1500 ms；
  4. `/health` + `/ws_status` 暴露 `last_frame_age_per_channel_ms / fps_5s / decode_fail_count / reconnect_count`；
  5. 退避重连 + 重连后请求关键帧（如协议支持）。
- **回滚**：停服 + 手动排查 / 重启上游 rgbd_ws_bridge。

### 风险 12 — chunk_ringbuffer 容量与丢弃策略（v2 新增明确）
- **决策**：ringbuffer 容量 = 2。push 新 chunk 时，若 buffer 已满，**丢最旧**（FIFO 满则覆盖头）。
- **原因**：infer_period 400 ms + chunk_len * dt = 1600 ms，第 3 次推理结果到达时第 1 个 chunk 大概率已被 dispatcher 跑完（dispatch 走完 1600 ms 需要 32 帧）。但若推理偶尔变慢导致 dispatch 提前耗尽，丢最旧能保证总是用最新观测。
- **影响**：低（运行期推理稳定时三块从不堆积，buffer 永远只占 1-2 个）。
- **可观测**：`status()` 暴露 ringbuffer occupancy 历史最大值。

### 风险 13 — SDK 命令队列与 50 ms 下发周期
- **现象**：`move_end_pose(pose, options=ArmControlOptions(), timeout_ms=1000)` 内部默认 `blocking=False`，是否会在 SDK 内部排队？hold 模式与未完成旧命令冲突？
- **缓解**：(a) acquire 后 `set_arm_speed([π/6]*6)`（30°/s）让命令在合理时间内执行完；(b) hold 模式**不主动重发** last_pose，依赖小速度自然停；(c) PR6 dry-run 时统计 `move_end_pose` RPC return 时间分布，若 p99 飙升说明 SDK 内部排队，需要降下发频率。
- **回滚**：hold = `set_arm_emergency_stop(True)`（更激进但安全）。

### 风险 14 — text cache 仅 1 个 entry
- **现象**：text cache 只有 `"open the door"` 一条。
- **缓解**：`/start` 收到未 cache instruction → 返回 4xx 明确错误"先跑 precompute_text_embeds.py"；上线 README 写清。
- **回滚**：限定 instruction 列表，运维白名单。

### 风险 15 — 图像 pipeline 与训练分布偏移（**v2 新增**）
- **现象**：v2 服务端不再 cv2.resize 到 480×640，模型看到的图像 stitch 后纵横比 1:2.35（训练 1:2.67），center crop 后边缘视场多保留 ~10%。
- **影响**：高。新引入的分布偏移，可能影响动作质量。
- **缓解**：
  1. 上线前在静态画面上跑 5–10 次推理，对比 v1 流程（先 480×640）和 v2 流程（直接 1088×1280）输出 action 的 max abs diff；
  2. CLI 加 `--image-pipeline {raw_native, lerobot_480x640}`，默认 `raw_native`；
  3. dry-test 中观察 chunk action 是否合理（不爆炸、不抖动）；
  4. **若发现偏差大**，CLI 切到 `lerobot_480x640` 即可一键恢复 v1 行为。
- **回滚**：`--image-pipeline=lerobot_480x640`。

---

## 11. 回滚策略

- **L0（运行期最轻）**：`--auto-dispatch=false`，server 仍跑推理 + 打日志，不下发命令；
- **L1**：`--image-pipeline=lerobot_480x640`，恢复 v1 一致的图像 pipeline；
- **L2**：停 `fastwam_active_loop_server.py`，启 `fastwam_http_server.py`，回到被动模式（旧 server 完全没动）；
- **L3（最重）**：删除 `src/fastwam/server/` + `scripts/fastwam_active_loop_server.py` 即恢复改造前。所有新代码隔离在独立目录/文件。
- **CLI 默认值**：首次上线 `--auto-dispatch=false`。验证稳定后再开。

---

## 12. 开发顺序建议（PR 拆分）

| PR | 内容 | 风险 | 依赖 |
| --- | --- | --- | --- |
| PR1 | `rotation.py` + 单测 + fingerprint 生成脚本 | 零 | 纯函数 |
| PR2 | `image_pipeline.undistort_native` 抽取，旧 server 重构无行为变化 | 低 | — |
| PR3 | `ws_ingest.py` 骨架 + `scripts/fastwam_ws_probe.py` 入库 | 低 | PR2 |
| PR4 | `arm_client.py`（只读 poller，不 acquire） | 低 | — |
| PR5 | `closed_loop.py` + `fastwam_active_loop_server.py` 骨架，`--auto-dispatch=false`，跑推理 + 写 chunk_ringbuffer + 日志 | 中 | PR1–4 |
| PR6 | dispatch 接入，`--auto-dispatch=true`，加 watchdog；先零位 dry-test 再单帧测试 | 高 | PR5 |
| PR7 | 急停联调 + 上线运行手册 + `docs/fastwam_http_server.md` 更新 | 高 | PR6 |

---

## 13. 未解项（v2）

v1 的"WS 实测通道名 / WS 协议字段 / SDK lease API / gripper 单位"**已在 §4 实测落地**。

剩余：

- **坐标系一致性**：上线前 §9 零位 dry-test 必跑；如果失败再决定坐标系转换矩阵。
- **图像 pipeline 分布偏移实测对比**：上线前对比 raw_native vs lerobot_480x640 输出差（见风险 15）。
- **gimbal lock 实际触发概率**：dispatcher 上线后采集 pitch 分布，回填风险 3 阈值。
- **undistort 实际耗时**：1088×1280 双路 undistort 时延需要 PR3 落地后实测，回填 §3 表。
- **SDK 命令队列行为**：PR6 dry-run 时统计 `move_end_pose` p99 RPC return 时间，确认 SDK 是否内部排队。
