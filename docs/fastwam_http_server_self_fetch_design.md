# FastWAM HTTP Server — 服务器主动采集 + 主动下发改造设计

> 版本：v1（设计稿） · 日期：2026-05-18 · 状态：仅设计，未实现

## 0. TL;DR

把 `scripts/fastwam_http_server.py` 从「客户端推 images + proprio + current_position」改成「服务器内置 WS 图像 ingester、ARM gRPC 异步状态缓存、chunked 异步下发循环、急停 watchdog」。

最终客户端只剩两件事：

1. `POST /start`（可选携带 `instruction` 覆盖默认 prompt）触发闭环；
2. `POST /stop` 或 `POST /emergency` 终止。

`POST /infer` 保留并向后兼容（payload 缺字段则从内置缓存自取），用 `--enable-self-fetch=false` 可一键回退到旧的「客户端推数据」模式。

实测端到端闭环延迟约 **225 ms**（capture → 第一帧下发，图像传输 ~45 ms p50 + ARM RPC ~1 ms + 推理 ~170 ms p50 / 10 steps + 调度几 ms），控制环 20 Hz，推理环 2.5 Hz，chunk 内部用 4 帧（200 ms）线性混合做无缝拼接。

---

## 1. 设计目标

- **接口最小化**：客户端不再承担图像 / 状态采集职责，只发触发信号；
- **延迟可控**：闭环 < 400 ms，控制频率 20 Hz、推理频率 2.5 Hz；
- **失败安全**：WS 断流、ARM RPC 失败、推理超时、chunk 过期，任何一项都必须收敛到「机械臂可控的安全保持状态」或「急停」；
- **向后兼容**：旧 `POST /infer` 路径保留，CLI flag 控制是否启用 self-fetch；
- **可观测**：每段延迟（图像准备 / 模型推理 / 下发调度）都有 `perf_counter` 打点 + 周期性日志；
- **风险隔离**：所有新模块放在独立目录 `src/fastwam/server/`，删除该目录可整体回滚。

---

## 2. 架构图

图中所有标签为纯 ASCII，等宽字体下严格对齐；中文说明放图外。

```text
                       +---------------------+
                       |    HTTP Client      |
                       +----------+----------+
                                  |
              /start  /stop  /emergency  /status  /infer
                                  v
+-----------------------------------------------------------------+
| FastWAM HTTP Server  (single process)                           |
|                                                                 |
|   +------------------+              +------------------+        |
|   | WS Ingester      |              | ARM Poller       |        |
|   | thread           |              | thread @ 50 Hz   |        |
|   | - PyAV H.264     |              | - get_arm_joint  |        |
|   | - auto reconnect |              | - get_eef_joint  |        |
|   | latest_frame[k]  |              | - get_end_pose   |        |
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
|             | 2. FastWAMModelClient.infer  |                    |
|             | 3. denorm + delta_to_abs     |                    |
|             | 4. write chunk_cache [32 x 7]|                    |
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
|                            |                                    |
+----------------------------+------------------------------------+
                             |
                             v
                 +---------------------------+
                 | External services         |
                 | WS  192.168.31.66:19095   |
                 | ARM 192.168.31.34:50051   |
                 +---------------------------+
```

说明：
- `WS Ingester` 与 `ARM Poller` 并行运行，各自维护一个带 TTL 的 latest cache。
- `ClosedLoopRunner` 是核心推理线程，按 2.5 Hz 触发：snapshot 两个 cache，跑 FastWAM，得到 `[32 x 7]` 绝对 action 写回 chunk_cache。
- `Dispatcher` 是 20 Hz 定时器，决定每个 tick 该下发 chunk 的哪一帧（含丢陈旧 + 重叠混合 + rpy→quat 转换）。
- `Watchdog` 监控三类异常并升级到急停或保持模式。

---

## 3. 数据流时序

一次完整闭环的事件序列，`t=0` 是相机捕获时刻；每段标「实测」或「估算」。

```text
t=0       camera capture (head_left + right_wrist_left @ 20 Hz)
t≈45      WS frame arrives in frame_cache         (real, p50; H.264 + LAN + PyAV)
t≈47      ARM RPC triple returns to state_cache   (real, ~1 ms; runs every 20 ms)
t≈50      ClosedLoopRunner snapshot(frame, state, t_capture = 0)
t≈50..220 FastWAMModelClient.infer                (real, p50 167.5 ms @ 10 steps; Agent I)
t≈225     chunk_cache write: 32 frames @ 50 ms
          chunk[i] logical time = i * 50 ms  (chunk[0] = 0, chunk[31] = 1550)

t≈225     Dispatcher picks up new chunk
          drop_n = ceil(225 / 50) = 5  (frames already in the past)
t≈225     send chunk[5],  blend w = 0.25
t≈275     send chunk[6],  blend w = 0.50
t≈325     send chunk[7],  blend w = 0.75
t≈375     send chunk[8],  blend w = 1.00          (blend window ends)
t≈425..   send chunk[9..31] without blending,
          until next chunk lands at t ≈ 625..675 and takes over

(in parallel)
t=400     next inference triggered (infer period = 400 ms)
```

延迟分解：

| 段 | 时长 | 来源 |
| --- | --- | --- |
| 摄像头 → frame_cache | 45–87 ms (p50–p99) | **实测**（Agent F：head 57.6 / wrist 44.5 mean） |
| ARM gRPC 三连发 | 0.9–1.2 ms (p50–p99) | **实测**（Agent C） |
| 图像预处理 (`_preprocess_image_batch`) | ~2.4 ms (p50) | **实测**（Agent I，长尾 max 104 ms 来自首次内存分配） |
| `FastWAMModelClient.infer` @ 10 steps | 167.5–192.5 ms (p50–p99) | **实测**（Agent I，5090 GPU，含图像预处理；mean 168.9 ms / std 6.2 ms） |
| chunk_cache → dispatcher 接管 | ≤ 50 ms | 设计目标 |
| `move_end_pose` + `move_eef` RPC | < 1 ms | 实测同 ARM 三连发 |
| **总闭环（捕获 → 第一帧下发）** | **~225 ms** | 实测；比初版估算 350 ms 短 125 ms，预算余 ~175 ms |

**推理 scaling**（Agent I 拟合 `mean_ms = 12.95 * steps + 42.31`，R² = 0.9993）：

| `num_inference_steps` | mean (ms) | 说明 |
| ---: | ---: | --- |
| 4 | 96.2 | 极快，需评估动作质量 |
| 6 | 116.8 | 调优甜点 |
| 8 | 146.3 | — |
| **10** | **173.4** | **训练 eval 默认，当前推荐** |
| 15 | 235.1 | 接近 400 ms 推理周期，无收益 |
| 20 | 302.1 | 不建议 |

单步 diffusion ~13 ms，固定开销 ~42 ms（VAE encode + text emb + denorm + py 调度）。如需进一步压延迟，调 `--num-inference-steps` 是最直接的杠杆。

---

## 4. CLI / 配置

`scripts/fastwam_http_server.py` 新增 flag（保持旧 flag 不变）：

| Flag | 默认值 | 含义 |
| --- | --- | --- |
| `--enable-self-fetch` | `false` | 总开关。`false` 走旧 client-push 路径，向后兼容 |
| `--ws-url` | `ws://192.168.31.66:19095` | rgbd_ws_bridge 地址 |
| `--ws-channel-map` | `head_left=<TBD>,right_wrist_left=<TBD>` | 上游通道名 → 模型 image_key 映射；通道名待 WS probe 确认 |
| `--ws-frame-max-age-ms` | `200` | frame_cache 内单帧最大允许年龄，超过视为陈旧 |
| `--ws-reconnect-backoff-ms` | `500,1000,2000,5000,10000` | 断线重连退避（逗号分隔，封顶最后一个值） |
| `--arm-host` | `192.168.31.34` | arm_sdk gRPC host |
| `--arm-port` | `50051` | arm_sdk gRPC port |
| `--arm-poll-hz` | `50` | ARM 状态轮询频率 |
| `--arm-state-max-age-ms` | `100` | state_cache 最大允许年龄 |
| `--infer-period-ms` | `400` | 推理触发周期（2.5 Hz） |
| `--send-period-ms` | `50` | 下发周期（20 Hz） |
| `--blend-frames` | `4` | 新旧 chunk 重叠区帧数（线性混合） |
| `--chunk-len` | `None`（auto） | 单次推理输出帧数。None 时从训练 config（`runs/.../config.yaml`）读 `data.train.num_frames - 1`（frame-aligned backward delta：第 0 帧 delta=0，可执行帧数 = num_frames - 1）。CLI 提供 override，但默认走 auto，避免与训练 horizon 漂移 |
| `--train-config-path` | 从 ckpt 推导 | 训练 config yaml 路径；默认从 `--ckpt-path` 同级目录推导。用于读取 `num_frames` / `action_output_dim` / `context_len` / `eval_num_inference_steps` |
| `--num-inference-steps` | `None`（auto） | 推理步数；None 时读训练 config 的 `eval_num_inference_steps`（当前训练 = 10） |
| `--chunk-max-stale-ms` | `2000` | chunk 老于此值则进入安全保持模式 |
| `--auto-dispatch` | `false` | `false` 时只跑推理 + 写日志，不调 SDK |
| `--emergency-on-failure` | `true` | 失败时是否立刻触发 `set_arm_emergency_stop(True)` |
| `--watchdog-period-ms` | `10` | watchdog 检查周期 |
| `--instruction` | `"open the door"` | 默认 task；与现有 default fallback 共用 |
| `--lease-renew-ms` | `4000` | lease 续期周期（SDK lease 5s，取 80%） |
| `--ws-startup-timeout-ms` | `30000` | 启动时等首帧的最大时间，超时直接 abort |
| `--ws-warn-stale-ms` | `200` | 单通道 last_frame_age 超过则 log WARNING（陈旧帧） |
| `--ws-hold-stale-ms` | `500` | 单通道 last_frame_age 超过则 dispatcher 进入 hold mode |
| `--ws-estop-stale-ms` | `1500` | 单通道 last_frame_age 超过则触发 emergency_stop |

---

## 5. 逐文件代码改动清单

> 本节只描述「将要怎么改」，不实际改任何代码。所有新文件均放在 `src/fastwam/server/` 目录下。

### 5.1 `src/fastwam/server/__init__.py`（新增）

空 `__init__.py`，仅为打包目的。后续可暴露 `ArmClient` / `WSFrameIngester` / `ClosedLoopRunner` 顶层别名。

### 5.2 `src/fastwam/server/rotation.py`（新增，工具函数）

签名：

```python
def quat_xyzw_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray   # shape (3,), float32
def rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray         # shape (4,), float32
def unwrap_rpy_sequence(rpy_seq: np.ndarray) -> np.ndarray  # shape (N,3)，处理 2π 跳变
def quat_canonicalize(quat_xyzw: np.ndarray, ref: np.ndarray | None) -> np.ndarray
    # 强制 w>=0 或与参考四元数同号，防止 SDK 帧间符号翻转
```

约定（**与上游 mcap_preprocess_pipeline 一致**）：scipy `Rotation.from_quat(..).as_euler("xyz", degrees=False)`，extrinsic XYZ。

单元测试要点：

- 往返一致：`rpy_to_quat_xyzw(quat_xyzw_to_rpy(q)) ≈ ±q`（同号化后严格等于）；
- 数值稳定：随机 N=1000 个旋转，最大相对误差 < 1e-6；
- gimbal-lock 边界：pitch ∈ {±89.9°, ±90°, ±90.1°} 各跑一例，验证 rpy 是否仍可还原相同旋转；
- 符号翻转：连续两帧四元数翻号时，`quat_canonicalize` 必须把第二帧改回参考符号。

被调用方：`arm_client.py`（读 end_pose 时）、`closed_loop.py`（下发前 rpy→quat）。

### 5.3 `src/fastwam/server/arm_client.py`（新增）

类 `ArmClient`，封装 `airbot.AirbotClient`（arm_sdk 5.2.3，已装入 `.venv`）。

构造参数：`host, port, poll_hz, state_max_age_ms, lease_renew_ms, logger`。

主要方法：

```python
class ArmStateSnapshot(NamedTuple):
    angles_rad: np.ndarray       # (6,)
    gripper_m: float
    eef_xyz: np.ndarray          # (3,)
    eef_rpy: np.ndarray          # (3,)  extrinsic XYZ，已 unwrap + canonicalize
    eef_quat_xyzw: np.ndarray    # (4,)  保留原始
    capture_ts_ns: int

class ArmClient:
    def __init__(self, ...): ...
    def start(self) -> None: ...                       # 起后台 poller
    def stop(self) -> None: ...
    def acquire_control(self) -> None: ...             # 包 acquire_control + switch_controller(servo)
    def release_control(self) -> None: ...
    def set_speeds(self, arm_rad_s: list[float], eef_m_s: float) -> None: ...
    def latest(self) -> ArmStateSnapshot | None: ...   # state_cache 读
    def send_pose(self,
                  target_xyz: np.ndarray,
                  target_rpy: np.ndarray,
                  gripper_m: float | None) -> bool: ...
    def emergency_stop(self, enable: bool = True) -> None: ...
    def health(self) -> dict: ...                      # for /health
```

行为细节：

- 后台 poller 线程串行调用三个 RPC（实测约 1 ms），写入 `state_cache`。失败计数 `_consecutive_fail`，连续 N（默认 5）次失败则把 `health()` 标 RED。
- `send_pose` 内部：`rpy → quat_xyzw → CartesianPose(position=(x,y,z), orientation=(qx,qy,qz,qw))`，调 `client.move_end_pose(pose, blocking=False)`；gripper 单独 `client.move_eef(g, options)`。两者均返回 bool，任一 False 时上抛 `ArmSdkError` 给 watchdog 决策。
- lease 续期：单独 timer 每 `lease_renew_ms` 调用一次（SDK 提供哪个 API 待对照 `airbot_example_record_and_replay.py`，预留位置）。
- `emergency_stop` 阻塞 0.15 s（SDK 已知行为），调用前后打日志。

接口：被 `closed_loop.py` 构造、`fastwam_http_server.py` 在 `--enable-self-fetch=true` 时初始化。

### 5.4 `src/fastwam/server/ws_ingest.py`（新增）

类 `WSFrameIngester`，基于仓库外参考脚本 `encoded_pair_ai_client.py`。

构造参数：`ws_url, channel_map: dict[str,str], frame_max_age_ms, reconnect_backoff_ms_list, logger`。

主要方法：

```python
class FrameSnapshot(NamedTuple):
    bgr: np.ndarray            # H×W×3 uint8
    capture_ts_ns: int         # 由 V4 meta JSON 中的时间戳填，缺则用本地 ts
    decode_ts_ns: int

class WSFrameIngester:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self, image_key: str) -> FrameSnapshot | None: ...
    def health(self) -> dict: ...
```

行为细节：

- 后台线程跑 `websocket-client` 同步循环（无 asyncio），收到 bytes 后按 V4 协议拆包：
  1. 读 8 字节 magic，校验 `b"RGBDWS4\x00"`；
  2. 读 `header_size:uint32`、`size_a:uint32`、`size_b:uint32`；
  3. 读 `header_size` 字节解析 JSON meta（含 `channel_a`、`channel_b`、时间戳等）；
  4. 读 `size_a` 字节 H.264 数据，喂入 PyAV codec context 解码 → BGR；
  5. 同样处理 channel_b。
- 解码失败（关键帧丢失等）跳过单帧，不报错。
- 通道名 → image_key 映射通过 CLI `--ws-channel-map` 注入，例如：
  ```
  head_left=cam_head_left,right_wrist_left=cam_rwrist_left
  ```
  上游通道名以 WS probe 实测为准（**未解项**）。
- 断线检测：socket 异常 / 心跳超时 → 关闭、按退避列表延迟重连；重连成功立刻请求关键帧（如协议支持）。
- 内存：`latest_frame[image_key]` 用 `threading.Lock` + 一份 deep copy；解码缓冲池复用，避免每帧分配。

被调用方：`closed_loop.py` 推理前 `latest()` 取两路图像；`fastwam_http_server.py` `/health` 报告 ingester 健康。

#### 5.4.1 启动自检 + 运行期监控（应对 WS 上游不稳定）

WSFrameIngester 启动时自检流程：

1. 连接 `ws://<host>:19095`，`--ws-startup-timeout-ms`（默认 30 s）内未收到第一帧 → 抛 `RuntimeError` 终止启动；
2. 收到首帧后校验：
   - `codec == "h264_annexb_pair_v1"`（payload_format）；
   - shape ∈ {(1088, 1280, 3), (480, 640, 3)}；
   - channel.meta['name'] 在 `expected_channels`（默认 `head_left`、`right_wrist_left`）；
   - 任一校验失败 → 抛 `RuntimeError` 并打印实际值，引导用户检查 `--ws-channel-map`；
3. 启动后 5 秒滑动窗口监测：fps、解码失败率、pair_seq 缺口；
   - fps < 期望 70%（e.g. < 14 Hz 当期望 20 Hz）→ log WARNING；
   - 解码失败 / pair_seq 跳跃 → log WARNING + 计数。

运行期监控：

- WSFrameIngester 暴露：`last_frame_age_per_channel_ms`、`fps_5s`、`decode_fail_count`、`reconnect_count`；
- Watchdog 每 100 ms 检查 `last_frame_age_per_channel_ms`：
  - `> --ws-warn-stale-ms`（200 ms）：log WARNING（陈旧帧）；
  - `> --ws-hold-stale-ms`（500 ms）：进入 hold mode（暂停下发）；
  - `> --ws-estop-stale-ms`（1500 ms）：触发 `emergency_stop`；
- `GET /health` 暴露所有上述指标；
- `GET /ws_status` 单独端点返回 ingester 详细状态（每通道最近 N 次解码耗时、key frame 间隔、最近一次重连原因）。

### 5.5 `src/fastwam/server/closed_loop.py`（新增）

核心调度类 `ClosedLoopRunner`。

构造参数：`model_client, arm_client, ws_ingester, channel_map, train_config_path, infer_period_ms, send_period_ms, chunk_len, num_inference_steps, blend_frames, chunk_max_stale_ms, auto_dispatch, emergency_on_failure, watchdog_period_ms, logger`。

构造时强制校验（**不一致直接 abort，不允许 fall back**）：

```python
# 在 __init__ 内
train_cfg = yaml.safe_load(open(self.train_config_path))
cfg_num_frames        = train_cfg["data"]["train"]["num_frames"]            # 33
cfg_action_dim        = train_cfg["data"]["train"]["processor"]["action_output_dim"]  # 7
cfg_context_len       = train_cfg["data"]["train"]["context_len"]           # 128
cfg_eval_steps        = train_cfg["eval_num_inference_steps"]               # 10

expected_chunk_len    = cfg_num_frames - 1                                   # 32
if self.chunk_len is None:
    self.chunk_len = expected_chunk_len
elif self.chunk_len != expected_chunk_len:
    raise RuntimeError(
        f"chunk_len mismatch: CLI={self.chunk_len} vs train_config num_frames-1={expected_chunk_len}"
    )

# 与 FastWAMModelClient.action_horizon 二次校验
if self.chunk_len != self.model_client.action_horizon:
    raise RuntimeError(
        f"chunk_len({self.chunk_len}) != model_client.action_horizon({self.model_client.action_horizon})"
    )

# action_output_dim 必须等于 7（xyz + rpy + gripper）
assert cfg_action_dim == 7, f"unexpected action_output_dim={cfg_action_dim}"

# num_inference_steps fallback 到训练 eval 配置
if self.num_inference_steps is None:
    self.num_inference_steps = cfg_eval_steps
```

> 训练 config 当前值（`runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/config.yaml`，已从 h200-1 拷贝到 5090_1 并 md5 验证一致）：
> - `data.train.num_frames: 33` → 可执行 chunk_len = 32
> - `data.train.processor.action_output_dim: 7`
> - `data.train.context_len: 128`
> - `eval_num_inference_steps: 10`

主要状态：

```python
@dataclass
class ChunkEntry:
    action_abs: np.ndarray         # (chunk_len, 7) — xyz, rpy, gripper（已 cumsum 反积分）
    base_capture_ts_ns: int        # 推理输入观测的时间戳锚点
    step_dt_ns: int = 50_000_000
    chunk_id: int

class ClosedLoopRunner:
    def __init__(self, ...): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def emergency(self) -> None: ...
    def status(self) -> dict: ...
```

三条线程：

1. **InferLoop**（threading.Timer 模式或精确 `time.monotonic` 调度）：
   - 每 `infer_period_ms` 触发一次；
   - snapshot：从 `ws_ingester.latest()` 取所有 image_key，从 `arm_client.latest()` 取 ArmStateSnapshot；
   - 校验：任一图像 / state 老于阈值则跳过本次推理，记录 `skip_reason`；
   - 组装 model input：`images` dict、`proprio_raw = concat(angles_rad, [gripper_m])`、`current_position = concat(eef_xyz, eef_rpy)`；
   - 调 `FastWAMModelClient.predict_action(...)`（**复用**现 `scripts/fastwam_http_server.py` 内已有的 model 封装，不复制逻辑）；
   - 调 `_delta_to_absolute(denorm_action, current_position)` 反积分 →（chunk_len, 7）绝对动作；
   - 构造 `ChunkEntry`，写到 `chunk_cache`（双缓冲：current + next）；
   - 整段用 `perf_counter` 打 image_prep / model / postproc 三段时间。

2. **DispatchLoop**（精确 50 ms timer，使用 `time.monotonic_ns()` 漂移补偿）：
   - 每槽 `t_slot`，计算应该播哪一帧：`idx = round((t_slot - chunk.base_capture_ts_ns) / chunk.step_dt_ns)`；
   - 若 `idx < 0` 跳过（chunk 比当前时刻还新，理论不会出现）；
   - 若 `idx >= chunk_len` 标记 chunk 耗尽 → 进入安全保持（保留最后一帧位姿，gripper 不变，速度 0）；
   - 若已有 next_chunk 且 idx 落在新旧重叠区前 `blend_frames` 帧内：
     - 老 chunk 在 `idx_old = (t_slot - old.base)/dt`；
     - 新 chunk 在 `idx_new = (t_slot - new.base)/dt`；
     - 权重 `w = (idx_in_blend + 1) / blend_frames`（0.25→1.0）；
     - 在 **xyz 空间** 线性混合（毫无问题）；
     - 在 **rpy 空间** 线性混合（**讨论**：因为训练侧 delta 也是 rpy 分量减法 + cumsum，rpy 累计值是连续的——前提是没有 ±π 跳变，所以小窗口（200 ms）内直接 lerp 是合理的；为防 unwrap 失败，混合前先 `unwrap_rpy_sequence` 对两条 rpy 序列做联合 unwrap，再 lerp）；
     - gripper 线性混合；
   - 调 `arm_client.send_pose(xyz, rpy, gripper)`；
   - send 失败 → `emergency_on_failure` 决定急停或 hold。

3. **Watchdog**（默认 10 ms 周期）：
   - 检查 chunk_age = now - chunk.base_capture_ts_ns - chunk_len * dt；若 > `chunk_max_stale_ms`，**且** 没有 next_chunk → 进入 hold 模式（DispatchLoop 改播 last_pose，速度 0）；
   - 检查 `arm_client.health()` / `ws_ingester.health()`：连续 RED 超过 200 ms → 紧急停止；
   - 检查 InferLoop 心跳：上次推理结束至今 > 2 × infer_period_ms → 记 WARN；> 5 × → 进入 hold。

`status()` 报告：current_chunk_id、next_chunk_id、last_dispatch_idx、blend_state、hold_mode、各健康灯、最近 N 次推理耗时直方图。

### 5.6 `scripts/fastwam_http_server.py`（修改）

**保留原 527 行所有逻辑**，仅做最小增量：

- 顶部 `argparse` 新增 5.4 节所有 flag（默认值如上）；
- `main()`：
  - 旧 `model = FastWAMModelClient(...)` 初始化保持不变；
  - 若 `--enable-self-fetch=true`：
    - 构造 `arm_client = ArmClient(host, port, ...)`、`ws_ingester = WSFrameIngester(...)`；
    - 启动两者（poller + ws ingester 线程）；
    - 构造 `runner = ClosedLoopRunner(model_client=model, ...)`；
    - 把 `runner / arm_client / ws_ingester` 挂到 HTTPServer 实例属性，供 handler 读；
- handler 新增 endpoint：
  - `POST /start`：body `{"instruction": "open the door"}`（可选）；调 `runner.start(instruction=...)`；
  - `POST /stop`：`runner.stop()`；
  - `POST /emergency`：`runner.emergency()`；
  - `GET /closed_loop_status`：返回 `runner.status()`；
- `POST /infer` 行为：
  - 若 `--enable-self-fetch=false` → 原行为不变；
  - 若 `true` → payload 中 `images / proprio_raw / current_position` 缺失时，从 ingester / arm_client 自取（与现有 `instruction` / `undistort` fallback 同一风格）；
- `GET /health`：原报告基础上追加 `arm_client.health()` 和 `ws_ingester.health()` 子字段；
- `_perform_infer`（旧函数）整段加 `perf_counter` 打点：image_decode / model_forward / postproc，写到响应 `timings` 字段（旧客户端可忽略）；
- `_INFER_LOCK` 保留——self-fetch 模式下 InferLoop 也走它，跟旧 `/infer` 互斥（避免两条入口同时打模型）。

#### 5.6.1 server 启动日志（INFO 级，**强制打印**）

启动 banner 必须包含以下字段，便于运维一眼核对配置/版本一致性：

```
[startup] train_config_path = runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/config.yaml
[startup] train.num_frames  = 33
[startup] chunk_len          = 32        (= num_frames - 1, frame-aligned backward delta)
[startup] action_output_dim  = 7         (xyz + rpy + gripper)
[startup] context_len        = 128
[startup] num_inference_steps= 10        (from eval_num_inference_steps)
[startup] ws_url             = ws://192.168.31.66:19095
[startup] ws_channel_map     = head_left=cam_head_left, right_wrist_left=cam_rwrist_left
[startup] arm_host:port      = 192.168.31.34:50051
[startup] infer_period_ms    = 400       (2.5 Hz)
[startup] send_period_ms     = 50        (20 Hz)
[startup] blend_frames       = 4
[startup] scipy_version      = <pinned>  (rotation 约定 API 行为依赖)
[startup] rotation fingerprint test: PASS
[startup] warmup infer (5 calls) ... last latency = NNN ms
[startup] benchmark infer (10 calls): p50=NNNms p95=NNNms p99=NNNms
```

如果训练 config 与 CLI override 不一致、或 fingerprint 单测失败、或 benchmark p50 > 500 ms（见 8 节风险 8）→ 直接 `sys.exit(1)`。

### 5.7 `docs/fastwam_http_server.md`（修改）

新增章节：
- self-fetch 模式开关与各 CLI flag；
- 启动两种典型命令行（兼容模式 / self-fetch 模式）；
- `/start /stop /emergency /closed_loop_status` 端点说明；
- 向后兼容声明：`--enable-self-fetch=false` 时行为与历史版本字节级一致；
- 运行手册：切换 instruction 时需先跑 `scripts/precompute_text_embeds.py` 生成 text cache，否则 500。

---

## 6. 错误处理 / 安全策略

| 事件 | 触发条件 | 响应 |
| --- | --- | --- |
| WS 单帧解码失败 | PyAV throw | 跳过该帧，计数 +1，连续 30 帧失败 → 关闭重连 |
| WS 断流 | socket 断 / 心跳超时（默认 1 s） | 进入退避重连；断流 > 1 s 且闭环运行中 → DispatchLoop 进入 hold；> 5 s → 急停 |
| ARM RPC 单次失败 | `move_*` 返回 False 或 RPC 抛异常 | 立即 hold；若 `emergency_on_failure=true` 立刻急停 |
| ARM RPC 连续失败 N=5 | poller / dispatcher 共用计数 | 急停 + 标记 health RED + 拒绝 `POST /start` |
| 推理异常 | model.predict_action throw | 当前 chunk 继续耗尽；若耗尽时仍无新 chunk → hold |
| 推理超时 | 单次 > 2 × infer_period_ms | WARN；> 5 × → hold；连续 3 次 → 急停 |
| chunk 跑完无新 chunk | dispatch idx >= chunk_len 且 next_chunk 缺 | hold（位姿保持，重发 last_pose，gripper 不变） |
| lease 被抢 | acquire_control 失败 / 后续 RPC 报无权限 | 急停（机械臂仍在前一条命令上）+ /health RED + 拒绝 /start |
| `set_arm_emergency_stop` 调用 | watchdog 决策 | 阻塞 0.15 s，记录原因 + 状态；后续 `/start` 必须先调 `/emergency` 复位 |
| watchdog 反应窗口 | 10 ms 周期 + ≤ 1 ms RPC + 150 ms 急停阻塞 | 最坏约 165 ms 从异常出现到机械臂收到急停 |
| 主进程崩溃 | Python 未捕获异常 | 终止前 try-finally 中调一次 emergency_stop；systemd / supervisor 重启策略由部署侧决定 |

---

## 7. 测试计划

- **单元测试**（pytest）：
  - `rotation.py`：往返一致 / unwrap / canonicalize / gimbal-lock 边界；
  - `ws_ingest.py`：喂一条固定字节流（事先 dump V4 包）→ 验证解析 + 解码后图像 hash；
  - `closed_loop.py`：`ChunkBuffer` 重叠混合的数学正确性（构造两条已知 chunk，验证混合后的轨迹严格等于 0.25→1.0 lerp）；
  - `arm_client.py`：用 mock gRPC stub 验证 `send_pose` 转换链路（rpy→quat→CartesianPose 字段顺序）。
- **集成测试**（不连真硬件）：
  - mock WS server（asyncio）回放预录二进制流；
  - mock arm_sdk（grpc 服务端 stub）回放 angle / pose；
  - 端到端跑 60 s `--auto-dispatch=false`，验证调度时序日志：每槽间隔标准差 < 2 ms。
- **dry-run**（连真 SDK 但不动机械臂）：
  - `--auto-dispatch=false`，机械臂围栏内但断电；
  - 跑 60 s，看 chunk_cache 写入频率、混合区帧索引、watchdog 心跳；
- **上线测试**（最后一步）：
  - `--auto-dispatch=true`，安全工位，单次开门；
  - 第一次启动先做 **零位 dry-test**：`current_position` 读出来，cumsum 32 帧全 0 delta，下发，验证机械臂确实不动（确认坐标系一致）。

---

## 8. 风险分析

> 这一节是核心。每条按「现象 / 原因 / 影响等级 / 缓解 / 回滚」给。

### 风险 1 — 欧拉约定 ground truth 在仓库外
- **现象**：训练侧用 `scipy.Rotation.from_quat(..).as_euler("xyz")` 在 `mcap_preprocess_pipeline/scripts/step01_extract_mcap_rgb_and_params.py:1117`。FastWAM 仓库不依赖它，无 import 关系。
- **原因**：约定靠口头/文档同步。一旦上游改成 "XYZ"（大写 intrinsic）或 "zyx"，FastWAM 这边推断出的 rpy 会跟训练数据语义不一致，反积分后位姿大幅偏差。
- **影响**：高。
- **缓解**：
  1. 在 `src/fastwam/server/rotation.py` 模块文件头**注释**记录三件事：
     - ground truth 上游路径：`/data/home/Lyle/Projects/mcap_preprocess_pipeline/scripts/step01_extract_mcap_rgb_and_params.py:1117`；
     - 关键代码片段：`Rotation.from_quat(quat).as_euler("xyz", degrees=False)`；
     - 上游 commit hash（**部署时**ssh 到 `mcap_preprocess_pipeline` 仓库 `git rev-parse HEAD` 填进来）。
  2. 在 rotation.py 同目录加 `test_rotation_fingerprint.py` 单测：
     - 含 3–5 组 `(quat_xyzw, expected_rpy)` ground truth，从 dataset action 反算出来的真实采样点；
     - `assert max abs error < 1e-5`；
     - server 启动时自动 run 一次，失败 abort（见 5.6.1）。
  3. 在 README / CLAUDE.md 写一行 reminder：「如果 `mcap_preprocess_pipeline` 改了欧拉约定，需要同步更新 FastWAM 的 `rotation.py` 并重生成 fingerprint 单测的 ground truth。」
  4. 启动日志打印 fingerprint 校验通过 + `scipy.__version__`（约定 API 行为依赖 scipy）。
- **回滚**：发现不一致时立刻 `--auto-dispatch=false` + 修 `rotation.py` 中的 convention 字符串 + 重新生成 fingerprint。

### 风险 2 — SDK 帧间四元数符号翻转
- **现象**：`get_end_pose` 返回 `(qx,qy,qz,qw)` 与 `(-qx,-qy,-qz,-qw)` 表示同一姿态，但 `as_euler` 算出的 rpy 在 ±π 边界会跳 2π。
- **原因**：很多 IK / 滤波内部用最短路径选符号，但 SDK 不保证。
- **影响**：高。任何 rpy 跳变都会被 cumsum 放大成爆炸级偏差。
- **缓解（必须实现，否则会爆 cumsum）**：
  1. 在 `src/fastwam/server/rotation.py` 提供：
     ```python
     def unwrap_quat_sign(q_new: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
         """同一旋转 q 和 -q 等价；为了保证连续帧 quat 不跳变，
         若 dot(q_new, q_prev) < 0 则取 -q_new。"""
         return q_new if np.dot(q_new, q_prev) >= 0 else -q_new
     ```
  2. 在 `ArmClient.ArmPoller` 维护 `last_quat`：
     - 第一次：直接保存；
     - 后续：`unwrap_quat_sign(new, last_quat)` → 写入 `state_cache` 之前修正；
     - 同时检测：unwrap 触发频率 > 1 Hz 视为异常，log WARNING（SDK 可能在 NaN/边界值附近反复翻转）。
  3. 写入 rpy 之前再叠一层 `unwrap_rpy_sequence`（rpy 域 unwrap）做兜底——双层防御。
- **测试**：
  - rotation 单测加 `unwrap_quat_sign` 双向往返：q 和 -q 都喂进来，输出应等价（同号）；
  - ArmPoller 测试：mock 一个故意翻转符号的 quat 序列，验证 unwrap 后无 2π 跳变。
- **回滚**：发现跳变时 watchdog 立即急停；离线修 `unwrap_quat_sign` / `unwrap_rpy_sequence` 逻辑。

### 风险 3 — Gimbal lock
- **现象**：当 pitch ≈ ±90°，roll / yaw 解不唯一，`as_euler` 返回的 rpy 数值会不连续。
- **原因**：欧拉角固有数学奇点。
- **影响**：中。「open the door」任务通常 pitch 不会逼近 ±90°，但右手腕摄像头视角下 pitch 可能接近 ±60°，要看实际工位。
- **缓解（分层）**：

  **A) 检测层（必做）**：
   在 Dispatcher 每帧检查 `rpy[1]` (pitch)：
   - `|pitch| > 75°` (~1.31 rad)：log WARNING("approaching gimbal lock")；
   - `|pitch| > 85°` (~1.48 rad)：log ERROR + 进入 hold mode。

  **B) 不连续跳变检测（必做）**：
   连续两帧 rpy 任一维度变化 > π/4 视为异常：
   - log WARNING；
   - 该帧不下发（hold 上一帧目标），等下一次推理或下一帧 chunk；
   - 连续 3 帧异常 → `emergency_stop`。

  **C) 任务先验（必做）**：
   open-the-door 任务 EEF 朝向基本水平，理论上不会接近 pitch = ±90°；
   如果上线后 WARNING 频繁触发，说明：
   1. dataset 坐标系定义跟 SDK 不同；或
   2. SDK 返回的 quat 已经在 gimbal 区。
   两种情况都需要排查标定，**不能简单调阈值**。

  **D) 终极缓解（如 A/B 不够，再考虑实现，本期不实施）**：
   在 chunk 接近 gimbal 区域时切到四元数 slerp delta，
   但这会破坏跟训练侧的 cumsum 等价性，**仅作 future work，本期明确不实现**。

- **回滚**：watchdog 检测到 rpy 帧间跳变 > π/2 → 急停（实质就是 B 的兜底）。

### 风险 4 — WS 推流帧率波动
- **现象**：上游 rgbd_ws_bridge 实际推流不严格等间隔，偶尔 100 ms+ 空隙。
- **原因**：H.264 关键帧周期、上游编码器抖动、网络抖动。
- **影响**：中。模型输入用陈旧帧 → 动作滞后。
- **缓解**：`frame_max_age_ms=200` 卡控；InferLoop 内若图像 > 阈值，跳过本次推理（chunk 继续耗尽，hold 兜底）。
- **回滚**：调大 max_age 或调小推理频率。

### 风险 5 — chunk 时间戳锚点漂移
- **现象**：DispatchLoop 用 `base_capture_ts_ns + idx*50ms` 对齐时间，若 `base` 取错（用 server local time 而不是图像 capture ts），不同 chunk 间会出现「时间跳跃」。
- **原因**：实现混淆「推理时刻」「图像采集时刻」「下发时刻」。
- **影响**：高。混合区会出现非物理跳变。
- **缓解**：**强制规定** `chunk.base_capture_ts_ns = image.capture_ts_ns`（不是 ARM state，不是 server now），并在代码里做 invariant 检查。日志里同时打三个时间戳便于核对。
- **回滚**：发现漂移时切到「不混合，硬切换」模式做对照。

### 风险 6 — delta_rpy 是分量差而非物理 delta（**核心讨论点**）
- **现象**：训练数据 `delta_rpy[t] = rpy[t] - rpy[t-1]`，不是 `R_{t-1}^{-1} @ R_t`。Server 现 `_delta_to_absolute` 用 `cumsum` 反积分，跟训练侧严格一致 → 不要改。
- **混合区问题**：因为 rpy 是「分量累加」语义而不是真物理欧拉，重叠区在 rpy 空间做 lerp 也是「同语义内的线性混合」，理论上没问题。但**前提**是两条 chunk 的 rpy 在混合窗口内没有 ±π 跳变。
- **影响**：中。
- **缓解**：(a) 混合前对 [old_chunk.rpy, new_chunk.rpy] 做联合 unwrap（统一参考帧）；(b) 给两条 rpy 序列限制最大跳变 π/4，超过则放弃混合，硬切；(c) 在 `closed_loop.py` 内加单测：手工构造两条带 -π 跳变的序列，验证 unwrap+lerp 仍连续。
- **回滚**：`--blend-frames=0` 关闭混合。

### 风险 7 — lease 续期失败
- **现象**：arm_sdk lease 默认 5 s，server 内部定时器若续期失败（network / SDK 异常），后续 RPC 会被拒。
- **原因**：lease 续期超时 / SDK 内部状态异常。**注意**：本风险仅讨论「同一个 server 进程内的 lease 续期问题」，不涉及多 HTTP 客户端 / 多 controller 抢占（FastWAM 部署模型是单 server 单 controller，多 HTTP 客户端竞争 `/start` 不在本设计范围内）。
- **影响**：高。机械臂仍在执行 server 的前一条命令，但 server 已失控。
- **缓解**：(a) lease 续期定时器，周期 `--lease-renew-ms`（默认 4000，即 lease 5 s 的 80%）；(b) 单次续期失败立即重试一次，仍失败则进入 hold；连续 2 次失败 → 急停；(c) RPC 失败（move_end_pose / move_eef 返回 False 或抛权限异常）立刻急停；(d) `acquire_control` 启动时调用一次，不做 retry。
- **回滚**：手动急停 + 重启 server。

### 风险 8 — 模型推理时延（已实测，仍需运行期监控）
- **现象**：实测 `FastWAMModelClient.infer @ 10 steps` 在 5090 GPU 上 p50 167.5 ms / p99 192.5 ms（Agent I，n=30，含图像预处理）。但**实测是在 GPU 空闲 + 同卡独占**条件下取得；运行期若 GPU 被其它进程抢占（5090_1 上 GPU 0 已被 `airdc` 占 95%，**本服务必须用 GPU 1**）或显存碎片化，时延可能跳到 250+ ms。
- **原因**：单步 diffusion ~13 ms + 固定 ~42 ms（线性拟合 R²=0.9993），10 步 ~173 ms 是设计点。`num_inference_steps` 是最敏感的杠杆。
- **影响**：中。当前预算 400 ms 推理周期下余 ~230 ms，余量充足；但若 > infer_period_ms (400 ms)，链路会反复 hold。
- **缓解**：
  1. **启动 warmup + benchmark**：server 启动时 warmup 5 次 + 实测 10 次推理耗时（用 dummy 输入），把 p50/p95/p99 写入启动日志（见 5.6.1）；
  2. **强制使用 GPU 1**：CLI 默认 `CUDA_VISIBLE_DEVICES=1`（或 `--device cuda:1` 直接指定），避开 5090_1 上 GPU 0 的常驻 airdc 进程；启动时检查显存 < 80% 占用，否则 WARN；
  3. **`/infer` 打点**：在 handler 加 `time.perf_counter` 打点（每次 ≤ 5 字段：`preprocess / infer / denorm / format / total`），用 `logger.info`；
  4. **启动日志告警阈值**：
     - p50 > 220 ms（实测 + 30% margin）：log WARNING，提示 GPU 可能被抢占或 num_inference_steps 设高了；
     - p50 > 350 ms：log WARNING + 建议降 `--num-inference-steps`（10 → 6 可省 ~50 ms，到 ~117 ms）；
     - p50 > 500 ms：log ERROR 并 `sys.exit(1)`，建议检查 ckpt / GPU 状态；
  5. **运行期监控**：单次推理 > `2 × infer_period_ms` (800 ms) → WARN，> 5× (2000 ms) → 进入 hold；连续 3 次超 `infer_period_ms` → emergency_stop；
  6. **调优空间**（备用）：`num_inference_steps=6` 时 mean 116.8 ms，端到端可降到 ~175 ms。若上线后发现 quality 仍足够，可视情况下调。
- **回滚**：调大 `--infer-period-ms`，降低 `--num-inference-steps`，或减小 chunk_len（需重训）。

### 风险 9 — 坐标系一致性（world vs base）
- **现象**：训练时的 6D pose 是哪个坐标系（机器人 base？world？相机？）暂未在本仓库文档中明确；SDK `get_end_pose` 返回的是哪个坐标系也需核。
- **原因**：训练数据 pipeline 在外部仓库，转换链路长。
- **影响**：高。坐标系错配 = 直接漂飞。
- **缓解**：(a) 上线前做「读 pose → cumsum 0 delta 32 帧 → 下发」零位测试；(b) 若 0 delta 下机械臂不动，可以认为坐标系一致；(c) 文档加坐标系核验 checklist。
- **回滚**：发现错配 → `--auto-dispatch=false` 退场。

### 风险 10 — 急停反应窗口
- **现象**：watchdog 10 ms 周期 + RPC 约 1 ms + `set_arm_emergency_stop` 阻塞 0.15 s，最坏约 165 ms 才生效。同时 dispatch 还在以 50 ms 一发下命令。
- **原因**：watchdog 与 dispatcher 异步。
- **影响**：中。
- **缓解**：(a) watchdog 触发急停前先把 dispatcher 的 `auto_dispatch` 标志置 False（同步原子操作），dispatcher 下一槽不再发；(b) 急停期间忽略所有 send 失败的「重试」逻辑；(c) `set_arm_emergency_stop` 阻塞期间 InferLoop 也暂停。
- **回滚**：物理按钮兜底（部署侧职责）。

### 风险 11 — WS 上游 RGBD 节点不稳定
- **现象**：上游 rgbd_ws_bridge 历史上经常无数据，需要手动重启上游服务；运行中也可能突然卡顿、丢帧、fps 抖动到期望值的 50% 以下。
- **原因**：上游 rgbd_ws_bridge 本身的健壮性问题，FastWAM 不负责修复，只能在客户端做防御。
- **影响**：高。WS 是闭环的关键输入，长时间陈旧帧 → 模型生成滞后动作 → 实际位姿偏离目标。
- **缓解**（详细策略见 5.4.1 节）：
  1. **启动自检**：30 s 内未收到首帧 → abort；首帧 codec / shape / channel 名校验不过 → abort；
  2. **5 s 滑动窗口监测**：fps < 期望 70% / 解码失败 / pair_seq 跳跃 → log WARNING；
  3. **运行期 watchdog**（每 100 ms 检查 `last_frame_age_per_channel_ms`）：
     - `> --ws-warn-stale-ms`（200 ms）：log WARNING；
     - `> --ws-hold-stale-ms`（500 ms）：dispatcher 进入 hold mode（暂停下发）；
     - `> --ws-estop-stale-ms`（1500 ms）：触发 `emergency_stop`；
  4. **指标暴露**：`/health` + `/ws_status` 报告 `last_frame_age_per_channel_ms / fps_5s / decode_fail_count / reconnect_count`；
  5. **退避重连**：socket 异常 / 心跳超时按 `--ws-reconnect-backoff-ms` 退避列表重连，重连成功后请求关键帧。
- **回滚**：手动 `--enable-self-fetch=false`，退回客户端推图模式；同时排查 / 重启上游 rgbd_ws_bridge。

### 风险 12 — chunk 跑完保持模式 vs SDK 命令队列
- **现象**：SDK `move_end_pose(..., blocking=False)` 返回后命令是否真已执行完成？hold 模式下重发 last_pose 是否会与 SDK 内部仍未完成的旧命令冲突？
- **原因**：SDK 内部命令队列行为不完全透明。
- **影响**：中。
- **缓解**：(a) hold 时把 arm_speed 设为非常小（如 π/12 rad/s），让 SDK 自然停在 last_pose；(b) 不主动重发 last_pose，而是「不发新命令」配合小速度；(c) 上线前在 dry-run 测一次 hold 行为。
- **回滚**：hold = `set_arm_emergency_stop(True)` 直接急停（更激进但安全）。

### 风险 13 — `mixtures.*` namespace 加载兼容性
- **现象**：默认 ckpt `runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/checkpoints/step_020000.pt` 已确认为新 namespace 格式，且 step_015000 已通过验证。
- **原因**：历史上 ckpt key prefix 变更过。
- **影响**：低（已 mitigated）。
- **缓解**：保持现 `FastWAMModelClient` 加载链路不变；本次改造不动模型加载。
- **回滚**：N/A。

### 风险 14 — text cache 仅 1 个 entry
- **现象**：当前 text cache 只有 hash `243062ca...` 一个 entry，对应 `"open the door"`。切换 instruction 会 500。
- **原因**：text encoder 离线预计算，按 prompt hash 落盘。
- **影响**：中（运营层面）。
- **缓解**：(a) `/start` 若收到未 cache 的 instruction，server 返回 4xx 明确错误，提示「先跑 precompute_text_embeds.py」；(b) 文档 + 部署 README 列清楚；(c) 后续考虑 server 启动时自动 precompute（独立 PR，不在本次范围）。
- **回滚**：限定 instruction 列表，运维白名单。

### 风险 15 — `output_action_format = cartesian_absolute` vs `action_mode = delta6_abs_gripper`
- **现象**：模型输出是「delta6 + abs_gripper」，反积分后是「cartesian_absolute」。两个字段含义易混淆，可能在某处 transform 误用。
- **原因**：命名历史遗留。
- **影响**：中。
- **缓解**：(a) 改造时不动现有 `_delta_to_absolute` 逻辑，仅在其外加一层 `send_pose` 适配（rpy→quat）；(b) 单元测试覆盖反积分 + 转换链路；(c) 注释里写清两个字段语义。
- **回滚**：N/A（逻辑不动）。

---

## 9. 回滚策略

- **L1（最轻）**：`--auto-dispatch=false`。Server 仍跑推理 + 打日志，但不下发命令。任何疑问优先切到这一档观察。
- **L2**：`--enable-self-fetch=false`。完全退回旧 client-push 模式，行为与历史版本字节级一致。
- **L3（最重）**：删除 `src/fastwam/server/` 目录 + 还原 `scripts/fastwam_http_server.py` 即恢复到改造前。所有新代码隔离在独立目录，无侵入。
- **CLI 默认值**：首次上线 `--enable-self-fetch=false` + `--auto-dispatch=false`。验证稳定后逐档放开。

---

## 10. 开发顺序建议（PR 拆分）

按风险递增：

1. **PR1 — perf_counter 打点**：仅在现 `scripts/fastwam_http_server.py` 加 timing 字段，零风险，立刻可上。
2. **PR2 — rotation 工具 + 单测**：纯函数，零运行时影响。
3. **PR3 — ArmClient（poller + state cache，不 dispatch）**：只读 RPC，无副作用。挂到 `/health` 做观察。
4. **PR4 — WSFrameIngester 骨架**：只 latest()，不接 InferLoop。挂到 `/health` 报告。
5. **PR5 — `/infer` fallback 用 ingester/arm_client**：仍由 client 触发，但缺字段自取。先在内部测试桩上验证。
6. **PR6 — ClosedLoopRunner（dry-run）**：`--auto-dispatch=false`，跑推理 + 写 chunk_cache + 计算应播帧，仅日志。
7. **PR7 — dispatch 接入**：开 `--auto-dispatch=true`，加 watchdog；先单帧测试再连续。
8. **PR8 — 急停联调 + 上线运行手册**：完整覆盖 6 节所有错误事件。

---

## 11. 未解项

- **WS 实测通道名**：另一个 agent 正在跑 ws probe，未回写。`--ws-channel-map` 默认值待确认。
- **模型推理时延实测**：需要 PR1 落地 + server 启动 benchmark 跑一次 calibration（见风险 8），再回填 3 节延迟分解表。
- **坐标系一致性（world vs base）**：上线前必须做零位测试（读 pose → cumsum 0 delta 32 帧 → 下发，机械臂应不动）。
- **gimbal lock 实际触发概率**：要在 open-the-door 任务真实工位下采集 pitch 分布（运行期 dispatcher 已有 detection，见风险 3）。
- **SDK lease 续期 API 具体名称**：需对照 `airbot_example_record_and_replay.py` 确认；先在 `ArmClient` 预留 timer 槽位。
- **mcap_preprocess_pipeline commit hash**：rotation.py 文件头注释里需要填上游欧拉约定来源的 commit hash，首次部署时通过 ssh 取（见风险 1）。
