# FastWAM Active Loop Server — 机械臂集成参考

> 范围：`scripts/fastwam_active_loop_server.py` v2 主动模式下，从模型输出到 AIRBOT 电机的完整链路；ArmClient 子模块（`src/fastwam/server/arm_client.py`）的接口契约、控制器选型、加固项与故障处置。本文档合并自原 `airbot_sdk_cartesian_armclient_guide.md`（实施 spec）与 `fastwam_active_loop_arm_integration.md`（v2 运维手册），是 ArmClient 与 v2 server 的**唯一权威参考**。
>
> 配套：
> - 设计稿：`docs/fastwam_http_server_self_fetch_design.md` §5.3 / §7.3 / §7.6 / §8
> - 总览运维：`docs/fastwam_http_server.md`（Active Loop Server v2 章节）
> - SDK 版本：airbot-arm 5.2.3 / arm_sdk 0.1.0.dev51+
> - SDK PDF 权威：《AIRBOT Arm SDK 开发指南》p.5（关节限位）/ p.21（控制器与接口矩阵）/ p.31（同步控夹爪警告）

---

## 1. 总览：模型输出到电机的 5 步

```
+----------------------------+
| FastWAMModelClient.infer   |  每 400 ms 一次 (2.5 Hz)
| 输出: [32, 7] actions      |  含义: 32 帧 (xyz_m, rpy_rad, gripper_m) 绝对位姿
+--------------+-------------+
               |
               v
+----------------------------+
| ChunkRingBuffer.push       |  容量 2，满则丢最旧 (FIFO)
| 锚点 base_capture_ts_ns    |  = head_left.stamp_ns (上游采集时刻)
+--------------+-------------+
               |
               v
+----------------------------+
| DispatchLoop  @ 20 Hz      |  每 50 ms 一次 tick
| - 算 idx = round((now-base)/50ms)
| - blend 4 帧 (rpy 联合 unwrap)
| - gimbal / rpy_jump guard
| - 转 quat
+--------------+-------------+
               |
               v
+----------------------------+
| ArmClient.send_pose        |  单次 RPC，不可分两次！
| - 组 CartesianPose(xyz,quat)
| - 组 ArmControlOptions(eef_pos=gripper_m)
| - move_end_pose(pose, options)
+--------------+-------------+
               |
               v
+----------------------------+
| arm_sdk → gRPC → AIRBOT    |  servo_control 控制器
| 内部 250 Hz 平滑滤波       |  相位滞后 ~20-40 ms
+----------------------------+
```

**全程目标延迟（捕获 → 第一帧下发）≈ 225 ms**，预算余量 ~175 ms。

ArmClient 作为 self-fetch HTTP server 与机械臂之间的**唯一边界**，对外只暴露三件事：
1. 后台 50 Hz 状态轮询 → `state_cache`（供 `ClosedLoopRunner` 推理时快照）；
2. 笛卡尔末端 + 夹爪的**单点下发**（供 `Dispatcher` 20 Hz 调用）；
3. lease 续期、急停、health 报告。

**不负责**：图像、推理、chunk 反积分、rpy unwrap（这些在 `closed_loop.py` / `rotation.py` 里）。

---

## 2. 关键设计决策

### 2.1 控制器必须是 `servo_control`

`acquire_control` 之后必须 `switch_controller(Controller.servo_control)`，否则 `move_end_pose` 行为不可预测。SDK PDF p.21 控制器矩阵：

| Controller | 行为 | 适合 v2 server | 原因 |
| --- | --- | --- | --- |
| `direct_control` | 直发关节力矩，无平滑 | 否 | 50 ms 相邻帧之间会抖；不接 `move_end_pose` |
| `servo_control` | 250 Hz 内部平滑，接受 streaming 末端目标 | 是 | v2 唯一支持，可接 `move_end_pose` 同时控夹爪 |
| `planning_control` | 内置轨迹规划 | 否 | 不接受高频流式输入 |
| `mit_control` | 力位混合 | 否 | SDK PDF 明确"慎用" |

启动 ArmClient 时顺序：

```python
client.acquire_control(lease_ms=15000, renew_period_s=4.0)  # 1. 拿 lease，SDK 自动续期
client.switch_controller(Controller.servo_control)          # 2. 切控制器
client.set_arm_speed([arm_speed_rad_s] * 6)                 # 3. 每关节速度上限
client.set_eef_speed(eef_speed_m_s)                         # 4. 夹爪速度上限
```

启动日志要看到：

```
[arm] poller started host=192.168.31.34 port=50051 hz=50.0
[arm] switched to servo_control
[arm] set_arm_speed=0.500 rad/s per joint
```

任何一步失败 → `acquire_control` 返回 False → `POST /start` 拒绝（503）。

### 2.2 `move_end_pose` 与夹爪必须**一次 RPC**

这是整个模块第一个"易踩坑"。SDK PDF p.31 原话：

> 在 servo_control 控制器下进行笛卡尔空间或关节空间控制时，末端执行器的运动会受到 `options.eef_pos` 参数的影响。当采用 `options.blocking = False` 即非阻塞方式时，**如果同时调用 `move_end_pose`（或 `move_joint`）与 `move_eef` 接口，后一个指令会中断前一个任务的执行**。

```python
# 错误（任一线程把另一个打断，机械臂动作不可预测）
client.move_end_pose(pose, options)
client.move_eef(gripper_m, options)

# 正确（PR8：夹爪目标通过 options 顺带）
options = ArmControlOptions()
options.eef_pos = gripper_m   # 关键
options.blocking = False
client.move_end_pose(pose, options)
```

SDK 源码 `arm_sdk/client.py:1650-1668` 也证实：servo_control 分支下，`move_end_pose` 内部直接把 `options.eef_pos` 作为夹爪目标一起序列化进 RPC payload。

### 2.3 时间锚点：`base_capture_ts_ns = head_left.stamp_ns`

DispatchLoop 计算 `idx` 时用：

```python
idx = round((time.time_ns() - chunk.base_capture_ts_ns) / chunk.step_dt_ns)
```

这里的 `chunk.base_capture_ts_ns` **不是**推理时刻、**不是**机械臂状态采集时刻、**不是** dispatch 时刻 —— 是上游 WS bridge 中 `channel.meta['stamp_ns']`，即**摄像头采集那一刻**。三个 server 内部时钟域必须对齐（都是 wall clock `time.time_ns()`），所以 PR3 的 `FrameSnapshot.capture_ts_ns` 必须保持 wall clock 不能切到 monotonic。

实测验证：probe 抓到 `stamp_ns ≈ 1779104237318198519 ns` ≈ 2026-05-18 UTC，符合 wall clock 量级，与 `time.time_ns()` 同 nanosecond 量级。

### 2.4 ringbuffer 满则丢最旧（不是丢最新）

InferLoop 2.5 Hz 推理一次（每 400 ms），DispatchLoop 20 Hz 接管（每 50 ms × 32 = 1600 ms / chunk）。理论上正常运行时 ringbuffer 不会满（第 2 次推理到达时，第 1 个 chunk 已被 dispatcher 走完一大半）。

但若推理偶尔变慢（GPU 抢占、显存压力），可能出现"第 3 次推理结果到达但前两个 chunk 都还没耗尽"。这时丢最旧的（FIFO 满则丢头），保证 dispatcher 总是用最近一次推理的结果。

```python
ChunkRingBuffer(capacity=2)
# 满了时 push 返回被丢的最旧 entry，logger 打印
[runner] ringbuffer evicted chunk_id=42 (new=44)
```

---

## 3. ArmClient 接口契约

### 3.1 状态数据结构

```python
class ArmStateSnapshot(NamedTuple):
    angles_rad: np.ndarray       # (6,)  关节角，弧度
    gripper_m: float             # 夹爪开度，米（G2: 0~0.072）
    eef_xyz: np.ndarray          # (3,)  末端笛卡尔位置，米
    eef_rpy: np.ndarray          # (3,)  extrinsic XYZ，已 unwrap + canonicalize
    eef_quat_xyzw: np.ndarray    # (4,)  原始四元数，符号已 canonicalize
    capture_ts_ns: int           # poller 取到该状态的本地时间戳
```

`eef_rpy` 在写入 cache 前完成：
1. `unwrap_quat_sign(q_new, q_prev)` —— quat 帧间符号 unwrap；
2. `as_euler("xyz", degrees=False)` —— scipy extrinsic XYZ；
3. `unwrap_rpy_sequence` —— rpy 域 ±π 兜底 unwrap。

### 3.2 50 Hz Poller

```python
def _poll_once(self) -> ArmStateSnapshot | None:
    joint = self.client.get_arm_joint_state()
    eef   = self.client.get_eef_joint_state()
    pose  = self.client.get_end_pose()
    if joint is None or eef is None or pose is None:
        self._consecutive_fail += 1
        return None
    self._consecutive_fail = 0

    q = np.array(pose.orientation, dtype=np.float32)            # (qx,qy,qz,qw)
    if self._last_quat is not None:
        q = unwrap_quat_sign(q, self._last_quat)
    self._last_quat = q

    rpy = R.from_quat(q).as_euler("xyz", degrees=False)
    if self._last_rpy is not None:
        rpy = unwrap_rpy_pair(self._last_rpy, rpy)
    self._last_rpy = rpy

    return ArmStateSnapshot(
        angles_rad=np.asarray(joint.angles, dtype=np.float32),
        gripper_m=float(eef.eef_pos),
        eef_xyz=np.asarray(pose.position, dtype=np.float32),
        eef_rpy=rpy.astype(np.float32),
        eef_quat_xyzw=q,
        capture_ts_ns=time.monotonic_ns(),
    )
```

**读取接口**（带新鲜度检查）：

```python
def latest(self) -> ArmStateSnapshot | None:
    with self._lock:
        if self._snapshot is None:
            return None
        age_ms = (time.monotonic_ns() - self._snapshot.capture_ts_ns) / 1e6
        if age_ms > self.state_max_age_ms:                      # 默认 100ms
            return None
        return self._snapshot
```

### 3.3 单点下发 `send_pose`

```python
def send_pose(
    self,
    target_xyz: np.ndarray,        # (3,) 米
    target_rpy: np.ndarray,        # (3,) extrinsic XYZ，弧度
    gripper_m: float | None,       # 米；None 表示沿用上次值
) -> bool:
    # PR11 R14: sanity 检查（|xyz|<=1m / |rpy|<=π / gripper∈[0,0.15]）超阈值 raise ValueError
    self._validate_pose(target_xyz, target_rpy, gripper_m)

    q = R.from_euler("xyz", target_rpy).as_quat()               # scipy: [x,y,z,w]
    pose = CartesianPose(
        position=(float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])),
        orientation=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
    )

    options = ArmControlOptions()
    options.eff = self._eff_thresholds          # e.g. [8.0] * 6
    options.eef_eff = self._eef_eff_threshold   # e.g. 6.0
    options.blocking = False                    # 流式下发必须 False
    if gripper_m is not None:
        options.eef_pos = float(gripper_m)      # PR8: 通过 options 顺带下发夹爪

    ok = self.client.move_end_pose(pose, options, timeout_ms=200)
    if not ok:
        # PR11 R15: lease 被抢时自动 reacquire 一次
        if not self._lease_alive():
            self._logger.warning("[arm] lease lost, attempting reacquire")
            if self._reacquire():
                ok = self.client.move_end_pose(pose, options, timeout_ms=200)
        if not ok:
            self._send_fail += 1
            return False
    self._send_fail = 0
    return True
```

### 3.4 PR8-PR11 加固项汇总

| 加固 | PR | 行为 |
| --- | --- | --- |
| 单次 RPC（end_pose + eef_pos 合并） | PR8 | `move_end_pose(pose, options)`，禁止再分两次 |
| `/emergency` 先 set_hold 再急停 | PR9 R3 | dispatcher 立即停发，再调 `set_arm_emergency_stop` |
| `DispatchLoop.start()` 自动清 `hold_mode` | PR9 R11 | 软恢复 `/start` 不必硬重启 |
| `send_pose` sanity 入参校验 | PR11 R14 | xyz/rpy/gripper 越界 raise ValueError，server 在调 SDK 前拦 |
| lease 丢失自动 reacquire 一次 | PR11 R15 | 含 `switch_controller(servo_control)` + `set_arm_speed`；失败才 bump `consecutive_send_fail` |
| `health()` 三档 + lease_alive | PR11 R6/R7 | GREEN/YELLOW/RED；暴露 `lease_renew_count` / `last_quat_unwrap_count` |

### 3.5 ArmControlOptions 字段表

| 字段 | 默认 | 单位 / 范围 | 用途 | v2 server 设值 |
| --- | --- | --- | --- | --- |
| `blocking` | `False` | bool | 阻塞调用 | **必须 False**（流式下发） |
| `eef_pos` | 0.0 | 米 G2:[0,0.072] / G2T:[0,0.090] / G2L:[0,0.010]（PDF p.5） | 夹爪目标位置 | 每帧从 chunk 取 |
| `eff` | 8.0 | 安培 | 每关节电流阈值 | SDK 默认（够用） |
| `eef_eff` | 8.0 | 安培 | 夹爪电流阈值 | 可下调到 6.0 增加安全裕量 |

SDK 内部会 clamp `eef_pos` 到夹爪 spec 范围（`client.py:1654-1655`），所以即便模型输出超界（理论不会，因为 dataset 训练时就是 [0, 0.0904] 米范围）也不会损坏硬件。关节限位 `ARM_JOINT_LIMITS` 见 PDF p.5；超限服务端拒发。

### 3.6 Lease / 急停 / Health

**Lease**：`arm_sdk 0.1.0.dev51+` 实测 `acquire_control(lease_ms=15000, renew_period_s=4.0)` 后台自动续租，无需单独 timer。PR11 R15 之后 `send_pose` 检测到 `_lease_id` 被 SDK 清空时会自动 `acquire_control` 重试一次。

**Emergency stop**：

```python
def emergency_stop(self, enable: bool = True) -> None:
    # PDF: set_arm_emergency_stop 阻塞 ~150ms
    self._logger.warning("[arm] emergency_stop(enable=%s)", enable)
    self._auto_dispatch_flag.clear()              # 同步停 dispatcher
    ok = self.client.set_arm_emergency_stop(enable)
    self._logger.warning("[arm] emergency_stop done, ok=%s", ok)
```

**Health 三档**（PR11 R6/R7）：

```python
def health(self) -> dict:
    snap = self._snapshot
    return {
        "connected": self._connected,
        "controller": "servo_control",
        "poll_hz_actual": self._poll_hz_meter.value(),
        "last_state_age_ms": (time.monotonic_ns() - snap.capture_ts_ns) / 1e6 if snap else None,
        "consecutive_poll_fail": self._consecutive_fail,
        "consecutive_send_fail": self._send_fail,
        "lease_renew_count": self._lease_renew_count,
        "lease_alive": self._lease_alive(),
        "last_quat_unwrap_count": self._quat_unwrap_count,
        "status": "GREEN" if self._consecutive_fail < 3 else ("YELLOW" if self._consecutive_fail < 5 else "RED"),
    }
```

`last_quat_unwrap_count` 触发频率 > 1 Hz 视为异常，watchdog 据此决策。

### 3.7 线程模型

```python
def start(self):
    # 主线程：连接 + acquire + switch_controller + set_speed
    # 后台：启动 poller 线程

def stop(self):
    # 顺序：停 poller → release_control → close()
    self._poller_stop.set()
    self._poller_thread.join(timeout=2.0)
    self.client.release_control()
    self.client.close()
```

- `state_cache` 用 `threading.Lock` 保护，读写一个不可变 `ArmStateSnapshot`；
- `send_pose` 不加锁（SDK 客户端内部已经线程安全；本场景只有 Dispatcher 一个线程发）；
- 急停标志位 `_auto_dispatch_flag` 用 `threading.Event`，保证 watchdog 与 dispatcher 之间同步可见。

---

## 4. 安全机制：4 层防御

从最快到最慢：

| 层 | 触发条件 | 响应时间 | 响应动作 |
| --- | --- | --- | --- |
| **L1 dispatch tick 内联** | rpy 帧间跳变 > π/4 / `\|pitch\|` > 85° | < 1 ms | 本帧跳过；连续 3 帧 → emergency_stop |
| **L2 dispatch.set_hold(True)** | watchdog 通知 | ≤ 50 ms | 停发新命令，机械臂在 servo 平滑下自然停 |
| **L3 watchdog 周期检查** | chunk_stale / WS stale / ARM RPC RED / infer 心跳超时 | ≤ 10 ms | set_hold / 升级 emergency_stop |
| **L4 emergency_stop** | watchdog 决策 / send_pose 异常 / 物理按钮 | ≤ 165 ms | 调 `set_arm_emergency_stop(True)`，dispatcher 同步停 |

**响应窗口最坏 165 ms**：watchdog 10 ms 周期 + 1 ms RPC + 150 ms `set_arm_emergency_stop` SDK 阻塞。在这段时间内 dispatcher 可能还在发命令，所以 L4 触发前先把 `auto_dispatch` 标志原子置 False（dispatcher 下一槽就不发了）。

### 4.1 急停的两种来源

```
来源 1: watchdog 自动触发
   chunk_stale  →  hold
   WS stale > 1500 ms  →  emergency_stop
   ARM RPC RED > 200 ms  →  emergency_stop
   推理连续超时 3 次  →  emergency_stop

来源 2: 客户端 POST /emergency {"enable": true}
   人工触发，等价于 set_arm_emergency_stop(True)
```

### 4.2 急停后恢复

```bash
# 1. 调清急停标志（必须）
curl -X POST http://server:8118/emergency -d '{"enable": false}'

# 2. 重新启动闭环（必须）
curl -X POST http://server:8118/start -d '{}'
```

`set_arm_emergency_stop(False)` 不会自动 `acquire_control`，所以必须重发 `/start`（handler 内部会调 acquire_control）。

> PR9 R3 后 `/emergency` handler 在调 `set_arm_emergency_stop` 前先 `dispatcher.set_hold(True, "emergency")`，dispatcher 不会再发新槽；PR9 R11 后 `DispatchLoop.start()` 会自动清 `hold_mode`，所以软恢复 `/start` 流程仍然可走（不必硬重启）—— 但 watchdog 自动急停的 case **生产环境建议硬重启** server 进程，以重置 watchdog 内部累计计数。

---

## 5. 启动前 checklist + 端到端 demo

### 5.1 checklist（任何一项不过不要继续）

1. **GPU 选 cuda:1** —— `5090_1` 上 cuda:0 被 AirDC 占着。`--device=cuda:1`（默认）。
2. **WS 上游先启** —— 启 server 时 `rgbd_ws_bridge` 必须已经在推帧（19095 端口）。30 秒等不到首帧 → server abort。
3. **ARM 服务可达** —— `192.168.31.34:50051` 必须开。
4. **text cache 命中** —— `data/text_embeds_cache/real_1048/<sha>.t5_len128.wan22ti2v5b.pt` 必须存在，否则 `/start` 返回 500。
5. **零位 dry-test** —— **每次新工位 / 新标定 / 新模型必跑**：
   ```bash
   curl -X POST http://server:8118/debug/zero_pose_test -d '{"duration_s": 5.0}'
   ```
   机械臂应**完全不动**。任何漂移 = 坐标系/欧拉约定不对，立刻 abort + diagnose。零位测试参考实现：
   ```python
   def test_zero_delta_does_not_move(arm: ArmClient):
       snap0 = arm.latest()
       for _ in range(32):                                  # 模拟 chunk_len=32 帧
           arm.send_pose(target_xyz=snap0.eef_xyz, target_rpy=snap0.eef_rpy, gripper_m=snap0.gripper_m)
           time.sleep(0.05)
       snap1 = arm.latest()
       assert np.linalg.norm(snap1.eef_xyz - snap0.eef_xyz) < 2e-3      # < 2 mm
       assert np.linalg.norm(snap1.eef_rpy - snap0.eef_rpy) < np.deg2rad(0.5)
   ```
6. **首次正式启动 `--auto-dispatch=false`** —— 跑推理 + 写日志，但不调 `send_pose`。看 `closed_loop_status` 中 ringbuffer 写入频率、dispatcher last_pose_target 合理后再开 `--auto-dispatch=true`。

### 5.2 端到端最小可工作 demo

```bash
# 一台 shell: 启 server
ssh 5090_1
cd /home/Lyle/Projects/FastWAM
export NO_PROXY=192.168.31.67,192.168.31.34,127.0.0.1,localhost
.venv/bin/python scripts/fastwam_active_loop_server.py \
    --device cuda:1 \
    --auto-dispatch     # 真发！跑前先做零位测试

# 另一台 shell: 控制
# 1. 健康检查
curl http://5090_1:8118/health | jq .

# 2. 零位 dry-test（机械臂应完全不动）
curl -X POST http://5090_1:8118/debug/zero_pose_test \
    -H 'Content-Type: application/json' \
    -d '{"duration_s": 5.0}' | jq .

# 3. 启动闭环
curl -X POST http://5090_1:8118/start \
    -H 'Content-Type: application/json' \
    -d '{"instruction": "open the door"}' | jq .

# 4. 实时看状态
watch -n 0.5 'curl -s http://5090_1:8118/closed_loop_status | jq "{
    chunk_id: .current_chunk_id,
    p50: .infer_latency_ms_p50,
    dispatch: .dispatch.last_dispatch_idx,
    hold: .hold_mode
}"'

# 5. 完成后停
curl -X POST http://5090_1:8118/stop | jq .
```

---

## 6. 故障速查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `POST /start` → 503 `arm.acquire_control failed` | 其它客户端持 lease / `switch_controller(servo_control)` 失败 | 杀掉抢占进程；查 ARM 服务端日志确认是否需要重启 |
| `send_pose ValueError` (xyz/rpy/gripper 越界) | 模型输出超 sanity 阈值（PR11 R14：`\|xyz\|≤1m` / `\|rpy\|≤π` / `gripper∈[0,0.15]`） | server 在调 SDK 之前拦的，看 `runner.status()` 的 `dispatch.last_pose_target` 找异常源头 |
| `move_end_pose returned False` (后跟 reacquire log) | lease 被抢，已自动重试一次（PR11 R15） | 看 `arm.health()` 的 `lease_renew_count` 增长；如频繁触发说明有其它客户端在抢 lease |
| `move_end_pose returned False (after reacquire if attempted)` | reacquire 后仍失败 / 关节超限 | 检查 `consecutive_send_fail`、`status` 三档；看 SDK 服务端日志 |
| 机械臂动了一下停住 | 漏改 `blocking=False` 或两次 RPC 互打断（PR8 后单次 RPC，理论已修） | grep `send_pose` 确认是 `move_end_pose(pose, opts)` 单次调用 |
| 夹爪一直不动 | `opts.eef_pos` 没传 / SDK clamp 到 0 | 看 dispatcher `last_pose_target` 中 gripper 字段，确认 0.0~0.09 之间；参考 SDK PDF p.31 警告 |
| rpy 突然 2π 跳 + 位姿乱飞 | quat 帧间符号翻转，未 unwrap | poller 应该有 `unwrap_quat_sign` + `unwrap_rpy_sequence` 兜底（PR11 R7），看 `arm.health()` 的 `last_quat_unwrap_count` |
| pitch 接近 ±90° 姿态乱跳 | gimbal lock | dispatcher 应触发 hold（`\|pitch\|`>85°）；排查任务工位是否合理 |
| `chunk_stale` 一直 hold | 推理超时 / 推理线程死 | 查 `infer_latency_ms_p99`、GPU 占用、ckpt 加载状态 |
| WS 频繁 reconnect | 上游 `rgbd_ws_bridge` 不稳 | 跑 `scripts/fastwam_ws_probe.py` 单测；重启上游 |
| 急停后 `/start` 仍返回 503 | 没发 `/emergency {"enable": false}` 解锁 | 按 §4.2 流程；PR9 R11 后 dispatch `hold_mode` 在 start() 内会自动清 |
| `health().status == "RED"` | `consecutive_fail/send_fail >= 5` 或 lease 丢失（`lease_alive=False` && `_acquired=True`） | 看具体哪个 counter 高；lease 丢失说明被其它客户端抢占 |
| 启动报 `controller=False` | `switch_controller` 没调或失败 | 启动 banner 打印 controller_state |

---

## 7. SDK 真实行为陷阱 + 文件锚点

### 7.1 SDK 文档与实测对应

- **SDK PDF p.5**：关节限位 `ARM_JOINT_LIMITS` + 夹爪 spec（G2: [0, 0.072] m / G2T: [0, 0.090] m / G2L: [0, 0.010] m）
- **SDK PDF p.21**：控制器与接口矩阵（哪个 controller 支持哪种 move_*）
- **SDK PDF p.31**：servo_control + blocking=False 下，`move_end_pose` 与 `move_eef` 互相中断的同步控夹爪警告 —— 必须用 `options.eef_pos` 单次 RPC
- 上游欧拉约定参考：`/data/home/Lyle/Projects/mcap_preprocess_pipeline/scripts/step01_extract_mcap_rgb_and_params.py:1117`

### 7.2 项目源码锚点

```
src/fastwam/server/arm_client.py        ArmClient + poller + send_pose
src/fastwam/server/dispatch.py          DispatchLoop (20 Hz)
src/fastwam/server/watchdog.py          Watchdog (100 Hz)
src/fastwam/server/closed_loop.py       ClosedLoopRunner + InferLoop (2.5 Hz)
src/fastwam/server/chunk_ringbuffer.py  ChunkRingBuffer (capacity=2, FIFO drop oldest)
src/fastwam/server/rotation.py          quat <-> rpy + unwrap (extrinsic XYZ scipy)
scripts/fastwam_active_loop_server.py   HTTP 入口 + endpoints + 启动 banner
docs/fastwam_http_server_self_fetch_design.md 完整设计稿（§5.3 / §7.3 / §7.6 / §8）
docs/fastwam_http_server.md             v2 总览运维文档
```

### 7.3 arm_sdk 源码锚点（`.venv/lib/python3.10/site-packages/arm_sdk/client.py`）

```
Controller enum:        line 31
ArmControlOptions:      line 89   (eef_pos: line 159)
servo_control 分支:     line 1650 (move_end_pose 内部塞 eef_pos: line 1668)
eef_pos clamp 到 spec:  line 1654-1655
```

---

## 8. 已知局限 / future work

- **dispatch 仍依赖 wall clock** `time.time_ns()`。若机器 NTP 跳秒，可能出现 idx 漂移。当前无 monotonic mode（design risk #5 提到的兜底）。
- **gimbal lock 阈值（85°）是经验值**。open-the-door 任务 EEF 朝向基本水平，理论不接近 ±90°；future work 是 chunk 接近时切到 quaternion slerp。
- **text cache 只有 1 条 entry**（"open the door"）。切 instruction 必须先离线跑 `precompute_text_embeds.py`。
- **lease 续期失败的兜底**：`send_pose` 内部检测到 `_lease_id` 被 SDK 清空时会自动 `acquire_control` 重试一次（含 `switch_controller(servo_control)` + `set_arm_speed`），失败才 bump `consecutive_send_fail` 并 raise。`health()` 暴露 `lease_renew_count` 计数（PR11 R15）。
- **基坐标系 / 欧拉约定核对**（原 design risk #9）：上线前必跑 §5.1 第 5 步零位 dry-test，漂移 > 2 mm 或 > 0.5° 立刻 abort。
