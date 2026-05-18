# FastWAM Active Loop Server — 机械臂控制集成指南

> 配套文件：
> - 设计稿：`docs/fastwam_http_server_self_fetch_design.md` §7.3 / §7.6 / §8
> - ArmClient 子模块指南：`docs/airbot_sdk_cartesian_armclient_guide.md`
> - 总览运维文档：`docs/fastwam_http_server.md`（Active Loop Server v2 章节）
> - SDK 版本：airbot-arm 5.2.3 / arm_sdk 0.1.0.dev51+

## 0. 这份文档解决的问题

`scripts/fastwam_active_loop_server.py` 是闭环 server，但**模型 inference 的 [T,7] action chunk 到底怎么走到机械臂的电机指令？** 中间还隔着 ringbuffer、dispatcher、watchdog、SDK 多层。这份文档从一帧 action 的视角说清楚整个链路 + 关键设计决策 + 出错时如何恢复。

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

## 2. 关键设计决策

### 2.1 控制器必须是 `servo_control`

`acquire_control` 之后必须 `switch_controller(Controller.servo_control)`，否则 `move_end_pose` 行为不可预测。

| Controller | 行为 | 适合 v2 server | 原因 |
| --- | --- | --- | --- |
| `direct_control` | 直发关节力矩，无平滑 | ❌ | 50 ms 相邻帧之间会抖；不接 `move_end_pose` |
| `servo_control` ✅ | 250 Hz 内部平滑，接受 streaming 末端目标 | ✅ | v2 唯一支持 |
| `planning_control` | 内置轨迹规划 | ❌ | 不接受高频流式输入 |
| `mit_control` | 力位混合 | ❌ | SDK 文档明确"慎用" |

启动 server 时 ArmClient.acquire_control() 内部按顺序做：

```python
client.acquire_control(lease_ms=15000, renew_period_s=5.0)  # 1. 拿 lease，SDK 自动续期
client.switch_controller(Controller.servo_control)          # 2. 切控制器
client.set_arm_speed([arm_speed_rad_s] * 6)                 # 3. 每关节速度上限
```

启动日志要看到：

```
[arm] poller started host=192.168.31.34 port=50051 hz=50.0
[arm] switched to servo_control
[arm] set_arm_speed=0.500 rad/s per joint
```

任何一步失败 → `acquire_control` 返回 False → `POST /start` 拒绝。

### 2.2 `move_end_pose` 与夹爪必须**一次 RPC**

这是文档要解决的第一个"易踩坑"。

```python
# ❌ 错误（任一线程把另一个打断，机械臂动作不可预测）
client.move_end_pose(pose, options)
client.move_eef(gripper_m, options)

# ✅ 正确（夹爪目标通过 options 顺带）
options = ArmControlOptions()
options.eef_pos = gripper_m   # 关键
options.blocking = False
client.move_end_pose(pose, options)
```

**为什么**：SDK 文档 p.31 明确警告 —— `servo_control` + `blocking=False` 下，**第二个 RPC 会中断第一个未完成的任务**。一次 RPC 内部把"末端 + 夹爪"打包发出，SDK 才能保证两者同时到达且不互相抵消。

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

## 3. 安全机制：4 层防御

从最快到最慢：

| 层 | 触发条件 | 响应时间 | 响应动作 |
| --- | --- | --- | --- |
| **L1 dispatch tick 内联** | rpy 帧间跳变 > π/4 / `\|pitch\|` > 85° | < 1 ms | 本帧跳过；连续 3 帧 → emergency_stop |
| **L2 dispatch.set_hold(True)** | watchdog 通知 | ≤ 50 ms | 停发新命令，机械臂在 servo 平滑下自然停 |
| **L3 watchdog 周期检查** | chunk_stale / WS stale / ARM RPC RED / infer 心跳超时 | ≤ 10 ms | set_hold / 升级 emergency_stop |
| **L4 emergency_stop** | watchdog 决策 / send_pose 异常 / 物理按钮 | ≤ 165 ms | 调 `set_arm_emergency_stop(True)`，dispatcher 同步停 |

**响应窗口最坏 165 ms**：watchdog 10 ms 周期 + 1 ms RPC + 150 ms `set_arm_emergency_stop` SDK 阻塞。在这段时间内 dispatcher 可能还在发命令，所以 L4 触发前先把 `auto_dispatch` 标志原子置 False（dispatcher 下一槽就不发了）。

### 3.1 急停的两种来源

```
来源 1: watchdog 自动触发
   chunk_stale  →  hold
   WS stale > 1500 ms  →  emergency_stop
   ARM RPC RED > 200 ms  →  emergency_stop
   推理连续超时 3 次  →  emergency_stop

来源 2: 客户端 POST /emergency {"enable": true}
   人工触发，等价于 set_arm_emergency_stop(True)
```

### 3.2 急停后恢复

```bash
# 1. 调清急停标志（必须）
curl -X POST http://server:8118/emergency -d '{"enable": false}'

# 2. 重新启动闭环（必须）
curl -X POST http://server:8118/start -d '{}'
```

`set_arm_emergency_stop(False)` 不会自动 `acquire_control`，所以必须重发 `/start`（handler 内部会调 acquire_control）。watchdog 自动急停的 case，强烈建议**重启 server 进程**而不是软恢复 —— 因为 watchdog 之前可能已经积累了 RED 状态。

## 4. ArmControlOptions 各字段配置

| 字段 | 默认 | 单位 / 范围 | 用途 | v2 server 设值 |
| --- | --- | --- | --- | --- |
| `blocking` | `False` | bool | 阻塞调用 | **必须 False**（流式下发） |
| `eef_pos` | 0.0 | 米 G2: [0, 0.072] / G2T: [0, 0.090] | 夹爪目标位置 | 每帧从 chunk 取 |
| `eff` | 8.0 | 安培 | 每关节电流阈值 | SDK 默认（够用） |
| `eef_eff` | 8.0 | 安培 | 夹爪电流阈值 | 可下调到 6.0 增加安全裕量 |

SDK 内部会 clamp `eef_pos` 到夹爪 spec 范围（`client.py:1654-1655`），所以即便模型输出超界（理论不会，因为 dataset 训练时就是 [0, 0.0904] 米范围）也不会损坏硬件。

## 5. 启动前 checklist

按 design doc §11 + ArmClient guide §9 整理，**任何一项不过不要继续**：

1. **GPU 选 cuda:1** —— `5090_1` 上 cuda:0 被 AirDC 占着。`--device=cuda:1`（默认）。
2. **WS 上游先启** —— 启 server 时 `rgbd_ws_bridge` 必须已经在推帧（19095 端口）。30 秒等不到首帧 → server abort。
3. **ARM 服务可达** —— `192.168.31.34:50051` 必须开。
4. **text cache 命中** —— `data/text_embeds_cache/real_1048/<sha>.t5_len128.wan22ti2v5b.pt` 必须存在，否则 `/start` 返回 500。
5. **零位 dry-test** —— **每次新工位 / 新标定 / 新模型必跑**：
   ```bash
   curl -X POST http://server:8118/debug/zero_pose_test -d '{"duration_s": 5.0}'
   ```
   机械臂应**完全不动**。任何漂移 = 坐标系/欧拉约定不对，立刻 abort + diagnose。
6. **首次正式启动 `--auto-dispatch=false`** —— 跑推理 + 写日志，但不调 `send_pose`。看 `closed_loop_status` 中 ringbuffer 写入频率、dispatcher last_pose_target 合理后再开 `--auto-dispatch=true`。

## 6. 端到端最小可工作 demo

```bash
# 一台 shell: 启 server
ssh 5090_1
cd /home/Lyle/Projects/FastWAM
export NO_PROXY=192.168.31.66,192.168.31.34,127.0.0.1,localhost
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

## 7. 故障速查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `acquire_control returned False` | 其它客户端持有 lease | 杀掉抢占进程；不要硬抢 |
| `switch_controller(servo_control) returned False` | acquire 后立即 abort | 排查 ARM 服务端日志；可能上一次 lease 残留 |
| `move_end_pose returned False` | 关节超限 / lease 失效 | 查 `arm.health()` `lease_alive` / `consecutive_send_fail` |
| 机械臂动了一下停住 | 漏改 `blocking=False`，或漏带 `options.eef_pos` 双 RPC 互打断 | grep 看 `send_pose` 实现是否是单次 RPC |
| 夹爪一直不动 | `options.eef_pos` 没传 | 看 dispatcher last_pose_target 中 gripper 字段 |
| rpy 突然 2π 跳 + 位姿乱飞 | quat 帧间符号翻转，未 unwrap | poller 应该有 `unwrap_quat_sign`，看 `arm.health()` 的 `last_quat_unwrap_count` |
| pitch 接近 ±90° 姿态乱跳 | gimbal lock | dispatcher 应触发 hold（`|pitch|`>85°）；排查任务工位是否合理 |
| `chunk_stale` 一直 hold | 推理超时 / 推理线程死 | 查 `infer_latency_ms_p99`、GPU 占用、ckpt 加载状态 |
| WS 频繁 reconnect | 上游 `rgbd_ws_bridge` 不稳 | 跑 `scripts/fastwam_ws_probe.py` 单测；重启上游 |
| 急停后 `/start` 仍返回 503 | 没发 `/emergency {"enable": false}` 解锁 | 按 §3.2 流程 |

## 8. 相关文件锚点

```
src/fastwam/server/arm_client.py        ArmClient + poller + send_pose
src/fastwam/server/dispatch.py          DispatchLoop (20 Hz)
src/fastwam/server/watchdog.py          Watchdog (100 Hz)
src/fastwam/server/closed_loop.py       ClosedLoopRunner + InferLoop (2.5 Hz)
src/fastwam/server/chunk_ringbuffer.py  ChunkRingBuffer (capacity=2, FIFO drop oldest)
src/fastwam/server/rotation.py          quat <-> rpy + unwrap (extrinsic XYZ scipy)
scripts/fastwam_active_loop_server.py   HTTP 入口 + endpoints + 启动 banner
docs/airbot_sdk_cartesian_armclient_guide.md  ArmClient 实现指南（含 SDK p.31 警告）
docs/fastwam_http_server_self_fetch_design.md 完整设计稿（§7.3 / §7.6 / §8）

.venv/lib/python3.10/site-packages/arm_sdk/client.py
   Controller enum:     line 31
   ArmControlOptions:   line 89  (eef_pos: line 159)
   servo_control 分支:  line 1650 (move_end_pose 内部塞 eef_pos: line 1668)
```

## 9. 已知局限 / future work

- **dispatch 仍依赖 wall clock** `time.time_ns()`。若机器 NTP 跳秒，可能出现 idx 漂移。当前无 monotonic mode（design risk #5 提到的兜底）。
- **gimbal lock 阈值（85°）是经验值**。open-the-door 任务 EEF 朝向基本水平，理论不接近 ±90°；future work 是 chunk 接近时切到 quaternion slerp。
- **text cache 只有 1 条 entry**（"open the door"）。切 instruction 必须先离线跑 `precompute_text_embeds.py`。
- **lease 续期失败的兜底**目前是直接 emergency_stop。future work 可以加一次自动 reacquire 重试。
