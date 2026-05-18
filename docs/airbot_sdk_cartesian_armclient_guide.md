# ArmClient 子模块实现指南（笛卡尔末端控制）

> 配套设计文档：`fastwam_http_server_self_fetch_design.md` 第 5.3 节
> 目标读者：实现 `src/fastwam/server/arm_client.py` 的开发者
> SDK 版本：airbot-arm 5.2.3 / arm-sdk 0.1.0.dev51+

---

## 0. 模块职责

`ArmClient` 是 self-fetch HTTP server 与机械臂之间的**唯一边界**，对外提供：

1. 后台 50Hz 状态轮询 → `state_cache`（供 `ClosedLoopRunner` 推理时快照）；
2. 笛卡尔末端 + 夹爪的**单点下发**（供 `Dispatcher` 20Hz 调用）；
3. lease 续期、急停、health 报告。

**不负责**：图像、推理、chunk 反积分、rpy unwrap（这些在 `closed_loop.py` / `rotation.py` 里）。

---

## 1. 控制器选择

设计文档 5.3 节 `acquire_control` 内部固定走 `Controller.servo_control`：

```python
self.client = AirbotClient(host=self.host, port=self.port)
self.client.acquire_control(lease_ms=15000, renew_period_s=4.0)
self.client.switch_controller(Controller.servo_control)
self.client.set_arm_speed([self.arm_speed_rad_s] * 6)
self.client.set_eef_speed(self.eef_speed_m_s)
```

**为什么选 servo 不选 direct_control / planning_control**：

| 控制器 | 适用 | 不适用于本场景的原因 |
|---|---|---|
| `direct_control` | VLA 模型直控、低延迟 | 无平滑，相邻 50ms 帧之间易抖动；不接受 `move_end_pose`（PDF p.21 表格） |
| `servo_control` ✅ | 遥操作、20Hz 末端下发 | 默认 250Hz 平滑滤波，相位滞后 ~20–40ms，可接 `move_end_pose` 同时控夹爪 |
| `planning_control` | 上层轨迹规划 | 内置规划器，不接受高频流式输入 |
| `mit_control` | 力位混合 | 慎用（PDF 原话） |

---

## 2. 状态数据结构

```python
class ArmStateSnapshot(NamedTuple):
    angles_rad: np.ndarray       # (6,)  关节角，弧度
    gripper_m: float             # 夹爪开度，米（G2: 0~0.072）
    eef_xyz: np.ndarray          # (3,)  末端笛卡尔位置，米
    eef_rpy: np.ndarray          # (3,)  extrinsic XYZ，已 unwrap + canonicalize
    eef_quat_xyzw: np.ndarray    # (4,)  原始四元数，符号已 canonicalize
    capture_ts_ns: int           # poller 取到该状态的本地时间戳
```

`eef_rpy` 必须在写入 cache 前完成：
1. `unwrap_quat_sign(q_new, q_prev)` —— quat 帧间符号 unwrap（设计文档风险 2）；
2. `as_euler("xyz", degrees=False)` —— scipy extrinsic XYZ；
3. `unwrap_rpy_sequence` —— rpy 域 ±π 兜底 unwrap。

---

## 3. Poller 后台线程

50Hz（`--arm-poll-hz`），串行调三个 RPC（实测每次 ~1ms）：

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
        q = unwrap_quat_sign(q, self._last_quat)                # 风险 2
    self._last_quat = q

    rpy = R.from_quat(q).as_euler("xyz", degrees=False)
    if self._last_rpy is not None:
        rpy = unwrap_rpy_pair(self._last_rpy, rpy)              # ±π 兜底
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

**健康标记**：`_consecutive_fail >= 5` 时 `health()` 转 RED，watchdog 据此决定急停。

**读取接口**：

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

---

## 4. 单点下发：`send_pose`

**这是整个模块的核心**。要点：

```python
def send_pose(
    self,
    target_xyz: np.ndarray,        # (3,) 米
    target_rpy: np.ndarray,        # (3,) extrinsic XYZ，弧度
    gripper_m: float | None,       # 米；None 表示沿用上次值
) -> bool:
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
        options.eef_pos = float(gripper_m)      # 关键：通过 options 顺带下发夹爪

    ok = self.client.move_end_pose(pose, options, timeout_ms=200)
    if not ok:
        self._send_fail += 1
        return False
    self._send_fail = 0
    return True
```

### 4.1 为什么不分两次调 `move_eef`

PDF p.31 明确警告：

> 在 servo_control 控制器下进行笛卡尔空间或关节空间控制时，末端执行器的运动会受到 `options.eef_pos` 参数的影响。当采用 `options.blocking = False` 即非阻塞方式时，**如果同时调用 `move_end_pose`（或 `move_joint`）与 `move_eef` 接口，后一个指令会中断前一个任务的执行**。

⚠️ 设计文档 5.3 节写的 `move_end_pose + move_eef` 两次调用是错误的，实现时合并到一次 `move_end_pose(pose, options)` 里，用 `options.eef_pos` 携带夹爪目标。

### 4.2 量纲与约定核对清单

| 字段 | 单位 | 范围 / 约定 |
|---|---|---|
| `CartesianPose.position` | 米 | 基坐标系（**风险 9 未决**，上线前零位测试核对） |
| `CartesianPose.orientation` | quat [x,y,z,w] | scipy `as_quat()` 默认输出，无需翻转 |
| `options.eef_pos` | 米 | G2: `[0, 0.072]`，G2T: `[0, 0.090]`，G2L: `[0, 0.010]`（PDF p.5） |
| `options.eff` | 安培 | 每关节电流阈值，`elem_max=20.0`；默认 8.0 |
| `options.eef_eff` | 安培 | 夹爪电流阈值，默认 8.0 |
| 关节限位 | 弧度 | `ARM_JOINT_LIMITS` 见 PDF p.5；超限服务端拒发 |

---

## 5. Lease 续期

设计文档 5.3 节预留位置，目前 `arm-sdk 0.1.0.dev51+` 实测 `acquire_control` 接受 `lease_ms` + `renew_period_s` 参数，**后台自动续租**，无需再单独起 timer：

```python
ok = self.client.acquire_control(lease_ms=15000, renew_period_s=4.0)
if not ok:
    raise RuntimeError("acquire_control failed — another client may hold the lease")
```

**未解项**（设计文档 11 节）：若发现自动续租不可靠，回退到手动 timer，调用 `self.client.acquire_control()` 重新续。具体 API 名以 `airbot_example_record_and_replay.py` 实测为准。

---

## 6. 急停

```python
def emergency_stop(self, enable: bool = True) -> None:
    # PDF: set_arm_emergency_stop 阻塞 ~150ms
    self._logger.warning("[arm] emergency_stop(enable=%s)", enable)
    self._auto_dispatch_flag.clear()              # 同步停 dispatcher（风险 10）
    ok = self.client.set_arm_emergency_stop(enable)
    self._logger.warning("[arm] emergency_stop done, ok=%s", ok)
```

**急停恢复**：`set_arm_emergency_stop(False)` 不需要重新 acquire_control，但**必须**先释放再恢复 dispatcher（手动调 `/start` 重置）。

---

## 7. Health 报告

供 `/health` 端点聚合：

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
        "quat_unwrap_count": self._quat_unwrap_count,
        "status": "GREEN" if self._consecutive_fail < 3 else ("YELLOW" if self._consecutive_fail < 5 else "RED"),
    }
```

`quat_unwrap_count` 触发频率 > 1 Hz 视为异常（设计文档风险 2），watchdog 据此决策。

---

## 8. 生命周期与线程模型

```python
class ArmClient:
    def __init__(self, ...): ...

    def start(self) -> None:
        # 主线程：连接 + acquire + switch_controller + set_speed
        # 后台：启动 poller 线程
        ...

    def stop(self) -> None:
        # 顺序：停 poller → release_control → close()
        self._poller_stop.set()
        self._poller_thread.join(timeout=2.0)
        self.client.release_control()
        self.client.close()
```

**线程安全**：
- `state_cache` 用 `threading.Lock` 保护，读写一个不可变 `ArmStateSnapshot`；
- `send_pose` 不加锁（SDK 客户端内部已经线程安全；多线程同时 send 才需要锁，本场景只有 Dispatcher 一个线程发）；
- 急停标志位 `_auto_dispatch_flag` 用 `threading.Event`，保证 watchdog 与 dispatcher 之间同步可见。

---

## 9. 上线前必做：零位测试

设计文档风险 9 提到的坐标系/欧拉约定核对，建议作为 `tests/test_arm_client_zero_drift.py`：

```python
def test_zero_delta_does_not_move(arm: ArmClient):
    snap0 = arm.latest()
    assert snap0 is not None

    for _ in range(32):                                  # 模拟 chunk_len=32 帧
        arm.send_pose(
            target_xyz=snap0.eef_xyz,                    # 完全照搬当前位姿
            target_rpy=snap0.eef_rpy,
            gripper_m=snap0.gripper_m,
        )
        time.sleep(0.05)

    snap1 = arm.latest()
    drift_xyz = np.linalg.norm(snap1.eef_xyz - snap0.eef_xyz)
    drift_rpy = np.linalg.norm(snap1.eef_rpy - snap0.eef_rpy)
    assert drift_xyz < 2e-3, f"xyz drift {drift_xyz*1000:.2f} mm > 2 mm"
    assert drift_rpy < np.deg2rad(0.5), f"rpy drift {np.rad2deg(drift_rpy):.2f}° > 0.5°"
```

**任何一项漂移超阈值都说明坐标系或欧拉约定不对，不要继续后续测试**。

---

## 10. 常见坑速查表

| 现象 | 大概率原因 | 排查 |
|---|---|---|
| `move_end_pose` 返回 False | 关节超限 / lease 失效 / 服务端未启动 | 看 server log；先调 `get_service_state()` |
| 机械臂只动了一下就卡住 | `blocking=True` 漏改 | grep `options.blocking` |
| 夹爪不动 | 漏设 `options.eef_pos` 或单独调了 `move_eef` 被 `move_end_pose` 顶掉 | 看 PDF p.31 警告 |
| rpy 突然跳 2π 后位姿爆飞 | quat 帧间符号翻转，未 unwrap | poller 里加 `unwrap_quat_sign` |
| pitch 接近 ±90° 时姿态乱跳 | gimbal lock | 检查工位 / 切四元数 slerp（future work） |
| 启动报 `controller=False` | `switch_controller` 没调或失败 | 启动 banner 打印 controller_state |
| 远程客户端报无权限 | lease 续期失败 / 被其它客户端抢占 | health 看 `lease_renew_count` 是否在增加 |
| 急停后无法恢复 | 没调 `set_arm_emergency_stop(False)` | 走 `/emergency` 复位流程 |

---

## 11. 相关文件 / 行号锚点

- 设计文档：`docs/fastwam_http_server_self_fetch_design.md` 第 5.3 节
- SDK PDF：`AIRBOT Arm SDK 开发指南` p.5（关节限位）/ p.21（控制器与接口矩阵）/ p.31（同步控夹爪警告）
- 上游欧拉约定：`/data/home/Lyle/Projects/mcap_preprocess_pipeline/scripts/step01_extract_mcap_rgb_and_params.py:1117`
- 旋转工具：`src/fastwam/server/rotation.py`（设计文档 5.2）
- 闭环调度：`src/fastwam/server/closed_loop.py`（设计文档 5.5）

---

## 12. PR 落地顺序建议

参考设计文档第 10 节：

1. **PR3a**：仅 `ArmClient` 骨架 + poller，挂 `/health`，不实现 `send_pose`；
2. **PR3b**：补 `send_pose`，但 server 入口仍 `--auto-dispatch=false`，只打日志不真发；
3. **PR3c**：零位测试通过后，开 `--auto-dispatch=true`，单帧测；
4. **PR3d**：联调 watchdog → 急停链路，完整覆盖第 6 节错误事件。
