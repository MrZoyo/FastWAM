"""DispatchLoop — 20 Hz absolute-pose dispatcher (design §3, §7.6.2, §8)."""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .rotation import unwrap_rpy_sequence


_PI_OVER_4 = math.pi / 4.0
_GIMBAL_WARN_RAD = math.radians(75.0)
_GIMBAL_ERROR_RAD = math.radians(85.0)
_NS_PER_MS = 1_000_000


@dataclass
class _DispatchMetrics:
    last_dispatch_idx: int = -1
    last_chunk_id: Optional[int] = None
    blend_state: str = "off"
    hold_mode: bool = False
    hold_reason: str = ""
    last_pose_target: Optional[dict] = None
    gimbal_warn_count: int = 0
    gimbal_error_count: int = 0
    rpy_jump_count: int = 0
    rpy_jump_streak: int = 0
    send_fail_count: int = 0
    tick_count: int = 0
    skip_count: int = 0


class DispatchLoop:
    """20 Hz dispatcher that turns chunked absolute poses into ARM commands.

    Each ``send_period_ms`` tick reads the latest ``ChunkEntry`` from the ring
    buffer, computes ``idx`` from ``t_slot - base_capture_ts_ns``, optionally
    blends with the previous chunk (4-frame lerp; rpy jointly unwrapped),
    runs gimbal / jump guards, and dispatches via ``arm_client.send_pose``.
    """

    def __init__(
        self,
        ringbuffer: Any,
        arm_client: Any,
        send_period_ms: int = 50,
        blend_frames: int = 4,
        chunk_max_stale_ms: int = 2000,
        auto_dispatch: bool = False,
        emergency_on_failure: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if send_period_ms <= 0:
            raise ValueError("send_period_ms must be > 0")
        if blend_frames < 0:
            raise ValueError("blend_frames must be >= 0")

        self._ringbuffer = ringbuffer
        self._arm = arm_client
        self._period_ns = int(send_period_ms) * _NS_PER_MS
        self._blend_frames = int(blend_frames)
        self._chunk_max_stale_ns = int(chunk_max_stale_ms) * _NS_PER_MS
        self._auto_dispatch = bool(auto_dispatch)
        self._emergency_on_failure = bool(emergency_on_failure)
        self._logger = logger or logging.getLogger(__name__)

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._metrics = _DispatchMetrics()
        self._last_target_rpy: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("DispatchLoop.start called twice")
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, name="DispatchLoop", daemon=True
        )
        self._thread.start()
        self._logger.info(
            "[dispatch] start period=%d ms blend=%d auto_dispatch=%s",
            self._period_ns // _NS_PER_MS, self._blend_frames, self._auto_dispatch,
        )

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None

    # ---------------------------------------------------------------- public
    def set_hold(self, hold: bool, reason: str = "") -> None:
        """Toggle hold mode. While held no new poses are sent (SDK coasts)."""
        with self._state_lock:
            prev = self._metrics.hold_mode
            self._metrics.hold_mode = bool(hold)
            self._metrics.hold_reason = reason if hold else ""
        if hold != prev:
            self._logger.warning("[dispatch] hold=%s reason=%s", bool(hold), reason)

    def status(self) -> dict:
        with self._state_lock:
            m = self._metrics
            return {
                "last_dispatch_idx": m.last_dispatch_idx,
                "last_chunk_id": m.last_chunk_id,
                "blend_state": m.blend_state,
                "hold_mode": m.hold_mode,
                "hold_reason": m.hold_reason,
                "last_pose_target": dict(m.last_pose_target) if m.last_pose_target else None,
                "gimbal_warn_count": m.gimbal_warn_count,
                "gimbal_error_count": m.gimbal_error_count,
                "rpy_jump_count": m.rpy_jump_count,
                "send_fail_count": m.send_fail_count,
                "tick_count": m.tick_count,
                "skip_count": m.skip_count,
                "auto_dispatch": self._auto_dispatch,
                "running": self._thread is not None and self._thread.is_alive(),
            }

    # ---------------------------------------------------------------- main loop
    def _run(self) -> None:
        next_t = time.monotonic_ns()
        while not self._stop_evt.is_set():
            self._tick(time.monotonic_ns())
            next_t += self._period_ns
            sleep_ns = next_t - time.monotonic_ns()
            if sleep_ns > 0:
                if self._stop_evt.wait(timeout=sleep_ns / 1e9):
                    return
            else:
                next_t = time.monotonic_ns()  # behind schedule; resync

    def _tick(self, _now_mono_ns: int) -> None:
        with self._state_lock:
            self._metrics.tick_count += 1
            held = self._metrics.hold_mode

        chunk = self._ringbuffer.latest()
        if chunk is None:
            self._mark_skip("no_chunk", blend_state="off")
            return

        t_slot_ns = time.time_ns()
        idx = self._compute_idx(t_slot_ns, chunk.base_capture_ts_ns, chunk.step_dt_ns)
        chunk_len = int(chunk.action_abs.shape[0])

        if idx < 0:
            self._logger.warning("[dispatch] idx<0 (%d), skipping", idx)
            self._mark_skip("idx_negative", blend_state="off")
            return
        if idx >= chunk_len:
            self._mark_skip(
                "chunk_exhausted", blend_state="off",
                last_chunk_id=chunk.chunk_id, last_dispatch_idx=chunk_len - 1,
            )
            return

        xyz_target, rpy_target, gripper_target = self._sample_chunk(chunk, idx)
        blend_state = "off"

        if held:
            with self._state_lock:
                self._metrics.blend_state = "hold"
                self._metrics.last_chunk_id = chunk.chunk_id
            return

        if self._blend_frames > 0:
            prev = self._safe_prev()
            if prev is not None:
                blended = self._maybe_blend(prev, chunk, idx, t_slot_ns)
                if blended is not None:
                    xyz_target, rpy_target, gripper_target, blend_state = blended

        if not self._gimbal_check(rpy_target):
            self._record_target(xyz_target, rpy_target, gripper_target,
                                blend_state, chunk.chunk_id, idx, dispatched=False)
            return

        if not self._rpy_jump_check(rpy_target):
            self._record_target(xyz_target, rpy_target, gripper_target,
                                blend_state, chunk.chunk_id, idx, dispatched=False)
            return

        dispatched = False
        if self._auto_dispatch:
            dispatched = self._dispatch(xyz_target, rpy_target, gripper_target)
        else:
            self._last_target_rpy = rpy_target.astype(np.float32)

        self._record_target(
            xyz_target, rpy_target, gripper_target,
            blend_state, chunk.chunk_id, idx, dispatched=dispatched,
        )

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _compute_idx(t_slot_ns: int, base_ns: int, step_dt_ns: int) -> int:
        if step_dt_ns <= 0:
            return -1
        return int(round((t_slot_ns - base_ns) / step_dt_ns))

    @staticmethod
    def _sample_chunk(chunk: Any, idx: int):
        action = np.asarray(chunk.action_abs[idx], dtype=np.float32).reshape(-1)
        xyz = action[0:3].copy()
        rpy = action[3:6].copy()
        gripper = float(action[6]) if action.shape[0] >= 7 else 0.0
        return xyz, rpy, gripper

    def _safe_prev(self) -> Any:
        try:
            return self._ringbuffer.prev()
        except Exception:  # pragma: no cover
            self._logger.exception("[dispatch] ringbuffer.prev() raised")
            return None

    def _maybe_blend(self, prev: Any, new: Any, idx_new: int, t_slot_ns: int):
        bf = self._blend_frames
        idx_in_blend = (t_slot_ns - new.base_capture_ts_ns) / new.step_dt_ns
        if not (0 <= idx_in_blend < bf):
            return None
        idx_old = self._compute_idx(t_slot_ns, prev.base_capture_ts_ns, prev.step_dt_ns)
        prev_len = int(prev.action_abs.shape[0])
        if not (0 <= idx_old < prev_len):
            self._logger.warning("[dispatch] prev idx oob (%d/%d) -> hard switch",
                                 idx_old, prev_len)
            return None

        w = float(max(0.0, min(1.0, (idx_in_blend + 1) / bf)))
        old_act = np.asarray(prev.action_abs[idx_old], dtype=np.float32).reshape(-1)
        new_act = np.asarray(new.action_abs[idx_new], dtype=np.float32).reshape(-1)

        rpy_stack = np.stack([old_act[3:6], new_act[3:6]], axis=0)
        rpy_unwrapped = unwrap_rpy_sequence(rpy_stack)
        diff = np.abs(rpy_unwrapped[1] - rpy_unwrapped[0])
        if np.any(diff > _PI_OVER_4):
            self._logger.warning("[dispatch] rpy blend diff > pi/4 %s -> hard switch",
                                 diff.tolist())
            return None
        rpy_target = (1.0 - w) * rpy_unwrapped[0] + w * rpy_unwrapped[1]
        xyz_target = (1.0 - w) * old_act[0:3] + w * new_act[0:3]
        gripper_target = float((1.0 - w) * float(old_act[6]) + w * float(new_act[6]))
        return (
            xyz_target.astype(np.float32),
            rpy_target.astype(np.float32),
            gripper_target,
            f"active w={w:.2f}",
        )

    def _gimbal_check(self, rpy_target: np.ndarray) -> bool:
        pitch = float(rpy_target[1])
        if abs(pitch) > _GIMBAL_ERROR_RAD:
            with self._state_lock:
                self._metrics.gimbal_error_count += 1
            self._logger.error("[dispatch] gimbal-lock ERROR pitch=%.3f rad, holding",
                               pitch)
            self.set_hold(True, reason="gimbal_lock")
            return False
        if abs(pitch) > _GIMBAL_WARN_RAD:
            with self._state_lock:
                self._metrics.gimbal_warn_count += 1
            self._logger.warning("[dispatch] gimbal-lock WARN pitch=%.3f rad", pitch)
        return True

    def _rpy_jump_check(self, rpy_target: np.ndarray) -> bool:
        if self._last_target_rpy is None:
            with self._state_lock:
                self._metrics.rpy_jump_streak = 0
            return True
        diff = np.abs(rpy_target - self._last_target_rpy)
        if np.any(diff > _PI_OVER_4):
            with self._state_lock:
                self._metrics.rpy_jump_count += 1
                self._metrics.rpy_jump_streak += 1
                streak = self._metrics.rpy_jump_streak
            self._logger.warning("[dispatch] rpy jump > pi/4 diff=%s streak=%d",
                                 diff.tolist(), streak)
            if streak >= 3:
                self._logger.error("[dispatch] rpy streak %d -> emergency_stop", streak)
                self._trigger_emergency("rpy_jump_streak")
            return False
        with self._state_lock:
            self._metrics.rpy_jump_streak = 0
        return True

    def _dispatch(self, xyz: np.ndarray, rpy: np.ndarray, gripper: float) -> bool:
        try:
            ok = bool(self._arm.send_pose(xyz, rpy, float(gripper)))
        except Exception as exc:
            ok = False
            self._logger.exception("[dispatch] send_pose raised: %s", exc)
        if not ok:
            with self._state_lock:
                self._metrics.send_fail_count += 1
            self._logger.error("[dispatch] send_pose returned False")
            if self._emergency_on_failure:
                self._trigger_emergency("send_pose_failed")
            else:
                self.set_hold(True, reason="send_pose_failed")
            return False
        self._last_target_rpy = rpy.astype(np.float32)
        return True

    def _record_target(self, xyz, rpy, gripper, blend_state, chunk_id, idx,
                       *, dispatched: bool) -> None:
        target = {
            "xyz": [float(v) for v in xyz],
            "rpy": [float(v) for v in rpy],
            "gripper": float(gripper),
            "dispatched": bool(dispatched),
        }
        with self._state_lock:
            self._metrics.last_pose_target = target
            self._metrics.blend_state = blend_state
            self._metrics.last_chunk_id = chunk_id
            self._metrics.last_dispatch_idx = idx

    def _mark_skip(self, reason: str, *, blend_state: str = "skipped",
                   last_chunk_id: Optional[int] = None,
                   last_dispatch_idx: Optional[int] = None) -> None:
        with self._state_lock:
            self._metrics.skip_count += 1
            self._metrics.blend_state = blend_state
            if last_chunk_id is not None:
                self._metrics.last_chunk_id = last_chunk_id
            if last_dispatch_idx is not None:
                self._metrics.last_dispatch_idx = last_dispatch_idx
        self._logger.debug("[dispatch] skip reason=%s", reason)

    def _trigger_emergency(self, reason: str) -> None:
        self.set_hold(True, reason=f"emergency:{reason}")
        try:
            self._arm.emergency_stop(True)
        except Exception:  # pragma: no cover
            self._logger.exception("[dispatch] emergency_stop raised")
        self._stop_evt.set()
