"""Unit tests for ``fastwam.server.dispatch.DispatchLoop``.

Everything but the rotation helpers is mocked. The ring buffer is replaced
with a tiny duck-typed stub and the ARM client is a ``MagicMock``.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from fastwam.server.dispatch import DispatchLoop


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@dataclass
class ChunkEntry:
    action_abs: np.ndarray
    base_capture_ts_ns: int
    step_dt_ns: int
    chunk_id: int


class FakeRing:
    """Duck-typed ring buffer with latest()/prev() returning fixed entries."""

    def __init__(
        self,
        latest: Optional[ChunkEntry] = None,
        prev: Optional[ChunkEntry] = None,
    ):
        self._latest = latest
        self._prev = prev

    def set(self, latest=None, prev=None):
        self._latest = latest
        self._prev = prev

    def latest(self):
        return self._latest

    def prev(self):
        return self._prev


def make_chunk(
    n: int = 32,
    base_ns: Optional[int] = None,
    chunk_id: int = 1,
    rpy: tuple = (0.0, 0.0, 0.0),
    pitch_override: Optional[float] = None,
    step_dt_ns: int = 50_000_000,
) -> ChunkEntry:
    """Build a (n,7) ChunkEntry. Pose is constant unless pitch_override given."""
    base = base_ns if base_ns is not None else time.time_ns()
    action = np.zeros((n, 7), dtype=np.float32)
    action[:, 0] = 0.5  # x
    action[:, 1] = 0.0
    action[:, 2] = 0.3  # z
    action[:, 3] = rpy[0]
    action[:, 4] = rpy[1]
    action[:, 5] = rpy[2]
    action[:, 6] = 0.05  # gripper m
    if pitch_override is not None:
        action[:, 4] = pitch_override
    return ChunkEntry(
        action_abs=action,
        base_capture_ts_ns=base,
        step_dt_ns=step_dt_ns,
        chunk_id=chunk_id,
    )


def _silent_logger() -> logging.Logger:
    log = logging.getLogger("test_dispatch")
    log.setLevel(logging.CRITICAL)
    return log


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
class TestDispatchBasic:
    def test_tick_runs_about_four_times_in_200ms(self):
        ring = FakeRing(latest=make_chunk(base_ns=time.time_ns()))
        arm = MagicMock()
        loop = DispatchLoop(
            ringbuffer=ring, arm_client=arm,
            send_period_ms=50, blend_frames=0,
            auto_dispatch=False, logger=_silent_logger(),
        )
        loop.start()
        time.sleep(0.21)
        loop.stop()
        ticks = loop.status()["tick_count"]
        assert 3 <= ticks <= 6, f"expected ~4 ticks, got {ticks}"

    def test_idx_calculation_uses_round_of_dt(self):
        """t_slot ~ base + 5*dt → idx=5 should read action[5]."""
        dt = 50_000_000
        base = time.time_ns() - 5 * dt  # so idx == 5 right now
        action = np.zeros((32, 7), dtype=np.float32)
        for i in range(32):
            action[i, 0] = 1.0 + i * 0.01  # uniquely identify each idx
            action[i, 6] = 0.05
        chunk = ChunkEntry(action_abs=action, base_capture_ts_ns=base,
                           step_dt_ns=dt, chunk_id=42)
        ring = FakeRing(latest=chunk)
        arm = MagicMock()
        loop = DispatchLoop(ring, arm, send_period_ms=50, blend_frames=0,
                            auto_dispatch=False, logger=_silent_logger())
        loop._tick(time.monotonic_ns())
        st = loop.status()
        assert st["last_dispatch_idx"] == 5
        assert st["last_pose_target"]["xyz"][0] == pytest.approx(1.0 + 5 * 0.01, rel=1e-5)

    def test_idx_negative_skips(self):
        # base far in the future → idx negative
        chunk = make_chunk(base_ns=time.time_ns() + 5 * 50_000_000)
        ring = FakeRing(latest=chunk)
        loop = DispatchLoop(ring, MagicMock(), send_period_ms=50, blend_frames=0,
                            auto_dispatch=False, logger=_silent_logger())
        loop._tick(time.monotonic_ns())
        assert loop.status()["last_dispatch_idx"] == -1
        assert loop.status()["skip_count"] == 1

    def test_chunk_exhausted_holds(self):
        dt = 50_000_000
        base = time.time_ns() - 100 * dt
        chunk = ChunkEntry(action_abs=np.zeros((32, 7), dtype=np.float32),
                           base_capture_ts_ns=base, step_dt_ns=dt, chunk_id=1)
        ring = FakeRing(latest=chunk)
        arm = MagicMock()
        loop = DispatchLoop(ring, arm, send_period_ms=50, blend_frames=0,
                            auto_dispatch=True, logger=_silent_logger())
        loop._tick(time.monotonic_ns())
        arm.send_pose.assert_not_called()
        assert loop.status()["last_dispatch_idx"] == 31
        assert loop.status()["last_pose_target"] is None or loop.status()["skip_count"] >= 1


class TestBlend:
    def test_blend_weights_quarter_half_threequarter_one(self):
        """4-frame blend: w should be 0.25, 0.5, 0.75, 1.0 across idx_in_blend=0..3."""
        dt = 50_000_000
        bf = 4

        prev_act = np.zeros((32, 7), dtype=np.float32)
        new_act = np.zeros((32, 7), dtype=np.float32)
        prev_act[:, 0] = 0.0  # x
        new_act[:, 0] = 1.0
        prev_act[:, 3:6] = 0.0
        new_act[:, 3:6] = 0.0
        prev_act[:, 6] = 0.0
        new_act[:, 6] = 1.0

        # set base so that, at "now", idx_in_blend = k
        for k, expected_w in enumerate([0.25, 0.5, 0.75, 1.0]):
            now_ns = time.time_ns()
            new_base = now_ns - k * dt
            prev_base = new_base - 5 * dt  # so idx_old ≈ k+5, well in range
            new = ChunkEntry(new_act, new_base, dt, chunk_id=2)
            prev = ChunkEntry(prev_act, prev_base, dt, chunk_id=1)
            ring = FakeRing(latest=new, prev=prev)
            loop = DispatchLoop(ring, MagicMock(), send_period_ms=50,
                                blend_frames=bf, auto_dispatch=False,
                                logger=_silent_logger())
            loop._tick(time.monotonic_ns())
            st = loop.status()
            # blend_state encodes weight
            assert st["blend_state"].startswith("active w="), st["blend_state"]
            got_w = float(st["blend_state"].split("=")[1])
            assert got_w == pytest.approx(expected_w, abs=0.02)
            # xyz at w=expected_w: x = (1-w)*0 + w*1 = w
            assert st["last_pose_target"]["xyz"][0] == pytest.approx(expected_w, abs=0.02)
            assert st["last_pose_target"]["gripper"] == pytest.approx(expected_w, abs=0.02)

    def test_blend_rpy_unwrap_no_2pi_jump(self):
        """prev.rpy near -pi, new.rpy near +pi → after unwrap, lerp must be continuous."""
        dt = 50_000_000
        n = 4
        prev_act = np.zeros((n, 7), dtype=np.float32)
        new_act = np.zeros((n, 7), dtype=np.float32)
        # roll: prev = -pi + 0.05, new = pi - 0.05  (physically very close)
        prev_act[:, 3] = -math.pi + 0.05
        new_act[:, 3] = math.pi - 0.05

        now_ns = time.time_ns()
        new_base = now_ns  # idx_in_blend=0 → w=0.25
        prev_base = new_base - 2 * dt
        ring = FakeRing(
            latest=ChunkEntry(new_act, new_base, dt, 2),
            prev=ChunkEntry(prev_act, prev_base, dt, 1),
        )
        loop = DispatchLoop(ring, MagicMock(), send_period_ms=50, blend_frames=4,
                            auto_dispatch=False, logger=_silent_logger())
        loop._tick(time.monotonic_ns())
        st = loop.status()
        roll = st["last_pose_target"]["rpy"][0]
        # without unwrap, lerp would give roll ~ 0; with proper unwrap, roll
        # should sit near ±pi (i.e. abs(roll) > 3.0)
        assert abs(roll) > 3.0, f"unwrap blend should stay near pi, got {roll}"

    def test_blend_giveup_when_rpy_diff_huge(self):
        """If unwrapped rpy still differs by > pi/4 → hard switch (no blend)."""
        dt = 50_000_000
        n = 4
        prev_act = np.zeros((n, 7), dtype=np.float32)
        new_act = np.zeros((n, 7), dtype=np.float32)
        prev_act[:, 3] = 0.0  # roll
        new_act[:, 3] = 1.5   # > pi/4

        now_ns = time.time_ns()
        new_base = now_ns
        prev_base = new_base - 2 * dt
        ring = FakeRing(
            latest=ChunkEntry(new_act, new_base, dt, 2),
            prev=ChunkEntry(prev_act, prev_base, dt, 1),
        )
        loop = DispatchLoop(ring, MagicMock(), send_period_ms=50, blend_frames=4,
                            auto_dispatch=False, logger=_silent_logger())
        loop._tick(time.monotonic_ns())
        st = loop.status()
        # hard switch → blend_state stays "off"
        assert st["blend_state"] == "off"
        assert st["last_pose_target"]["rpy"][0] == pytest.approx(1.5, abs=1e-3)


class TestGimbalAndJump:
    def test_gimbal_lock_pitch_90_triggers_hold(self):
        chunk = make_chunk(pitch_override=math.radians(89.5))
        ring = FakeRing(latest=chunk)
        arm = MagicMock()
        loop = DispatchLoop(ring, arm, send_period_ms=50, blend_frames=0,
                            auto_dispatch=True, logger=_silent_logger())
        loop._tick(time.monotonic_ns())
        st = loop.status()
        assert st["gimbal_error_count"] >= 1
        assert st["hold_mode"] is True
        assert "gimbal" in st["hold_reason"]
        arm.send_pose.assert_not_called()

    def test_gimbal_warn_pitch_80(self):
        chunk = make_chunk(pitch_override=math.radians(80.0))
        ring = FakeRing(latest=chunk)
        arm = MagicMock()
        loop = DispatchLoop(ring, arm, send_period_ms=50, blend_frames=0,
                            auto_dispatch=True, logger=_silent_logger())
        loop._tick(time.monotonic_ns())
        st = loop.status()
        assert st["gimbal_warn_count"] == 1
        assert st["gimbal_error_count"] == 0
        assert st["hold_mode"] is False
        arm.send_pose.assert_called_once()

    def test_rpy_jump_three_strikes_emergency(self):
        """Three consecutive ticks each with rpy diff > pi/4 → emergency_stop."""
        arm = MagicMock()
        arm.send_pose.return_value = True
        # We'll swap chunks between ticks; each new chunk's rpy differs by 1.0 rad.
        dt = 50_000_000
        seq = [0.0, 1.0, 2.0, 3.0]  # each step jumps by 1.0 rad > pi/4
        ring = FakeRing(latest=None)
        loop = DispatchLoop(ring, arm, send_period_ms=50, blend_frames=0,
                            auto_dispatch=True, logger=_silent_logger())
        # prime first dispatch (no jump check yet)
        ring.set(latest=make_chunk(base_ns=time.time_ns(), rpy=(seq[0], 0, 0), chunk_id=1))
        loop._tick(time.monotonic_ns())
        # now force three jumps
        for i, roll in enumerate(seq[1:], start=2):
            ring.set(latest=make_chunk(base_ns=time.time_ns(),
                                       rpy=(roll, 0, 0), chunk_id=i))
            loop._tick(time.monotonic_ns())
        arm.emergency_stop.assert_called_with(True)
        assert loop.status()["rpy_jump_count"] >= 3


class TestAutoDispatchAndFailures:
    def test_auto_dispatch_false_records_but_does_not_send(self):
        chunk = make_chunk()
        ring = FakeRing(latest=chunk)
        arm = MagicMock()
        loop = DispatchLoop(ring, arm, send_period_ms=50, blend_frames=0,
                            auto_dispatch=False, logger=_silent_logger())
        loop._tick(time.monotonic_ns())
        arm.send_pose.assert_not_called()
        assert loop.status()["last_pose_target"] is not None
        assert loop.status()["last_pose_target"]["dispatched"] is False

    def test_send_pose_failure_triggers_emergency(self):
        chunk = make_chunk()
        ring = FakeRing(latest=chunk)
        arm = MagicMock()
        arm.send_pose.return_value = False
        loop = DispatchLoop(ring, arm, send_period_ms=50, blend_frames=0,
                            auto_dispatch=True, emergency_on_failure=True,
                            logger=_silent_logger())
        loop._tick(time.monotonic_ns())
        arm.send_pose.assert_called_once()
        arm.emergency_stop.assert_called_with(True)
        assert loop.status()["send_fail_count"] == 1
        assert loop.status()["hold_mode"] is True

    def test_send_pose_failure_without_emergency_just_holds(self):
        chunk = make_chunk()
        ring = FakeRing(latest=chunk)
        arm = MagicMock()
        arm.send_pose.return_value = False
        loop = DispatchLoop(ring, arm, send_period_ms=50, blend_frames=0,
                            auto_dispatch=True, emergency_on_failure=False,
                            logger=_silent_logger())
        loop._tick(time.monotonic_ns())
        arm.emergency_stop.assert_not_called()
        assert loop.status()["hold_mode"] is True
        assert loop.status()["send_fail_count"] == 1

    def test_set_hold_pauses_dispatch(self):
        chunk = make_chunk()
        ring = FakeRing(latest=chunk)
        arm = MagicMock()
        arm.send_pose.return_value = True
        loop = DispatchLoop(ring, arm, send_period_ms=50, blend_frames=0,
                            auto_dispatch=True, logger=_silent_logger())
        loop.set_hold(True, reason="testing")
        loop._tick(time.monotonic_ns())
        arm.send_pose.assert_not_called()
        st = loop.status()
        assert st["hold_mode"] is True
        assert st["blend_state"] == "hold"
        # release
        loop.set_hold(False)
        loop._tick(time.monotonic_ns())
        arm.send_pose.assert_called()

    def test_status_keys_present(self):
        loop = DispatchLoop(FakeRing(), MagicMock(), logger=_silent_logger())
        st = loop.status()
        for k in (
            "last_dispatch_idx", "blend_state", "hold_mode", "last_pose_target",
            "gimbal_warn_count", "rpy_jump_count", "send_fail_count",
            "tick_count", "auto_dispatch", "running",
        ):
            assert k in st
