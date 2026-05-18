"""Unit tests for ``fastwam.server.watchdog.Watchdog``.

All collaborators are mocked: ring buffer is a dataclass-based stub, arm /
ws / dispatcher are ``MagicMock`` instances.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from fastwam.server.watchdog import Watchdog


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
    def __init__(self, latest=None):
        self._latest = latest

    def set(self, latest):
        self._latest = latest

    def latest(self):
        return self._latest

    def prev(self):
        return None


def make_chunk(n=32, base_ns=None, chunk_id=1, step_dt_ns=50_000_000):
    base = base_ns if base_ns is not None else time.time_ns()
    return ChunkEntry(
        action_abs=np.zeros((n, 7), dtype=np.float32),
        base_capture_ts_ns=base, step_dt_ns=step_dt_ns, chunk_id=chunk_id,
    )


def _silent_logger() -> logging.Logger:
    log = logging.getLogger("test_watchdog")
    log.setLevel(logging.CRITICAL)
    return log


def make_arm(age_ms=10.0, lease_alive=True, consecutive_fail=0, running=True):
    arm = MagicMock()
    arm.health.return_value = {
        "last_poll_age_ms": age_ms,
        "consecutive_fail": consecutive_fail,
        "lease_alive": lease_alive,
        "running": running,
    }
    return arm


def make_ws(age_ms=20.0):
    ws = MagicMock()
    ws.health.return_value = {
        "last_frame_age_per_channel_ms": {"head_left": age_ms, "right_wrist_left": age_ms},
        "fps_5s": 20.0,
        "decode_fail_count": 0,
        "reconnect_count": 0,
    }
    return ws


def make_wd(ring, arm, ws, dispatcher, **kwargs):
    defaults = dict(
        watchdog_period_ms=10, infer_period_ms=400, arm_red_grace_ms=200,
        arm_health_max_age_ms=200,
    )
    defaults.update(kwargs)
    return Watchdog(
        ringbuffer=ring, arm_client=arm, ws_ingester=ws, dispatcher=dispatcher,
        logger=_silent_logger(), **defaults,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
class TestChunkStale:
    def test_chunk_stale_holds_dispatcher(self):
        # chunk used up long ago
        dt = 50_000_000
        old_base = time.time_ns() - 5 * 32 * dt  # 5 chunks' worth in past
        ring = FakeRing(latest=make_chunk(base_ns=old_base))
        arm, ws, disp = make_arm(), make_ws(), MagicMock()
        wd = make_wd(ring, arm, ws, disp, chunk_max_stale_ms=2000)
        # simulate watching the same chunk for > chunk_max_stale_ms
        wd._tick()
        # fake "we saw this 3 s ago"
        wd._last_chunk_seen_mono_ns = time.monotonic_ns() - 3_000_000_000
        wd._tick()
        # dispatcher.set_hold should have been called with chunk_stale
        calls = [c for c in disp.set_hold.call_args_list
                 if c.kwargs.get("reason", "").startswith("chunk_stale")]
        assert calls, f"expected chunk_stale hold; got {disp.set_hold.call_args_list}"
        assert wd.status()["chunk_stale_count"] >= 1

    def test_chunk_missing_after_seen(self):
        ring = FakeRing(latest=None)
        arm, ws, disp = make_arm(), make_ws(), MagicMock()
        wd = make_wd(ring, arm, ws, disp, chunk_max_stale_ms=100)
        # simulate that we previously saw a chunk 1 s ago
        wd._last_chunk_seen_mono_ns = time.monotonic_ns() - 1_000_000_000
        wd._tick()
        disp.set_hold.assert_called()
        args, kwargs = disp.set_hold.call_args
        assert kwargs.get("reason") == "chunk_missing"


class TestWsStaleThreshold:
    def test_ws_warn_only(self):
        ring = FakeRing(latest=make_chunk())
        arm, disp = make_arm(), MagicMock()
        ws = make_ws(age_ms=300.0)  # > 200 warn, < 500 hold
        wd = make_wd(ring, arm, ws, disp,
                     ws_warn_stale_ms=200, ws_hold_stale_ms=500, ws_estop_stale_ms=1500)
        wd._tick()
        assert wd.status()["ws_warn_count"] >= 1
        assert wd.status()["ws_hold_count"] == 0
        assert wd.status()["ws_estop_count"] == 0
        # warn alone should NOT call set_hold for ws
        ws_hold_calls = [c for c in disp.set_hold.call_args_list
                         if c.kwargs.get("reason", "").startswith("ws_stale")]
        assert not ws_hold_calls

    def test_ws_hold_threshold(self):
        ring = FakeRing(latest=make_chunk())
        arm, disp = make_arm(), MagicMock()
        ws = make_ws(age_ms=700.0)  # > 500 hold, < 1500 estop
        wd = make_wd(ring, arm, ws, disp,
                     ws_warn_stale_ms=200, ws_hold_stale_ms=500, ws_estop_stale_ms=1500)
        wd._tick()
        assert wd.status()["ws_hold_count"] >= 1
        disp.set_hold.assert_called()
        args, kwargs = disp.set_hold.call_args
        assert kwargs.get("reason", "").startswith("ws_stale")

    def test_ws_estop_threshold(self):
        ring = FakeRing(latest=make_chunk())
        arm, disp = make_arm(), MagicMock()
        ws = make_ws(age_ms=2000.0)  # > 1500 estop
        wd = make_wd(ring, arm, ws, disp,
                     ws_warn_stale_ms=200, ws_hold_stale_ms=500, ws_estop_stale_ms=1500)
        wd._tick()
        assert wd.status()["ws_estop_count"] >= 1
        arm.emergency_stop.assert_called_with(True)
        disp.stop.assert_called()


class TestArmHealth:
    def test_arm_red_streak_triggers_emergency(self):
        ring = FakeRing(latest=make_chunk())
        ws, disp = make_ws(), MagicMock()
        # ARM age way above max-age threshold → RED
        arm = make_arm(age_ms=500.0)
        wd = make_wd(ring, arm, ws, disp,
                     arm_health_max_age_ms=200, arm_red_grace_ms=200)
        # First tick: enter RED, start streak timer
        wd._tick()
        assert wd.status()["arm_red_streak_ms"] >= 0.0
        # Simulate streak grew past grace by rewinding the streak start
        wd._arm_red_start_mono_ns = time.monotonic_ns() - 300_000_000  # 300 ms ago
        wd._tick()
        arm.emergency_stop.assert_called_with(True)

    def test_arm_healthy_clears_streak(self):
        ring = FakeRing(latest=make_chunk())
        arm = make_arm(age_ms=10.0)
        ws, disp = make_ws(), MagicMock()
        wd = make_wd(ring, arm, ws, disp,
                     arm_health_max_age_ms=200, arm_red_grace_ms=200)
        # pretend we had a streak in flight
        wd._arm_red_start_mono_ns = time.monotonic_ns() - 100_000_000
        wd._tick()
        assert wd._arm_red_start_mono_ns is None
        assert wd.status()["arm_red_streak_ms"] == 0.0


class TestInferHeartbeat:
    def test_no_heartbeat_yet_no_action(self):
        ring = FakeRing(latest=make_chunk())
        arm, ws, disp = make_arm(), make_ws(), MagicMock()
        wd = make_wd(ring, arm, ws, disp)
        wd._tick()
        # nothing should fire because no heartbeat has ever arrived
        assert wd.status()["infer_heartbeat_age_ms"] is None
        # set_hold may have been called for other reasons; verify not for infer
        for c in disp.set_hold.call_args_list:
            assert "infer_heartbeat" not in c.kwargs.get("reason", "")

    def test_infer_warn(self):
        ring = FakeRing(latest=make_chunk())
        arm, ws, disp = make_ws(), MagicMock(), MagicMock()
        # actually we need real make_arm + make_ws
        arm = make_arm()
        ws = make_ws()
        wd = make_wd(ring, arm, ws, disp,
                     infer_period_ms=400, infer_warn_factor=2.0, infer_hold_factor=5.0,
                     infer_timeout_factor=5.0)
        # 900 ms ago heartbeat → warn (> 800 ms), not hold (< 2000 ms)
        wd._infer_heartbeat_ns = time.monotonic_ns() - 900_000_000
        wd._tick()
        assert wd.status()["infer_warn_count"] >= 1
        # ensure no infer hold
        infer_holds = [c for c in disp.set_hold.call_args_list
                       if "infer_heartbeat" in c.kwargs.get("reason", "")]
        assert not infer_holds

    def test_infer_hold(self):
        ring = FakeRing(latest=make_chunk())
        arm, ws, disp = make_arm(), make_ws(), MagicMock()
        wd = make_wd(ring, arm, ws, disp,
                     infer_period_ms=400, infer_warn_factor=2.0,
                     infer_hold_factor=5.0, infer_timeout_factor=5.0)
        # 2.5 s ago → > 5× = 2 s → hold
        wd._infer_heartbeat_ns = time.monotonic_ns() - 2_500_000_000
        wd._tick()
        infer_holds = [c for c in disp.set_hold.call_args_list
                       if "infer_heartbeat" in c.kwargs.get("reason", "")]
        assert infer_holds


class TestStatus:
    def test_status_shape(self):
        ring = FakeRing(latest=make_chunk())
        arm, ws, disp = make_arm(), make_ws(), MagicMock()
        wd = make_wd(ring, arm, ws, disp)
        wd._tick()
        st = wd.status()
        for k in (
            "hold_reason", "last_emergency_reason", "last_ws_age_ms",
            "last_arm_age_ms", "infer_heartbeat_age_ms",
            "tick_count", "running",
        ):
            assert k in st

    def test_infer_heartbeat_updates_age(self):
        ring = FakeRing(latest=make_chunk())
        arm, ws, disp = make_arm(), make_ws(), MagicMock()
        wd = make_wd(ring, arm, ws, disp)
        wd.infer_heartbeat()
        wd._tick()
        assert wd.status()["infer_heartbeat_age_ms"] is not None
        assert wd.status()["infer_heartbeat_age_ms"] < 100.0


# ---------------------------------------------------------------------------
# PR12: start() must reset state; hold must auto-clear on fresh chunk
# ---------------------------------------------------------------------------
class TestStartResetAndHoldRelease:
    def test_watchdog_start_resets_chunk_tracking_state(self):
        ring = FakeRing(latest=None)
        arm, ws, disp = make_arm(), make_ws(), MagicMock()
        wd = make_wd(ring, arm, ws, disp)
        # Pre-populate with ancient state from a previous /start cycle
        ancient = time.monotonic_ns() - 9 * 60 * 1_000_000_000  # 9 minutes ago
        wd._last_seen_chunk_id = 42
        wd._last_seen_chunk_base_ns = 1
        wd._last_chunk_seen_mono_ns = ancient
        wd._arm_red_start_mono_ns = ancient
        wd._infer_heartbeat_ns = ancient
        with wd._state_lock:
            wd._metrics.hold_reason = "chunk_missing"
            wd._metrics.chunk_stale_count = 5
            wd._metrics.tick_count = 1234

        wd.start()
        try:
            assert wd._last_seen_chunk_id is None
            assert wd._last_seen_chunk_base_ns is None
            assert wd._last_chunk_seen_mono_ns is None
            assert wd._arm_red_start_mono_ns is None
            assert wd._infer_heartbeat_ns is None
            st = wd.status()
            assert st["hold_reason"] == ""
            assert st["chunk_stale_count"] == 0
        finally:
            wd.stop()

    def test_watchdog_no_chunk_missing_on_fresh_start(self):
        """Reproduces the bug: 2nd /start with stale internal state + empty ring."""
        ring = FakeRing(latest=None)
        arm, ws, disp = make_arm(), make_ws(), MagicMock()
        wd = make_wd(ring, arm, ws, disp, chunk_max_stale_ms=2000)
        # Simulate prior cycle's lingering state (would normally trigger chunk_missing)
        wd._last_chunk_seen_mono_ns = time.monotonic_ns() - 9 * 60 * 1_000_000_000

        wd.start()
        try:
            wd._tick()  # exercise the same code path the loop runs
            chunk_missing_calls = [
                c for c in disp.set_hold.call_args_list
                if c.args and c.args[0] is True
                and "chunk_missing" in (c.kwargs.get("reason", "") or (c.args[1] if len(c.args) > 1 else ""))
            ]
            assert not chunk_missing_calls, (
                f"unexpected chunk_missing hold on fresh /start: {disp.set_hold.call_args_list}"
            )
        finally:
            wd.stop()

    def test_watchdog_clears_hold_when_new_chunk_arrives(self):
        ring = FakeRing(latest=None)
        arm, ws, disp = make_arm(), make_ws(), MagicMock()
        wd = make_wd(ring, arm, ws, disp, chunk_max_stale_ms=100)
        # Pretend we are already held due to chunk_missing
        with wd._state_lock:
            wd._metrics.hold_reason = "chunk_missing"
        # Now a brand-new chunk arrives
        ring.set(make_chunk(chunk_id=7, base_ns=time.time_ns()))
        wd._tick()
        release_calls = [
            c for c in disp.set_hold.call_args_list
            if c.args and c.args[0] is False
        ]
        assert release_calls, (
            f"expected dispatcher.set_hold(False) call; got {disp.set_hold.call_args_list}"
        )
        assert wd.status()["hold_reason"] == ""

    def test_watchdog_keeps_hold_on_unrelated_reason(self):
        ring = FakeRing(latest=None)
        arm, ws, disp = make_arm(), make_ws(), MagicMock()
        wd = make_wd(ring, arm, ws, disp, chunk_max_stale_ms=100)
        with wd._state_lock:
            wd._metrics.hold_reason = "ws_stale:600ms"
        ring.set(make_chunk(chunk_id=9, base_ns=time.time_ns()))
        wd._tick()
        # chunk-related auto-clear must NOT touch ws_stale hold
        release_calls = [
            c for c in disp.set_hold.call_args_list
            if c.args and c.args[0] is False
        ]
        assert not release_calls, (
            f"set_hold(False) must not fire for non-chunk reasons; got {disp.set_hold.call_args_list}"
        )
        assert wd.status()["hold_reason"] == "ws_stale:600ms"
