"""Unit tests for fastwam.server.arm_client.

The arm_sdk client is fully mocked: tests verify the rpy->quat->CartesianPose
conversion chain, the background poller lifecycle, and quaternion sign
unwrapping. No real gRPC connection is needed.
"""
from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from fastwam.server.arm_client import ArmClient, ArmSdkError


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _CartesianPoseStub:
    """Mirror of arm_sdk.client.CartesianPose (position, orientation tuples)."""

    def __init__(self, position, orientation):
        self.position = tuple(position)
        self.orientation = tuple(orientation)

    def __eq__(self, other):  # pragma: no cover - convenience
        return (
            isinstance(other, _CartesianPoseStub)
            and self.position == other.position
            and self.orientation == other.orientation
        )


class _OptionsStub:
    """Mirror of ArmControlOptions: ``blocking`` + ``eef_pos`` are asserted."""

    def __init__(self):
        self.blocking = False
        self.eef_pos = 0.0


def _make_mock_sdk_client(
    *,
    angles=(0.0, 0.0, 0.0, 1.57, 0.0, -1.57),
    gripper=0.0887,
    pose_xyz=(0.2867, 0.0004, 0.2150),
    pose_quat=(0.0, 0.0, 0.0, 1.0),
):
    c = MagicMock()
    c.get_arm_joint_state.return_value = SimpleNamespace(angles=tuple(angles))
    c.get_eef_joint_state.return_value = SimpleNamespace(eef_pos=gripper)
    c.get_end_pose.return_value = SimpleNamespace(
        position=tuple(pose_xyz), orientation=tuple(pose_quat)
    )
    c.move_end_pose.return_value = True
    c.move_eef.return_value = True
    c.set_arm_emergency_stop.return_value = True
    c.acquire_control.return_value = True
    c.switch_controller.return_value = True
    c.set_arm_speed.return_value = None
    c.release_control.return_value = None
    c.close.return_value = None
    c._lease_id = None
    return c


def _make_client(mock_sdk, **overrides):
    kwargs = dict(
        host="127.0.0.1",
        port=50051,
        poll_hz=200.0,
        state_max_age_ms=500.0,
        lease_ms=15000,
        logger=logging.getLogger("test_arm_client"),
        client_factory=lambda: mock_sdk,
    )
    kwargs.update(overrides)
    arm = ArmClient(**kwargs)
    # Inject the stub CartesianPose / ArmControlOptions classes so that
    # start() does not need the real arm_sdk import.
    mock_sdk._cart_pose_cls = _CartesianPoseStub
    mock_sdk._opts_cls = _OptionsStub

    def _start_no_real_import():
        # Manually replay ArmClient.start without importing arm_sdk.
        if arm._thread is not None:
            raise RuntimeError("already started")
        arm._client = mock_sdk
        arm._cart_pose_cls = _CartesianPoseStub
        arm._opts_cls = _OptionsStub
        arm._stop_evt.clear()
        arm._thread = threading.Thread(
            target=arm._poll_loop, name="ArmClientPoller", daemon=True
        )
        arm._thread.start()

    arm._test_start = _start_no_real_import  # type: ignore[attr-defined]
    return arm


# ---------------------------------------------------------------------------
# send_pose conversion chain
# ---------------------------------------------------------------------------
def test_send_pose_converts_rpy_to_quat_and_packs_gripper_in_opts():
    mock_sdk = _make_mock_sdk_client()
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        rpy = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        xyz = np.array([0.3, 0.0, 0.25], dtype=np.float64)
        ok = arm.send_pose(xyz, rpy, gripper_m=0.05)
        assert ok is True

        # PR8: a single move_end_pose RPC carries both pose and gripper.
        assert mock_sdk.move_end_pose.call_count == 1
        assert mock_sdk.move_eef.call_count == 0

        call = mock_sdk.move_end_pose.call_args
        pose_arg = call.args[0]
        opts_arg = call.args[1]
        assert isinstance(pose_arg, _CartesianPoseStub)
        assert isinstance(opts_arg, _OptionsStub)
        assert opts_arg.blocking is False
        assert opts_arg.eef_pos == pytest.approx(0.05)

        # position passed through verbatim
        np.testing.assert_allclose(pose_arg.position, (0.3, 0.0, 0.25), atol=1e-7)

        # quaternion matches scipy ground truth
        expected_quat = R.from_euler("xyz", rpy, degrees=False).as_quat()
        np.testing.assert_allclose(pose_arg.orientation, expected_quat, atol=1e-6)
    finally:
        arm.stop()


def test_send_pose_without_gripper_leaves_opts_eef_pos_default():
    mock_sdk = _make_mock_sdk_client()
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        arm.send_pose(np.zeros(3), np.zeros(3), gripper_m=None)
        assert mock_sdk.move_end_pose.call_count == 1
        assert mock_sdk.move_eef.call_count == 0
        opts_arg = mock_sdk.move_end_pose.call_args.args[1]
        # opts.eef_pos must stay at the stub's default (0.0) when no gripper.
        assert opts_arg.eef_pos == pytest.approx(0.0)
    finally:
        arm.stop()


def test_send_pose_raises_on_move_end_pose_false():
    mock_sdk = _make_mock_sdk_client()
    mock_sdk.move_end_pose.return_value = False
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        with pytest.raises(ArmSdkError):
            arm.send_pose(np.zeros(3), np.zeros(3), gripper_m=0.05)
        # PR8: move_eef is never called in the new single-RPC path.
        assert mock_sdk.move_eef.call_count == 0
    finally:
        arm.stop()


# ---------------------------------------------------------------------------
# poller lifecycle / latest() / age tracking
# ---------------------------------------------------------------------------
def _wait_until(predicate, timeout_s=2.0, interval_s=0.01):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def test_poller_writes_state_cache_and_latest_returns_fresh():
    mock_sdk = _make_mock_sdk_client(
        angles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        gripper=0.07,
        pose_xyz=(0.31, 0.02, 0.22),
        pose_quat=(0.0, 0.0, 0.0, 1.0),
    )
    arm = _make_client(mock_sdk, poll_hz=200.0, state_max_age_ms=500.0)
    arm._test_start()
    try:
        assert _wait_until(lambda: arm.latest() is not None, timeout_s=1.0), \
            "poller did not populate state_cache"

        snap = arm.latest()
        assert snap is not None
        np.testing.assert_allclose(snap.angles_rad, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        assert snap.gripper_m == pytest.approx(0.07)
        np.testing.assert_allclose(snap.eef_xyz, [0.31, 0.02, 0.22])
        np.testing.assert_allclose(snap.eef_quat_xyzw, [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(snap.eef_rpy, [0.0, 0.0, 0.0], atol=1e-6)
        assert snap.capture_ts_ns > 0

        h = arm.health()
        assert h["running"] is True
        assert h["consecutive_fail"] == 0
        assert h["last_poll_age_ms"] is not None
        assert h["last_poll_age_ms"] < 200.0  # @200 Hz the poll is fresh
    finally:
        arm.stop()
    assert arm.health()["running"] is False


def test_latest_returns_none_when_state_is_stale():
    mock_sdk = _make_mock_sdk_client()
    arm = _make_client(mock_sdk, poll_hz=100.0, state_max_age_ms=50.0)
    arm._test_start()
    try:
        assert _wait_until(lambda: arm.latest() is not None, timeout_s=1.0)
        # Stop polling — state will become stale after state_max_age_ms.
        arm._stop_evt.set()
        time.sleep(0.2)  # > state_max_age_ms (50 ms)
        assert arm.latest() is None, "latest() must drop stale snapshots"
    finally:
        arm.stop()


# ---------------------------------------------------------------------------
# quaternion sign unwrap
# ---------------------------------------------------------------------------
class _QuatFlipSdk:
    """Mock SDK that returns a quaternion sequence with sign flips."""

    def __init__(self, quat_sequence: List[tuple]):
        self._seq = list(quat_sequence)
        self._idx = 0
        self._lock = threading.Lock()
        self._cart_pose_cls = _CartesianPoseStub
        self._opts_cls = _OptionsStub
        self._lease_id = None

    def _advance(self):
        with self._lock:
            idx = min(self._idx, len(self._seq) - 1)
            self._idx += 1
        return self._seq[idx]

    def get_arm_joint_state(self):
        return SimpleNamespace(angles=(0.0,) * 6)

    def get_eef_joint_state(self):
        return SimpleNamespace(eef_pos=0.0887)

    def get_end_pose(self):
        q = self._advance()
        return SimpleNamespace(position=(0.28, 0.0, 0.21), orientation=q)

    def acquire_control(self, lease_ms=15000, renew_period_s=5.0):
        return True

    def release_control(self):
        return None

    def close(self):
        return None


def test_unwrap_quat_sign_keeps_cache_quat_consistent():
    # A non-trivial quaternion (rpy [0.1, 0.2, 0.3])
    q = R.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat()
    q_neg = (-q).tolist()
    q_pos = q.tolist()
    sequence = [tuple(q_pos), tuple(q_neg), tuple(q_pos)]
    # Repeat last value so the poller keeps producing valid samples after the
    # 3-step sequence is exhausted.
    sequence_padded = sequence + [tuple(q_pos)] * 50

    mock_sdk = _QuatFlipSdk(sequence_padded)
    arm = _make_client(mock_sdk, poll_hz=200.0)
    arm._test_start()
    try:
        # wait for all three flips to be consumed
        assert _wait_until(
            lambda: mock_sdk._idx >= 3, timeout_s=2.0
        ), f"poller only consumed {mock_sdk._idx} samples"
        # Give the poller a couple more ticks so unwrap counter settles.
        time.sleep(0.05)

        h = arm.health()
        # The middle frame (-q) must have triggered unwrap; the third (q
        # after sign-corrected -q) may or may not trigger again depending on
        # cache state. We just assert at least one unwrap happened.
        assert h["last_quat_unwrap_count"] >= 1, h

        snap = arm.latest()
        assert snap is not None
        # The cached quat must be on the same hemisphere as the original q
        # (since the sequence ends with q_pos and any earlier flips were
        # rewritten in-place by unwrap_quat_sign).
        assert float(np.dot(snap.eef_quat_xyzw, q)) > 0.0, snap.eef_quat_xyzw
    finally:
        arm.stop()


# ---------------------------------------------------------------------------
# health when RPC fails
# ---------------------------------------------------------------------------
def test_poller_counts_consecutive_failures():
    mock_sdk = _make_mock_sdk_client()
    mock_sdk.get_arm_joint_state.return_value = None  # treated as failure
    arm = _make_client(mock_sdk, poll_hz=500.0)
    arm._test_start()
    try:
        assert _wait_until(
            lambda: arm.health()["consecutive_fail"] >= 5, timeout_s=2.0
        ), arm.health()
        # latest() must be None since no successful sample was ever cached.
        assert arm.latest() is None
    finally:
        arm.stop()


# ---------------------------------------------------------------------------
# emergency stop and acquire_control plumbing
# ---------------------------------------------------------------------------
def test_acquire_release_and_emergency_forward_to_sdk():
    mock_sdk = _make_mock_sdk_client()
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        assert arm.acquire_control() is True
        mock_sdk.acquire_control.assert_called_once_with(
            lease_ms=15000, renew_period_s=5.0
        )
        # PR8: switch_controller(servo_control) + set_arm_speed must follow.
        from arm_sdk.client import Controller as _Controller
        mock_sdk.switch_controller.assert_called_once_with(
            _Controller.servo_control
        )
        mock_sdk.set_arm_speed.assert_called_once_with([0.5] * 6)

        arm.emergency_stop(True)
        mock_sdk.set_arm_emergency_stop.assert_called_with(True)

        arm.release_control()
        mock_sdk.release_control.assert_called()
    finally:
        arm.stop()


def test_acquire_control_switches_to_servo_and_sets_speed_custom():
    """Custom arm_speed_rad_s flows through to set_arm_speed unchanged."""
    mock_sdk = _make_mock_sdk_client()
    arm = _make_client(mock_sdk, arm_speed_rad_s=0.8)
    arm._test_start()
    try:
        from arm_sdk.client import Controller as _Controller
        assert arm.acquire_control() is True

        # Verify call order: acquire_control -> switch_controller -> set_arm_speed.
        names = [c[0] for c in mock_sdk.method_calls if c[0] in (
            "acquire_control", "switch_controller", "set_arm_speed",
        )]
        assert names == ["acquire_control", "switch_controller", "set_arm_speed"], names
        mock_sdk.switch_controller.assert_called_once_with(
            _Controller.servo_control
        )
        mock_sdk.set_arm_speed.assert_called_once_with([0.8] * 6)
    finally:
        arm.stop()


def test_acquire_control_switch_failure_releases_lease():
    mock_sdk = _make_mock_sdk_client()
    mock_sdk.switch_controller.return_value = False
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        assert arm.acquire_control() is False
        mock_sdk.acquire_control.assert_called_once()
        mock_sdk.switch_controller.assert_called_once()
        # On switch failure we must release the lease and not call set_arm_speed.
        mock_sdk.release_control.assert_called_once()
        mock_sdk.set_arm_speed.assert_not_called()
        assert arm._acquired is False
    finally:
        arm.stop()


def test_acquire_control_acquire_returns_false_skips_switch():
    mock_sdk = _make_mock_sdk_client()
    mock_sdk.acquire_control.return_value = False
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        assert arm.acquire_control() is False
        # When lease is not granted, no controller switch / speed set should happen.
        mock_sdk.switch_controller.assert_not_called()
        mock_sdk.set_arm_speed.assert_not_called()
        mock_sdk.release_control.assert_not_called()
        assert arm._acquired is False
    finally:
        arm.stop()


# ---------------------------------------------------------------------------
# PR11 R6: health().status tri-level + consecutive_send_fail
# ---------------------------------------------------------------------------
def test_health_status_green_baseline():
    mock_sdk = _make_mock_sdk_client()
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        # wait for poller to populate one good sample
        assert _wait_until(lambda: arm.latest() is not None, timeout_s=1.0)
        h = arm.health()
        assert h["status"] == "GREEN", h
        assert h["consecutive_send_fail"] == 0
        assert h["lease_renew_count"] == 0
        assert h["consecutive_fail"] == 0
    finally:
        arm.stop()


def test_health_status_yellow_at_3_send_fails():
    mock_sdk = _make_mock_sdk_client()
    mock_sdk.move_end_pose.return_value = False
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        for _ in range(3):
            with pytest.raises(ArmSdkError):
                arm.send_pose(np.zeros(3), np.zeros(3), gripper_m=0.05)
        h = arm.health()
        assert h["consecutive_send_fail"] == 3, h
        assert h["status"] == "YELLOW", h
    finally:
        arm.stop()


def test_health_status_red_at_5_send_fails():
    mock_sdk = _make_mock_sdk_client()
    mock_sdk.move_end_pose.return_value = False
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        for _ in range(5):
            with pytest.raises(ArmSdkError):
                arm.send_pose(np.zeros(3), np.zeros(3), gripper_m=0.05)
        h = arm.health()
        assert h["consecutive_send_fail"] == 5, h
        assert h["status"] == "RED", h
    finally:
        arm.stop()


def test_health_status_recovers_to_green_after_success():
    # First 4 calls fail, 5th succeeds -> counter resets, status back to GREEN.
    mock_sdk = _make_mock_sdk_client()
    fail_then_pass = [False, False, False, False, True]
    mock_sdk.move_end_pose.side_effect = lambda *a, **kw: fail_then_pass.pop(0)
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        for _ in range(4):
            with pytest.raises(ArmSdkError):
                arm.send_pose(np.zeros(3), np.zeros(3), gripper_m=0.05)
        # success on 5th
        assert arm.send_pose(np.zeros(3), np.zeros(3), gripper_m=0.05) is True
        # wait for poller to also be in good state
        assert _wait_until(lambda: arm.latest() is not None, timeout_s=1.0)
        h = arm.health()
        assert h["consecutive_send_fail"] == 0, h
        assert h["status"] == "GREEN", h
    finally:
        arm.stop()


def test_health_status_red_on_lease_lost():
    mock_sdk = _make_mock_sdk_client()
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        # Force "we believe we own a lease but SDK has cleared it".
        arm._acquired = True
        mock_sdk._lease_id = None
        h = arm.health()
        assert h["lease_alive"] is False, h
        assert h["status"] == "RED", h
    finally:
        arm.stop()


# ---------------------------------------------------------------------------
# PR11 R7: poller unwraps rpy across ±π
# ---------------------------------------------------------------------------
class _RpyEdgeSdk:
    """Mock SDK that emits a fixed quaternion sequence (xyzw)."""

    def __init__(self, quat_sequence):
        self._seq = list(quat_sequence)
        self._idx = 0
        self._lock = threading.Lock()
        self._cart_pose_cls = _CartesianPoseStub
        self._opts_cls = _OptionsStub
        self._lease_id = None

    def _advance(self):
        with self._lock:
            idx = min(self._idx, len(self._seq) - 1)
            self._idx += 1
        return self._seq[idx]

    def get_arm_joint_state(self):
        return SimpleNamespace(angles=(0.0,) * 6)

    def get_eef_joint_state(self):
        return SimpleNamespace(eef_pos=0.0887)

    def get_end_pose(self):
        q = self._advance()
        return SimpleNamespace(position=(0.28, 0.0, 0.21), orientation=q)

    def acquire_control(self, lease_ms=15000, renew_period_s=5.0):
        return True

    def release_control(self):
        return None

    def close(self):
        return None


def test_poller_unwraps_rpy_across_pi_boundary():
    # Craft two rpy that as_euler will return as roll≈+3.1 then roll≈-3.1
    # (i.e. crossing ±π). After unwrap, the second sample's roll should be
    # close to +3.1 + (2π - 6.2) ≈ +3.1832, NOT -3.1.
    rpy_a = np.array([3.1, 0.1, 0.2], dtype=np.float64)
    rpy_b = np.array([-3.1, 0.1, 0.2], dtype=np.float64)  # represents roll near -π
    q_a = R.from_euler("xyz", rpy_a).as_quat()
    q_b = R.from_euler("xyz", rpy_b).as_quat()
    # Repeat to keep poller alive after the sequence is consumed.
    seq = [tuple(q_a), tuple(q_b)] + [tuple(q_b)] * 80

    mock_sdk = _RpyEdgeSdk(seq)
    arm = _make_client(mock_sdk, poll_hz=200.0)
    arm._test_start()
    try:
        # Wait for at least both samples to be consumed.
        assert _wait_until(lambda: mock_sdk._idx >= 2, timeout_s=2.0)
        # Give the poller one more tick to land the second snapshot in cache.
        time.sleep(0.05)
        snap = arm.latest()
        assert snap is not None
        unwrapped_roll = float(snap.eef_rpy[0])
        # Without unwrap this would be ~ -3.1 (diff > π from prev +3.1).
        # With unwrap it lands near +3.1 + (2π - 6.2) ≈ +3.1832.
        assert unwrapped_roll > 3.0, snap.eef_rpy
        # And specifically the jump from previous frame must be < π.
        diff = abs(unwrapped_roll - 3.1)
        assert diff < np.pi, f"rpy not unwrapped (diff={diff})"
    finally:
        arm.stop()


# ---------------------------------------------------------------------------
# PR11 R14: send_pose sanity checks
# ---------------------------------------------------------------------------
def test_send_pose_rejects_unreachable_xyz():
    mock_sdk = _make_mock_sdk_client()
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        with pytest.raises(ValueError, match="target_xyz"):
            arm.send_pose(np.array([2.0, 0.0, 0.0]), np.zeros(3), gripper_m=0.05)
        # No RPC must have been issued.
        assert mock_sdk.move_end_pose.call_count == 0
    finally:
        arm.stop()


def test_send_pose_rejects_out_of_range_rpy():
    mock_sdk = _make_mock_sdk_client()
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        with pytest.raises(ValueError, match="target_rpy"):
            arm.send_pose(np.zeros(3), np.array([4.0, 0.0, 0.0]), gripper_m=0.05)
        assert mock_sdk.move_end_pose.call_count == 0
    finally:
        arm.stop()


def test_send_pose_rejects_invalid_gripper():
    mock_sdk = _make_mock_sdk_client()
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        with pytest.raises(ValueError, match="gripper_m"):
            arm.send_pose(np.zeros(3), np.zeros(3), gripper_m=-0.01)
        with pytest.raises(ValueError, match="gripper_m"):
            arm.send_pose(np.zeros(3), np.zeros(3), gripper_m=0.2)
        assert mock_sdk.move_end_pose.call_count == 0
    finally:
        arm.stop()


def test_send_pose_accepts_boundary_values():
    mock_sdk = _make_mock_sdk_client()
    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        # xyz boundary 1.0 m, rpy boundary ±π, gripper 0.15 m.
        assert arm.send_pose(
            np.array([1.0, 1.0, 1.0]),
            np.array([np.pi, np.pi, np.pi]),
            gripper_m=0.15,
        ) is True
        # Lower-bound boundary for gripper.
        assert arm.send_pose(
            np.array([-1.0, -1.0, -1.0]),
            np.array([-np.pi, -np.pi, -np.pi]),
            gripper_m=0.0,
        ) is True
        # gripper=None is allowed (no eef_pos override).
        assert arm.send_pose(
            np.zeros(3), np.zeros(3), gripper_m=None
        ) is True
    finally:
        arm.stop()


# ---------------------------------------------------------------------------
# PR11 R15: send_pose auto-reacquire on lease loss
# ---------------------------------------------------------------------------
def test_send_pose_reacquires_on_lease_loss():
    mock_sdk = _make_mock_sdk_client()
    # First call to move_end_pose returns False AND clears the lease id;
    # second call (after reacquire) returns True.
    call_order = {"n": 0}

    def _move_side_effect(*args, **kwargs):
        call_order["n"] += 1
        if call_order["n"] == 1:
            mock_sdk._lease_id = None  # simulate lease kicked
            return False
        mock_sdk._lease_id = "lease-renewed"
        return True

    mock_sdk.move_end_pose.side_effect = _move_side_effect
    # acquire_control will be called by reacquire path -> return True + set lease id.
    def _acquire_side_effect(*args, **kwargs):
        mock_sdk._lease_id = "lease-renewed"
        return True
    mock_sdk.acquire_control.side_effect = _acquire_side_effect

    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        # Simulate that we already own the lease (so reacquire path is eligible).
        arm._acquired = True
        mock_sdk._lease_id = "lease-original"

        assert arm.send_pose(np.zeros(3), np.zeros(3), gripper_m=0.05) is True
        # Two move_end_pose calls: first failed, second succeeded.
        assert mock_sdk.move_end_pose.call_count == 2
        # One reacquire happened.
        h = arm.health()
        assert h["lease_renew_count"] == 1, h
        # Counter cleared by final success.
        assert h["consecutive_send_fail"] == 0, h
    finally:
        arm.stop()


def test_send_pose_no_reacquire_when_lease_alive():
    mock_sdk = _make_mock_sdk_client()
    mock_sdk.move_end_pose.return_value = False
    mock_sdk._lease_id = "lease-still-alive"  # not cleared

    arm = _make_client(mock_sdk)
    arm._test_start()
    try:
        arm._acquired = True
        with pytest.raises(ArmSdkError):
            arm.send_pose(np.zeros(3), np.zeros(3), gripper_m=0.05)
        # Only ONE move_end_pose call (no retry because lease is alive).
        assert mock_sdk.move_end_pose.call_count == 1
        # acquire_control must NOT have been called by send_pose retry path.
        mock_sdk.acquire_control.assert_not_called()
        h = arm.health()
        assert h["lease_renew_count"] == 0, h
        assert h["consecutive_send_fail"] == 1, h
    finally:
        arm.stop()
