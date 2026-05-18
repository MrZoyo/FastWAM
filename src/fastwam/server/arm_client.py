"""ARM client wrapper around arm_sdk.AirbotClient.

PR4 scope: background poller that reads joint / EEF / end_pose RPCs, exposes
the latest snapshot, and converts target poses to SDK CartesianPose for
``move_end_pose``. Quaternion sign is unwrapped against the previous sample.

PR8 scope: ``acquire_control`` now also switches the controller to
``servo_control`` and configures per-joint arm speed (guide §1). ``send_pose``
issues a single ``move_end_pose`` per tick with the gripper target carried in
``ArmControlOptions.eef_pos`` — calling ``move_eef`` separately would interrupt
the previous pose command (SDK PDF p.31 / guide §4.1).

PR11 scope (server-side ARM hardening, guide §7 + §2 + §4.2):
  - R6: ``health()`` exposes a GREEN/YELLOW/RED ``status`` derived from both
    poller and send-pose consecutive failures, plus a hard RED on lease loss
    (lease_alive==False while we still believe we are the owner).
  - R7: poller post-processes rpy via ``unwrap_rpy_sequence`` against the
    previous frame to eliminate ±π → ∓π jumps from ``Rotation.as_euler``.
  - R14: ``send_pose`` performs sanity-checks (workspace, rpy range, gripper
    range) before issuing the RPC; bad inputs raise ``ValueError`` (input
    validation), distinct from ``ArmSdkError`` (SDK-side failure).
  - R15: ``send_pose`` makes ONE auto-reacquire attempt when ``move_end_pose``
    returns False AND the SDK has cleared ``_lease_id`` (lease kicked); the
    re-RPC follows the reacquire. ``lease_renew_count`` in ``health()`` tracks
    these reacquires.

The lease is fully delegated to ``AirbotClient.acquire_control`` — its built-in
``_lease_thread`` auto-renews on the SDK side, so we never start our own
timer (design doc risk #7).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, NamedTuple, Optional

import numpy as np

from .rotation import (
    quat_xyzw_to_rpy,
    rpy_to_quat_xyzw,
    unwrap_quat_sign,
    unwrap_rpy_sequence,
)


class ArmSdkError(RuntimeError):
    """Raised when an SDK call reports a hard failure (return False)."""


class ArmStateSnapshot(NamedTuple):
    """Latest snapshot returned by ``ArmClient.latest``.

    All numeric fields use SI units: radians for joints / RPY, meters for
    positions, and ``time.time_ns()`` for the capture timestamp (SDK does not
    expose an upstream sample time).
    """

    angles_rad: np.ndarray         # (6,) float32
    gripper_m: float               # gripper opening in meters
    eef_xyz: np.ndarray            # (3,) float32, end-effector position (m)
    eef_rpy: np.ndarray            # (3,) float32, extrinsic XYZ Euler (rad)
    eef_quat_xyzw: np.ndarray      # (4,) float32, sign-unwrapped against last
    capture_ts_ns: int             # wall-clock ns when the RPC trio completed


class ArmClient:
    """Threaded wrapper around ``arm_sdk.AirbotClient``.

    Construction does *not* call ``acquire_control``. The background poller
    only issues read-only RPCs (which work without a lease) so the device
    remains controllable by other clients until ``acquire_control`` is invoked
    explicitly.
    """

    def __init__(
        self,
        host: str,
        port: int,
        poll_hz: float,
        state_max_age_ms: float,
        lease_ms: int,
        logger: Optional[logging.Logger] = None,
        *,
        arm_speed_rad_s: float = 0.5,
        client_factory: Optional[Any] = None,
    ) -> None:
        if poll_hz <= 0:
            raise ValueError(f"poll_hz must be > 0, got {poll_hz}")
        if state_max_age_ms <= 0:
            raise ValueError("state_max_age_ms must be > 0")
        if lease_ms < 1000:
            raise ValueError("lease_ms must be >= 1000 (SDK constraint)")

        self._host = host
        self._port = port
        self._poll_hz = float(poll_hz)
        self._poll_period_s = 1.0 / float(poll_hz)
        self._state_max_age_ns = int(state_max_age_ms * 1e6)
        self._lease_ms = int(lease_ms)
        self._logger = logger or logging.getLogger(__name__)
        # arm_speed in rad/s per joint; servo_control needs an explicit cap.
        # 0.5 rad/s (~28.6 deg/s) is the conservative default suggested by
        # the ArmClient guide §1 (set_arm_speed call right after switch_controller).
        self._arm_speed_rad_s = float(arm_speed_rad_s)

        # Allow tests to inject a fake SDK client without monkeypatching imports.
        if client_factory is None:
            from arm_sdk.client import AirbotClient  # local import: heavy gRPC stub
            client_factory = lambda: AirbotClient(host=host, port=port)
        self._client_factory = client_factory

        self._client: Any = None
        self._cart_pose_cls: Any = None  # filled in start()
        self._opts_cls: Any = None       # filled in start()

        self._state_lock = threading.Lock()
        self._state: Optional[ArmStateSnapshot] = None

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consecutive_fail = 0
        self._last_poll_ts_ns: int = 0
        self._last_quat: Optional[np.ndarray] = None
        self._last_quat_unwrap_count = 0
        self._acquired = False
        # PR11 R6: track consecutive send_pose failures for status tri-level.
        self._consecutive_send_fail = 0
        # PR11 R7: previous rpy sample for unwrap_rpy_sequence().
        self._last_rpy: Optional[np.ndarray] = None
        # PR11 R15: count auto-reacquires triggered by send_pose lease-loss path.
        self._lease_renew_count = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("ArmClient.start called twice")
        self._client = self._client_factory()
        # Resolve CartesianPose / ArmControlOptions lazily so tests can stub
        # the entire arm_sdk module.
        try:
            from arm_sdk.client import (
                ArmControlOptions as _Opts,
                CartesianPose as _CP,
            )
            self._cart_pose_cls = _CP
            self._opts_cls = _Opts
        except Exception:  # pragma: no cover - only triggered in stub envs
            self._cart_pose_cls = getattr(self._client, "_cart_pose_cls", None)
            self._opts_cls = getattr(self._client, "_opts_cls", None)

        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, name="ArmClientPoller", daemon=True
        )
        self._thread.start()
        self._logger.info(
            "[arm] poller started host=%s port=%d hz=%.1f",
            self._host, self._port, self._poll_hz,
        )

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._thread = None
        if self._client is not None:
            try:
                if self._acquired:
                    self._client.release_control()
            except Exception:  # pragma: no cover - best-effort cleanup
                self._logger.exception("[arm] release_control on stop failed")
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                self._logger.exception("[arm] client.close on stop failed")
            self._client = None
            self._acquired = False

    # ------------------------------------------------------------------
    # lease / control
    # ------------------------------------------------------------------
    def acquire_control(self) -> bool:
        """Acquire lease, switch to servo_control, set arm speed.

        servo_control is required for streaming 20 Hz pose targets via
        ``move_end_pose`` (guide §1, SDK PDF p.21). On any failure after
        ``acquire_control`` succeeds we release the lease before returning.
        """
        if self._client is None:
            raise RuntimeError("ArmClient.acquire_control before start()")
        ok = bool(
            self._client.acquire_control(
                lease_ms=self._lease_ms, renew_period_s=5.0
            )
        )
        if not ok:
            self._logger.error("[arm] acquire_control returned False")
            self._acquired = False
            return False
        # Lazy import the Controller enum so tests can keep arm_sdk fully stubbed.
        try:
            from arm_sdk.client import Controller as _Controller
        except Exception:  # pragma: no cover - only in stub envs
            _Controller = getattr(self._client, "_controller_enum", None)
        if _Controller is not None and hasattr(self._client, "switch_controller"):
            switched = bool(
                self._client.switch_controller(_Controller.servo_control)
            )
            if not switched:
                self._logger.error(
                    "[arm] switch_controller(servo_control) returned False"
                )
                try:
                    self._client.release_control()
                finally:
                    self._acquired = False
                return False
            self._logger.info("[arm] switched to servo_control")
        if hasattr(self._client, "set_arm_speed"):
            try:
                self._client.set_arm_speed([self._arm_speed_rad_s] * 6)
                self._logger.info(
                    "[arm] set_arm_speed=%.3f rad/s per joint",
                    self._arm_speed_rad_s,
                )
            except Exception:  # pragma: no cover - best-effort
                self._logger.exception(
                    "[arm] set_arm_speed raised; continuing"
                )
        self._acquired = True
        return True

    def release_control(self) -> None:
        if self._client is None:
            return
        try:
            self._client.release_control()
        finally:
            self._acquired = False

    def emergency_stop(self, enable: bool = True) -> None:
        if self._client is None:
            raise RuntimeError("ArmClient.emergency_stop before start()")
        self._client.set_arm_emergency_stop(bool(enable))

    # ------------------------------------------------------------------
    # state cache
    # ------------------------------------------------------------------
    def latest(self) -> Optional[ArmStateSnapshot]:
        with self._state_lock:
            snap = self._state
        if snap is None:
            return None
        age_ns = time.time_ns() - snap.capture_ts_ns
        if age_ns > self._state_max_age_ns:
            return None
        return snap

    def health(self) -> dict:
        # PR11 R6: ``status`` is GREEN/YELLOW/RED from the worst of
        # consecutive_fail (poll) and consecutive_send_fail (move_end_pose):
        # 0-2 GREEN, 3-4 YELLOW, >=5 RED. Lease loss while ``_acquired``
        # forces RED regardless of the counters.
        now_ns = time.time_ns()
        with self._state_lock:
            last_poll = self._last_poll_ts_ns
            fail = self._consecutive_fail
            unwrap = self._last_quat_unwrap_count
            send_fail = self._consecutive_send_fail
            lease_renew_count = self._lease_renew_count
        if last_poll == 0:
            last_age_ms: Optional[float] = None
        else:
            last_age_ms = (now_ns - last_poll) / 1e6
        lease_alive = False
        try:
            lease_alive = (
                self._client is not None
                and getattr(self._client, "_lease_id", None) is not None
            )
        except Exception:  # pragma: no cover
            lease_alive = False

        worst = max(fail, send_fail)
        if worst >= 5:
            status = "RED"
        elif worst >= 3:
            status = "YELLOW"
        else:
            status = "GREEN"
        # PR11 R6: lease-loss while still believing we own it is hard RED.
        if (not lease_alive) and self._acquired:
            status = "RED"

        return {
            "last_poll_age_ms": last_age_ms,
            "consecutive_fail": fail,
            "consecutive_send_fail": send_fail,
            "lease_alive": lease_alive,
            "lease_renew_count": lease_renew_count,
            "last_quat_unwrap_count": unwrap,
            "running": self._thread is not None and self._thread.is_alive(),
            "status": status,
        }

    # ------------------------------------------------------------------
    # command
    # ------------------------------------------------------------------
    def send_pose(
        self,
        target_xyz: np.ndarray,
        target_rpy: np.ndarray,
        gripper_m: Optional[float],
    ) -> bool:
        # PR11 order: sanity-check (R14, ValueError) -> build pose ->
        # move_end_pose -> on False+lease cleared: ONE reacquire + re-RPC (R15)
        # -> on still False: bump consecutive_send_fail (R6) + raise ArmSdkError.
        if self._client is None:
            raise RuntimeError("ArmClient.send_pose before start()")

        xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3)
        rpy = np.asarray(target_rpy, dtype=np.float64).reshape(3)

        # PR11 R14: sanity check (guide §4.2). |xyz|<=1 m (Airbot reach <0.7m),
        # rpy in (-π, π] with ε tolerance, gripper 0..0.15 m (G2/G2T/G2L).
        if not np.all(np.isfinite(xyz)) or not np.all(np.abs(xyz) <= 1.0):
            raise ValueError(f"target_xyz out of workspace |xyz|<=1m, got {xyz.tolist()}")
        if not np.all(np.isfinite(rpy)) or not np.all(np.abs(rpy) <= np.pi + 0.01):
            raise ValueError(f"target_rpy out of (-π, π], got {rpy.tolist()}")
        if gripper_m is not None and (
            not np.isfinite(gripper_m) or not (0.0 <= float(gripper_m) <= 0.15)
        ):
            raise ValueError(f"gripper_m out of [0, 0.15] m, got {gripper_m}")

        quat = rpy_to_quat_xyzw(rpy)

        if self._cart_pose_cls is None or self._opts_cls is None:
            raise RuntimeError(
                "CartesianPose / ArmControlOptions class not resolved; was start() called?"
            )

        pose = self._cart_pose_cls(
            position=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            orientation=(
                float(quat[0]), float(quat[1]),
                float(quat[2]), float(quat[3]),
            ),
        )
        opts = self._opts_cls()  # blocking=False by default
        # Gripper rides inside opts.eef_pos so that pose + gripper update is
        # a single RPC. Calling move_eef separately would interrupt the
        # previous move_end_pose (SDK PDF p.31 / guide §4.1).
        if gripper_m is not None:
            opts.eef_pos = float(gripper_m)

        try:
            ok = bool(self._client.move_end_pose(pose, opts))
        except Exception:
            with self._state_lock:
                self._consecutive_send_fail += 1
            raise

        if not ok:
            # PR11 R15: lease may have been kicked by another client. The SDK
            # clears _lease_id when its lease thread sees the server take it
            # away. If we still believe we hold it, try ONE reacquire.
            lease_cleared = (
                hasattr(self._client, "_lease_id")
                and self._client._lease_id is None
                and self._acquired
            )
            if lease_cleared:
                self._logger.warning(
                    "[arm] move_end_pose failed and lease cleared; "
                    "attempting one reacquire"
                )
                # acquire_control() re-runs switch_controller + set_arm_speed.
                if self.acquire_control():
                    with self._state_lock:
                        self._lease_renew_count += 1
                    try:
                        ok = bool(self._client.move_end_pose(pose, opts))
                    except Exception:
                        with self._state_lock:
                            self._consecutive_send_fail += 1
                        raise

        if not ok:
            with self._state_lock:
                self._consecutive_send_fail += 1
            raise ArmSdkError(
                "move_end_pose returned False (after reacquire if attempted)"
            )

        with self._state_lock:
            self._consecutive_send_fail = 0
        return True

    # ------------------------------------------------------------------
    # poller
    # ------------------------------------------------------------------
    def _poll_once(self) -> Optional[ArmStateSnapshot]:
        try:
            joint = self._client.get_arm_joint_state()
            eef = self._client.get_eef_joint_state()
            pose = self._client.get_end_pose()
        except Exception:
            self._logger.exception("[arm] poll RPC raised")
            return None
        if joint is None or eef is None or pose is None:
            return None

        capture_ts_ns = time.time_ns()
        try:
            angles = np.asarray(joint.angles, dtype=np.float32).reshape(-1)
        except Exception:
            self._logger.exception("[arm] failed to parse joint.angles")
            return None
        if angles.shape != (6,):
            self._logger.warning("[arm] unexpected angles shape: %s", angles.shape)
            # still keep going; ARM might use a different DOF in the future

        gripper_m = float(eef.eef_pos)
        try:
            xyz = np.asarray(pose.position, dtype=np.float32).reshape(3)
            quat_raw = np.asarray(pose.orientation, dtype=np.float64).reshape(4)
        except Exception:
            self._logger.exception("[arm] failed to parse end_pose tuple")
            return None

        quat = unwrap_quat_sign(quat_raw, self._last_quat)
        unwrap_inc = 0
        if self._last_quat is not None:
            # sign got flipped iff dot(prev, raw) < 0
            if float(np.dot(quat_raw, self._last_quat)) < 0.0:
                unwrap_inc = 1
        self._last_quat = quat.astype(np.float64)
        rpy = quat_xyzw_to_rpy(quat)

        # PR11 R7: even with sign-unwrapped quaternion, ``as_euler("xyz")`` can
        # still produce ±π discontinuities (e.g. roll near π flips to -π on
        # the next sample). Unwrap rpy against the previous frame so that the
        # snapshot stream stays continuous for downstream consumers.
        if self._last_rpy is not None:
            seq = np.stack([
                self._last_rpy.astype(np.float64),
                rpy.astype(np.float64),
            ])
            seq = unwrap_rpy_sequence(seq)
            rpy = seq[-1].astype(np.float32)
        self._last_rpy = rpy.copy()

        snap = ArmStateSnapshot(
            angles_rad=angles,
            gripper_m=gripper_m,
            eef_xyz=xyz,
            eef_rpy=rpy,
            eef_quat_xyzw=quat.astype(np.float32),
            capture_ts_ns=capture_ts_ns,
        )

        with self._state_lock:
            self._state = snap
            self._last_poll_ts_ns = capture_ts_ns
            self._last_quat_unwrap_count += unwrap_inc
        return snap

    def _poll_loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop_evt.is_set():
            snap = self._poll_once()
            if snap is None:
                with self._state_lock:
                    self._consecutive_fail += 1
                    fail = self._consecutive_fail
                if fail == 5:
                    self._logger.error(
                        "[arm] poller RED: %d consecutive RPC failures", fail
                    )
            else:
                with self._state_lock:
                    self._consecutive_fail = 0

            next_t += self._poll_period_s
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                # use Event.wait so stop() interrupts immediately
                if self._stop_evt.wait(timeout=sleep_s):
                    return
            else:
                # behind schedule; reset to avoid runaway catch-up
                next_t = time.monotonic()
