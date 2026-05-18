"""Closed-loop runner. PR5 scope = InferLoop only.

Dispatch + watchdog are NoOp placeholders whose signatures match PR6
(see TEMP comments below). See design doc §7.6 for full design.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import yaml

from .arm_client import ArmClient
from .chunk_ringbuffer import ChunkEntry, ChunkRingBuffer
from .dispatch import DispatchLoop
from .image_pipeline import undistort_native
from .watchdog import Watchdog
from .ws_ingest import FrameSnapshot, WSFrameIngester


# TEMP: replaced by PR6 dispatch.DispatchLoop
#   Signature: DispatchLoop(ringbuffer, arm_client, send_period_ms,
#                           blend_frames, chunk_max_stale_ms, ...)
#   Methods: start() / stop() / status() -> dict
class NoOpDispatch:
    """PR5 placeholder for the PR6 DispatchLoop."""

    def __init__(self, ringbuffer, arm_client, send_period_ms, blend_frames,
                 chunk_max_stale_ms, **kwargs: Any) -> None:
        self.ringbuffer = ringbuffer
        self.arm_client = arm_client
        self.send_period_ms = int(send_period_ms)
        self.blend_frames = int(blend_frames)
        self.chunk_max_stale_ms = int(chunk_max_stale_ms)
        self._extra = dict(kwargs)
        self._running = False

    def start(self) -> None: self._running = True
    def stop(self) -> None: self._running = False

    def status(self) -> dict:
        return {"kind": "noop", "running": self._running,
                "send_period_ms": self.send_period_ms,
                "blend_frames": self.blend_frames,
                "chunk_max_stale_ms": self.chunk_max_stale_ms}


# TEMP: replaced by PR6 watchdog.Watchdog
#   Signature: Watchdog(ringbuffer, arm_client, ws_ingester,
#                       runner_status_provider, watchdog_period_ms, ...)
#   Methods: start() / stop() / status() -> dict
class NoOpWatchdog:
    """PR5 placeholder for the PR6 Watchdog."""

    def __init__(self, ringbuffer, arm_client, ws_ingester,
                 watchdog_period_ms=10, runner_status_provider=None,
                 **kwargs: Any) -> None:
        self.ringbuffer = ringbuffer
        self.arm_client = arm_client
        self.ws_ingester = ws_ingester
        self.runner_status_provider = runner_status_provider
        self.watchdog_period_ms = int(watchdog_period_ms)
        self._extra = dict(kwargs)
        self._running = False

    def start(self) -> None: self._running = True
    def stop(self) -> None: self._running = False

    def status(self) -> dict:
        return {"kind": "noop", "running": self._running,
                "watchdog_period_ms": self.watchdog_period_ms}


class ClosedLoopRunner:
    """Owns InferLoop thread + chunk ring buffer.

    Dispatch / watchdog are constructed via the injected factories
    (default NoOps in PR5). PR6 factories preserve the same
    ``start()/stop()/status()`` interface.
    """

    def __init__(
        self,
        model_client: Any,
        arm_client: ArmClient,
        ws_ingester: WSFrameIngester,
        train_config_path: Path | str,
        default_camera_info: dict[str, dict],
        stereo_pair: dict[str, str],
        infer_period_ms: int = 400,
        send_period_ms: int = 50,
        chunk_len: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        blend_frames: int = 4,
        chunk_max_stale_ms: int = 2000,
        auto_dispatch: bool = False,
        emergency_on_failure: bool = True,
        watchdog_period_ms: int = 10,
        image_pipeline: str = "raw_native",
        undistort: bool = True,
        default_instruction: Optional[str] = None,
        ws_warn_stale_ms: int = 200,
        ws_hold_stale_ms: int = 500,
        ws_estop_stale_ms: int = 1500,
        dispatcher_factory: Optional[Callable[..., Any]] = None,
        watchdog_factory: Optional[Callable[..., Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if infer_period_ms <= 0:
            raise ValueError("infer_period_ms must be > 0")
        if send_period_ms <= 0:
            raise ValueError("send_period_ms must be > 0")
        if blend_frames < 0:
            raise ValueError("blend_frames must be >= 0")
        if image_pipeline not in ("raw_native", "lerobot_480x640"):
            raise ValueError(
                f"image_pipeline must be 'raw_native' or 'lerobot_480x640', got {image_pipeline!r}"
            )
        if "left" not in stereo_pair or "right" not in stereo_pair:
            raise ValueError("stereo_pair must contain 'left' and 'right' keys")

        self.model_client = model_client
        self.arm_client = arm_client
        self.ws_ingester = ws_ingester
        self.train_config_path = Path(train_config_path).expanduser().resolve()
        self.default_camera_info = default_camera_info
        self.stereo_pair = dict(stereo_pair)
        self.infer_period_ms = int(infer_period_ms)
        self.send_period_ms = int(send_period_ms)
        self.blend_frames = int(blend_frames)
        self.chunk_max_stale_ms = int(chunk_max_stale_ms)
        self.auto_dispatch = bool(auto_dispatch)
        self.emergency_on_failure = bool(emergency_on_failure)
        self.watchdog_period_ms = int(watchdog_period_ms)
        self.image_pipeline = str(image_pipeline)
        self.undistort = bool(undistort)
        self.default_instruction = default_instruction
        self.ws_warn_stale_ms = int(ws_warn_stale_ms)
        self.ws_hold_stale_ms = int(ws_hold_stale_ms)
        self.ws_estop_stale_ms = int(ws_estop_stale_ms)
        self._logger = logger or logging.getLogger("fastwam.closed_loop")

        # Config-driven validation (design doc §7.6).
        train_cfg = self._load_train_config(self.train_config_path)
        cfg_num_frames = int(train_cfg["data"]["train"]["num_frames"])
        cfg_action_dim = int(train_cfg["data"]["train"]["processor"]["action_output_dim"])
        cfg_eval_steps = int(train_cfg.get("eval_num_inference_steps", 10))
        expected_chunk_len = cfg_num_frames - 1

        if chunk_len is None:
            chunk_len = expected_chunk_len
        elif int(chunk_len) != expected_chunk_len:
            raise RuntimeError(
                f"chunk_len={chunk_len} != expected_chunk_len={expected_chunk_len} "
                f"(num_frames={cfg_num_frames})"
            )
        client_horizon = getattr(model_client, "action_horizon", None)
        if client_horizon is not None and int(client_horizon) != int(chunk_len):
            raise RuntimeError(
                f"model_client.action_horizon={client_horizon} != chunk_len={chunk_len}"
            )
        if cfg_action_dim != 7:
            raise RuntimeError(
                f"PR5 only supports 7-DoF; train action_output_dim={cfg_action_dim}"
            )
        if num_inference_steps is None:
            num_inference_steps = cfg_eval_steps
        self.chunk_len = int(chunk_len)
        self.num_inference_steps = int(num_inference_steps)

        # Defensive: server pipeline assumes 7-DoF proprio (6 joints + gripper).
        # Triggered when CLI points at a config whose proprio dim != 7 (e.g. cup task).
        # Use isinstance(int) so MagicMock-style attribute access in tests is tolerated.
        client_proprio = getattr(model_client, "proprio_dim", None)
        if isinstance(client_proprio, int) and client_proprio != 7:
            raise RuntimeError(
                f"model_client.proprio_dim={client_proprio} != 7; "
                "active loop server expects [j0..j5, gripper_m]. "
                "Check --config matches the 7-DoF real_1048 task."
            )

        # Runtime state.
        self.ringbuffer = ChunkRingBuffer(capacity=2)
        self._chunk_id_counter = 0
        self._instruction: Optional[str] = None
        self._stop_evt = threading.Event()
        self._infer_thread: Optional[threading.Thread] = None
        self._infer_latency_history: deque[dict[str, float]] = deque(maxlen=100)
        self._last_skip_reason: Optional[str] = None
        self._last_infer_end_ns: int = 0
        self._consecutive_infer_fail = 0
        self._hold_mode = False  # PR6 watchdog flips this

        # Injected dispatcher / watchdog. Default = NoOp (PR5 unit tests stay
        # green). active_loop_server.py injects PR6's DispatchLoop / Watchdog
        # via make_production_factories() below.
        df = dispatcher_factory or NoOpDispatch
        wf = watchdog_factory or NoOpWatchdog
        # Pass auto_dispatch to dispatcher (PR6 needs it; NoOp absorbs via **kwargs).
        self._dispatch = df(
            ringbuffer=self.ringbuffer, arm_client=self.arm_client,
            send_period_ms=self.send_period_ms, blend_frames=self.blend_frames,
            chunk_max_stale_ms=self.chunk_max_stale_ms,
            auto_dispatch=self.auto_dispatch,
            emergency_on_failure=self.emergency_on_failure, logger=self._logger,
        )
        # Pass dispatcher to watchdog (PR6 real Watchdog requires it; NoOp
        # absorbs via **kwargs). runner_status_provider stays NoOp-only and is
        # not passed here.
        self._watchdog = wf(
            ringbuffer=self.ringbuffer, arm_client=self.arm_client,
            ws_ingester=self.ws_ingester, dispatcher=self._dispatch,
            watchdog_period_ms=self.watchdog_period_ms,
            chunk_max_stale_ms=self.chunk_max_stale_ms,
            infer_period_ms=self.infer_period_ms,
            ws_warn_stale_ms=self.ws_warn_stale_ms,
            ws_hold_stale_ms=self.ws_hold_stale_ms,
            ws_estop_stale_ms=self.ws_estop_stale_ms,
            logger=self._logger,
        )

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self, instruction: Optional[str] = None) -> None:
        if self._infer_thread is not None:
            raise RuntimeError("ClosedLoopRunner.start called twice")
        self._instruction = instruction if instruction is not None else self.default_instruction
        self._stop_evt.clear()
        self._consecutive_infer_fail = 0
        self._last_skip_reason = None
        self.ringbuffer.clear()
        if self.auto_dispatch:
            self._dispatch.start()
        self._watchdog.start()
        self._infer_thread = threading.Thread(
            target=self._infer_loop, name="ClosedLoopInfer", daemon=True
        )
        self._infer_thread.start()
        self._logger.info(
            "[runner] started instruction=%r chunk_len=%d steps=%d pipeline=%s auto_dispatch=%s",
            self._instruction, self.chunk_len, self.num_inference_steps,
            self.image_pipeline, self.auto_dispatch,
        )

    def stop(self) -> None:
        self._stop_evt.set()
        try: self._dispatch.stop()
        except Exception: self._logger.exception("[runner] dispatch.stop raised")
        try: self._watchdog.stop()
        except Exception: self._logger.exception("[runner] watchdog.stop raised")
        t = self._infer_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._infer_thread = None

    def emergency(self) -> None:
        self._stop_evt.set()
        try: self.arm_client.emergency_stop(True)
        except Exception: self._logger.exception("[runner] arm.emergency_stop raised")
        self.stop()

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------
    def status(self) -> dict:
        latest = self.ringbuffer.latest()
        prev = self.ringbuffer.prev()
        latencies = [item["total_ms"] for item in self._infer_latency_history]
        p50 = p95 = p99 = None
        if latencies:
            a = np.asarray(latencies, dtype=np.float64)
            p50 = float(np.percentile(a, 50))
            p95 = float(np.percentile(a, 95))
            p99 = float(np.percentile(a, 99))
        return {
            "running": self._infer_thread is not None and self._infer_thread.is_alive(),
            "current_chunk_id": latest.chunk_id if latest else None,
            "prev_chunk_id": prev.chunk_id if prev else None,
            "ringbuffer_occupancy": self.ringbuffer.occupancy(),
            "ringbuffer_max_occupancy_seen": self.ringbuffer.max_occupancy_seen(),
            "infer_count": len(self._infer_latency_history),
            "infer_latency_ms_p50": p50,
            "infer_latency_ms_p95": p95,
            "infer_latency_ms_p99": p99,
            "last_skip_reason": self._last_skip_reason,
            "consecutive_infer_fail": self._consecutive_infer_fail,
            "hold_mode": self._hold_mode,
            "instruction": self._instruction,
            "chunk_len": self.chunk_len,
            "num_inference_steps": self.num_inference_steps,
            "image_pipeline": self.image_pipeline,
            "auto_dispatch": self.auto_dispatch,
            "dispatch": self._dispatch.status() if hasattr(self._dispatch, "status") else None,
            "watchdog": self._watchdog.status() if hasattr(self._watchdog, "status") else None,
        }

    # ------------------------------------------------------------------
    # debug hook for PR7 /debug/zero_pose_test (design doc §9)
    # ------------------------------------------------------------------
    def inject_chunk_for_debug(
        self, action_abs: np.ndarray, base_capture_ts_ns: Optional[int] = None
    ) -> ChunkEntry:
        """Push a synthetic ChunkEntry directly into the ringbuffer, bypassing
        the model. Used by /debug/zero_pose_test to verify the coordinate frame:
        actions=repeat(current_pose, chunk_len) should leave the arm stationary.
        """
        action_abs = np.asarray(action_abs, dtype=np.float32)
        if action_abs.ndim != 2 or action_abs.shape != (self.chunk_len, 7):
            raise ValueError(
                f"inject_chunk_for_debug expected shape=({self.chunk_len}, 7), got {action_abs.shape}"
            )
        self._chunk_id_counter += 1
        entry = ChunkEntry(
            action_abs=action_abs,
            base_capture_ts_ns=int(base_capture_ts_ns) if base_capture_ts_ns is not None else time.time_ns(),
            step_dt_ns=self.send_period_ms * 1_000_000,
            chunk_id=self._chunk_id_counter,
        )
        self.ringbuffer.push(entry)
        return entry

    # ------------------------------------------------------------------
    # InferLoop
    # ------------------------------------------------------------------
    def _infer_loop(self) -> None:
        period_s = self.infer_period_ms / 1000.0
        next_t = time.monotonic()
        while not self._stop_evt.is_set():
            self._infer_once()
            next_t += period_s
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                if self._stop_evt.wait(timeout=sleep_s):
                    return
            else:
                next_t = time.monotonic()  # behind schedule -- reset

    def _infer_once(self) -> None:
        left_key = self.stereo_pair["left"]
        right_key = self.stereo_pair["right"]

        # 1. Snapshot frames (WS first) then ARM state.
        head_frame = self.ws_ingester.latest(left_key)
        wrist_frame = self.ws_ingester.latest(right_key)
        if head_frame is None:
            self._record_skip(f"stale_frame:{left_key}"); return
        if wrist_frame is None:
            self._record_skip(f"stale_frame:{right_key}"); return
        state = self.arm_client.latest()
        if state is None:
            self._record_skip("stale_arm_state"); return

        # 2. Image pipeline.
        t0 = time.perf_counter()
        try:
            images = self._build_image_dict(head_frame, wrist_frame)
        except Exception as exc:
            self._logger.exception("[runner] image pipeline failed")
            self._record_skip(f"image_pipeline_error:{exc}"); return
        t_image_end = time.perf_counter()

        # 3. Assemble model input (proprio_raw = angles+gripper; current_position = xyz+rpy).
        try:
            proprio_raw = np.concatenate([
                np.asarray(state.angles_rad, dtype=np.float32).reshape(-1),
                np.asarray([state.gripper_m], dtype=np.float32),
            ])
            current_position = np.concatenate([
                np.asarray(state.eef_xyz, dtype=np.float32).reshape(-1),
                np.asarray(state.eef_rpy, dtype=np.float32).reshape(-1),
            ])
            model_input: dict[str, Any] = {
                "images": images,
                "proprio_raw": proprio_raw,
                "current_position": current_position,
                "instruction": self._instruction or "",
            }
        except Exception as exc:
            self._logger.exception("[runner] model_input assembly failed")
            self._record_skip(f"input_assembly_error:{exc}"); return

        # 4. Model inference.
        t_model_start = time.perf_counter()
        try:
            result = self.model_client.infer(model_input)
        except Exception as exc:
            self._consecutive_infer_fail += 1
            self._logger.exception("[runner] model_client.infer raised")
            self._record_skip(f"model_error:{exc}"); return
        t_model_end = time.perf_counter()

        # 5. Validate + push to ringbuffer.
        try:
            actions = np.asarray(result["actions"], dtype=np.float32)
        except Exception as exc:
            self._consecutive_infer_fail += 1
            self._record_skip(f"missing_actions:{exc}"); return
        if actions.shape != (self.chunk_len, 7):
            self._consecutive_infer_fail += 1
            msg = f"action shape mismatch: got {actions.shape}, expected ({self.chunk_len}, 7)"
            self._logger.error("[runner] %s", msg)
            self._record_skip(msg)
            raise RuntimeError(msg)

        self._consecutive_infer_fail = 0
        self._chunk_id_counter += 1
        entry = ChunkEntry(
            action_abs=actions,
            base_capture_ts_ns=int(head_frame.capture_ts_ns),
            step_dt_ns=int(self.send_period_ms) * 1_000_000,
            chunk_id=self._chunk_id_counter,
        )
        evicted = self.ringbuffer.push(entry)
        t_post_end = time.perf_counter()
        self._last_infer_end_ns = time.time_ns()
        self._infer_latency_history.append({
            "image_prep_ms": (t_image_end - t0) * 1000.0,
            "model_ms": (t_model_end - t_model_start) * 1000.0,
            "postproc_ms": (t_post_end - t_model_end) * 1000.0,
            "total_ms": (t_post_end - t0) * 1000.0,
        })
        if evicted is not None:
            self._logger.debug(
                "[runner] ringbuffer evicted chunk_id=%d (new=%d)",
                evicted.chunk_id, entry.chunk_id,
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_image_dict(self, head_frame: FrameSnapshot,
                          wrist_frame: FrameSnapshot) -> dict[str, np.ndarray]:
        raw = {self.stereo_pair["left"]: head_frame.bgr,
               self.stereo_pair["right"]: wrist_frame.bgr}
        if not self.undistort:
            return raw
        if self.image_pipeline == "raw_native":
            return undistort_native(raw, self.default_camera_info, self.stereo_pair, alpha=0.0)
        # lerobot_480x640: undistort, then resize to training resolution (v1 fallback).
        undist = undistort_native(raw, self.default_camera_info, self.stereo_pair, alpha=0.0)
        import cv2
        return {k: cv2.resize(v, (640, 480), interpolation=cv2.INTER_AREA)
                for k, v in undist.items()}

    def _record_skip(self, reason: str) -> None:
        self._last_skip_reason = reason
        self._logger.warning("[runner] skip: %s", reason)

    @staticmethod
    def _load_train_config(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def make_production_factories() -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Return (dispatcher_factory, watchdog_factory) wired to PR6 real classes.

    active_loop_server.py passes the result into ClosedLoopRunner kwargs to
    swap in production DispatchLoop / Watchdog (NoOps are the default for
    unit tests).
    """

    def dispatcher_factory(**kw: Any) -> DispatchLoop:
        return DispatchLoop(
            ringbuffer=kw["ringbuffer"],
            arm_client=kw["arm_client"],
            send_period_ms=kw["send_period_ms"],
            blend_frames=kw["blend_frames"],
            chunk_max_stale_ms=kw["chunk_max_stale_ms"],
            auto_dispatch=kw.get("auto_dispatch", False),
            emergency_on_failure=kw.get("emergency_on_failure", True),
            logger=kw.get("logger"),
        )

    def watchdog_factory(**kw: Any) -> Watchdog:
        return Watchdog(
            ringbuffer=kw["ringbuffer"],
            arm_client=kw["arm_client"],
            ws_ingester=kw["ws_ingester"],
            dispatcher=kw["dispatcher"],
            watchdog_period_ms=kw["watchdog_period_ms"],
            chunk_max_stale_ms=kw.get("chunk_max_stale_ms", 2000),
            infer_period_ms=kw.get("infer_period_ms", 400),
            ws_warn_stale_ms=kw.get("ws_warn_stale_ms", 200),
            ws_hold_stale_ms=kw.get("ws_hold_stale_ms", 500),
            ws_estop_stale_ms=kw.get("ws_estop_stale_ms", 1500),
            logger=kw.get("logger"),
        )

    return dispatcher_factory, watchdog_factory


__all__ = [
    "ClosedLoopRunner",
    "NoOpDispatch",
    "NoOpWatchdog",
    "make_production_factories",
]
