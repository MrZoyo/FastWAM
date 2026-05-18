"""Unit tests for fastwam.server.closed_loop.

Strategy: fully mock the heavy collaborators (model_client / arm_client /
ws_ingester). Drive InferLoop synchronously via the private
``_infer_once`` method so we never wait on real timing.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from fastwam.server.chunk_ringbuffer import ChunkEntry, ChunkRingBuffer
from fastwam.server.closed_loop import (
    ClosedLoopRunner,
    NoOpDispatch,
    NoOpWatchdog,
)
from fastwam.server.ws_ingest import FrameSnapshot

CHUNK_LEN = 32  # = num_frames(33) - 1
ACTION_DIM = 7


def _write_train_config(
    tmp_path: Path,
    *,
    num_frames: int = 33,
    action_output_dim: int = 7,
    eval_num_inference_steps: int = 10,
) -> Path:
    cfg = {
        "eval_num_inference_steps": eval_num_inference_steps,
        "data": {
            "train": {
                "num_frames": num_frames,
                "processor": {"action_output_dim": action_output_dim},
            }
        },
    }
    p = tmp_path / "train_config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def _make_frame(stamp_ns: int = 1_000_000_000, h: int = 1088, w: int = 1280) -> FrameSnapshot:
    return FrameSnapshot(
        bgr=np.zeros((h, w, 3), dtype=np.uint8),
        capture_ts_ns=stamp_ns,
        decode_ts_ns=stamp_ns + 1,
        pair_seq=0,
    )


def _make_state() -> SimpleNamespace:
    return SimpleNamespace(
        angles_rad=np.zeros(6, dtype=np.float32),
        gripper_m=0.05,
        eef_xyz=np.array([0.2, 0.0, 0.3], dtype=np.float32),
        eef_rpy=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        eef_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        capture_ts_ns=1_000_000_500,
    )


def _make_mock_model_client(action_horizon: int = CHUNK_LEN) -> MagicMock:
    mc = MagicMock()
    mc.action_horizon = action_horizon

    def _infer(model_input):
        # Echo a deterministic chunk shaped (chunk_len, 7).
        return {
            "actions": np.zeros((action_horizon, ACTION_DIM), dtype=np.float32),
            "action_format": "cartesian_absolute",
        }
    mc.infer.side_effect = _infer
    return mc


def _make_mock_ws_ingester(left="head_left", right="right_wrist_left") -> MagicMock:
    ws = MagicMock()
    frames = {left: _make_frame(stamp_ns=1_000_000_000),
              right: _make_frame(stamp_ns=1_000_000_100)}
    ws._frames = frames  # exposed for tests to mutate
    ws.latest.side_effect = lambda key: frames.get(key)
    return ws


def _make_mock_arm_client() -> MagicMock:
    ac = MagicMock()
    ac._state: Optional[SimpleNamespace] = _make_state()
    ac.latest.side_effect = lambda: ac._state
    return ac


def _make_default_camera_info() -> dict:
    K = [[700.0, 0.0, 640.0], [0.0, 700.0, 544.0], [0.0, 0.0, 1.0]]
    D = [0.0, 0.0, 0.0, 0.0, 0.0]
    base = {
        "K": K, "D": D,
        "width": 1280, "height": 1088,
        "distortion_model": "plumb_bob",
        "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "P": [[700.0, 0.0, 640.0, 0.0], [0.0, 700.0, 544.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
    }
    return {"head_left": base, "right_wrist_left": dict(base)}


# ---------- factories that record calls (used to bypass real undistort) ----
def _make_runner(tmp_path: Path, **overrides) -> ClosedLoopRunner:
    cfg_path = _write_train_config(tmp_path)
    kwargs = dict(
        model_client=_make_mock_model_client(),
        arm_client=_make_mock_arm_client(),
        ws_ingester=_make_mock_ws_ingester(),
        train_config_path=cfg_path,
        default_camera_info=_make_default_camera_info(),
        stereo_pair={"left": "head_left", "right": "right_wrist_left"},
        infer_period_ms=400,
        send_period_ms=50,
        blend_frames=4,
        chunk_max_stale_ms=2000,
        auto_dispatch=False,
        emergency_on_failure=True,
        watchdog_period_ms=10,
        logger=logging.getLogger("test_closed_loop"),
    )
    kwargs.update(overrides)
    runner = ClosedLoopRunner(**kwargs)
    # Replace the real undistort with an identity passthrough so tests don't
    # depend on opencv calibration quality; we still call it once per loop.
    runner._build_image_dict = lambda hf, wf: {
        runner.stereo_pair["left"]: hf.bgr, runner.stereo_pair["right"]: wf.bgr,
    }
    return runner


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------
def test_chunk_len_mismatch_raises(tmp_path: Path) -> None:
    cfg_path = _write_train_config(tmp_path, num_frames=33)
    with pytest.raises(RuntimeError, match="chunk_len"):
        ClosedLoopRunner(
            model_client=_make_mock_model_client(action_horizon=CHUNK_LEN),
            arm_client=_make_mock_arm_client(),
            ws_ingester=_make_mock_ws_ingester(),
            train_config_path=cfg_path,
            default_camera_info=_make_default_camera_info(),
            stereo_pair={"left": "head_left", "right": "right_wrist_left"},
            chunk_len=10,  # != 32
        )


def test_action_horizon_mismatch_raises(tmp_path: Path) -> None:
    cfg_path = _write_train_config(tmp_path)
    with pytest.raises(RuntimeError, match="action_horizon"):
        ClosedLoopRunner(
            model_client=_make_mock_model_client(action_horizon=999),
            arm_client=_make_mock_arm_client(),
            ws_ingester=_make_mock_ws_ingester(),
            train_config_path=cfg_path,
            default_camera_info=_make_default_camera_info(),
            stereo_pair={"left": "head_left", "right": "right_wrist_left"},
        )


def test_action_dim_must_be_7(tmp_path: Path) -> None:
    cfg_path = _write_train_config(tmp_path, action_output_dim=6)
    with pytest.raises(RuntimeError, match="7-DoF"):
        ClosedLoopRunner(
            model_client=_make_mock_model_client(),
            arm_client=_make_mock_arm_client(),
            ws_ingester=_make_mock_ws_ingester(),
            train_config_path=cfg_path,
            default_camera_info=_make_default_camera_info(),
            stereo_pair={"left": "head_left", "right": "right_wrist_left"},
        )


def test_image_pipeline_must_be_known(tmp_path: Path) -> None:
    cfg_path = _write_train_config(tmp_path)
    with pytest.raises(ValueError):
        ClosedLoopRunner(
            model_client=_make_mock_model_client(),
            arm_client=_make_mock_arm_client(),
            ws_ingester=_make_mock_ws_ingester(),
            train_config_path=cfg_path,
            default_camera_info=_make_default_camera_info(),
            stereo_pair={"left": "head_left", "right": "right_wrist_left"},
            image_pipeline="bogus",
        )


# ---------------------------------------------------------------------------
# InferLoop happy / skip paths
# ---------------------------------------------------------------------------
def test_infer_normal_path_pushes_one_chunk(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    runner._instruction = "open the door"
    runner._infer_once()

    assert runner.ringbuffer.occupancy() == 1
    entry = runner.ringbuffer.latest()
    assert entry is not None
    assert entry.action_abs.shape == (CHUNK_LEN, ACTION_DIM)
    assert entry.chunk_id == 1
    assert entry.base_capture_ts_ns == 1_000_000_000  # head_frame capture ts
    assert entry.step_dt_ns == 50_000_000
    # model_input observed by mock
    runner.model_client.infer.assert_called_once()
    sent = runner.model_client.infer.call_args[0][0]
    assert set(sent.keys()) >= {"images", "proprio_raw", "current_position", "instruction"}
    assert sent["instruction"] == "open the door"
    assert sent["proprio_raw"].shape == (7,)
    assert sent["current_position"].shape == (6,)


def test_snapshot_call_order_ws_before_arm(tmp_path: Path) -> None:
    """Design doc §7.6.1: WS.latest() must run before arm.latest()."""
    runner = _make_runner(tmp_path)
    call_log: list[str] = []

    def _ws_latest(key):
        call_log.append(f"ws:{key}")
        return runner.ws_ingester._frames[key]

    def _arm_latest():
        call_log.append("arm")
        return runner.arm_client._state

    runner.ws_ingester.latest.side_effect = _ws_latest
    runner.arm_client.latest.side_effect = _arm_latest
    runner._infer_once()

    # ws calls must precede the arm call
    ws_indices = [i for i, e in enumerate(call_log) if e.startswith("ws:")]
    arm_indices = [i for i, e in enumerate(call_log) if e == "arm"]
    assert ws_indices and arm_indices
    assert max(ws_indices) < min(arm_indices)


def test_frame_stale_records_skip(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    # ws_ingester.latest already enforces TTL by returning None when stale.
    runner.ws_ingester._frames["head_left"] = None
    runner.ws_ingester.latest.side_effect = lambda key: runner.ws_ingester._frames.get(key)
    runner._infer_once()
    assert runner.ringbuffer.occupancy() == 0
    assert (runner._last_skip_reason or "").startswith("stale_frame:head_left")
    runner.model_client.infer.assert_not_called()


def test_wrist_frame_stale_records_skip(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    runner.ws_ingester._frames["right_wrist_left"] = None
    runner.ws_ingester.latest.side_effect = lambda key: runner.ws_ingester._frames.get(key)
    runner._infer_once()
    assert runner.ringbuffer.occupancy() == 0
    assert (runner._last_skip_reason or "").startswith("stale_frame:right_wrist_left")


def test_state_stale_records_skip(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    runner.arm_client._state = None
    runner.arm_client.latest.side_effect = lambda: None
    runner._infer_once()
    assert runner.ringbuffer.occupancy() == 0
    assert runner._last_skip_reason == "stale_arm_state"


def test_consecutive_pushes_evict_oldest(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    for _ in range(5):
        runner._infer_once()
    # capacity=2 -> only the two newest chunks remain
    assert runner.ringbuffer.occupancy() == 2
    assert runner.ringbuffer.max_occupancy_seen() == 2
    latest = runner.ringbuffer.latest()
    prev = runner.ringbuffer.prev()
    assert latest.chunk_id == 5
    assert prev.chunk_id == 4


def test_action_shape_mismatch_raises(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    runner.model_client.infer.side_effect = lambda mi: {
        "actions": np.zeros((10, ACTION_DIM), dtype=np.float32)
    }
    with pytest.raises(RuntimeError, match="action shape mismatch"):
        runner._infer_once()
    assert runner.ringbuffer.occupancy() == 0


def test_model_exception_records_skip(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    runner.model_client.infer.side_effect = RuntimeError("kaboom")
    runner._infer_once()
    assert runner.ringbuffer.occupancy() == 0
    assert (runner._last_skip_reason or "").startswith("model_error:")
    assert runner._consecutive_infer_fail == 1


# ---------------------------------------------------------------------------
# status()
# ---------------------------------------------------------------------------
def test_status_shape(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    runner._infer_once()
    s = runner.status()
    required = {
        "running", "current_chunk_id", "prev_chunk_id",
        "ringbuffer_occupancy", "ringbuffer_max_occupancy_seen",
        "infer_count", "infer_latency_ms_p50", "infer_latency_ms_p95",
        "infer_latency_ms_p99", "last_skip_reason", "consecutive_infer_fail",
        "hold_mode", "instruction", "chunk_len", "num_inference_steps",
        "image_pipeline", "auto_dispatch", "dispatch", "watchdog",
    }
    missing = required - set(s.keys())
    assert not missing, f"status missing keys: {missing}"
    assert s["chunk_len"] == CHUNK_LEN
    assert s["num_inference_steps"] == 10
    assert s["ringbuffer_occupancy"] == 1
    assert s["current_chunk_id"] == 1
    assert s["prev_chunk_id"] is None  # only one chunk so far
    assert s["infer_count"] == 1
    assert isinstance(s["infer_latency_ms_p50"], float)
    assert s["hold_mode"] is False
    assert s["auto_dispatch"] is False
    assert s["image_pipeline"] == "raw_native"
    assert s["dispatch"]["kind"] == "noop"
    assert s["watchdog"]["kind"] == "noop"


def test_status_empty_history(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    s = runner.status()
    assert s["infer_count"] == 0
    assert s["infer_latency_ms_p50"] is None
    assert s["ringbuffer_occupancy"] == 0


# ---------------------------------------------------------------------------
# lifecycle (real thread, brief)
# ---------------------------------------------------------------------------
def test_start_stop_runs_infer_thread(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path, infer_period_ms=20)  # very fast for test
    runner.start(instruction="hello")
    try:
        threading.Event().wait(0.12)
    finally:
        runner.stop()
    assert runner.model_client.infer.call_count >= 1
    assert runner.ringbuffer.occupancy() >= 1


def test_double_start_raises(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path, infer_period_ms=20)
    runner.start()
    try:
        with pytest.raises(RuntimeError, match="called twice"):
            runner.start()
    finally:
        runner.stop()


def test_noop_dispatch_records_kwargs(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path, blend_frames=7)
    assert isinstance(runner._dispatch, NoOpDispatch)
    assert isinstance(runner._watchdog, NoOpWatchdog)
    assert runner._dispatch.blend_frames == 7
    assert runner._dispatch.status()["running"] is False
    runner._dispatch.start()
    assert runner._dispatch.status()["running"] is True


def test_custom_dispatcher_factory_used(tmp_path: Path) -> None:
    captured: dict = {}

    class _RecorderDispatch(NoOpDispatch):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    runner = _make_runner(tmp_path, dispatcher_factory=_RecorderDispatch)
    assert isinstance(runner._dispatch, _RecorderDispatch)
    assert captured["send_period_ms"] == 50
    assert captured["blend_frames"] == 4
