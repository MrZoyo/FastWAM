#!/usr/bin/env python
"""FastWAM active-loop HTTP server (v2 self-fetch mode).

Pulls RGB frames from rgbd_ws_bridge and ARM state from arm_sdk on its own,
runs FastWAM @ 2.5 Hz, and dispatches 20 Hz pose chunks. Companion to
``scripts/fastwam_http_server.py`` (v1 passive, port 8117); this script
listens on port 8118.

Endpoints: POST /start /stop /emergency /debug/zero_pose_test ;
GET /health /closed_loop_status /ws_status. See
``docs/fastwam_http_server_self_fetch_design.md`` §5/§7.7/§7.7.1/§8/§9/§11.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.closed_loop_eval.model_clients import FastWAMModelClient
from fastwam.server.arm_client import ArmClient
from fastwam.server.rotation import quat_xyzw_to_rpy
from fastwam.server.ws_ingest import WSFrameIngester

try:
    from fastwam.server.closed_loop import ClosedLoopRunner, make_production_factories  # type: ignore
    _CLOSED_LOOP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only before PR5
    _CLOSED_LOOP_AVAILABLE = False

    class ClosedLoopRunner:  # type: ignore[no-redef]
        def __init__(self, *_a: Any, **_k: Any) -> None:
            raise RuntimeError(
                "fastwam.server.closed_loop.ClosedLoopRunner unavailable "
                "(PR5 not merged). Tests must inject a mock runner."
            )


logger = logging.getLogger("fastwam.active_loop_server")
DEFAULT_INSTRUCTION = "open the door"
EXPECTED_WS_CHANNELS = {"head_left", "right_wrist_left"}
BANNER = "[startup]"

# All CLI flags from design §5. Kept as a data table so the function body
# stays small. (See test_active_loop_server.test_argparser_has_all_v2_flags.)
_FLAGS: list[tuple[str, dict[str, Any]]] = [
    ("--host", {"default": "0.0.0.0"}),
    ("--port", {"type": int, "default": 8118}),
    ("--config", {"default": "configs/task/real_1048_uncond_2cam224_1e-4.yaml"}),
    ("--checkpoint", {"default": "runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/checkpoints/step_020000.pt"}),
    ("--dataset-stats", {"default": "runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/dataset_stats.json"}),
    ("--text-cache-dir", {"default": "data/text_embeds_cache/real_1048"}),
    ("--default-camera-info", {"default": "configs/camera_info/real_1048_default.json"}),
    ("--ws-url", {"default": "ws://192.168.31.66:19095"}),
    ("--ws-frame-max-age-ms", {"type": int, "default": 250}),
    ("--ws-reconnect-backoff-ms", {"default": "500,1000,2000,5000,10000"}),
    ("--ws-startup-timeout-ms", {"type": int, "default": 30_000}),
    ("--ws-warn-stale-ms", {"type": int, "default": 200}),
    ("--ws-hold-stale-ms", {"type": int, "default": 500}),
    ("--ws-estop-stale-ms", {"type": int, "default": 1500}),
    ("--arm-host", {"default": "192.168.31.34"}),
    ("--arm-port", {"type": int, "default": 50051}),
    ("--arm-poll-hz", {"type": float, "default": 50.0}),
    ("--arm-state-max-age-ms", {"type": float, "default": 100.0}),
    ("--arm-lease-ms", {"type": int, "default": 15_000}),
    ("--arm-acquire-on", {"choices": ("start", "init"), "default": "start"}),
    ("--infer-period-ms", {"type": int, "default": 400}),
    ("--send-period-ms", {"type": int, "default": 50}),
    ("--blend-frames", {"type": int, "default": 4}),
    ("--chunk-len", {"type": int, "default": None}),
    ("--num-inference-steps", {"type": int, "default": None}),
    ("--chunk-max-stale-ms", {"type": int, "default": 2000}),
    ("--auto-dispatch", {"action": "store_true", "default": False}),
    ("--emergency-on-failure", {"action": "store_true", "default": True}),
    ("--watchdog-period-ms", {"type": int, "default": 10}),
    ("--instruction", {"default": DEFAULT_INSTRUCTION}),
    ("--device", {"default": "cuda:1"}),
    ("--require-gpu-mem-free-gb", {"type": float, "default": 8.0}),
    ("--image-pipeline", {"choices": ("raw_native", "lerobot_480x640"), "default": "raw_native"}),
    ("--undistort", {"action": "store_true", "default": True}),
    ("--no-undistort", {"dest": "undistort", "action": "store_false"}),
    ("--log-level", {"default": "INFO"}),
    ("--warmup-infer-calls", {"type": int, "default": 5}),
    ("--benchmark-infer-calls", {"type": int, "default": 10}),
    ("--benchmark-p50-budget-ms", {"type": float, "default": 500.0}),
    ("--skip-warmup", {"action": "store_true", "default": False}),
]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    for flag, kw in _FLAGS:
        p.add_argument(flag, **kw)
    return p


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------

def setup_logging(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _resolve_train_config_path(checkpoint: str) -> Path:
    return Path(checkpoint).expanduser().parent.parent / "config.yaml"


def check_gpu_free_mem(device: str, require_gb: float) -> None:
    if not device.startswith("cuda"):
        logger.warning("device=%s not CUDA; skipping GPU mem check", device)
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        logger.error("torch unavailable: %s", exc); sys.exit(1)
    if not torch.cuda.is_available():
        logger.error("CUDA unavailable but --device=%s", device); sys.exit(1)
    try:
        idx = 0 if ":" not in device else int(device.split(":", 1)[1])
        free_bytes, _ = torch.cuda.mem_get_info(idx)
    except Exception as exc:
        logger.error("torch.cuda.mem_get_info(%s) failed: %s", device, exc); sys.exit(1)
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < require_gb:
        logger.error("%s GPU free %.2f GiB < required %.2f GiB on %s; abort.",
                     BANNER, free_gb, require_gb, device)
        sys.exit(1)
    logger.info("%s gpu_free_mem_gb      = %.2f      (require >= %.2f)",
                BANNER, free_gb, require_gb)


def init_model_client(args: argparse.Namespace) -> FastWAMModelClient:
    kw: dict[str, Any] = dict(
        config_path=args.config, checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats, text_cache_dir=args.text_cache_dir,
        instruction=args.instruction, device=args.device, preload_text_context=True,
    )
    if args.chunk_len is not None:
        kw["action_horizon"] = int(args.chunk_len)
    if args.num_inference_steps is not None:
        kw["num_inference_steps"] = int(args.num_inference_steps)
    logger.info("loading FastWAMModelClient checkpoint=%s", args.checkpoint)
    return FastWAMModelClient(**kw)


def run_rotation_fingerprint_or_exit() -> None:
    fixture = (Path(__file__).resolve().parents[1]
               / "tests" / "fixtures" / "rotation_fingerprint.json")
    if not fixture.exists():
        logger.error("%s fingerprint fixture missing at %s", BANNER, fixture); sys.exit(1)
    try:
        samples = json.loads(fixture.read_text())["samples"]
        for s in samples:
            got = quat_xyzw_to_rpy(np.array(s["quat_xyzw"], dtype=np.float64)).astype(np.float64)
            expected = np.array(s["expected_rpy"], dtype=np.float64)
            err = float(np.max(np.abs(got - expected)))
            if err >= 1e-5:
                raise RuntimeError(f"sample {s['label']!r} max abs err {err:.3e}")
    except Exception as exc:
        logger.error("%s rotation fingerprint test FAILED: %s", BANNER, exc); sys.exit(1)
    logger.info("%s rotation fingerprint test (%d GT samples from opendoor_real_1048): PASS",
                BANNER, len(samples))


def _build_dummy_infer_input(model: FastWAMModelClient) -> dict[str, Any]:
    images: dict[str, np.ndarray] = {}
    for key, shape in model.image_shapes.items():
        if len(shape) == 3:
            _, h, w = (int(v) for v in shape)
        elif len(shape) == 2:
            h, w = (int(v) for v in shape)
        else:
            h, w = 224, 448
        images[str(key)] = np.zeros((h, w, 3), dtype=np.uint8)
    return {"images": images,
            "proprio_raw": np.zeros((model.proprio_dim,), dtype=np.float32),
            "current_position": np.zeros((6,), dtype=np.float32),
            "instruction": model.instruction}


def warmup_benchmark_or_exit(model: FastWAMModelClient, *, warmup_calls: int,
                             benchmark_calls: int, p50_budget_ms: float) -> dict[str, float]:
    sample = _build_dummy_infer_input(model)
    last_warmup_ms = 0.0
    for _ in range(max(1, warmup_calls)):
        t0 = time.perf_counter()
        model.infer(sample)
        last_warmup_ms = (time.perf_counter() - t0) * 1e3
    logger.info("%s warmup infer (%d calls) ... last latency = %.1f ms",
                BANNER, warmup_calls, last_warmup_ms)
    times_ms = []
    for _ in range(max(1, benchmark_calls)):
        t0 = time.perf_counter()
        model.infer(sample)
        times_ms.append((time.perf_counter() - t0) * 1e3)
    arr = np.asarray(times_ms, dtype=np.float64)
    p50, p95, p99 = (float(np.percentile(arr, q)) for q in (50, 95, 99))
    logger.info("%s benchmark infer (%d calls): p50=%.1fms p95=%.1fms p99=%.1fms",
                BANNER, benchmark_calls, p50, p95, p99)
    if p50 > p50_budget_ms:
        logger.error("%s benchmark p50 %.1f ms > budget %.1f ms; abort.",
                     BANNER, p50, p50_budget_ms)
        sys.exit(1)
    return {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "last_warmup_ms": last_warmup_ms}


def print_startup_banner(args: argparse.Namespace,
                         model: FastWAMModelClient | None = None) -> None:
    """Print banner rows from design §7.7.1. ``model=None`` prints part 1 only."""
    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:
        scipy_v = "<unknown>"
    rows: list[tuple[str, str]] = [
        ("script              ", "fastwam_active_loop_server.py v2"),
        ("train_config_path   ", str(_resolve_train_config_path(args.checkpoint))),
    ]
    if model is not None:
        nframes = int(model.cfg.get("data", {}).get("train", {}).get("num_frames", 33))
        clen = args.chunk_len if args.chunk_len is not None else (nframes - 1)
        nsteps = (args.num_inference_steps if args.num_inference_steps is not None
                  else int(model.num_inference_steps))
        rows += [
            ("train.num_frames    ", str(nframes)),
            ("chunk_len           ",
             f"{clen}        (= num_frames - 1, frame-aligned backward delta)"),
            ("action_output_dim   ", "7         (xyz + rpy + gripper)"),
            ("num_inference_steps ", f"{nsteps}        (from eval_num_inference_steps)"),
        ]
    rows += [
        ("device              ", args.device),
        ("ws_url              ", args.ws_url),
        ("ws_channels         ", ", ".join(sorted(EXPECTED_WS_CHANNELS)) + "  (identity mapping)"),
        ("arm_host:port       ", f"{args.arm_host}:{args.arm_port}"),
        ("arm_lease_ms        ", f"{args.arm_lease_ms}     (SDK auto-renew @ 5s)"),
        ("infer_period_ms     ",
         f"{args.infer_period_ms}       ({1000.0 / max(1, args.infer_period_ms):.1f} Hz)"),
        ("send_period_ms      ",
         f"{args.send_period_ms}        ({1000.0 / max(1, args.send_period_ms):.1f} Hz)"),
        ("blend_frames        ", str(args.blend_frames)),
        ("image_pipeline      ",
         f"{args.image_pipeline} (1088x1280 -> undistort -> model_clients)"),
        ("scipy_version       ", scipy_v),
    ]
    for label, value in rows:
        logger.info("%s %s = %s", BANNER, label, value)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def make_handler(arm: ArmClient, ws: WSFrameIngester, runner: Any) -> type[BaseHTTPRequestHandler]:
    op_lock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        server_version = "FastWAMActiveLoop/0.1"

        def log_message(self, fmt: str, *a: Any) -> None:
            logger.info("%s - %s", self.client_address[0], fmt % a)

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(_to_jsonable(payload), ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0:
                return {}
            raw = self.rfile.read(n)
            return json.loads(raw.decode("utf-8")) if raw else {}

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._json(HTTPStatus.OK, {"status": "ok",
                                           "server": "fastwam_active_loop_server",
                                           "arm": arm.health(), "ws": ws.health()})
            elif self.path == "/closed_loop_status":
                self._json(HTTPStatus.OK, runner.status())
            elif self.path == "/ws_status":
                self._json(HTTPStatus.OK, ws.health())
            else:
                self._json(HTTPStatus.NOT_FOUND,
                           {"error": f"Unknown endpoint: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read() or {}
                if self.path == "/start":
                    inst = payload.get("instruction")
                    if inst is not None and not isinstance(inst, str):
                        return self._json(HTTPStatus.BAD_REQUEST,
                                          {"error": "instruction must be a string"})
                    with op_lock:
                        # PR9 R9: refuse to start when lease + servo_control + speed setup fails;
                        # otherwise InferLoop runs but every send_pose returns False -> repeated estop.
                        if not arm.acquire_control():
                            return self._json(HTTPStatus.SERVICE_UNAVAILABLE,
                                              {"error": "arm.acquire_control failed (lease held by another client, or switch_controller failed)"})
                        runner.start(inst)
                    return self._json(HTTPStatus.OK,
                                      {"status": "started", "instruction": inst})
                if self.path == "/stop":
                    with op_lock:
                        runner.stop()
                        arm.release_control()
                    return self._json(HTTPStatus.OK, {"status": "stopped"})
                if self.path == "/emergency":
                    enable = payload.get("enable", True)
                    if not isinstance(enable, bool):
                        return self._json(HTTPStatus.BAD_REQUEST,
                                          {"error": "enable must be bool"})
                    with op_lock:
                        # PR9 R3: stop dispatcher synchronously BEFORE the 150 ms
                        # set_arm_emergency_stop SDK blocking call; otherwise dispatch
                        # could fire 2-3 more 50 ms slots inside the estop window.
                        if enable:
                            try:
                                runner._dispatch.set_hold(True, reason="emergency")
                            except Exception:
                                logger.exception("[emergency] dispatch.set_hold failed")
                        arm.emergency_stop(enable)
                    return self._json(HTTPStatus.OK,
                                      {"status": "emergency_stop_set", "enable": enable})
                if self.path == "/debug/zero_pose_test":
                    return self._zero_pose_test(payload)
                self._json(HTTPStatus.NOT_FOUND,
                           {"error": f"Unknown endpoint: {self.path}"})
            except Exception as exc:  # noqa: BLE001
                logger.error("handler %s failed: %s\n%s",
                             self.path, exc, traceback.format_exc())
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR,
                           {"error": f"{type(exc).__name__}: {exc}"})

        def _zero_pose_test(self, payload: dict[str, Any]) -> None:
            duration_s = float(payload.get("duration_s", 5.0))
            if duration_s <= 0 or duration_s > 60:
                return self._json(HTTPStatus.BAD_REQUEST,
                                  {"error": "duration_s must be in (0, 60]"})
            snap = arm.latest()
            if snap is None:
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE,
                                  {"error": "no arm state yet"})
            chunk_len = 32
            pose_row = np.concatenate([
                np.asarray(snap.eef_xyz, dtype=np.float32),
                np.asarray(snap.eef_rpy, dtype=np.float32),
                np.asarray([snap.gripper_m], dtype=np.float32),
            ], axis=0)
            action_abs = np.tile(pose_row[None, :], (chunk_len, 1))
            base_ts_ns = int(snap.capture_ts_ns)
            with op_lock:
                # PR9 R4: bypass the model. Starting runner.start() would let
                # InferLoop push a real chunk ~400 ms later and overwrite the
                # zero-pose entry. Only start the dispatcher; no InferLoop, no
                # Watchdog. arm.acquire_control() is required for send_pose
                # to take effect on hardware.
                if not arm.acquire_control():
                    return self._json(HTTPStatus.SERVICE_UNAVAILABLE,
                                      {"error": "arm.acquire_control failed; cannot run zero-pose dry-test"})
                try:
                    runner.inject_chunk_for_debug(action_abs, base_ts_ns)
                    runner.start_dispatch_only()
                    time.sleep(duration_s)
                finally:
                    try: runner.stop_dispatch_only()
                    except Exception: logger.exception("[zero_pose] dispatch.stop raised")
                    try: arm.release_control()
                    except Exception: logger.exception("[zero_pose] arm.release_control raised")
            self._json(HTTPStatus.OK, {"status": "zero_pose_completed",
                                       "chunk_len": chunk_len,
                                       "duration_s": duration_s,
                                       "base_ts_ns": base_ts_ns})

    return H


# ---------------------------------------------------------------------------
# bootstrap helpers
# ---------------------------------------------------------------------------

def _parse_backoff_ms(text: str) -> list[int]:
    out = [int(c.strip()) for c in text.split(",") if c.strip()]
    return out or [1000]


def _load_default_camera_info(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"default camera-info not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    cameras = data.get("cameras") if isinstance(data, dict) else None
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"default camera-info {path} missing 'cameras'")
    pair = data.get("stereo_pair") if isinstance(data, dict) else None
    if (not isinstance(pair, dict) or not isinstance(pair.get("left"), str)
            or not isinstance(pair.get("right"), str)):
        raise ValueError(f"default camera-info {path} missing stereo_pair")
    return ({str(k): v for k, v in cameras.items() if isinstance(v, dict)},
            {"left": pair["left"], "right": pair["right"]})


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    setup_logging(args)

    print_startup_banner(args, model=None)
    check_gpu_free_mem(args.device, args.require_gpu_mem_free_gb)
    model = init_model_client(args)
    run_rotation_fingerprint_or_exit()
    print_startup_banner(args, model=model)

    repo_root = Path(__file__).resolve().parents[1]
    cam_path = Path(args.default_camera_info).expanduser()
    if not cam_path.is_absolute():
        cam_path = (repo_root / cam_path).resolve()
    default_camera_info, stereo_pair = _load_default_camera_info(cam_path)

    arm = ArmClient(host=args.arm_host, port=args.arm_port,
                    poll_hz=args.arm_poll_hz, state_max_age_ms=args.arm_state_max_age_ms,
                    lease_ms=args.arm_lease_ms)
    arm.start()
    if args.arm_acquire_on == "init" and not arm.acquire_control():
        logger.error("%s arm.acquire_control() returned False; abort.", BANNER)
        sys.exit(1)

    ws = WSFrameIngester(
        ws_url=args.ws_url, expected_channels=set(EXPECTED_WS_CHANNELS),
        frame_max_age_ms=args.ws_frame_max_age_ms,
        reconnect_backoff_ms_list=_parse_backoff_ms(args.ws_reconnect_backoff_ms),
        startup_timeout_ms=args.ws_startup_timeout_ms,
    )
    try:
        ws.start()
    except Exception as exc:
        logger.error("%s WSFrameIngester.start() failed: %s", BANNER, exc)
        arm.stop()
        sys.exit(1)

    runner = ClosedLoopRunner(
        model_client=model, arm_client=arm, ws_ingester=ws,
        train_config_path=str(_resolve_train_config_path(args.checkpoint)),
        default_camera_info=default_camera_info, stereo_pair=stereo_pair,
        infer_period_ms=args.infer_period_ms, send_period_ms=args.send_period_ms,
        blend_frames=args.blend_frames, chunk_max_stale_ms=args.chunk_max_stale_ms,
        auto_dispatch=args.auto_dispatch,
        emergency_on_failure=args.emergency_on_failure,
        watchdog_period_ms=args.watchdog_period_ms,
        image_pipeline=args.image_pipeline, undistort=args.undistort,
        ws_warn_stale_ms=args.ws_warn_stale_ms,
        ws_hold_stale_ms=args.ws_hold_stale_ms,
        ws_estop_stale_ms=args.ws_estop_stale_ms,
        default_instruction=args.instruction,
        dispatcher_factory=make_production_factories()[0],
        watchdog_factory=make_production_factories()[1],
    )

    if not args.skip_warmup:
        warmup_benchmark_or_exit(
            model,
            warmup_calls=args.warmup_infer_calls,
            benchmark_calls=args.benchmark_infer_calls,
            p50_budget_ms=args.benchmark_p50_budget_ms,
        )

    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(arm, ws, runner))
    logger.info("FastWAM active-loop server ready at http://%s:%d  (auto_dispatch=%s)",
                args.host, args.port, args.auto_dispatch)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping FastWAM active-loop server.")
    finally:
        for fn, label in (
            (runner.stop, "runner.stop"),
            ((lambda: arm.emergency_stop(True)) if args.emergency_on_failure
             else (lambda: None), "arm.emergency_stop(True)"),
            (ws.stop, "ws.stop"),
            (arm.stop, "arm.stop"),
        ):
            try:
                fn()
            except Exception:
                logger.exception("%s in shutdown failed", label)
        httpd.server_close()


if __name__ == "__main__":
    main()
