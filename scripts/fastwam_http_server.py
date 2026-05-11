#!/usr/bin/env python
"""HTTP service for FastWAM action inference.

Start from the repo root:

  .venv/bin/python -B scripts/fastwam_http_server.py \
    --config configs/task/real_1048_uncond_2cam224_1e-4.yaml \
    --checkpoint runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/checkpoints/weights/step_020000.pt \
    --dataset-stats runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/dataset_stats.json \
    --text-cache-dir data/text_embeds_cache/real_1048

POST /infer expects JSON:

  {
    "instruction": "open the door",
    "images": {"head_left": "<base64 png/jpeg>", "right_wrist_left": "<base64 png/jpeg>"},
    "proprio_raw": [q0, q1, q2, q3, q4, q5, gripper]
  }
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.closed_loop_eval.model_clients import FastWAMModelClient


logger = logging.getLogger(__name__)

_CLIENT: FastWAMModelClient | None = None
_INFER_LOCK = threading.Lock()
_SERVER_INFO: dict[str, Any] = {}


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


def _decode_image(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("base64") or value.get("image_base64")
    if not isinstance(value, str):
        raise ValueError("Image values must be base64 strings or {'base64': ...} objects.")
    if "," in value and value.split(",", 1)[0].startswith("data:"):
        value = value.split(",", 1)[1]
    try:
        raw = base64.b64decode(value, validate=True)
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid base64 image: {exc}") from exc


def _require_array(payload: dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"Missing required field '{key}'.")
    arr = np.asarray(payload[key], dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Field '{key}' must not be empty.")
    return arr


def _request_to_model_input(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    images_payload = payload.get("images")
    if not isinstance(images_payload, dict) or not images_payload:
        raise ValueError("Field 'images' must be a non-empty object.")
    if _CLIENT is not None:
        expected_keys = set(_CLIENT.image_shapes.keys())
        missing = sorted(expected_keys - set(str(key) for key in images_payload.keys()))
        if missing:
            raise ValueError(f"Missing required image keys: {missing}.")

    images = {str(key): _decode_image(value) for key, value in images_payload.items()}
    proprio_raw = _require_array(payload, "proprio_raw")
    if _CLIENT is not None and proprio_raw.size != _CLIENT.proprio_dim:
        raise ValueError(f"Field 'proprio_raw' must have length {_CLIENT.proprio_dim}, got {proprio_raw.size}.")
    model_input = {
        "images": images,
        "proprio_raw": proprio_raw,
    }
    for key in ("current_position", "joint_position", "cartesian_position"):
        if key in payload and payload[key] is not None:
            model_input["current_position"] = _require_array(payload, key)
            if model_input["current_position"].size < 6:
                raise ValueError(f"Field '{key}' must contain at least 6 values.")
            break
    if "current_position" not in model_input:
        if proprio_raw.size < 6:
            raise ValueError("Field 'proprio_raw' must contain at least 6 values.")
        model_input["current_position"] = proprio_raw[:6]
    if "instruction" in payload and payload["instruction"] is not None:
        model_input["instruction"] = str(payload["instruction"])
    return model_input


def _schema() -> dict[str, Any]:
    return {
        "endpoints": {
            "GET /health": "server/model metadata",
            "GET /schema": "request and response schema",
            "POST /infer": "run FastWAM and return an action chunk",
            "POST /predict_action": "alias of /infer",
        },
        "request": {
            "instruction": "optional string; must have a matching text cache when text encoder is not loaded",
            "images": "object mapping training camera keys to base64 encoded PNG/JPEG RGB images",
            "proprio_raw": "raw 7D real_1048 proprio vector [joint0..joint5, gripper]",
            "current_position": "optional 6D base position for delta accumulation; defaults to proprio_raw[:6]",
        },
        "response": {
            "action_format": "joint_absolute",
            "actions": "[T,7] absolute action chunk",
            "action_mode": "delta6_abs_gripper",
            "normalized_action_shape": "[T,D] model output shape before denormalization",
            "full_prompt": "prompt used to load the text embedding cache",
        },
        "example": {
            "instruction": "open the door",
            "images": {"head_left": "<base64>", "right_wrist_left": "<base64>"},
            "proprio_raw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
        },
    }


class FastWAMHandler(BaseHTTPRequestHandler):
    server_version = "FastWAMHTTP/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.client_address[0], fmt % args)

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(_to_jsonable(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Request body is empty.")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok", **_SERVER_INFO})
            return
        if self.path == "/schema":
            self._send_json(HTTPStatus.OK, _schema())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/infer", "/predict_action"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {self.path}"})
            return
        if _CLIENT is None:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "Model is not initialized."})
            return
        try:
            payload = self._read_json()
            model_input = _request_to_model_input(payload)
        except Exception as exc:  # noqa: BLE001
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        try:
            with _INFER_LOCK:
                result = _CLIENT.infer(model_input)
            self._send_json(HTTPStatus.OK, result)
        except Exception as exc:  # noqa: BLE001
            logger.error("FastWAM inference failed: %s\n%s", exc, traceback.format_exc())
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})


def _init_client(args: argparse.Namespace) -> FastWAMModelClient:
    client = FastWAMModelClient(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        text_cache_dir=args.text_cache_dir,
        instruction=args.instruction,
        action_horizon=args.action_horizon,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        rand_device=args.rand_device,
        preload_text_context=False,
        output_action_format=args.output_action_format,
    )
    return client


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8117)
    parser.add_argument("--config", default="configs/task/real_1048_uncond_2cam224_1e-4.yaml")
    parser.add_argument(
        "--checkpoint",
        default="runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/checkpoints/weights/step_020000.pt",
    )
    parser.add_argument(
        "--dataset-stats",
        default="runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/dataset_stats.json",
    )
    parser.add_argument("--text-cache-dir", default="data/text_embeds_cache/real_1048")
    parser.add_argument("--instruction", default="open the door")
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--output-action-format", default="joint_absolute")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    global _CLIENT, _SERVER_INFO

    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger.info("Loading FastWAM model from %s", args.checkpoint)
    _CLIENT = _init_client(args)
    _SERVER_INFO = {
        "model_loaded": True,
        "config": str(Path(args.config).expanduser()),
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "dataset_stats": str(Path(args.dataset_stats).expanduser()),
        "text_cache_dir": str(Path(args.text_cache_dir).expanduser()),
        "image_keys": list(_CLIENT.image_shapes.keys()),
        "image_shapes": {k: list(v) for k, v in _CLIENT.image_shapes.items()},
        "video_size": list(_CLIENT.video_size),
        "proprio_dim": _CLIENT.proprio_dim,
        "action_horizon": _CLIENT.action_horizon,
        "num_inference_steps": _CLIENT.num_inference_steps,
        "device": _CLIENT.device,
    }

    httpd = ThreadingHTTPServer((args.host, args.port), FastWAMHandler)
    logger.info("FastWAM HTTP server ready at http://%s:%d", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping FastWAM HTTP server.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
