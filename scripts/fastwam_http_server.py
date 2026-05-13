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

Images must be either (H=1088, W=1280) raw frames (the payload then MUST
include an "undistort" object with left_camera_info / right_camera_info, and
the server undistorts at native resolution before resizing each camera to
480x640 with cv2.INTER_AREA) or (H=480, W=640) frames already aligned with
training (no undistort, no resize). All camera streams in a single request
must share the same resolution; any other resolution returns 400.
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
from fastwam.utils.rgb_undistort import (
    _require_cv2,
    opencv_available,
    undistort_stereo_side_from_camera_info,
)


logger = logging.getLogger(__name__)

_CLIENT: FastWAMModelClient | None = None
_INFER_LOCK = threading.Lock()
_SERVER_INFO: dict[str, Any] = {}

_NATIVE_SHAPE: tuple[int, int] = (1088, 1280)  # (H, W) raw stereo frames
_TRAIN_SHAPE: tuple[int, int] = (480, 640)     # (H, W) training-aligned frames


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


def _image_hw(arr: np.ndarray) -> tuple[int, int]:
    if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(
            f"Each image must be H x W x 3 uint8; got shape={getattr(arr, 'shape', None)} "
            f"dtype={getattr(arr, 'dtype', None)}."
        )
    return int(arr.shape[0]), int(arr.shape[1])


def _stereo_image_keys(options: dict[str, Any]) -> tuple[str, str]:
    left_key = options.get("left_image_key") or options.get("left_key")
    right_key = options.get("right_image_key") or options.get("right_key")
    if left_key is not None and right_key is not None:
        return str(left_key), str(right_key)
    if _CLIENT is None:
        raise RuntimeError("Model is not initialized.")
    keys = list(_CLIENT.image_shapes.keys())
    if len(keys) != 2:
        raise ValueError(
            "undistort requires left_image_key/right_image_key when the model uses more than 2 images."
        )
    return keys[0], keys[1]


def _normalize_image_resolution(
    *,
    images: dict[str, np.ndarray],
    payload: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Branch on the actual (H, W) of incoming images.

    - 1088x1280: require stereo calibration, undistort at native resolution, then
      cv2.resize each camera down to 480x640 with INTER_AREA.
    - 480x640:   pass through; any 'undistort' field in the payload is ignored.
    - other:     raise ValueError -> the handler turns it into a 400.
    """
    shapes = {k: _image_hw(v) for k, v in images.items()}
    unique = set(shapes.values())
    if len(unique) != 1:
        raise ValueError(f"All images must share the same (H, W); got {shapes}.")
    hw = next(iter(unique))

    if hw == _TRAIN_SHAPE:
        return images

    if hw == _NATIVE_SHAPE:
        options = payload.get("undistort")
        if not isinstance(options, dict):
            raise ValueError(
                "Images at 1088x1280 require an 'undistort' object with "
                "left_camera_info/right_camera_info for native-resolution undistortion."
            )
        left_key, right_key = _stereo_image_keys(options)
        if left_key not in images or right_key not in images:
            raise ValueError(
                f"Field 'images' must contain stereo keys '{left_key}' and '{right_key}'."
            )
        left_ci = options.get("left_camera_info")
        right_ci = options.get("right_camera_info")
        if not isinstance(left_ci, dict) or not isinstance(right_ci, dict):
            raise ValueError(
                "Field 'undistort' must provide 'left_camera_info' and "
                "'right_camera_info' objects when images are 1088x1280."
            )
        leftover = [k for k in images if k not in (left_key, right_key)]
        if leftover:
            raise ValueError(
                f"Image keys {leftover} are 1088x1280 but not designated as "
                "left/right stereo; cannot undistort."
            )

        cv = _require_cv2()
        kwargs = dict(
            left_camera_info=left_ci,
            right_camera_info=right_ci,
            output_size="native",
            left_to_right=options.get("left_to_right"),
            rotation=options.get("rotation"),
            translation=options.get("translation"),
            alpha=float(options.get("alpha", 0.0)),
        )
        out = dict(images)
        out[left_key] = undistort_stereo_side_from_camera_info(
            rgb=images[left_key], eye="left", **kwargs
        )
        out[right_key] = undistort_stereo_side_from_camera_info(
            rgb=images[right_key], eye="right", **kwargs
        )
        target_wh = (_TRAIN_SHAPE[1], _TRAIN_SHAPE[0])  # cv2.resize takes (W, H)
        for k in (left_key, right_key):
            out[k] = cv.resize(out[k], target_wh, interpolation=cv.INTER_AREA)
        return out

    raise ValueError(
        f"Unsupported image resolution (H, W)={hw}; expected "
        f"{_NATIVE_SHAPE} (with stereo calibration) or {_TRAIN_SHAPE}."
    )


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
    images = _normalize_image_resolution(images=images, payload=payload)
    proprio_raw = _require_array(payload, "proprio_raw")
    if _CLIENT is not None and proprio_raw.size != _CLIENT.proprio_dim:
        raise ValueError(f"Field 'proprio_raw' must have length {_CLIENT.proprio_dim}, got {proprio_raw.size}.")
    model_input = {
        "images": images,
        "proprio_raw": proprio_raw,
    }
    for key in ("current_position", "cartesian_position"):
        if key in payload and payload[key] is not None:
            model_input["current_position"] = _require_array(payload, key)
            if model_input["current_position"].size < 6:
                raise ValueError(f"Field '{key}' must contain at least 6 values.")
            break
    if "current_position" not in model_input:
        action_mode = "delta6_abs_gripper" if _CLIENT is None else _CLIENT.action_mode
        if action_mode.startswith("delta6_abs_gripper"):
            raise ValueError(
                "Field 'current_position' or 'cartesian_position' is required for "
                "delta6_abs_gripper action modes. It must be the current 6D EEF pose, "
                "not joint proprio_raw[:6]."
            )
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
            "images": (
                "object mapping training camera keys to base64 PNG/JPEG RGB images; "
                "all cameras must share the same resolution, either 1088x1280 (raw, "
                "requires 'undistort' calibration) or 480x640 (already training-aligned)"
            ),
            "undistort": (
                "required when images are 1088x1280; ignored when images are 480x640. "
                "Stereo calibration object used to undistort at native resolution before "
                "the server resizes each camera to 480x640 with cv2.INTER_AREA."
            ),
            "undistort.left_image_key": "optional image key for the left stereo frame; defaults to the first model key",
            "undistort.right_image_key": "optional image key for the right stereo frame; defaults to the second model key",
            "undistort.left_camera_info": "required CameraInfo-style object for the left camera",
            "undistort.right_camera_info": "required CameraInfo-style object for the right camera",
            "undistort.left_to_right": "optional 4x4 transform or flattened 16 values for stereo rectification",
            "undistort.rotation": "optional 3x3 stereo rotation, alternative to left_to_right",
            "undistort.translation": "optional 3D stereo translation, alternative to left_to_right",
            "undistort.alpha": "optional OpenCV free scaling parameter, default 0.0",
            "proprio_raw": "raw model proprio vector; real_1048 uses [joint0..joint5, gripper]",
            "current_position": "required 6D current EEF pose [x,y,z,roll,pitch,yaw] for delta6_abs_gripper modes; cartesian_position is accepted as an alias",
        },
        "response": {
            "action_format": "cartesian_absolute",
            "actions": "[T,7] absolute EEF action chunk [x,y,z,roll,pitch,yaw,gripper]",
            "action_mode": "delta6_abs_gripper",
            "action_semantics": "description of delta shifting/integration before the absolute action chunk is returned",
            "normalized_action_shape": "[T,D] model output shape before denormalization",
            "full_prompt": "prompt used to load the text embedding cache",
        },
        "example": {
            "instruction": "open the door",
            "images": {"head_left": "<base64>", "right_wrist_left": "<base64>"},
            "proprio_raw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
            "undistort": {
                "left_image_key": "head_left",
                "right_image_key": "right_wrist_left",
                "left_camera_info": {"width": 1280, "height": 1088, "k": "[...]", "d": "[...]"},
                "right_camera_info": {"width": 1280, "height": 1088, "k": "[...]", "d": "[...]"},
            },
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
        action_mode=args.model_action_mode,
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
    parser.add_argument(
        "--model-action-mode",
        choices=("delta6_abs_gripper", "delta6_abs_gripper_forward", "absolute", "absolute_joint"),
        default="delta6_abs_gripper",
    )
    parser.add_argument("--output-action-format", default="cartesian_absolute")
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
        "action_mode": _CLIENT.action_mode,
        "num_inference_steps": _CLIENT.num_inference_steps,
        "device": _CLIENT.device,
        "opencv_available": opencv_available(),
        "accepted_image_resolutions": [list(_NATIVE_SHAPE), list(_TRAIN_SHAPE)],
        "undistort_required_at": list(_NATIVE_SHAPE),
        "resize_interpolation": "cv2.INTER_AREA",
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
