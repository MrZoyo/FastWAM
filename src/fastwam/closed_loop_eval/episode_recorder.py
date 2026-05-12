"""Episode recording utilities for AAO closed-loop evaluation."""

from __future__ import annotations

import gzip
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .observation_adapter import SimFrame


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


class _SafeEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        converted = to_jsonable(o)
        return converted if converted is not o else str(o)


class EpisodeRecorder:
    def __init__(
        self,
        *,
        episode_dir: Path,
        episode_name: str,
        task_name: str,
        camera_names: list[str],
        fps: int = 10,
        save_video: bool = True,
    ) -> None:
        self.episode_dir = episode_dir
        self.episode_name = episode_name
        self.task_name = task_name
        self.camera_names = list(camera_names)
        self.fps = int(fps)
        self.save_video = bool(save_video)
        self.video_frames: list[np.ndarray] = []
        self.trace_steps: list[dict[str, Any]] = []

    def record(
        self,
        *,
        step_index: int,
        chunk_index: int,
        chunk_step_index: int,
        sim_frame: SimFrame,
        update: Any,
        action_cartesian: np.ndarray | None,
        action: np.ndarray | None = None,
        action_format: str | None = None,
        remote_action: dict[str, Any] | None = None,
        model_response: dict[str, Any] | None = None,
        chunk_action_index: int | None = None,
        repeat_index: int | None = None,
    ) -> None:
        if self.save_video:
            self.video_frames.append(self._compose_multicam_frame(sim_frame))
        self.trace_steps.append(
            {
                "step_index": int(step_index),
                "chunk_index": int(chunk_index),
                "chunk_step_index": int(chunk_step_index),
                "chunk_action_index": None if chunk_action_index is None else int(chunk_action_index),
                "repeat_index": None if repeat_index is None else int(repeat_index),
                "timestamp_ns": float(sim_frame.timestamp_ns),
                "action_format": action_format,
                "action": None if action is None else np.asarray(action, dtype=np.float32),
                "action_cartesian": None if action_cartesian is None else np.asarray(action_cartesian, dtype=np.float32),
                "remote_action": remote_action,
                "update": update,
                "robot_state": sim_frame.robot_state,
                "model_response_summary": self._summarize_model_response(model_response),
            }
        )

    def finalize(
        self,
        *,
        summary: Any,
        records: list[Any],
        metadata: dict[str, Any],
        error: str | None = None,
    ) -> None:
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {
            "task_name": self.task_name,
            "episode_name": self.episode_name,
            "episode_dir": str(self.episode_dir),
            "camera_names": list(self.camera_names),
            "num_video_frames": len(self.video_frames),
            "num_recorded_steps": len(self.trace_steps),
            "summary": summary,
            "records": records,
            "error": error,
        }
        with (self.episode_dir / "summary.json").open("w", encoding="utf-8") as fp:
            json.dump(to_jsonable(summary_payload), fp, indent=2, cls=_SafeEncoder)

        trace_payload = {
            "metadata": metadata,
            "steps": self.trace_steps,
            "summary": summary,
            "records": records,
            "error": error,
        }
        with gzip.open(self.episode_dir / "client_trace.json.gz", "wt", encoding="utf-8") as fp:
            json.dump(to_jsonable(trace_payload), fp, cls=_SafeEncoder)

        if self.save_video and self.video_frames:
            with imageio.get_writer(
                str(self.episode_dir / "multicam.mp4"),
                fps=float(self.fps),
                macro_block_size=None,
            ) as writer:
                for frame in self.video_frames:
                    writer.append_data(frame)

    def _compose_multicam_frame(self, sim_frame: SimFrame) -> np.ndarray:
        tiles: list[np.ndarray] = []
        tile_hw: tuple[int, int] | None = None
        for camera_name in self.camera_names:
            camera = sim_frame.cameras.get(camera_name)
            if camera is None:
                raise RuntimeError(f"Missing camera '{camera_name}' in recorded SimFrame.")
            rgb = self._coerce_rgb(camera.get("rgb"), camera_name=camera_name)
            if tile_hw is None:
                tile_hw = rgb.shape[:2]
            else:
                rgb = self._resize_rgb(rgb, tile_hw)
            tiles.append(self._annotate_rgb(rgb, camera_name))
        if not tiles:
            raise RuntimeError("EpisodeRecorder has no cameras to compose.")
        return np.concatenate(tiles, axis=1)

    @staticmethod
    def _coerce_rgb(value: Any, *, camera_name: str) -> np.ndarray:
        if value is None:
            raise RuntimeError(f"Missing RGB frame for camera '{camera_name}'.")
        rgb = np.asarray(value, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise RuntimeError(f"Camera '{camera_name}' expected RGB shape [H,W,3], got {rgb.shape}.")
        return rgb

    @staticmethod
    def _resize_rgb(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        height, width = target_hw
        if rgb.shape[:2] == (height, width):
            return rgb
        return np.asarray(Image.fromarray(rgb).resize((width, height), Image.BILINEAR), dtype=np.uint8)

    @staticmethod
    def _annotate_rgb(rgb: np.ndarray, label: str) -> np.ndarray:
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, rgb.shape[0] // 24))
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        margin = 8
        pad = 6
        draw.rectangle(
            (margin, margin, margin + bbox[2] - bbox[0] + 2 * pad, margin + bbox[3] - bbox[1] + 2 * pad),
            fill=(0, 0, 0),
        )
        draw.text((margin + pad, margin + pad), label, fill=(255, 255, 255), font=font)
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _summarize_model_response(model_response: dict[str, Any] | None) -> dict[str, Any] | None:
        if model_response is None:
            return None
        summary: dict[str, Any] = {}
        for key, value in model_response.items():
            if isinstance(value, np.ndarray):
                summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            else:
                summary[key] = value
        return summary
