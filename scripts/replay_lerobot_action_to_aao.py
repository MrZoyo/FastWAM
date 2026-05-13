#!/usr/bin/env python
"""Replay LeRobot dataset actions in AAO and save comparison videos.

The current sim/real LeRobot action convention is:

    [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]

The first six values are frame-aligned backward deltas, i.e.
``pose[t] - pose[t-1]``.  This script resets AAO from the source MCAP with
AAO DataReplay semantics, skips row 0, cumulatively integrates rows 1..N from
the reset EEF pose, and sends absolute EEF pose targets to AAO.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def _parse_early_gpu(argv: list[str]) -> int:
    for index, item in enumerate(argv):
        if item == "--gpu" and index + 1 < len(argv):
            return int(argv[index + 1])
        if item.startswith("--gpu="):
            return int(item.split("=", 1)[1])
    return 0


_EARLY_GPU = _parse_early_gpu(sys.argv[1:])
if _EARLY_GPU >= 0:
    _gpu = str(_EARLY_GPU)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _gpu)
    os.environ.setdefault("EGL_VISIBLE_DEVICES", _gpu)
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _gpu)
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.closed_loop_eval.episode_recorder import to_jsonable
from fastwam.closed_loop_eval.model_clients import FastWAMModelClient
from fastwam.closed_loop_eval.observation_adapter import AAOObservationAdapter, split_batched_observation
from fastwam.closed_loop_eval.runner import _extract_mujoco_diagnostics, _numeric_delta
from fastwam.closed_loop_eval.sim_service_client import DEFAULT_AAO_ROOT

LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET = Path("/DATA/disk1/zoyo/sim/open_door_augmented_sim_lerobot")
DEFAULT_MCAP_ROOT = Path("/DATA/disk1/zoyo/mcap/real_1048")
DEFAULT_SOURCE_VIEWS = (
    "observation.images.head_left",
    "observation.images.right_wrist_left",
)


def _set_env_or_fail_on_conflict(key: str, value: str, *, reason: str) -> None:
    existing = os.environ.get(key)
    if existing is not None and existing != value:
        raise RuntimeError(
            f"{key} is already set to {existing!r}, but {reason} requires {value!r}. "
            f"Unset {key} or pass matching GPU settings explicitly."
        )
    os.environ[key] = value


def _prepare_runtime_env(gpu: int) -> None:
    python_bin = str(Path(sys.prefix) / "bin")
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([python_bin, *path_parts])
    os.environ.setdefault("MUJOCO_GL", "egl")
    if gpu >= 0:
        gpu_str = str(gpu)
        reason = f"AAO dataset action replay --gpu={gpu_str}"
        _set_env_or_fail_on_conflict("CUDA_VISIBLE_DEVICES", gpu_str, reason=reason)
        _set_env_or_fail_on_conflict("EGL_VISIBLE_DEVICES", gpu_str, reason=reason)
        _set_env_or_fail_on_conflict("MUJOCO_EGL_DEVICE_ID", gpu_str, reason=reason)


def _load_aao_replay_modules(aao_root: Path):
    renderer_src = Path(__file__).resolve().parents[1] / "third_party" / "GaussianRenderer" / "src"
    if renderer_src.exists():
        renderer_src_str = str(renderer_src)
        if renderer_src_str not in sys.path:
            sys.path.insert(0, renderer_src_str)

    root = aao_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"AAO root does not exist: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    try:
        from mcap.reader import make_reader as _make_reader  # noqa: F401
        from mcap_ros2idl_support import Ros2DecodeFactory as _Ros2DecodeFactory  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "AAO MCAP replay dependencies are missing. Install them in the FastWAM "
            "environment before running this script:\n"
            "  uv pip install --python .venv/bin/python mcap mcap-ros2idl-support"
        ) from exc

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from auto_atom.runner.data_replay import DataReplayRunner
    from auto_atom.utils.pose import euler_to_quaternion

    return compose, initialize_config_dir, OmegaConf, open_dict, DataReplayRunner, euler_to_quaternion


def _episode_parquet(dataset_dir: Path, episode_index: int) -> Path:
    candidates = sorted((dataset_dir / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet"))
    if not candidates:
        raise FileNotFoundError(f"Episode parquet not found for episode {episode_index}: {dataset_dir / 'data'}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple episode parquet files found for episode {episode_index}: {candidates}")
    return candidates[0]


def _load_report(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "meta" / "augmented_sim_conversion_report.json"
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _episode_source(report: dict[str, Any], episode_index: int) -> str:
    for episode in report.get("episodes", []):
        if int(episode.get("episode_index", -1)) == int(episode_index):
            return str(episode["source"])
    raise KeyError(f"episode_index={episode_index} not found in conversion report.")


def _source_parts(source: str) -> tuple[str, str]:
    if "::" not in source:
        raise ValueError(f"Unsupported episode source format: {source!r}")
    mcap_name, variant = source.split("::", 1)
    if not mcap_name.endswith(".mcap"):
        mcap_name = f"{mcap_name}.mcap"
    return mcap_name, variant


def _variant_index(variant: str) -> int:
    match = re.fullmatch(r"door_(\d+)", variant)
    if not match:
        raise ValueError(f"Cannot infer env index from variant {variant!r}; pass --variant-env-index.")
    return int(match.group(1))


def _load_actions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    if "action" not in df.columns:
        raise KeyError(f"{path} does not contain an 'action' column.")
    actions = np.stack([np.asarray(item, dtype=np.float32).reshape(-1) for item in df["action"]], axis=0)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise RuntimeError(f"Expected action shape [T,7+], got {actions.shape}.")
    if "observation.state" in df.columns:
        states = np.stack([np.asarray(item, dtype=np.float32).reshape(-1) for item in df["observation.state"]], axis=0)
    else:
        states = np.empty((len(actions), 0), dtype=np.float32)
    return actions[:, :7].astype(np.float32, copy=False), states


def _resize_rgb(frame: np.ndarray, *, height: int) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.shape[0] != height:
        width = max(1, int(round(arr.shape[1] * (height / arr.shape[0]))))
        arr = np.asarray(Image.fromarray(arr).resize((width, height), Image.BILINEAR), dtype=np.uint8)
    return arr


def _annotate(frame: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(16, frame.shape[0] // 28))
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((6, 6, bbox[2] + 14, bbox[3] + 14), fill=(0, 0, 0))
    draw.text((10, 10), label, fill=(255, 255, 255), font=font)
    return np.asarray(image, dtype=np.uint8)


def _stack_h(frames: list[np.ndarray], *, height: int) -> np.ndarray:
    resized = [_resize_rgb(frame, height=height) for frame in frames]
    return np.concatenate(resized, axis=1)


def _stack_v(rows: list[np.ndarray]) -> np.ndarray:
    width = max(row.shape[1] for row in rows)
    padded = []
    for row in rows:
        if row.shape[1] == width:
            padded.append(row)
            continue
        pad = np.zeros((row.shape[0], width - row.shape[1], 3), dtype=np.uint8)
        padded.append(np.concatenate([row, pad], axis=1))
    return np.concatenate(padded, axis=0)


def _source_video_paths(dataset_dir: Path, episode_index: int, source_views: tuple[str, ...]) -> list[Path]:
    paths = []
    for view in source_views:
        candidates = sorted((dataset_dir / "videos").glob(f"chunk-*/{view}/episode_{episode_index:06d}.mp4"))
        if candidates:
            paths.append(candidates[0])
    return paths


def _read_source_frames(paths: list[Path], *, max_frames: int | None = None) -> list[np.ndarray]:
    if not paths:
        return []
    readers = [imageio.get_reader(str(path)) for path in paths]
    frames: list[np.ndarray] = []
    try:
        lengths = []
        for reader in readers:
            try:
                lengths.append(reader.count_frames())
            except Exception:  # noqa: BLE001
                lengths.append(max_frames or 0)
        limit = min(lengths) if all(item > 0 for item in lengths) else (max_frames or 0)
        if max_frames is not None:
            limit = min(limit, max_frames) if limit > 0 else max_frames
        for index in range(limit):
            frames.append(_stack_h([reader.get_data(index) for reader in readers], height=480))
    finally:
        for reader in readers:
            reader.close()
    return frames


def _extract_camera_frame(observation: dict[str, dict[str, Any]], camera_name: str, env_index: int) -> np.ndarray:
    payload = observation.get(f"{camera_name}/color/image_raw")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Missing camera observation: {camera_name}/color/image_raw")
    data = np.asarray(payload.get("data"), dtype=np.uint8)
    if data.ndim == 4:
        return data[env_index]
    if data.ndim == 3:
        return data
    raise RuntimeError(f"Unexpected camera data shape for {camera_name}: {data.shape}")


def _env_mask(batch_size: int, env_index: int) -> np.ndarray:
    if env_index < 0 or env_index >= batch_size:
        raise ValueError(f"variant env index {env_index} is outside AAO batch size {batch_size}.")
    mask = np.zeros(batch_size, dtype=bool)
    mask[env_index] = True
    return mask


def _task_update_to_dict(update: Any) -> dict[str, Any]:
    return {
        "stage_index": getattr(update, "stage_index", None),
        "stage_name": getattr(update, "stage_name", None),
        "status": getattr(update, "status", None),
        "done": getattr(update, "done", None),
        "success": getattr(update, "success", None),
        "details": getattr(update, "details", None),
        "phase": getattr(update, "phase", None),
        "phase_step": getattr(update, "phase_step", None),
    }


def _build_replay_cfg(
    *,
    args: argparse.Namespace,
    mcap_path: Path,
    compose: Any,
    initialize_config_dir: Any,
    open_dict: Any,
) -> Any:
    config_dir = Path(args.aao_root).expanduser().resolve() / "aao_configs"
    overrides = list(args.override)
    if not any(item.startswith("env.batch_size=") for item in overrides):
        overrides.append(f"env.batch_size={args.batch_size}")
    if not any(item.startswith("assets_dir=") for item in overrides):
        overrides.append(f"assets_dir={Path(args.aao_root).expanduser().resolve() / 'assets'}")
    if not any("env.viewer.disable" in item for item in overrides):
        overrides.append("++env.viewer.disable=true")
    overrides.extend(
        [
            f"+replay.mcap_path={mcap_path}",
            "+replay.load_on_initialize=true",
            "+replay.reset_from_first_frame=true",
            "+replay.done_on_success=false",
        ]
    )

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=args.task, overrides=overrides)
    with open_dict(cfg):
        cfg.env.update_freq = int(args.aao_update_freq)
        cfg.replay.steps_per_action = 1
    return cfg


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(to_jsonable(payload), fp, ensure_ascii=False, indent=2)


def _write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fp:
        json.dump(to_jsonable(rows), fp, ensure_ascii=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _prepare_runtime_env(args.gpu)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    report = _load_report(dataset_dir)
    source = _episode_source(report, args.episode_index)
    mcap_name, source_variant = _source_parts(source)
    variant = args.variant or source_variant
    variant_env_index = args.variant_env_index if args.variant_env_index is not None else _variant_index(variant)
    mcap_path = Path(args.mcap_root).expanduser().resolve() / mcap_name
    if not mcap_path.exists():
        raise FileNotFoundError(f"Source MCAP not found: {mcap_path}")

    episode_path = _episode_parquet(dataset_dir, args.episode_index)
    raw_actions, raw_states = _load_actions(episode_path)
    output_dir = Path(args.output_dir).expanduser()
    if args.timestamp_output:
        output_dir = output_dir / f"episode_{args.episode_index:06d}_{variant}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        compose,
        initialize_config_dir,
        _OmegaConf,
        open_dict,
        DataReplayRunner,
        euler_to_quaternion,
    ) = _load_aao_replay_modules(Path(args.aao_root))
    cfg = _build_replay_cfg(args=args, mcap_path=mcap_path, compose=compose, initialize_config_dir=initialize_config_dir, open_dict=open_dict)

    runner = DataReplayRunner().from_config(cfg)
    trace: list[dict[str, Any]] = []
    replay_frames: list[np.ndarray] = []
    last_update = None
    started = time.perf_counter()
    try:
        reset_update = runner.reset()
        last_update = reset_update
        env = runner.get_env()
        batch_size = int(runner.batch_size)
        mask = _env_mask(batch_size, variant_env_index)

        reset_obs = runner.get_observation()
        split_obs = split_batched_observation(reset_obs, batch_size)
        adapter = AAOObservationAdapter(
            env.get_info(),
            selected_cameras=[args.head_camera, args.wrist_camera],
            history_frames=1,
        )
        adapter.reset()
        adapter.extend([split_obs[variant_env_index]])
        initial_model_input = adapter.build_model_input(
            {
                "head_left": args.head_camera,
                "right_wrist_left": args.wrist_camera,
            }
        )
        current_pose = np.asarray(initial_model_input["cartesian_position"], dtype=np.float32)
        initial_eef_pose = {
            "position": current_pose[:3].tolist(),
            "rpy": current_pose[3:6].tolist(),
        }
        initial_diagnostics = _extract_mujoco_diagnostics_from_runner(runner)

        replay_frames.append(
            _annotate(
                _stack_h(
                    [
                        _extract_camera_frame(reset_obs, args.head_camera, variant_env_index),
                        _extract_camera_frame(reset_obs, args.wrist_camera, variant_env_index),
                    ],
                    height=480,
                ),
                "AAO replay reset",
            )
        )

        max_action_row = min(len(raw_actions) - 1, args.max_actions or len(raw_actions) - 1)
        for action_index in range(1, max_action_row + 1):
            raw_action = raw_actions[action_index]
            current_pose = FastWAMModelClient._delta_to_absolute(
                raw_action.reshape(1, -1),
                current_pose,
                frame_aligned_backward_delta=False,
            )[0, :6]
            absolute_action = np.concatenate([current_pose, raw_action[6:7]], axis=0).astype(np.float32)
            remote_action = {
                "position": np.repeat(absolute_action[:3][None, :], batch_size, axis=0),
                "orientation": np.repeat(
                    np.asarray(euler_to_quaternion(tuple(float(v) for v in absolute_action[3:6])), dtype=np.float32)[None, :],
                    batch_size,
                    axis=0,
                ),
                "gripper": np.repeat(absolute_action[6:7][None, :], batch_size, axis=0),
            }
            for repeat_index in range(args.action_repeat):
                last_update = runner._require_evaluator().update(remote_action, env_mask=mask)  # noqa: SLF001
            obs = runner.get_observation()
            replay_frames.append(
                _annotate(
                    _stack_h(
                        [
                            _extract_camera_frame(obs, args.head_camera, variant_env_index),
                            _extract_camera_frame(obs, args.wrist_camera, variant_env_index),
                        ],
                        height=480,
                    ),
                    f"AAO replay action row {action_index}",
                )
            )
            if action_index <= 5 or action_index == max_action_row:
                trace.append(
                    {
                        "action_index": int(action_index),
                        "raw_lerobot_action_row": raw_action.tolist(),
                        "absolute_action": absolute_action.tolist(),
                    }
                )

        final_diagnostics = _extract_mujoco_diagnostics_from_runner(runner)
        summary_obj = runner.summarize(
            last_update,
            max_updates=max_action_row * args.action_repeat,
            updates_used=max_action_row * args.action_repeat,
            elapsed_time_sec=time.perf_counter() - started,
        )
    finally:
        runner.close()

    replay_video = output_dir / "aao_replay_multicam.mp4"
    imageio.mimsave(replay_video, replay_frames, fps=args.fps, codec="libx264", quality=8)

    comparison_video = None
    source_paths = _source_video_paths(dataset_dir, args.episode_index, tuple(args.source_views.split(",")))
    source_frames = _read_source_frames(source_paths, max_frames=len(replay_frames))
    if source_frames:
        comparison_frames = []
        for index, replay_frame in enumerate(replay_frames):
            source_frame = source_frames[min(index, len(source_frames) - 1)]
            comparison_frames.append(
                _stack_v(
                    [
                        _annotate(_resize_rgb(source_frame, height=480), "dataset video"),
                        replay_frame,
                    ]
                )
            )
        comparison_video = output_dir / "dataset_vs_aao_replay.mp4"
        imageio.mimsave(comparison_video, comparison_frames, fps=args.fps, codec="libx264", quality=8)

    summary = {
        "dataset": str(dataset_dir),
        "episode_index": int(args.episode_index),
        "episode_parquet": str(episode_path),
        "source": source,
        "source_mcap": str(mcap_path),
        "variant": variant,
        "variant_env_index": int(variant_env_index),
        "num_dataset_frames": int(len(raw_actions)),
        "num_replayed_actions": int(min(len(raw_actions) - 1, args.max_actions or len(raw_actions) - 1)),
        "action_repeat": int(args.action_repeat),
        "fps": int(args.fps),
        "reset_semantics": "AAO DataReplay reset from MCAP first frame + transform_resets, then LeRobot action[1:] backward deltas integrated into absolute EEF targets.",
        "action_semantics": "LeRobot row t stores pose[t]-pose[t-1] for first 6 dims; row 0 is skipped for replay. Gripper is absolute.",
        "initial_lerobot_state": raw_states[0].tolist() if len(raw_states) else None,
        "raw_action_row0": raw_actions[0].tolist(),
        "first_applied_raw_action_row": raw_actions[1].tolist() if len(raw_actions) > 1 else None,
        "initial_eef_pose_base": initial_eef_pose,
        "trace_samples": trace,
        "initial_diagnostics": initial_diagnostics,
        "final_diagnostics": final_diagnostics,
        "diagnostic_delta": _numeric_delta(final_diagnostics, initial_diagnostics),
        "last_update": _task_update_to_dict(last_update),
        "aao_summary": summary_obj,
        "outputs": {
            "replay_video": str(replay_video),
            "comparison_video": None if comparison_video is None else str(comparison_video),
            "source_videos": [str(path) for path in source_paths],
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_trace(output_dir / "trace.json.gz", trace)
    return summary


def _extract_mujoco_diagnostics_from_runner(runner: Any) -> dict[str, Any]:
    class _Shim:
        def _require_evaluator(self) -> Any:
            return runner._require_evaluator()  # noqa: SLF001

    return _extract_mujoco_diagnostics(_Shim())


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--mcap-root", default=str(DEFAULT_MCAP_ROOT))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--variant", default=None, help="Door variant tag such as door_2; defaults to the conversion report source.")
    parser.add_argument("--variant-env-index", type=int, default=None, help="AAO batch index for the selected variant; defaults to the numeric suffix in --variant.")
    parser.add_argument("--aao-root", default=str(DEFAULT_AAO_ROOT))
    parser.add_argument("--task", default="open_door_airbot_play_back_gs")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--aao-update-freq", type=int, default=100)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--head-camera", default="env2_cam")
    parser.add_argument("--wrist-camera", default="eef_wrist_cam")
    parser.add_argument("--source-views", default=",".join(DEFAULT_SOURCE_VIEWS))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-dir", default="runs/aao_dataset_action_replay")
    parser.add_argument("--timestamp-output", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    try:
        summary = run(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = Path(summary["outputs"]["replay_video"]).parent
    print(json.dumps(to_jsonable({"summary_path": str(output_dir / "summary.json"), **summary["outputs"]}), indent=2))


if __name__ == "__main__":
    main()
