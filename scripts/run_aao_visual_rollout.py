#!/usr/bin/env python
"""Run AAO closed-loop rollout and save pred/recon/actual visual comparison."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
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
from PIL import Image, ImageDraw, ImageFont

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.closed_loop_eval.episode_recorder import to_jsonable
from fastwam.closed_loop_eval.model_clients import FastWAMModelClient
from fastwam.closed_loop_eval.observation_adapter import AAOObservationAdapter, split_batched_observation
from fastwam.closed_loop_eval.runner import (
    DEFAULT_MIX_CKPT,
    DEFAULT_MIX_CONFIG,
    DEFAULT_MIX_STATS,
    _clamp_gripper,
    _extract_mujoco_diagnostics,
    _numeric_delta,
    _parse_camera_map,
    _validate_camera_map,
    _validated_actions,
)
from fastwam.closed_loop_eval.sim_service_client import DEFAULT_AAO_ROOT, SimulatorServiceClient

LOGGER = logging.getLogger(__name__)


def _rgb(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        arr = np.asarray(value.convert("RGB"))
    else:
        arr = np.asarray(value)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr.astype(np.uint8, copy=False)


def _resize_rgb(value: Any, width: int = 224, height: int = 224) -> np.ndarray:
    arr = _rgb(value)
    if arr.shape[:2] != (height, width):
        arr = np.asarray(Image.fromarray(arr).resize((width, height), Image.BILINEAR), dtype=np.uint8)
    return arr


def _split_two_views(value: Any, *, view_width: int = 224, height: int = 224) -> tuple[np.ndarray, np.ndarray]:
    arr = _rgb(value)
    if arr.shape[:2] != (height, view_width * 2):
        arr = np.asarray(Image.fromarray(arr).resize((view_width * 2, height), Image.BILINEAR), dtype=np.uint8)
    return arr[:, :view_width], arr[:, view_width:]


def _annotate(cell: np.ndarray, label: str) -> np.ndarray:
    image = Image.fromarray(cell)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((5, 5, bbox[2] + 13, bbox[3] + 13), fill=(0, 0, 0))
    draw.text((9, 9), label, fill=(255, 255, 255), font=font)
    return np.asarray(image, dtype=np.uint8)


def _compose_grid(
    *,
    pred_frame: Any,
    recon_frame: Any,
    actual_views: tuple[np.ndarray, np.ndarray],
    window_index: int,
    model_step: str,
    sim_update: int,
    video_index: int,
) -> np.ndarray:
    pred_left, pred_right = _split_two_views(pred_frame)
    recon_left, recon_right = _split_two_views(recon_frame)
    actual_left = _resize_rgb(actual_views[0])
    actual_right = _resize_rgb(actual_views[1])
    rows = [
        [
            _annotate(pred_left, f"pred head w{window_index} m{model_step} u{sim_update} v{video_index}"),
            _annotate(pred_right, f"pred wrist w{window_index} m{model_step} u{sim_update} v{video_index}"),
        ],
        [
            _annotate(recon_left, f"vae head w{window_index} m{model_step} u{sim_update} v{video_index}"),
            _annotate(recon_right, f"vae wrist w{window_index} m{model_step} u{sim_update} v{video_index}"),
        ],
        [
            _annotate(actual_left, f"actual head w{window_index} m{model_step} u{sim_update}"),
            _annotate(actual_right, f"actual wrist w{window_index} m{model_step} u{sim_update}"),
        ],
    ]
    return np.concatenate([np.concatenate(row, axis=1) for row in rows], axis=0)


def _model_view_pair(client: FastWAMModelClient, model_input: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    # Show what the model actually sees: training-aligned preview, split horizontally.
    if len(client.image_shapes) != 2:
        raise RuntimeError(f"Visual rollout requires exactly two model cameras, got {len(client.image_shapes)}.")
    if client.concat_multi_camera != "horizontal":
        raise RuntimeError(
            f"Visual rollout split assumes concat_multi_camera='horizontal', got {client.concat_multi_camera!r}."
        )
    preview = client.preview_uint8(model_input["images"])  # (H, W, 3) uint8
    width = preview.shape[1]
    if width % 2 != 0:
        raise RuntimeError(f"Preview width must be even for horizontal split, got {width}.")
    half = width // 2
    return preview[:, :half], preview[:, half:]


def _video_frame_index(
    *,
    model_step_in_window: float,
    action_horizon: int,
    num_video_frames: int,
) -> int:
    if model_step_in_window <= 0:
        return 0
    ratio = action_horizon / float(max(num_video_frames - 1, 1))
    return min(num_video_frames - 1, int(np.floor(model_step_in_window / ratio)))


def _build_model_client(args: argparse.Namespace) -> FastWAMModelClient:
    return FastWAMModelClient(
        config_path=args.fastwam_config,
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        text_cache_dir=args.text_cache_dir,
        instruction=args.instruction,
        action_horizon=args.action_horizon,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        rand_device=args.rand_device,
        action_mode=args.model_action_mode,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive.")
    if args.num_windows <= 0:
        raise ValueError("--num-windows must be positive.")
    if args.action_repeat <= 0:
        raise ValueError("--action-repeat must be positive.")
    if args.action_horizon % max(args.num_video_frames - 1, 1) != 0:
        raise ValueError(
            "--action-horizon must be divisible by --num-video-frames - 1 "
            f"for exact video/action alignment, got {args.action_horizon} and {args.num_video_frames}."
        )
    if args.num_video_frames % 4 != 1:
        raise ValueError("--num-video-frames must satisfy T % 4 == 1 for FastWAM video decoding.")

    max_model_steps = int(args.max_model_steps or (args.num_windows * args.action_horizon))
    max_sim_updates = max_model_steps * args.action_repeat
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sim = SimulatorServiceClient(
        aao_root=args.aao_root,
        gpu=args.gpu,
        sim_loop_frequency=0.0,
    )
    client: FastWAMModelClient | None = None
    combined_frames: list[np.ndarray] = []
    windows: list[dict[str, Any]] = []
    model_steps_used = 0
    sim_updates_used = 0
    error: str | None = None
    initial_diagnostics: dict[str, Any] = {}
    final_diagnostics: dict[str, Any] = {}
    summary: Any = None
    start_time = time.perf_counter()
    try:
        sim.connect()
        init_result = sim.init(
            config_name=args.task,
            overrides=list(args.override),
            action_format="cartesian_absolute",
        )
        sim_info = init_result["info"]
        camera_map = _parse_camera_map(args.camera_map)
        selected_cameras = _validate_camera_map(camera_map, sim_info)
        adapter = AAOObservationAdapter(
            sim_info,
            selected_cameras=selected_cameras,
            history_frames=1,
        )
        client = _build_model_client(args)

        reset_obs = sim.reset()
        if not isinstance(reset_obs, dict):
            reset_obs = sim.get_observation()
        observations = split_batched_observation(reset_obs, sim.batch_size)
        adapter.reset()
        adapter.extend([observations[0]])
        initial_diagnostics = _extract_mujoco_diagnostics(sim)

        for window_index in range(args.num_windows):
            if model_steps_used >= max_model_steps:
                break
            window_start_model_step = model_steps_used
            window_start_sim_update = sim_updates_used
            model_input = adapter.build_model_input(camera_map, proprio_mode=args.proprio_mode)
            response = client.infer_joint_video(
                model_input,
                num_video_frames=args.num_video_frames,
                test_action_with_infer_action=args.test_action_consistency,
            )
            actions = _clamp_gripper(
                _validated_actions(response, expected_format="cartesian_absolute", action_dim=7),
                gripper_min=args.gripper_min,
                gripper_max=args.gripper_max,
                gripper_index=-1,
            )
            remaining_model_steps = max_model_steps - model_steps_used
            chunk_steps = min(args.action_horizon, len(actions), remaining_model_steps)
            if chunk_steps <= 0:
                break

            frame_entries: list[dict[str, Any]] = []
            recon_inputs = [model_input]
            capture_steps = [0]
            video_stride = args.action_horizon // (args.num_video_frames - 1)
            for chunk_step_index in range(chunk_steps):
                action = actions[chunk_step_index]
                repeat_steps = min(args.action_repeat, max_sim_updates - sim_updates_used)
                last_actual_input: dict[str, Any] | None = None
                for repeat_index in range(repeat_steps):
                    sim.update_cartesian_action(action)
                    sim_updates_used += 1
                    obs = sim.get_observation()
                    observations = split_batched_observation(obs, sim.batch_size)
                    adapter.extend([observations[0]])
                    last_actual_input = adapter.build_model_input(camera_map, proprio_mode=args.proprio_mode)
                    if args.frame_sampling == "sim-update":
                        local_progress = float(chunk_step_index) + float(repeat_index + 1) / float(args.action_repeat)
                        frame_entries.append(
                            {
                                "actual_views": _model_view_pair(client, last_actual_input),
                                "model_step_label": f"{window_start_model_step + local_progress:.1f}",
                                "model_step_in_window": local_progress,
                                "sim_update": sim_updates_used,
                            }
                        )

                model_steps_used += 1
                step_in_window = chunk_step_index + 1
                if last_actual_input is None:
                    raise RuntimeError("No AAO observation was captured for the current model action.")
                if args.frame_sampling == "model-action":
                    frame_entries.append(
                        {
                            "actual_views": _model_view_pair(client, last_actual_input),
                            "model_step_label": str(window_start_model_step + step_in_window),
                            "model_step_in_window": float(step_in_window),
                            "sim_update": sim_updates_used,
                        }
                    )
                if step_in_window % video_stride == 0:
                    recon_inputs.append(last_actual_input)
                    capture_steps.append(step_in_window)

                if model_steps_used >= max_model_steps:
                    break

            while len(recon_inputs) < args.num_video_frames:
                recon_inputs.append(recon_inputs[-1])
                capture_steps.append(capture_steps[-1])
            if len(recon_inputs) > args.num_video_frames:
                recon_inputs = recon_inputs[: args.num_video_frames]
                capture_steps = capture_steps[: args.num_video_frames]

            recon_frames = client.reconstruct_video_from_model_inputs(recon_inputs)
            pred_frames = list(response["video"])
            if len(pred_frames) != args.num_video_frames or len(recon_frames) != args.num_video_frames:
                raise RuntimeError(
                    "Unexpected pred/recon frame count: "
                    f"pred={len(pred_frames)} recon={len(recon_frames)} expected={args.num_video_frames}."
                )

            for entry in frame_entries:
                video_index = _video_frame_index(
                    model_step_in_window=float(entry["model_step_in_window"]),
                    action_horizon=args.action_horizon,
                    num_video_frames=args.num_video_frames,
                )
                combined_frames.append(
                    _compose_grid(
                        pred_frame=pred_frames[video_index],
                        recon_frame=recon_frames[video_index],
                        actual_views=entry["actual_views"],
                        window_index=window_index,
                        model_step=str(entry["model_step_label"]),
                        sim_update=int(entry["sim_update"]),
                        video_index=video_index,
                    )
                )

            windows.append(
                {
                    "window_index": window_index,
                    "start_model_step": window_start_model_step,
                    "end_model_step": model_steps_used,
                    "start_sim_update": window_start_sim_update,
                    "end_sim_update": sim_updates_used,
                    "model_action_steps": chunk_steps,
                    "output_frames": len(frame_entries),
                    "frame_sampling": args.frame_sampling,
                    "sim_updates": sim_updates_used - window_start_sim_update,
                    "pred_video_frames": len(pred_frames),
                    "recon_video_frames": len(recon_frames),
                    "video_capture_model_steps": capture_steps,
                }
            )

        elapsed = time.perf_counter() - start_time
        final_diagnostics = _extract_mujoco_diagnostics(sim)
        summary = sim.summarize(
            max_updates=max_sim_updates,
            updates_used=sim_updates_used,
            elapsed_time_sec=elapsed,
        )
    except Exception as exc:  # noqa: BLE001
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        LOGGER.exception("Visual rollout failed")
    finally:
        if client is not None:
            client.close()
        sim.close()

    if not combined_frames:
        raise RuntimeError(f"No visual frames were generated. error={error}")
    video_path = output_root / "pred_vae_actual_3x2.mp4"
    with imageio.get_writer(str(video_path), fps=float(args.video_fps), macro_block_size=None) as writer:
        for frame in combined_frames:
            writer.append_data(frame)

    payload = {
        "task": args.task,
        "output_dir": str(output_root),
        "video_path": str(video_path),
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "fastwam_config": str(Path(args.fastwam_config).expanduser()),
        "dataset_stats": str(Path(args.dataset_stats).expanduser()),
        "text_cache_dir": str(Path(args.text_cache_dir).expanduser()),
        "model_action_mode": args.model_action_mode,
        "camera_map": _parse_camera_map(args.camera_map),
        "selected_cameras": selected_cameras if "selected_cameras" in locals() else None,
        "control": {
            "num_windows": int(args.num_windows),
            "action_horizon_model_steps": int(args.action_horizon),
            "max_model_steps": int(max_model_steps),
            "action_repeat_sim_updates": int(args.action_repeat),
            "max_sim_updates": int(max_sim_updates),
            "num_video_frames_per_window": int(args.num_video_frames),
            "frame_sampling": args.frame_sampling,
            "output_frames": len(combined_frames),
            "video_fps": float(args.video_fps),
            "gripper_min": args.gripper_min,
            "gripper_max": args.gripper_max,
        },
        "model_steps_used": model_steps_used,
        "sim_updates_used": sim_updates_used,
        "windows": windows,
        "summary": summary,
        "mujoco_diagnostics": {
            "initial": initial_diagnostics,
            "final": final_diagnostics,
            "delta": _numeric_delta(final_diagnostics, initial_diagnostics),
        },
        "error": error,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(to_jsonable(payload), fp, indent=2)
    if error is not None:
        raise RuntimeError(error)
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aao-root", default=str(DEFAULT_AAO_ROOT))
    parser.add_argument("--task", default="open_door_airbot_play_gs")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-dir", default="runs/aao_closed_loop/mix20k_open_door_gs_2win_visual")
    parser.add_argument("--num-windows", type=int, default=2)
    parser.add_argument("--max-model-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--num-video-frames", type=int, default=9)
    parser.add_argument(
        "--frame-sampling",
        choices=("model-action", "sim-update"),
        default="model-action",
        help="Save one output frame per model action or per AAO simulator update.",
    )
    parser.add_argument("--camera-map", default="head_left=env2_cam,right_wrist_left=eef_wrist_cam")
    parser.add_argument("--fastwam-config", default=str(DEFAULT_MIX_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_MIX_CKPT))
    parser.add_argument("--dataset-stats", default=str(DEFAULT_MIX_STATS))
    parser.add_argument("--text-cache-dir", default="data/text_embeds_cache/mix")
    parser.add_argument("--instruction", default="open the door")
    parser.add_argument(
        "--model-action-mode",
        choices=("delta6_abs_gripper", "delta6_abs_gripper_forward", "absolute"),
        default="delta6_abs_gripper",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gripper-min", type=float, default=0.02)
    parser.add_argument("--gripper-max", type=float, default=0.0945)
    parser.add_argument("--proprio-mode", choices=("cartesian", "joint"), default="joint")
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--test-action-consistency", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    result = run(args)
    compact = {
        "task": result["task"],
        "video_path": result["video_path"],
        "summary_path": str(Path(result["output_dir"]) / "summary.json"),
        "model_steps_used": result["model_steps_used"],
        "sim_updates_used": result["sim_updates_used"],
        "output_frames": result["control"]["output_frames"],
    }
    print(json.dumps(to_jsonable(compact), indent=2))


if __name__ == "__main__":
    main()
