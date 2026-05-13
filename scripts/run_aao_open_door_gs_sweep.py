#!/usr/bin/env python
"""Sweep FastWAM closed-loop eval over open-door GS visual variants."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.closed_loop_eval.episode_recorder import EpisodeRecorder, to_jsonable
from fastwam.closed_loop_eval.model_clients import BaseModelClient, FastWAMModelClient, HoldModelClient
from fastwam.closed_loop_eval.observation_adapter import AAOObservationAdapter, split_batched_observation
from fastwam.closed_loop_eval.runner import (
    DEFAULT_MIX_CKPT,
    DEFAULT_MIX_CONFIG,
    DEFAULT_MIX_STATS,
    _clamp_gripper,
    _continuous_hold_seconds,
    _control_metadata,
    _effective_sim_loop_frequency,
    _parse_camera_map,
    _resolve_control_mode,
    _validate_camera_map,
    _validated_actions,
)
from fastwam.closed_loop_eval.sim_service_client import DEFAULT_AAO_ROOT, SimulatorServiceClient

LOGGER = logging.getLogger(__name__)

DEFAULT_DOORS = ("door1", "door2", "door3", "door4", "door11", "door14", "door15", "door17", "door19")
DEFAULT_WALLS = (
    "wall0",
    "wall1",
    "wall2",
    "wall3",
    "wall4",
    "wall5",
    "wall6",
    "wall7",
    "wall8",
    "wall9",
    "wall10",
    "wall11",
    "wall12",
    "wall13",
)
DEFAULT_STRIDES = (4, 8)


def _build_model_client(args: argparse.Namespace) -> BaseModelClient:
    if args.model_client == "hold":
        return HoldModelClient(horizon=args.action_horizon)
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


def _open_door_overrides(args: argparse.Namespace, door: str, wall: str) -> list[str]:
    aao_root = Path(args.aao_root).expanduser().resolve()
    gs_dir = aao_root / "assets" / "gs" / "scenes" / "open_door"
    bg_dir = aao_root / "assets" / "gs" / "backgrounds" / "door_bg"
    knob_path = gs_dir / "real_knob1.ply"
    lock_path = gs_dir / "real_lock1.ply"
    wall_path = bg_dir / "wall" / f"{wall}.ply"
    inside_path = bg_dir / "inside" / "inside0.ply"
    required = [gs_dir / f"{door}.ply", knob_path, lock_path, wall_path, inside_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing GS assets for {door}/{wall}: {missing}")

    overrides = [
        f"door_name={door}",
        f"wall_name={wall}",
        "inside_name=inside0",
        f"env.gaussian_render.body_gaussians.handle_gs_frame={knob_path}",
        f"env.gaussian_render.body_gaussians.lock_gs_frame={lock_path}",
        "env.gaussian_render.background_transform_randomization.inside.y=[0.0,0.0]",
    ]
    overrides.extend(args.override)
    return overrides


def _extract_mujoco_diagnostics(sim: SimulatorServiceClient) -> dict[str, Any]:
    try:
        import mujoco
    except Exception as exc:  # noqa: BLE001
        return {"error": f"mujoco import failed: {type(exc).__name__}: {exc}"}

    try:
        ctx = sim._require_evaluator()._require_context()  # noqa: SLF001
        env = ctx.backend.env
        if hasattr(env, "envs") and env.envs:
            env = env.envs[0]
        model = getattr(env, "model", None)
        data = getattr(env, "data", None)
        if model is None or data is None:
            return {"error": "mujoco model/data unavailable"}

        diagnostics: dict[str, Any] = {"joints": {}, "bodies": {}, "sites": {}}
        for joint_name in ("door_hinge", "handle_hinge", "eef_claw_joint"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_addr = int(model.jnt_qposadr[joint_id])
                diagnostics["joints"][joint_name] = float(data.qpos[qpos_addr])

        for body_name in ("door_body", "handle_body_phys", "handle_gs_frame", "lock_gs_frame"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                diagnostics["bodies"][body_name] = {
                    "xpos": np.asarray(data.xpos[body_id], dtype=np.float64).tolist(),
                    "xquat_wxyz": np.asarray(data.xquat[body_id], dtype=np.float64).tolist(),
                }

        for site_name in ("handle_grasp_front_site", "handle_grasp_back_site", "eef_pose"):
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:
                diagnostics["sites"][site_name] = {
                    "xpos": np.asarray(data.site_xpos[site_id], dtype=np.float64).tolist(),
                    "xmat": np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).tolist(),
                }
        return diagnostics
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _numeric_delta(final: Any, initial: Any) -> Any:
    if isinstance(final, dict) and isinstance(initial, dict):
        keys = sorted(set(final) | set(initial))
        return {key: _numeric_delta(final.get(key), initial.get(key)) for key in keys}
    try:
        final_arr = np.asarray(final, dtype=np.float64)
        initial_arr = np.asarray(initial, dtype=np.float64)
        if final_arr.shape == initial_arr.shape and final_arr.size > 0:
            return (final_arr - initial_arr).tolist()
    except Exception:  # noqa: BLE001
        pass
    return None


def _select_combos(
    *,
    doors: list[str],
    walls: list[str],
    num_combos: int | None,
    seed: int,
) -> list[tuple[str, str]]:
    all_combos = [(door, wall) for door in doors for wall in walls]
    if num_combos is None:
        return all_combos
    if num_combos <= 0:
        raise ValueError("num_combos must be positive when provided.")
    if num_combos > len(all_combos):
        raise ValueError(f"num_combos={num_combos} exceeds available unique combos={len(all_combos)}.")

    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    used: set[tuple[str, str]] = set()
    door_order = list(doors)
    while len(selected) < num_combos:
        rng.shuffle(door_order)
        for door in door_order:
            available_walls = [wall for wall in walls if (door, wall) not in used]
            if not available_walls:
                continue
            wall = rng.choice(available_walls)
            combo = (door, wall)
            selected.append(combo)
            used.add(combo)
            if len(selected) >= num_combos:
                break
    return selected


def _run_one(
    *,
    args: argparse.Namespace,
    model_client: BaseModelClient,
    output_root: Path,
    door: str,
    wall: str,
    stride: int,
) -> dict[str, Any]:
    if hasattr(model_client, "call_index"):
        setattr(model_client, "call_index", 0)
    overrides = _open_door_overrides(args, door, wall)
    episode_name = f"{door}_{wall}_stride{stride}"
    episode_dir = output_root / f"stride_{stride}" / f"{door}_{wall}"
    sim = SimulatorServiceClient(
        aao_root=args.aao_root,
        gpu=args.gpu,
        sim_loop_frequency=_effective_sim_loop_frequency(args),
    )
    error: str | None = None
    updates_used = 0
    sim_info: dict[str, Any] = {}
    initial_diagnostics: dict[str, Any] = {}
    final_diagnostics: dict[str, Any] = {}
    start_time = time.perf_counter()
    try:
        sim.connect()
        init_result = sim.init(
            config_name=args.task,
            overrides=overrides,
            action_format="cartesian_absolute",
        )
        sim_info = init_result["info"]
        control_mode = _resolve_control_mode(args)
        continuous_hold_sec = _continuous_hold_seconds(args, sim_info)
        camera_map = _parse_camera_map(args.camera_map)
        selected_cameras = _validate_camera_map(camera_map, sim_info)
        recorder = EpisodeRecorder(
            episode_dir=episode_dir,
            episode_name=episode_name,
            task_name=args.task,
            camera_names=selected_cameras,
            fps=args.video_fps,
            save_video=not args.no_video,
        )
        metadata = {
            "task": args.task,
            "door": door,
            "wall": wall,
            "overrides": overrides,
            "aao_root": str(Path(args.aao_root).expanduser()),
            "camera_map": camera_map,
            "selected_cameras": selected_cameras,
            "stage_plans": sim.stage_plans,
            "model_client": args.model_client,
            "model_action_mode": args.model_action_mode,
            "control": _control_metadata(args, sim_info=sim_info, stride=stride),
        }
        adapter = AAOObservationAdapter(
            sim_info,
            selected_cameras=selected_cameras,
            history_frames=1,
        )

        try:
            reset_obs = sim.reset()
            if not isinstance(reset_obs, dict):
                reset_obs = sim.get_observation()
            observations = split_batched_observation(reset_obs, sim.batch_size)
            adapter.reset()
            adapter.extend([observations[0]])
            initial_diagnostics = _extract_mujoco_diagnostics(sim)
            recorder.record(
                step_index=0,
                chunk_index=-1,
                chunk_step_index=-1,
                chunk_action_index=-1,
                repeat_index=-1,
                sim_frame=adapter.latest_frame(),
                update=None,
                action_cartesian=None,
                remote_action=None,
                model_response=None,
            )

            chunk_index = 0
            while updates_used < args.max_updates:
                model_input = adapter.build_model_input(camera_map)
                model_response = model_client.infer(model_input)
                actions = _clamp_gripper(
                    _validated_actions(model_response, expected_format="cartesian_absolute", action_dim=7),
                    gripper_min=args.gripper_min,
                    gripper_max=args.gripper_max,
                    gripper_index=-1,
                )
                remaining_updates = args.max_updates - updates_used
                chunk_steps = min(
                    int(stride),
                    len(actions),
                    max(1, int(np.ceil(remaining_updates / args.action_repeat))),
                )
                episode_done = False
                for chunk_step_index in range(chunk_steps):
                    action = actions[chunk_step_index]
                    repeat_steps = min(args.action_repeat, args.max_updates - updates_used)
                    if control_mode == "continuous":
                        remote_action = sim.set_cartesian_action(action)
                        if continuous_hold_sec > 0:
                            time.sleep(continuous_hold_sec)
                        updates_used += repeat_steps
                        task_state = sim.get_task_state()
                        obs = sim.get_observation()
                        observations = split_batched_observation(obs, sim.batch_size)
                        adapter.extend([observations[0]])
                        recorder.record(
                            step_index=updates_used,
                            chunk_index=chunk_index,
                            chunk_step_index=chunk_step_index * args.action_repeat + repeat_steps - 1,
                            chunk_action_index=chunk_step_index,
                            repeat_index=repeat_steps - 1,
                            sim_frame=adapter.latest_frame(),
                            update=task_state,
                            action_cartesian=action,
                            remote_action=remote_action,
                            model_response=model_response if chunk_step_index == 0 else None,
                        )
                        done = np.asarray(task_state.get("done", [False]), dtype=bool).reshape(-1)
                        if not args.ignore_done and done.size and bool(done[0]):
                            episode_done = True
                    else:
                        for repeat_index in range(repeat_steps):
                            update, remote_action = sim.update_cartesian_action(action)
                            updates_used += 1
                            obs = sim.get_observation()
                            observations = split_batched_observation(obs, sim.batch_size)
                            adapter.extend([observations[0]])
                            recorder.record(
                                step_index=updates_used,
                                chunk_index=chunk_index,
                                chunk_step_index=chunk_step_index * args.action_repeat + repeat_index,
                                chunk_action_index=chunk_step_index,
                                repeat_index=repeat_index,
                                sim_frame=adapter.latest_frame(),
                                update=update,
                                action_cartesian=action,
                                remote_action=remote_action,
                                model_response=model_response if chunk_step_index == 0 and repeat_index == 0 else None,
                            )
                            task_state = sim.get_task_state()
                            done = np.asarray(task_state.get("done", [False]), dtype=bool).reshape(-1)
                            if not args.ignore_done and done.size and bool(done[0]):
                                episode_done = True
                                break
                    if episode_done or updates_used >= args.max_updates:
                        break
                chunk_index += 1
                task_state = sim.get_task_state()
                done = np.asarray(task_state.get("done", [False]), dtype=bool).reshape(-1)
                if not args.ignore_done and done.size and bool(done[0]):
                    break
        except Exception as exc:  # noqa: BLE001
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            metadata["traceback"] = traceback.format_exc()
            LOGGER.exception("Episode failed: %s", episode_name)

        elapsed = time.perf_counter() - start_time
        final_diagnostics = _extract_mujoco_diagnostics(sim)
        summary = sim.summarize(
            max_updates=args.max_updates,
            updates_used=updates_used,
            elapsed_time_sec=elapsed,
        )
        diagnostics = {
            "initial": initial_diagnostics,
            "final": final_diagnostics,
            "delta": _numeric_delta(final_diagnostics, initial_diagnostics),
        }
        metadata["mujoco_diagnostics"] = diagnostics
        recorder.finalize(
            summary=summary,
            records=sim.records,
            metadata=metadata,
            error=error,
        )
        return {
            "episode_name": episode_name,
            "episode_dir": str(episode_dir),
            "video_path": str(episode_dir / "multicam.mp4") if not args.no_video else None,
            "door": door,
            "wall": wall,
            "stride": int(stride),
            "stride_model_actions": int(stride),
            "action_repeat_sim_updates": int(args.action_repeat),
            "action_horizon": int(args.action_horizon),
            "max_updates": int(args.max_updates),
            "gripper_min": args.gripper_min,
            "gripper_max": args.gripper_max,
            "control": _control_metadata(args, sim_info=sim_info, stride=stride),
            "updates_used": updates_used,
            "elapsed_time_sec": elapsed,
            "error": error,
            "summary": summary,
            "mujoco_diagnostics": diagnostics,
        }
    finally:
        sim.close()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "door",
        "wall",
        "stride",
        "action_repeat_sim_updates",
        "action_horizon",
        "max_updates",
        "gripper_min",
        "gripper_max",
        "updates_used",
        "final_done",
        "final_success",
        "final_status",
        "door_hinge_delta",
        "handle_hinge_delta",
        "handle_body_displacement",
        "error",
        "episode_dir",
        "video_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            summary = to_jsonable(row.get("summary") or {})
            diagnostics = row.get("mujoco_diagnostics") or {}
            delta = diagnostics.get("delta") or {}
            joint_delta = delta.get("joints") or {}
            body_delta = delta.get("bodies") or {}
            handle_body = body_delta.get("handle_body_phys") or {}
            handle_displacement = ""
            if isinstance(handle_body, dict) and handle_body.get("xpos") is not None:
                handle_displacement = float(np.linalg.norm(np.asarray(handle_body["xpos"], dtype=np.float64)))
            writer.writerow(
                {
                    "door": row.get("door"),
                    "wall": row.get("wall"),
                    "stride": row.get("stride"),
                    "action_repeat_sim_updates": row.get("action_repeat_sim_updates"),
                    "action_horizon": row.get("action_horizon"),
                    "max_updates": row.get("max_updates"),
                    "gripper_min": row.get("gripper_min"),
                    "gripper_max": row.get("gripper_max"),
                    "updates_used": row.get("updates_used"),
                    "final_done": (summary.get("final_done") or [""])[0],
                    "final_success": (summary.get("final_success") or [""])[0],
                    "final_status": (summary.get("final_status") or [""])[0],
                    "door_hinge_delta": joint_delta.get("door_hinge", ""),
                    "handle_hinge_delta": joint_delta.get("handle_hinge", ""),
                    "handle_body_displacement": handle_displacement,
                    "error": row.get("error"),
                    "episode_dir": row.get("episode_dir"),
                    "video_path": row.get("video_path"),
                }
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.action_repeat <= 0:
        raise ValueError("--action-repeat must be positive.")
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive.")
    if args.max_updates <= 0:
        raise ValueError("--max-updates must be positive.")
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.gpu >= 0:
        gpu_str = str(args.gpu)
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_str)
        os.environ.setdefault("EGL_VISIBLE_DEVICES", gpu_str)
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", gpu_str)

    doors = [item.strip() for item in args.doors.split(",") if item.strip()]
    walls = [item.strip() for item in args.walls.split(",") if item.strip()]
    strides = [int(item.strip()) for item in args.strides.split(",") if item.strip()]
    if not strides or any(stride <= 0 for stride in strides):
        raise ValueError("--strides must contain positive integers.")
    combos = _select_combos(
        doors=doors,
        walls=walls,
        num_combos=args.num_combos,
        seed=args.combo_seed,
    )
    if args.limit_combos is not None:
        combos = combos[: int(args.limit_combos)]

    model_client = _build_model_client(args)
    rows: list[dict[str, Any]] = []
    try:
        total = len(combos) * len(strides)
        run_index = 0
        for stride in strides:
            for door, wall in combos:
                run_index += 1
                LOGGER.info("Running %d/%d: door=%s wall=%s stride=%s", run_index, total, door, wall, stride)
                row = _run_one(
                    args=args,
                    model_client=model_client,
                    output_root=output_root,
                    door=door,
                    wall=wall,
                    stride=stride,
                )
                rows.append(row)
                aggregate_payload = {
                    "task": args.task,
                    "output_dir": str(output_root),
                    "doors": doors,
                    "walls": walls,
                    "strides": strides,
                    "control": _control_metadata(args),
                    "episodes": rows,
                }
                with (output_root / "aggregate_summary.json").open("w", encoding="utf-8") as fp:
                    json.dump(to_jsonable(aggregate_payload), fp, indent=2)
                _write_csv(output_root / "sweep_results.csv", rows)
    finally:
        model_client.close()

    return {
        "task": args.task,
        "output_dir": str(output_root),
        "doors": doors,
        "walls": walls,
        "strides": strides,
        "control": _control_metadata(args),
        "episodes": rows,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aao-root", default=str(DEFAULT_AAO_ROOT))
    parser.add_argument("--task", default="open_door_airbot_play_gs")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-dir", default="runs/aao_closed_loop/fastwam_mix20k_open_door_gs_30env_random_bg_sweep")
    parser.add_argument("--doors", default=",".join(DEFAULT_DOORS))
    parser.add_argument("--walls", default=",".join(DEFAULT_WALLS))
    parser.add_argument("--strides", default=",".join(str(item) for item in DEFAULT_STRIDES))
    parser.add_argument("--num-combos", type=int, default=30)
    parser.add_argument("--combo-seed", type=int, default=20260508)
    parser.add_argument("--limit-combos", type=int, default=None)
    parser.add_argument("--max-updates", type=int, default=160)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--train-action-hz", type=float, default=20.0)
    parser.add_argument("--gripper-min", type=float, default=0.02)
    parser.add_argument("--gripper-max", type=float, default=0.0945)
    parser.add_argument("--camera-map", default="head_left=env2_cam,right_wrist_left=eef_wrist_cam")
    parser.add_argument("--model-client", choices=("hold", "fastwam"), default="fastwam")
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
    parser.add_argument("--sim-loop-frequency", type=float, default=0.0)
    parser.add_argument("--obs-interval", type=float, default=-1.0)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--ignore-done", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    result = run(args)
    output_root = Path(result["output_dir"])
    episodes = result["episodes"]
    compact = {
        "task": result["task"],
        "output_dir": str(output_root),
        "episode_count": len(episodes),
        "aggregate_summary": str(output_root / "aggregate_summary.json"),
        "sweep_results_csv": str(output_root / "sweep_results.csv"),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
