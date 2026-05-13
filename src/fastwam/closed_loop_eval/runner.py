"""Command-line runner for FastWAM + AAO closed-loop evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .episode_recorder import EpisodeRecorder, to_jsonable
from .model_clients import BaseModelClient, FastWAMModelClient, HoldJointModelClient, HoldModelClient
from .observation_adapter import AAOObservationAdapter, split_batched_observation
from .sim_service_client import DEFAULT_AAO_ROOT, SimulatorServiceClient

logger = logging.getLogger(__name__)


def _parse_camera_map(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Camera map item must be train=sim, got '{item}'.")
        train_key, sim_key = item.split("=", 1)
        mapping[train_key.strip()] = sim_key.strip()
    if not mapping:
        raise argparse.ArgumentTypeError("Camera map must not be empty.")
    return mapping


def _validate_camera_map(camera_map: dict[str, str], sim_info: dict[str, Any]) -> list[str]:
    available = list(sim_info.get("cameras", {}).keys())
    missing = [sim_key for sim_key in camera_map.values() if sim_key not in available]
    if missing:
        raise RuntimeError(f"AAO cameras {missing} are not available. Available cameras: {available}")
    return list(dict.fromkeys(camera_map.values()))


def _validated_actions(
    action_payload: dict[str, Any],
    *,
    expected_format: str,
    action_dim: int | None = None,
) -> np.ndarray:
    if action_payload.get("action_format") != expected_format:
        raise RuntimeError(f"Model action_format must be '{expected_format}'.")
    actions = np.asarray(action_payload.get("actions"), dtype=np.float32)
    if actions.ndim != 2:
        raise RuntimeError(f"Model actions must have shape [T,D], got {actions.shape}.")
    if expected_format == "cartesian_absolute":
        action_dim = 7 if action_dim is None else int(action_dim)
    elif expected_format == "joint_absolute":
        if action_dim is None:
            if actions.shape[1] < 1:
                raise RuntimeError(f"Model actions must have shape [T,D] with D >= 1, got {actions.shape}.")
            return actions
        action_dim = int(action_dim)
    else:
        raise RuntimeError(f"Unsupported expected action format: {expected_format}")
    if actions.shape[1] != action_dim:
        raise RuntimeError(f"Model actions must have shape [T,{action_dim}], got {actions.shape}.")
    return actions


def _clamp_gripper(
    actions: np.ndarray,
    *,
    gripper_min: float | None,
    gripper_max: float | None,
    gripper_index: int,
) -> np.ndarray:
    if gripper_min is None and gripper_max is None:
        return actions
    clipped = np.asarray(actions, dtype=np.float32).copy()
    index = int(gripper_index)
    if index < 0:
        index = clipped.shape[1] + index
    if index < 0 or index >= clipped.shape[1]:
        raise RuntimeError(f"gripper_index={gripper_index} is out of bounds for action shape {clipped.shape}.")
    low = -np.inf if gripper_min is None else float(gripper_min)
    high = np.inf if gripper_max is None else float(gripper_max)
    clipped[:, index] = np.clip(clipped[:, index], low, high)
    return clipped


def _effective_gripper_bounds(args: argparse.Namespace) -> tuple[float | None, float | None]:
    # No hardcoded defaults: clamping range is dataset-specific (e.g. cup GT max=0.1144,
    # real_1048 GT max=0.0904). Users / benchmark profiles must opt in via
    # --gripper-min / --gripper-max when they actually want clamping.
    return args.gripper_min, args.gripper_max


def _infer_action_dim(sim_info: dict[str, Any], action_format: str, operator: str = "arm") -> int | None:
    if action_format == "cartesian_absolute":
        return 7
    if action_format != "joint_absolute":
        return None
    operators = sim_info.get("operators")
    op_cfg: dict[str, Any] | None = None
    if isinstance(operators, dict):
        candidate = operators.get(operator)
        if isinstance(candidate, dict):
            op_cfg = candidate
    elif isinstance(operators, list):
        for candidate in operators:
            if isinstance(candidate, dict) and candidate.get("name", operator) == operator:
                op_cfg = candidate
                break
    if op_cfg is None:
        return None
    arm = op_cfg.get("arm_actuators") or []
    eef = op_cfg.get("eef_actuators") or []
    return len(arm) + len(eef)


def _resolve_control_mode(args: argparse.Namespace) -> str:
    return "continuous" if float(getattr(args, "sim_loop_frequency", 0.0)) > 0 else "lockstep"


def _effective_sim_loop_frequency(args: argparse.Namespace) -> float:
    return float(getattr(args, "sim_loop_frequency", 0.0))


def _resolve_overrides(args: argparse.Namespace) -> list[str]:
    """Build the final hydra override list, prepending implicit toggles before user overrides
    so a later --override on the same key still wins."""
    implicit: list[str] = []
    if getattr(args, "disable_arm_eef_randomization", False):
        implicit.extend([
            "task.randomization.arm.eef.x=[0.0,0.0]",
            "task.randomization.arm.eef.y=[0.0,0.0]",
            "task.randomization.arm.eef.z=[0.0,0.0]",
        ])
    return implicit + list(args.override)


def _continuous_hold_seconds(args: argparse.Namespace, sim_info: dict[str, Any] | None = None) -> float:
    obs_interval = float(getattr(args, "obs_interval", -1.0))
    if obs_interval >= 0:
        return obs_interval
    loop_hz = _effective_sim_loop_frequency(args)
    if loop_hz <= 0 and sim_info is not None and sim_info.get("env_update_freq") is not None:
        loop_hz = float(sim_info["env_update_freq"])
    if loop_hz <= 0:
        return 0.0
    return float(args.action_repeat) / loop_hz


def _control_metadata(
    args: argparse.Namespace,
    *,
    sim_info: dict[str, Any] | None = None,
    stride: int | None = None,
) -> dict[str, Any]:
    aao_update_hz = None if sim_info is None else sim_info.get("env_update_freq")
    control_mode = _resolve_control_mode(args)
    sim_loop_frequency = _effective_sim_loop_frequency(args)
    gripper_min, gripper_max = _effective_gripper_bounds(args)
    effective_action_hz = None
    if control_mode == "continuous" and sim_loop_frequency > 0:
        effective_action_hz = sim_loop_frequency / float(args.action_repeat)
    elif aao_update_hz is not None:
        effective_action_hz = float(aao_update_hz) / float(args.action_repeat)
    stride_value = getattr(args, "stride", None) if stride is None else stride
    return {
        "action_horizon": int(args.action_horizon),
        "prediction_window_model_actions": int(args.action_horizon),
        "stride_model_actions": None if stride_value is None else int(stride_value),
        "action_repeat_sim_updates": int(args.action_repeat),
        "max_updates": int(args.max_updates),
        "train_action_hz": float(getattr(args, "train_action_hz", 20.0)),
        "aao_update_hz": None if aao_update_hz is None else float(aao_update_hz),
        "effective_action_hz_in_sim": effective_action_hz,
        "action_format": args.action_format,
        "proprio_mode": args.proprio_mode,
        "gripper_min": gripper_min,
        "gripper_max": gripper_max,
        "gripper_index": int(args.gripper_index),
        "ignore_done": bool(getattr(args, "ignore_done", False)),
        "stop_on_done": not bool(getattr(args, "ignore_done", False)),
        "control_mode": control_mode,
        "sim_loop_frequency": sim_loop_frequency,
        "continuous_target_hold_sec": (
            _continuous_hold_seconds(args, sim_info) if control_mode == "continuous" else None
        ),
    }


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


def _build_model_client(args: argparse.Namespace) -> BaseModelClient:
    if args.model_client == "hold":
        return HoldModelClient(horizon=args.action_horizon)
    if args.model_client == "hold-joint":
        return HoldJointModelClient(horizon=args.action_horizon)
    if args.model_client == "fastwam":
        missing = [
            name for name, value in (
                ("--fastwam-config", args.fastwam_config),
                ("--checkpoint", args.checkpoint),
                ("--dataset-stats", args.dataset_stats),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"--model-client=fastwam requires {', '.join(missing)} to be set explicitly."
            )
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
            output_action_format=args.output_action_format or args.action_format,
        )
    raise ValueError(f"Unsupported model client: {args.model_client}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.action_repeat <= 0:
        raise ValueError("--action-repeat must be positive.")
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive.")
    if args.max_updates <= 0:
        raise ValueError("--max-updates must be positive.")
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sim = SimulatorServiceClient(
        aao_root=args.aao_root,
        gpu=args.gpu,
        sim_loop_frequency=_effective_sim_loop_frequency(args),
    )
    model_client: BaseModelClient | None = None
    aggregate: list[dict[str, Any]] = []
    try:
        sim.connect()
        init_result = sim.init(
            config_name=args.task,
            overrides=_resolve_overrides(args),
            action_format=args.action_format,
        )
        sim_info = init_result["info"]
        action_dim = _infer_action_dim(sim_info, args.action_format)
        gripper_min, gripper_max = _effective_gripper_bounds(args)
        control_mode = _resolve_control_mode(args)
        continuous_hold_sec = _continuous_hold_seconds(args, sim_info)
        camera_map = _parse_camera_map(args.camera_map)
        selected_cameras = _validate_camera_map(camera_map, sim_info)
        model_client = _build_model_client(args)

        for episode_index in range(args.episodes):
            episode_name = f"{args.task}_ep{episode_index:03d}"
            episode_dir = output_root / episode_name
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
                "aao_root": str(Path(args.aao_root).expanduser()),
                "camera_map": camera_map,
                "selected_cameras": selected_cameras,
                "stage_plans": sim.stage_plans,
                "model_client": args.model_client,
                "model_action_mode": getattr(args, "model_action_mode", None),
                "control": _control_metadata(args, sim_info=sim_info),
            }
            error: str | None = None
            updates_used = 0
            initial_diagnostics: dict[str, Any] = {}
            final_diagnostics: dict[str, Any] = {}
            start_time = time.perf_counter()
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
                    action=None,
                    action_format=args.action_format,
                    remote_action=None,
                    model_response=None,
                )

                chunk_index = 0
                while updates_used < args.max_updates:
                    model_input = adapter.build_model_input(camera_map, proprio_mode=args.proprio_mode)
                    model_response = model_client.infer(model_input)
                    actions = _clamp_gripper(
                        _validated_actions(
                            model_response,
                            expected_format=args.action_format,
                            action_dim=action_dim,
                        ),
                        gripper_min=gripper_min,
                        gripper_max=gripper_max,
                        gripper_index=args.gripper_index,
                    )
                    remaining_updates = args.max_updates - updates_used
                    chunk_steps = min(
                        args.stride,
                        len(actions),
                        max(1, int(np.ceil(remaining_updates / args.action_repeat))),
                    )
                    episode_done = False
                    for chunk_step_index in range(chunk_steps):
                        action = actions[chunk_step_index]
                        repeat_steps = min(args.action_repeat, args.max_updates - updates_used)
                        if control_mode == "continuous":
                            if args.action_format == "joint_absolute":
                                remote_action = sim.set_joint_action(action)
                            else:
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
                                action_cartesian=action if args.action_format == "cartesian_absolute" else None,
                                action=action,
                                action_format=args.action_format,
                                remote_action=remote_action,
                                model_response=model_response if chunk_step_index == 0 else None,
                            )
                            done = np.asarray(task_state.get("done", [False]), dtype=bool).reshape(-1)
                            if not args.ignore_done and done.size and bool(done[0]):
                                episode_done = True
                        else:
                            for repeat_index in range(repeat_steps):
                                if args.action_format == "joint_absolute":
                                    update, remote_action = sim.update_joint_action(action)
                                else:
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
                                    action_cartesian=action if args.action_format == "cartesian_absolute" else None,
                                    action=action,
                                    action_format=args.action_format,
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
                logger.exception("Episode failed: %s", episode_name)

            elapsed = time.perf_counter() - start_time
            final_diagnostics = _extract_mujoco_diagnostics(sim)
            diagnostics = {
                "initial": initial_diagnostics,
                "final": final_diagnostics,
                "delta": _numeric_delta(final_diagnostics, initial_diagnostics),
            }
            metadata["mujoco_diagnostics"] = diagnostics
            summary = sim.summarize(
                max_updates=args.max_updates,
                updates_used=updates_used,
                elapsed_time_sec=elapsed,
            )
            recorder.finalize(
                summary=summary,
                records=sim.records,
                metadata=metadata,
                error=error,
            )
            aggregate.append(
                {
                    "episode_name": episode_name,
                    "episode_dir": str(episode_dir),
                    "updates_used": updates_used,
                    "stride_model_actions": int(args.stride),
                    "action_repeat_sim_updates": int(args.action_repeat),
                    "control": _control_metadata(args, sim_info=sim_info),
                    "elapsed_time_sec": elapsed,
                    "error": error,
                    "summary": summary,
                    "mujoco_diagnostics": diagnostics,
                }
            )

        aggregate_payload = {
            "task": args.task,
            "output_dir": str(output_root),
            "control": _control_metadata(args, sim_info=sim_info),
            "episodes": aggregate,
        }
        with (output_root / "aggregate_summary.json").open("w", encoding="utf-8") as fp:
            json.dump(to_jsonable(aggregate_payload), fp, indent=2)
        return aggregate_payload
    finally:
        if model_client is not None:
            model_client.close()
        sim.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aao-root", default=str(DEFAULT_AAO_ROOT))
    parser.add_argument("--task", default="open_door_airbot_play_gs")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--disable-arm-eef-randomization",
        action="store_true",
        help="Force task.randomization.arm.eef.{x,y,z} to [0,0] so the arm "
             "always resets to its task_operators.arm.initial_state pose. Useful "
             "when the training dataset was collected without arm-eef randomization.",
    )
    parser.add_argument("--output-dir", default="runs/aao_closed_loop/open_door_airbot_play_gs")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-updates", type=int, default=40)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--action-repeat", type=int, default=5)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-format", choices=("cartesian_absolute", "joint_absolute"), default="cartesian_absolute")
    parser.add_argument("--proprio-mode", choices=("cartesian", "joint"), default="joint")
    parser.add_argument("--train-action-hz", type=float, default=20.0)
    parser.add_argument("--gripper-min", type=float, default=None)
    parser.add_argument("--gripper-max", type=float, default=None)
    parser.add_argument("--gripper-index", type=int, default=-1)
    parser.add_argument("--camera-map", default="head_left=env2_cam,right_wrist_left=eef_wrist_cam")
    parser.add_argument("--model-client", choices=("hold", "hold-joint", "fastwam"), default="hold")
    parser.add_argument("--fastwam-config", default=None,
        help="Required when --model-client=fastwam. Path to the training config.yaml.")
    parser.add_argument("--checkpoint", default=None,
        help="Required when --model-client=fastwam. Path to the model checkpoint .pt file.")
    parser.add_argument("--dataset-stats", default=None,
        help="Required when --model-client=fastwam. Path to the dataset_stats.json paired with the checkpoint; "
             "different runs/datasets have different normalization stats, do not rely on a fallback.")
    parser.add_argument("--text-cache-dir", default="data/text_embeds_cache/mix")
    parser.add_argument("--instruction", default="open the door")
    parser.add_argument(
        "--model-action-mode",
        choices=("delta6_abs_gripper", "delta6_abs_gripper_forward", "absolute", "absolute_joint"),
        default="delta6_abs_gripper",
    )
    parser.add_argument("--output-action-format", choices=("cartesian_absolute", "joint_absolute"), default=None)
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
    print(json.dumps(to_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
