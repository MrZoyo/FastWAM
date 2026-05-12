"""Batch benchmark runner for FastWAM + AAO closed-loop evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .episode_recorder import to_jsonable
from .model_clients import BaseModelClient, FastWAMModelClient, HoldModelClient, ParallelFastWAMModelClient
from .observation_adapter import AAOObservationAdapter, split_batched_observation
from .runner import (
    _clamp_gripper,
    _control_metadata,
    _effective_gripper_bounds,
    _effective_sim_loop_frequency,
    _infer_action_dim,
    _parse_camera_map,
    _resolve_control_mode,
    _validate_camera_map,
    _validated_actions,
)
from .sim_service_client import DEFAULT_AAO_ROOT, SimulatorServiceClient

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BenchmarkProfile:
    name: str
    task: str
    instruction: str
    camera_map: str
    action_repeat: int
    train_action_hz: float
    max_updates: int
    proprio_mode: str = "cartesian"
    proprio_dim: int | None = None
    fastwam_config: str | None = None
    text_cache_dir: str | None = None


PROFILES: dict[str, BenchmarkProfile] = {
    "open_door_airbot_play_gs": BenchmarkProfile(
        name="open_door_airbot_play_gs",
        task="open_door_airbot_play_gs",
        instruction="open the door",
        camera_map="head_left=env2_cam,right_wrist_left=eef_wrist_cam",
        action_repeat=5,
        train_action_hz=20.0,
        max_updates=160,
        proprio_mode="cartesian",
        proprio_dim=7,
        fastwam_config=str(_REPO_ROOT / "configs/task/mix_uncond_2cam224_1e-4.yaml"),
        text_cache_dir="data/text_embeds_cache/mix",
    ),
    "cup_on_coaster_gs_airbot_p7": BenchmarkProfile(
        name="cup_on_coaster_gs_airbot_p7",
        task="cup_on_coaster_gs_airbot_p7",
        instruction="pick up the cup and place it on the coaster",
        camera_map="head_left=env1_cam,right_wrist_left=eef_wrist_cam",
        action_repeat=1,
        train_action_hz=50.0,
        max_updates=650,
        proprio_mode="joint",
        proprio_dim=8,
        fastwam_config=str(_REPO_ROOT / "configs/task/cup_uncond_2cam224_1e-4.yaml"),
        text_cache_dir="data/text_embeds_cache/cup",
    ),
}


DEFAULT_SENSOR_OVERRIDES = (
    "enable_depth=false",
    "enable_mask=false",
    "enable_heat_map=false",
)


def _profile_names() -> tuple[str, ...]:
    return tuple(sorted(PROFILES))


def _parse_model_gpus(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _resolve_profile(args: argparse.Namespace) -> BenchmarkProfile:
    profile = PROFILES[args.profile]
    values = {
        "task": args.task or profile.task,
        "instruction": args.instruction or profile.instruction,
        "camera_map": args.camera_map or profile.camera_map,
        "action_repeat": int(args.action_repeat or profile.action_repeat),
        "train_action_hz": float(args.train_action_hz or profile.train_action_hz),
        "max_updates": int(args.max_updates or profile.max_updates),
        "proprio_mode": args.proprio_mode or profile.proprio_mode,
        "proprio_dim": args.proprio_dim if args.proprio_dim is not None else profile.proprio_dim,
        "fastwam_config": args.fastwam_config or profile.fastwam_config,
        "text_cache_dir": args.text_cache_dir or profile.text_cache_dir,
    }
    return replace(profile, **values)


@dataclass
class EnvEpisodeState:
    env_index: int
    episode_index: int | None = None
    updates_used: int = 0
    model_steps_used: int = 0
    chunk_index: int = 0
    chunk_action_index: int = 0
    chunk_actions: np.ndarray | None = None
    last_action: np.ndarray | None = None
    start_perf: float = 0.0
    end_perf: float | None = None
    done: bool = False
    finished: bool = False
    success: Any = False
    status: str | None = None
    error: str | None = None
    model_infer_calls: int = 0
    model_infer_time_sec: float = 0.0
    sim_update_time_sec: float = 0.0
    first_done_update: int | None = None
    model_worker_indices: list[int] = field(default_factory=list)
    model_worker_gpus: list[str] = field(default_factory=list)


def _success_to_bool(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        return bool(value.astype(bool).any())
    return bool(value)


def _build_model_client(args: argparse.Namespace, profile: BenchmarkProfile) -> BaseModelClient:
    if args.model_client == "hold":
        if _parse_model_gpus(args.model_gpus):
            raise ValueError("--model-gpus only applies to --model-client fastwam.")
        return HoldModelClient(horizon=args.action_horizon)
    missing: list[str] = []
    if not profile.fastwam_config:
        missing.append("--fastwam-config")
    if not args.checkpoint:
        missing.append("--checkpoint")
    if not args.dataset_stats:
        missing.append("--dataset-stats")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"model-client=fastwam requires {joined}; checkpoint paths are intentionally not hard-coded.")
    client_kwargs = {
        "config_path": profile.fastwam_config,
        "checkpoint_path": args.checkpoint,
        "dataset_stats_path": args.dataset_stats,
        "text_cache_dir": profile.text_cache_dir,
        "instruction": profile.instruction,
        "action_horizon": args.action_horizon,
        "device": args.device,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "rand_device": args.rand_device,
        "action_mode": args.model_action_mode,
        "output_action_format": args.output_action_format or args.action_format,
    }
    model_gpus = _parse_model_gpus(args.model_gpus)
    if not model_gpus:
        return FastWAMModelClient(**client_kwargs)
    logger.info("Launching %s FastWAM model worker(s) on physical GPU(s): %s", len(model_gpus), ",".join(model_gpus))
    return ParallelFastWAMModelClient(
        model_gpus=model_gpus,
        worker_timeout_sec=args.model_worker_timeout_sec,
        worker_start_method=args.model_worker_start_method,
        **client_kwargs,
    )


def _validate_model_client(profile: BenchmarkProfile, model_client: BaseModelClient) -> None:
    proprio_dim = getattr(model_client, "proprio_dim", None)
    if profile.proprio_dim is not None and proprio_dim is not None and int(proprio_dim) != int(profile.proprio_dim):
        raise RuntimeError(
            f"Profile '{profile.name}' expects proprio_dim={profile.proprio_dim}, "
            f"but model config expects {proprio_dim}."
        )
    model_action_dim = getattr(model_client, "model_action_dim", None)
    if model_action_dim is not None and int(model_action_dim) != 7:
        raise RuntimeError(
            f"Benchmark mode expects FastWAM to predict 7D EEF pose + gripper actions, got {model_action_dim}D."
        )


def _resize_vector(values: np.ndarray, dim: int | None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if dim is None:
        return array
    if array.size < dim:
        array = np.pad(array, (0, int(dim) - array.size), constant_values=0.0)
    return array[: int(dim)].astype(np.float32, copy=False)


def _prepare_model_input(
    adapter: AAOObservationAdapter,
    *,
    camera_map: dict[str, str],
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    model_input = adapter.build_model_input(camera_map, proprio_mode=profile.proprio_mode)
    model_input["proprio_raw"] = _resize_vector(model_input["proprio_raw"], profile.proprio_dim)
    model_input["instruction"] = profile.instruction
    return model_input


def _as_array(value: Any, *, dtype: Any, default: Any, batch_size: int) -> np.ndarray:
    if value is None:
        return np.asarray([default] * batch_size, dtype=dtype)
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.size == 0:
        return np.asarray([default] * batch_size, dtype=dtype)
    if array.size == 1 and batch_size > 1:
        return np.repeat(array, batch_size)
    if array.size != batch_size:
        out = np.asarray([default] * batch_size, dtype=dtype)
        out[: min(batch_size, array.size)] = array[: min(batch_size, array.size)]
        return out
    return array


def _status_array(value: Any, *, batch_size: int) -> list[str | None]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, np.ndarray):
        items = value.reshape(-1).tolist()
    elif isinstance(value, list):
        items = value
    else:
        items = [value]
    if len(items) == 1 and batch_size > 1:
        items = items * batch_size
    if len(items) < batch_size:
        items = [*items, *([None] * (batch_size - len(items)))]
    return [None if item is None else str(getattr(item, "value", item)) for item in items[:batch_size]]


def _extract_update_arrays(update: Any, task_state: dict[str, Any], batch_size: int) -> tuple[np.ndarray, np.ndarray, list[str | None]]:
    done_source = getattr(update, "done", None)
    success_source = getattr(update, "success", None)
    status_source = getattr(update, "status", None)
    if done_source is None:
        done_source = task_state.get("done")
    if success_source is None:
        success_source = task_state.get("success")
    return (
        _as_array(done_source, dtype=bool, default=False, batch_size=batch_size),
        _as_array(success_source, dtype=object, default=False, batch_size=batch_size),
        _status_array(status_source, batch_size=batch_size),
    )


def _row_for_episode(
    *,
    state: EnvEpisodeState,
    profile: BenchmarkProfile,
    args: argparse.Namespace,
) -> dict[str, Any]:
    elapsed = None if state.end_perf is None else state.end_perf - state.start_perf
    return {
        "profile": profile.name,
        "task": profile.task,
        "env_index": int(state.env_index),
        "episode_index": None if state.episode_index is None else int(state.episode_index),
        "success": _success_to_bool(state.success),
        "done": bool(state.done),
        "finished": bool(state.finished),
        "status": state.status,
        "updates_used": int(state.updates_used),
        "model_steps_used": int(state.model_steps_used),
        "first_done_update": state.first_done_update,
        "elapsed_time_sec": elapsed,
        "model_infer_calls": int(state.model_infer_calls),
        "model_infer_time_sec": float(state.model_infer_time_sec),
        "sim_update_time_sec": float(state.sim_update_time_sec),
        "model_worker_indices": state.model_worker_indices,
        "model_worker_gpus": state.model_worker_gpus,
        "error": state.error,
        "action_repeat_sim_updates": int(profile.action_repeat),
        "action_horizon": int(args.action_horizon),
        "stride_model_actions": int(args.stride),
        "max_updates": int(profile.max_updates),
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "profile",
        "task",
        "env_index",
        "episode_index",
        "success",
        "done",
        "finished",
        "status",
        "updates_used",
        "model_steps_used",
        "first_done_update",
        "elapsed_time_sec",
        "model_infer_calls",
        "model_infer_time_sec",
        "sim_update_time_sec",
        "model_worker_indices",
        "model_worker_gpus",
        "error",
        "action_repeat_sim_updates",
        "action_horizon",
        "stride_model_actions",
        "max_updates",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_summary(
    *,
    output_root: Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    with (output_root / "benchmark_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(to_jsonable(payload), fp, indent=2)
    _write_rows_csv(output_root / "benchmark_results.csv", rows)


def _start_episode(
    *,
    sim: SimulatorServiceClient,
    env_state: EnvEpisodeState,
    episode_index: int,
    adapters: list[AAOObservationAdapter],
    camera_map: dict[str, str],
    profile: BenchmarkProfile,
) -> None:
    mask = np.zeros(sim.batch_size, dtype=bool)
    mask[env_state.env_index] = True
    reset_obs = sim.reset(env_mask=mask)
    if not isinstance(reset_obs, dict):
        reset_obs = sim.get_observation()
    observations = split_batched_observation(reset_obs, sim.batch_size)
    adapter = adapters[env_state.env_index]
    adapter.reset()
    adapter.extend([observations[env_state.env_index]])
    _prepare_model_input(adapter, camera_map=camera_map, profile=profile)
    env_state.episode_index = int(episode_index)
    env_state.updates_used = 0
    env_state.model_steps_used = 0
    env_state.chunk_index = 0
    env_state.chunk_action_index = 0
    env_state.chunk_actions = None
    env_state.last_action = None
    env_state.start_perf = time.perf_counter()
    env_state.end_perf = None
    env_state.done = False
    env_state.finished = False
    env_state.success = False
    env_state.status = None
    env_state.error = None
    env_state.model_infer_calls = 0
    env_state.model_infer_time_sec = 0.0
    env_state.sim_update_time_sec = 0.0
    env_state.first_done_update = None
    env_state.model_worker_indices = []
    env_state.model_worker_gpus = []


def _complete_episode(
    *,
    state: EnvEpisodeState,
    rows: list[dict[str, Any]],
    profile: BenchmarkProfile,
    args: argparse.Namespace,
    error: str | None = None,
) -> None:
    if state.finished:
        return
    if error is not None:
        state.error = error
    state.finished = True
    state.end_perf = time.perf_counter()
    rows.append(_row_for_episode(state=state, profile=profile, args=args))
    state.episode_index = None


def _record_task_update(
    *,
    state: EnvEpisodeState,
    done: bool,
    success: Any,
    status: str | None,
) -> None:
    if done:
        state.done = True
        if state.first_done_update is None:
            state.first_done_update = state.updates_used
    elif not state.done:
        state.done = False
    if _success_to_bool(success):
        state.success = success
    if status is not None:
        state.status = status


def _assign_new_episodes(
    *,
    sim: SimulatorServiceClient,
    env_states: list[EnvEpisodeState],
    adapters: list[AAOObservationAdapter],
    camera_map: dict[str, str],
    profile: BenchmarkProfile,
    next_episode_index: int,
    total_episodes: int,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> int:
    for state in env_states:
        if state.episode_index is not None or next_episode_index >= total_episodes:
            continue
        state.finished = False
        try:
            _start_episode(
                sim=sim,
                env_state=state,
                episode_index=next_episode_index,
                adapters=adapters,
                camera_map=camera_map,
                profile=profile,
            )
            next_episode_index += 1
        except Exception as exc:  # noqa: BLE001
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            state.episode_index = next_episode_index
            state.start_perf = time.perf_counter()
            _complete_episode(state=state, rows=rows, profile=profile, args=args, error=error)
            logger.exception("Failed to start benchmark episode env=%s episode=%s", state.env_index, next_episode_index)
            next_episode_index += 1
    return next_episode_index


def _active_states(env_states: list[EnvEpisodeState]) -> list[EnvEpisodeState]:
    return [state for state in env_states if state.episode_index is not None and not state.finished]


def _ensure_actions_for_states(
    *,
    states: list[EnvEpisodeState],
    adapters: list[AAOObservationAdapter],
    camera_map: dict[str, str],
    profile: BenchmarkProfile,
    model_client: BaseModelClient,
    args: argparse.Namespace,
    action_dim: int | None,
    rows: list[dict[str, Any]],
) -> None:
    needs_actions = [
        state
        for state in states
        if state.chunk_actions is None or state.chunk_action_index >= len(state.chunk_actions)
    ]
    if not needs_actions:
        return
    model_inputs = []
    for state in needs_actions:
        model_input = _prepare_model_input(
            adapters[state.env_index],
            camera_map=camera_map,
            profile=profile,
        )
        model_input["env_index"] = int(state.env_index)
        model_inputs.append(model_input)
    start = time.perf_counter()
    responses = model_client.infer_batch(model_inputs)
    infer_elapsed = time.perf_counter() - start
    per_env_elapsed = infer_elapsed / max(len(needs_actions), 1)
    gripper_min, gripper_max = _effective_gripper_bounds(args)
    for state, response in zip(needs_actions, responses, strict=True):
        try:
            actions = _clamp_gripper(
                _validated_actions(
                    response,
                    expected_format=args.action_format,
                    action_dim=action_dim,
                ),
                gripper_min=gripper_min,
                gripper_max=gripper_max,
                gripper_index=args.gripper_index,
            )
            state.chunk_actions = actions[: max(1, int(args.stride))]
            state.chunk_action_index = 0
            state.chunk_index += 1
            state.model_infer_calls += 1
            state.model_infer_time_sec += float(response.get("model_infer_time_sec_per_item", per_env_elapsed))
            worker_index = response.get("worker_index")
            if worker_index is not None and int(worker_index) not in state.model_worker_indices:
                state.model_worker_indices.append(int(worker_index))
            model_gpu = response.get("model_gpu")
            if model_gpu is not None and str(model_gpu) not in state.model_worker_gpus:
                state.model_worker_gpus.append(str(model_gpu))
        except Exception as exc:  # noqa: BLE001
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            _complete_episode(state=state, rows=rows, profile=profile, args=args, error=error)
            logger.exception("Model action validation failed env=%s episode=%s", state.env_index, state.episode_index)


def run(args: argparse.Namespace) -> dict[str, Any]:
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.total_episodes <= 0:
        raise ValueError("--total-episodes must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive.")
    if args.action_format != "cartesian_absolute":
        raise ValueError("Benchmark mode currently expects cartesian_absolute AAO actions.")

    profile = _resolve_profile(args)
    args.task = profile.task
    args.instruction = profile.instruction
    args.camera_map = profile.camera_map
    args.action_repeat = profile.action_repeat
    args.train_action_hz = profile.train_action_hz
    args.max_updates = profile.max_updates
    args.proprio_mode = profile.proprio_mode
    args.proprio_dim = profile.proprio_dim
    args.fastwam_config = profile.fastwam_config
    args.text_cache_dir = profile.text_cache_dir
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    overrides = [*args.override]
    if not any(item.startswith("env.batch_size=") for item in overrides):
        overrides.append(f"env.batch_size={args.batch_size}")
    for override in DEFAULT_SENSOR_OVERRIDES:
        key = override.split("=", 1)[0]
        if not any(item.startswith(f"{key}=") or item.startswith(f"++{key}=") for item in overrides):
            overrides.append(override)

    sim = SimulatorServiceClient(
        aao_root=args.aao_root,
        gpu=args.gpu,
        sim_loop_frequency=_effective_sim_loop_frequency(args),
    )
    model_client: BaseModelClient | None = None
    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    sim_info: dict[str, Any] = {}
    selected_cameras: list[str] = []
    camera_map: dict[str, str] = {}
    try:
        sim.connect()
        init_result = sim.init(
            config_name=profile.task,
            overrides=overrides,
            action_format=args.action_format,
        )
        sim_info = init_result["info"]
        if sim.batch_size != args.batch_size:
            raise RuntimeError(f"AAO batch_size={sim.batch_size} does not match requested {args.batch_size}.")
        action_dim = _infer_action_dim(sim_info, args.action_format)
        camera_map = _parse_camera_map(profile.camera_map)
        selected_cameras = _validate_camera_map(camera_map, sim_info)
        adapters = [
            AAOObservationAdapter(sim_info, selected_cameras=selected_cameras, history_frames=1)
            for _ in range(sim.batch_size)
        ]
        model_client = _build_model_client(args, profile)
        _validate_model_client(profile, model_client)
        env_states = [EnvEpisodeState(env_index=index) for index in range(sim.batch_size)]
        next_episode_index = _assign_new_episodes(
            sim=sim,
            env_states=env_states,
            adapters=adapters,
            camera_map=camera_map,
            profile=profile,
            next_episode_index=0,
            total_episodes=args.total_episodes,
            rows=rows,
            args=args,
        )

        while next_episode_index < args.total_episodes or _active_states(env_states):
            active = _active_states(env_states)
            if not active:
                next_episode_index = _assign_new_episodes(
                    sim=sim,
                    env_states=env_states,
                    adapters=adapters,
                    camera_map=camera_map,
                    profile=profile,
                    next_episode_index=next_episode_index,
                    total_episodes=args.total_episodes,
                    rows=rows,
                    args=args,
                )
                continue
            _ensure_actions_for_states(
                states=active,
                adapters=adapters,
                camera_map=camera_map,
                profile=profile,
                model_client=model_client,
                args=args,
                action_dim=action_dim,
                rows=rows,
            )
            active = _active_states(env_states)
            if not active:
                continue

            stepping_states: list[EnvEpisodeState] = []
            for state in active:
                if state.updates_used >= profile.max_updates:
                    state.status = state.status or "max_updates"
                    _complete_episode(state=state, rows=rows, profile=profile, args=args)
                    continue
                if state.chunk_actions is None or state.chunk_action_index >= len(state.chunk_actions):
                    continue
                action = np.asarray(state.chunk_actions[state.chunk_action_index], dtype=np.float32)
                state.last_action = action
                stepping_states.append(state)
                state.model_steps_used += 1
                state.chunk_action_index += 1

            if not stepping_states:
                next_episode_index = _assign_new_episodes(
                    sim=sim,
                    env_states=env_states,
                    adapters=adapters,
                    camera_map=camera_map,
                    profile=profile,
                    next_episode_index=next_episode_index,
                    total_episodes=args.total_episodes,
                    rows=rows,
                    args=args,
                )
                continue

            repeat_steps = min(profile.action_repeat, min(profile.max_updates - state.updates_used for state in stepping_states))
            for _ in range(repeat_steps):
                current_states = [state for state in stepping_states if state.episode_index is not None and not state.finished]
                if not current_states:
                    break
                action_rows = np.stack([state.last_action for state in current_states], axis=0)
                env_mask = np.zeros(sim.batch_size, dtype=bool)
                for state in current_states:
                    env_mask[state.env_index] = True
                start = time.perf_counter()
                update, _remote_action = sim.update_cartesian_actions(action_rows, env_mask=env_mask)
                sim_elapsed = time.perf_counter() - start
                obs = sim.get_observation()
                observations = split_batched_observation(obs, sim.batch_size)
                for state in current_states:
                    state.updates_used += 1
                    state.sim_update_time_sec += sim_elapsed / max(len(current_states), 1)
                    adapters[state.env_index].extend([observations[state.env_index]])
                task_state = sim.get_task_state()
                done_arr, success_arr, status_arr = _extract_update_arrays(update, task_state, sim.batch_size)
                for state in list(current_states):
                    done = bool(done_arr[state.env_index])
                    success = success_arr[state.env_index]
                    status = status_arr[state.env_index]
                    _record_task_update(state=state, done=done, success=success, status=status)
                    if (done and not args.ignore_done) or state.updates_used >= profile.max_updates:
                        if done or _success_to_bool(success):
                            state.success = success
                        if state.updates_used >= profile.max_updates and not state.done:
                            state.status = "max_updates"
                        _complete_episode(state=state, rows=rows, profile=profile, args=args)
                        stepping_states.remove(state)
                if not stepping_states:
                    break

            next_episode_index = _assign_new_episodes(
                sim=sim,
                env_states=env_states,
                adapters=adapters,
                camera_map=camera_map,
                profile=profile,
                next_episode_index=next_episode_index,
                total_episodes=args.total_episodes,
                rows=rows,
                args=args,
            )

        payload = _build_payload(
            args=args,
            profile=profile,
            model_client=model_client,
            output_root=output_root,
            rows=rows,
            sim_info=sim_info,
            camera_map=camera_map,
            selected_cameras=selected_cameras,
            overrides=overrides,
            wall_start=wall_start,
        )
        _write_summary(output_root=output_root, payload=payload, rows=rows)
        return payload
    finally:
        if model_client is not None:
            model_client.close()
        sim.close()


def _build_payload(
    *,
    args: argparse.Namespace,
    profile: BenchmarkProfile,
    model_client: BaseModelClient | None,
    output_root: Path,
    rows: list[dict[str, Any]],
    sim_info: dict[str, Any],
    camera_map: dict[str, str],
    selected_cameras: list[str],
    overrides: list[str],
    wall_start: float,
) -> dict[str, Any]:
    completed = len(rows)
    successes = sum(1 for row in rows if _success_to_bool(row.get("success")))
    elapsed = time.perf_counter() - wall_start
    return {
        "profile": to_jsonable(profile),
        "task": profile.task,
        "output_dir": str(output_root),
        "batch_size": int(args.batch_size),
        "total_episodes_requested": int(args.total_episodes),
        "episodes_completed": completed,
        "successes": successes,
        "success_rate": None if completed == 0 else successes / completed,
        "elapsed_time_sec": elapsed,
        "episodes_per_sec": None if elapsed <= 0 else completed / elapsed,
        "model_client": args.model_client,
        "model_gpus": _parse_model_gpus(args.model_gpus),
        "model_worker_start_method": args.model_worker_start_method,
        "model_worker_timeout_sec": float(args.model_worker_timeout_sec),
        "model_worker_metadata": None if model_client is None else getattr(model_client, "worker_metadata", None),
        "checkpoint": None if args.checkpoint is None else str(Path(args.checkpoint).expanduser()),
        "dataset_stats": None if args.dataset_stats is None else str(Path(args.dataset_stats).expanduser()),
        "fastwam_config": profile.fastwam_config,
        "text_cache_dir": profile.text_cache_dir,
        "camera_map": camera_map,
        "selected_cameras": selected_cameras,
        "overrides": overrides,
        "control": _control_metadata(args, sim_info=sim_info, stride=args.stride),
        "control_mode": _resolve_control_mode(args),
        "results_csv": str(output_root / "benchmark_results.csv"),
        "summary_json": str(output_root / "benchmark_summary.json"),
        "episodes": rows,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aao-root", default=str(DEFAULT_AAO_ROOT))
    parser.add_argument("--profile", choices=_profile_names(), default="open_door_airbot_play_gs")
    parser.add_argument("--task", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--output-dir", default="runs/aao_benchmark")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-episodes", type=int, default=4)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--action-format", choices=("cartesian_absolute",), default="cartesian_absolute")
    parser.add_argument("--proprio-mode", choices=("cartesian", "joint"), default=None)
    parser.add_argument("--proprio-dim", type=int, default=None)
    parser.add_argument("--train-action-hz", type=float, default=None)
    parser.add_argument("--gripper-min", type=float, default=None)
    parser.add_argument("--gripper-max", type=float, default=None)
    parser.add_argument("--gripper-index", type=int, default=-1)
    parser.add_argument("--camera-map", default=None)
    parser.add_argument("--model-client", choices=("hold", "fastwam"), default="hold")
    parser.add_argument("--model-gpus", default=None, help="Comma-separated physical GPU ids for FastWAM model workers.")
    parser.add_argument("--model-worker-start-method", choices=("spawn",), default="spawn")
    parser.add_argument("--model-worker-timeout-sec", type=float, default=600.0)
    parser.add_argument("--fastwam-config", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-stats", default=None)
    parser.add_argument("--text-cache-dir", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--model-action-mode", choices=("delta6_abs_gripper", "absolute"), default="delta6_abs_gripper")
    parser.add_argument("--output-action-format", choices=("cartesian_absolute",), default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sim-loop-frequency", type=float, default=0.0)
    parser.add_argument("--obs-interval", type=float, default=-1.0)
    parser.add_argument("--ignore-done", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    result = run(args)
    compact = {
        "profile": result["profile"]["name"],
        "task": result["task"],
        "output_dir": result["output_dir"],
        "episodes_completed": result["episodes_completed"],
        "success_rate": result["success_rate"],
        "summary_json": result["summary_json"],
        "results_csv": result["results_csv"],
    }
    print(json.dumps(to_jsonable(compact), indent=2))


if __name__ == "__main__":
    main()
