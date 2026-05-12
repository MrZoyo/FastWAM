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
from omegaconf import OmegaConf

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
_DEFAULT_PROFILE_DIR = _REPO_ROOT / "configs" / "aao_benchmark"


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


DEFAULT_PROFILE_FILES: dict[str, Path] = {
    "open_door_airbot_play_gs": _DEFAULT_PROFILE_DIR / "open_door_airbot_play_gs.yaml",
    "cup_on_coaster_gs_airbot_p7": _DEFAULT_PROFILE_DIR / "cup_on_coaster_gs_airbot_p7.yaml",
}


DEFAULT_SENSOR_OVERRIDES = (
    "enable_depth=false",
    "enable_mask=false",
    "enable_heat_map=false",
)


def _profile_names() -> tuple[str, ...]:
    return tuple(sorted(DEFAULT_PROFILE_FILES))


def _resolve_repo_path(value: str | None, *, base_dir: Path | None = None) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    if base_dir is not None:
        candidate = (base_dir / path).resolve()
        if candidate.exists():
            return str(candidate)
    return str((_REPO_ROOT / path).resolve())


def _load_profile_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (_REPO_ROOT / resolved).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Benchmark profile config does not exist: {resolved}")
    payload = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Benchmark profile config must be a mapping: {resolved}")
    payload = dict(payload)
    payload.setdefault("name", resolved.stem)
    payload["fastwam_config"] = _resolve_repo_path(payload.get("fastwam_config"), base_dir=resolved.parent)
    payload["text_cache_dir"] = _resolve_repo_path(payload.get("text_cache_dir"), base_dir=resolved.parent)
    return payload


def _profile_from_dict(payload: dict[str, Any], *, source: Path | str) -> BenchmarkProfile:
    required = ("name", "task", "instruction", "camera_map", "action_repeat", "train_action_hz", "max_updates")
    missing = [key for key in required if payload.get(key) is None]
    if missing:
        raise RuntimeError(f"Benchmark profile {source} is missing required field(s): {', '.join(missing)}")
    proprio_mode = str(payload.get("proprio_mode", "cartesian"))
    if proprio_mode not in {"cartesian", "joint"}:
        raise RuntimeError(f"Benchmark profile {source} has unsupported proprio_mode={proprio_mode!r}.")
    proprio_dim = payload.get("proprio_dim")
    return BenchmarkProfile(
        name=str(payload["name"]),
        task=str(payload["task"]),
        instruction=str(payload["instruction"]),
        camera_map=str(payload["camera_map"]),
        action_repeat=int(payload["action_repeat"]),
        train_action_hz=float(payload["train_action_hz"]),
        max_updates=int(payload["max_updates"]),
        proprio_mode=proprio_mode,
        proprio_dim=None if proprio_dim is None else int(proprio_dim),
        fastwam_config=None if payload.get("fastwam_config") is None else str(payload["fastwam_config"]),
        text_cache_dir=None if payload.get("text_cache_dir") is None else str(payload["text_cache_dir"]),
    )


def _load_profile_file(path: str | Path) -> BenchmarkProfile:
    return _profile_from_dict(_load_profile_yaml(path), source=path)


def _load_named_profile(name: str) -> BenchmarkProfile:
    try:
        path = DEFAULT_PROFILE_FILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark profile '{name}'. Expected one of {sorted(DEFAULT_PROFILE_FILES)}.") from exc
    return _load_profile_file(path)


def _parse_model_gpus(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _resolve_profile(args: argparse.Namespace) -> BenchmarkProfile:
    profile = _load_profile_file(args.profile_config) if args.profile_config is not None else _load_named_profile(args.profile)
    values = {
        "task": args.task if args.task is not None else profile.task,
        "instruction": args.instruction if args.instruction is not None else profile.instruction,
        "camera_map": args.camera_map if args.camera_map is not None else profile.camera_map,
        "action_repeat": int(args.action_repeat if args.action_repeat is not None else profile.action_repeat),
        "train_action_hz": float(args.train_action_hz if args.train_action_hz is not None else profile.train_action_hz),
        "max_updates": int(args.max_updates if args.max_updates is not None else profile.max_updates),
        "proprio_mode": args.proprio_mode if args.proprio_mode is not None else profile.proprio_mode,
        "proprio_dim": args.proprio_dim if args.proprio_dim is not None else profile.proprio_dim,
        "fastwam_config": args.fastwam_config if args.fastwam_config is not None else profile.fastwam_config,
        "text_cache_dir": args.text_cache_dir if args.text_cache_dir is not None else profile.text_cache_dir,
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
    stage_index: int | None = None
    stage_name: str | None = None
    phase: str | None = None
    phase_step: int | None = None
    task_details: Any = None


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


def _require_vector_dim(values: np.ndarray, dim: int | None, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if dim is None:
        if array.size == 0:
            raise RuntimeError(f"{name} must not be empty.")
        return array
    if array.size != int(dim):
        raise RuntimeError(f"{name} expected exactly {int(dim)} values, got {array.size}.")
    return array.astype(np.float32, copy=False)


def _prepare_model_input(
    adapter: AAOObservationAdapter,
    *,
    camera_map: dict[str, str],
    profile: BenchmarkProfile,
) -> dict[str, Any]:
    model_input = adapter.build_model_input(camera_map, proprio_mode=profile.proprio_mode)
    model_input["proprio_raw"] = _require_vector_dim(
        model_input["proprio_raw"],
        profile.proprio_dim,
        name=f"profile '{profile.name}' proprio_raw",
    )
    model_input["instruction"] = profile.instruction
    return model_input


def _required_array(value: Any, *, dtype: Any, name: str, batch_size: int) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"AAO task update is missing required field '{name}'.")
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.size != batch_size:
        raise RuntimeError(f"AAO task update field '{name}' expected {batch_size} values, got {array.size}.")
    return array


def _optional_items(value: Any, *, name: str, batch_size: int) -> list[Any]:
    if value is None:
        return [None] * batch_size
    if isinstance(value, np.ndarray):
        items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]
    if len(items) != batch_size:
        raise RuntimeError(f"AAO task update field '{name}' expected {batch_size} values, got {len(items)}.")
    return list(items)


def _optional_array(value: Any, *, dtype: Any, name: str, batch_size: int) -> np.ndarray | None:
    if value is None:
        return None
    return _required_array(value, dtype=dtype, name=name, batch_size=batch_size)


def _task_update_source(update: Any, task_state: dict[str, Any], name: str) -> Any:
    value = getattr(update, name, None) if update is not None else None
    if value is None:
        value = task_state.get(name)
    return value


def _extract_task_update(
    update: Any,
    task_state: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    status_items = _optional_items(
        _task_update_source(update, task_state, "status"),
        name="status",
        batch_size=batch_size,
    )
    stage_name_items = _optional_items(
        _task_update_source(update, task_state, "stage_name"),
        name="stage_name",
        batch_size=batch_size,
    )
    details_items = _optional_items(
        _task_update_source(update, task_state, "details"),
        name="details",
        batch_size=batch_size,
    )
    phase_items = _optional_items(
        _task_update_source(update, task_state, "phase"),
        name="phase",
        batch_size=batch_size,
    )
    return {
        "done": _required_array(
            _task_update_source(update, task_state, "done"),
            dtype=bool,
            name="done",
            batch_size=batch_size,
        ),
        "success": _required_array(
            _task_update_source(update, task_state, "success"),
            dtype=object,
            name="success",
            batch_size=batch_size,
        ),
        "status": [None if item is None else str(getattr(item, "value", item)) for item in status_items],
        "stage_index": _optional_array(
            _task_update_source(update, task_state, "stage_index"),
            dtype=np.int64,
            name="stage_index",
            batch_size=batch_size,
        ),
        "stage_name": [None if item is None else str(item) for item in stage_name_items],
        "details": details_items,
        "phase": [None if item is None else str(item) for item in phase_items],
        "phase_step": _optional_array(
            _task_update_source(update, task_state, "phase_step"),
            dtype=np.int64,
            name="phase_step",
            batch_size=batch_size,
        ),
    }


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
        "stage_index": state.stage_index,
        "stage_name": state.stage_name,
        "phase": state.phase,
        "phase_step": state.phase_step,
        "task_details": state.task_details,
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
        "stage_index",
        "stage_name",
        "phase",
        "phase_step",
        "task_details",
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
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray, np.generic)):
        return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _append_row_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def _persist_rows(output_root: Path, rows: list[dict[str, Any]], row: dict[str, Any] | None = None) -> None:
    if row is not None:
        _append_row_jsonl(output_root / "benchmark_results.jsonl", row)
    _write_rows_csv(output_root / "benchmark_results.csv", rows)


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
    env_state.stage_index = None
    env_state.stage_name = None
    env_state.phase = None
    env_state.phase_step = None
    env_state.task_details = None


def _complete_episode(
    *,
    state: EnvEpisodeState,
    rows: list[dict[str, Any]],
    profile: BenchmarkProfile,
    args: argparse.Namespace,
    error: str | None = None,
    output_root: Path | None = None,
) -> None:
    if state.finished:
        return
    if error is not None:
        state.error = error
    state.finished = True
    state.end_perf = time.perf_counter()
    row = _row_for_episode(state=state, profile=profile, args=args)
    rows.append(row)
    if output_root is not None:
        _persist_rows(output_root, rows, row=row)
    state.episode_index = None


def _record_task_update(
    *,
    state: EnvEpisodeState,
    done: bool,
    success: Any,
    status: str | None,
    stage_index: int | None,
    stage_name: str | None,
    phase: str | None,
    phase_step: int | None,
    details: Any,
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
    state.stage_index = stage_index
    state.stage_name = stage_name
    state.phase = phase
    state.phase_step = phase_step
    state.task_details = to_jsonable(details)


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
    output_root: Path,
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
            state.updates_used = 0
            state.model_steps_used = 0
            state.chunk_index = 0
            state.chunk_action_index = 0
            state.chunk_actions = None
            state.last_action = None
            state.start_perf = time.perf_counter()
            state.end_perf = None
            state.done = False
            state.success = False
            state.status = None
            state.error = None
            state.model_infer_calls = 0
            state.model_infer_time_sec = 0.0
            state.sim_update_time_sec = 0.0
            state.first_done_update = None
            state.model_worker_indices = []
            state.model_worker_gpus = []
            state.stage_index = None
            state.stage_name = None
            state.phase = None
            state.phase_step = None
            state.task_details = None
            _complete_episode(
                state=state,
                rows=rows,
                profile=profile,
                args=args,
                error=error,
                output_root=output_root,
            )
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
    output_root: Path,
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
            _complete_episode(
                state=state,
                rows=rows,
                profile=profile,
                args=args,
                error=error,
                output_root=output_root,
            )
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
    if args.sim_loop_frequency < 0:
        raise ValueError("--sim-loop-frequency must be non-negative.")
    if args.sim_loop_frequency > 0:
        raise ValueError(
            "Batch AAO benchmark currently supports lockstep mode only. "
            "Use --sim-loop-frequency 0 so update counts and per-env episode accounting stay exact."
        )
    if args.model_worker_timeout_sec <= 0:
        raise ValueError("--model-worker-timeout-sec must be positive.")

    profile = _resolve_profile(args)
    if profile.action_repeat <= 0:
        raise ValueError("--action-repeat/profile.action_repeat must be positive.")
    if profile.max_updates <= 0:
        raise ValueError("--max-updates/profile.max_updates must be positive.")
    if profile.train_action_hz <= 0:
        raise ValueError("--train-action-hz/profile.train_action_hz must be positive.")
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
    for stale_name in ("benchmark_results.jsonl", "benchmark_results.csv", "benchmark_summary.json"):
        stale_path = output_root / stale_name
        if stale_path.exists():
            stale_path.unlink()

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
            output_root=output_root,
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
                    output_root=output_root,
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
                output_root=output_root,
            )
            active = _active_states(env_states)
            if not active:
                continue

            stepping_states: list[EnvEpisodeState] = []
            for state in active:
                if state.updates_used >= profile.max_updates:
                    state.status = state.status or "max_updates"
                    _complete_episode(
                        state=state,
                        rows=rows,
                        profile=profile,
                        args=args,
                        output_root=output_root,
                    )
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
                    output_root=output_root,
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
                task_update = _extract_task_update(update, task_state, sim.batch_size)
                for state in list(current_states):
                    env_index = state.env_index
                    done = bool(task_update["done"][env_index])
                    success = task_update["success"][env_index]
                    status = task_update["status"][env_index]
                    stage_index_array = task_update["stage_index"]
                    phase_step_array = task_update["phase_step"]
                    _record_task_update(
                        state=state,
                        done=done,
                        success=success,
                        status=status,
                        stage_index=None if stage_index_array is None else int(stage_index_array[env_index]),
                        stage_name=task_update["stage_name"][env_index],
                        phase=task_update["phase"][env_index],
                        phase_step=None if phase_step_array is None else int(phase_step_array[env_index]),
                        details=task_update["details"][env_index],
                    )
                    if (done and not args.ignore_done) or state.updates_used >= profile.max_updates:
                        if done or _success_to_bool(success):
                            state.success = success
                        if state.updates_used >= profile.max_updates and not state.done:
                            state.status = "max_updates"
                        _complete_episode(
                            state=state,
                            rows=rows,
                            profile=profile,
                            args=args,
                            output_root=output_root,
                        )
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
                output_root=output_root,
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
    except Exception as exc:
        if rows:
            partial_payload = _build_payload(
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
            partial_payload["incomplete"] = True
            partial_payload["run_error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            _write_summary(output_root=output_root, payload=partial_payload, rows=rows)
        raise
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
        "incomplete": False,
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
        "results_jsonl": str(output_root / "benchmark_results.jsonl"),
        "summary_json": str(output_root / "benchmark_summary.json"),
        "episodes": rows,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aao-root", default=str(DEFAULT_AAO_ROOT))
    parser.add_argument("--profile", choices=_profile_names(), default="open_door_airbot_play_gs")
    parser.add_argument(
        "--profile-config",
        default=None,
        help="Path to a benchmark profile YAML. Overrides --profile when provided.",
    )
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
        "results_jsonl": result["results_jsonl"],
    }
    print(json.dumps(to_jsonable(compact), indent=2))


if __name__ == "__main__":
    main()
