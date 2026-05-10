"""Local in-process client for AAO PolicyEvaluator."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AAO_ROOT = _REPO_ROOT / "third_party" / "auto-atomic-operation"
DEFAULT_GAUSSIAN_RENDERER_ROOT = _REPO_ROOT / "third_party" / "GaussianRenderer"


def _load_aao_modules(aao_root: Path):
    renderer_src = DEFAULT_GAUSSIAN_RENDERER_ROOT / "src"
    if renderer_src.exists():
        renderer_src_str = str(renderer_src)
        if renderer_src_str not in sys.path:
            sys.path.insert(0, renderer_src_str)

    root = Path(aao_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"AAO root does not exist: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    policy_eval = importlib.import_module("auto_atom.policy_eval")
    try:
        config_loader = importlib.import_module("auto_atom.config_loader")
        load_task_file_hydra = config_loader.load_task_file_hydra
    except (ImportError, AttributeError):
        runtime = importlib.import_module("auto_atom.runtime")
        load_task_file_hydra = runtime.load_task_file_hydra
    pose = importlib.import_module("auto_atom.utils.pose")
    return policy_eval.PolicyEvaluator, load_task_file_hydra, pose.euler_to_quaternion


def _default_action_applier(context: Any, action: Any, env_mask: np.ndarray | None = None) -> None:
    if action is None:
        return
    if isinstance(action, dict):
        context.backend.env.apply_pose_action(
            "arm",
            action["position"],
            action["orientation"],
            action["gripper"],
            env_mask=env_mask,
        )


def _default_observation_getter(context: Any) -> Any:
    return context.backend.env.capture_observation()


def _select_config_value(config: Any, path: str) -> Any:
    current = config
    for key in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


class SimulatorServiceClient:
    def __init__(
        self,
        *,
        aao_root: str | Path = DEFAULT_AAO_ROOT,
        gpu: int = 0,
        sim_loop_frequency: float = 0.0,
    ) -> None:
        self.aao_root = Path(aao_root).expanduser().resolve()
        self.config_dir = self.aao_root / "aao_configs"
        self.assets_dir = self.aao_root / "assets"
        self.gpu = int(gpu)
        self.sim_loop_frequency = float(sim_loop_frequency)
        self.evaluator: Any | None = None
        self._load_task_file_hydra: Any | None = None
        self._euler_to_quaternion: Any | None = None
        self.info: dict[str, Any] | None = None
        self.task_name = ""

    def connect(self) -> dict[str, Any]:
        python_bin = str(Path(sys.prefix) / "bin")
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if python_bin not in path_parts:
            os.environ["PATH"] = os.pathsep.join([python_bin, *path_parts])
        os.environ.setdefault("MUJOCO_GL", "egl")
        if self.gpu >= 0:
            gpu_str = str(self.gpu)
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_str)
            os.environ.setdefault("EGL_VISIBLE_DEVICES", gpu_str)
            os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", gpu_str)
        policy_evaluator, load_task_file_hydra, euler_to_quaternion = _load_aao_modules(self.aao_root)
        self._load_task_file_hydra = load_task_file_hydra
        self._euler_to_quaternion = euler_to_quaternion
        self.evaluator = policy_evaluator(
            action_applier=_default_action_applier,
            observation_getter=_default_observation_getter,
        )
        return {"status": "ok", "mode": "local", "aao_root": str(self.aao_root)}

    def init(
        self,
        *,
        config_name: str,
        overrides: list[str],
        action_format: str = "cartesian_absolute",
    ) -> dict[str, Any]:
        if action_format != "cartesian_absolute":
            raise ValueError("Only cartesian_absolute action_format is supported.")
        evaluator = self._require_evaluator()
        load_task_file_hydra = self._require_load_task_file_hydra()
        resolved_overrides = self._normalize_overrides(overrides)
        task_file = load_task_file_hydra(
            config_name,
            config_dir=self.config_dir,
            overrides=resolved_overrides,
        )
        evaluator.from_config(task_file, sim_loop_frequency=self.sim_loop_frequency)
        self.info = dict(evaluator._require_context().backend.env.get_info())
        update_freq = _select_config_value(task_file, "env.update_freq")
        if update_freq is not None:
            self.info["env_update_freq"] = int(update_freq)
        self.task_name = str(config_name)
        return {
            "status": "ok",
            "config_name": config_name,
            "overrides": resolved_overrides,
            "action_format": action_format,
            "info": self.info,
        }

    def reset(self) -> Any:
        return self._require_evaluator().reset()

    def get_observation(self) -> dict[str, dict[str, Any]]:
        observation = self._require_evaluator().get_observation()
        if not isinstance(observation, dict):
            raise RuntimeError(f"Expected AAO observation dict, got {type(observation).__name__}.")
        return observation

    def get_task_state(self) -> dict[str, Any]:
        task_update = self._require_evaluator()._build_task_update()
        return {
            "done": np.asarray(task_update.done, dtype=bool),
            "success": task_update.success,
        }

    def summarize(
        self,
        *,
        max_updates: int | None,
        updates_used: int,
        elapsed_time_sec: float,
    ) -> Any:
        return self._require_evaluator().summarize(
            max_updates=max_updates,
            updates_used=updates_used,
            elapsed_time_sec=elapsed_time_sec,
        )

    @property
    def records(self) -> list[Any]:
        return list(self._require_evaluator().records)

    @property
    def stage_plans(self) -> list[dict[str, Any]]:
        plans = self._require_evaluator().stage_plans
        return [
            {
                "stage_name": plan.stage_name,
                "operator_name": plan.operator_name,
                "operation": plan.stage.operation.value,
                "object": plan.stage.object,
            }
            for plan in plans
        ]

    @property
    def batch_size(self) -> int:
        ctx = self._require_evaluator()._require_context()
        return int(ctx.backend.batch_size)

    def update_cartesian_action(self, action_row: np.ndarray) -> tuple[Any, dict[str, np.ndarray]]:
        return self.update_cartesian_actions(np.asarray(action_row, dtype=np.float32).reshape(1, -1))

    def update_cartesian_actions(
        self,
        action_rows: np.ndarray,
        env_mask: np.ndarray | None = None,
    ) -> tuple[Any, dict[str, np.ndarray]]:
        evaluator = self._require_evaluator()
        batch_size = self.batch_size
        mask = self._normalize_env_mask(batch_size, env_mask)
        remote_action = self._expand_encoded_action(
            self._encode_action(action_rows),
            batch_size=batch_size,
            env_mask=mask,
        )
        update = evaluator.update(remote_action, env_mask=mask)
        return update, remote_action

    def set_cartesian_action(self, action_row: np.ndarray) -> dict[str, np.ndarray]:
        return self.set_cartesian_actions(np.asarray(action_row, dtype=np.float32).reshape(1, -1))

    def set_cartesian_actions(
        self,
        action_rows: np.ndarray,
        env_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        evaluator = self._require_evaluator()
        batch_size = self.batch_size
        mask = self._normalize_env_mask(batch_size, env_mask)
        remote_action = self._expand_encoded_action(
            self._encode_action(action_rows),
            batch_size=batch_size,
            env_mask=mask,
        )
        with evaluator.sim_lock:
            ctx = evaluator._require_context()
            ctx.backend.env.apply_pose_action(
                "arm",
                remote_action["position"],
                remote_action["orientation"],
                remote_action["gripper"],
                env_mask=mask,
            )
        return remote_action

    def close(self) -> None:
        if self.evaluator is not None:
            self.evaluator.close()
            self.evaluator = None

    def _encode_action(self, action_rows: np.ndarray) -> dict[str, np.ndarray]:
        action_array = np.asarray(action_rows, dtype=np.float32)
        if action_array.ndim == 1:
            action_array = action_array.reshape(1, -1)
        if action_array.ndim != 2 or action_array.shape[1] < 7:
            raise ValueError(
                "cartesian_absolute action must have shape (T, 7+) with "
                "[x, y, z, roll, pitch, yaw, gripper]."
            )
        euler_to_quaternion = self._require_euler_to_quaternion()
        orientation = np.asarray(
            [
                euler_to_quaternion(tuple(float(v) for v in row[3:6]))
                for row in action_array
            ],
            dtype=np.float32,
        )
        return {
            "position": np.asarray(action_array[:, :3], dtype=np.float32),
            "orientation": orientation,
            "gripper": np.asarray(action_array[:, 6:7], dtype=np.float32),
        }

    @staticmethod
    def _normalize_env_mask(batch_size: int, env_mask: np.ndarray | None) -> np.ndarray:
        if env_mask is None:
            return np.ones(batch_size, dtype=bool)
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        if mask.shape != (batch_size,):
            raise ValueError(f"env_mask must have shape ({batch_size},), got {mask.shape}.")
        return mask

    @classmethod
    def _expand_encoded_action(
        cls,
        encoded: dict[str, np.ndarray],
        *,
        batch_size: int,
        env_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        selected = int(env_mask.sum())

        def _expand(value: np.ndarray, trailing_shape: tuple[int, ...]) -> np.ndarray:
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == len(trailing_shape):
                array = array.reshape((1,) + trailing_shape)
            if array.shape[1:] != trailing_shape:
                raise ValueError(f"Expected trailing shape {trailing_shape}, got {array.shape}.")
            if array.shape[0] == batch_size:
                return array
            if array.shape[0] == 1 and selected > 1:
                array = np.repeat(array, selected, axis=0)
            if array.shape[0] != selected:
                raise ValueError(f"Action rows must match selected envs={selected}, got {array.shape[0]}.")
            full = np.zeros((batch_size,) + trailing_shape, dtype=np.float32)
            full[env_mask] = array
            return full

        return {
            "position": _expand(encoded["position"], (3,)),
            "orientation": _expand(encoded["orientation"], (4,)),
            "gripper": _expand(encoded["gripper"], (1,)),
        }

    def _normalize_overrides(self, overrides: list[str]) -> list[str]:
        resolved = list(overrides)
        if not any(item.startswith("env.batch_size=") for item in resolved):
            resolved.append("env.batch_size=1")
        if not any("env.viewer.disable" in item for item in resolved):
            resolved.append("++env.viewer.disable=true")
        if not any(item.startswith("assets_dir=") for item in resolved):
            resolved.append(f"assets_dir={self.assets_dir}")
        return resolved

    def _require_evaluator(self) -> Any:
        if self.evaluator is None:
            raise RuntimeError("SimulatorServiceClient is not connected.")
        return self.evaluator

    def _require_load_task_file_hydra(self) -> Any:
        if self._load_task_file_hydra is None:
            raise RuntimeError("AAO runtime module has not been loaded.")
        return self._load_task_file_hydra

    def _require_euler_to_quaternion(self) -> Any:
        if self._euler_to_quaternion is None:
            raise RuntimeError("AAO pose module has not been loaded.")
        return self._euler_to_quaternion

    def __enter__(self) -> "SimulatorServiceClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
