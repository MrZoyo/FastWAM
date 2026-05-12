"""Adapters from AAO observations to FastWAM closed-loop inputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

import numpy as np


def _squeeze_single_env(data: Any) -> Any:
    if isinstance(data, np.ndarray) and data.shape[:1] == (1,):
        return data[0]
    if isinstance(data, list) and len(data) == 1:
        return data[0]
    return data


def _slice_batched_value(value: Any, env_index: int, batch_size: int) -> Any:
    if isinstance(value, np.ndarray) and value.shape[:1] == (batch_size,):
        return value[env_index]
    if isinstance(value, list) and len(value) == batch_size:
        return value[env_index]
    if isinstance(value, tuple) and len(value) == batch_size:
        return value[env_index]
    return value


def _extract_data(observation: dict[str, dict[str, Any]], key: str) -> Any | None:
    payload = observation.get(key)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return None
    return _squeeze_single_env(payload.get("data"))


def _infer_timestamp_ns(observation: dict[str, dict[str, Any]]) -> float:
    for payload in observation.values():
        if not isinstance(payload, dict):
            continue
        value = payload.get("t")
        if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
            return float(value[0])
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, list) and value:
            return float(value[0])
    return 0.0


def _coerce_rgb(value: Any, *, camera_name: str) -> np.ndarray:
    if value is None:
        raise RuntimeError(f"Missing RGB observation for AAO camera '{camera_name}'.")
    rgb = np.asarray(value)
    if rgb.ndim != 3:
        raise RuntimeError(
            f"Expected RGB observation for AAO camera '{camera_name}' to have shape [H,W,3], got {rgb.shape}."
        )
    if rgb.shape[-1] != 3:
        raise RuntimeError(
            f"Expected RGB observation for AAO camera '{camera_name}' to have 3 channels, got shape {rgb.shape}."
        )
    return rgb.astype(np.uint8, copy=False)


def _coerce_vector(value: Any, dim: int | None, *, name: str) -> np.ndarray:
    if value is None:
        expected = "non-empty" if dim is None else f"{dim}D"
        raise RuntimeError(f"Missing {name}; expected {expected} vector from AAO observation.")
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if dim is None:
        if array.size == 0:
            raise RuntimeError(f"AAO observation field '{name}' is empty.")
        return array.astype(np.float32, copy=False)
    if array.size != dim:
        raise RuntimeError(f"AAO observation field '{name}' expected {dim} values, got {array.size}.")
    return array.astype(np.float32, copy=False)


def _resolve_operator_names(sim_info: dict[str, Any]) -> tuple[str, str, str]:
    operators = sim_info.get("operators")
    if isinstance(operators, dict) and operators:
        operator_name, op_cfg = next(iter(operators.items()))
        arm_output = op_cfg.get("arm_output_name") or operator_name
        eef_output = op_cfg.get("eef_output_name", "eef")
        return str(operator_name), str(arm_output), str(eef_output)
    if isinstance(operators, list) and operators:
        op_cfg = operators[0]
        operator_name = op_cfg.get("name", "arm")
        arm_output = op_cfg.get("arm_output_name") or operator_name
        eef_output = op_cfg.get("eef_output_name", "eef")
        return str(operator_name), str(arm_output), str(eef_output)
    raise RuntimeError("AAO sim_info does not contain operator metadata.")


def _resolve_operator_joint_dims(sim_info: dict[str, Any], operator_name: str) -> tuple[int | None, int | None]:
    operators = sim_info.get("operators")
    op_cfg: dict[str, Any] | None = None
    if isinstance(operators, dict):
        candidate = operators.get(operator_name)
        if isinstance(candidate, dict):
            op_cfg = candidate
    elif isinstance(operators, list):
        for candidate in operators:
            if isinstance(candidate, dict) and candidate.get("name", operator_name) == operator_name:
                op_cfg = candidate
                break
    if op_cfg is None:
        return None, None
    arm = op_cfg.get("arm_actuators")
    eef = op_cfg.get("eef_actuators")
    return (
        len(arm) if isinstance(arm, list) else None,
        len(eef) if isinstance(eef, list) else None,
    )


def _extract_joint_position(observation: dict[str, dict[str, Any]], limb: str) -> Any | None:
    value = _extract_data(observation, f"{limb}/joint_state/position")
    if value is not None:
        return value
    payload = observation.get(f"{limb}/joint_state")
    if not isinstance(payload, dict):
        return None
    data = _squeeze_single_env(payload.get("data"))
    if isinstance(data, dict):
        return data.get("position")
    return None


def split_batched_observation(
    observation: dict[str, dict[str, Any]],
    batch_size: int,
) -> list[dict[str, dict[str, Any]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    split: list[dict[str, dict[str, Any]]] = [{} for _ in range(batch_size)]
    for key, payload in observation.items():
        if not isinstance(payload, dict):
            continue
        for env_index in range(batch_size):
            split[env_index][key] = {
                field: _slice_batched_value(value, env_index, batch_size)
                for field, value in payload.items()
            }
    return split


@dataclass
class SimFrame:
    timestamp_ns: float
    cameras: dict[str, dict[str, Any]]
    robot_state: dict[str, Any]
    raw_observation: dict[str, dict[str, Any]]


class AAOObservationAdapter:
    """Keep recent AAO frames and build model-facing FastWAM payloads."""

    def __init__(
        self,
        sim_info: dict[str, Any],
        *,
        selected_cameras: list[str],
        history_frames: int = 1,
    ) -> None:
        self.sim_info = sim_info
        self.selected_cameras = list(selected_cameras)
        if not self.selected_cameras:
            raise ValueError("selected_cameras must not be empty.")
        self.history_frames = max(int(history_frames), 1)
        self.operator_name, self.arm_output_name, self.eef_output_name = _resolve_operator_names(sim_info)
        self.arm_joint_dim, self.eef_joint_dim = _resolve_operator_joint_dims(sim_info, self.operator_name)
        self.frames: Deque[SimFrame] = deque(maxlen=self.history_frames)

    def reset(self) -> None:
        self.frames.clear()

    def extend(self, observations: list[dict[str, dict[str, Any]]]) -> None:
        for observation in observations:
            self.frames.append(self.parse_observation(observation))

    def latest_frame(self) -> SimFrame:
        if not self.frames:
            raise RuntimeError("No AAO observations have been parsed yet.")
        return self.frames[-1]

    def parse_observation(self, observation: dict[str, dict[str, Any]]) -> SimFrame:
        cameras: dict[str, dict[str, Any]] = {}
        for camera_name in self.selected_cameras:
            rgb = _coerce_rgb(
                _extract_data(observation, f"{camera_name}/color/image_raw"),
                camera_name=camera_name,
            )
            cameras[camera_name] = {
                "rgb": rgb,
                "depth_m": _extract_data(
                    observation,
                    f"{camera_name}/aligned_depth_to_color/image_raw",
                ),
                "mask": _extract_data(observation, f"{camera_name}/mask/image_raw"),
                "intrinsics": self._camera_intrinsics(camera_name),
                "extrinsics": self._camera_extrinsics(camera_name),
            }

        op = self.operator_name
        arm = self.arm_output_name
        eef = self.eef_output_name
        robot_state = {
            "arm_joint_position": _extract_joint_position(observation, arm),
            "eef_joint_position": _extract_joint_position(observation, eef),
            "eef_position": _extract_data(observation, f"{op}/pose/position"),
            "eef_orientation_xyzw": _extract_data(observation, f"{op}/pose/orientation"),
            "eef_rotation_rpy": _extract_data(observation, f"{op}/pose/rotation"),
            "target_eef_position": _extract_data(observation, f"action/{op}/pose/position"),
            "target_eef_orientation_xyzw": _extract_data(observation, f"action/{op}/pose/orientation"),
        }
        return SimFrame(
            timestamp_ns=_infer_timestamp_ns(observation),
            cameras=cameras,
            robot_state=robot_state,
            raw_observation=observation,
        )

    def build_model_input(self, camera_map: dict[str, str], *, proprio_mode: str = "cartesian") -> dict[str, Any]:
        frame = self.latest_frame()
        missing = [sim_name for sim_name in camera_map.values() if sim_name not in frame.cameras]
        if missing:
            raise RuntimeError(f"Camera map references unavailable AAO cameras: {missing}")

        cartesian = self.current_cartesian_position(frame.robot_state)
        gripper = self.current_gripper_position(frame.robot_state)
        joint = self.current_joint_position(frame.robot_state)
        if proprio_mode == "cartesian":
            proprio_raw = np.concatenate([cartesian, gripper], axis=0).astype(np.float32)
        elif proprio_mode == "joint":
            proprio_raw = joint.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported proprio_mode='{proprio_mode}'.")
        images = {
            train_key: frame.cameras[sim_key]["rgb"]
            for train_key, sim_key in camera_map.items()
        }
        return {
            "timestamp_ns": frame.timestamp_ns,
            "camera_map": dict(camera_map),
            "images": images,
            "cartesian_position": cartesian,
            "joint_position": joint,
            "gripper_position": gripper,
            "proprio_mode": proprio_mode,
            "proprio_raw": proprio_raw,
            "sim_frame": frame,
        }

    def _camera_intrinsics(self, camera_name: str) -> np.ndarray:
        info = self.sim_info.get("cameras", {}).get(camera_name)
        if not isinstance(info, dict):
            raise RuntimeError(f"AAO sim_info is missing camera metadata for '{camera_name}'.")
        color_k = info.get("camera_info", {}).get("color", {}).get("k")
        depth_k = info.get("camera_info", {}).get("depth", {}).get("k")
        k = color_k if color_k is not None else depth_k
        if k is None:
            raise RuntimeError(f"AAO sim_info camera '{camera_name}' is missing color/depth intrinsics.")
        matrix = np.asarray(k, dtype=np.float32)
        if matrix.size != 9:
            raise RuntimeError(f"AAO sim_info camera '{camera_name}' intrinsics expected 9 values, got {matrix.size}.")
        return matrix.reshape(3, 3)

    def _camera_extrinsics(self, camera_name: str) -> np.ndarray:
        info = self.sim_info.get("cameras", {}).get(camera_name)
        if not isinstance(info, dict):
            raise RuntimeError(f"AAO sim_info is missing camera metadata for '{camera_name}'.")
        extr = info.get("camera_extrinsics")
        if not isinstance(extr, dict):
            raise RuntimeError(f"AAO sim_info camera '{camera_name}' is missing extrinsics.")
        if "rotation_matrix" not in extr or "translation" not in extr:
            raise RuntimeError(f"AAO sim_info camera '{camera_name}' extrinsics must include rotation_matrix and translation.")
        rotation = np.asarray(extr["rotation_matrix"], dtype=np.float32)
        translation = np.asarray(extr["translation"], dtype=np.float32)
        if rotation.size != 9:
            raise RuntimeError(f"AAO sim_info camera '{camera_name}' rotation_matrix expected 9 values, got {rotation.size}.")
        if translation.size != 3:
            raise RuntimeError(f"AAO sim_info camera '{camera_name}' translation expected 3 values, got {translation.size}.")
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation.reshape(3, 3)
        matrix[:3, 3] = translation.reshape(3)
        return matrix

    @staticmethod
    def current_cartesian_position(robot_state: dict[str, Any]) -> np.ndarray:
        position = _coerce_vector(robot_state.get("eef_position"), 3, name="eef_position")
        rotation = _coerce_vector(robot_state.get("eef_rotation_rpy"), 3, name="eef_rotation_rpy")
        return np.concatenate([position, rotation], axis=0).astype(np.float32)

    @staticmethod
    def current_gripper_position(robot_state: dict[str, Any]) -> np.ndarray:
        return _coerce_vector(robot_state.get("eef_joint_position"), 1, name="eef_joint_position")

    def current_joint_position(self, robot_state: dict[str, Any]) -> np.ndarray:
        arm = _coerce_vector(robot_state.get("arm_joint_position"), self.arm_joint_dim, name="arm_joint_position")
        eef = _coerce_vector(robot_state.get("eef_joint_position"), self.eef_joint_dim, name="eef_joint_position")
        return np.concatenate([arm, eef], axis=0).astype(np.float32)
