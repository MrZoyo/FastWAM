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


def _coerce_rgb(value: Any, fallback_hw: tuple[int, int]) -> np.ndarray:
    if value is None:
        height, width = fallback_hw
        return np.zeros((height, width, 3), dtype=np.uint8)
    rgb = np.asarray(value)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    return rgb.astype(np.uint8, copy=False)


def _coerce_vector(value: Any, dim: int, *, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full((dim,), default, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < dim:
        array = np.pad(array, (0, dim - array.size), constant_values=default)
    return array[:dim].astype(np.float32, copy=False)


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
    return "arm", "arm", "eef"


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
            fallback_hw = self._camera_hw(camera_name)
            rgb = _coerce_rgb(
                _extract_data(observation, f"{camera_name}/color/image_raw"),
                fallback_hw=fallback_hw,
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
            "arm_joint_position": _extract_data(observation, f"{arm}/joint_state/position"),
            "eef_joint_position": _extract_data(observation, f"{eef}/joint_state/position"),
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

    def build_model_input(self, camera_map: dict[str, str]) -> dict[str, Any]:
        frame = self.latest_frame()
        missing = [sim_name for sim_name in camera_map.values() if sim_name not in frame.cameras]
        if missing:
            raise RuntimeError(f"Camera map references unavailable AAO cameras: {missing}")

        cartesian = self.current_cartesian_position(frame.robot_state)
        gripper = self.current_gripper_position(frame.robot_state)
        images = {
            train_key: frame.cameras[sim_key]["rgb"]
            for train_key, sim_key in camera_map.items()
        }
        return {
            "timestamp_ns": frame.timestamp_ns,
            "camera_map": dict(camera_map),
            "images": images,
            "cartesian_position": cartesian,
            "gripper_position": gripper,
            "proprio_raw": np.concatenate([cartesian, gripper], axis=0).astype(np.float32),
            "sim_frame": frame,
        }

    def _camera_hw(self, camera_name: str) -> tuple[int, int]:
        info = self.sim_info.get("cameras", {}).get(camera_name, {})
        color_info = info.get("camera_info", {}).get("color", {})
        depth_info = info.get("camera_info", {}).get("depth", {})
        height = int(color_info.get("height", depth_info.get("height", 480)))
        width = int(color_info.get("width", depth_info.get("width", 640)))
        return height, width

    def _camera_intrinsics(self, camera_name: str) -> np.ndarray:
        info = self.sim_info.get("cameras", {}).get(camera_name, {})
        color_k = info.get("camera_info", {}).get("color", {}).get("k")
        depth_k = info.get("camera_info", {}).get("depth", {}).get("k")
        k = color_k if color_k is not None else depth_k
        if k is None:
            return np.zeros((3, 3), dtype=np.float32)
        return np.asarray(k, dtype=np.float32).reshape(3, 3)

    def _camera_extrinsics(self, camera_name: str) -> np.ndarray:
        info = self.sim_info.get("cameras", {}).get(camera_name, {})
        extr = info.get("camera_extrinsics")
        matrix = np.eye(4, dtype=np.float32)
        if not extr:
            return matrix
        matrix[:3, :3] = np.asarray(
            extr.get("rotation_matrix", np.eye(3)),
            dtype=np.float32,
        ).reshape(3, 3)
        matrix[:3, 3] = np.asarray(
            extr.get("translation", np.zeros(3)),
            dtype=np.float32,
        ).reshape(3)
        return matrix

    @staticmethod
    def current_cartesian_position(robot_state: dict[str, Any]) -> np.ndarray:
        position = _coerce_vector(robot_state.get("eef_position"), 3)
        rotation = _coerce_vector(robot_state.get("eef_rotation_rpy"), 3)
        return np.concatenate([position, rotation], axis=0).astype(np.float32)

    @staticmethod
    def current_gripper_position(robot_state: dict[str, Any]) -> np.ndarray:
        return _coerce_vector(robot_state.get("eef_joint_position"), 1)
