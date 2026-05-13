from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fastwam.closed_loop_eval.benchmark import (
    _extract_task_update,
    _load_named_profile,
    _load_profile_file,
    _resolve_profile,
    _require_vector_dim,
)
from fastwam.closed_loop_eval.model_clients import FastWAMModelClient
from fastwam.closed_loop_eval.observation_adapter import AAOObservationAdapter
from fastwam.closed_loop_eval.runner import _validated_actions
from fastwam.closed_loop_eval.runner import _resolve_overrides
from fastwam.closed_loop_eval.runner import _default_disable_arm_randomization
from fastwam.closed_loop_eval.sim_service_client import SimulatorServiceClient


def _sim_info() -> dict:
    return {
        "operators": {
            "arm": {
                "arm_output_name": "arm",
                "eef_output_name": "eef",
                "arm_actuators": ["j0", "j1"],
                "eef_actuators": ["gripper"],
            }
        },
        "cameras": {
            "cam0": {
                "camera_info": {
                    "color": {
                        "k": [1.0, 0.0, 2.5, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0],
                    }
                },
                "camera_extrinsics": {
                    "rotation_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "translation": [0.0, 0.0, 0.0],
                },
            }
        },
    }


def _observation() -> dict:
    return {
        "cam0/color/image_raw": {"data": np.zeros((4, 5, 3), dtype=np.uint8), "t": 10},
        "arm/joint_state/position": {"data": np.asarray([0.1, 0.2], dtype=np.float32)},
        "eef/joint_state/position": {"data": np.asarray([0.03], dtype=np.float32)},
        "arm/pose/position": {"data": np.asarray([0.4, 0.0, 0.2], dtype=np.float32)},
        "arm/pose/rotation": {"data": np.asarray([0.0, 0.1, 0.2], dtype=np.float32)},
    }


def test_proprio_vector_requires_exact_dim() -> None:
    assert _require_vector_dim(np.zeros(7, dtype=np.float32), 7, name="proprio").shape == (7,)
    with pytest.raises(RuntimeError, match="expected exactly 7 values, got 6"):
        _require_vector_dim(np.zeros(6, dtype=np.float32), 7, name="proprio")


def test_default_benchmark_profiles_load_from_yaml() -> None:
    open_door = _load_named_profile("open_door_airbot_play_gs")
    cup = _load_named_profile("cup_on_coaster_gs_airbot_p7")
    assert open_door.fastwam_config.endswith("configs/task/mix_uncond_2cam224_1e-4.yaml")
    assert open_door.proprio_mode == "joint"
    assert open_door.stride == 32
    assert open_door.disable_arm_randomization is True
    assert cup.fastwam_config.endswith("configs/task/cup_uncond_2cam224_1e-4.yaml")
    assert cup.proprio_mode == "joint"


def test_custom_profile_config_loads_relative_paths(tmp_path) -> None:
    profile_path = tmp_path / "custom_env.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "name: custom_env",
                "task: custom_task",
                "instruction: do the thing",
                "camera_map: head_left=cam0",
                "action_repeat: 3",
                "train_action_hz: 20.0",
                "max_updates: 12",
                "proprio_mode: cartesian",
                "proprio_dim: 7",
                "fastwam_config: configs/task/mix_uncond_2cam224_1e-4.yaml",
                "text_cache_dir: data/text_embeds_cache/mix",
            ]
        ),
        encoding="utf-8",
    )
    profile = _load_profile_file(profile_path)
    assert profile.name == "custom_env"
    assert profile.task == "custom_task"
    assert profile.action_repeat == 3
    assert profile.fastwam_config.endswith("configs/task/mix_uncond_2cam224_1e-4.yaml")


def test_benchmark_profile_stride_can_be_overridden() -> None:
    args = SimpleNamespace(
        profile="open_door_airbot_play_gs",
        profile_config=None,
        task=None,
        instruction=None,
        camera_map=None,
        action_repeat=None,
        train_action_hz=None,
        max_updates=None,
        stride=12,
        disable_arm_randomization=None,
        proprio_mode=None,
        proprio_dim=None,
        fastwam_config=None,
        text_cache_dir=None,
    )

    profile = _resolve_profile(args)

    assert profile.stride == 12
    assert profile.disable_arm_randomization is True


def test_benchmark_profile_arm_randomization_can_be_reenabled() -> None:
    args = SimpleNamespace(
        profile="open_door_airbot_play_gs",
        profile_config=None,
        task=None,
        instruction=None,
        camera_map=None,
        action_repeat=None,
        train_action_hz=None,
        max_updates=None,
        stride=None,
        disable_arm_randomization=False,
        proprio_mode=None,
        proprio_dim=None,
        fastwam_config=None,
        text_cache_dir=None,
    )

    profile = _resolve_profile(args)

    assert profile.disable_arm_randomization is False


def test_disable_arm_randomization_adds_base_and_eef_overrides() -> None:
    args = SimpleNamespace(
        disable_arm_randomization=True,
        disable_arm_eef_randomization=False,
        override=["task.randomization.arm.base.x=[0.1,0.1]"],
    )

    overrides = _resolve_overrides(args)

    assert "task.randomization.arm.base.x=[0.0,0.0]" in overrides
    assert "task.randomization.arm.base.y=[0.0,0.0]" in overrides
    assert "task.randomization.arm.base.z=[0.0,0.0]" in overrides
    assert "task.randomization.arm.eef.x=[0.0,0.0]" in overrides
    assert "task.randomization.arm.eef.y=[0.0,0.0]" in overrides
    assert "task.randomization.arm.eef.z=[0.0,0.0]" in overrides
    assert overrides[-1] == "task.randomization.arm.base.x=[0.1,0.1]"


def test_open_door_runner_disables_arm_randomization_by_default() -> None:
    assert _default_disable_arm_randomization("open_door_airbot_play_back_gs") is True
    assert _default_disable_arm_randomization("cup_on_coaster_gs_airbot_p7") is False


def test_task_update_requires_batch_sized_done_and_success() -> None:
    update = SimpleNamespace(done=np.asarray([False]), success=np.asarray([False], dtype=object))
    with pytest.raises(RuntimeError, match="done.*expected 2 values"):
        _extract_task_update(update, {}, batch_size=2)


def test_task_update_preserves_optional_metadata() -> None:
    update = SimpleNamespace(
        done=np.asarray([False, True]),
        success=np.asarray([None, True], dtype=object),
        status=np.asarray(["running", "success"], dtype=object),
        stage_index=np.asarray([0, 1]),
        stage_name=["approach", "finish"],
        details=[{"distance": 0.1}, {"distance": 0.0}],
        phase=["policy", None],
        phase_step=np.asarray([3, -1]),
    )
    parsed = _extract_task_update(update, {}, batch_size=2)
    assert parsed["status"] == ["running", "success"]
    assert parsed["stage_name"] == ["approach", "finish"]
    assert parsed["details"][0]["distance"] == 0.1
    assert parsed["phase_step"].tolist() == [3, -1]


def test_observation_adapter_rejects_missing_rgb() -> None:
    adapter = AAOObservationAdapter(_sim_info(), selected_cameras=["cam0"])
    obs = _observation()
    obs.pop("cam0/color/image_raw")
    with pytest.raises(RuntimeError, match="Missing RGB observation"):
        adapter.extend([obs])


def test_observation_adapter_rejects_bad_proprio_dim() -> None:
    adapter = AAOObservationAdapter(_sim_info(), selected_cameras=["cam0"])
    obs = _observation()
    obs["arm/joint_state/position"] = {"data": np.asarray([0.1], dtype=np.float32)}
    adapter.extend([obs])
    with pytest.raises(RuntimeError, match="arm_joint_position.*expected 2 values"):
        adapter.build_model_input({"head_left": "cam0"}, proprio_mode="joint")


def test_observation_adapter_builds_expected_state_shapes() -> None:
    adapter = AAOObservationAdapter(_sim_info(), selected_cameras=["cam0"])
    adapter.extend([_observation()])
    cartesian = adapter.build_model_input({"head_left": "cam0"}, proprio_mode="cartesian")
    joint = adapter.build_model_input({"head_left": "cam0"}, proprio_mode="joint")
    assert cartesian["proprio_raw"].shape == (7,)
    assert joint["proprio_raw"].shape == (3,)


def test_cartesian_actions_reject_extra_columns() -> None:
    with pytest.raises(RuntimeError, match=r"shape \[T,7\]"):
        _validated_actions(
            {"action_format": "cartesian_absolute", "actions": np.zeros((2, 8), dtype=np.float32)},
            expected_format="cartesian_absolute",
            action_dim=7,
        )


def test_delta6_abs_gripper_shifts_backward_delta_and_cumsum() -> None:
    current = np.asarray([10.0, 20.0, 30.0, 0.1, 0.2, 0.3], dtype=np.float32)
    raw = np.asarray(
        [
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.20],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.30],
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.40],
        ],
        dtype=np.float32,
    )

    converted = FastWAMModelClient._delta_to_absolute(raw, current)

    np.testing.assert_allclose(converted[:, 0], [11.0, 13.0, 16.0, 16.0])
    np.testing.assert_allclose(converted[:, 5], [0.4, 0.6, 0.9, 0.9], atol=1e-6)
    np.testing.assert_allclose(converted[:, 6], [0.20, 0.30, 0.40, 0.40], atol=1e-6)


def test_delta6_abs_gripper_forward_cumsums_without_shift() -> None:
    current = np.asarray([10.0, 20.0, 30.0, 0.1, 0.2, 0.3], dtype=np.float32)
    raw = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.20],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.30],
        ],
        dtype=np.float32,
    )

    converted = FastWAMModelClient._delta_to_absolute(raw, current, frame_aligned_backward_delta=False)

    np.testing.assert_allclose(converted[:, 0], [11.0, 13.0])
    np.testing.assert_allclose(converted[:, 5], [0.4, 0.6], atol=1e-6)
    np.testing.assert_allclose(converted[:, 6], [0.20, 0.30], atol=1e-6)


def test_action_expansion_rejects_single_row_broadcast() -> None:
    env_mask = np.asarray([True, True, False])
    with pytest.raises(ValueError, match="selected envs=2"):
        SimulatorServiceClient._expand_array(
            np.zeros((1, 3), dtype=np.float32),
            (3,),
            batch_size=3,
            env_mask=env_mask,
        )
