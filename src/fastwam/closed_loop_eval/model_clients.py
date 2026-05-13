"""Model clients used by FastWAM AAO closed-loop evaluation."""

from __future__ import annotations

import hashlib
import contextlib
import logging
import multiprocessing as mp
import os
import queue
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from fastwam.datasets.dataset_utils import (
    CenterCrop,
    Normalize,
    ResizeSmallestSideAspectPreserving,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


_REPO_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _looks_composed_fastwam_config(cfg: DictConfig) -> bool:
    if "data" not in cfg or "model" not in cfg:
        return False
    model_cfg = cfg.get("model")
    return model_cfg is not None and model_cfg.get("_target_") is not None


def load_fastwam_config(config_path: str | Path) -> DictConfig:
    """Load either a composed FastWAM config or a configs/task/*.yaml file."""
    path = Path(config_path).expanduser().resolve()
    cfg = OmegaConf.load(path)
    if _looks_composed_fastwam_config(cfg):
        OmegaConf.resolve(cfg)
        return cfg

    configs_dir = path.parents[1] if path.parent.name == "task" else _REPO_ROOT / "configs"
    task_name = path.stem
    with initialize_config_dir(config_dir=str(configs_dir), version_base=None):
        cfg = compose(config_name="train", overrides=[f"task={task_name}"])
    OmegaConf.resolve(cfg)
    return cfg


def _select_inference_data_cfg(cfg: DictConfig) -> DictConfig:
    data_cfg = cfg.data.get("val")
    if data_cfg is None:
        data_cfg = cfg.data.train
    return data_cfg


def _prepare_model_cfg_for_checkpoint_inference(cfg: DictConfig) -> None:
    model_cfg = cfg.get("model")
    if model_cfg is None:
        return
    path_value = model_cfg.get("action_dit_pretrained_path")
    if path_value is None or bool(model_cfg.get("skip_dit_load_from_pretrain", False)):
        return
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if path.exists():
        return
    logger.warning(
        "ActionDiT pretrained path is missing (%s); skipping pretrained ActionDiT load "
        "and expecting the provided FastWAM checkpoint to populate weights.",
        path,
    )
    model_cfg.skip_dit_load_from_pretrain = True
    model_cfg.action_dit_pretrained_path = None


class BaseModelClient(ABC):
    @abstractmethod
    def infer(self, model_input: dict[str, Any]) -> dict[str, Any]:
        """Return a chunked action payload with model actions."""

    def infer_batch(self, model_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.infer(model_input) for model_input in model_inputs]

    def close(self) -> None:
        return None


class HoldModelClient(BaseModelClient):
    def __init__(self, *, horizon: int = 32) -> None:
        self.horizon = int(horizon)

    def infer(self, model_input: dict[str, Any]) -> dict[str, Any]:
        cartesian = np.asarray(model_input["cartesian_position"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(model_input["gripper_position"], dtype=np.float32).reshape(-1)
        if cartesian.size < 6 or gripper.size < 1:
            raise RuntimeError("HoldModelClient requires 6D cartesian pose and 1D gripper.")
        action = np.concatenate([cartesian[:6], gripper[:1]], axis=0).astype(np.float32)
        return {
            "action_format": "cartesian_absolute",
            "actions": np.repeat(action[None, :], self.horizon, axis=0),
            "source": "hold",
        }


class HoldJointModelClient(BaseModelClient):
    def __init__(self, *, horizon: int = 32) -> None:
        self.horizon = int(horizon)

    def infer(self, model_input: dict[str, Any]) -> dict[str, Any]:
        joint = np.asarray(model_input["joint_position"], dtype=np.float32).reshape(-1)
        if joint.size < 1:
            raise RuntimeError("HoldJointModelClient requires joint_position.")
        return {
            "action_format": "joint_absolute",
            "actions": np.repeat(joint[None, :], self.horizon, axis=0),
            "source": "hold_joint",
        }


class FastWAMModelClient(BaseModelClient):
    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        dataset_stats_path: str | Path,
        text_cache_dir: str | Path | None = None,
        instruction: str = "open the door",
        action_horizon: int = 32,
        device: str = "cuda:0",
        num_inference_steps: int = 10,
        seed: int | None = 42,
        rand_device: str = "cpu",
        action_mode: str = "delta6_abs_gripper",
        preload_text_context: bool = True,
        output_action_format: str = "cartesian_absolute",
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.dataset_stats_path = Path(dataset_stats_path).expanduser().resolve()
        self.text_cache_dir = None if text_cache_dir is None else Path(text_cache_dir).expanduser()
        self.instruction = str(instruction)
        self.action_horizon = int(action_horizon)
        self.device = str(device)
        self.num_inference_steps = int(num_inference_steps)
        self.seed = None if seed is None else int(seed)
        self.rand_device = str(rand_device)
        self.action_mode = str(action_mode)
        self.preload_text_context = bool(preload_text_context)
        self.output_action_format = str(output_action_format)
        self.call_index = 0

        supported_action_modes = {
            "delta6_abs_gripper",
            "delta6_abs_gripper_forward",
            "absolute",
            "absolute_joint",
        }
        if self.action_mode not in supported_action_modes:
            raise ValueError(
                f"Unsupported action_mode='{self.action_mode}'. "
                f"Expected one of {sorted(supported_action_modes)}."
            )

        self.cfg = load_fastwam_config(self.config_path)
        _prepare_model_cfg_for_checkpoint_inference(self.cfg)
        self.data_cfg = _select_inference_data_cfg(self.cfg)
        self.processor = instantiate(self.data_cfg.processor)
        stats = load_dataset_stats_from_json(str(self.dataset_stats_path))
        self.processor.set_normalizer_from_stats(stats)
        self.processor.eval()

        if self.text_cache_dir is None:
            self.text_cache_dir = Path(str(self.data_cfg.text_embedding_cache_dir))
        self.text_cache_dir = self.text_cache_dir.expanduser()

        mixed_precision = str(self.cfg.get("mixed_precision", "bf16"))
        model_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        if mixed_precision in ("no", "fp32", "float32"):
            model_dtype = torch.float32

        self.model = instantiate(self.cfg.model, model_dtype=model_dtype, device=self.device)
        self.model.load_checkpoint(str(self.checkpoint_path))
        self.model.eval()
        self._text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
        self.full_prompt = DEFAULT_PROMPT.format(task=self.instruction)
        if self.preload_text_context:
            self.context, self.context_mask, self.full_prompt = self._get_text_context(self.instruction)
        else:
            self.context = None
            self.context_mask = None

        self.video_size = tuple(int(x) for x in self.data_cfg.video_size)
        self.concat_multi_camera = str(self.data_cfg.get("concat_multi_camera", "horizontal"))
        self.image_shapes = {
            str(meta["key"]): tuple(int(v) for v in meta["shape"])
            for meta in self.data_cfg.shape_meta.images
        }
        self.state_key = str(self.data_cfg.shape_meta.state[0]["key"])
        self.proprio_dim = int(self.data_cfg.shape_meta.state[0]["raw_shape"])
        self.model_action_dim = int(self.data_cfg.shape_meta.action[0]["raw_shape"])

        # Image transforms mirroring RobotVideoDataset (training side):
        # concat cameras at native resolution -> aspect-preserving resize -> center crop -> normalize.
        self._image_resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": int(self.video_size[1]), "img_h": int(self.video_size[0])}
        )
        self._image_crop_transform = CenterCrop(
            args={"img_w": int(self.video_size[1]), "img_h": int(self.video_size[0])}
        )
        self._image_normalize_transform = Normalize(args={"mean": 0.5, "std": 0.5})

    def infer(self, model_input: dict[str, Any]) -> dict[str, Any]:
        return self.infer_batch([model_input])[0]

    def infer_batch(self, model_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not model_inputs:
            return []
        input_image = self._preprocess_image_batch([item["images"] for item in model_inputs])
        proprio_raw = np.stack(
            [np.asarray(item["proprio_raw"], dtype=np.float32).reshape(-1) for item in model_inputs],
            axis=0,
        )
        proprio_norm = self._normalize_proprio(proprio_raw)
        instructions = [str(item.get("instruction", self.instruction)) for item in model_inputs]
        context, context_mask, full_prompts = self._get_batch_text_context(instructions)
        seed = None if self.seed is None else self.seed + self.call_index
        self.call_index += 1

        with torch.no_grad():
            output = self.model.infer_action(
                prompt=None,
                input_image=input_image,
                action_horizon=self.action_horizon,
                proprio=proprio_norm,
                context=context,
                context_mask=context_mask,
                num_inference_steps=self.num_inference_steps,
                seed=seed,
                rand_device=self.rand_device,
            )
        action_norm = output["action"].detach().to(device="cpu", dtype=torch.float32)
        denorm_action = self._denormalize_action(action_norm, proprio_norm)
        if denorm_action.ndim == 2:
            denorm_action = denorm_action[None, ...]
        return [
            {
                "action_format": self.output_action_format,
                "actions": self._format_actions(denorm_action[index], model_input).astype(np.float32, copy=False),
                "source": "fastwam",
                "action_mode": self.action_mode,
                "action_semantics": self._action_semantics(),
                "normalized_action_shape": list(action_norm.shape),
                "batch_size": len(model_inputs),
                "batch_index": index,
                "full_prompt": full_prompts[index],
                "instruction": instructions[index],
            }
            for index, model_input in enumerate(model_inputs)
        ]

    def _resolve_current_position(self, model_input: dict[str, Any]) -> np.ndarray:
        for key in ("current_position", "cartesian_position"):
            if key in model_input and model_input[key] is not None:
                current = np.asarray(model_input[key], dtype=np.float32).reshape(-1)
                break
        else:
            raise RuntimeError("Model input is missing current/cartesian_position required for delta6_abs_gripper actions.")
        if current.size != 6:
            raise RuntimeError(f"Current cartesian position must contain exactly 6 values, got {current.size}.")
        return current

    def infer_joint_video(
        self,
        model_input: dict[str, Any],
        *,
        num_video_frames: int,
        test_action_with_infer_action: bool = False,
    ) -> dict[str, Any]:
        """Infer a 32-action chunk plus a future video clip for visualization."""
        input_image = self._preprocess_images(model_input["images"])
        proprio_norm = self._normalize_proprio(model_input["proprio_raw"])
        instruction = str(model_input.get("instruction", self.instruction))
        context, context_mask, full_prompt = self._get_text_context(instruction)
        seed = None if self.seed is None else self.seed + self.call_index
        self.call_index += 1

        with torch.no_grad():
            output = self.model.infer_joint(
                prompt=None,
                input_image=input_image,
                num_video_frames=int(num_video_frames),
                action_horizon=self.action_horizon,
                proprio=proprio_norm,
                context=context,
                context_mask=context_mask,
                num_inference_steps=self.num_inference_steps,
                seed=seed,
                rand_device=self.rand_device,
                test_action_with_infer_action=test_action_with_infer_action,
        )
        action_norm = output["action"].detach().to(device="cpu", dtype=torch.float32)
        denorm_action = self._denormalize_action(action_norm, proprio_norm)
        actions = self._format_actions(denorm_action, model_input)
        return {
            "action_format": self.output_action_format,
            "actions": actions.astype(np.float32, copy=False),
            "video": list(output["video"]),
            "source": "fastwam_joint_video",
            "action_mode": self.action_mode,
            "action_semantics": self._action_semantics(),
            "num_video_frames": int(num_video_frames),
            "normalized_action_shape": list(action_norm.shape),
            "full_prompt": full_prompt,
            "instruction": instruction,
        }

    def reconstruct_video_from_model_inputs(self, model_inputs: list[dict[str, Any]]) -> list[Image.Image]:
        """VAE-reconstruct model-facing actual frames for pred/recon/actual comparison."""
        if not model_inputs:
            return []
        frame_tensors = [self._preprocess_images(item["images"])[0] for item in model_inputs]
        video = torch.stack(frame_tensors, dim=1).unsqueeze(0)
        video = video.to(device=self.model.device, dtype=self.model.torch_dtype)
        with torch.no_grad():
            latents = self.model._encode_video_latents(video, tiled=False)
            return list(self.model._decode_latents(latents, tiled=False))

    def _preprocess_images(self, images: dict[str, np.ndarray]) -> torch.Tensor:
        return self._preprocess_image(images).unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)

    def _preprocess_image_batch(self, images_batch: list[dict[str, np.ndarray]]) -> torch.Tensor:
        image = torch.stack([self._preprocess_image(images) for images in images_batch], dim=0)
        return image.to(device=self.model.device, dtype=self.model.torch_dtype)

    def _stitch_cameras_native(self, images: dict[str, np.ndarray]) -> torch.Tensor:
        """Concatenate per-camera RGB inputs at their native resolution.

        Returns a uint8 tensor (C, H, W) where the spatial layout matches the training
        pipeline (training calls torch.cat on the raw video frames before any resize).
        """
        camera_tensors: list[torch.Tensor] = []
        for camera_key in self.image_shapes:
            if camera_key not in images:
                raise RuntimeError(f"Missing FastWAM camera '{camera_key}' in model input.")
            rgb = np.asarray(images[camera_key], dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise RuntimeError(f"FastWAM camera '{camera_key}' expected RGB shape [H,W,3], got {rgb.shape}.")
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()
            camera_tensors.append(tensor)

        if self.concat_multi_camera == "horizontal":
            return torch.cat(camera_tensors, dim=-1)
        if self.concat_multi_camera == "vertical":
            return torch.cat(camera_tensors, dim=-2)
        if len(camera_tensors) == 1:
            return camera_tensors[0]
        raise ValueError(f"Unsupported concat_multi_camera='{self.concat_multi_camera}'.")

    def _preprocess_image(self, images: dict[str, np.ndarray]) -> torch.Tensor:
        # Mirror RobotVideoDataset: concat at native res -> resize (keep aspect) -> center crop -> normalize.
        image = self._stitch_cameras_native(images)
        image = self._image_resize_transform(image)
        image = self._image_crop_transform(image)
        image = self._image_normalize_transform(image)
        return image.contiguous()

    def preview_uint8(self, images: dict[str, np.ndarray]) -> np.ndarray:
        """Return the uint8 RGB view of the model input (training-aligned), shape (H, W, 3).

        Useful for visualization scripts that want to show what the model actually sees,
        without recomputing the resize/crop pipeline locally.
        """
        image = self._stitch_cameras_native(images)
        image = self._image_resize_transform(image)
        image = self._image_crop_transform(image)
        return image.permute(1, 2, 0).contiguous().to(dtype=torch.uint8).numpy()

    def _normalize_proprio(self, proprio_raw: np.ndarray) -> torch.Tensor:
        proprio = torch.as_tensor(proprio_raw, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2:
            raise RuntimeError(f"Expected proprio_raw shape [D] or [B,D], got {tuple(proprio.shape)}.")
        if proprio.shape[-1] != self.proprio_dim:
            raise RuntimeError(f"Expected proprio_raw length {self.proprio_dim}, got {proprio.shape[-1]}.")
        batch = {"state": {self.state_key: proprio.clone()}}
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        return batch["state"].to(device=self.model.device, dtype=self.model.torch_dtype)

    def _denormalize_action(self, action_norm: torch.Tensor, proprio_norm: torch.Tensor) -> np.ndarray:
        single = action_norm.ndim == 2
        if single:
            action_batch = action_norm.unsqueeze(0)
        elif action_norm.ndim == 3:
            action_batch = action_norm
        else:
            raise ValueError(f"Expected normalized action [T,D] or [B,T,D], got {tuple(action_norm.shape)}.")
        state_flat = proprio_norm.detach().to(device="cpu", dtype=torch.float32)
        if state_flat.ndim == 1:
            state_flat = state_flat.unsqueeze(0)
        if state_flat.shape[0] != action_batch.shape[0]:
            raise ValueError(
                f"Action/state batch mismatch: action={tuple(action_batch.shape)}, state={tuple(state_flat.shape)}."
            )
        state_norm = state_flat.reshape(state_flat.shape[0], 1, -1)
        batch = {
            "action": action_batch.detach().to(device="cpu", dtype=torch.float32),
            "state": state_norm,
        }
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        action_meta = self.processor.shape_meta["action"]
        denorm = torch.cat([batch["action"][meta["key"]] for meta in action_meta], dim=-1)
        if single:
            denorm = denorm.squeeze(0)
        return denorm.detach().cpu().numpy().astype(np.float32)

    def _format_actions(self, denorm_action: np.ndarray, model_input: dict[str, Any]) -> np.ndarray:
        if self.action_mode == "delta6_abs_gripper":
            current_position = self._resolve_current_position(model_input)
            return self._delta_to_absolute(denorm_action, current_position, frame_aligned_backward_delta=True)
        if self.action_mode == "delta6_abs_gripper_forward":
            current_position = self._resolve_current_position(model_input)
            return self._delta_to_absolute(denorm_action, current_position, frame_aligned_backward_delta=False)
        if self.action_mode in {"absolute", "absolute_joint"}:
            return np.asarray(denorm_action, dtype=np.float32)
        raise ValueError(f"Unsupported action_mode='{self.action_mode}'.")

    def _action_semantics(self) -> str:
        if self.action_mode == "delta6_abs_gripper":
            return (
                "LeRobot frame-aligned EEF delta: raw row t is pose[t]-pose[t-1]. "
                "The bridge shifts rows left once, cumulatively integrates the first 6 dims "
                "from the current EEF pose, and sends an absolute gripper target."
            )
        if self.action_mode == "delta6_abs_gripper_forward":
            return (
                "Forward EEF delta: raw row t is already the next-step pose delta. "
                "The bridge cumulatively integrates the first 6 dims from the current EEF pose "
                "and sends an absolute gripper target."
            )
        return "Model output is already absolute in the requested action format."

    @staticmethod
    def _delta_to_absolute(
        action_delta: np.ndarray,
        current_cartesian: np.ndarray,
        *,
        frame_aligned_backward_delta: bool = True,
    ) -> np.ndarray:
        action = np.asarray(action_delta, dtype=np.float32).copy()
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.ndim != 2 or action.shape[1] < 7:
            raise RuntimeError(f"delta6_abs_gripper action must have shape [T,7+], got {action.shape}.")
        current = np.asarray(current_cartesian, dtype=np.float32).reshape(-1)
        if current.size < 6:
            raise RuntimeError("current_cartesian must contain at least 6 values.")
        step_delta = action[:, :6].copy()
        if frame_aligned_backward_delta:
            # LeRobot eef_delta_gripper data stores delta[t] = pose[t] - pose[t-1].
            # For a command after obs[t], row 0 is the previous transition, so use
            # rows 1..T-1 as future increments and hold the last target.
            shifted_delta = np.zeros_like(step_delta)
            if step_delta.shape[0] > 1:
                shifted_delta[:-1] = step_delta[1:]
                shifted_gripper = action[:, 6].copy()
                shifted_gripper[:-1] = action[1:, 6]
                shifted_gripper[-1] = action[-1, 6]
                action[:, 6] = shifted_gripper
            step_delta = shifted_delta
        action[:, :6] = current[:6][None, :] + np.cumsum(step_delta, axis=0)
        return action

    def _get_text_context(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        cached = self._text_context_cache.get(instruction)
        if cached is not None:
            return cached
        context = self._load_text_context(instruction)
        self._text_context_cache[instruction] = context
        return context

    def _get_batch_text_context(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        contexts: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        full_prompts: list[str] = []
        for instruction in instructions:
            context, context_mask, full_prompt = self._get_text_context(instruction)
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.shape[0] != 1 or context_mask.shape[0] != 1:
                raise RuntimeError(
                    f"Expected cached text context batch size 1, got {tuple(context.shape)} and {tuple(context_mask.shape)}."
                )
            contexts.append(context[0])
            masks.append(context_mask[0])
            full_prompts.append(full_prompt)
        return torch.stack(contexts, dim=0), torch.stack(masks, dim=0), full_prompts

    def _load_text_context(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        full_prompt = DEFAULT_PROMPT.format(task=instruction)
        hashed = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()
        cache_path = self.text_cache_dir / f"{hashed}.t5_len{int(self.data_cfg.context_len)}.wan22ti2v5b.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py with the same prompt first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"].detach().clone()
        context_mask = payload["mask"].bool().detach().clone()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask, dtype=torch.bool)
        return (
            context.to(device=self.model.device, dtype=self.model.torch_dtype),
            context_mask.to(device=self.model.device, dtype=torch.bool),
            full_prompt,
        )


def _fastwam_model_worker_loop(
    *,
    worker_index: int,
    model_gpu: str,
    request_queue: Any,
    response_queue: Any,
    client_kwargs: dict[str, Any],
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(model_gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    client: FastWAMModelClient | None = None
    try:
        worker_kwargs = dict(client_kwargs)
        worker_kwargs["device"] = "cuda:0"
        client = FastWAMModelClient(**worker_kwargs)
        response_queue.put(
            {
                "type": "ready",
                "worker_index": int(worker_index),
                "metadata": {
                    "worker_index": int(worker_index),
                    "model_gpu": str(model_gpu),
                    "device": "cuda:0",
                    "proprio_dim": int(client.proprio_dim),
                    "model_action_dim": int(client.model_action_dim),
                    "action_mode": client.action_mode,
                    "checkpoint_path": str(client.checkpoint_path),
                    "dataset_stats_path": str(client.dataset_stats_path),
                    "config_path": str(client.config_path),
                },
            }
        )
    except BaseException as exc:  # noqa: BLE001
        response_queue.put(
            {
                "type": "error",
                "stage": "init",
                "worker_index": int(worker_index),
                "model_gpu": str(model_gpu),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return

    try:
        while True:
            message = request_queue.get()
            if not isinstance(message, dict):
                continue
            message_type = message.get("type")
            if message_type == "close":
                return
            if message_type != "infer":
                continue
            request_id = str(message["request_id"])
            model_inputs = list(message["model_inputs"])
            try:
                start = time.perf_counter()
                responses = client.infer_batch(model_inputs)
                elapsed = time.perf_counter() - start
                per_item_elapsed = elapsed / max(len(responses), 1)
                for batch_index, response in enumerate(responses):
                    response["worker_index"] = int(worker_index)
                    response["model_gpu"] = str(model_gpu)
                    response["worker_batch_index"] = int(batch_index)
                    response["worker_batch_size"] = int(len(responses))
                    response["worker_infer_time_sec"] = float(elapsed)
                    response["model_infer_time_sec_per_item"] = float(per_item_elapsed)
                response_queue.put(
                    {
                        "type": "result",
                        "request_id": request_id,
                        "worker_index": int(worker_index),
                        "responses": responses,
                    }
                )
            except BaseException as exc:  # noqa: BLE001
                response_queue.put(
                    {
                        "type": "error",
                        "stage": "infer",
                        "request_id": request_id,
                        "worker_index": int(worker_index),
                        "model_gpu": str(model_gpu),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        if client is not None:
            client.close()


class ParallelFastWAMModelClient(BaseModelClient):
    """Run one FastWAM model process per physical GPU and shard env batches across them."""

    def __init__(
        self,
        *,
        model_gpus: Sequence[str | int],
        worker_timeout_sec: float = 600.0,
        worker_start_method: str = "spawn",
        **fastwam_kwargs: Any,
    ) -> None:
        self.model_gpus = [str(item).strip() for item in model_gpus if str(item).strip()]
        if not self.model_gpus:
            raise ValueError("ParallelFastWAMModelClient requires at least one model GPU.")
        if len(set(self.model_gpus)) != len(self.model_gpus):
            raise ValueError(f"Duplicate model GPU ids are not supported: {self.model_gpus}.")
        if worker_timeout_sec <= 0:
            raise ValueError("worker_timeout_sec must be positive.")

        self.worker_timeout_sec = float(worker_timeout_sec)
        self.worker_start_method = str(worker_start_method)
        self.ctx = mp.get_context(self.worker_start_method)
        self._response_queue = self.ctx.Queue()
        self._request_queues: list[Any] = []
        self._processes: list[mp.Process] = []
        self._closed = False
        self._next_request_id = 0
        self.worker_metadata: list[dict[str, Any]] = []

        try:
            for worker_index, model_gpu in enumerate(self.model_gpus):
                request_queue = self.ctx.Queue()
                process = self.ctx.Process(
                    target=_fastwam_model_worker_loop,
                    kwargs={
                        "worker_index": worker_index,
                        "model_gpu": model_gpu,
                        "request_queue": request_queue,
                        "response_queue": self._response_queue,
                        "client_kwargs": dict(fastwam_kwargs),
                    },
                    daemon=True,
                )
                with _temporary_env({"CUDA_VISIBLE_DEVICES": model_gpu}):
                    process.start()
                self._request_queues.append(request_queue)
                self._processes.append(process)
            self.worker_metadata = self._await_worker_ready()
            self._validate_worker_metadata()
        except Exception:
            self.close()
            raise

    @property
    def num_workers(self) -> int:
        return len(self.model_gpus)

    def infer(self, model_input: dict[str, Any]) -> dict[str, Any]:
        return self.infer_batch([model_input])[0]

    def infer_batch(self, model_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not model_inputs:
            return []
        if self._closed:
            raise RuntimeError("ParallelFastWAMModelClient is already closed.")

        grouped_inputs: dict[int, list[dict[str, Any]]] = {}
        grouped_positions: dict[int, list[int]] = {}
        for position, model_input in enumerate(model_inputs):
            worker_index = self._select_worker(model_input, fallback_index=position)
            grouped_inputs.setdefault(worker_index, []).append(model_input)
            grouped_positions.setdefault(worker_index, []).append(position)

        pending: dict[str, tuple[int, list[int]]] = {}
        for worker_index, worker_inputs in sorted(grouped_inputs.items()):
            request_id = f"{os.getpid()}-{self._next_request_id}-{worker_index}"
            self._next_request_id += 1
            pending[request_id] = (worker_index, grouped_positions[worker_index])
            self._request_queues[worker_index].put(
                {
                    "type": "infer",
                    "request_id": request_id,
                    "model_inputs": worker_inputs,
                }
            )

        responses: list[dict[str, Any] | None] = [None] * len(model_inputs)
        deadline = time.monotonic() + self.worker_timeout_sec
        while pending:
            message = self._get_response_until(deadline)
            message_type = message.get("type")
            request_id = str(message.get("request_id", ""))
            if message_type == "result" and request_id in pending:
                _worker_index, positions = pending.pop(request_id)
                worker_responses = list(message["responses"])
                if len(worker_responses) != len(positions):
                    raise RuntimeError(
                        f"Model worker returned {len(worker_responses)} responses for {len(positions)} inputs."
                    )
                for position, response in zip(positions, worker_responses, strict=True):
                    responses[position] = response
                continue
            if message_type == "error" and (not request_id or request_id in pending):
                raise RuntimeError(self._format_worker_error(message))
            logger.debug("Ignoring unrelated model worker message: %s", message)

        missing = [index for index, response in enumerate(responses) if response is None]
        if missing:
            raise RuntimeError(f"Missing model responses for input positions: {missing}.")
        return [response for response in responses if response is not None]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for request_queue in self._request_queues:
            try:
                request_queue.put({"type": "close"}, block=False)
            except Exception:  # noqa: BLE001
                pass
        for process in self._processes:
            process.join(timeout=10.0)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            if process.is_alive():
                process.join(timeout=5.0)
        for request_queue in self._request_queues:
            try:
                request_queue.cancel_join_thread()
                request_queue.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self._response_queue.cancel_join_thread()
            self._response_queue.close()
        except Exception:  # noqa: BLE001
            pass

    def _select_worker(self, model_input: dict[str, Any], *, fallback_index: int) -> int:
        env_index = model_input.get("env_index", fallback_index)
        return int(env_index) % self.num_workers

    def _await_worker_ready(self) -> list[dict[str, Any]]:
        metadata_by_worker: dict[int, dict[str, Any]] = {}
        deadline = time.monotonic() + self.worker_timeout_sec
        while len(metadata_by_worker) < self.num_workers:
            message = self._get_response_until(deadline, ready_workers=set(metadata_by_worker))
            message_type = message.get("type")
            if message_type == "ready":
                worker_index = int(message["worker_index"])
                metadata_by_worker[worker_index] = dict(message["metadata"])
                continue
            if message_type == "error":
                raise RuntimeError(self._format_worker_error(message))
            logger.debug("Ignoring model worker message before ready: %s", message)
        return [metadata_by_worker[index] for index in range(self.num_workers)]

    def _validate_worker_metadata(self) -> None:
        proprio_dims = {int(item["proprio_dim"]) for item in self.worker_metadata}
        action_dims = {int(item["model_action_dim"]) for item in self.worker_metadata}
        action_modes = {str(item.get("action_mode")) for item in self.worker_metadata}
        if len(proprio_dims) != 1:
            raise RuntimeError(f"Model workers disagree on proprio_dim: {sorted(proprio_dims)}.")
        if len(action_dims) != 1:
            raise RuntimeError(f"Model workers disagree on model_action_dim: {sorted(action_dims)}.")
        if len(action_modes) != 1:
            raise RuntimeError(f"Model workers disagree on action_mode: {sorted(action_modes)}.")
        self.proprio_dim = proprio_dims.pop()
        self.model_action_dim = action_dims.pop()
        self.action_mode = action_modes.pop()

    def _get_response_until(self, deadline: float, ready_workers: set[int] | None = None) -> dict[str, Any]:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for model workers after {self.worker_timeout_sec:.1f}s.")
            try:
                return self._response_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                dead = []
                for index, process in enumerate(self._processes):
                    if ready_workers is not None and index in ready_workers:
                        continue
                    if not process.is_alive():
                        dead.append(f"{index}(exitcode={process.exitcode})")
                if dead:
                    raise RuntimeError(f"Model worker process exited unexpectedly: {', '.join(dead)}.")

    def _format_worker_error(self, message: dict[str, Any]) -> str:
        worker_index = message.get("worker_index")
        model_gpu = message.get("model_gpu")
        stage = message.get("stage", "unknown")
        error = message.get("error", "unknown error")
        tb = message.get("traceback", "")
        return f"Model worker {worker_index} on GPU {model_gpu} failed during {stage}: {error}\n{tb}"
