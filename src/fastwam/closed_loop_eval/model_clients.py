"""Model clients used by FastWAM AAO closed-loop evaluation."""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


_REPO_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)


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

        supported_action_modes = {"delta6_abs_gripper", "absolute", "absolute_joint"}
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

    def infer(self, model_input: dict[str, Any]) -> dict[str, Any]:
        input_image = self._preprocess_images(model_input["images"])
        proprio_norm = self._normalize_proprio(model_input["proprio_raw"])
        instruction = str(model_input.get("instruction", self.instruction))
        context, context_mask, full_prompt = self._get_text_context(instruction)
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
        actions = self._format_actions(denorm_action, model_input)
        return {
            "action_format": self.output_action_format,
            "actions": actions.astype(np.float32, copy=False),
            "source": "fastwam",
            "action_mode": self.action_mode,
            "normalized_action_shape": list(action_norm.shape),
            "full_prompt": full_prompt,
            "instruction": instruction,
        }

    def _resolve_current_position(self, model_input: dict[str, Any]) -> np.ndarray:
        for key in ("current_position", "joint_position", "cartesian_position"):
            if key in model_input and model_input[key] is not None:
                current = np.asarray(model_input[key], dtype=np.float32).reshape(-1)
                break
        else:
            current = np.asarray(model_input["proprio_raw"], dtype=np.float32).reshape(-1)[:6]
        if current.size < 6:
            raise RuntimeError(f"Current position must contain at least 6 values, got {current.size}.")
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
        camera_tensors: list[torch.Tensor] = []
        for camera_key in self.image_shapes:
            if camera_key not in images:
                raise RuntimeError(f"Missing FastWAM camera '{camera_key}' in model input.")
            _, height, width = self.image_shapes[camera_key]
            rgb = np.asarray(images[camera_key], dtype=np.uint8)
            resized = Image.fromarray(rgb).resize((width, height), Image.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            camera_tensors.append(tensor)

        if self.concat_multi_camera == "horizontal":
            image = torch.cat(camera_tensors, dim=-1)
        elif self.concat_multi_camera == "vertical":
            image = torch.cat(camera_tensors, dim=-2)
        elif len(camera_tensors) == 1:
            image = camera_tensors[0]
        else:
            raise ValueError(f"Unsupported concat_multi_camera='{self.concat_multi_camera}'.")

        target_h, target_w = self.video_size
        if tuple(image.shape[-2:]) != (target_h, target_w):
            pil = Image.fromarray(image.permute(1, 2, 0).numpy().astype(np.uint8))
            pil = pil.resize((target_w, target_h), Image.BILINEAR)
            image = torch.from_numpy(np.asarray(pil, dtype=np.float32)).permute(2, 0, 1).contiguous()

        image = image * (2.0 / 255.0) - 1.0
        return image.unsqueeze(0).to(device=self.model.device, dtype=self.model.torch_dtype)

    def _normalize_proprio(self, proprio_raw: np.ndarray) -> torch.Tensor:
        proprio = torch.as_tensor(proprio_raw, dtype=torch.float32).reshape(1, -1)
        if proprio.shape[-1] != self.proprio_dim:
            raise RuntimeError(f"Expected proprio_raw length {self.proprio_dim}, got {proprio.shape[-1]}.")
        batch = {"state": {self.state_key: proprio.clone()}}
        batch = self.processor.normalizer.forward(batch)
        batch = self.processor.action_state_merger.forward(batch)
        return batch["state"].to(device=self.model.device, dtype=self.model.torch_dtype)

    def _denormalize_action(self, action_norm: torch.Tensor, proprio_norm: torch.Tensor) -> np.ndarray:
        if action_norm.ndim != 2:
            raise ValueError(f"Expected normalized action [T,D], got {tuple(action_norm.shape)}.")
        state_norm = proprio_norm.detach().to(device="cpu", dtype=torch.float32).reshape(1, 1, -1)
        batch = {
            "action": action_norm.unsqueeze(0).detach().to(device="cpu", dtype=torch.float32),
            "state": state_norm,
        }
        batch = self.processor.action_state_merger.backward(batch)
        batch = self.processor.normalizer.backward(batch)
        action_meta = self.processor.shape_meta["action"]
        state_meta = self.processor.shape_meta["state"]
        merged_batch = {
            "action": {
                meta["key"]: batch["action"][meta["key"]].squeeze(0)
                for meta in action_meta
            },
            "state": {
                meta["key"]: batch["state"][meta["key"]].squeeze(0)
                for meta in state_meta
            },
        }
        merged_batch = self.processor.action_state_merger.forward(merged_batch)
        return merged_batch["action"].detach().cpu().numpy().astype(np.float32)

    def _format_actions(self, denorm_action: np.ndarray, model_input: dict[str, Any]) -> np.ndarray:
        if self.action_mode == "delta6_abs_gripper":
            current_position = self._resolve_current_position(model_input)
            return self._delta_to_absolute(denorm_action, current_position)
        if self.action_mode in {"absolute", "absolute_joint"}:
            return np.asarray(denorm_action, dtype=np.float32)
        raise ValueError(f"Unsupported action_mode='{self.action_mode}'.")

    def _delta_to_absolute(self, action_delta: np.ndarray, current_cartesian: np.ndarray) -> np.ndarray:
        action = np.asarray(action_delta, dtype=np.float32).copy()
        current = np.asarray(current_cartesian, dtype=np.float32).reshape(-1)
        if current.size < 6:
            raise RuntimeError("current_cartesian must contain at least 6 values.")
        action[:, :6] = action[:, :6] + current[:6][None, :]
        return action

    def _get_text_context(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        cached = self._text_context_cache.get(instruction)
        if cached is not None:
            return cached
        context = self._load_text_context(instruction)
        self._text_context_cache[instruction] = context
        return context

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
