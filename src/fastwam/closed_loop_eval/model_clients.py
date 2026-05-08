"""Model clients used by FastWAM AAO closed-loop evaluation."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json


class BaseModelClient(ABC):
    @abstractmethod
    def infer(self, model_input: dict[str, Any]) -> dict[str, Any]:
        """Return a chunked action payload with cartesian_absolute actions."""

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
        self.call_index = 0

        if self.action_mode != "delta6_abs_gripper":
            raise ValueError("Only action_mode='delta6_abs_gripper' is currently supported.")

        self.cfg = OmegaConf.load(self.config_path)
        self.processor = instantiate(self.cfg.data.val.processor)
        stats = load_dataset_stats_from_json(str(self.dataset_stats_path))
        self.processor.set_normalizer_from_stats(stats)
        self.processor.eval()

        if self.text_cache_dir is None:
            self.text_cache_dir = Path(str(self.cfg.data.val.text_embedding_cache_dir))
        self.text_cache_dir = self.text_cache_dir.expanduser()

        mixed_precision = str(self.cfg.get("mixed_precision", "bf16"))
        model_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
        if mixed_precision in ("no", "fp32", "float32"):
            model_dtype = torch.float32

        self.model = instantiate(self.cfg.model, model_dtype=model_dtype, device=self.device)
        self.model.load_checkpoint(str(self.checkpoint_path))
        self.model.eval()
        self.context, self.context_mask, self.full_prompt = self._load_text_context(self.instruction)

        train_cfg = self.cfg.data.val
        self.video_size = tuple(int(x) for x in train_cfg.video_size)
        self.concat_multi_camera = str(train_cfg.get("concat_multi_camera", "horizontal"))
        self.image_shapes = {
            str(meta["key"]): tuple(int(v) for v in meta["shape"])
            for meta in train_cfg.shape_meta.images
        }

    def infer(self, model_input: dict[str, Any]) -> dict[str, Any]:
        input_image = self._preprocess_images(model_input["images"])
        proprio_norm = self._normalize_proprio(model_input["proprio_raw"])
        seed = None if self.seed is None else self.seed + self.call_index
        self.call_index += 1

        with torch.no_grad():
            output = self.model.infer_action(
                prompt=None,
                input_image=input_image,
                action_horizon=self.action_horizon,
                proprio=proprio_norm,
                context=self.context,
                context_mask=self.context_mask,
                num_inference_steps=self.num_inference_steps,
                seed=seed,
                rand_device=self.rand_device,
            )
        action_norm = output["action"].detach().to(device="cpu", dtype=torch.float32)
        denorm_delta = self._denormalize_action(action_norm, proprio_norm)
        absolute = self._delta_to_absolute(
            denorm_delta,
            np.asarray(model_input["cartesian_position"], dtype=np.float32),
        )
        return {
            "action_format": "cartesian_absolute",
            "actions": absolute.astype(np.float32, copy=False),
            "source": "fastwam",
            "action_mode": self.action_mode,
            "normalized_action_shape": list(action_norm.shape),
            "full_prompt": self.full_prompt,
        }

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
        batch = {"state": {"default": proprio.clone()}}
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

    def _delta_to_absolute(self, action_delta: np.ndarray, current_cartesian: np.ndarray) -> np.ndarray:
        action = np.asarray(action_delta, dtype=np.float32).copy()
        current = np.asarray(current_cartesian, dtype=np.float32).reshape(-1)
        if current.size < 6:
            raise RuntimeError("current_cartesian must contain at least 6 values.")
        action[:, :6] = action[:, :6] + current[:6][None, :]
        return action

    def _load_text_context(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor, str]:
        full_prompt = DEFAULT_PROMPT.format(task=instruction)
        hashed = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()
        cache_path = self.text_cache_dir / f"{hashed}.t5_len{int(self.cfg.data.val.context_len)}.wan22ti2v5b.pt"
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
