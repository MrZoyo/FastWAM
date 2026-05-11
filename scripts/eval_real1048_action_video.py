#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if (REPO_ROOT / "src").exists():
    sys.path.insert(0, str(REPO_ROOT / "src"))
else:
    # Allow running from repo root after copying this file into scripts/.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastwam.closed_loop_eval.model_clients import FastWAMModelClient, load_fastwam_config
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim


DEFAULT_CKPT = (
    "runs/real_1048_uncond_2cam224_1e-4/"
    "real1048_20k_wandb_20260508_202105/checkpoints/weights/step_020000.pt"
)
DEFAULT_STATS = (
    "runs/real_1048_uncond_2cam224_1e-4/"
    "real1048_20k_wandb_20260508_202105/dataset_stats.json"
)
DEFAULT_OUT = (
    "runs/real_1048_uncond_2cam224_1e-4/"
    "real1048_20k_wandb_20260508_202105/action_video_eval"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _pil_resample_bilinear() -> int:
    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def _to_hwc_uint8(x: torch.Tensor) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(x.shape)}")
    arr = x.detach().cpu()
    if arr.dtype != torch.uint8:
        arr = arr.clamp(0, 255).to(torch.uint8)
    return arr.permute(1, 2, 0).contiguous().numpy()


def _sample_camera_images(sample: dict[str, Any], t: int) -> dict[str, np.ndarray]:
    return {
        key: _to_hwc_uint8(value[t])
        for key, value in sample["images"].items()
    }


def _composite_frame(images: dict[str, np.ndarray], client: FastWAMModelClient) -> Image.Image:
    frames: list[np.ndarray] = []
    resample = _pil_resample_bilinear()
    for camera_key, shape in client.image_shapes.items():
        if camera_key not in images:
            raise KeyError(f"Missing camera key {camera_key}")
        _, height, width = shape
        frame = Image.fromarray(images[camera_key]).convert("RGB")
        frame = frame.resize((width, height), resample)
        frames.append(np.asarray(frame, dtype=np.uint8))

    if client.concat_multi_camera == "horizontal":
        arr = np.concatenate(frames, axis=1)
    elif client.concat_multi_camera == "vertical":
        arr = np.concatenate(frames, axis=0)
    elif len(frames) == 1:
        arr = frames[0]
    else:
        raise ValueError(f"Unsupported concat_multi_camera={client.concat_multi_camera!r}")

    target_h, target_w = client.video_size
    if arr.shape[:2] != (target_h, target_w):
        arr = np.asarray(Image.fromarray(arr).resize((target_w, target_h), resample), dtype=np.uint8)
    return Image.fromarray(arr)


def _gt_video_frames(
    sample: dict[str, Any],
    client: FastWAMModelClient,
    frame_indices: list[int],
) -> list[Image.Image]:
    return [
        _composite_frame(_sample_camera_images(sample, int(t)), client)
        for t in frame_indices
    ]


def _annotate(frame: Image.Image, label: str) -> Image.Image:
    frame = frame.convert("RGB")
    width, height = frame.size
    bar_h = 32
    out = Image.new("RGB", (width, height + bar_h), (20, 20, 20))
    out.paste(frame, (0, bar_h))
    draw = ImageDraw.Draw(out)
    draw.text((8, 9), label, fill=(245, 245, 245))
    return out


def _diff_image(gt: Image.Image, pred: Image.Image, scale: float) -> Image.Image:
    gt_arr = np.asarray(gt.convert("RGB"), dtype=np.float32)
    pred_arr = np.asarray(pred.convert("RGB").resize(gt.size, _pil_resample_bilinear()), dtype=np.float32)
    diff = np.abs(pred_arr - gt_arr)
    gray = np.clip(diff.mean(axis=2) * float(scale), 0, 255).astype(np.uint8)
    heat = np.zeros((*gray.shape, 3), dtype=np.uint8)
    heat[..., 0] = gray
    heat[..., 1] = np.clip(gray.astype(np.int16) // 2, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip(255 - gray.astype(np.int16), 0, 255).astype(np.uint8)
    return Image.fromarray(heat)


def _even(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def save_comparison_video(
    gt_frames: list[Image.Image],
    pred_frames: list[Image.Image],
    path: Path,
    *,
    fps: int,
    diff_scale: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=max(int(fps), 1),
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
    )
    try:
        for t, (gt, pred) in enumerate(zip(gt_frames, pred_frames)):
            pred = pred.convert("RGB").resize(gt.size, _pil_resample_bilinear())
            diff = _diff_image(gt, pred, scale=diff_scale)
            panels = [
                _annotate(gt, f"GT t={t}"),
                _annotate(pred, f"Pred t={t}"),
                _annotate(diff, f"Abs diff x{diff_scale:g}"),
            ]
            row = np.concatenate([np.asarray(panel) for panel in panels], axis=1)
            writer.append_data(_even(row))
    finally:
        writer.close()


def _video_metrics(gt_frames: list[Image.Image], pred_frames: list[Image.Image]) -> dict[str, Any]:
    pred_tensor = pil_frames_to_video_tensor([p.convert("RGB").resize(gt_frames[0].size) for p in pred_frames])
    gt_tensor = pil_frames_to_video_tensor(gt_frames)
    if pred_tensor.shape != gt_tensor.shape:
        raise ValueError(f"Video shape mismatch: pred={tuple(pred_tensor.shape)} gt={tuple(gt_tensor.shape)}")

    diff = pred_tensor - gt_tensor
    per_frame_mse = diff.pow(2).mean(dim=(0, 2, 3)).numpy()
    per_frame_mae = diff.abs().mean(dim=(0, 2, 3)).numpy()
    per_frame_psnr = 10.0 * np.log10(1.0 / (per_frame_mse + 1e-8))

    metrics = {
        "mae": float(diff.abs().mean().item()),
        "mse": float(diff.pow(2).mean().item()),
        "psnr": float(video_psnr(pred_tensor, gt_tensor)),
        "ssim": float(video_ssim(pred_tensor, gt_tensor)),
        "per_frame_mae": per_frame_mae.tolist(),
        "per_frame_mse": per_frame_mse.tolist(),
        "per_frame_psnr": per_frame_psnr.tolist(),
    }
    if pred_tensor.shape[1] > 1:
        pred_future = pred_tensor[:, 1:]
        gt_future = gt_tensor[:, 1:]
        future_diff = pred_future - gt_future
        metrics.update(
            {
                "future_mae_excl_t0": float(future_diff.abs().mean().item()),
                "future_mse_excl_t0": float(future_diff.pow(2).mean().item()),
                "future_psnr_excl_t0": float(video_psnr(pred_future, gt_future)),
                "future_ssim_excl_t0": float(video_ssim(pred_future, gt_future)),
            }
        )
    return metrics


def _load_raw_dataset(args: argparse.Namespace) -> BaseLerobotDataset:
    cfg = load_fastwam_config(args.config)
    if args.split == "train":
        data_cfg = cfg.data.train
        is_training = True
        val_prop = float(data_cfg.val_set_proportion)
    elif args.split == "all":
        data_cfg = cfg.data.val if cfg.data.get("val") is not None else cfg.data.train
        is_training = False
        val_prop = 0.0
    else:
        data_cfg = cfg.data.val if cfg.data.get("val") is not None else cfg.data.train
        is_training = False
        val_prop = float(data_cfg.val_set_proportion)

    dataset_dirs = [args.dataset_dir] if args.dataset_dir else list(data_cfg.dataset_dirs)
    shape_meta = OmegaConf.to_container(data_cfg.shape_meta, resolve=True)
    return BaseLerobotDataset(
        dataset_dirs=dataset_dirs,
        shape_meta=shape_meta,
        obs_size=int(data_cfg.num_frames),
        action_size=int(data_cfg.num_frames) - 1,
        val_set_proportion=val_prop,
        is_training_set=is_training,
        global_sample_stride=int(data_cfg.global_sample_stride),
    )


def _parse_indices(text: str | None) -> list[int] | None:
    if text is None or text.strip() == "":
        return None
    return [int(part) for part in text.replace(" ", "").split(",") if part]


def _select_indices(
    dataset: BaseLerobotDataset,
    *,
    num_samples: int,
    seed: int,
    sample_indices: list[int] | None,
    min_xyz_motion: float,
    max_tries: int,
) -> list[int]:
    if sample_indices is not None:
        return sample_indices[:num_samples]

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    seen: set[int] = set()
    tries = 0
    while len(selected) < num_samples and tries < max_tries:
        tries += 1
        idx = int(rng.integers(0, len(dataset)))
        if idx in seen:
            continue
        seen.add(idx)
        sample = dataset[idx]
        action = sample["action"]["default"].detach().cpu().float().numpy()
        valid = ~sample["action_is_pad"].detach().cpu().bool().numpy()
        motion = float(np.linalg.norm(action[valid, :3], axis=1).sum()) if valid.any() else 0.0
        if motion >= float(min_xyz_motion):
            selected.append(idx)

    while len(selected) < num_samples:
        idx = int(rng.integers(0, len(dataset)))
        if idx not in selected:
            selected.append(idx)
    return selected


def _normalize_raw_action(
    client: FastWAMModelClient,
    action_raw: torch.Tensor,
    state_raw: torch.Tensor,
) -> torch.Tensor:
    batch = {
        "action": {"default": action_raw.detach().cpu().float().clone()},
        "state": {"default": state_raw.detach().cpu().float().clone()},
    }
    batch = client.processor.action_state_transform(batch)
    batch = client.processor.normalizer.forward(batch)
    batch = client.processor.action_state_merger.forward(batch)
    return batch["action"]


def _plot_action_errors(
    abs_err: np.ndarray,
    xyz_l2: np.ndarray,
    sample_labels: list[str],
    output_dir: Path,
) -> None:
    steps = np.arange(abs_err.shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    titles = ["x abs error", "y abs error", "z abs error", "xyz L2 error"]
    series = [abs_err[:, :, 0], abs_err[:, :, 1], abs_err[:, :, 2], xyz_l2]
    for ax, title, values in zip(axes.ravel(), titles, series):
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        ax.plot(steps, mean, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("action step")
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("abs error")
    axes[1, 0].set_ylabel("error")
    fig.savefig(output_dir / "action_error_xyz_total.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, max(3, 0.45 * len(sample_labels))), constrained_layout=True)
    im = ax.imshow(xyz_l2, aspect="auto", interpolation="nearest")
    ax.set_title("xyz L2 error per sample and action step")
    ax.set_xlabel("action step")
    ax.set_ylabel("sample")
    ax.set_yticks(np.arange(len(sample_labels)))
    ax.set_yticklabels(sample_labels)
    fig.colorbar(im, ax=ax, label="xyz L2")
    fig.savefig(output_dir / "action_error_xyz_l2_heatmap.png", dpi=180)
    plt.close(fig)


def _plot_action_value_curves(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, str]:
    """Plot actual GT/Pred action values so scalar losses are interpretable."""
    plot_dir = output_dir / "action_value_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    by_sample: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(int(row["sample_order"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []

    def add_summary(
        *,
        sample_order: int,
        dataset_index: int,
        space: str,
        dim: str,
        gt: np.ndarray,
        pred: np.ndarray,
    ) -> None:
        err = pred - gt
        summary_rows.append(
            {
                "sample_order": sample_order,
                "dataset_index": dataset_index,
                "space": space,
                "dim": dim,
                "gt_mean": float(gt.mean()),
                "gt_min": float(gt.min()),
                "gt_max": float(gt.max()),
                "pred_mean": float(pred.mean()),
                "pred_min": float(pred.min()),
                "pred_max": float(pred.max()),
                "mae": float(np.abs(err).mean()),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "max_abs_err": float(np.abs(err).max()),
            }
        )

    for sample_order, sample_rows in sorted(by_sample.items()):
        sample_rows = sorted(sample_rows, key=lambda row: int(row["step"]))
        dataset_index = int(sample_rows[0]["dataset_index"])
        steps = np.asarray([int(row["step"]) for row in sample_rows], dtype=np.int64)

        fig, axes = plt.subplots(4, 2, figsize=(13, 10), constrained_layout=True)
        for row_idx, dim in enumerate(("x", "y", "z", "gripper")):
            gt_key = "gt_gripper" if dim == "gripper" else f"gt_{dim}"
            pred_key = "pred_gripper" if dim == "gripper" else f"pred_{dim}"
            gt = np.asarray([float(row[gt_key]) for row in sample_rows], dtype=np.float64)
            pred = np.asarray([float(row[pred_key]) for row in sample_rows], dtype=np.float64)
            err = pred - gt

            ax = axes[row_idx, 0]
            ax.plot(steps, gt, label="GT", linewidth=2)
            ax.plot(steps, pred, label="Pred", linewidth=2)
            ax.set_title(f"delta action {dim}: GT vs Pred")
            ax.set_xlabel("action step")
            ax.set_ylabel(dim)
            ax.grid(alpha=0.25)
            if row_idx == 0:
                ax.legend(loc="best")

            err_ax = axes[row_idx, 1]
            err_ax.plot(steps, err, color="tab:red", linewidth=1.8)
            err_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            err_ax.set_title(f"delta action {dim}: Pred - GT")
            err_ax.set_xlabel("action step")
            err_ax.set_ylabel("error")
            err_ax.grid(alpha=0.25)

            add_summary(
                sample_order=sample_order,
                dataset_index=dataset_index,
                space="delta6_abs_gripper",
                dim=dim,
                gt=gt,
                pred=pred,
            )

        fig.suptitle(f"sample {sample_order:03d} / dataset idx {dataset_index}: actual action values")
        delta_plot = plot_dir / f"sample_{sample_order:03d}_idx_{dataset_index:06d}_delta_action_values.png"
        fig.savefig(delta_plot, dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(3, 2, figsize=(13, 7.5), constrained_layout=True)
        for row_idx, dim in enumerate(("x", "y", "z")):
            gt = np.asarray([float(row[f"gt_abs_{dim}"]) for row in sample_rows], dtype=np.float64)
            pred = np.asarray([float(row[f"pred_abs_{dim}"]) for row in sample_rows], dtype=np.float64)
            err = pred - gt

            ax = axes[row_idx, 0]
            ax.plot(steps, gt, label="GT abs", linewidth=2)
            ax.plot(steps, pred, label="Pred abs", linewidth=2)
            ax.set_title(f"absolute target {dim}: GT vs Pred")
            ax.set_xlabel("action step")
            ax.set_ylabel(dim)
            ax.grid(alpha=0.25)
            if row_idx == 0:
                ax.legend(loc="best")

            err_ax = axes[row_idx, 1]
            err_ax.plot(steps, err, color="tab:red", linewidth=1.8)
            err_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            err_ax.set_title(f"absolute target {dim}: Pred - GT")
            err_ax.set_xlabel("action step")
            err_ax.set_ylabel("error")
            err_ax.grid(alpha=0.25)

            add_summary(
                sample_order=sample_order,
                dataset_index=dataset_index,
                space="absolute_xyz",
                dim=dim,
                gt=gt,
                pred=pred,
            )

        fig.suptitle(f"sample {sample_order:03d} / dataset idx {dataset_index}: absolute x/y/z targets")
        abs_plot = plot_dir / f"sample_{sample_order:03d}_idx_{dataset_index:06d}_absolute_xyz_values.png"
        fig.savefig(abs_plot, dpi=170)
        plt.close(fig)

    summary_csv = plot_dir / "action_value_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    readme = plot_dir / "README_action_values.md"
    readme.write_text(
        "# Action Value Plots\n\n"
        "`*_delta_action_values.png` shows GT and Pred actual action values for "
        "delta x/y/z and gripper, plus Pred-GT error.\n\n"
        "`*_absolute_xyz_values.png` shows x/y/z after adding the current state to "
        "the delta action. For x/y/z, absolute-space error equals delta-space error, "
        "but the absolute targets are easier to interpret physically.\n\n"
        "`action_value_summary.csv` contains gt/pred min/mean/max plus "
        "MAE/RMSE/max_abs_err for each plotted dimension.\n",
        encoding="utf-8",
    )
    return {
        "action_value_plot_dir": str(plot_dir),
        "action_value_summary_csv": str(summary_csv),
        "action_value_readme": str(readme),
    }


def _summarize_actions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    err = np.asarray([[r["err_x"], r["err_y"], r["err_z"]] for r in rows], dtype=np.float64)
    abs_err = np.abs(err)
    xyz_l2 = np.asarray([r["xyz_l2"] for r in rows], dtype=np.float64)
    l2_6d = np.asarray([r["l2_6d"] for r in rows], dtype=np.float64)
    l2_7d = np.asarray([r["l2_7d"] for r in rows], dtype=np.float64)
    return {
        "num_action_points": int(len(rows)),
        "xyz_mean_abs": {
            "x": float(abs_err[:, 0].mean()),
            "y": float(abs_err[:, 1].mean()),
            "z": float(abs_err[:, 2].mean()),
        },
        "xyz_rmse": {
            "x": float(math.sqrt(np.mean(err[:, 0] ** 2))),
            "y": float(math.sqrt(np.mean(err[:, 1] ** 2))),
            "z": float(math.sqrt(np.mean(err[:, 2] ** 2))),
        },
        "xyz_max_abs": {
            "x": float(abs_err[:, 0].max()),
            "y": float(abs_err[:, 1].max()),
            "z": float(abs_err[:, 2].max()),
        },
        "xyz_l2_mean": float(xyz_l2.mean()),
        "xyz_l2_median": float(np.median(xyz_l2)),
        "xyz_l2_p95": float(np.percentile(xyz_l2, 95)),
        "l2_6d_mean": float(l2_6d.mean()),
        "l2_7d_mean": float(l2_7d.mean()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_raw_dataset(args)
    selected = _select_indices(
        dataset,
        num_samples=args.num_samples,
        seed=args.seed,
        sample_indices=_parse_indices(args.sample_indices),
        min_xyz_motion=args.min_xyz_motion,
        max_tries=args.max_sample_tries,
    )

    client = FastWAMModelClient(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_stats_path=args.dataset_stats,
        text_cache_dir=args.text_cache_dir,
        instruction=args.instruction,
        action_horizon=args.action_horizon,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        rand_device=args.rand_device,
        preload_text_context=False,
        output_action_format="joint_absolute",
    )

    frame_stride = max(1, args.action_horizon // max(args.num_video_frames - 1, 1))
    frame_indices = [min(i * frame_stride, args.action_horizon) for i in range(args.num_video_frames)]

    action_rows: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    abs_err_for_plot = np.full((len(selected), args.action_horizon, 3), np.nan, dtype=np.float32)
    xyz_l2_for_plot = np.full((len(selected), args.action_horizon), np.nan, dtype=np.float32)

    for sample_order, dataset_index in enumerate(selected):
        sample = dataset[int(dataset_index)]
        instruction = str(args.instruction_override or sample.get("task") or args.instruction)
        raw_action = sample["action"]["default"].detach().cpu().float()
        raw_state = sample["state"]["default"].detach().cpu().float()
        valid = ~sample["action_is_pad"].detach().cpu().bool().numpy()
        horizon = min(args.action_horizon, raw_action.shape[0])

        images0 = _sample_camera_images(sample, 0)
        input_image = client._preprocess_images(images0)
        proprio_raw = raw_state[0].numpy()
        proprio_norm = client._normalize_proprio(proprio_raw)
        context, context_mask, full_prompt = client._get_text_context(instruction)

        gt_action_norm = _normalize_raw_action(client, raw_action[: args.action_horizon], raw_state)
        action_condition = None
        if args.video_action_source == "gt":
            action_condition = gt_action_norm.to(device=client.model.device, dtype=client.model.torch_dtype)
        elif args.video_action_source == "pred":
            with torch.no_grad():
                action_only = client.model.infer_action(
                    prompt=None,
                    input_image=input_image,
                    action_horizon=args.action_horizon,
                    proprio=proprio_norm,
                    context=context,
                    context_mask=context_mask,
                    num_inference_steps=args.num_inference_steps,
                    seed=args.seed + sample_order,
                    rand_device=args.rand_device,
                )["action"]
            action_condition = action_only.to(device=client.model.device, dtype=client.model.torch_dtype)

        with torch.no_grad():
            output = client.model.infer_joint(
                prompt=None,
                input_image=input_image,
                num_video_frames=args.num_video_frames,
                action_horizon=args.action_horizon,
                action=action_condition,
                proprio=proprio_norm,
                context=context,
                context_mask=context_mask,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed + sample_order,
                rand_device=args.rand_device,
                tiled=False,
                test_action_with_infer_action=False,
            )

        pred_norm = output["action"].detach().cpu().float()
        pred_delta = client._denormalize_action(pred_norm, proprio_norm)
        pred_abs = client._delta_to_absolute(pred_delta, proprio_raw[:6])
        gt_delta = raw_action[:horizon].numpy().astype(np.float32)
        gt_abs = gt_delta.copy()
        gt_abs[:, :6] = gt_abs[:, :6] + proprio_raw[:6][None, :]

        pred_delta = pred_delta[:horizon]
        pred_abs = pred_abs[:horizon]
        valid_h = valid[:horizon]

        for step in range(horizon):
            if not bool(valid_h[step]):
                continue
            diff = pred_delta[step] - gt_delta[step]
            row = {
                "sample_order": sample_order,
                "dataset_index": int(dataset_index),
                "step": step,
                "valid": True,
                "gt_x": float(gt_delta[step, 0]),
                "gt_y": float(gt_delta[step, 1]),
                "gt_z": float(gt_delta[step, 2]),
                "pred_x": float(pred_delta[step, 0]),
                "pred_y": float(pred_delta[step, 1]),
                "pred_z": float(pred_delta[step, 2]),
                "err_x": float(diff[0]),
                "err_y": float(diff[1]),
                "err_z": float(diff[2]),
                "abs_err_x": float(abs(diff[0])),
                "abs_err_y": float(abs(diff[1])),
                "abs_err_z": float(abs(diff[2])),
                "xyz_l2": float(np.linalg.norm(diff[:3])),
                "l2_6d": float(np.linalg.norm(diff[:6])),
                "l2_7d": float(np.linalg.norm(diff[:7])),
                "gt_gripper": float(gt_delta[step, 6]),
                "pred_gripper": float(pred_delta[step, 6]),
                "err_gripper": float(diff[6]),
                "gt_abs_x": float(gt_abs[step, 0]),
                "gt_abs_y": float(gt_abs[step, 1]),
                "gt_abs_z": float(gt_abs[step, 2]),
                "pred_abs_x": float(pred_abs[step, 0]),
                "pred_abs_y": float(pred_abs[step, 1]),
                "pred_abs_z": float(pred_abs[step, 2]),
            }
            action_rows.append(row)
            abs_err_for_plot[sample_order, step] = [row["abs_err_x"], row["abs_err_y"], row["abs_err_z"]]
            xyz_l2_for_plot[sample_order, step] = row["xyz_l2"]

        gt_frames = _gt_video_frames(sample, client, frame_indices)
        pred_frames = [frame.convert("RGB").resize(gt_frames[0].size) for frame in output["video"][: len(gt_frames)]]
        video_metrics = _video_metrics(gt_frames, pred_frames)
        video_path = output_dir / f"sample_{sample_order:03d}_idx_{int(dataset_index):06d}_gt_pred_diff.mp4"
        save_comparison_video(
            gt_frames,
            pred_frames,
            video_path,
            fps=args.video_fps,
            diff_scale=args.diff_scale,
        )

        sample_action_rows = [r for r in action_rows if r["sample_order"] == sample_order]
        sample_action_summary = _summarize_actions(sample_action_rows) if sample_action_rows else {}
        sample_summary = {
            "sample_order": sample_order,
            "dataset_index": int(dataset_index),
            "instruction": instruction,
            "full_prompt": full_prompt,
            "video_path": str(video_path),
            "frame_indices": frame_indices,
            "action_summary": sample_action_summary,
            "video_metrics": video_metrics,
        }
        sample_summaries.append(sample_summary)
        print(
            f"[{sample_order + 1}/{len(selected)}] idx={dataset_index} "
            f"xyz_l2_mean={sample_action_summary.get('xyz_l2_mean', float('nan')):.6f} "
            f"future_psnr={video_metrics.get('future_psnr_excl_t0', video_metrics['psnr']):.3f} "
            f"video={video_path}",
            flush=True,
        )

    action_csv = output_dir / "action_errors.csv"
    with action_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(action_rows[0].keys()))
        writer.writeheader()
        writer.writerows(action_rows)

    _plot_action_errors(
        abs_err=abs_err_for_plot,
        xyz_l2=xyz_l2_for_plot,
        sample_labels=[f"{i}:idx{idx}" for i, idx in enumerate(selected)],
        output_dir=output_dir,
    )
    action_value_artifacts = {}
    if not args.no_action_value_plots:
        action_value_artifacts = _plot_action_value_curves(action_rows, output_dir)

    video_metric_keys = [
        "mae",
        "mse",
        "psnr",
        "ssim",
        "future_mae_excl_t0",
        "future_mse_excl_t0",
        "future_psnr_excl_t0",
        "future_ssim_excl_t0",
    ]
    video_summary = {}
    for key in video_metric_keys:
        values = [s["video_metrics"][key] for s in sample_summaries if key in s["video_metrics"]]
        if values:
            video_summary[key] = float(np.mean(values))

    summary = {
        "output_dir": str(output_dir),
        "config": args.config,
        "checkpoint": args.checkpoint,
        "dataset_stats": args.dataset_stats,
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "selected_indices": selected,
        "num_samples": len(selected),
        "action_horizon": args.action_horizon,
        "num_video_frames": args.num_video_frames,
        "num_inference_steps": args.num_inference_steps,
        "video_action_source": args.video_action_source,
        "action_space_note": (
            "action_errors.csv compares denormalized model action with raw dataset action in "
            "delta6_abs_gripper space. Adding current state to both gives the absolute x/y/z "
            "columns and leaves the error unchanged for the first 6 dims."
        ),
        "action_summary": _summarize_actions(action_rows),
        "video_summary": video_summary,
        "sample_summaries": sample_summaries,
        "artifacts": {
            "action_csv": str(action_csv),
            "action_plot": str(output_dir / "action_error_xyz_total.png"),
            "action_heatmap": str(output_dir / "action_error_xyz_l2_heatmap.png"),
            **action_value_artifacts,
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, ensure_ascii=False, indent=2)

    print(json.dumps(_jsonable({"summary_path": str(summary_path), **summary["action_summary"], **video_summary}), indent=2))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate real_1048 FastWAM action and future-video errors.")
    parser.add_argument("--config", default="configs/task/real_1048_uncond_2cam224_1e-4.yaml")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--dataset-stats", default=DEFAULT_STATS)
    parser.add_argument("--dataset-dir", default="/DATA/disk1/zoyo/real_1048")
    parser.add_argument("--text-cache-dir", default="data/text_embeds_cache/real_1048")
    parser.add_argument("--output-dir", default=DEFAULT_OUT)
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-indices", default=None, help="Comma-separated dataset indices; overrides random sampling.")
    parser.add_argument("--min-xyz-motion", type=float, default=0.02)
    parser.add_argument("--max-sample-tries", type=int, default=200)
    parser.add_argument("--instruction", default="open the door")
    parser.add_argument("--instruction-override", default=None)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--num-video-frames", type=int, default=9)
    parser.add_argument("--video-action-source", choices=("gt", "pred", "none"), default="gt")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-fps", type=int, default=8)
    parser.add_argument("--diff-scale", type=float, default=4.0)
    parser.add_argument("--no-action-value-plots", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if args.num_video_frames <= 1 or args.num_video_frames % 4 != 1:
        raise ValueError("--num-video-frames must be >1 and satisfy T % 4 == 1")
    run(args)


if __name__ == "__main__":
    main()
