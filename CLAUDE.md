# FastWAM Notes

## h200-1 Normal LIBERO Training

Use multi-GPU training. A 1-step smoke test passed with 4 GPUs; single-GPU training can OOM during the Adam optimizer step because ZeRO1 does not shard optimizer state across a single GPU.

```bash
cd /home/Lyle/Projects/FastWAM

RUN_ID=libero_init_20k_$(date +%Y%m%d_%H%M%S) \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DIFFSYNTH_MODEL_BASE_PATH=$PWD/checkpoints \
bash scripts/train_zero1.sh 4 \
  task=libero_uncond_2cam224_1e-4 \
  model.redirect_common_files=false \
  +data.train.pretrained_norm_stats=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  max_steps=20000 \
  gradient_accumulation_steps=2 \
  save_every=5000 \
  eval_every=5000 \
  log_every=10 \
  wandb.enabled=true \
  wandb.project=fast-wam \
  wandb.name=libero_20k_from_init \
  wandb.mode=online
```

To use 8 GPUs, set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` and change `scripts/train_zero1.sh 4` to `scripts/train_zero1.sh 8`.

Weights/data are symlinked from the project root:

- `checkpoints -> /DATA/disk7/Lyle/FastWAM/checkpoints`
- `data -> /DATA/disk7/Lyle/FastWAM/data`
- `runs -> /DATA/disk7/Lyle/FastWAM/runs`

## real_1048 Action/Video Eval

Use this script to compare `step_020000.pt` action predictions against real_1048 ground-truth actions and to visualize predicted future frames against GT future frames:

```bash
cd /home/Lyle/Projects/FastWAM

.venv/bin/python scripts/eval_real1048_action_video.py \
  --device cuda:2 \
  --num-samples 12 \
  --num-inference-steps 10 \
  --video-action-source gt
```

Default inputs:

- checkpoint: `runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/checkpoints/weights/step_020000.pt`
- dataset stats: `runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/dataset_stats.json`
- dataset: `/DATA/disk1/zoyo/real_1048`
- output root: `runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/action_video_eval/<timestamp>`

Output files:

- `summary.json`: aggregate action/video metrics and artifact paths.
- `action_errors.csv`: per-sample, per-step GT/Pred action values and errors. It includes delta-space columns like `gt_x`, `pred_x`, `err_x` and absolute target columns like `gt_abs_x`, `pred_abs_x`.
- `action_error_xyz_total.png`: mean/std action error curves for x/y/z and xyz L2.
- `action_error_xyz_l2_heatmap.png`: sample-by-step xyz L2 error heatmap.
- `action_value_plots/*_delta_action_values.png`: per-sample actual GT vs Pred delta action curves for x/y/z and gripper, with Pred-GT error. These make the scalar action loss easier to interpret.
- `action_value_plots/*_absolute_xyz_values.png`: per-sample GT vs Pred absolute x/y/z target curves after adding current state to delta action.
- `action_value_plots/action_value_summary.csv`: min/mean/max GT/Pred plus MAE/RMSE/max_abs_err for each plotted action dimension.
- `sample_*_gt_pred_diff.mp4`: visual comparison video. Columns are GT future frame, predicted future frame, and amplified absolute pixel difference. Inside each column the two cameras are concatenated horizontally: `head_left` then `right_wrist_left`.

Notes:

- Action comparison is in `delta6_abs_gripper` space. For x/y/z, adding current state to both GT and Pred produces absolute target columns but leaves the error unchanged.
- `--video-action-source gt` matches the trainer eval style: future video is generated conditioned on GT action, while the action prediction is still compared against GT action.
- `--video-action-source pred` conditions future video on the predicted action; this is useful for seeing compounding action/video mismatch.
- Use `--sample-indices 976,8469,...` for reproducible specific samples.
- Use a lower `--video-fps` if the 9-frame videos are too short to inspect, or repeat frames after generation for a longer playback without changing model output.
