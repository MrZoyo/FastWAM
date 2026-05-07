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
  wandb.enabled=false
```

To use 8 GPUs, set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` and change `scripts/train_zero1.sh 4` to `scripts/train_zero1.sh 8`.

Weights/data are symlinked from the project root:

- `checkpoints -> /DATA/disk7/Lyle/FastWAM/checkpoints`
- `data -> /DATA/disk7/Lyle/FastWAM/data`
- `runs -> /DATA/disk7/Lyle/FastWAM/runs`
