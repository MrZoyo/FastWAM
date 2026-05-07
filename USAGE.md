# FastWAM 本地使用简明说明

## 进入项目和虚拟环境

```bash
cd /DATA/disk1/zoyo/FastWAM
source .venv/bin/activate
```

不想激活环境时，也可以直接用 `.venv/bin/python`、`.venv/bin/wandb`。

如果需要重建环境：

```bash
uv sync
```

## 常用环境变量

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/DATA/disk1/zoyo/FastWAM/checkpoints
export CUDA_VISIBLE_DEVICES=4,5,6,7
```

当前本地 Wan2.2、ActionDiT 初始化权重、release checkpoint 都在 `checkpoints/` 下。训练时建议加
`model.redirect_common_files=false`，避免模型 loader 重定向到缺失的公共 safetensors 路径。

## wandb

```bash
.venv/bin/wandb login
.venv/bin/wandb status
```

训练时加：

```bash
wandb.enabled=true wandb.project=fast-wam wandb.mode=online
```

## LIBERO 文本 embedding 预计算

通常只需要做一次，cache 会写到 `data/text_embeds_cache/libero`：

```bash
CUDA_VISIBLE_DEVICES=4 \
DIFFSYNTH_MODEL_BASE_PATH=/DATA/disk1/zoyo/FastWAM/checkpoints \
.venv/bin/python -u scripts/precompute_text_embeds.py \
  task=libero_uncond_2cam224_1e-4 \
  model.redirect_common_files=false
```

## LIBERO 从初始化权重训练

4 卡示例：

```bash
RUN_ID=libero_init_20k_$(date +%Y%m%d_%H%M%S) \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
DIFFSYNTH_MODEL_BASE_PATH=/DATA/disk1/zoyo/FastWAM/checkpoints \
bash scripts/train_zero1.sh 4 \
  task=libero_uncond_2cam224_1e-4 \
  model.redirect_common_files=false \
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

说明：

- 不加 `resume=...` 就不是从 release checkpoint 继续训练。
- `save_every=5000` 会每 5000 step 保存一次完整训练状态，便于中断后恢复，但每个 checkpoint 可能很大。
- 输出目录在 `runs/libero_uncond_2cam224_1e-4/<RUN_ID>/`。

## 查看训练

```bash
nvidia-smi
pgrep -af "scripts/train.py|train_zero1|accelerate"
tail -f runs/libero_uncond_2cam224_1e-4/<RUN_ID>/wandb/*/files/output.log
```

如果后台运行，建议把命令放进 `nohup` 或稳定的任务管理器里，避免终端或 tmux session 被清掉导致训练中断。
