# FastWAM

Official codebase for **Fast-WAM: Do World Action Models Need Test-time Future Imagination?**

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2603.16666-b31b1b.svg)](https://arxiv.org/abs/2603.16666)
[![Project Page](https://img.shields.io/badge/Project_Page-Fast--WAM-2ea44f.svg)](https://yuantianyuan01.github.io/FastWAM/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/yuanty/fastwam)
[![Hugging Face Dataset - LIBERO](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20LIBERO-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![Hugging Face Dataset - RoboTwin](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20RoboTwin-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

This repository contains the training and evaluation code for FastWAM on LIBERO / RoboTwin.

## AAO Closed-Loop Integration

This workspace also contains a local integration of FastWAM with
`auto-atomic-operation` (AAO) for closed-loop simulator validation. The current
bridge supports the GS open-door Airbot Play setup and the non-GS P7 cup task.
The integration is intentionally kept separate from the training pipeline:

- AAO is vendored as `third_party/auto-atomic-operation` and pinned to the
  latest DISCOVER fork commit used here. The current pin is `449119b`, which
  includes the upstream rename commit `d1530c7`.
- Gaussian rendering support is vendored as `third_party/GaussianRenderer`.
- The FastWAM closed-loop bridge lives in `src/fastwam/closed_loop_eval/`.
- Single-episode evaluation entrypoint:
  `scripts/run_aao_closed_loop_eval.py`.
- Multi-door/background GS sweep entrypoint:
  `scripts/run_aao_open_door_gs_sweep.py`.
- Pred/VAE-recon/actual simulator comparison video entrypoint:
  `scripts/run_aao_visual_rollout.py`.
- Batch benchmark entrypoint:
  `scripts/run_aao_benchmark.py`; see `docs/aao_benchmark.md`.

The default open-door task is `open_door_airbot_play_gs`. The current bridge
uses the mix 20k checkpoint by default:

```text
runs/mix_uncond_2cam224_1e-4/mix_uncond_20k_20260507_024400/checkpoints/weights/step_020000.pt
```

Model run directories under `runs/` are intentionally ignored by git. Copy or
download the corresponding `config.yaml`, `dataset_stats.json`, text embedding
cache, and checkpoint before using `--model-client fastwam`; use
`--fastwam-config`, `--dataset-stats`, `--text-cache-dir`, and `--checkpoint`
to point at a different run.

Control timing is handled by `--sim-loop-frequency`:

- `--sim-loop-frequency 0` means lockstep mode. The runner calls AAO
  `update(action)` synchronously, then reads the latest observation.
- `--sim-loop-frequency >0` means continuous mode. AAO runs a background
  simulation loop at that frequency, while the runner updates the current
  target action with `set_cartesian_action()` and replans from new observations.

For the mix/open-door data, `/home/zoyo/mix/meta/info.json` reports 20Hz
actions, while AAO open-door runs at 100Hz. Therefore the default closed-loop
settings are:

```bash
.venv/bin/python -B scripts/run_aao_closed_loop_eval.py \
  --model-client fastwam \
  --task open_door_airbot_play_gs \
  --action-horizon 32 \
  --stride 8 \
  --action-repeat 5 \
  --gripper-min 0.02 \
  --gripper-max 0.0945 \
  --sim-loop-frequency 0
```

The batch benchmark mode currently supports `open_door_airbot_play_gs` and
`cup_on_coaster_gs_airbot_p7`. Both profiles use 7D EEF pose + gripper model
actions and send `cartesian_absolute` commands to AAO. The cup profile uses 8D
joint + gripper proprio as model input, but it does not use the AAO P7 v3 UMI
operator or `joint_absolute` control. For multi-env and multi-model-GPU
benchmark usage, see `docs/aao_benchmark.md`.

Important caveat: AAO `final_success` alone is not a reliable open-door
criterion for the current setup. Always inspect `multicam.mp4` and the
MuJoCo diagnostics in `client_trace.json.gz` / `aggregate_summary.json`,
especially `door_hinge` and `handle_hinge` deltas.

For visual debugging of the model's imagined future against VAE reconstruction
and actual AAO observations, use the visual rollout script. `--frame-sampling
sim-update` writes one output frame per AAO update; for two 32-action windows
with `--action-repeat 5`, this produces 320 frames:

```bash
.venv/bin/python -B scripts/run_aao_visual_rollout.py \
  --task open_door_airbot_play_gs \
  --num-windows 2 \
  --action-horizon 32 \
  --action-repeat 5 \
  --num-video-frames 9 \
  --frame-sampling sim-update \
  --output-dir runs/aao_closed_loop/mix20k_open_door_gs_2win_visual_simupdate
```

The working notes for this integration are:

- `docs/260508-auto-atomic-open-door-plan.md`
- `docs/260508-auto-atomic-open-door-progress.md`

### AAO Setup

For a fresh checkout, clone submodules first:

```bash
git clone --recurse-submodules https://github.com/MrZoyo/FastWAM.git
cd FastWAM

# If the repo was cloned without --recurse-submodules:
git submodule update --init --recursive
```

Install the normal FastWAM environment first, then add AAO and the optional
GS dependencies into the same Python environment. AAO itself is installed from
the local submodule, while GaussianRenderer is installed from the pinned local
submodule instead of downloading a floating Git dependency:

```bash
uv pip install --python .venv/bin/python \
  -e "third_party/auto-atomic-operation[mujoco]" \
  -e "third_party/GaussianRenderer[shs,mujoco]" \
  gsplat==1.5.3 \
  ninja \
  natsort \
  PyOpenGL_accelerate
```

If you use conda instead of `.venv`, replace `.venv/bin/python` with the
Python executable from that environment.

The AAO MuJoCo meshes are managed by Git LFS inside the AAO submodule. Pull
them once after cloning submodules:

```bash
git -C third_party/auto-atomic-operation lfs pull \
  --include "assets/meshes/**" \
  --exclude "assets/videos/**"
```

The 3DGS assets are not committed to this repository. Download the open-door
GS assets from the Hugging Face dataset before running GS open-door evaluation:

```bash
uv pip install --python .venv/bin/python huggingface_hub

huggingface-cli download OpenGHz/auto-atom-assets \
  --repo-type dataset \
  --local-dir third_party/auto-atomic-operation \
  --include "assets/gs/robots/airbot_play/*" \
  --include "assets/gs/robots/airbot_g2p/*" \
  --include "assets/gs/scenes/open_door/door1.ply" \
  --include "assets/gs/scenes/open_door/door2.ply" \
  --include "assets/gs/scenes/open_door/door3.ply" \
  --include "assets/gs/scenes/open_door/door4.ply" \
  --include "assets/gs/scenes/open_door/door11.ply" \
  --include "assets/gs/scenes/open_door/door14.ply" \
  --include "assets/gs/scenes/open_door/door15.ply" \
  --include "assets/gs/scenes/open_door/door17.ply" \
  --include "assets/gs/scenes/open_door/door19.ply" \
  --include "assets/gs/scenes/open_door/real_knob1.ply" \
  --include "assets/gs/scenes/open_door/real_lock1.ply" \
  --include "assets/gs/backgrounds/door_bg/**"
```

For a minimal simulator sanity check:

```bash
.venv/bin/python -B scripts/run_aao_closed_loop_eval.py \
  --model-client hold \
  --episodes 1 \
  --max-updates 1 \
  --stride 1 \
  --action-repeat 1 \
  --sim-loop-frequency 0 \
  --no-video \
  --output-dir runs/aao_closed_loop/smoke_hold_open_door_gs
```

The 30-environment GS sweep uses the door list
`door1,door2,door3,door4,door11,door14,door15,door17,door19`, random wall
backgrounds, and the shared `real_knob1.ply` / `real_lock1.ply` handle assets:

```bash
.venv/bin/python -B scripts/run_aao_open_door_gs_sweep.py \
  --gpu 0 \
  --device cuda:0 \
  --output-dir runs/aao_closed_loop/fastwam_mix20k_open_door_gs_30env_repeat5_lockstep \
  --num-combos 30 \
  --strides 4,8 \
  --max-updates 160 \
  --action-repeat 5 \
  --action-horizon 32 \
  --num-inference-steps 10 \
  --sim-loop-frequency 0
```

## Index

- [AAO Closed-Loop Integration](#aao-closed-loop-integration)
  - [AAO Setup](#aao-setup)
- [File Structure](#file-structure)
- [Environment Setup](#environment-setup)
- [Model Preparation](#model-preparation)
- [Dataset Download](#dataset-download)
- [Inference with Released Checkpoints](#inference-with-released-checkpoints)
- [Training](#training)
- [Inference with Your Trained Checkpoints](#inference-with-your-trained-checkpoints)
- [Acknowledgements](#acknowledgements)
- [BibTeX](#bibtex)

## File Structure

```text
FastWAM/
├── configs/
│   ├── data/                 # Dataset configs (LIBERO, RoboTwin, etc.)
│   ├── model/                # Model architecture and component configs
│   └── task/                 # Task-level configs (training task names)
├── scripts/
│   ├── train.py
│   ├── train_zero1.sh        # Deepspeed zero1 training entrypoint
│   ├── preprocess_action_dit_backbone.py  # Preprocess ActionDiT backbone before training
│   └── precompute_text_embeds.py  # Precompute T5 text embedding cache before training
├── experiments/
│   ├── libero/
│   │   └── run_libero_manager.py
│   └── robotwin/
│       └── run_robotwin_manager.py
├── src/fastwam/              # Core code
├── runs/                     # Training outputs (ckpt, logs)
├── checkpoints/              # Pretrained or external checkpoints
├── data/                     # Data directory
└── evaluate_results/         # Inference / evaluation results
```

## Environment Setup

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## Model Preparation

This step is required before both training and inference.

Step 1: set the Wan model directory first (opional, default `./checkpoints`):

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

Step 2: pre-generate the ActionDiT backbone (interpolated from Wan22 DiT):

```bash
# uncond (fastwam)
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## Dataset Download

### LIBERO

The preprocessed LIBERO dataset used by Fast-WAM is available at:

- https://huggingface.co/datasets/yuanty/LIBERO-fastwam

Download all compressed files first, then extract them all:

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2

# Run after downloading all 4 tar.gz files
for f in *.tar.gz; do
  tar -xzf "$f"
done
```

The extracted directory structure should be:

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin

The preprocessed RoboTwin dataset used by Fast-WAM is available at:

- https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

Download all split archive files first, then concatenate and extract:

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0

# Run after downloading all robotwin2.0.tar.gz.part-* files
cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

The extracted directory structure should be:

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

If you also keep:

```text
data/robotwin2.0/dataset_stats.json
```

in the root directory, it can be used directly as the statistics file for the current configs in this repo. You can also recompute it.

## Inference with Released Checkpoints

The released checkpoints and their corresponding dataset stats are available on [Hugging Face](https://huggingface.co/yuanty/fastwam).

Optional: download released checkpoints and dataset stats from Hugging Face:

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

After downloading, the local directory is expected to contain:

```text
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
├── libero_uncond_2cam224_dataset_stats.json
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

Before running the `LIBERO` benchmark, install the official LIBERO environment first
from the [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO).
Then run this final step:

```bash
pip install mujoco==3.3.2
```

The `mujoco` environment should ideally stay consistent with the LIBERO data version.

We have already copied the `RoboTwin` evaluation-related code into `third_party/RoboTwin`.
You still need to follow the official RoboTwin instructions from the
[RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin) to finish environment installation and download the required assets, then create the policy symlink:

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

Optional: evaluate released LIBERO checkpoint:

The released `LIBERO` / `RoboTwin` evaluation managers default to `8` GPUs
(`MULTIRUN.num_gpus=8` in `configs/sim_libero.yaml` and `configs/sim_robotwin.yaml`).
If you want to evaluate with fewer GPUs, pass a smaller value such as
`MULTIRUN.num_gpus=4`.

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

Optional: evaluate released RoboTwin checkpoint:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

For faster RoboTwin evaluation, we have enabled `EVALUATION.skip_get_obs_within_replan=true` in [`configs/sim_robotwin.yaml`](./configs/sim_robotwin.yaml).
This skips RGB rendering while consecutively executing an action chunk within one replan window, which speeds up evaluation but makes the saved video look very low-FPS.
Set it to `false` if you want to save a fully rendered video.

**Note:** We evaluate with **unseen** instructions, following Motus. [Lingbot-VA](https://github.com/Robbyant/lingbot-va/blob/661d52a59dc634a650efcd10a79d06bbb17ea81f/evaluation/robotwin/eval_polict_client_openpi.py#L308) uses **seen** instructions instead. You can try `EVALUATION.instruction_type=seen` to use **seen** instructions, which should theoretically improve performance by one or two points.

## Training

### 1) Precompute T5 embedding cache before training

Use `scripts/precompute_text_embeds.py` to precompute embeddings for each training task:

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

For multi-GPU:

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

### 2) Training (using `fastwam` as an example)

When running a new task for the first time, set `pretrained_norm_stats` in the corresponding `configs/data/*.yaml` to `null` first.
After one training run, a `dataset_stats.json` file will be generated in the current run directory (for example, `runs/{task_name}/{run_id}/dataset_stats.json`).
You can then update `pretrained_norm_stats` to that file path for subsequent runs.

```bash
# LIBERO
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# RoboTwin
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

For LIBERO, we train on a single node with 8 GPUs. For RoboTwin, we use 64 GPUs to accelerate training. You can try reducing the GPU count or training epochs.

## Inference with Your Trained Checkpoints

The `mujoco` environment should ideally stay consistent with the LIBERO data version. Then run LIBERO evaluation:

```bash
# LIBERO
python experiments/libero/run_libero_manager.py task={task_name} ckpt={ckpt_path}
```

We have already copied the `RoboTwin` evaluation-related code into `third_party/RoboTwin`.
You still need to follow the official RoboTwin instructions from the
[RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin).
Finish installation and download the required assets, then create the policy symlink:

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

Then run RoboTwin evaluation:

```bash
python experiments/robotwin/run_robotwin_manager.py task={task_name} ckpt={ckpt_path}
```

Common `task_name` examples:

```text
libero_uncond_2cam224_1e-4
robotwin_uncond_3cam_384_1e-4
```

## Acknowledgements

The RoboTwin evaluation code in this repository is adapted from the official [RoboTwin repository](https://github.com/RoboTwin-Platform/RoboTwin). We thank the RoboTwin team for releasing their codebase and assets.

## BibTeX

If you find our work helpful, please consider citing:

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```
