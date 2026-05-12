# FastWAM

**Fast-WAM: Do World Action Models Need Test-time Future Imagination?** 的官方代码仓库。

[![English](https://img.shields.io/badge/README-English-111111.svg)](./README.md)
[![中文](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-d14836.svg)](./README_zh.md)

[![arXiv](https://img.shields.io/badge/arXiv-2603.16666-b31b1b.svg)](https://arxiv.org/abs/2603.16666)
[![Project Page](https://img.shields.io/badge/Project_Page-Fast--WAM-2ea44f.svg)](https://yuantianyuan01.github.io/FastWAM/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-f7c843)](https://huggingface.co/yuanty/fastwam)
[![Hugging Face Dataset - LIBERO](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20LIBERO-f7c843)](https://huggingface.co/datasets/yuanty/LIBERO-fastwam)
[![Hugging Face Dataset - RoboTwin](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20RoboTwin-f7c843)](https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam)

本仓库包含 FastWAM 在 LIBERO / RoboTwin 上的训练与评估代码。

## AAO 闭环仿真集成

当前工作区额外集成了 `auto-atomic-operation`（AAO），用于闭环仿真验证。
目前桥接代码支持 GS 版本 open door Airbot Play 场景，以及 P7 cup-on-coaster
场景。这部分和训练主流程解耦：

- AAO 以 submodule 形式放在 `third_party/auto-atomic-operation`，当前 pin
  到 DISCOVER fork 的 `449119b`，该版本包含上游重命名提交 `d1530c7`。
- Gaussian renderer 以 submodule 形式放在 `third_party/GaussianRenderer`。
- FastWAM 到 AAO 的闭环桥接代码在 `src/fastwam/closed_loop_eval/`。
- 单 episode 入口：`scripts/run_aao_closed_loop_eval.py`。
- 多门/多背景 sweep 入口：`scripts/run_aao_open_door_gs_sweep.py`。
- pred / VAE recon / 实际仿真对比视频入口：
  `scripts/run_aao_visual_rollout.py`。
- batch benchmark 入口：`scripts/run_aao_benchmark.py`，详细用法见
  `docs/aao_benchmark.md`。
- batch benchmark smoke 入口：`scripts/run_aao_benchmark_smoke_tests.py`。

默认任务是 `open_door_airbot_play_gs`。当前闭环代码默认使用 mix 20k
权重：

```text
runs/mix_uncond_2cam224_1e-4/mix_uncond_20k_20260507_024400/checkpoints/weights/step_020000.pt
```

`runs/` 下的模型运行目录不会提交进 git。使用 `--model-client fastwam`
前，需要先复制或下载对应的 `config.yaml`、`dataset_stats.json`、text
embedding cache 和 checkpoint；如果使用别的 run，用 `--fastwam-config`、
`--dataset-stats`、`--text-cache-dir`、`--checkpoint` 显式指定。

控制方式只由 `--sim-loop-frequency` 决定：

- `--sim-loop-frequency 0`：lockstep，同步调用 AAO `update(action)`，
  再读取最新 observation。
- `--sim-loop-frequency >0`：continuous，AAO 后台按该频率持续仿真，
  runner 只用 `set_cartesian_action()` 更新当前目标动作，并从新
  observation 重新规划。

对于当前 mix/open-door 数据，`/home/zoyo/mix/meta/info.json` 显示 action
频率是 20Hz，AAO open door 的 `env.update_freq` 是 100Hz。因此默认闭环
参数为：

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

batch benchmark 的 profile 配置在 `configs/aao_benchmark/`，当前预设
`open_door_airbot_play_gs` 和 `cup_on_coaster_gs_airbot_p7`。也可以通过
`--profile-config <yaml>` 接入新的测试 env。两个预设 profile 的模型输出都按
7D EEF pose + gripper 处理，并统一向 AAO 下发 `cartesian_absolute`；cup
profile 的模型输入 state 是 8D joint + gripper，但不走 `joint_absolute`
控制。batch benchmark 当前只支持 lockstep，即 `--sim-loop-frequency 0`。
多 env、多模型 GPU 的 benchmark 用法见 `docs/aao_benchmark.md`。

注意：当前 AAO `success_rate` / `final_success` 不能单独作为真实开门成功标准。
batch benchmark 会在 `benchmark_results.csv` / `benchmark_results.jsonl` 中记录
`stage_name`、`phase` 和 `task_details`，需要结合这些字段判断失败原因和
success 语义。需要视觉排查时，再用单 episode runner 或 visual rollout。

需要对比模型想象视频、VAE recon 和 AAO 实际观测时，可以使用 visual
rollout 脚本。`--frame-sampling sim-update` 会按每个 AAO update 输出一帧；
例如 2 个 32-action window、`--action-repeat 5` 会生成 320 帧：

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

本次集成的工作记录在：

- `docs/260508-auto-atomic-open-door-plan.md`
- `docs/260508-auto-atomic-open-door-progress.md`

### AAO 部署

全新 clone 时需要拉取 submodule：

```bash
git clone --recurse-submodules https://github.com/MrZoyo/FastWAM.git
cd FastWAM

# 如果 clone 时没有加 --recurse-submodules，则补执行：
git submodule update --init --recursive
```

先安装 FastWAM 常规环境，然后在同一个 Python 环境中补 AAO/GS 可选依赖。
AAO 从本地 submodule 安装，GaussianRenderer 也使用已经 pin 住的本地
submodule，避免重新拉一个浮动的 Git 依赖：

```bash
uv pip install --python .venv/bin/python \
  -e "third_party/auto-atomic-operation[mujoco]" \
  -e "third_party/GaussianRenderer[shs,mujoco]" \
  gsplat==1.5.3 \
  ninja \
  natsort \
  PyOpenGL_accelerate
```

如果你使用 conda 环境，把 `.venv/bin/python` 换成对应 conda 环境里的
Python 路径即可。

AAO MuJoCo mesh 由 AAO submodule 内的 Git LFS 管理。clone submodule 后
需要拉一次 LFS mesh：

```bash
git -C third_party/auto-atomic-operation lfs pull \
  --include "assets/meshes/**" \
  --exclude "assets/videos/**"
```

3DGS 资产不会提交进本仓库，运行 GS open door 前需要从 Hugging Face
dataset 下载 open-door 需要的 GS assets：

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

最小仿真 sanity check：

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

30 环境 GS sweep 默认使用门列表
`door1,door2,door3,door4,door11,door14,door15,door17,door19`，随机 wall
背景，并统一使用 `real_knob1.ply` / `real_lock1.ply`：

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

## 目录

- [AAO 闭环仿真集成](#aao-闭环仿真集成)
  - [AAO 部署](#aao-部署)
- [File Structure](#file-structure)
- [环境安装](#环境安装)
- [模型准备](#模型准备)
- [数据集下载](#数据集下载)
- [使用 Release 权重推理](#使用-release-权重推理)
- [训练](#训练)
- [使用自己训练的权重推理](#使用自己训练的权重推理)
- [致谢](#致谢)
- [BibTeX](#bibtex)

## File Structure

```text
FastWAM/
├── configs/
│   ├── data/                 # 数据集配置（LIBERO、RoboTwin 等）
│   ├── model/                # 模型结构与组件配置
│   └── task/                 # 任务级配置（训练 task 名）
├── scripts/
│   ├── train.py
│   ├── train_zero1.sh        # deepspeed zero1 训练入口
│   ├── preprocess_action_dit_backbone.py  # 训练前预处理 ActionDiT backbone
│   └── precompute_text_embeds.py  # 训练前预计算 T5 文本 embedding cache
├── experiments/
│   ├── libero/
│   │   └── run_libero_manager.py
│   └── robotwin/
│       └── run_robotwin_manager.py
├── src/fastwam/              # 核心代码
├── runs/                     # 训练输出（ckpt、日志）
├── checkpoints/              # 预训练或外部 checkpoint
├── data/                     # data目录
└── evaluate_results/         # 推理/评估结果
```

## 环境安装

```bash
conda create -n fastwam python=3.10 -y
conda activate fastwam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## 模型准备

这一步同时是训练和推理的前置项。

第一步，先设置 Wan 模型目录（可选，默认 `./checkpoints`）：

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

第二步，预生成 ActionDiT backbone（从Wan22 DiT插值）：

```bash
# uncond (fastwam)
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

## 数据集下载

### LIBERO

Fast-WAM 使用的 LIBERO 预处理数据已发布到：

- https://huggingface.co/datasets/yuanty/LIBERO-fastwam

先下载全部压缩包，再全部解压：

```bash
mkdir -p data/libero_mujoco3.3.2
cd data/libero_mujoco3.3.2

# 下载 4 个 tar.gz 文件后执行
for f in *.tar.gz; do
  tar -xzf "$f"
done
```

解压后目录结构应为：

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

### RoboTwin

Fast-WAM 使用的 RoboTwin 预处理数据已发布到：

- https://huggingface.co/datasets/yuanty/robotwin2.0-fastwam

先下载全部分卷文件，再拼接并解压：

```bash
mkdir -p data/robotwin2.0
cd data/robotwin2.0

# 下载全部 robotwin2.0.tar.gz.part-* 文件后执行
cat robotwin2.0.tar.gz.part-* | tar -xzf -
```

解压后目录结构应为：

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

根目录下如果同时保留：

```text
data/robotwin2.0/dataset_stats.json
```

可直接作为本仓库当前配置使用的统计文件，也可重新计算。

## 使用 Release 权重推理

release 的模型权重以及对应的 dataset stats 已经发布到 [Hugging Face](https://huggingface.co/yuanty/fastwam).

从 Hugging Face 下载 release 权重和 dataset stats：

```bash
pip install -U huggingface_hub

huggingface-cli download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --local-dir ./checkpoints/fastwam_release
```

下载后，本地目录应为：

```text
checkpoints/fastwam_release/
├── libero_uncond_2cam224.pt
├── libero_uncond_2cam224_dataset_stats.json
├── robotwin_uncond_3cam_384.pt
└── robotwin_uncond_3cam_384_dataset_stats.json
```

`LIBERO` benchmark 评测前，请先按 [LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) 安装环境：
最后一步执行：

```bash
pip install mujoco==3.3.2
```

`mujoco` 环境和 LIBERO 数据版本相关，最好保持一致。

我们已经把 `RoboTwin` 评测相关代码copy到了 `third_party/RoboTwin`。
但仍需按 [RoboTwin 官方仓库](https://github.com/RoboTwin-Platform/RoboTwin) 中的教程完成环境安装并下载相关assets：
再创建 policy 软链接：

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

一键评测 release 的 LIBERO 权重：

当前 `LIBERO` / `RoboTwin` 的评测 manager 默认使用 `8` 张 GPU
（`configs/sim_libero.yaml` 和 `configs/sim_robotwin.yaml` 中的
`MULTIRUN.num_gpus=8`）。
如果你想用更少的卡，直接在命令行里传更小的值，例如
`MULTIRUN.num_gpus=4`。

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

一键评测 release 的 RoboTwin 权重：

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  MULTIRUN.num_gpus=8
```

为了加速 RoboTwin 评测，我们在 [`configs/sim_robotwin.yaml`](./configs/sim_robotwin.yaml) 中打开了 `EVALUATION.skip_get_obs_within_replan=true`。
它会在一次 replan 窗口内连续执行一个 action chunk 时跳过 RGB 渲染，评测更快，但保存下来的视频帧率会低。
如果想保存完整视频，可以把它设为 `false`。

**注意：**我们测试用的是**unseen**指令，这点和Motus对齐。而[Lingbot-VA](https://github.com/Robbyant/lingbot-va/blob/661d52a59dc634a650efcd10a79d06bbb17ea81f/evaluation/robotwin/eval_polict_client_openpi.py#L308)使用的是**seen**，你可以尝试设置`EVALUATION.instruction_type=seen`来使用**seen**指令，理论上会提高一两个点。

## 训练

### 1) 训练前先预计算 T5 embedding cache

使用 `scripts/precompute_text_embeds.py`，按训练 task 预计算：

```bash
# LIBERO
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# RoboTwin
python scripts/precompute_text_embeds.py task=robotwin_uncond_3cam_384_1e-4
```

如需多卡可用：

```bash
torchrun --standalone --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```


### 2) 训练（以 fastwam 为例）

首次跑某个新任务时，请先把对应 `configs/data/*.yaml` 里的 `pretrained_norm_stats` 设为 `null`。
跑完一次训练后，会在当前 run 目录生成 `dataset_stats.json`（例如 `runs/{task_name}/{run_id}/dataset_stats.json`），
后续就可以把 `pretrained_norm_stats` 改成该文件路径。

```bash
# LIBERO
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# RoboTwin
bash scripts/train_zero1.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

对于LIBERO，我们使用单机8卡训练。对于RoboTwin，我们使用了64卡来加速训练，你可以尝试调小卡数和训练总epoch数。

## 使用自己训练的权重推理

`mujoco` 环境和 LIBERO 数据版本相关，最好保持一致。之后再运行 LIBERO 评测：

```bash
# LIBERO
python experiments/libero/run_libero_manager.py task={task_name} ckpt={ckpt_path}
```

我们已经把 `RoboTwin` 评测相关代码copy到了 `third_party/RoboTwin`。
但仍需按 [RoboTwin 官方仓库](https://github.com/RoboTwin-Platform/RoboTwin) 中的教程完成安装并下载相关assets：
再创建 policy 软链接：

```bash
ln -sfn "$(pwd)/experiments/robotwin/fastwam_policy" "$(pwd)/third_party/RoboTwin/policy/fastwam_policy"
```

之后再运行 RoboTwin 评测：

```bash
python experiments/robotwin/run_robotwin_manager.py task={task_name} ckpt={ckpt_path}
```


常用 `task_name` 示例：

```text
libero_uncond_2cam224_1e-4
robotwin_uncond_3cam_384_1e-4
```

## 致谢

本仓库中的 RoboTwin 评测代码基于官方 [RoboTwin 仓库](https://github.com/RoboTwin-Platform/RoboTwin) 适配而来。感谢 RoboTwin 团队公开其代码仓库和相关 assets。

## BibTeX

如果你觉得我们的工作有帮助，欢迎引用：

```bibtex
@article{yuan2026fastwam,
  title={Fast-WAM: Do World Action Models Need Test-time Future Imagination?},
  author={Tianyuan Yuan and Zibin Dong and Yicheng Liu and Hang Zhao},
  journal={arXiv preprint arXiv:2603.16666},
  year={2026},
  url={https://arxiv.org/abs/2603.16666}
}
```
