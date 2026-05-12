# AAO Benchmark 使用文档

本文档说明 `scripts/run_aao_benchmark.py` 的实际用法。这个入口用于在 AAO
里批量跑 closed-loop episode，统计成功率、耗时、模型推理时间、仿真更新时间
等指标。

当前 batch benchmark 只支持 lockstep 控制，也就是 `--sim-loop-frequency 0`。
如果传入 `--sim-loop-frequency > 0`，程序会直接报错；这样可以避免连续仿真
后台线程和 per-env episode 计数产生歧义。

## Profile

benchmark 的任务默认值已经抽到 YAML：

| profile | YAML | AAO task | 模型输入 state | 模型输出 action | 相机映射 |
| --- | --- | --- | --- | --- | --- |
| `open_door_airbot_play_gs` | `configs/aao_benchmark/open_door_airbot_play_gs.yaml` | `open_door_airbot_play_gs` | 7D cartesian + gripper | 7D EEF pose + gripper | `head_left=env2_cam,right_wrist_left=eef_wrist_cam` |
| `cup_on_coaster_gs_airbot_p7` | `configs/aao_benchmark/cup_on_coaster_gs_airbot_p7.yaml` | `cup_on_coaster_gs_airbot_p7` | 8D joint + gripper | 7D EEF pose + gripper | `head_left=env1_cam,right_wrist_left=eef_wrist_cam` |

可以继续用内置名字：

```bash
.venv/bin/python scripts/run_aao_benchmark.py \
  --profile open_door_airbot_play_gs \
  ...
```

也可以显式传 YAML，适合新测试 env：

```bash
.venv/bin/python scripts/run_aao_benchmark.py \
  --profile-config configs/aao_benchmark/open_door_airbot_play_gs.yaml \
  ...
```

YAML 字段示例：

```yaml
name: open_door_airbot_play_gs
task: open_door_airbot_play_gs
instruction: open the door
camera_map: head_left=env2_cam,right_wrist_left=eef_wrist_cam
action_repeat: 5
train_action_hz: 20.0
max_updates: 160
proprio_mode: cartesian
proprio_dim: 7
fastwam_config: configs/task/mix_uncond_2cam224_1e-4.yaml
text_cache_dir: data/text_embeds_cache/mix
```

注意：

- benchmark 统一向 AAO 下发 `cartesian_absolute`。
- 不支持 `joint_absolute` 控制；cup 只是模型输入 state 使用 8D joint + gripper。
- 两个 profile 的模型输出都必须是 7D action：`[x, y, z, roll, pitch, yaw, gripper]`。
- `fastwam_config` 和 `text_cache_dir` 在 YAML 中可以写仓库相对路径。
- `checkpoint` 和 `dataset_stats` 不写进 profile，正式模型评测时必须显式传入。
- 默认会关闭 depth、mask、heat map，只取 RGB，避免 GS segmentation 相关崩溃并减少开销。

## Strict 校验

benchmark 现在默认 fail fast，不再静默补零、截断或广播：

- AAO observation 缺 RGB、pose、joint state、camera intrinsics/extrinsics 会直接报错。
- `proprio_raw` 维度必须和 profile 的 `proprio_dim` 完全一致。
- 模型 action 维度必须正好是 7D，额外列不会被截断。
- batch task update 的 `done/success/status/details` 等字段必须和 AAO batch size 对齐。
- `CUDA_VISIBLE_DEVICES` / `EGL_VISIBLE_DEVICES` / `MUJOCO_EGL_DEVICE_ID`
  如果已经设置且和 `--gpu` 冲突，会直接报错。

这些错误会暴露环境、配置或数据接口不一致的问题，不再用默认值掩盖。

## 并行模型

AAO env batch 和模型 GPU 是分开的：

- `--gpu`：AAO 仿真使用的物理 GPU。
- `--batch-size`：同一个 AAO `PolicyEvaluator` 里并行的 env 数。
- `--model-gpus`：FastWAM 模型 worker 使用的物理 GPU 列表，例如 `1,7`。

启用 `--model-gpus` 后，主进程只跑 AAO 和调度逻辑；每张模型卡启动一个
子进程，每个子进程加载同一个 checkpoint。子进程内部会设置
`CUDA_VISIBLE_DEVICES=<physical_gpu>`，因此 worker 内部固定使用 `cuda:0`。

env 到模型 worker 的分配规则是：

```text
worker_index = env_index % len(model_gpus)
```

没有 `--model-gpus` 时，FastWAM 模型在主进程中加载，使用 `--device` 指定的
device；这适合单模型 smoke，但不适合多卡并行 benchmark。

## 准备项

从仓库根目录运行：

```bash
cd /DATA/disk1/zoyo/FastWAM
```

确认 AAO/GS 依赖已经按 README 的 AAO 部署步骤装好。正式模型评测还需要确认
checkpoint、dataset stats 和 text cache 存在：

```bash
ls -lh "$CHECKPOINT"
ls -lh "$DATASET_STATS"
ls -lh data/text_embeds_cache/mix
ls -lh data/text_embeds_cache/cup
```

`dataset_stats.json` 用于把 AAO observation 转成训练分布里的归一化 state，
并把模型输出的归一化 action 反归一化回真实 action。它必须和 checkpoint
对应的训练数据一致；mix 和 cup 不能混用。

text cache 必须和 profile 的 instruction 一致：

- mix/open door：`open the door`
- cup：`pick up the cup and place it on the coaster`

如果缺 text cache，模型 worker 会在初始化时报 `Missing text embedding cache`。

## 快速测试

先跑不加载 FastWAM checkpoint 的真实 AAO hold smoke。这个脚本会默认依次跑
两个预设 profile，每个 profile 都是 `batch-size=1`、`total-episodes=1`、
`max-updates=1`：

```bash
.venv/bin/python scripts/run_aao_benchmark_smoke_tests.py --gpu 0
```

只跑一个 profile：

```bash
.venv/bin/python scripts/run_aao_benchmark_smoke_tests.py \
  --profile open_door_airbot_play_gs \
  --gpu 0
```

直接调用 benchmark 入口也可以：

```bash
.venv/bin/python scripts/run_aao_benchmark.py \
  --profile-config configs/aao_benchmark/open_door_airbot_play_gs.yaml \
  --model-client hold \
  --gpu 0 \
  --batch-size 1 \
  --total-episodes 1 \
  --max-updates 1 \
  --stride 1 \
  --action-horizon 2 \
  --output-dir /DATA/disk3/tmp/fastwam_aao_benchmark_hold_smoke \
  --log-level INFO
```

smoke 常用 `--max-updates 1`，因此 `success_rate=0.0` 是正常现象；
它只说明 AAO 初始化、reset、observation adapter、update 和输出统计链路能跑通。

不依赖 AAO/GPU 的 strict 单元测试：

```bash
.venv/bin/python -m pytest -q tests/test_aao_benchmark_strict.py
```

## mix 开门模型测试

本地已验证过的 mix run 示例：

```bash
MIX_CKPT=runs/mix_uncond_2cam224_1e-4/mix_uncond_20k_20260507_024400/checkpoints/weights/step_020000.pt
MIX_STATS=runs/mix_uncond_2cam224_1e-4/mix_uncond_20k_20260507_024400/dataset_stats.json

.venv/bin/python scripts/run_aao_benchmark.py \
  --profile-config configs/aao_benchmark/open_door_airbot_play_gs.yaml \
  --model-client fastwam \
  --model-gpus 1,7 \
  --gpu 0 \
  --checkpoint "$MIX_CKPT" \
  --dataset-stats "$MIX_STATS" \
  --batch-size 2 \
  --total-episodes 2 \
  --max-updates 1 \
  --stride 1 \
  --action-horizon 32 \
  --num-inference-steps 1 \
  --output-dir /DATA/disk3/tmp/fastwam_aao_benchmark_smoke_mix_parallel \
  --log-level INFO
```

正式统计时去掉 smoke 限制，把 `--total-episodes` 调大，并让 profile 使用默认
`max_updates`：

```bash
.venv/bin/python scripts/run_aao_benchmark.py \
  --profile-config configs/aao_benchmark/open_door_airbot_play_gs.yaml \
  --model-client fastwam \
  --model-gpus 1,7 \
  --gpu 0 \
  --checkpoint "$MIX_CKPT" \
  --dataset-stats "$MIX_STATS" \
  --batch-size 8 \
  --total-episodes 100 \
  --stride 8 \
  --action-horizon 32 \
  --num-inference-steps 10 \
  --output-dir runs/aao_benchmark/mix_open_door_gs_b8_e100
```

open-door profile 的默认 `action_repeat=5`，用于把 20Hz 训练 action 对齐到
100Hz AAO update。

## cup 模型测试

本地已验证过的 cup checkpoint 示例：

```bash
CUP_CKPT=/DATA/disk1/zoyo/checkpoints/fastwam/cup_20k_resume5000_20260512_112808/weights/step_020000.pt
CUP_STATS=/DATA/disk1/zoyo/checkpoints/fastwam/cup_20k_resume5000_20260512_112808/dataset_stats.json

.venv/bin/python scripts/run_aao_benchmark.py \
  --profile-config configs/aao_benchmark/cup_on_coaster_gs_airbot_p7.yaml \
  --model-client fastwam \
  --model-gpus 1,7 \
  --gpu 0 \
  --checkpoint "$CUP_CKPT" \
  --dataset-stats "$CUP_STATS" \
  --batch-size 2 \
  --total-episodes 2 \
  --max-updates 1 \
  --stride 1 \
  --action-horizon 32 \
  --num-inference-steps 1 \
  --output-dir /DATA/disk3/tmp/fastwam_aao_benchmark_smoke_cup_parallel \
  --log-level INFO
```

正式统计示例：

```bash
.venv/bin/python scripts/run_aao_benchmark.py \
  --profile-config configs/aao_benchmark/cup_on_coaster_gs_airbot_p7.yaml \
  --model-client fastwam \
  --model-gpus 1,7 \
  --gpu 0 \
  --checkpoint "$CUP_CKPT" \
  --dataset-stats "$CUP_STATS" \
  --batch-size 8 \
  --total-episodes 100 \
  --stride 8 \
  --action-horizon 32 \
  --num-inference-steps 10 \
  --output-dir runs/aao_benchmark/cup_p7_b8_e100
```

cup profile 的默认 `action_repeat=1`，因为当前 profile 按 50Hz 训练 action
和 AAO update 对齐。

## 输出文件

每次运行会在 `--output-dir` 下写：

- `benchmark_summary.json`：整次 benchmark 的汇总。正常结束和部分异常退出都会尽量写。
- `benchmark_results.csv`：每个 episode 一行；每完成一个 episode 就刷新。
- `benchmark_results.jsonl`：每个 episode 一行 JSON；每完成一个 episode 就追加。

`benchmark_summary.json` 的关键字段：

- `episodes_completed`：已完成 episode 数。
- `successes` / `success_rate`：AAO 返回的成功统计。
- `incomplete` / `run_error`：异常退出时标记 partial summary。
- `elapsed_time_sec` / `episodes_per_sec`：整次运行 wall time 和吞吐。
- `model_gpus`：本次配置的模型物理 GPU。
- `model_worker_metadata`：每个 worker 的 GPU、checkpoint、stats、config、`proprio_dim`、`model_action_dim`。
- `control`：action repeat、stride、AAO update hz 等控制元信息。
- `overrides`：实际传给 AAO Hydra 的 override。
- `results_csv` / `results_jsonl` / `summary_json`：输出路径。

`benchmark_results.csv` / `benchmark_results.jsonl` 的关键字段：

- `env_index`：AAO batch 中的 env 编号。
- `episode_index`：全局 episode 编号。
- `success` / `done` / `status`：AAO task state。
- `stage_index` / `stage_name` / `phase` / `phase_step`：AAO stage/phase 状态。
- `task_details`：AAO task update details，常用于确认失败原因或 success 语义。
- `updates_used`：这个 episode 消耗的 AAO update 数。
- `model_steps_used`：实际下发的模型 action 数。
- `model_infer_calls`：模型 chunk 推理次数。
- `model_infer_time_sec`：摊到该 env 的模型推理耗时。
- `sim_update_time_sec`：摊到该 env 的 AAO update 耗时。
- `model_worker_indices` / `model_worker_gpus`：这个 episode 实际经过的模型 worker。

AAO `success_rate` 不应该脱离 `task_details` 单独解释。open-door 尤其要检查
handle/door 是否真的产生目标位移；batch benchmark 不保存视频，需要视觉排查时
用 `scripts/run_aao_visual_rollout.py` 或单 episode runner。

## 常用参数

- `--profile`：选择内置 profile 名；内置 profile 也来自 `configs/aao_benchmark/*.yaml`。
- `--profile-config`：指定任意 benchmark profile YAML；提供后覆盖 `--profile`。
- `--checkpoint`：FastWAM checkpoint，`--model-client fastwam` 时必须显式传入。
- `--dataset-stats`：和 checkpoint 配套的 stats，`--model-client fastwam` 时必须显式传入。
- `--model-gpus`：模型 worker 物理 GPU，例如 `1,7`。
- `--gpu`：AAO 仿真物理 GPU。
- `--batch-size`：AAO batch env 数。
- `--total-episodes`：总 episode 数，runner 会持续给空闲 env 分配新 episode。
- `--stride`：每个模型 chunk 只执行前几个 action，然后重新取 observation 推理。
- `--action-horizon`：模型一次输出的 action chunk 长度。
- `--num-inference-steps`：扩散推理步数。smoke 可设 1，正式统计用训练/评测约定值。
- `--max-updates`：覆盖 profile 默认 episode 最大 update 数。只建议 smoke 时用小值。
- `--override`：追加 AAO Hydra override。benchmark 会自动补 `env.batch_size=<batch_size>`。
- `--ignore-done`：诊断用。AAO done 后不提前停，用于排查成功判据或过早 done。

## 常见问题

### 模型 worker 报 missing text embedding cache

确认 profile instruction 与 cache 一致。cup 当前使用：

```text
pick up the cup and place it on the coaster
```

如果确实缺 cache，需要先运行 `scripts/precompute_text_embeds.py` 生成。

### `--gpu` 和 `--device` 怎么配

使用 `--model-gpus` 时不需要改 `--device`；worker 内部固定用 `cuda:0`。
此时 `--gpu` 只影响 AAO 仿真。

不使用 `--model-gpus` 时，模型在主进程里跑，`--device` 才直接决定模型 device。

如果外层已经设置 `CUDA_VISIBLE_DEVICES`、`EGL_VISIBLE_DEVICES` 或
`MUJOCO_EGL_DEVICE_ID`，它们必须和 `--gpu` 一致；不一致时程序会报错，而不是
静默忽略 `--gpu`。

### 多个 env 是否真的分到了多张模型卡

看输出 CSV：

```bash
sed -n '1,6p' runs/aao_benchmark/your_run/benchmark_results.csv
```

每行的 `model_worker_indices` 和 `model_worker_gpus` 会记录实际使用的 worker。

也可以看 summary：

```bash
.venv/bin/python -c "import json; d=json.load(open('runs/aao_benchmark/your_run/benchmark_summary.json')); print(json.dumps(d['model_worker_metadata'], indent=2))"
```

### cup state/action 维度检查

cup worker metadata 应该显示：

```json
{
  "proprio_dim": 8,
  "model_action_dim": 7
}
```

如果 `proprio_dim`、`model_action_dim`、AAO observation 或模型 action 维度不匹配，
benchmark 会直接报错，不会补零或截断。

### GS mask/segmentation 相关崩溃

benchmark 默认追加：

```text
enable_depth=false
enable_mask=false
enable_heat_map=false
```

如果手动打开这些传感器，需要单独确认 AAO/GS renderer 的对应资源完整。

## 已验证 smoke

以下链路已经跑通过：

- hold/open door：`--profile-config configs/aao_benchmark/open_door_airbot_play_gs.yaml --model-client hold --batch-size 1 --total-episodes 1 --max-updates 1`
- hold/cup：`--profile-config configs/aao_benchmark/cup_on_coaster_gs_airbot_p7.yaml --model-client hold --batch-size 1 --total-episodes 1 --max-updates 1`
- YAML profile/open door：显式 `--profile-config configs/aao_benchmark/open_door_airbot_play_gs.yaml` 的真实 AAO hold smoke。
- strict 单元测试：`.venv/bin/python -m pytest -q tests/test_aao_benchmark_strict.py`
