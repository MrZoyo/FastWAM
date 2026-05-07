# FastWAM 训练数据要求（LeRobot v2.1）

这份文档只写训练数据怎么准备，按代码实际读取路径整理，不混训练超参。

## 先说结论

FastWAM 训练数据必须是一个或多个 **LeRobot v2.1** 数据集根目录。每个数据集根目录都要有标准的 `meta/`、`data/`，以及视觉模态对应的 `videos/` 或 `images/`。

FastWAM 不是直接吃原始图片或视频文件，而是吃 **LeRobot v2.1 数据集 + FastWAM 配置** 组合出来的样本。真正要对齐的是：

- LeRobot 数据集 `info.json` 里的 feature 定义
- `episodes.jsonl` / `episodes_stats.jsonl` / `tasks.jsonl`
- FastWAM 的 `configs/data/*.yaml` 里的 `shape_meta`
- 训练时使用的 `pretrained_norm_stats`
- 文本缓存 `text_embedding_cache_dir`

如果这些不一致，最常见的问题是维度报错、视频解码失败、归一化错位、或者 prompt cache 缺失。

## 1. LeRobot v2.1 数据集目录结构

一个标准数据集根目录一般长这样：

```text
my_dataset/
  data/
    chunk-000/
      episode_000000.parquet
      episode_000001.parquet
  videos/
    chunk-000/
      <video_key>/
        episode_000000.mp4
        episode_000001.mp4
  meta/
    info.json
    episodes.jsonl
    episodes_stats.jsonl
    tasks.jsonl
```

可选项：

- `images/` 目录：当视觉特征以 `dtype: image` 存储时使用
- `annotations/` 目录：任务标注类附加信息，FastWAM 训练本身不依赖

### 1.1 `meta/info.json` 的字段

这是 LeRobot v2.1 的核心元数据。FastWAM 会通过它识别数据版本、文件路径模板、特征定义。

| 字段 | 含义 | 是否必需 |
| --- | --- | --- |
| `codebase_version` | 数据集代码版本，v2.1 应写 `v2.1` | 是 |
| `robot_type` | 机器人类型说明 | 否 |
| `total_episodes` | 总 episode 数 | 是 |
| `total_frames` | 总帧数 | 是 |
| `total_tasks` | 总任务数 | 是 |
| `total_videos` | 总视频数 | 是 |
| `total_chunks` | chunk 数 | 是 |
| `chunks_size` | 每个 chunk 最大 episode 数，默认 1000 | 是 |
| `fps` | 数据采样帧率 | 是 |
| `splits` | 数据切分描述 | 是 |
| `data_path` | parquet 路径模板 | 是 |
| `video_path` | 视频路径模板，若不用视频可为 `null` | 是 |
| `features` | 特征定义字典 | 是 |

默认路径模板是：

```text
data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
```

如果使用 `dtype: image`，逐帧图片通常放在：

```text
images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpeg
```

### 1.1.1 LeRobot 默认帧级字段

这些字段会和自定义特征一起进入 `info.json.features`，也是当前写入器默认维护的列：

| 字段 | 含义 | shape |
| --- | --- | --- |
| `timestamp` | 帧时间戳 | `[1]` |
| `frame_index` | episode 内帧序号 | `[1]` |
| `episode_index` | episode 序号 | `[1]` |
| `index` | 全局帧序号 | `[1]` |
| `task_index` | 任务文本索引 | `[1]` |
| `coarse_task_index` | 高层任务索引 | `[1]` |
| `quality_index` | 质量/子任务索引 | `[1]` |
| `coarse_quality_index` | 高层质量/子任务索引 | `[1]` |

如果你是通过当前 LeRobot 写入器生成数据，frame 级 `task` 会是 4 元组：

```text
[coarse_task, task, coarse_quality, quality]
```

FastWAM 最终只取其中的低层任务文本来做 prompt，但这些默认列建议完整保留。

### 1.2 `meta/tasks.jsonl`

每行一个任务：

```json
{"task_index": 0, "task": "pick up the black bowl and place it on the plate"}
```

要求：

- 每条任务都要有唯一 `task_index`
- `task` 必须是字符串
- 这个文件是 `precompute_text_embeds.py` 读取 prompt 的来源

### 1.3 `meta/episodes.jsonl`

每行一个 episode：

```json
{"episode_index": 0, "tasks": ["..."], "length": 98}
```

要求：

- `episode_index` 唯一
- `tasks` 是字符串列表
- `length` 是 episode 帧数

### 1.4 `meta/episodes_stats.jsonl`

v2.1 标准使用这个文件保存每个 episode 的统计，而不是只靠旧版 `stats.json`。

每行形如：

```json
{"episode_index": 0, "stats": {...}}
```

FastWAM 的训练归一化 `dataset_stats.json` 不是这个文件。两者不是一回事：

- `episodes_stats.jsonl`：LeRobot 数据集自带的 per-episode 统计
- `dataset_stats.json`：FastWAM 训练/推理用的归一化统计

### 1.5 `meta/stats.json`

这是旧版兼容路径。新数据优先写 `episodes_stats.jsonl`；只有老的 v2.0 风格数据才需要依赖它。

## 2. `features` 该怎么写

`info.json.features` 里的每个 feature 都必须和真实数据一致。

### 2.1 特征命名规则

- 特征名不能包含 `/`
- 视觉特征推荐使用 `observation.images.xxx`
- 机器人状态推荐使用 `observation.state` 或拆成多个 `observation.state.xxx`
- 动作推荐使用 `action`

### 2.2 feature 的常见类型

| dtype | 说明 | 典型 shape |
| --- | --- | --- |
| `video` | 视频模态，存 mp4 | `[H, W, C]` |
| `image` | 图片模态，存逐帧图片 | `[H, W, C]` |
| `float32` / `int64` 等 | 数值向量 | `[D]` 或 `[1]` |
| `string` | 文本 | 标量 |

视觉特征在 LeRobot 元数据里通常按 **HWC** 记录，比如 `[512, 512, 3]`。

FastWAM 侧的 `shape_meta` 是另一套东西，那里通常写成 **CHW**，例如 `[3, 224, 224]`。

对数值向量特征，`names` 也必须和维度一一对应，写入器会按 `names` 组装数组。

### 2.3 你必须分清的两套 shape

| 位置 | 含义 |
| --- | --- |
| `info.json.features[*].shape` | LeRobot 数据集原始存储/读取侧的 shape |
| `configs/data/*.yaml.shape_meta` | FastWAM 训练时的输入 shape |

这两个 shape 可以不同，但必须逻辑一致。

比如 LIBERO 当前配置：

- 原始相机：`[512, 512, 3]`
- FastWAM 每路相机输入：`[3, 224, 224]`
- 两路横向拼接后的最终视频：`[224, 448]`

## 3. FastWAM 训练真正需要的数据字段

FastWAM 训练时，数据会被整理成几类张量：

- `video`: `[B, C, T, H, W]`
- `action`: `[B, T_action, D_action]`
- `proprio`: `[B, T, D_state]`
- `context` / `context_mask`: 文本 embedding

训练脚本不会直接使用原始 parquet 里的所有列，而是通过 `RobotVideoDataset` + `FastWAMProcessor` 拼出来。

### 3.1 视觉数据

要求：

- 每个 camera 对应一个 feature
- 多相机必须在配置里声明 `num_output_cameras`
- 多相机需要 `concat_multi_camera`
- 每路相机的 raw shape 要和数据一致
- 每路相机先经过 processor 的单路 `train_transforms` / `val_transforms`
- `RobotVideoDataset` 再按 `concat_multi_camera` 拼接，之后统一做 `video_size` 的 resize/crop/normalize

当前代码支持的多相机拼接方式：

- `horizontal`
- `vertical`
- `robotwin`

### 3.2 动作数据

动作默认要求是一个向量，维度由 `shape_meta.action[*].raw_shape` 决定。

LIBERO 当前是 7 维：

- eef delta pose 6 维
- gripper 1 维

如果你自己的机器人不是这个控制空间，就要改：

- `shape_meta.action`
- `action_output_dim`
- `delta_action_dim_mask`
- 训练和推理时的归一化统计

### 3.3 状态 / proprio 数据

状态默认会被整理成 `proprio`。

LIBERO 当前是 8 维，来源是：

- eef pose 6 维
- gripper state 2 维

如果你的机器人状态更复杂，可以拆成多个子字段，例如：

- `observation.state.ee_state`
- `observation.state.joint_state`
- `observation.state.gripper_state`

然后由 `ConcatLeftAlign` 在 FastWAM 侧拼接。

## 4. `shape_meta` 要怎么配

`shape_meta` 是 FastWAM 训练里最重要的配置之一。它决定：

- 数据集中哪些字段会被读取
- 这些字段原始维度是多少
- 经过预处理后要变成什么形状

### 4.1 基本格式

```yaml
shape_meta:
  images:
    - key: image
      raw_shape: [3, 512, 512]
      shape: [3, 224, 224]
    - key: wrist_image
      raw_shape: [3, 512, 512]
      shape: [3, 224, 224]
  action:
    - key: default
      raw_shape: 7
      shape: 7
  state:
    - key: default
      raw_shape: 8
      shape: 8
```

### 4.2 `key` 的映射规则

在 FastWAM 里，`shape_meta` 的 `key` 不是随便写的，它会映射到 LeRobot 的字段名：

- `image` -> `observation.images.image`
- `wrist_image` -> `observation.images.wrist_image`
- `default` in image -> `observation.images`
- `default` in action -> `action`
- `default` in state -> `observation.state`

如果你有多个 action/state 子字段，`shape_meta` 里就分别列出来，随后由 merger 拼接。
列表顺序就是拼接顺序，`ConcatLeftAlign` 会按同样顺序拆回去。

### 4.3 `raw_shape` 和 `shape`

| 字段 | 意义 |
| --- | --- |
| `raw_shape` | 原始数据维度，必须和 parquet/原始 feature 一致 |
| `shape` | 经过 transforms 后的维度，必须和 processor 输出一致 |

举例：

- 原始图像 `512x512`
- 训练输入图像 `224x224`
- 那么就写 `raw_shape: [3, 512, 512]`，`shape: [3, 224, 224]`

## 5. LeRobot v2.1 多数据集训练要求

`dataset_dirs` 可以是多个数据集根目录，但有两个硬要求：

1. 所有数据集的 `fps` 必须一致
2. 多数据集只保留 **共同 feature 的交集**，不在所有数据集中都出现的 key 会被禁用
3. 同名 key 的 dtype / shape 必须兼容，否则会在读取或 transform 阶段报错

这意味着：

- 如果你把多个不同机器人数据混在一起，必须保证共用字段完全对齐
- 如果相机 key 或 state/action key 不一致，数据会被裁掉，甚至直接报错
- 只要是 FastWAM 训练真正要用的字段，必须同时存在于所有数据集里

## 6. FastWAM 训练配置里跟数据强相关的参数

| 参数 | 位置 | 作用 | 要求 |
| --- | --- | --- | --- |
| `dataset_dirs` | `configs/data/*.yaml` | 数据集根目录列表 | 每个目录都必须是 LeRobot v2.1 数据集 |
| `shape_meta` | `configs/data/*.yaml` | 数据字段和维度定义 | 必须和真实数据一致 |
| `num_frames` | `configs/data/*.yaml` | 观测窗口长度 | 默认 33 |
| `action_video_freq_ratio` | `configs/data/*.yaml` | 视频采样间隔 | 必须满足窗口约束 |
| `video_size` | `configs/data/*.yaml` | 最终输入视频尺寸 | 建议 H/W 都是 32 的倍数 |
| `concat_multi_camera` | `configs/data/*.yaml` | 多相机拼接方式 | 多相机时必须设置 |
| `global_sample_stride` | `configs/data/*.yaml` | 全局采样步长 | 默认 1，改了会影响 delta_timestamps |
| `val_set_proportion` | `configs/data/*.yaml` | episode 级 train/val 划分比例 | 训练和验证要保持一致的划分逻辑 |
| `is_training_set` | `configs/data/*.yaml` | 当前数据对象是训练集还是验证集 | train=true, val/test=false |
| `skip_padding_as_possible` | `configs/data/*.yaml` | 边界有 padding 时是否重采样 | 默认 false，最稳 |
| `train_transforms` / `val_transforms` | processor | 图像预处理序列 | 输出 shape 必须和 `shape_meta.shape` 一致 |
| `num_output_cameras` | processor | 模型期望的 camera 数量 | 应和真实 camera 数量一致 |
| `action_output_dim` | processor | 合并后的动作维度 | 必须和 action 一致 |
| `proprio_output_dim` | processor | 合并后的状态维度 | 必须和 state 一致 |
| `delta_action_dim_mask` | processor | 哪些动作维度是 delta | 长度必须和动作维度一致 |
| `action_state_transforms` | processor | 动作/状态的专用变换 | 改了就要重算 stats |
| `use_stepwise_action_norm` | processor | 动作归一化是否按 stepwise 统计 | 影响 `dataset_stats.json` 的使用方式 |
| `norm_default_mode` | processor | 默认归一化方式 | 常见是 `min/max` 或 `z-score` |
| `norm_exception_mode` | processor | 某些字段的例外归一化方式 | 只对特定 key 生效 |
| `context_len` | `configs/data/*.yaml` | 文本 token 长度 | 文本缓存文件名依赖它 |
| `text_embedding_cache_dir` | `configs/data/*.yaml` | T5 prompt cache 输出目录 | 必须可写 |
| `pretrained_norm_stats` | `configs/data/*.yaml` / runtime | 归一化统计文件 | 训练可自动生成，val/test 必须可用 |

### 6.1 `num_frames` 的硬约束

`RobotVideoDataset` 里有两个关键约束：

```text
(num_frames - 1) % action_video_freq_ratio == 0
((num_frames - 1) / action_video_freq_ratio) % 4 == 0
```

含义是：

- 先从 `num_frames` 里采样出视频帧
- 采样后的视频帧数必须满足 `T % 4 == 1`

LIBERO 默认值：

- `num_frames = 33`
- `action_video_freq_ratio = 4`
- 采样后视频帧数 = 9
- 满足 `9 % 4 == 1`

### 6.2 `video_size` 的实际要求

对当前 FastWAM 模型，**最终输入的 H 和 W 都必须是 32 的倍数**，而且不要求正方形。

原因是：

- VAE 空间下采样是 16 倍
- DiT patch size 是 `[1, 2, 2]`

所以虽然辅助函数只会把尺寸补到 16 的倍数，但真正进入 DiT 以后，latent 的 H/W 还要能被 2 整除。

例如：

- `224 x 448` 可以
- `384 x 320` 可以
- `240 x 320` 不行，latent 高度会变成 15，不能被 patch size 的 2 整除

## 7. 归一化统计要求

FastWAM 用的是自己的 `dataset_stats.json`，不是 LeRobot 的 `episodes_stats.jsonl`。

### 7.1 什么时候需要 `dataset_stats.json`

- 第一次训练：可以不写 `pretrained_norm_stats`，训练集会自动计算并保存
- 验证 / 测试 / 推理：必须提供可用的 `pretrained_norm_stats`
- 只要你改了 `dataset_dirs`、`num_frames`（等价于动作窗口长度）或 `action_state_transforms`，旧 stats cache 就不再匹配，需要重算

### 7.2 `dataset_stats.json` 存什么

里面是 FastWAM 训练侧要用的归一化统计，例如：

- `min`
- `max`
- `mean`
- `std`
- `q01`
- `q99`

这些统计是按 action/state 维度生成的。
action 会保存 `stepwise_*` 和 `global_*`，state 主要使用 `global_*`。如果 `use_stepwise_action_norm=True`，action 会优先用 `stepwise_*`。

### 7.3 不能混用的情况

以下情况不能直接复用旧 stats：

- 动作维度变了
- 状态维度变了
- 机器人控制方式变了
- 单位变了
- 夹爪编码变了
- action/state merge 规则变了

## 8. 文本缓存要求

FastWAM 训练要预计算 prompt embedding。

### 8.1 缓存来源

`scripts/precompute_text_embeds.py` 会扫描：

- `meta/tasks.jsonl`
- `configs/data/*.yaml` 里的 `text_embedding_cache_dir`
- `context_len`

如果一个 `data` 配置里有多个 dataset node，脚本会先把所有 prompt 去重，再统一编码后写入每个 cache 目录。所有 dataset node 的 `context_len` 必须一致。

### 8.2 缓存文件命名

文件名是 prompt 的 hash 加上 `context_len` 和模型标识，类似：

```text
<sha256>.t5_len128.wan22ti2v5b.pt
```

### 8.3 要求

- `tasks.jsonl` 必须先准备好
- `context_len` 必须在同一训练配置里保持一致
- cache 目录要可写
- 同一句 prompt 只会编码一次，重复任务会复用同名缓存

## 9. 训练样本边界与 padding

FastWAM 以固定窗口切样本，样本靠近 episode 边界时可能会出现 padding。

loader 会生成这些 mask：

- `action_is_pad`
- `image_is_pad`
- `proprio_is_pad`
- 以及必要时的维度 pad mask

要求：

- episode 最好足够长，尽量避免大量 padding
- 如果很多 episode 短于 `num_frames`，训练会变差

## 10. 自己做数据时的最小检查清单

1. 每个数据集根目录都是 LeRobot v2.1 格式
2. `meta/info.json` 里 `codebase_version` 是 `v2.1`
3. `fps` 在所有 `dataset_dirs` 里一致
4. `tasks.jsonl`、`episodes.jsonl`、`episodes_stats.jsonl` 都存在
5. `features` 里的 key、dtype、shape 和真实数据一致
6. 视频/图片模态的 shape 记录为 `[H, W, C]`
7. FastWAM 的 `shape_meta` 与真实数据对齐
8. 多相机数量和 `num_output_cameras` 一致
9. 最终输入 `video_size` 的 H/W 都是 32 的倍数
10. `num_frames`、`action_video_freq_ratio` 满足窗口约束
11. 生成或提供正确的 `dataset_stats.json`
12. 预计算好 `text_embedding_cache_dir`

## 11. 参考当前项目里 LIBERO 的配置

LIBERO 当前就是一套可工作的参考样例：

- 每路相机原始尺寸：`512 x 512`
- 每路相机训练输入：`224 x 224`
- 两路横向拼接后：`224 x 448`
- `num_frames = 33`
- `action_video_freq_ratio = 4`
- `action_output_dim = 7`
- `proprio_output_dim = 8`
- `context_len = 128`

如果你要做自己的数据，最稳妥的做法是先按这个结构跑通，再改相机数量、控制维度和归一化统计。

## 12. 自有数据的最小准备顺序

1. 先定 `info.json.features`，把视觉、动作、状态和 LeRobot 默认帧级字段一次列清。
2. 再写 `tasks.jsonl`，保证每个任务文本唯一且和 `task_index` 对上。
3. 再写 `episodes.jsonl`，每个 episode 记录自己的 `tasks` 和 `length`。
4. 再写 `episodes_stats.jsonl`，不要只留旧的 `stats.json`。
5. 再确认 `data_path` / `video_path` / `images/` 的真实文件名和 `info.json` 模板完全一致。
6. 再把 `configs/data/*.yaml` 里的 `shape_meta`、`num_frames`、`action_video_freq_ratio`、`video_size`、`num_output_cameras`、`action_output_dim`、`proprio_output_dim` 对齐。
7. 再跑 `scripts/precompute_text_embeds.py`，生成 `text_embedding_cache_dir`。
8. 最后再训练。
