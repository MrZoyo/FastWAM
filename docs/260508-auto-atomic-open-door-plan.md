# Auto Atomic Open Door Closed-Loop Plan

## 目标

在 FastWAM 中集成 `auto-atomic-operation` 仿真器，复用 `WorldModel_3d` 的闭环验证思路，用本地已训练的 open door 权重跑 AAO open door 任务闭环仿真，输出每个 episode 的视频、trace、summary 和 aggregate 指标。

## 当前事实

- 当前代码的最新使用说明以 `README.md` 和 `docs/aao_benchmark.md` 为准：
  open-door 默认任务是 `open_door_airbot_play_back_gs`，对应门把手在左侧的
  back 版本；FastWAM checkpoint/config/stats/text cache 必须通过 CLI 显式
  指定；open-door 默认 `stride=32`，并默认关闭 arm base/EEF reset
  randomization。
- `WorldModel_3d` 已同步到 `a08e872`，闭环入口在 `closed_loop_eval/run_closed_loop_eval.py`。
- `WorldModel_3d` 使用 `auto-atomic-operation` 作为仓库内 submodule，而不是外部软链接。
- `WorldModel_3d/auto-atomic-operation` 最初参考 pin 为 `2f7a10b`；FastWAM 侧需要使用更新的 AAO open door/Airbot/GS 配置。
- `2f7a10b` 之后包含大量 open door 相关修复和新增配置，包括 `open_door_airbot_play_gs.yaml`、`open_door_airbot_play_g2p.yaml`、`open_door_airbot_play_back_gs.yaml`、door 物理、GS、相机和 Airbot 配置更新。
- FastWAM 已新增自己的 AAO submodule：`third_party/auto-atomic-operation`，当前 pin 到 DISCOVER 远端 commit `d831ee7cecd4b87df2337bab7ec856d5a342b412`。
- FastWAM 已新增 `third_party/GaussianRenderer` submodule，跟踪 `main`，当前 pin 到 `8eb95dd690626bdece989b6f1b2cad10371ce652`，用于 AAO GS open door。
- FastWAM 已新增进程内 AAO 集成层，不依赖 WorldModel 风格的 WebSocket inference server。
- FastWAM 的模型动作输出在模型归一化空间内，需要使用训练时 `FastWAMProcessor` 的 normalizer 做反归一化后再送给仿真器。
- 当前 open-door 模型评测使用显式传入的 run 资产；代码不再内置任何
  checkpoint/config/stats/text-cache 路径。示例见 `README.md` 和
  `docs/aao_benchmark.md`。
- 当前 real+sim open-door 配置训练动作窗口为 `num_frames - 1 = 32` 个
  action；本地 LeRobot 数据 FPS 是 `20`。
- AAO `open_door_airbot_play_gs` 继承的 `env.update_freq` 是 `100`，因此数据中 1 个 20Hz action 在 AAO 中应保持 `5` 个 update。
- 历史记录：2026-05-08 初始闭环代码曾默认 `stride=8`，sweep 默认测试
  `stride=4,8`。当前代码已经改为 open-door 默认 `stride=32`。
- 控制模式不再暴露单独 `control-mode` 参数，只由 `--sim-loop-frequency` 控制：`0` 为同步 lockstep，`>0` 为 AAO 后台持续仿真 continuous。

## 2026-05-08 新闭环控制设计

- 每次用最新 AAO observation 做一次 FastWAM 推理，得到未来一个 action chunk。
- 当前 open-door checkpoint 训练窗口是 32 个 action，默认保持
  `action_horizon=32`。
- 默认执行完整 32-action chunk 后再重新规划，即 `stride=32`。较小 stride
  只作为 receding-horizon 诊断参数，不作为当前 open-door 正式评测默认值。
- 每个模型 action 在 AAO 中重复执行 `action_repeat=5` 个 update，把 20Hz 训练 action 对齐到 100Hz AAO update。
- 每个 chunk 执行完后重新读取 AAO 最新 observation，再做下一次 FastWAM 推理。
- `--sim-loop-frequency 0` 使用同步 lockstep：每个 action 通过 AAO `update(action)` 推进一步，然后读取 observation。
- `--sim-loop-frequency >0` 使用 continuous：AAO 后台线程按指定频率持续 `env.update()`，主线程用 `set_cartesian_action()` 更新当前控制目标，再按 `action_repeat / sim_loop_frequency` 的间隔取 observation 重推理。
- 当前大规模 sweep 默认先用 `--sim-loop-frequency 0`，因为它最容易复现并且 trace 与 action repeat 一一对应；continuous 模式已保留用于后续加速验证。
- 夹爪最后一维按长度值传递，默认 clamp 到 `[0.02, 0.0945]`，避免 AAO 中夹爪闭到 0 后穿/夹死门把手。
- 每个 episode 需要保存视频、动作 trace、AAO summary，以及 MuJoCo door/handle 关节和 body pose 诊断；成功判断不能只看 AAO `final_success`。
- 如果怀疑 AAO 过早 done，可用 `--ignore-done` 保持每步评估 task state 但不提前停止，用于诊断后续动作是否还能继续推动门。

## WorldModel 集成方式摘要

WorldModel 的闭环结构可以直接借鉴：

- `SimulatorServiceClient`
  - 直接在进程内实例化 `auto_atom.policy_eval.PolicyEvaluator`
  - 通过 Hydra config 名称加载 AAO task，例如 `cup_on_coaster_gs`
  - 默认补齐 override：`env.batch_size=1`、`++env.viewer.disable=true`、`assets_dir=<AAO assets>`
  - 每步把模型输出的 `[x, y, z, roll, pitch, yaw, gripper]` 转成 AAO 需要的 pose action dict
- `ObservationWindowAdapter`
  - 从 AAO observation 中取相机 RGB/depth/mask/heatmap 和机器人状态
  - 维护 history window
  - 构造模型 payload
- `BaseModelClient`
  - 统一 `infer(model_input) -> {"action_format": "cartesian_absolute", "actions": (T, 7)}`
  - WorldModel 有 `PayloadValidatingHoldModelClient` 和 `WebSocketModelClient`
- `run_closed_loop_eval`
  - 初始化 simulator
  - reset 后记录初始 observation
  - 循环：build input -> model infer -> apply stride/action_repeat 步动作 -> record -> summary

## FastWAM 需要适配的关键差异

- 输入：FastWAM `infer_action` 需要单张拼接后的首帧图像 `[1, 3, H, W]`、prompt 或 cached text context、可选 proprio。
- 图像：当前 open-door 训练配置是双相机水平拼接，最终
  `video_size=[224,448]`；AAO 应映射两路相机到 FastWAM 训练相机：
  - 当前默认：`env2_cam -> head_left`
  - 当前默认：`eef_wrist_cam -> right_wrist_left`
- 状态：当前 open-door 训练 `observation.state` 是 6D arm joint + gripper，
  action 是 7D EEF delta + gripper。AAO observation 同时提供 joint state、
  EEF pose 和 gripper；当前按以下语义适配：
  - 训练动作名：`delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, right_gripper_position`
  - AAO apply 期望：`cartesian_absolute`，即绝对 `[x, y, z, r, p, y, gripper]`
  - 当前 LeRobot action 的前 6 维是帧对齐 backward delta：`pose[t] - pose[t-1]`，不是 `pose[t+1] - pose[t]`。AAO bridge 下发前会先把 action chunk 左移一帧，再累计积分到当前 EEF pose；最后 1 维 gripper length 不是 delta，只做同样的一帧时间对齐并按 absolute 值传给 AAO。
- 文本：open-door instruction 是 `open the door`，text embedding cache 必须由
  `--text-cache-dir` 显式传入。
- 权重：不再有默认权重。闭环 runner、benchmark、visual rollout 和 sweep 在
  `--model-client fastwam` 下都要求显式传
  `--fastwam-config`、`--checkpoint`、`--dataset-stats` 和
  `--text-cache-dir`。

## 执行阶段

1. 文档与上下文固定
   - 创建本 plan 文档和 progress 文档。
   - 持续把发现、决策、命令、失败点写进 progress。

2. 仿真器依赖确认
   - 决定是在 FastWAM 仓库内新增 `auto-atomic-operation` submodule，还是复用 `/DATA/disk1/zoyo/WorldModel_3d/auto-atomic-operation`。
   - 第一版实现使用可配置 `--aao-root`，默认指向 FastWAM 内部 `third_party/auto-atomic-operation` submodule。
   - submodule 跟踪 AAO `main`，当前 pin 到 DISCOVER 远端 `d831ee7cecd4b87df2337bab7ec856d5a342b412`，避免长期依赖 WorldModel 项目目录。
   - GS 渲染依赖固定到 FastWAM 内部 `third_party/GaussianRenderer` submodule。
   - 用户已确认本轮可直接补环境；后续新增依赖仍需在 progress 中记录具体变更。

3. AAO open door smoke test
   - 用最新 AAO 的 open door config 做 `PolicyEvaluator` 初始化和 reset。
   - 先用 hold/mock policy 跑最小 episode，确认 observation、camera、EEF state、stage success、summary 可用。
   - 记录可用 task config 名称，候选为 `open_door`、`open_door_airbot_play_gs`、`open_door_airbot_play_g2p`。
   - 历史记录：初始 smoke test 使用 `open_door_airbot_play_gs` 做闭环验证，贴近
     open door 的 GS 仿真版本；当前默认已切到
     `open_door_airbot_play_back_gs`。

4. FastWAM inference client
   - 新增 FastWAM 专用 closed-loop model client。
   - 输入 AAO observation，输出 `cartesian_absolute` 动作 chunk。
   - 实现图像拼接、resize/normalize、prompt/context 加载、proprio 提取、checkpoint 加载。
   - 实现动作反归一化和 delta-to-absolute 转换。

5. 闭环 runner
   - 先复用 WorldModel 的 runner 结构，裁剪出 FastWAM 所需字段。
   - 支持 `--task open_door...`、`--checkpoint`、`--dataset-stats`、`--text-cache-dir`、`--camera-map`、`--stride`、`--action-repeat`、`--max-updates`、`--sim-loop-frequency`。
   - 输出 `multicam.mp4`、`summary.json`、`client_trace.json.gz`、`aggregate_summary.json`。
   - trace 中显式记录 `chunk_action_index` 和 `repeat_index`，用于确认每个模型 action 被重复了多少个仿真 update。

6. 视觉对比 rollout
   - 新增 `scripts/run_aao_visual_rollout.py`，用于输出 pred / VAE recon / actual simulator 的 3x2 拼接视频。
   - `--frame-sampling model-action` 每个模型 action 输出一帧；`--frame-sampling sim-update` 每个 AAO update 输出一帧。
   - 当前 open-door 配置下 `action_horizon=32, num_video_frames=9`，
     pred/recon 每个 window 只有 9 个视频帧；sim-update 模式按最近邻展开到
     每个 AAO update。

7. 验证顺序
   - 只初始化 simulator，不跑模型。
   - mock hold action 跑通闭环记录。
   - 加载 FastWAM 权重做单步 inference smoke test。
   - 跑 1 episode open door，检查动作尺度和方向。
   - 根据 trace 调整相机映射、delta/absolute 转换和 stride。

## 已通过的最小验证

- `open_door_airbot_play_gs` + hold policy + 1 update：通过。
- `open_door_airbot_play_gs` + hold policy + 1 update + `multicam.mp4`：通过。
- `open_door_airbot_play_gs` + FastWAM mix 20k + 1-step inference/update：通过。
- `open_door_airbot_play_gs` + FastWAM mix 20k + 完整 episode：`action_horizon=32, stride=16` 下 AAO 返回 `final_done=true, final_success=true`，但人工视频检查认为没有真实完成开门；后续必须以视频和 trace 双重判断，不只依赖 AAO success flag。

## 当前未决问题

- 当前要评测的 open-door 权重、data config、dataset stats、text embedding
  cache 路径必须在命令行显式指定，且 stats/cache 必须与 checkpoint 的训练 run
  匹配。
- gripper length 与 AAO `FingerDistanceMapper` 的几何标定是否完全一致仍需通过视频和 trace 继续验证；若不一致，加入显式 length mapping，而不是改成“开合度”语义。
- 当前 AAO open door 默认使用 `open_door_airbot_play_back_gs`；front 对照可显式
  传 `open_door_airbot_play_gs`，普通 MuJoCo/G2P 版本
  `open_door_airbot_play_g2p` 作为依赖排查时的降级验证。
- AAO/GS 部署方式已写入 README/README_zh：包括 submodule、AAO/GaussianRenderer 本地安装、AAO Git LFS mesh、HF open-door GS assets 下载。

## 暂不做

- 不改系统 CUDA/MuJoCo 环境。
- 不直接覆盖或升级现有 FastWAM 训练代码。
- 不把 WorldModel 的所有闭环代码原样复制进来；只迁移 FastWAM 所需的最小部分。
