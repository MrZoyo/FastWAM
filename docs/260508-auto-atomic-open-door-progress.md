# Auto Atomic Open Door Closed-Loop Progress

## 当前状态

- 最新使用说明以 `README.md` 和 `docs/aao_benchmark.md` 为准。
- open-door 默认任务已改为 `open_door_airbot_play_back_gs`，对应门把手在左侧的
  back 版本；`open_door_airbot_play_gs` 仅作为 front/right-handle 对照场景显式使用。
- `runner.py`、benchmark、visual rollout、sweep 和 replay 脚本不再隐式指定
  FastWAM checkpoint；使用 `--model-client fastwam` 时必须显式传入
  `--fastwam-config`、`--checkpoint`、`--dataset-stats` 和对应的
  `--text-cache-dir`。
- open-door 默认 `stride=32`、`action_repeat=5`、`proprio_mode=joint`，并默认关闭
  AAO arm base/EEF reset randomization，除非显式传 `--enable-arm-randomization`。

## 2026-05-08

### 已完成

- 同步并检查了 `/DATA/disk1/zoyo/WorldModel_3d`。
- 确认 `WorldModel_3d` 主仓库当前在 `a08e872 fix resume bug`。
- 初始化了 `WorldModel_3d` 的两个 submodule：
  - `GaussianRenderer` at `8eb95dd`
  - `auto-atomic-operation` at `2f7a10b`
- 阅读了以下关键文件：
  - `WorldModel_3d/AGENTS.md`
  - `WorldModel_3d/CLAUDE.md`
  - `WorldModel_3d/closed_loop_eval/CLAUDE.md`
  - `WorldModel_3d/docs/260405/closed-loop-sim-runbook.md`
  - `WorldModel_3d/closed_loop_eval/sim_service_client.py`
  - `WorldModel_3d/closed_loop_eval/model_clients.py`
  - `WorldModel_3d/closed_loop_eval/observation_adapter.py`
  - `WorldModel_3d/closed_loop_eval/run_closed_loop_eval.py`
  - `WorldModel_3d/auto-atomic-operation/docs/260404-worldmodel-closed-loop-eval.md`
  - `WorldModel_3d/auto-atomic-operation/docs/policy_evaluation.md`
  - `WorldModel_3d/auto-atomic-operation/aao_configs/open_door.yaml`
- 确认 WorldModel 闭环入口采用 `SimulatorServiceClient + ObservationWindowAdapter + ModelClient + EpisodeRecorder`。
- 确认 AAO `PolicyEvaluator` 的一步动作接口是 pose action dict，WorldModel 中间层负责把模型 `[x,y,z,r,p,y,gripper]` 转成 quaternion pose。
- 确认 FastWAM `infer_action` 输出仍需通过训练 processor 的 normalizer 做动作反归一化，不能直接送入 AAO。
- 抓取了 `auto-atomic-operation` 上游引用。
- 确认 `auto-atomic-operation` 本地 pin `2f7a10b` 落后 `origin/main`，上游最新 `f12f75c` 包含大量 open door 相关修复和新配置。
- 2026-05-08 20:10 CST 重新 fetch 了 AAO 上游，确认 `origin/main` 仍为 `f12f75c2220ad7a7ffce54d349a323c4a431d869`，本地工作树仍停在 `2f7a10b0181ddf0549dc48a435ea9d5c37c4231`。
- 用户明确要求 FastWAM 自己集成仿真器项目时使用 AAO 远端最新版本。
- 已在 FastWAM 新增 submodule：
  - path: `third_party/auto-atomic-operation`
  - url: `git@github.com:DISCOVER-Robotics/auto-atomic-operation.git`
  - branch: `main`
  - HEAD: `f12f75c2220ad7a7ffce54d349a323c4a431d869`
- 已在 FastWAM 新增 GS renderer submodule：
  - path: `third_party/GaussianRenderer`
  - url: `https://github.com/OpenGHz/GaussianRenderer.git`
  - branch: `main`
  - HEAD: `8eb95dd690626bdece989b6f1b2cad10371ce652`
- 找到 Lyle 侧 real_1048 训练 run：
  - `/DATA/disk7/Lyle/FastWAM/runs/real_1048_uncond_2cam224_1e-4/real1048_20k_20260508_200316`
  - 已有 `config.yaml` 和 `dataset_stats.json`
  - `checkpoints/weights/` 当前为空，尚无 `.pt` checkpoint
- 通过进程和日志确认同事的 real_1048 训练正在 `/home/Lyle/Projects/FastWAM` 环境运行，命令包含 `task=real_1048_uncond_2cam224_1e-4 max_steps=20000 save_every=2500`。
- 用户确认第一轮测试先使用 `/home/zoyo/mix` 训练出的 20k 权重。
- 已确认 mix 20k 资产存在：
  - checkpoint: `/DATA/disk1/zoyo/FastWAM/runs/mix_uncond_2cam224_1e-4/mix_uncond_20k_20260507_024400/checkpoints/weights/step_020000.pt`
  - config: `/DATA/disk1/zoyo/FastWAM/runs/mix_uncond_2cam224_1e-4/mix_uncond_20k_20260507_024400/config.yaml`
  - stats: `/DATA/disk1/zoyo/FastWAM/runs/mix_uncond_2cam224_1e-4/mix_uncond_20k_20260507_024400/dataset_stats.json`
  - data dir: `/home/zoyo/mix`

### 关键发现

- WorldModel 的 `closed_loop_eval` 已经不依赖独立 AAO server；当前 `SimulatorServiceClient` 是进程内直接调用 `PolicyEvaluator`。
- WorldModel 原 runbook 仍保留 rpyc server 流程，但实际当前代码默认走 in-process client。
- AAO current pinned config 已有 `open_door.yaml`，但上游新增了更多 Airbot/GS/P7 open door 配置和门物理修复。
- FastWAM 第一轮测试权重使用 mix 20k。当前可见权重包括：
  - `checkpoints/fastwam_release/libero_uncond_2cam224.pt`
  - `checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt`
  - `runs/mix_uncond_2cam224_1e-4/.../step_*.pt`
  - `/DATA/disk7/Lyle/FastWAM/runs/real_1048_uncond_2cam224_1e-4/...` 当前只有 config/stats，没有权重
- `real_1048` 数据集 text cache 在 `/DATA/disk7/Lyle/FastWAM/data/text_embeds_cache/real_1048/` 可见。
- real_1048 配置确认使用双相机水平拼接，`video_size=[224,448]`，`head_left + right_wrist_left`，动作和 state 都是 7 维。
- real_1048 配置确认 `delta_action_dim_mask.default=[true,true,true,true,true,true,false]`，所以模型输出经反归一化后前 6 维按 delta EEF pose 处理，最后 1 维按 absolute gripper 处理。
- mix 20k 配置与 real_1048 的模型输入/动作语义一致：双相机水平拼接、`video_size=[224,448]`、7 维 state/action、前 6 维 delta + gripper absolute。

### 当前决策

- 第一版 FastWAM 闭环代码采用可配置 `--aao-root`，默认使用 FastWAM 自己的 `third_party/auto-atomic-operation`。
- 运行 smoke test 时使用 AAO 上游 `origin/main` 版本 `f12f75c`，因为 WorldModel 当前 pin `2f7a10b` 缺少多项 open door/Airbot/GS 修复。
- 先用 mix 20k 实现和验证 FastWAM model client；real_1048 checkpoint 产出后只需要替换 config/stats/checkpoint/text cache。
- 历史记录：当时用户要求默认使用 GS 版本 open door，因此默认 AAO task 曾改为
  `open_door_airbot_play_gs`；当前默认已改为
  `open_door_airbot_play_back_gs`，对应门把手在左侧的 back 版本。
- 最新 Airbot/G2P open door 配置暴露的相机是 `env2_cam` 和 `eef_wrist_cam`，默认 FastWAM 相机映射改为 `head_left=env2_cam,right_wrist_left=eef_wrist_cam`。
- 按用户确认已直接补齐当前 `.venv` 的 GS 依赖：
  - `gaussian-renderer==0.2.0` editable from `third_party/GaussianRenderer`
  - `gsplat==1.5.3`
  - `mujoco` 从 `3.3.2` 升到 `3.8.0`
  - `plyfile==1.1.3`
  - `trimesh==4.12.2`
  - `e3nn==0.6.0`
  - `natsort==8.4.0`
  - `ninja`
  - `pyopengl-accelerate==3.1.10`
- 从 HuggingFace dataset `OpenGHz/auto-atom-assets` 下载了 AAO GS 资产到 `third_party/auto-atomic-operation/assets/gs`。全量下载途中遇到 HF 429 限流，但 open door GS 必需资产已补齐：
  - `assets/gs/robots/airbot_g2p/*.ply`
  - `assets/gs/robots/airbot_play/*.ply`
  - `assets/gs/scenes/open_door/door19.ply`
  - `assets/gs/scenes/open_door/real_knob1.ply`
  - `assets/gs/scenes/open_door/real_lock1.ply`
  - `assets/gs/backgrounds/door_bg/inside/inside0.ply`
  - `assets/gs/backgrounds/door_bg/wall/wall*.ply`
- 修复了 `gsplat` JIT 找不到 `ninja` 的问题：`SimulatorServiceClient.connect()` 现在会把 `sys.prefix/bin` prepend 到 `PATH`。
- 新增 FastWAM 侧闭环代码：
  - `src/fastwam/closed_loop_eval/sim_service_client.py`
  - `src/fastwam/closed_loop_eval/observation_adapter.py`
  - `src/fastwam/closed_loop_eval/model_clients.py`
  - `src/fastwam/closed_loop_eval/episode_recorder.py`
  - `src/fastwam/closed_loop_eval/runner.py`
  - `scripts/run_aao_closed_loop_eval.py`
- `py_compile` 和 CLI `--help` 静态验证通过。
- GS hold smoke 通过：
  - command: `.venv/bin/python -B scripts/run_aao_closed_loop_eval.py --gpu 3 --model-client hold --episodes 1 --max-updates 1 --stride 1 --no-video --output-dir runs/aao_closed_loop/smoke_hold_open_door_gs --log-level INFO`
  - output: `runs/aao_closed_loop/smoke_hold_open_door_gs/aggregate_summary.json`
  - result: `updates_used=1`, `error=null`
  - 首次运行触发 `gsplat` CUDA extension JIT，耗时约 68.7 秒；后续使用缓存。
- FastWAM mix20k 单步闭环 smoke 通过：
  - command: `.venv/bin/python -B scripts/run_aao_closed_loop_eval.py --gpu 3 --model-client fastwam --episodes 1 --max-updates 1 --stride 1 --action-horizon 4 --num-inference-steps 1 --no-video --output-dir runs/aao_closed_loop/smoke_fastwam_mix20k_open_door_gs --log-level INFO`
  - output: `runs/aao_closed_loop/smoke_fastwam_mix20k_open_door_gs/aggregate_summary.json`
  - result: `updates_used=1`, `error=null`
- FastWAM mix20k 短闭环 smoke 通过：
  - command: `.venv/bin/python -B scripts/run_aao_closed_loop_eval.py --gpu 3 --model-client fastwam --episodes 1 --max-updates 8 --stride 4 --action-horizon 8 --num-inference-steps 3 --no-video --output-dir runs/aao_closed_loop/smoke_fastwam_mix20k_open_door_gs_8updates --log-level INFO`
  - output: `runs/aao_closed_loop/smoke_fastwam_mix20k_open_door_gs_8updates/aggregate_summary.json`
  - result: `updates_used=8`, `error=null`
  - action range: position around `[0.30, -0.05, 0.30]`, gripper in `0.02-0.09`
  - roll 维度在 `pi/-pi` 附近有欧拉角环绕，当前不算接口错误；长 rollout 若抖动再加角度 wrap/限幅。
- GS hold video smoke 通过：
  - command: `.venv/bin/python -B scripts/run_aao_closed_loop_eval.py --gpu 3 --model-client hold --episodes 1 --max-updates 1 --stride 1 --output-dir runs/aao_closed_loop/smoke_hold_open_door_gs_video --log-level INFO`
  - output video: `runs/aao_closed_loop/smoke_hold_open_door_gs_video/open_door_airbot_play_gs_ep000/multicam.mp4`
  - result: `updates_used=1`, `error=null`
- FastWAM mix20k open door GS 完整 episode 对照验证：
  - `stride=4`：
    - command: `.venv/bin/python -B scripts/run_aao_closed_loop_eval.py --gpu 3 --device cuda:0 --model-client fastwam --episodes 1 --max-updates 160 --stride 4 --action-horizon 32 --num-inference-steps 10 --output-dir runs/aao_closed_loop/fastwam_mix20k_open_door_gs_full_ep --log-level INFO`
    - output video: `runs/aao_closed_loop/fastwam_mix20k_open_door_gs_full_ep/open_door_airbot_play_gs_ep000/multicam.mp4`
    - result: `updates_used=160`, `final_done=false`, `final_success=null`, `error=null`
    - trace: 40 次模型推理，每次执行 4 个 action；AAO 持续返回 `stage_postcondition_failed/no_displacement`。
  - `stride=8`：
    - command: `.venv/bin/python -B scripts/run_aao_closed_loop_eval.py --gpu 3 --device cuda:0 --model-client fastwam --episodes 1 --max-updates 160 --stride 8 --action-horizon 32 --num-inference-steps 10 --output-dir runs/aao_closed_loop/fastwam_mix20k_open_door_gs_full_ep_stride8 --log-level INFO`
    - output video: `runs/aao_closed_loop/fastwam_mix20k_open_door_gs_full_ep_stride8/open_door_airbot_play_gs_ep000/multicam.mp4`
    - result: `updates_used=160`, `final_done=false`, `final_success=null`, `error=null`
    - trace: 20 次模型推理，每次执行 8 个 action；末段更靠近把手，但 AAO 仍未检测到把手位移成功。
  - `stride=16`：
    - command: `.venv/bin/python -B scripts/run_aao_closed_loop_eval.py --gpu 3 --device cuda:0 --model-client fastwam --episodes 1 --max-updates 160 --stride 16 --action-horizon 32 --num-inference-steps 10 --output-dir runs/aao_closed_loop/fastwam_mix20k_open_door_gs_full_ep_stride16 --log-level INFO`
    - output video: `runs/aao_closed_loop/fastwam_mix20k_open_door_gs_full_ep_stride16/open_door_airbot_play_gs_ep000/multicam.mp4`
    - result: `updates_used=32`, `final_done=true`, `final_success=true`, `error=null`
    - trace: 2 次模型推理，每次最多执行 16 个 action；最后返回 `stage_success_condition_met`，stage `grasp_and_open` 成功。
    - 人工视频检查：用户认为没有真实完成开门，因此该 run 记为“AAO 判据通过，但视觉验证未通过”。
  - 结论修正：`stride=8/16` 目前都不能算真实成功；`stride=16` 只能说明 AAO success flag 可能出现视觉假阳性，后续要结合视频、门/把手物理状态和位移量判断。
  - 备注：AAO 当前 `open_door_airbot_play_gs` 继承 `open_door_airbot_play_g2p`，stage 的 `operation` 是 `push`，不是 `grasp`；因此 trace 里的 `is_operator_grasping=false` 不等价于任务失败，最终成功判据是把手/门位移达到 AAO 的 stage success condition。
- 已把 `scripts/run_aao_closed_loop_eval.py` 对应 runner 的默认 `--stride` 从 `4` 调整为 `16`，用于减少过早重规划；但这不代表视觉任务成功，完整验证仍需人工视频检查。
- 确认 `open_door_airbot_play_gs.yaml` 支持更换 GS 门和背景：
  - 门体：`door_name`，默认 `door19`，映射到 `${gs_dir}/${door_name}.ply`。
  - 把手：默认 `handle_gs_frame: ${gs_dir}/real_${knob_name}.ply`，也可直接 override `env.gaussian_render.body_gaussians.handle_gs_frame=<path>`。
  - 锁：默认 `lock_gs_frame: ${gs_dir}/real_${lock_name}.ply`，也可直接 override 或删除。
  - 墙面背景：`wall_name`，默认 `wall*`，映射到 `${bg3dgs_dir}/wall/${wall_name}.ply`。
  - 室内背景：`inside_name`，默认 `inside0`，映射到 `${bg3dgs_dir}/inside/${inside_name}.ply`。
  - 本地已有 open door GS 门体/把手资产：`door0/1/10/11/12/13/14/19.ply`，部分门有 `door*_knob.ply`；默认真实把手/锁为 `real_knob1.ply` 和 `real_lock1.ply`。
  - 本地已有背景：`wall0..wall13.ply` 和 `inside0.ply`。
  - 注意：这些 override 更换的是视觉高斯，不会自动更换 MuJoCo 物理门、物理把手、抓取点或 stage success 逻辑；视觉门/把手与物理体不对齐时，视频和 success flag 会更加不一致。
- 用户指定门列表为 `door1,door2,door3,door4,door11,door14,door15,door17,door19`。
  - 已从 HF dataset `OpenGHz/auto-atom-assets` 补齐缺失的 `door2/door3/door4/door15/door17.ply`。
  - 用户澄清“不要单独 knob”指不要使用 `door*_knob.ply` 这种每个门自己的 knob；所有门统一使用 `real_knob1.ply` 和 `real_lock1.ply`。
- 开始过一轮 30 随机门/背景组合 sweep，但在第 9 个 `stride=8` 组合后停止，原因是发现动作时间尺度和 gripper 下限尚未修正，该 sweep 不作为有效评估。
- 关键修正：
  - `/home/zoyo/mix/meta/info.json` 显示数据 FPS 为 `20`。
  - AAO open door config 中 `env.update_freq=100`，因此训练数据 1 个 action 对应 AAO `5` 个 update。
  - 之前每个模型 action 只执行 1 个 AAO update，等价于把 20Hz 训练 action 按 100Hz 播放，时间压缩 5 倍。
  - 已在 runner 和 sweep 脚本增加 `--action-repeat`，默认 `5`。
  - 用户确认训练数据最后一维 `right_gripper_position` 是夹爪长度值，范围 `0..0.0945`；模型学到的也是这个长度值，不是抽象开合度。
  - AAO `FingerDistanceMapper` 接收的是 finger distance 长度值。下一步需要确认训练数据里的夹爪长度和 AAO finger distance 是否是同一个测量定义；如果不是，需要加线性/非线性映射，而不是直接透传。
  - AAO replay 配置里对 `eef_claw_joint` 有 `min: 0.02`，用于避免夹爪闭到 0 穿过门把手；闭环 runner/sweep 已加 `--gripper-min 0.02 --gripper-max 0.0945`，默认启用。
  - `smoke_fastwam_repeat5_gripper_clamp` 通过：`stride=8, action_repeat=5, max_updates=40` 下，实际喂给 AAO 的 gripper 为 `0.0893..0.0927`，没有立刻闭合。
- 新闭环设计已同步到代码：
  - 历史记录：单 episode runner 曾默认改为
    `stride=8, action_repeat=5, action_horizon=32`，sweep 曾默认测试
    `stride=4,8`。当前 open-door 默认已改为 `stride=32`。
  - 每次推理后只执行前 `stride` 个模型 action；每个模型 action 在 AAO 中 repeat `5` 个 update；chunk 执行结束后用最新 observation 重新推理。
  - `client_trace.json.gz` 每步新增 `chunk_action_index` 和 `repeat_index`，避免把模型 action index 和仿真 repeat index 混在一起。
  - 单 episode runner 和 sweep 都在 metadata/aggregate 中记录 `action_horizon`、`stride_model_actions`、`action_repeat_sim_updates`、`train_action_hz`、`aao_update_hz`、`effective_action_hz_in_sim`、`gripper_min/max`。
  - `SimulatorServiceClient.init()` 已把 Hydra task 的 `env.update_freq` 写入 `sim_info.env_update_freq`，用于核对 20Hz/100Hz 时间尺度。
  - 单 episode runner 也补了 MuJoCo diagnostics，记录 door/handle joint、body/site 初末状态和 delta；后续不再只依赖 AAO `final_success`。
  - 新增 `--ignore-done` 诊断参数：每步仍调用 AAO task state 评估，但不因为 done 提前停止，用于排查 AAO success flag 假阳性后的完整 rollout。
- 对齐 WorldModel 的 continuous 控制实现：
  - WorldModel 的 continuous 模式不是每步 `PolicyEvaluator.update(action)`，而是 AAO 后台线程按 `sim_loop_frequency` 持续 `env.update()`，主线程用 `set_actions()` 更新当前控制目标。
  - FastWAM 侧已新增 `SimulatorServiceClient.set_cartesian_action(s)`，用于 continuous 模式只更新目标动作。
  - 已移除额外 `--control-mode` 参数，避免接口分叉；现在只用 `--sim-loop-frequency` 控制：`0` 为 lockstep，`>0` 为 continuous。
  - `--obs-interval` 仅用于覆盖 continuous 的等待时间；默认 `-1` 时按 `action_repeat / sim_loop_frequency` 自动计算。
  - `aggregate_summary.json` 和 `client_trace.json.gz` 仍记录推断出的 `control_mode`，方便回看。
- 更新后静态验证通过：
  - `.venv/bin/python -B -m py_compile ...`
  - `.venv/bin/python -B scripts/run_aao_closed_loop_eval.py --help`

## 2026-05-13 闭环 state/proprio 问题定位

- 现象：`real_sim_open_door_uncond_2cam224_1e-4_20k` 的模型在预测视频和离线 action
  看起来正常，但用默认闭环跑 AAO open door 不稳定；同一数据集 episode 用
  `scripts/replay_lerobot_action_to_aao.py` 从 MCAP 首帧初始化并直接 replay
  数据集 action 可以开门成功。
- 结论：闭环默认 `proprio_mode=cartesian` 是错误的。real_1048 和
  sim open door 的 LeRobot `meta/info.json` 都明确写着
  `observation.state` 是
  `[right_arm_joint_0..right_arm_joint_5, right_gripper_position]`，不是
  `[x,y,z,roll,pitch,yaw,gripper]`。
- 影响：
  - 模型条件输入收到分布外 state。
  - 下发到 AAO 后表现为较多 IK failure 或只能转动把手、不能稳定开门。
- 验证：
  - 错误 cartesian proprio 跑 3 个 episode：
    `runs/aao_closed_loop/real_sim_open_door_20k_gs_3ep_480_20260513/`，
    AAO final_success 为 `1/3`。
  - 只改 `--proprio-mode joint`，同 checkpoint/stats/action 语义/执行节奏跑
    3 个 episode：
    `runs/aao_closed_loop/real_sim_open_door_20k_gs_3ep_480_jointprop_20260513/`，
    AAO final_success 为 `2/3`，说明主问题在 state/proprio 语义。
- 已修正：
  - `configs/aao_benchmark/open_door_airbot_play_gs.yaml` 改为
    `proprio_mode: joint`。
  - 单 episode runner 默认 `--proprio-mode joint`。
  - open door sweep 和 visual rollout 脚本显式使用 `proprio_mode=joint`。
  - `.venv/bin/python -B scripts/run_aao_open_door_gs_sweep.py --help`
- 频率控制验证通过：
  - `--sim-loop-frequency 0` + hold policy：记录 `control_mode=lockstep`，`error=null`。
  - `--sim-loop-frequency 100` + hold policy：记录 `control_mode=continuous`，`continuous_target_hold_sec=0.05`，`error=null`。
- 新设计单环境验证结果：
  - `stride=4, action_repeat=5, action_horizon=32, max_updates=160`
    - output: `runs/aao_closed_loop/fastwam_mix20k_open_door_gs_repeat5_stride4`
    - result: `updates_used=160`, `final_done=false`, `final_success=null`, `error=null`
    - diagnostics: `door_hinge_delta≈0`, `handle_hinge_delta≈0.0333 rad`
    - trace: `effective_action_hz_in_sim=20.0`，`repeat_index=0..4` 正确；gripper 范围约 `0.0200..0.0904`
    - 视觉末帧：门未打开。
  - `stride=8, action_repeat=5, action_horizon=32, max_updates=160`
    - output: `runs/aao_closed_loop/fastwam_mix20k_open_door_gs_repeat5_stride8`
    - result: `updates_used=53`, `final_done=true`, `final_success=true`, `error=null`
    - diagnostics: `door_hinge_delta≈0.0192 rad`, `handle_hinge_delta≈0.0186 rad`
    - 视觉末帧：门只发生很小位移，不能视作真实开门；该结果继续按 AAO success flag 假阳性处理。
  - `stride=8, action_repeat=5, action_horizon=32, max_updates=80, --ignore-done`
    - output: `runs/aao_closed_loop/fastwam_mix20k_open_door_gs_repeat5_stride8_ignore_done_80`
    - result: `updates_used=80`, `final_done=false`, `final_success=null`, `error=null`
    - diagnostics: `door_hinge_delta≈0.0047 rad`, `handle_hinge_delta≈0.0243 rad`
    - 结论：继续执行并没有带来稳定真实开门；AAO done 判据和实际视觉/物理开门仍需分开看。
- 注意：`runs/aao_closed_loop/fastwam_mix20k_open_door_gs_repeat5_stride8_ignore_done` 是 `--ignore-done` 第一次修正前的调试目录，不作为有效结果使用。
- 已按用户要求删除旧 smoke 测试目录：
  - `runs/aao_closed_loop/smoke_*`
- 2026-05-08 晚间启动过新的 30 环境 lockstep sweep：
  - command: `.venv/bin/python -B scripts/run_aao_open_door_gs_sweep.py --gpu 3 --device cuda:0 --output-dir runs/aao_closed_loop/fastwam_mix20k_open_door_gs_30env_repeat5_lockstep_20260508 --num-combos 30 --strides 4,8 --max-updates 160 --action-repeat 5 --action-horizon 32 --num-inference-steps 10 --sim-loop-frequency 0 --log-level INFO`
  - 因 GPU 资源过满，用户要求先停止；已终止 PID `1647397`。
  - 该目录只包含部分已完成 episode，不作为完整 30 环境结果使用。
- README 顶部已新增 AAO open-door closed-loop integration 总说明，记录入口脚本、
  显式模型路径要求、`--sim-loop-frequency` 语义、默认时间尺度和成功判据注意事项。
- README/README_zh 已同步部署说明：
  - fresh clone 使用 `git clone --recurse-submodules`，未递归 clone 时用 `git submodule update --init --recursive`。
  - AAO 从本地 `third_party/auto-atomic-operation[mujoco]` 安装，GaussianRenderer 从本地 `third_party/GaussianRenderer[shs,mujoco]` 安装，避免重新拉浮动 Git 依赖。
  - AAO MuJoCo mesh 需要在 submodule 内执行 `git lfs pull --include "assets/meshes/**" --exclude "assets/videos/**"`。
  - open-door GS `.ply` 资产需要从 HF dataset `OpenGHz/auto-atom-assets` 下载到 `third_party/auto-atomic-operation`，README 中的 `--include` 已改为 `assets/gs/...` 路径，确保落盘为 `third_party/auto-atomic-operation/assets/gs/...`。
  - `runs/` 下的 mix 20k 模型产物不会提交，README 已说明需要单独复制/下载 `config.yaml`、`dataset_stats.json`、text embedding cache 和 checkpoint，或用 CLI 参数显式指定其他 run。
- 历史记录：代码曾默认使用仓库根目录下的 mix 20k 相对路径。当前
  runner/benchmark/visual/sweep 都不再隐式指定 checkpoint；FastWAM 模型评测必须
  显式传 `--fastwam-config`、`--checkpoint`、`--dataset-stats` 和
  `--text-cache-dir`。
- `.gitignore` 已补充 `.hf_cache/`、`.local/`、`data.local_before_origin_main_*/`，避免本地缓存和远端同步前备份目录误提交。

## 2026-05-11

### 已完成

- 重新检查 AAO 上游：
  - OpenGHz upstream `main`: `5303f50e0366a0c14560da133b8c85871bf0b95d`
  - DISCOVER submodule remote `main`: `8a9d5c76136ba7e02e14d25646d9b614ab984081`
  - FastWAM submodule 指针已切到 DISCOVER 远端可拉取的 `8a9d5c7`，取代本地临时 merge commit `4b9f432`。
  - `4b9f432` 和 `8a9d5c7` 的 tree 完全一致；最终使用 `8a9d5c7` 是为了保证其他用户递归 clone/submodule update 时能从 DISCOVER 远端取到 commit。
- 新 AAO API 兼容：
  - AAO 将 `load_task_file_hydra` 从 `auto_atom.runtime` 移到 `auto_atom.config_loader`。
  - `SimulatorServiceClient` 已优先从 `auto_atom.config_loader` 加载，旧版本 fallback 到 `auto_atom.runtime`。
- 新 AAO smoke 验证：
  - `SimulatorServiceClient.connect()` 通过。
  - `open_door_airbot_play_gs` Hydra config resolve 通过，`env_update_freq=100`，operator 为 `arm`。
  - `open_door_airbot_play_gs` 能 `init/reset/update` 一步；默认相机 `env2_cam` 和 `eef_wrist_cam` 仍可用。
- 永久化 visual rollout：
  - 新增 `scripts/run_aao_visual_rollout.py`，用于生成 pred / VAE recon / actual simulator 的 3x2 对比视频。
  - `FastWAMModelClient` 新增 `infer_joint_video()`，一次 joint 推理得到动作 chunk 和预测视频。
  - `FastWAMModelClient` 新增 `reconstruct_video_from_model_inputs()`，用模型 VAE 对实际 AAO 观测帧做 recon。
  - 新增 `--frame-sampling {model-action,sim-update}`：
    - `model-action`：每个模型 action 输出一帧。
    - `sim-update`：每个 AAO update 输出一帧。
  - 已用 mix 20k + `open_door_airbot_play_gs` 生成过 2-window AAO update 级别视频：
    - output: `runs/aao_closed_loop/mix20k_open_door_gs_2win_visual_simupdate_20260509/pred_vae_actual_3x2.mp4`
    - `model_steps_used=64`
    - `sim_updates_used=320`
    - `output_frames=320`
    - video shape: `672x448`, fps `10`
  - 说明：mix 配置下 `action_horizon=32, num_video_frames=9`，pred/recon 每个 window 只有 9 个模型视频帧；`sim-update` 模式按最近邻展开到每个 AAO update，actual 行为逐 AAO update 真实取图。
- README / README_zh 已补充：
  - 当时 AAO submodule pin 为 `8a9d5c7`，包含 OpenGHz upstream `5303f50`。
  - visual rollout 入口和 `--frame-sampling sim-update` 示例。

### 需要确认

- open door FastWAM checkpoint 的准确路径。
- 是否需要同步 AAO 的 LFS/大文件资产；如果 smoke test 提示 mesh/ply/xml 资产缺失，需要补充 `git lfs pull`。
- open door 仿真任务当前默认使用 `open_door_airbot_play_back_gs`；front 对照显式
  使用 `open_door_airbot_play_gs`。
- 若 GS 依赖或资产缺失，临时降级排查可用 `open_door_airbot_play_g2p`。

### 下一步

1. 后续 open-door 正式闭环/benchmark 默认使用
   `open_door_airbot_play_back_gs`、`stride=32`、`action_repeat=5`、joint
   proprio、关闭 arm randomization，并显式传入
   `--fastwam-config/--checkpoint/--dataset-stats/--text-cache-dir`。
2. 继续用 `mujoco_diagnostics` 的 door/handle hinge delta 和视频末帧作为主判据；
   不再只依赖 AAO `final_success`。
3. 如果 `stride=32` 下仍不能稳定真实开门，优先排查坐标系、相机域差异和
   gripper length 标定；较小 stride 仅作为 receding-horizon 诊断参数。

## 2026-05-13

### 已完成

- 重新同步 AAO submodule 到 DISCOVER 远端 `d831ee7cecd4b87df2337bab7ec856d5a342b412`。
- 确认 AAO `apply_pose_action()` 需要绝对 EEF target，不接受 LeRobot delta 直接下发。
- 修正 FastWAM bridge 默认 action 语义：当前 `delta6_abs_gripper` 会把 LeRobot
  `pose[t] - pose[t-1]` backward delta 左移一帧，然后从当前 EEF pose 累计积分；
  gripper 是绝对 target，只做同样的一帧时间对齐，不做积分。
- HTTP inference 服务不再在 delta action mode 下把 `proprio_raw[:6]` 当作积分基准；
  调用方必须传当前 EEF pose `current_position` 或 `cartesian_position`。
- 新增 `scripts/replay_lerobot_action_to_aao.py`，用于把本地 LeRobot sim episode
  的 action 直接 replay 到 AAO，输出 `aao_replay_multicam.mp4`、
  `dataset_vs_aao_replay.mp4`、`summary.json` 和 `trace.json.gz`。
- 已用 `/DATA/disk1/zoyo/sim/open_door_augmented_sim_lerobot` 的
  `episode_000000` / `door_2` 验证 replay：输出在
  `runs/aao_dataset_action_replay/episode_000000_door2_20260513`，dataset 118 帧，
  replay 117 个 action，door hinge 从约 `0` 到 `-0.231 rad`，handle hinge
  增加约 `0.053 rad`。
- 已用 mix 20k ckpt 跑 3 个 open-door AAO episode，视频在
  `runs/aao_closed_loop/mix20k_open_door_gs_3ep_20260513_actionfix/`。
- 之后的正式 open-door 运行已改为使用 real+sim open-door 20k run，并要求
  显式传入 config、checkpoint、dataset stats 和 text cache；mix 20k 路径不再是
  任何脚本默认值。

### 注意

- 旧文档中把 AAO 当前 pin 写成 `8a9d5c7` 或把 HTTP response 写成
  `joint_absolute` 的描述已经过时；当前 AAO pin 是 `d831ee7`，AAO bridge 默认
  输出 `cartesian_absolute`。
