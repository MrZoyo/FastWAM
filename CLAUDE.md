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

## v2 Active-Loop Server (real-world AIRBOT control)

`scripts/fastwam_active_loop_server.py` (port 8118) is the self-fetch closed-loop
server: it pulls 1088×1280 stereo frames from `rgbd_ws_bridge`, ARM joint /
EEF state from `arm_sdk`, runs FastWAM @ 2.5 Hz, and dispatches 20 Hz
absolute pose targets through `arm_client.send_pose`.

`scripts/fastwam_http_server.py` (port 8117) is the passive v1 server kept
for offline replay / unit tests. Both servers share `undistort_native` and
`FastWAMModelClient._preprocess_image` (1088×1280 → stitch → aspect-preserving
resize → center crop 224×448) — no `cv2.resize` to 480×640 happens server-side.

Authoritative docs:

- `docs/fastwam_arm_integration.md` — operator manual + arm-control reference
- `docs/fastwam_http_server.md` — both servers, CLI / endpoint / troubleshooting
- `docs/fastwam_http_server_self_fetch_design.md` — full design (§3 timing,
  §5 CLI, §7 modules, §8 errors, §10 risks)

### Start / stop

```bash
# defaults: --port 8118 --device cuda:1 --ws-url ws://192.168.31.67:19095
#           --arm-host 192.168.31.34 --arm-port 50051 --auto-dispatch=False
#           --emergency-on-failure (use --no-emergency-on-failure to disable)
export NO_PROXY=192.168.31.67,192.168.31.34,127.0.0.1,localhost
.venv/bin/python scripts/fastwam_active_loop_server.py

# detached background launch (model + warmup + benchmark ≈ 34s before serve)
nohup .venv/bin/python scripts/fastwam_active_loop_server.py \
    --port 8118 --log-level INFO \
    > /tmp/active_loop_server.log 2>&1 &
disown $!

# stop / cleanup
pgrep -af "fastwam_active_loop\|fastwam_http_server"
ss -tlnp 2>/dev/null | grep -E "8117|8118"
kill <pid>
```

### Probe upstream before launch

```bash
# WS upstream (ws://192.168.31.67:19095) — must be pushing frames, not just open
.venv/bin/python scripts/fastwam_ws_probe.py --max-packets 5

# ARM gRPC (192.168.31.34:50051) — read-only smoke, does NOT acquire control
.venv/bin/python - <<'PY'
import logging, time
from fastwam.server.arm_client import ArmClient
logging.basicConfig(level=logging.INFO)
c = ArmClient(host="192.168.31.34", port=50051, poll_hz=10,
              state_max_age_ms=500, lease_ms=15000,
              logger=logging.getLogger("arm"))
c.start(); time.sleep(1.0)
print(c.latest()); print(c.health()); c.stop()
PY
```

### Operate the running server (cURL)

```bash
# 1. health (always safe)
curl -s http://localhost:8118/health | jq .

# 2. zero-pose dry-test (arm MUST NOT move; verifies coord frame; PR9 R4)
curl -s -X POST http://localhost:8118/debug/zero_pose_test \
    -H 'Content-Type: application/json' -d '{"duration_s": 5.0}' | jq .

# 3. start closed loop (uses cached "open the door" text embed by default)
curl -s -X POST http://localhost:8118/start \
    -H 'Content-Type: application/json' -d '{}' | jq .

# 4. live status
watch -n 0.5 'curl -s http://localhost:8118/closed_loop_status | jq "{
    chunk_id: .current_chunk_id,
    p50: .infer_latency_ms_p50,
    dispatch: .dispatch.last_dispatch_idx,
    hold: .hold_mode,
    arm: .dispatch.send_fail_count
}"'

# 5. stop / emergency (estop is latching — must clear before next /start)
curl -s -X POST http://localhost:8118/stop -d '{}' | jq .
curl -s -X POST http://localhost:8118/emergency -d '{"enable": true}'  | jq .
curl -s -X POST http://localhost:8118/emergency -d '{"enable": false}' | jq .  # release
```

### Bring-up checklist (mandatory before real `/start`)

1. **Pre-compute text embeddings** for any instruction other than `"open the door"`:
   `python scripts/precompute_text_embeds.py --instructions "..." --output-dir data/text_embeds_cache/real_1048`
2. **Start upstream in this order**: `arm_sdk` gRPC → `rgbd_ws_bridge` → server.
   Server aborts after 30 s if no WS first frame.
3. **GPU 1, not GPU 0** (default `--device=cuda:1`). GPU 0 is reserved for AirDC.
4. **Zero-pose dry-test FIRST** — arm in safety enclosure, run step 2 above. Arm
   must not move; any drift = coordinate-frame bug, abort.
5. **First real run uses `--no-auto-dispatch`** (default). InferLoop runs and
   logs everything but never calls `send_pose`. Promote to `--auto-dispatch`
   only after `closed_loop_status` looks healthy.

### Run all tests

```bash
.venv/bin/python -m pytest tests/ -v          # 160 passed + 1 skipped expected
.venv/bin/python -m pytest tests/test_arm_client.py        # ArmClient unit
.venv/bin/python -m pytest tests/test_active_loop_server.py  # HTTP handlers
```

### Known infrastructure quirks

- Shared venv has `fastwam` installed editable from main worktree
  (`/data/home/Lyle/Projects/FastWAM/src`). Working in a `FastWAM-wt/*`
  worktree requires `tests/conftest.py` (already in repo) to prepend the
  local `src/` to `sys.path`; runtime smoke smartens need
  `PYTHONPATH=$worktree/src`.
- `tests/fixtures/ws_packets_v4.bin` is **not** in git; libx264 output is
  not stable across PyAV decoder contexts. `tests/conftest.py` regenerates
  the V4 packet fixture every session (cheap, ~5 frames).
- v2 server defaults `--device cuda:1` because `5090_1` GPU 0 is owned by
  the AirDC visual-capture process. Override only if `airdc` is offline.
- All LAN traffic (`192.168.31.x`) must bypass the proxy. The server's main
  injects `NO_PROXY` for `ws-url` and `arm-host` automatically, but cURL /
  smoke scripts run outside the server need it in the calling shell.
