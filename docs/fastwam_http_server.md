# FastWAM HTTP Server

The server loads one FastWAM checkpoint and exposes action chunk inference over JSON HTTP.
It uses only the Python standard library for HTTP, so it does not require FastAPI or uvicorn.

Incoming RGB frames are routed by their actual `(H, W)`:

- **1088x1280** raw stereo frames: the server undistorts each eye at native resolution
  and hands the resulting 1088x1280 frames directly to the model client. There is
  *no* server-side `cv2.resize` to 480x640 — the aspect-preserving short-side resize
  and center crop to `224x448` (each eye 224x224) happen inside
  `FastWAMModelClient._preprocess_image`, matching the training-time transform.
  Calibration may come from the request (`undistort.left_camera_info` /
  `right_camera_info`) or from the server-side default file loaded at startup via
  `--default-camera-info` (see [Server-side defaults](#server-side-defaults)).
  Anything provided in the payload takes precedence over the default for that camera.
- **480x640** training-aligned frames: passed through as-is. Any `undistort` field in the
  payload is ignored. OpenCV is not required.
- Any other resolution returns HTTP 400. All camera streams in a single request must share
  the same resolution.

## Prepare real_1048 Assets

The default service target is `real_1048_uncond_2cam224_1e-4`.

Required local paths:

- dataset: `/data_hdd/Lyle/Datasets/real_1048`
- checkpoint: `runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/checkpoints/step_020000.pt`
- dataset stats: `runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/dataset_stats.json`
- text cache: `data/text_embeds_cache/real_1048`

Generate stats and text cache from the repo root if they are missing:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.utils.misc import register_work_dir

run_dir = Path("runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15").resolve()
register_work_dir(run_dir)
with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
    cfg = compose(config_name="train", overrides=["task=real_1048_uncond_2cam224_1e-4"])
OmegaConf.resolve(cfg)
_ = instantiate(cfg.data.train)
PY

CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -u scripts/precompute_text_embeds.py \
  task=real_1048_uncond_2cam224_1e-4 \
  +overwrite=false \
  model.redirect_common_files=true
```

## Start

From `/home/Lyle/Projects/FastWAM`:

```bash
CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -B scripts/fastwam_http_server.py \
  --host 0.0.0.0 \
  --port 8117 \
  --config configs/task/real_1048_uncond_2cam224_1e-4.yaml \
  --checkpoint runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/checkpoints/step_020000.pt \
  --dataset-stats runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/dataset_stats.json \
  --text-cache-dir data/text_embeds_cache/real_1048
```

`CUDA_VISIBLE_DEVICES=1` starts the service on physical GPU 1. Inside the process,
PyTorch still reports the selected visible GPU as `cuda:0`.

## Endpoints

- `GET /health`: model metadata, required camera keys, `proprio_dim`, horizon. Also reports
  `accepted_image_resolutions=[[1088,1280],[480,640]]`, `undistort_required_at=[1088,1280]`,
  and `image_pipeline="undistort_only (model client handles stitch+resize+crop)"`. When the server is launched with
  `--default-camera-info` pointing at a valid JSON file, it additionally reports
  `default_camera_info_loaded=true`, `default_camera_info_path`,
  `default_camera_info_keys` (list of image keys with defaults), `default_stereo_pair`
  (e.g. `{"left": "head_left", "right": "right_wrist_left"}`), and the
  `default_instruction` that the server uses when a client omits `instruction`.
- `POST /infer`: run inference.
- `POST /predict_action`: alias of `/infer`.

## Server-side defaults

Two payload fields can be omitted by the client and resolved server-side:

- `instruction`: when the request omits this field, sends `null`, or sends an empty string,
  the server uses its `--instruction` CLI default (`"open the door"` by default for
  real_1048). A matching text-embedding cache must still exist under `--text-cache-dir`.
- `undistort` (1088x1280 path only): when the server starts with a valid
  `--default-camera-info` JSON (default: `configs/camera_info/real_1048_default.json`), it
  loads `{stereo_pair, cameras}` so that 1088x1280 requests no longer need to include
  any `undistort` block. The `cameras` map is keyed by image key (e.g. `head_left`,
  `right_wrist_left`) and provides the per-camera `k`/`d` used for native-resolution
  undistortion. The optional `stereo_pair` overrides the stereo-key lookup fallback
  (which would otherwise pull the first two keys from `image_shapes`).

Payload values always win over server defaults at field granularity: a request can supply
`left_camera_info` only and the server will still fill in `right_camera_info` from the
default file. Setting either `instruction` or any individual `undistort.*` field in the
payload disables only that specific override.

The minimal client-side request therefore needs just `images`, `proprio_raw`, and the
`current_position` required by delta action modes. Note that the `proprio_raw`,
`images`, and `current_position` validations are unchanged.

## POST /infer Request

The examples below show the new minimal payloads. `instruction` and `undistort` can
still be supplied — when present they take precedence over the server defaults.

### Raw `1088x1280` stereo (minimal request, server defaults supply calibration)

```json
{
  "images": {
    "head_left": "<base64 encoded png/jpeg, 1088x1280>",
    "right_wrist_left": "<base64 encoded png/jpeg, 1088x1280>"
  },
  "proprio_raw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
  "current_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

If you need to override the server defaults (e.g. you have a freshly recalibrated rig),
an explicit `undistort` block still works and wins per-field:

```json
{
  "instruction": "open the door",
  "images": {
    "head_left": "<base64 encoded png/jpeg, 1088x1280>",
    "right_wrist_left": "<base64 encoded png/jpeg, 1088x1280>"
  },
  "proprio_raw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
  "current_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "undistort": {
    "left_image_key": "head_left",
    "right_image_key": "right_wrist_left",
    "left_camera_info": {
      "width": 1280,
      "height": 1088,
      "k": [ ... ],
      "d": [ ... ]
    },
    "right_camera_info": {
      "width": 1280,
      "height": 1088,
      "k": [ ... ],
      "d": [ ... ]
    },
    "alpha": 0.0
  }
}
```

### Already-aligned `480x640` frames (minimal request)

```json
{
  "images": {
    "head_left": "<base64 encoded png/jpeg, 480x640>",
    "right_wrist_left": "<base64 encoded png/jpeg, 480x640>"
  },
  "proprio_raw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
  "current_position": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

`images` keys must match the model config. `real_1048` expects `head_left` and `right_wrist_left`.
`proprio_raw` is the 7D real_1048 state: six arm joint values followed by gripper position.
The default `action_mode` is `delta6_abs_gripper`, matching the current LeRobot
data: the first six action dimensions are frame-aligned EEF deltas
`pose[t] - pose[t-1]`, while the final gripper dimension is an absolute target.
The server shifts the predicted action chunk left by one frame, cumulatively
integrates the first six dimensions from the current 6D EEF pose, and keeps the
gripper as an absolute target. For delta modes, the request must include
`current_position` or `cartesian_position` as `[x,y,z,roll,pitch,yaw]`; this is
not the same as `proprio_raw[:6]` when `proprio_raw` is joint state. Use `--model-action-mode
delta6_abs_gripper_forward` only for checkpoints trained on already-forward
EEF deltas `pose[t+1] - pose[t]`.

For the `1088x1280` path, the server treats the two model images as a stereo pair and runs
OpenCV undistortion/rectification at **native resolution** (no `K` scaling — calibration must
match the source resolution). The undistorted 1088x1280 frames are handed straight to the
model client; `FastWAMModelClient._preprocess_image` stitches the two eyes horizontally,
applies aspect-preserving short-side resize, and center-crops to `224x448` (each eye 224x224).
`left_to_right` is optional: when provided, OpenCV performs stereo rectification; otherwise
each eye is undistorted independently with its own `K`/`d`. The legacy fields
`undistort.enabled` and `undistort.output_size` are no longer honored — they are silently
ignored.

For the `480x640` path, the server skips all OpenCV work; any `undistort` payload is ignored.
This is the right choice if the client has already undistorted and resized upstream.

`instruction` must have a matching precomputed text embedding in `--text-cache-dir`.
If the cache is missing, the response is HTTP 500 with the missing cache path.

The following malformed requests return HTTP 400:

- non-7D `proprio_raw` or missing camera keys
- images at a resolution other than `1088x1280` or `480x640`
- cameras in the same request with mismatched resolutions
- `1088x1280` images for which neither the payload nor the server-side default file
  provides `left_camera_info` / `right_camera_info` for the resolved stereo keys
- image bytes that are not `H x W x 3` `uint8` after base64 decoding

## Response

```json
{
  "action_format": "cartesian_absolute",
  "actions": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  "source": "fastwam",
  "action_mode": "delta6_abs_gripper",
  "action_semantics": "...",
  "normalized_action_shape": [32, 7],
  "full_prompt": "...",
  "instruction": "..."
}
```

`actions` is an `[action_horizon, 7]` list. The first six dimensions are
absolute targets after the one-frame shift and cumulative delta integration;
the last dimension is the predicted absolute gripper target.

## Verified On 5090

- `GET /health` reports `image_keys=["head_left","right_wrist_left"]`, `proprio_dim=7`, and checkpoint `step_020000.pt`.
- `GET /health` also reports `opencv_available=true` on the 5090 runtime.
- A real dataset frame returns HTTP 200 with `action_format="cartesian_absolute"` and `actions` shape `[32,7]` when the request includes the current EEF pose.

---

# Active Loop Server (v2, self-fetch mode)

`scripts/fastwam_active_loop_server.py` is the v2 active-loop companion to
the v1 passive server above. The v1 server stays unchanged on port `8117`;
the v2 server listens on port `8118` and is responsible for driving its own
closed loop.

| | v1 passive (`fastwam_http_server.py`) | v2 active (`fastwam_active_loop_server.py`) |
| --- | --- | --- |
| Port (default) | 8117 | 8118 |
| Frame source | Client POSTs base64 PNG/JPEG | Server pulls from `rgbd_ws_bridge` WS |
| ARM state | Client supplies `proprio_raw` + `current_position` | Server polls `arm_sdk` directly |
| Image normalization | undistort only; model_clients does stitch + resize + crop | same — no server-side resize on either path |
| Endpoints | `POST /infer`, `GET /health` | `POST /start /stop /emergency /debug/zero_pose_test`, `GET /health /closed_loop_status /ws_status` |
| Use when | Offline replay, integration testing | Real-time autonomy on hardware |

The v2 server never exposes `/infer`; the model is driven by the internal
`ClosedLoopRunner`. See
`docs/fastwam_http_server_self_fetch_design.md` for the full design.

## v2 Endpoints

### `POST /start`

Body (optional): `{"instruction": "open the door"}`. If `instruction` is
absent or `null`, the runner uses the `--instruction` default and the
existing precomputed text-embedding cache.

The handler calls `arm.acquire_control()` first, then
`runner.start(instruction)`. Both calls run under a single internal lock so
concurrent `/start /stop` requests cannot interleave.

### `POST /stop`

No body. Calls `runner.stop()` then `arm.release_control()`.

### `POST /emergency`

Body (optional): `{"enable": true}` (default) or `{"enable": false}`. Maps
directly to `arm_sdk.set_arm_emergency_stop(enable)`. After tripping the
estop you MUST call `POST /emergency {"enable": false}` before the next
`/start` request will succeed.

### `GET /health`

Returns `{"status": "ok", "server": "fastwam_active_loop_server", "arm":
arm.health(), "ws": ws.health()}`. `arm.health()` includes
`lease_alive`, `last_poll_age_ms`, `consecutive_fail`. `ws.health()` includes
`last_frame_age_per_channel_ms`, `fps_5s`, `decode_fail_count`, etc.

### `GET /closed_loop_status`

Forwards `runner.status()` — chunk index, dispatch latency stats, watchdog
status.

### `GET /ws_status`

Verbose `ws.health()` view (per-channel decode latencies, key-frame gaps,
last reconnect reason). Useful for debugging dropped frames.

### `POST /debug/zero_pose_test`

Body: `{"duration_s": 5.0}`. Reads `arm.latest()`, builds a 32-frame chunk
where every row is `[x, y, z, roll, pitch, yaw, gripper] = current EEF
pose`, and injects it directly into the dispatcher (bypassing the model).
The arm MUST NOT move. Any observed pose drift means the coordinate-system
plumbing in `arm_client.send_pose` or the dispatcher's blend-and-extend is
wrong; abort and fix the conversion before proceeding to a real `/start`.

Internally calls `runner.start(None)` followed by
`runner.inject_chunk_for_debug(action_abs, base_ts_ns)` (the latter ships
with PR5). Returns 503 if no fresh `arm.latest()` is available.

## Bring-up Checklist (mandatory before real `/start`)

1. **Pre-compute text embeddings.** If the instruction differs from the
   training-time prompt, run `scripts/precompute_text_embeds.py` and point
   `--text-cache-dir` at the new cache. The runtime uses a single
   text-embed cache entry — switching instructions live is unsupported.
2. **Start upstream services in this order**: `arm_sdk` gRPC server →
   `rgbd_ws_bridge` → `fastwam_active_loop_server.py`. The server's
   startup self-check waits for the first WS frame (default timeout 30 s);
   if it sees no frame it aborts with `exit(1)` rather than start with a
   silent stream.
3. **Pick GPU 1, not GPU 0.** Default `--device=cuda:1`. GPU 0 on `5090_1`
   is reserved for the AirDC visual capture process and is usually pinned
   at >40 % utilization. Override with `--device=cuda:0` only if AirDC is
   confirmed offline.
4. **Run the zero-pose dry-test FIRST.** With the arm powered on but
   inside its safety enclosure: start the server, then
   `POST /debug/zero_pose_test {"duration_s": 5.0}`. The arm must not
   move. If it does, halt and diagnose the coordinate frame.
5. **First closed-loop run uses `--auto-dispatch=false`.** This disables
   SDK calls; the runner still runs inference and logs everything. Once
   the chunk write rate, blend indices, and watchdog logs look healthy,
   relaunch with `--auto-dispatch=true`.

## Startup Banner (design §7.7.1)

On launch the server prints (INFO level) a fixed banner enumerating
`train_config_path`, `train.num_frames`, `chunk_len`, `action_output_dim`,
`context_len`, `num_inference_steps`, `device`, `gpu_free_mem_gb`,
`ws_url`, `ws_channels`, `arm_host:port`, `arm_lease_ms`,
`infer_period_ms`, `send_period_ms`, `blend_frames`, `image_pipeline`,
`scipy_version`, rotation-fingerprint result, warmup latency, benchmark
percentiles. Any of the following abort the launch with `exit(1)`:

- training-config / CLI mismatch
- rotation fingerprint failure (scipy convention regression)
- GPU free memory < `--require-gpu-mem-free-gb` (default 8 GiB)
- benchmark `p50 > 500 ms`
- WS startup self-check or 30 s first-frame timeout

## Rollback (design §11)

- **L0** — runtime softstop: pass `--auto-dispatch=false`. Inference and
  logging continue; nothing is sent to the manipulator.
- **L1** — image-pipeline rollback: pass
  `--image-pipeline=lerobot_480x640` to reproduce the v1 stitch path.
- **L2** — full process rollback: stop the v2 server, start the v1
  passive `scripts/fastwam_http_server.py` on port 8117. The v1 path is
  unchanged.
- **L3** — code rollback: delete `src/fastwam/server/` and
  `scripts/fastwam_active_loop_server.py`. All new code lives in
  isolated paths so a single revert restores the pre-PR state.
- Malformed requests return HTTP 400, including non-7D `proprio_raw` and missing camera keys.

## v2 CLI Reference

| Flag | Default | Description |
| --- | --- | --- |
| `--host` | `0.0.0.0` | HTTP bind host |
| `--port` | `8118` | HTTP bind port (v1 server uses 8117) |
| `--config` | `configs/task/real_1048_uncond_2cam224_1e-4.yaml` | Model config |
| `--checkpoint` | `runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/checkpoints/step_020000.pt` | Trained ckpt path |
| `--dataset-stats` | `runs/.../dataset_stats.json` | Normalizer stats path |
| `--text-cache-dir` | `data/text_embeds_cache/real_1048` | Pre-computed text embed cache |
| `--default-camera-info` | `configs/camera_info/real_1048_default.json` | Per-camera intrinsics for undistort |
| `--ws-url` | `ws://192.168.31.66:19095` | rgbd_ws_bridge address |
| `--ws-frame-max-age-ms` | `250` | Drop frames older than this in `latest()` |
| `--ws-reconnect-backoff-ms` | `500,1000,2000,5000,10000` | Reconnect backoff schedule (caps at last value) |
| `--ws-startup-timeout-ms` | `30000` | Abort if no first frame within this timeout |
| `--ws-warn-stale-ms` | `200` | Watchdog WARN on per-channel age |
| `--ws-hold-stale-ms` | `500` | Watchdog HOLD on per-channel age |
| `--ws-estop-stale-ms` | `1500` | Watchdog EMERGENCY_STOP on per-channel age |
| `--arm-host` | `192.168.31.34` | arm_sdk gRPC host |
| `--arm-port` | `50051` | arm_sdk gRPC port |
| `--arm-poll-hz` | `50` | ARM state poller frequency |
| `--arm-state-max-age-ms` | `100` | Drop state older than this in `latest()` |
| `--arm-lease-ms` | `15000` | acquire_control lease duration (SDK auto-renews @ 5s) |
| `--arm-acquire-on` | `start` | `start` = acquire on `/start` (default); `init` = acquire at boot |
| `--infer-period-ms` | `400` | Inference cadence (2.5 Hz) |
| `--send-period-ms` | `50` | Dispatch cadence (20 Hz) |
| `--blend-frames` | `4` | New/old chunk linear-blend overlap window |
| `--chunk-len` | `None` (auto) | Action chunk length. `None` → `train_config.num_frames - 1` (=32) |
| `--num-inference-steps` | `None` (auto) | Diffusion steps. `None` → `train_config.eval_num_inference_steps` (=10) |
| `--chunk-max-stale-ms` | `2000` | Watchdog HOLD when newest chunk older than this |
| `--auto-dispatch` | `False` | Send `arm.send_pose` only when set; otherwise dispatcher logs targets but does NOT command the arm |
| `--emergency-on-failure` | `True` | Trip `set_arm_emergency_stop(True)` on hard fault |
| `--watchdog-period-ms` | `10` | Watchdog tick period |
| `--instruction` | `"open the door"` | Default task; `/start` may override via JSON body |
| `--device` | `cuda:1` | GPU device. Default avoids `cuda:0` (reserved for AirDC capture) |
| `--require-gpu-mem-free-gb` | `8.0` | Abort if free GPU memory falls below this at startup |
| `--image-pipeline` | `raw_native` | `raw_native` = 1088×1280 → undistort → model. `lerobot_480x640` reproduces v1 stitch path |
| `--undistort` / `--no-undistort` | `--undistort` | Toggle undistort step (rarely needed) |
| `--warmup-infer-calls` | `5` | Warmup inferences during startup |
| `--benchmark-infer-calls` | `10` | Benchmark inferences for p50/p95/p99 |
| `--benchmark-p50-budget-ms` | `500` | Abort if benchmark p50 exceeds this |
| `--skip-warmup` | `False` | Skip warmup+benchmark (useful for dev / smoke tests) |
| `--log-level` | `INFO` | Standard `logging` level |

## Example Session (cURL)

```bash
# 1. Health check (always safe; cheap)
curl -s http://localhost:8118/health | jq .
# {
#   "status": "ok",
#   "server": "fastwam_active_loop_server",
#   "arm":  { "last_poll_age_ms": 19.5, "consecutive_fail": 0, "lease_alive": false, ... },
#   "ws":   { "last_frame_age_per_channel_ms": {"head_left": 113.5, "right_wrist_left": 130.0},
#             "fps_5s": 12.4, "decode_fail_count": 0, "reconnect_count": 0, "pair_seq_gaps": 0 }
# }

# 2. Zero-pose dry-test (arm MUST NOT move; verifies coordinate frame)
curl -s -X POST http://localhost:8118/debug/zero_pose_test \
  -H 'Content-Type: application/json' -d '{"duration_s": 5.0}' | jq .

# 3. Start a real closed loop (default instruction)
curl -s -X POST http://localhost:8118/start \
  -H 'Content-Type: application/json' -d '{}' | jq .

# 3b. Or override the instruction (must already exist in text-embed cache)
curl -s -X POST http://localhost:8118/start \
  -H 'Content-Type: application/json' -d '{"instruction": "open the door"}' | jq .

# 4. Live status while running
curl -s http://localhost:8118/closed_loop_status | jq .

# 5. Graceful stop
curl -s -X POST http://localhost:8118/stop -d '{}' | jq .

# 6. Emergency stop (latching; clear with {"enable": false})
curl -s -X POST http://localhost:8118/emergency \
  -H 'Content-Type: application/json' -d '{"enable": true}' | jq .
# ...later:
curl -s -X POST http://localhost:8118/emergency \
  -H 'Content-Type: application/json' -d '{"enable": false}' | jq .
```

## Real Startup Log (verified on `5090_1`)

Trimmed sample from a real boot (model load + fingerprint + ARM poller +
WS connect):

```
INFO [startup] script               = fastwam_active_loop_server.py v2
INFO [startup] train_config_path    = runs/real_1048_uncond_2cam224_1e-4/2026-05-14_10-51-15/config.yaml
INFO [startup] device               = cuda:1
INFO [startup] ws_url               = ws://192.168.31.66:19095
INFO [startup] ws_channels          = head_left, right_wrist_left  (identity mapping)
INFO [startup] arm_host:port        = 192.168.31.34:50051
INFO [startup] arm_lease_ms         = 15000     (SDK auto-renew @ 5s)
INFO [startup] infer_period_ms      = 400       (2.5 Hz)
INFO [startup] send_period_ms       = 50        (20.0 Hz)
INFO [startup] blend_frames         = 4
INFO [startup] image_pipeline       = raw_native (1088x1280 -> undistort -> model_clients)
INFO [startup] scipy_version        = 1.15.3
INFO [startup] gpu_free_mem_gb      = 30.83      (require >= 8.00)
INFO loading FastWAMModelClient checkpoint=runs/.../step_020000.pt
INFO Loading Wan2.2-TI2V-5B components...
INFO Finished loading Wan2.2-TI2V-5B components in 17.22 seconds.
INFO Initialized MoT with experts: ['video', 'action'], num_layers=30
INFO   Expert 'video': num_params=5.00 B
INFO   Expert 'action': num_params=1.02 B
INFO [startup] rotation fingerprint test (5 GT samples from opendoor_real_1048): PASS
INFO [startup] train.num_frames     = 33
INFO [startup] chunk_len            = 32        (= num_frames - 1, frame-aligned backward delta)
INFO [startup] action_output_dim    = 7         (xyz + rpy + gripper)
INFO [startup] num_inference_steps  = 10        (from eval_num_inference_steps)
INFO [arm] poller started host=192.168.31.34 port=50051 hz=50.0
INFO Websocket connected
# ... warmup + benchmark output (omitted when --skip-warmup) ...
INFO FastWAM active-loop server ready at http://0.0.0.0:8118  (auto_dispatch=False)
```

Cold-start timing on `5090_1` (RTX 5090, GPU 1 idle):

- model load (Wan2.2-TI2V-5B + ActionDiT) ≈ **17 s** (one-off)
- rotation fingerprint test ≈ instant
- WS first-frame wait ≤ 5 s (when upstream healthy)
- warmup × 5 + benchmark × 10 ≈ 10 s
- total cold-start ≈ **30–40 s** end-to-end

## Troubleshooting

### `ws error: [Errno 111] Connection refused` → server aborts after 30 s

Upstream `rgbd_ws_bridge` is not running. Restart it on the bridge host and
re-launch the server. Confirm with:

```bash
curl -sf -o /dev/null -w "%{http_code}\n" \
  --connect-timeout 2 "http://192.168.31.66:19095" || echo "WS port closed"
# Or one-shot probe:
python scripts/fastwam_ws_probe.py --max-packets 5
```

The server's 30 s timeout is intentional: a silent stream would mean the
runner produces actions from stale frames. Do not raise
`--ws-startup-timeout-ms` as a workaround.

### Server starts but `fps_5s` < 14 Hz in `/ws_status`

Upstream `rgbd_ws_bridge` is degraded (observed several times during
integration). Inference cadence (2.5 Hz) is unaffected as long as frame age
< 250 ms, but the dispatcher's frame-age guard tightens. Either:

1. restart the bridge upstream, or
2. raise `--ws-frame-max-age-ms` temporarily (still bounded by the dispatcher's `chunk_max_stale_ms`).

### `ConnectionRefusedError` to `192.168.31.34:50051`

`arm_sdk` gRPC server is down on the controller box. Check the controller
power / process, then re-launch.

### `[Errno 111]` over a LAN IP even though host is reachable

Process-wide proxy is intercepting LAN traffic. Set:

```bash
export NO_PROXY=192.168.31.66,192.168.31.34,127.0.0.1,localhost
```

The server already injects these into `os.environ` at boot, but matters for
the cURL examples above when run from a different shell.

### Startup aborts: `gpu_free_mem_gb=X.X < 8.0`

GPU is occupied. On `5090_1`, GPU 0 is reserved for AirDC (typical free
RAM < 25 GiB); GPU 1 is normally idle. Default `--device=cuda:1`; lower
`--require-gpu-mem-free-gb` only after verifying the workload fits.

### Startup aborts: `benchmark p50=XXX ms > 500 ms budget`

Either the GPU is shared with another workload or the checkpoint loaded
into the wrong dtype. Re-check `nvidia-smi`, then re-launch. As a last
resort, drop `--num-inference-steps` (default 10) to 6 — design doc §3
shows mean drops to ~117 ms with quality margin still untested in the
wild.

### `lease_alive: false` after `POST /start`

`acquire_control` failed silently. The likely cause is another process
holding the lease (concurrent server, leftover record-and-replay session).
`set_arm_emergency_stop(false)` does NOT free the lease — kill the other
client. Verify with the airbot controller logs.

### text-embed cache miss on `/start {"instruction": "..."}`

Only the cached instruction works at runtime. Pre-compute:

```bash
python scripts/precompute_text_embeds.py \
  --instructions "open the door" "another phrase" \
  --output-dir data/text_embeds_cache/real_1048
```

Then relaunch the server. Switching instruction live is unsupported by
design (model loads a fixed text embed at boot).

### Where to find the design rationale for any decision above

`docs/fastwam_http_server_self_fetch_design.md` — every CLI default,
endpoint, error path, and rollback level is cross-referenced from §3
onward. This README is the operator manual; the design doc is the
engineering reference.
