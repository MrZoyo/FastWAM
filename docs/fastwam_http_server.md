# FastWAM HTTP Server

The server loads one FastWAM checkpoint and exposes action chunk inference over JSON HTTP.
It uses only the Python standard library for HTTP, so it does not require FastAPI or uvicorn.

Incoming RGB frames are routed by their actual `(H, W)`:

- **1088x1280** raw stereo frames: the payload **must** include an `undistort` object with
  `left_camera_info` and `right_camera_info`. The server undistorts each eye at native
  resolution, then resizes to `480x640` per camera with `cv2.INTER_AREA`. Requires OpenCV.
- **480x640** training-aligned frames: passed through as-is. Any `undistort` field in the
  payload is ignored. OpenCV is not required.
- Any other resolution returns HTTP 400. All camera streams in a single request must share
  the same resolution.

Once images reach `480x640`, the model-side pipeline (short-side resize to 480, center-crop
to `224x224`, normalize to `[-1, 1]`) runs unchanged.

## Prepare real_1048 Assets

The default service target is `real_1048_uncond_2cam224_1e-4`.

Required local paths:

- dataset: `/data_hdd/Lyle/Datasets/real_1048`
- checkpoint: `runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/checkpoints/weights/step_020000.pt`
- dataset stats: `runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/dataset_stats.json`
- text cache: `data/text_embeds_cache/real_1048`

Generate stats and text cache from the repo root if they are missing:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fastwam.utils.misc import register_work_dir

run_dir = Path("runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105").resolve()
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
  --checkpoint runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/checkpoints/weights/step_020000.pt \
  --dataset-stats runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/dataset_stats.json \
  --text-cache-dir data/text_embeds_cache/real_1048
```

`CUDA_VISIBLE_DEVICES=1` starts the service on physical GPU 1. Inside the process,
PyTorch still reports the selected visible GPU as `cuda:0`.

## Endpoints

- `GET /health`: model metadata, required camera keys, `proprio_dim`, horizon. Also reports
  `accepted_image_resolutions=[[1088,1280],[480,640]]`, `undistort_required_at=[1088,1280]`,
  and `resize_interpolation="cv2.INTER_AREA"`.
- `GET /schema`: compact request/response schema.
- `POST /infer`: run inference.
- `POST /predict_action`: alias of `/infer`.

## POST /infer Request

### Raw `1088x1280` stereo (requires `undistort` calibration)

```json
{
  "instruction": "open the door",
  "images": {
    "head_left": "<base64 encoded png/jpeg, 1088x1280>",
    "right_wrist_left": "<base64 encoded png/jpeg, 1088x1280>"
  },
  "proprio_raw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
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

### Already-aligned `480x640` frames

```json
{
  "instruction": "open the door",
  "images": {
    "head_left": "<base64 encoded png/jpeg, 480x640>",
    "right_wrist_left": "<base64 encoded png/jpeg, 480x640>"
  },
  "proprio_raw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04]
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
match the source resolution). Each eye is then resized to `480x640` with `cv2.INTER_AREA`
before entering the training pipeline. `left_to_right` is optional: when provided, OpenCV
performs stereo rectification; otherwise each eye is undistorted independently with its own
`K`/`d`. The legacy fields `undistort.enabled` and `undistort.output_size` are no longer
honored — they are silently ignored.

For the `480x640` path, the server skips all OpenCV work; any `undistort` payload is ignored.
This is the right choice if the client has already undistorted and resized upstream.

`instruction` must have a matching precomputed text embedding in `--text-cache-dir`.
If the cache is missing, the response is HTTP 500 with the missing cache path.

The following malformed requests return HTTP 400:

- non-7D `proprio_raw` or missing camera keys
- images at a resolution other than `1088x1280` or `480x640`
- cameras in the same request with mismatched resolutions
- `1088x1280` images without an `undistort` object, or with `left_camera_info` /
  `right_camera_info` missing
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
- Malformed requests return HTTP 400, including non-7D `proprio_raw` and missing camera keys.
