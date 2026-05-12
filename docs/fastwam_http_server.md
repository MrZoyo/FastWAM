# FastWAM HTTP Server

The server loads one FastWAM checkpoint and exposes action chunk inference over JSON HTTP.
It uses only the Python standard library for HTTP, so it does not require FastAPI or uvicorn.
Optional stereo undistortion uses OpenCV if you send `undistort` calibration in the request.

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
.venv/bin/python -B scripts/fastwam_http_server.py \
  --host 0.0.0.0 \
  --port 8117 \
  --config configs/task/real_1048_uncond_2cam224_1e-4.yaml \
  --checkpoint runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/checkpoints/weights/step_020000.pt \
  --dataset-stats runs/real_1048_uncond_2cam224_1e-4/real1048_20k_wandb_20260508_202105/dataset_stats.json \
  --text-cache-dir data/text_embeds_cache/real_1048
```

## Endpoints

- `GET /health`: model metadata, required camera keys, `proprio_dim`, horizon.
- `GET /schema`: compact request/response schema.
- `POST /infer`: run inference.
- `POST /predict_action`: alias of `/infer`.

## POST /infer Request

```json
{
  "instruction": "open the door",
  "images": {
    "head_left": "<base64 encoded png/jpeg>",
    "right_wrist_left": "<base64 encoded png/jpeg>"
  },
  "proprio_raw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04],
  "undistort": {
    "enabled": true,
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
    "output_size": [224, 224],
    "alpha": 0.0
  }
}
```

`images` keys must match the model config. `real_1048` expects `head_left` and `right_wrist_left`.
`proprio_raw` is the 7D real_1048 state: six arm joint values followed by gripper position.
The server accumulates the predicted first six delta dimensions onto `proprio_raw[:6]` by default.
You may pass `current_position`, `joint_position`, or `cartesian_position` as a 6D override if needed.

If `undistort` is present, the server treats the two model images as a stereo pair, scales
`CameraInfo.K` from the calibration resolution to the received image resolution, applies
OpenCV stereo undistortion/rectification, and then resizes the result to the model-aligned
output size. If `output_size` is omitted, it defaults to the model image size, currently
`[224, 224]` per camera.

`instruction` must have a matching precomputed text embedding in `--text-cache-dir`.
If the cache is missing, the response is HTTP 500 with the missing cache path.
Malformed requests, such as missing camera keys or a non-7D `proprio_raw`, return HTTP 400.
If undistortion is requested but calibration is incomplete, the response is HTTP 400.

## Response

```json
{
  "action_format": "joint_absolute",
  "actions": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
  "source": "fastwam",
  "action_mode": "delta6_abs_gripper",
  "normalized_action_shape": [32, 7],
  "full_prompt": "...",
  "instruction": "..."
}
```

`actions` is an `[action_horizon, 7]` list. The first six dimensions are absolute joint targets,
computed as predicted delta plus the current six joint values; the last dimension is the predicted gripper target.

## Verified On 5090

- `GET /health` reports `image_keys=["head_left","right_wrist_left"]`, `proprio_dim=7`, and checkpoint `step_020000.pt`.
- `GET /health` also reports `opencv_available=true` on the 5090 runtime.
- A real dataset frame from `/data_hdd/Lyle/Datasets/real_1048` returns HTTP 200 with `action_format="joint_absolute"` and `actions` shape `[32,7]`.
- Malformed requests return HTTP 400, including non-7D `proprio_raw` and missing camera keys.
