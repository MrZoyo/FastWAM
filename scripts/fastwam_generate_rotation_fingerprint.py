#!/usr/bin/env python3
"""Generate ``tests/fixtures/rotation_fingerprint.json`` — 5 (quat_xyzw, expected_rpy)
ground-truth samples used by ``tests/test_rotation_fingerprint.py`` to pin the
extrinsic-XYZ scipy Euler convention.

Data source: the FastWAM training dataset
``/DATA/disk1/datasets_lerobot/opendoor_real_1048`` (h200-1) only exposes joint
angles and **delta** cartesian actions — no absolute quaternion field — so we
cannot lift quaternions straight out of the parquet. Instead, this script
**seeds** the fixture with quaternions covering the operating envelope
implied by §4.2 ARM probe in
``docs/fastwam_http_server_self_fetch_design.md``:

  - sample 0: ARM zero pose (~0,0,0,1)  ← exact probe value
  - samples 1-4: deterministic rotations spanning roll/pitch/yaw

For each sample, ``expected_rpy`` is recomputed with
``Rotation.from_quat(..).as_euler("xyz")`` — the same call the upstream training
pipeline uses (see ``step01_extract_mcap_rgb_and_params.py:1117``). The fixture
therefore pins our module to that exact scipy call: any deviation (e.g.
switching to intrinsic) flips the test red.

Re-running this script regenerates the JSON in-place. Commit the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


# (label, quat_xyzw) — quat_xyzw deliberately non-unit on input where possible
# to also exercise scipy's normalization, but kept unit here to make the
# expected_rpy reproducible bit-for-bit.
SEED_QUATS_XYZW: list[tuple[str, list[float]]] = [
    # ARM probe §4.2 starting pose (close to identity rotation).
    ("arm_zero_pose", [0.0, 0.0, 0.0, 1.0]),
    # Pure roll +30 deg.
    ("roll_+30deg", Rotation.from_euler("xyz", [np.deg2rad(30.0), 0.0, 0.0]).as_quat().tolist()),
    # Pure pitch -45 deg.
    ("pitch_-45deg", Rotation.from_euler("xyz", [0.0, np.deg2rad(-45.0), 0.0]).as_quat().tolist()),
    # Pure yaw +120 deg (past ±π/2 boundary).
    ("yaw_+120deg", Rotation.from_euler("xyz", [0.0, 0.0, np.deg2rad(120.0)]).as_quat().tolist()),
    # Mixed rpy = (20, 35, -75) deg — covers cross-axis term.
    (
        "mixed_20_35_-75deg",
        Rotation.from_euler(
            "xyz",
            [np.deg2rad(20.0), np.deg2rad(35.0), np.deg2rad(-75.0)],
        ).as_quat().tolist(),
    ),
]


def build_samples() -> list[dict]:
    samples = []
    for label, quat in SEED_QUATS_XYZW:
        q = np.asarray(quat, dtype=np.float64)
        rpy = Rotation.from_quat(q).as_euler("xyz", degrees=False)
        samples.append({
            "label": label,
            "quat_xyzw": q.tolist(),
            "expected_rpy": rpy.tolist(),
        })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "rotation_fingerprint.json",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "convention": "scipy Rotation.from_quat(..).as_euler('xyz', degrees=False)",
        "quat_order": "xyzw",
        "samples": build_samples(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(payload['samples'])} samples -> {args.output}")


if __name__ == "__main__":
    main()
