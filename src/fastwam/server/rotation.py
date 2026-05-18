"""Quaternion / Euler conversions for FastWAM active loop server.

Euler convention: scipy ``Rotation.from_quat(..).as_euler("xyz", degrees=False)``,
i.e. extrinsic XYZ (roll-pitch-yaw in fixed world axes). This MUST match the
training-side convention applied in
``mcap_preprocess_pipeline/scripts/step01_extract_mcap_rgb_and_params.py:1117``.
If the upstream convention changes, ``tests/test_rotation_fingerprint.py``
(driven by ``tests/fixtures/rotation_fingerprint.json``) will fail.

Quaternion order is xyzw throughout (scipy default). gripper / position are
handled elsewhere; this module is rotation-only.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_xyzw_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert a single xyzw quaternion to extrinsic XYZ rpy (radians, float32)."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    rpy = Rotation.from_quat(q).as_euler("xyz", degrees=False)
    return rpy.astype(np.float32)


def rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    """Convert a single extrinsic XYZ rpy (radians) to xyzw quaternion (float32)."""
    r = np.asarray(rpy, dtype=np.float64).reshape(3)
    q = Rotation.from_euler("xyz", r, degrees=False).as_quat()
    return q.astype(np.float32)


def unwrap_rpy_sequence(rpy_seq: np.ndarray) -> np.ndarray:
    """Remove 2π discontinuities along axis 0 for a (N,3) rpy sequence."""
    arr = np.asarray(rpy_seq, dtype=np.float64).reshape(-1, 3)
    out = np.unwrap(arr, axis=0)
    return out.astype(np.float32)


def quat_canonicalize(q_new: np.ndarray, q_ref: np.ndarray | None) -> np.ndarray:
    """Return q_new with sign aligned to q_ref (dot >= 0). If q_ref is None,
    force w >= 0. Output is float32 xyzw."""
    q = np.asarray(q_new, dtype=np.float64).reshape(4)
    if q_ref is None:
        if q[3] < 0.0:
            q = -q
    else:
        r = np.asarray(q_ref, dtype=np.float64).reshape(4)
        if float(np.dot(q, r)) < 0.0:
            q = -q
    return q.astype(np.float32)


def unwrap_quat_sign(q_new: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
    """Alias of ``quat_canonicalize(q_new, q_prev)``; flips q_new sign when
    consecutive frames return q vs -q for the same orientation."""
    return quat_canonicalize(q_new, q_prev)
