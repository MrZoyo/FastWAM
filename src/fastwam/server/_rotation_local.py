# TEMP: will be replaced by src/fastwam/server/rotation.py after PR1 merge
# Euler convention: scipy Rotation.from_quat(..).as_euler("xyz") extrinsic XYZ.
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _R


def unwrap_quat_sign(q_new, q_prev):
    q_new = np.asarray(q_new, dtype=np.float64).reshape(4)
    if q_prev is None:
        return q_new.astype(np.float32)
    q_prev = np.asarray(q_prev, dtype=np.float64).reshape(4)
    if float(np.dot(q_new, q_prev)) < 0.0:
        q_new = -q_new
    return q_new.astype(np.float32)


def quat_xyzw_to_rpy(quat_xyzw):
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return _R.from_quat(q).as_euler("xyz", degrees=False).astype(np.float32)


def rpy_to_quat_xyzw(rpy):
    r = np.asarray(rpy, dtype=np.float64).reshape(3)
    return _R.from_euler("xyz", r, degrees=False).as_quat().astype(np.float32)
