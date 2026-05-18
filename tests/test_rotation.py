"""Unit tests for ``fastwam.server.rotation``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastwam.server.rotation import (
    quat_canonicalize,
    quat_xyzw_to_rpy,
    rpy_to_quat_xyzw,
    unwrap_quat_sign,
    unwrap_rpy_sequence,
)


def _canonical(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return -q if q[3] < 0.0 else q


def test_dtypes_and_shapes():
    rpy = quat_xyzw_to_rpy(np.array([0.0, 0.0, 0.0, 1.0]))
    assert rpy.shape == (3,) and rpy.dtype == np.float32
    q = rpy_to_quat_xyzw(np.array([0.1, 0.2, 0.3]))
    assert q.shape == (4,) and q.dtype == np.float32


def test_roundtrip_identity():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    rt = rpy_to_quat_xyzw(quat_xyzw_to_rpy(q))
    np.testing.assert_allclose(_canonical(rt), _canonical(q), atol=1e-6)


def test_roundtrip_random_1000():
    rng = np.random.default_rng(20260518)
    quats = Rotation.random(1000, random_state=rng).as_quat()
    worst = 0.0
    for q in quats:
        rt = rpy_to_quat_xyzw(quat_xyzw_to_rpy(q))
        worst = max(worst, float(np.max(np.abs(_canonical(rt) - _canonical(q)))))
    assert worst < 1e-6, worst


@pytest.mark.parametrize("pitch_deg", [-90.1, -90.0, -89.9, 89.9, 90.0, 90.1])
def test_gimbal_lock_rotation_matrix_preserved(pitch_deg: float):
    rpy = np.array([0.3, np.deg2rad(pitch_deg), -0.4])
    q = rpy_to_quat_xyzw(rpy)
    rpy_back = quat_xyzw_to_rpy(q)
    q_back = rpy_to_quat_xyzw(rpy_back)
    R1 = Rotation.from_quat(q).as_matrix()
    R2 = Rotation.from_quat(q_back).as_matrix()
    np.testing.assert_allclose(R1, R2, atol=1e-6)


def test_unwrap_rpy_sequence_removes_2pi_jump():
    seq = np.array([
        [0.0, 0.0, np.pi - 0.05],
        [0.0, 0.0, np.pi + 0.05],   # raw would jump to -π+0.05
        [0.0, 0.0, np.pi + 0.10],
    ])
    raw = seq.copy()
    raw[1, 2] = -np.pi + 0.05
    raw[2, 2] = -np.pi + 0.10
    out = unwrap_rpy_sequence(raw)
    diffs = np.diff(out, axis=0)
    assert np.all(np.abs(diffs) < np.pi)


def test_quat_canonicalize_no_ref_forces_w_positive():
    q = np.array([0.1, 0.2, 0.3, -0.9])
    out = quat_canonicalize(q, None)
    assert out[3] >= 0.0
    np.testing.assert_allclose(out, -q.astype(np.float32), atol=0.0)


def test_unwrap_quat_sign_flips_back():
    q = Rotation.from_euler("xyz", [0.2, -0.3, 1.1]).as_quat()
    q_prev = q.copy()
    q_new_flipped = -q
    fixed = unwrap_quat_sign(q_new_flipped, q_prev)
    np.testing.assert_allclose(fixed, q.astype(np.float32), atol=1e-7)


def test_unwrap_quat_sign_keeps_aligned_input():
    q_prev = Rotation.from_euler("xyz", [0.2, -0.3, 1.1]).as_quat()
    q_new = Rotation.from_euler("xyz", [0.21, -0.29, 1.11]).as_quat()
    assert float(np.dot(q_new, q_prev)) > 0.0
    fixed = unwrap_quat_sign(q_new, q_prev)
    np.testing.assert_allclose(fixed, q_new.astype(np.float32), atol=1e-7)
