"""Tests for `fastwam.server.image_pipeline.undistort_native` and the
backward-compatible refactor of `scripts/fastwam_http_server._normalize_image_resolution`.

The undistortion code needs a real K + D matrix to run, so the fixtures load
`configs/camera_info/real_1048_default.json` from the repo root. Image data is
synthetic noise -- we only assert shape/dtype invariants here, not pixel
values.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERA_INFO_PATH = REPO_ROOT / "configs" / "camera_info" / "real_1048_default.json"


# Make `scripts/fastwam_http_server.py` importable.
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="module")
def camera_info_payload() -> dict:
    with CAMERA_INFO_PATH.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def stereo_pair(camera_info_payload) -> dict:
    return camera_info_payload["stereo_pair"]


@pytest.fixture(scope="module")
def default_camera_info(camera_info_payload) -> dict:
    return camera_info_payload["cameras"]


@pytest.fixture
def native_images(stereo_pair):
    """Two 1088x1280x3 uint8 frames keyed by the stereo pair."""
    rng = np.random.default_rng(0)
    return {
        stereo_pair["left"]: rng.integers(0, 256, size=(1088, 1280, 3), dtype=np.uint8),
        stereo_pair["right"]: rng.integers(0, 256, size=(1088, 1280, 3), dtype=np.uint8),
    }


def _load_http_server_module():
    """Load scripts/fastwam_http_server.py as a module without invoking its CLI."""
    if "fastwam_http_server" in sys.modules:
        return sys.modules["fastwam_http_server"]
    spec = importlib.util.spec_from_file_location(
        "fastwam_http_server",
        REPO_ROOT / "scripts" / "fastwam_http_server.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["fastwam_http_server"] = module
    spec.loader.exec_module(module)
    return module


def test_undistort_native_preserves_shape(native_images, default_camera_info, stereo_pair):
    from fastwam.server.image_pipeline import undistort_native

    out = undistort_native(
        native_images,
        default_camera_info=default_camera_info,
        stereo_pair=stereo_pair,
    )

    assert set(out.keys()) == set(native_images.keys())
    for key, arr in out.items():
        assert arr.shape == (1088, 1280, 3), f"{key} shape={arr.shape}"
        assert arr.dtype == np.uint8


def test_undistort_native_missing_camera_info_raises(native_images, stereo_pair):
    from fastwam.server.image_pipeline import undistort_native

    with pytest.raises(ValueError, match="default_camera_info missing"):
        undistort_native(
            native_images,
            default_camera_info={},
            stereo_pair=stereo_pair,
        )


def test_undistort_native_missing_stereo_key_raises(default_camera_info, stereo_pair):
    from fastwam.server.image_pipeline import undistort_native

    rng = np.random.default_rng(0)
    only_left = {
        stereo_pair["left"]: rng.integers(0, 256, size=(1088, 1280, 3), dtype=np.uint8),
    }
    with pytest.raises(ValueError, match="must contain stereo keys"):
        undistort_native(
            only_left,
            default_camera_info=default_camera_info,
            stereo_pair=stereo_pair,
        )


def test_legacy_normalize_image_resolution_resizes_to_train_shape(
    native_images, default_camera_info, stereo_pair
):
    """Old server still returns 480x640 to keep /infer behaviour unchanged."""
    module = _load_http_server_module()

    # Inject server-side defaults so the legacy helper accepts the request
    # without an explicit `undistort` payload section.
    module._DEFAULT_CAMERA_INFO.clear()
    module._DEFAULT_CAMERA_INFO.update(default_camera_info)
    module._DEFAULT_STEREO_PAIR.clear()
    module._DEFAULT_STEREO_PAIR.update(stereo_pair)

    out = module._normalize_image_resolution(images=native_images, payload={})

    assert set(out.keys()) == set(native_images.keys())
    for key, arr in out.items():
        assert arr.shape == (480, 640, 3), f"{key} shape={arr.shape}"
        assert arr.dtype == np.uint8


def test_legacy_normalize_passes_through_train_shape(stereo_pair):
    module = _load_http_server_module()
    rng = np.random.default_rng(0)
    images = {
        stereo_pair["left"]: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
        stereo_pair["right"]: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
    }
    out = module._normalize_image_resolution(images=images, payload={})

    # Train-shape requests must pass through untouched -- same objects in, same
    # objects out (no copy, no resize).
    assert out is images
    for key, arr in out.items():
        assert arr.shape == (480, 640, 3)
