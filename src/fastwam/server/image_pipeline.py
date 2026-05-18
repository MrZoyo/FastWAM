"""Server-side image preprocessing helpers.

`undistort_native` is the shared stereo undistortion step extracted from
`scripts/fastwam_http_server.py:_normalize_image_resolution`. It applies the
same stereo undistortion at native (1088x1280) resolution that the old HTTP
server does, but does NOT do the cv2.resize to 480x640 -- callers that need
training-aligned 480x640 frames apply that resize themselves. New active-loop
code feeds the native-resolution output directly to `FastWAMModelClient.infer`,
which performs stitch+resize+crop+normalize internally.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from fastwam.utils.rgb_undistort import undistort_stereo_side_from_camera_info


def undistort_native(
    images: dict[str, np.ndarray],
    default_camera_info: dict[str, dict[str, Any]],
    stereo_pair: dict[str, str],
    alpha: float = 0.0,
    *,
    left_to_right: Any | None = None,
    rotation: Any | None = None,
    translation: Any | None = None,
) -> dict[str, np.ndarray]:
    """Undistort a stereo pair at native resolution.

    Args:
        images: {image_key: HxWx3 uint8 RGB/BGR}. Must contain both stereo keys.
        default_camera_info: {image_key: camera_info dict (with K + D)}.
        stereo_pair: {"left": <image_key>, "right": <image_key>}.
        alpha: OpenCV free scaling parameter (0.0 = crop, 1.0 = keep all pixels).
        left_to_right / rotation / translation: optional stereo extrinsics
            overrides forwarded to `undistort_stereo_side_from_camera_info`.
            When None (default), the underlying helper auto-derives them from
            the camera_info dicts.

    Returns:
        New dict mirroring `images` with the stereo entries replaced by
        undistorted frames of the SAME shape (native resolution preserved).
        Non-stereo keys are passed through unchanged.
    """
    left_key = stereo_pair["left"]
    right_key = stereo_pair["right"]
    if left_key not in images or right_key not in images:
        raise ValueError(
            f"images must contain stereo keys '{left_key}' and '{right_key}'; "
            f"got {sorted(images.keys())}."
        )
    left_ci = default_camera_info.get(left_key)
    right_ci = default_camera_info.get(right_key)
    if not isinstance(left_ci, dict):
        raise ValueError(f"default_camera_info missing dict for left key '{left_key}'.")
    if not isinstance(right_ci, dict):
        raise ValueError(f"default_camera_info missing dict for right key '{right_key}'.")

    kwargs = dict(
        left_camera_info=left_ci,
        right_camera_info=right_ci,
        output_size="native",
        left_to_right=left_to_right,
        rotation=rotation,
        translation=translation,
        alpha=float(alpha),
    )
    out = dict(images)
    out[left_key] = undistort_stereo_side_from_camera_info(
        rgb=images[left_key], eye="left", **kwargs
    )
    out[right_key] = undistort_stereo_side_from_camera_info(
        rgb=images[right_key], eye="right", **kwargs
    )
    return out
