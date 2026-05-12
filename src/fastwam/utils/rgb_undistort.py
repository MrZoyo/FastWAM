"""RGB undistortion helpers used by inference-time services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency for the normal path
    cv2 = None


@dataclass(frozen=True)
class RGBUndistortMaps:
    image_size: tuple[int, int]
    output_size: tuple[int, int]
    map_x: np.ndarray
    map_y: np.ndarray
    rectification: np.ndarray
    projection: np.ndarray
    roi: tuple[int, int, int, int]
    stereo_rectified: bool
    eye: str | None = None


def _require_cv2() -> Any:
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for RGB undistortion")
    return cv2


def opencv_available() -> bool:
    return cv2 is not None


def camera_matrix_array(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3):
        raise ValueError(f"camera matrix must be shape (3, 3) or length 9, got {matrix.shape}")
    return np.ascontiguousarray(matrix)


def distortion_coefficients_array(value: Any) -> np.ndarray:
    coeffs = np.asarray([] if value is None else value, dtype=np.float64).reshape(-1, 1)
    return np.ascontiguousarray(coeffs)


def normalize_image_size(
    image_size: tuple[int, int] | list[int] | np.ndarray | None,
    *,
    rgb: np.ndarray | None = None,
) -> tuple[int, int]:
    if image_size is None:
        if rgb is None:
            raise ValueError("image_size is required when rgb is not provided")
        height, width = np.asarray(rgb).shape[:2]
        return int(width), int(height)
    values = tuple(int(v) for v in np.asarray(image_size).reshape(-1).tolist())
    if len(values) != 2:
        raise ValueError(f"image_size must contain width,height, got {image_size}")
    width, height = values
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive, got {(width, height)}")
    return width, height


def normalize_output_size(
    output_size: tuple[int, int] | list[int] | np.ndarray | str | None,
    *,
    image_size: tuple[int, int],
) -> tuple[int, int]:
    if output_size is None:
        return image_size
    if isinstance(output_size, str):
        if output_size.strip().lower() == "native":
            return image_size
        raise ValueError("output_size must be 'native' or [width, height]")
    values = tuple(int(v) for v in np.asarray(output_size).reshape(-1).tolist())
    if len(values) != 2:
        raise ValueError(f"output_size must contain width,height, got {output_size}")
    width, height = values
    if width <= 0 or height <= 0:
        raise ValueError(f"output_size must be positive, got {(width, height)}")
    return width, height


def scale_camera_matrix_for_resize(
    camera_matrix: Any,
    *,
    source_size: tuple[int, int] | list[int] | np.ndarray,
    target_size: tuple[int, int] | list[int] | np.ndarray,
) -> np.ndarray:
    """Scale pixel-space intrinsics K for a direct image resize.

    Distortion coefficients are not scaled because OpenCV applies them in
    normalized camera coordinates.
    """
    src_width, src_height = normalize_image_size(source_size)
    dst_width, dst_height = normalize_image_size(target_size)
    matrix = camera_matrix_array(camera_matrix).copy()
    if (src_width, src_height) == (dst_width, dst_height):
        return np.ascontiguousarray(matrix)
    matrix[0, :] *= dst_width / float(src_width)
    matrix[1, :] *= dst_height / float(src_height)
    return np.ascontiguousarray(matrix)


def left_to_right_from_rotation_translation(rotation: Any, translation: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def _parse_left_to_right(
    *,
    left_to_right: Any | None,
    rotation: Any | None,
    translation: Any | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if left_to_right is not None:
        transform = np.asarray(left_to_right, dtype=np.float64)
        if transform.shape == (16,):
            transform = transform.reshape(4, 4)
        if transform.shape != (4, 4):
            raise ValueError(f"left_to_right must be shape (4, 4) or length 16, got {transform.shape}")
        return (
            np.ascontiguousarray(transform),
            np.ascontiguousarray(transform[:3, :3]),
            np.ascontiguousarray(transform[:3, 3].reshape(3, 1)),
        )
    if rotation is None and translation is None:
        return None, None, None
    if rotation is None or translation is None:
        raise ValueError("rotation and translation must be provided together")
    transform = left_to_right_from_rotation_translation(rotation, translation)
    return (
        np.ascontiguousarray(transform),
        np.ascontiguousarray(transform[:3, :3]),
        np.ascontiguousarray(transform[:3, 3].reshape(3, 1)),
    )


def build_rgb_undistort_maps(
    *,
    camera_matrix: Any,
    distortion: Any,
    image_size: tuple[int, int] | list[int] | np.ndarray,
    output_size: tuple[int, int] | list[int] | np.ndarray | str | None = None,
    alpha: float = 0.0,
) -> RGBUndistortMaps:
    cv = _require_cv2()
    src_size = normalize_image_size(image_size)
    dst_size = normalize_output_size(output_size, image_size=src_size)
    k = camera_matrix_array(camera_matrix)
    d = distortion_coefficients_array(distortion)
    k_new, roi = cv.getOptimalNewCameraMatrix(k, d, src_size, float(alpha), dst_size)
    map_x, map_y = cv.initUndistortRectifyMap(
        k,
        d,
        np.eye(3, dtype=np.float64),
        k_new,
        dst_size,
        cv.CV_32FC1,
    )
    projection = np.zeros((3, 4), dtype=np.float64)
    projection[:3, :3] = k_new
    return RGBUndistortMaps(
        image_size=src_size,
        output_size=dst_size,
        map_x=np.asarray(map_x, dtype=np.float32),
        map_y=np.asarray(map_y, dtype=np.float32),
        rectification=np.eye(3, dtype=np.float32),
        projection=np.asarray(projection, dtype=np.float32),
        roi=tuple(map(int, roi)),
        stereo_rectified=False,
    )


def build_stereo_side_undistort_maps(
    *,
    left_camera_matrix: Any,
    left_distortion: Any,
    right_camera_matrix: Any,
    right_distortion: Any,
    image_size: tuple[int, int] | list[int] | np.ndarray,
    output_size: tuple[int, int] | list[int] | np.ndarray | str | None = None,
    left_to_right: Any | None = None,
    rotation: Any | None = None,
    translation: Any | None = None,
    alpha: float = 0.0,
    eye: str = "left",
) -> RGBUndistortMaps:
    cv = _require_cv2()
    selected_eye = str(eye).strip().lower()
    if selected_eye not in {"left", "right"}:
        raise ValueError("eye must be 'left' or 'right'")
    src_size = normalize_image_size(image_size)
    dst_size = normalize_output_size(output_size, image_size=src_size)
    k_left = camera_matrix_array(left_camera_matrix)
    d_left = distortion_coefficients_array(left_distortion)
    k_right = camera_matrix_array(right_camera_matrix)
    d_right = distortion_coefficients_array(right_distortion)
    _, stereo_rotation, stereo_translation = _parse_left_to_right(
        left_to_right=left_to_right,
        rotation=rotation,
        translation=translation,
    )
    if stereo_rotation is not None and stereo_translation is not None:
        r_left, r_right, p_left, p_right, _, left_roi, right_roi = cv.stereoRectify(
            k_left,
            d_left,
            k_right,
            d_right,
            src_size,
            stereo_rotation,
            stereo_translation,
            flags=cv.CALIB_ZERO_DISPARITY,
            alpha=float(alpha),
            newImageSize=dst_size,
        )
        stereo_rectified = True
    else:
        r_left = np.eye(3, dtype=np.float64)
        r_right = np.eye(3, dtype=np.float64)
        k_left_new, left_roi = cv.getOptimalNewCameraMatrix(k_left, d_left, src_size, float(alpha), dst_size)
        k_right_new, right_roi = cv.getOptimalNewCameraMatrix(k_right, d_right, src_size, float(alpha), dst_size)
        p_left = np.zeros((3, 4), dtype=np.float64)
        p_right = np.zeros((3, 4), dtype=np.float64)
        p_left[:3, :3] = k_left_new
        p_right[:3, :3] = k_right_new
        stereo_rectified = False
    if selected_eye == "left":
        k, d, r, p, roi = k_left, d_left, r_left, p_left, left_roi
    else:
        k, d, r, p, roi = k_right, d_right, r_right, p_right, right_roi
    map_x, map_y = cv.initUndistortRectifyMap(k, d, r, p[:3, :3], dst_size, cv.CV_32FC1)
    return RGBUndistortMaps(
        image_size=src_size,
        output_size=dst_size,
        map_x=np.asarray(map_x, dtype=np.float32),
        map_y=np.asarray(map_y, dtype=np.float32),
        rectification=np.asarray(r, dtype=np.float32),
        projection=np.asarray(p, dtype=np.float32),
        roi=tuple(map(int, roi)),
        stereo_rectified=stereo_rectified,
        eye=selected_eye,
    )


def apply_rgb_undistort_maps(rgb: np.ndarray, maps: RGBUndistortMaps) -> np.ndarray:
    cv = _require_cv2()
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {image.shape}")
    expected_width, expected_height = maps.image_size
    height, width = image.shape[:2]
    if (width, height) != (expected_width, expected_height):
        raise ValueError(f"rgb size {(width, height)} does not match map image_size {maps.image_size}")
    out = cv.remap(
        np.ascontiguousarray(image),
        maps.map_x,
        maps.map_y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
    )
    return np.ascontiguousarray(out, dtype=np.uint8)


def _camera_info_field(camera_info: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in camera_info:
            return camera_info[name]
    raise KeyError(f"missing camera info field, expected one of: {', '.join(names)}")


def _rgb_image_size(rgb: np.ndarray) -> tuple[int, int]:
    height, width = np.asarray(rgb).shape[:2]
    return int(width), int(height)


def _camera_info_size(camera_info: dict[str, Any], fallback_size: tuple[int, int]) -> tuple[int, int]:
    width = int(camera_info.get("width") or 0)
    height = int(camera_info.get("height") or 0)
    if width > 0 and height > 0:
        return width, height
    return fallback_size


def _camera_info_distortion(camera_info: dict[str, Any]) -> Any:
    return camera_info.get("d", camera_info.get("D", camera_info.get("distortion", [])))


def undistort_rgb_from_camera_info(
    *,
    rgb: np.ndarray,
    camera_info: dict[str, Any],
    output_size: tuple[int, int] | list[int] | np.ndarray | str | None = None,
    alpha: float = 0.0,
) -> np.ndarray:
    if not isinstance(camera_info, dict):
        raise TypeError("camera_info must be an object")
    image_size = _rgb_image_size(rgb)
    camera_info_size = _camera_info_size(camera_info, image_size)
    maps = build_rgb_undistort_maps(
        camera_matrix=scale_camera_matrix_for_resize(
            _camera_info_field(camera_info, "k", "K", "camera_matrix"),
            source_size=camera_info_size,
            target_size=image_size,
        ),
        distortion=_camera_info_distortion(camera_info),
        image_size=image_size,
        output_size=output_size,
        alpha=alpha,
    )
    return apply_rgb_undistort_maps(rgb, maps)


def undistort_stereo_side_from_camera_info(
    *,
    rgb: np.ndarray,
    left_camera_info: dict[str, Any],
    right_camera_info: dict[str, Any],
    output_size: tuple[int, int] | list[int] | np.ndarray | str | None = None,
    left_to_right: Any | None = None,
    rotation: Any | None = None,
    translation: Any | None = None,
    alpha: float = 0.0,
    eye: str = "left",
) -> np.ndarray:
    if not isinstance(left_camera_info, dict) or not isinstance(right_camera_info, dict):
        raise TypeError("left_camera_info and right_camera_info must be objects")
    image_size = _rgb_image_size(rgb)
    left_size = _camera_info_size(left_camera_info, image_size)
    right_size = _camera_info_size(right_camera_info, image_size)
    maps = build_stereo_side_undistort_maps(
        left_camera_matrix=scale_camera_matrix_for_resize(
            _camera_info_field(left_camera_info, "k", "K", "camera_matrix"),
            source_size=left_size,
            target_size=image_size,
        ),
        left_distortion=_camera_info_distortion(left_camera_info),
        right_camera_matrix=scale_camera_matrix_for_resize(
            _camera_info_field(right_camera_info, "k", "K", "camera_matrix"),
            source_size=right_size,
            target_size=image_size,
        ),
        right_distortion=_camera_info_distortion(right_camera_info),
        image_size=image_size,
        output_size=output_size,
        left_to_right=left_to_right,
        rotation=rotation,
        translation=translation,
        alpha=alpha,
        eye=eye,
    )
    return apply_rgb_undistort_maps(rgb, maps)
