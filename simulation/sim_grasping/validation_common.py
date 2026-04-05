from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stretch_mujoco.enums.stretch_cameras import StretchCameras


ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT_DIR / "logs" / "sim_validation"
DATA_ROOT = LOG_ROOT / "data_samples"
MOTION_ROOT = LOG_ROOT / "motion"
SCREENSHOT_ROOT = LOG_ROOT / "screenshots"
NOTES_ROOT = ROOT_DIR / "notes"

DEFAULT_SCENE_PATH = ROOT_DIR / "stretch_mujoco" / "stretch_mujoco" / "models" / "scene.xml"
TABLETOP_SCENE_PATH = ROOT_DIR / "sim_grasping" / "tabletop_minimal_scene.xml"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_validation_dirs() -> dict[str, Path]:
    return {
        "log_root": ensure_dir(LOG_ROOT),
        "data_root": ensure_dir(DATA_ROOT),
        "motion_root": ensure_dir(MOTION_ROOT),
        "screenshot_root": ensure_dir(SCREENSHOT_ROOT),
        "notes_root": ensure_dir(NOTES_ROOT),
    }


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_serializable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Enum):
        return value.name
    return value


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(to_serializable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def rgb_to_bgr(rgb_image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def save_rgb_image(path: Path, rgb_image: np.ndarray) -> None:
    ensure_dir(path.parent)
    cv2.imwrite(str(path), rgb_to_bgr(rgb_image))


def depth_to_color(depth_meters: np.ndarray) -> np.ndarray:
    valid_mask = depth_meters > 0
    color = np.zeros((*depth_meters.shape, 3), dtype=np.uint8)
    if not np.any(valid_mask):
        return color

    valid_depth = depth_meters[valid_mask]
    depth_min = float(valid_depth.min())
    depth_max = float(valid_depth.max())
    if np.isclose(depth_min, depth_max):
        normalized = np.zeros_like(depth_meters, dtype=np.uint8)
        normalized[valid_mask] = 255
    else:
        normalized = np.zeros_like(depth_meters, dtype=np.float32)
        normalized[valid_mask] = (depth_meters[valid_mask] - depth_min) / (depth_max - depth_min)
        normalized = (normalized * 255).clip(0, 255).astype(np.uint8)

    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid_mask] = 0
    return color


def save_depth_outputs(prefix: Path, depth_meters: np.ndarray) -> dict[str, Any]:
    ensure_dir(prefix.parent)
    np.save(str(prefix.with_suffix(".npy")), depth_meters)
    colorized = depth_to_color(depth_meters)
    cv2.imwrite(str(prefix.with_name(prefix.name + "_color.png")), colorized)

    valid_mask = depth_meters > 0
    stats = {
        "shape": list(depth_meters.shape),
        "dtype": str(depth_meters.dtype),
        "valid_pixel_count": int(valid_mask.sum()),
    }
    if np.any(valid_mask):
        stats.update(
            {
                "min_m": float(depth_meters[valid_mask].min()),
                "max_m": float(depth_meters[valid_mask].max()),
                "mean_m": float(depth_meters[valid_mask].mean()),
            }
        )
    else:
        stats.update({"min_m": None, "max_m": None, "mean_m": None})
    return stats


def matrix_to_rows(matrix: np.ndarray | None) -> list[list[float]] | None:
    if matrix is None:
        return None
    return np.asarray(matrix, dtype=float).tolist()


def summarize_array(array: np.ndarray | None) -> dict[str, Any] | None:
    if array is None:
        return None
    payload: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size:
        payload["min"] = float(np.min(array))
        payload["max"] = float(np.max(array))
    return payload


def rotate_intrinsics_clockwise(k_matrix: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
    rotated = np.eye(3, dtype=float)
    rotated[0, 0] = k_matrix[1, 1]
    rotated[1, 1] = k_matrix[0, 0]
    rotated[0, 2] = original_height - 1 - k_matrix[1, 2]
    rotated[1, 2] = k_matrix[0, 2]
    return rotated


def aligned_intrinsics_for_saved_image(camera: StretchCameras) -> np.ndarray | None:
    settings = camera.initial_camera_settings
    if camera == StretchCameras.cam_nav_rgb:
        return None

    base_k = np.array(
        [
            [settings.focal[0], 0.0, settings.width / 2.0],
            [0.0, settings.focal[1], settings.height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    if camera in [StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth]:
        return rotate_intrinsics_clockwise(base_k, settings.width, settings.height)

    return base_k
