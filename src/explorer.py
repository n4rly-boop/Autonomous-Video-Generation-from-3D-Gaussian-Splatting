"""Scene exploration routines based on NoField seeding heuristics."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from renderer import load_gaussians


@dataclass
class ExplorerConfig:
    """Configuration knobs for the exploration module."""

    ply_path: Path | None
    num_cameras: int = 12
    camera_height: float = 0.2
    is_scene: bool = False
    max_points: int = 120_000
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    @classmethod
    def from_metadata(cls, metadata: dict | None) -> "ExplorerConfig":
        if metadata is None:
            raise ValueError("scene_metadata must be provided.")

        path_value = metadata.get("ply_path") or metadata.get("path")
        ply_path = Path(path_value) if path_value is not None else None

        camera_count = metadata.get("num_cameras") or metadata.get("camera_count") or metadata.get("cameranum")
        num_cameras = int(camera_count) if camera_count is not None else 12
        num_cameras = max(1, num_cameras)

        height_value = (
            metadata.get("camera_height")
            or metadata.get("height")
            or metadata.get("radius")
            or 0.2
        )
        camera_height = float(height_value)

        scene_flag = metadata.get("scene_type")
        if scene_flag is None:
            scene_flag = metadata.get("scene_class")
        if scene_flag is None:
            scene_flag = metadata.get("isscene")
        is_scene = _bool_from_value(scene_flag)

        max_points_value = metadata.get("max_points")
        if max_points_value is None:
            max_points_value = metadata.get("point_budget", 120_000)
        max_points = int(max_points_value)

        seed = metadata.get("seed") or metadata.get("rng_seed")
        rng = np.random.default_rng(seed)

        return cls(
            ply_path=ply_path,
            num_cameras=num_cameras,
            camera_height=camera_height,
            is_scene=is_scene,
            max_points=max(1024, max_points),
            rng=rng,
        )


def plan_exploration_path(scene_metadata: dict) -> List[str]:
    """Return ordered camera commands that cover the scene."""
    config = ExplorerConfig.from_metadata(scene_metadata)

    points = _resolve_points(scene_metadata, config)
    if len(points) == 0:
        return []

    points = _downsample_points(points, config.max_points, config.rng)

    extent, p_low, p_high = _scene_statistics(points)
    bounds = _compute_bounds(p_low, p_high, extent, config.is_scene)
    effective_height = _effective_height(config.camera_height, extent, config.is_scene)

    normals = _estimate_normals(points, config.is_scene)

    anchor_indices = _farthest_point_sampling(points, config.num_cameras, config.rng)
    commands: List[str] = []
    for i, idx in enumerate(anchor_indices):
        surface_point = points[idx]
        surface_normal = normals[idx]
        camera_pos, look_at = _place_camera(surface_point, surface_normal, effective_height, bounds)
        up = _orthonormal_up_vector(look_at - camera_pos)
        commands.append(_format_camera_command(i, camera_pos, look_at, up))
    return commands


# ---------------------------------------------------------------------------
# Helpers (adapted from NoField seeding stage)
# ---------------------------------------------------------------------------

def _bool_from_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    str_value = str(value).strip().lower()
    return str_value in {"scene", "scenes", "room", "indoor", "true", "1", "yes"}


def _resolve_points(scene_metadata: dict, config: ExplorerConfig) -> np.ndarray:
    existing = scene_metadata.get("points")
    if existing is not None:
        points = np.asarray(existing, dtype=np.float64)
    else:
        if config.ply_path is None:
            raise ValueError("scene_metadata must contain either 'points' or 'ply_path'.")
        points, _, _, _ = load_gaussians(config.ply_path, max_points=config.max_points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be of shape (N, 3).")
    return np.nan_to_num(points, copy=False)


def _downsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[indices]


def _scene_statistics(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    p_low = np.percentile(points, 1, axis=0)
    p_high = np.percentile(points, 99, axis=0)
    extent = float(np.max(p_high - p_low))
    return extent, p_low, p_high


def _compute_bounds(p_low: np.ndarray, p_high: np.ndarray, extent: float, is_scene: bool, interior_scale: float = 0.7) -> Dict[str, np.ndarray]:
    if is_scene:
        center = (p_low + p_high) * 0.5
        extents = (p_high - p_low) * interior_scale
        safe_min = center - extents * 0.5
        safe_max = center + extents * 0.5
        print(f"[Explorer] Scene bounds tightened by {int(interior_scale * 100)}% for interior cameras.")
    else:
        margin = extent * 0.05
        safe_min = p_low - margin
        safe_max = p_high + margin
    return {"min": safe_min, "max": safe_max, "is_interior": is_scene}


def _effective_height(requested_height: float, extent: float, is_scene: bool) -> float:
    if extent <= 0:
        return requested_height
    if is_scene:
        cap = max(0.05 * extent, 1e-3)
        max_scene_height = 0.15 * extent
        effective_height = min(requested_height, max_scene_height)
        if requested_height > max_scene_height:
            print(
                f"[Explorer] Requested camera height {requested_height:.3f} is large for a scene; "
                f"using {effective_height:.3f}."
            )
        return max(effective_height, cap)
    if requested_height < 0.1 * extent:
        suggested = 0.5 * extent
        print(
            f"[Explorer] Camera height {requested_height:.3f} might be too small for object-scale exploration. "
            f"Suggested >= {suggested:.3f}."
        )
    return requested_height


def _estimate_normals(points: np.ndarray, is_scene: bool) -> np.ndarray:
    center = np.mean(points, axis=0)
    normals = points - center
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms + 1e-6)
    if is_scene:
        normals = -normals
    zero_normals = (np.linalg.norm(normals, axis=1) < 1e-6).nonzero()[0]
    normals[zero_normals] = np.array([0.0, 0.0, 1.0])
    return normals


def _farthest_point_sampling(points: np.ndarray, count: int, rng: np.random.Generator) -> Sequence[int]:
    if len(points) == 0:
        return []
    count = min(count, len(points))
    indices = np.empty(count, dtype=int)
    start_idx = int(rng.integers(0, len(points)))
    indices[0] = start_idx
    distances = np.linalg.norm(points - points[start_idx], axis=1)
    for i in range(1, count):
        next_idx = int(np.argmax(distances))
        indices[i] = next_idx
        new_dist = np.linalg.norm(points - points[next_idx], axis=1)
        distances = np.minimum(distances, new_dist)
    return indices


def _place_camera(surface_point: np.ndarray, normal: np.ndarray, height: float, bounds: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    normal = normal / (np.linalg.norm(normal) + 1e-6)
    camera_pos = surface_point + height * normal
    camera_pos = np.clip(camera_pos, bounds["min"], bounds["max"])
    return camera_pos, surface_point


def _orthonormal_up_vector(forward: np.ndarray) -> np.ndarray:
    forward = forward / (np.linalg.norm(forward) + 1e-6)
    up_hint = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, up_hint)) > 0.95:
        up_hint = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up_hint)
    right = right / (np.linalg.norm(right) + 1e-6)
    up = np.cross(right, forward)
    return up / (np.linalg.norm(up) + 1e-6)


def _format_camera_command(index: int, position: np.ndarray, look_at: np.ndarray, up: np.ndarray) -> str:
    return (
        f"camera_{index:02d}: pos=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) "
        f"look_at=({look_at[0]:.4f}, {look_at[1]:.4f}, {look_at[2]:.4f}) "
        f"up=({up[0]:.4f}, {up[1]:.4f}, {up[2]:.4f})"
    )
