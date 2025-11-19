"""Path planning utilities for smooth camera trajectories."""
from pathlib import Path
from typing import Iterable, Sequence, Optional, List, Tuple
import numpy as np
from renderer import load_gaussians


def optimize_path(
    waypoints: Optional[Iterable[str]] = None,
    ply_path: Optional[Path] = None,
    num_waypoints: int = 16,
    max_points: int = 120_000,
    is_scene: bool = False,
    camera_height: float = 0.2,
    smooth_trajectory: bool = True,
) -> Sequence[str]:
    """
    Refine coarse waypoints into an executable trajectory.

    If ply_path is provided, generates waypoints from the PLY file's point cloud.
    Otherwise, optimizes the provided waypoints.

    Args:
        waypoints: Existing camera command strings (ignored if ply_path is set).
        ply_path: Path to PLY file for generating waypoints from point cloud.
        num_waypoints: Number of waypoints to generate if using PLY.
        max_points: Max points to sample from PLY for analysis.
        is_scene: Whether PLY represents a scene (affects normal estimation).
        camera_height: Camera height above surface for waypoint placement.
        smooth_trajectory: Whether to apply trajectory smoothing.

    Returns:
        Sequence of optimized camera command strings.
    """
    if ply_path is not None:
        waypoints = generate_waypoints_from_ply(
            ply_path, num_waypoints, max_points, is_scene, camera_height
        )

    if not waypoints:
        return []

    # For now, just return the waypoints; TODO: implement full trajectory optimization
    optimized = list(waypoints)

    if smooth_trajectory:
        optimized = _smooth_trajectory(optimized)

    return optimized


def generate_waypoints_from_ply(
    ply_path: Path,
    num_waypoints: int = 16,
    max_points: int = 120_000,
    is_scene: bool = False,
    camera_height: float = 0.2,
) -> List[str]:
    """
    Generate camera waypoints by analyzing the point cloud in the PLY file.
    Uses farthest point sampling, normal estimation, and surface-based placement.
    """
    # Load point cloud
    points, _, _, _ = load_gaussians(ply_path, max_points=max_points)
    if len(points) == 0:
        return []

    # Downsample if necessary
    if len(points) > max_points:
        indices = np.random.choice(len(points), size=max_points, replace=False)
        points = points[indices]

    # Estimate normals (simple radial from center)
    normals = _estimate_normals(points, is_scene)

    # Compute scene statistics
    extent, p_low, p_high = _scene_statistics(points)

    # Compute bounds
    bounds = _compute_bounds(p_low, p_high, extent, is_scene)

    # Effective camera height
    effective_height = _effective_height(camera_height, extent, is_scene)

    # Select anchor points using farthest point sampling
    anchor_indices = _farthest_point_sampling(points, num_waypoints)

    # Generate camera commands
    commands = []
    for i, idx in enumerate(anchor_indices):
        surface_point = points[idx]
        surface_normal = normals[idx]
        camera_pos, look_at = _place_camera(surface_point, surface_normal, effective_height, bounds)
        up = _orthonormal_up_vector(look_at - camera_pos)
        commands.append(_format_camera_command(i, camera_pos, look_at, up))

    return commands


def _estimate_normals(points: np.ndarray, is_scene: bool) -> np.ndarray:
    """Estimate surface normals (radial from center for simplicity)."""
    center = np.mean(points, axis=0)
    normals = points - center
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norms + 1e-6)
    if is_scene:
        normals = -normals  # Flip for interior scenes
    zero_normals = (np.linalg.norm(normals, axis=1) < 1e-6).nonzero()[0]
    normals[zero_normals] = np.array([0.0, 0.0, 1.0])
    return normals


def _scene_statistics(points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute extent and percentiles."""
    p_low = np.percentile(points, 1, axis=0)
    p_high = np.percentile(points, 99, axis=0)
    extent = float(np.max(p_high - p_low))
    return extent, p_low, p_high


def _compute_bounds(p_low: np.ndarray, p_high: np.ndarray, extent: float, is_scene: bool, interior_scale: float = 0.7) -> dict:
    """Compute safe bounds for camera placement."""
    if is_scene:
        center = (p_low + p_high) * 0.5
        extents = (p_high - p_low) * interior_scale
        safe_min = center - extents * 0.5
        safe_max = center + extents * 0.5
    else:
        margin = extent * 0.05
        safe_min = p_low - margin
        safe_max = p_high + margin
    return {"min": safe_min, "max": safe_max}


def _effective_height(requested_height: float, extent: float, is_scene: bool) -> float:
    """Adjust camera height based on scene scale."""
    if extent <= 0:
        return requested_height
    if is_scene:
        cap = max(0.05 * extent, 1e-3)
        max_scene_height = 0.15 * extent
        effective_height = min(requested_height, max_scene_height)
        return max(effective_height, cap)
    return requested_height


def _farthest_point_sampling(points: np.ndarray, count: int) -> Sequence[int]:
    """Select points using farthest point sampling for coverage."""
    if len(points) == 0:
        return []
    count = min(count, len(points))
    indices = np.empty(count, dtype=int)
    start_idx = np.random.randint(0, len(points))
    indices[0] = start_idx
    distances = np.linalg.norm(points - points[start_idx], axis=1)
    for i in range(1, count):
        next_idx = int(np.argmax(distances))
        indices[i] = next_idx
        new_dist = np.linalg.norm(points - points[next_idx], axis=1)
        distances = np.minimum(distances, new_dist)
    return indices


def _place_camera(surface_point: np.ndarray, normal: np.ndarray, height: float, bounds: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Place camera above surface point along normal."""
    normal = normal / (np.linalg.norm(normal) + 1e-6)
    camera_pos = surface_point + height * normal
    camera_pos = np.clip(camera_pos, bounds["min"], bounds["max"])
    return camera_pos, surface_point


def _orthonormal_up_vector(forward: np.ndarray) -> np.ndarray:
    """Compute an orthonormal up vector."""
    forward = forward / (np.linalg.norm(forward) + 1e-6)
    up_hint = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, up_hint)) > 0.95:
        up_hint = np.array([1.0, 0.0, 0.0])
    right = np.cross(forward, up_hint)
    right = right / (np.linalg.norm(right) + 1e-6)
    up = np.cross(right, forward)
    return up / (np.linalg.norm(up) + 1e-6)


def _format_camera_command(index: int, position: np.ndarray, look_at: np.ndarray, up: np.ndarray) -> str:
    """Format as camera command string."""
    return (
        f"camera_{index:02d}: pos=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}) "
        f"look_at=({look_at[0]:.4f}, {look_at[1]:.4f}, {look_at[2]:.4f}) "
        f"up=({up[0]:.4f}, {up[1]:.4f}, {up[2]:.4f})"
    )


def _smooth_trajectory(waypoints: List[str]) -> List[str]:
    """
    Placeholder for trajectory smoothing (e.g., spline fitting).
    Currently returns waypoints unchanged.
    """
    # TODO: Implement spline smoothing or path optimization
    return waypoints
