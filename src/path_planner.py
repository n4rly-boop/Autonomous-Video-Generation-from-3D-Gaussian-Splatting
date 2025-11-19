"""Path planning utilities for smooth camera trajectories."""
from typing import Iterable, Sequence, List, Dict, Any
import numpy as np
from pathlib import Path
from plyfile import PlyData

def optimize_path(waypoints: Iterable[str]) -> Sequence[str]:
    """Refine coarse waypoints into an executable trajectory."""
    return list(waypoints)

def load_ply_points(ply_path: Path, max_points: int = 100000) -> np.ndarray:
    try:
        ply = PlyData.read(str(ply_path))
        vertex = ply['vertex']
        props = [p.name for p in vertex.properties]
        if 'x' in props and 'y' in props and 'z' in props:
            pts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
        else:
            return np.array([])
            
        # Filter NaNs and Infs
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        
        # Clamp extreme values to avoid covariance overflow
        # A typical scene is within +/- 10000 units
        pts = np.clip(pts, -1e5, 1e5)
        
        if len(pts) > max_points:
             pts = pts[::len(pts)//max_points]
        return pts
    except Exception:
        return np.array([])

def generate_interior_path(ply_path: Path, duration: float = 60.0, fps: int = 30) -> Dict[str, Any]:
    """
    Generates a figure-8 trajectory inside the scene boundaries.
    """
    points = load_ply_points(ply_path)
    if len(points) < 10:
        return {}
    
    # Filter extreme outliers (z-score or percentile)
    # Use 1st-99th percentile clipping to remove fly-away splats
    try:
        p_low, p_high = np.percentile(points, [1, 99], axis=0)
    except Exception:
        return {}

    mask = np.all((points >= p_low) & (points <= p_high), axis=1)
    points = points[mask]
    
    if len(points) < 10:
        return {}

    # PCA for orientation
    center = np.mean(points, axis=0)
    shifted = points - center
    
    # Double check finite
    if not np.isfinite(shifted).all():
        return {}

    try:
        cov = np.cov(shifted, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
    except Exception:
        # Fallback to identity if PCA fails
        vals = np.array([1, 1, 1])
        vecs = np.eye(3)
        
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    
    # Sanitize vectors
    vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Major axes
    axis_1 = vecs[:, 0] # Longest
    axis_2 = vecs[:, 1] # Second longest
    
    # Normalize axes just in case
    norm1 = np.linalg.norm(axis_1)
    if norm1 > 1e-6: axis_1 /= norm1
    else: axis_1 = np.array([1.0, 0.0, 0.0])
        
    norm2 = np.linalg.norm(axis_2)
    if norm2 > 1e-6: axis_2 /= norm2
    else: axis_2 = np.array([0.0, 0.0, 1.0])
    
    # Filter floor height
    y_coords = points[:, 1]
    floor_y = np.percentile(y_coords, 5)
    ceiling_y = np.percentile(y_coords, 95)
    scene_height = ceiling_y - floor_y
    
    # Eye level
    camera_y = floor_y + max(scene_height * 0.35, 1.0)
    
    # Project onto major plane (Axis 1 & 2) to find extent
    proj_1 = np.dot(shifted, axis_1)
    proj_2 = np.dot(shifted, axis_2)
    
    # Handle empty projections
    if len(proj_1) == 0: return {}
    
    p1_min, p1_max = np.percentile(proj_1, [20, 80])
    p2_min, p2_max = np.percentile(proj_2, [20, 80])
    
    scale_1 = (p1_max - p1_min) * 0.45
    scale_2 = (p2_max - p2_min) * 0.45
    
    # Sanity check scales
    if scale_1 <= 0.1 or not np.isfinite(scale_1): scale_1 = 1.0
    if scale_2 <= 0.1 or not np.isfinite(scale_2): scale_2 = 1.0
    
    frames = []
    total_frames = int(duration * fps)
    
    for i in range(total_frames):
        t = (i / total_frames) * 2 * np.pi
        
        u = np.sin(t)
        v = np.sin(2 * t) * 0.5
        
        pos = center + (axis_1 * u * scale_1) + (axis_2 * v * scale_2)
        pos[1] = camera_y
        
        du = np.cos(t)
        dv = np.cos(2 * t)
        tangent = (axis_1 * du * scale_1) + (axis_2 * dv * scale_2)
        
        look_at = pos + tangent
        look_at[1] = camera_y
        
        frames.append({
            "t": i / fps,
            "position": pos.tolist(),
            "look_at": look_at.tolist()
        })
        
    return {
        "fps": fps,
        "duration_sec": duration,
        "frames": frames
    }
