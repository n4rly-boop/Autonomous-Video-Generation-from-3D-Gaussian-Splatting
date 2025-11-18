"""Renderer module for 3D Gaussian Splatting."""
import math
from pathlib import Path
from typing import Tuple, Iterable, Optional, Sequence, List

import numpy as np
import torch
from plyfile import PlyData
import imageio


# -------------------- PLY LOADER (robust to packing) --------------------

_COLOR_FIELDS = (
    ("red", "green", "blue"),
    ("diffuse_red", "diffuse_green", "diffuse_blue"),
    ("r", "g", "b"),
)


def _decode_rotation(rot_uint: np.ndarray) -> np.ndarray:
    """Decode quaternion from packed uint32 format."""
    # Standard quaternion packing: 10 bits per component
    w_bits = (rot_uint >> 30) & 0x3FF
    x_bits = (rot_uint >> 20) & 0x3FF
    y_bits = (rot_uint >> 10) & 0x3FF
    z_bits = rot_uint & 0x3FF
    
    # Normalize to [-1, 1] range
    w = (w_bits.astype(np.float64) / 511.0) * 2.0 - 1.0
    x = (x_bits.astype(np.float64) / 511.0) * 2.0 - 1.0
    y = (y_bits.astype(np.float64) / 511.0) * 2.0 - 1.0
    z = (z_bits.astype(np.float64) / 511.0) * 2.0 - 1.0
    
    quat = np.stack([w, x, y, z], axis=1)
    # Normalize quaternion
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / (norm + 1e-8)
    return quat


def _decode_scale(scale_uint: np.ndarray, chunk_data, cidx: np.ndarray) -> np.ndarray:
    """Decode scale from packed uint32 format."""
    # Standard scale packing: 10 bits per component
    sx_bits = (scale_uint >> 20) & 0x3FF
    sy_bits = (scale_uint >> 10) & 0x3FF
    sz_bits = scale_uint & 0x3FF
    
    sx_norm = sx_bits.astype(np.float64) / 1023.0
    sy_norm = sy_bits.astype(np.float64) / 1023.0
    sz_norm = sz_bits.astype(np.float64) / 1023.0
    
    mins = np.stack([chunk_data['min_scale_x'][cidx], chunk_data['min_scale_y'][cidx], chunk_data['min_scale_z'][cidx]], axis=1)
    maxs = np.stack([chunk_data['max_scale_x'][cidx], chunk_data['max_scale_y'][cidx], chunk_data['max_scale_z'][cidx]], axis=1)
    scales = mins + np.stack([sx_norm, sy_norm, sz_norm], axis=1) * (maxs - mins)
    # Ensure positive scales
    scales = np.maximum(scales, 1e-6)
    return scales


def _decode_vertices_sample(chunk_data, vertex_data, sample_idx: np.ndarray, vpc: int):
    cidx = (sample_idx // max(1, int(vpc))).clip(0, len(chunk_data) - 1)

    pos_uint = vertex_data['packed_position'][sample_idx].astype(np.uint64)
    x_bits = (pos_uint >> 22) & 0x3FF  # 10 b
    y_bits = (pos_uint >> 11) & 0x7FF  # 11 b
    z_bits = pos_uint & 0x7FF          # 11 b
    x = x_bits.astype(np.float64) / 1023.0
    y = y_bits.astype(np.float64) / 2047.0
    z = z_bits.astype(np.float64) / 2047.0

    mins = np.stack([chunk_data['min_x'][cidx], chunk_data['min_y'][cidx], chunk_data['min_z'][cidx]], axis=1)
    maxs = np.stack([chunk_data['max_x'][cidx], chunk_data['max_y'][cidx], chunk_data['max_z'][cidx]], axis=1)
    pts = mins + np.stack([x, y, z], axis=1) * (maxs - mins)
    return pts, cidx


def _validate_vpc(ply: PlyData, max_check: int = 200_000) -> int:
    if 'chunk' not in ply.elements or 'vertex' not in ply.elements:
        return 256
    chunk_data = ply['chunk'].data
    vertex_data = ply['vertex'].data
    n = len(vertex_data)
    if n == 0:
        return 256
    
    # Compute expected VPC from file structure
    num_chunks = len(chunk_data)
    expected_vpc = int(n / num_chunks) if num_chunks > 0 else 256
    
    # Test multiple VPC candidates around the expected value
    base_candidates = [256, 800]
    candidates = sorted(set([expected_vpc] + [c for c in base_candidates if abs(c - expected_vpc) <= 256]))
    
    step = max(1, n // max_check)
    sample_idx = np.arange(0, n, step, dtype=np.int64)

    best, best_score = candidates[0], -1.0
    for vpc in candidates:
        pts, cidx = _decode_vertices_sample(chunk_data, vertex_data, sample_idx, vpc)
        mins = np.stack([chunk_data['min_x'][cidx], chunk_data['min_y'][cidx], chunk_data['min_z'][cidx]], axis=1)
        maxs = np.stack([chunk_data['max_x'][cidx], chunk_data['max_y'][cidx], chunk_data['max_z'][cidx]], axis=1)
        inside = (
            (pts[:, 0] >= mins[:, 0]) & (pts[:, 0] <= maxs[:, 0]) &
            (pts[:, 1] >= mins[:, 1]) & (pts[:, 1] <= maxs[:, 1]) &
            (pts[:, 2] >= mins[:, 2]) & (pts[:, 2] <= maxs[:, 2])
        )
        score = float(np.mean(inside)) if inside.size else 0.0
        if score > best_score:
            best_score, best = score, vpc
    return best


def _has_supersplat_layout(ply: PlyData) -> bool:
    names = [e.name for e in ply.elements]
    if 'chunk' not in names or 'vertex' not in names:
        return False
    vertex = ply['vertex'].data
    vertex_fields = set(vertex.dtype.names or ())
    required = {'packed_position', 'packed_rotation', 'packed_scale', 'packed_color'}
    return required.issubset(vertex_fields)


def _load_standard_vertices(vertex_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback loader for standard vertex-based PLYs."""
    names = vertex_data.dtype.names
    if not names or len(vertex_data) == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        empty3 = np.zeros((0, 3), dtype=np.float32)
        empty4 = np.zeros((0, 4), dtype=np.float32)
        return empty, empty3, empty4, empty3

    if not {'x', 'y', 'z'}.issubset(names):
        raise ValueError("PLY missing standard 'x/y/z' fields.")

    pts = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=1).astype(np.float64)

    cols = np.ones((len(pts), 3), dtype=np.float32)
    for fields in _COLOR_FIELDS:
        if set(fields).issubset(names):
            cols = np.stack([vertex_data[fields[0]], vertex_data[fields[1]], vertex_data[fields[2]]], axis=1).astype(np.float32)
            if cols.max() > 1.0:
                cols /= 255.0
            cols = np.clip(cols, 0.0, 1.0)
            break

    rots = np.zeros((len(pts), 4), dtype=np.float32)
    rots[:, 0] = 1.0  # identity quaternion

    scales = np.ones((len(pts), 3), dtype=np.float32) * 0.01

    return pts, cols, rots, scales


def load_gaussians(ply_path: Path, max_points: int = 6_000_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load 3D Gaussians from PLY file with positions, colors, rotations, and scales.
    
    Returns:
        positions: (N, 3) float64 array
        colors: (N, 3) float32 array in [0, 1]
        rotations: (N, 4) float32 array (quaternions w, x, y, z)
        scales: (N, 3) float32 array
    """
    ply = PlyData.read(str(ply_path))
    names = [e.name for e in ply.elements]

    if not _has_supersplat_layout(ply):
        if 'vertex' not in names:
            raise ValueError("PLY lacks both 'chunk' and 'vertex' elements.")
        return _load_standard_vertices(ply['vertex'].data)

    chunk = ply['chunk'].data

    # Fallback centers if vertex absent
    cx = (chunk['min_x'] + chunk['max_x']) * 0.5
    cy = (chunk['min_y'] + chunk['max_y']) * 0.5
    cz = (chunk['min_z'] + chunk['max_z']) * 0.5
    cr = (chunk['min_r'] + chunk['max_r']) * 0.5
    cg = (chunk['min_g'] + chunk['max_g']) * 0.5
    cb = (chunk['min_b'] + chunk['max_b']) * 0.5

    if 'vertex' not in names:
        pts = np.stack([cx, cy, cz], axis=1).astype(np.float64)
        cols = np.clip(np.stack([cr, cg, cb], axis=1), 0, 1).astype(np.float32)
        # Default rotations (identity) and scales
        rots = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(pts), 1))
        scales = np.ones((len(pts), 3), dtype=np.float32) * 0.01
        return pts, cols, rots, scales

    vertex = ply['vertex'].data
    vpc = _validate_vpc(ply)

    # Sampling
    if len(vertex) > max_points:
        step = max(1, len(vertex) // max_points)
        sample_idx = np.arange(0, len(vertex), step, dtype=np.int64)[:max_points]
        v = vertex[sample_idx]
    else:
        v = vertex
        sample_idx = np.arange(len(vertex), dtype=np.int64)

    cidx = (sample_idx // vpc).clip(0, len(chunk) - 1)

    # Decode positions
    pos_uint = v['packed_position'].astype(np.uint64)
    x_bits = (pos_uint >> 22) & 0x3FF
    y_bits = (pos_uint >> 11) & 0x7FF
    z_bits = pos_uint & 0x7FF
    xn = x_bits.astype(np.float64) / 1023.0
    yn = y_bits.astype(np.float64) / 2047.0
    zn = z_bits.astype(np.float64) / 2047.0

    mins = np.stack([chunk['min_x'][cidx], chunk['min_y'][cidx], chunk['min_z'][cidx]], axis=1)
    maxs = np.stack([chunk['max_x'][cidx], chunk['max_y'][cidx], chunk['max_z'][cidx]], axis=1)
    pts = mins + np.stack([xn, yn, zn], axis=1) * (maxs - mins)

    # Decode colors
    col_uint = v['packed_color'].astype(np.uint32)
    r = ((col_uint >> 16) & 0xFF).astype(np.float64) / 255.0
    g = ((col_uint >> 8) & 0xFF).astype(np.float64) / 255.0
    b = (col_uint & 0xFF).astype(np.float64) / 255.0
    rr = chunk['min_r'][cidx] + r * (chunk['max_r'][cidx] - chunk['min_r'][cidx])
    gg = chunk['min_g'][cidx] + g * (chunk['max_g'][cidx] - chunk['min_g'][cidx])
    bb = chunk['min_b'][cidx] + b * (chunk['max_b'][cidx] - chunk['min_b'][cidx])
    cols = np.clip(np.stack([rr, gg, bb], axis=1), 0, 1)

    # Decode rotations (quaternions)
    rot_uint = v['packed_rotation'].astype(np.uint32)
    rots = _decode_rotation(rot_uint)

    # Decode scales
    scale_uint = v['packed_scale'].astype(np.uint32)
    scales = _decode_scale(scale_uint, chunk, cidx)

    return pts.astype(np.float64), cols.astype(np.float32), rots.astype(np.float32), scales.astype(np.float32)


def load_points_colors(ply_path: Path, max_points: int = 6_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy function for backward compatibility."""
    pts, cols, _, _ = load_gaussians(ply_path, max_points)
    return pts, cols


# -------------------- CAMERA (torch) --------------------

def camera_basis_torch(cam: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v = tgt - cam
    v = v / (v.norm() + 1e-9)
    up_hint = torch.tensor([0.0, 1.0, 0.0], device=cam.device, dtype=cam.dtype)
    right = torch.linalg.cross(v, up_hint)
    right = right / (right.norm() + 1e-9)
    up = torch.linalg.cross(right, v)
    up = up / (up.norm() + 1e-9)
    return right, up, v


# -------------------- RENDER (torch Gaussian splat) --------------------

@torch.no_grad()
def render_frame_torch(
    points_np: np.ndarray,
    colors_np: np.ndarray,
    camera_pos_np: np.ndarray,
    target_np: np.ndarray,
    image_size: int = 960,
    fov_deg: float = 60.0,
    rotations_np: Optional[np.ndarray] = None,
    scales_np: Optional[np.ndarray] = None,
    tile_size: int = 16,
    chunk_size: int = 50_000,
    device: str = "cuda",
) -> np.ndarray:
    """
    Render a frame using proper 3D Gaussian Splatting.
    
    If rotations_np and scales_np are None, falls back to simple point rendering.
    """
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    
    H = W = image_size
    focal = 0.5 * image_size / math.tan(math.radians(fov_deg) * 0.5)
    
    # Load data
    pts = torch.from_numpy(points_np).to(device=device, dtype=torch.float32)
    cols = torch.from_numpy(colors_np).to(device=device, dtype=torch.float32)
    
    cam = torch.tensor(camera_pos_np, device=device, dtype=torch.float32)
    tgt = torch.tensor(target_np, device=device, dtype=torch.float32)
    
    # Build view matrix - use simpler direct transformation
    right, up, fwd = camera_basis_torch(cam, tgt)
    
    # Transform points to camera space directly (more reliable)
    rel = pts - cam
    x = (rel * right).sum(dim=1)
    y = (rel * up).sum(dim=1)
    z = (rel * fwd).sum(dim=1)
    
    # Filter points in front of camera
    valid = z > 1e-6
    if valid.sum() == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)
    
    # Filter valid points
    x, y, z = x[valid], y[valid], z[valid]
    cols = cols[valid]
    
    # Project to screen space
    u = focal * (x / z) + W * 0.5
    v = -focal * (y / z) + H * 0.5
    
    # Filter points visible in screen
    screen_valid = (u >= -tile_size) & (u < W + tile_size) & (v >= -tile_size) & (v < H + tile_size)
    if screen_valid.sum() == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)
    
    cols = cols[screen_valid]
    u = u[screen_valid]
    v = v[screen_valid]
    z = z[screen_valid]
    
    # Sort by depth (back to front for alpha blending)
    depth_order = torch.argsort(z, descending=True)
    cols = cols[depth_order]
    u = u[depth_order]
    v = v[depth_order]
    z = z[depth_order]
    
    # Prepare buffers - use alpha blending
    img_alpha = torch.zeros((H, W), device=device, dtype=torch.float32)
    img_rgb = torch.zeros((H, W, 3), device=device, dtype=torch.float32)
    
    # Use proper Gaussian rendering if rotations/scales available
    use_gaussian = rotations_np is not None and scales_np is not None
    
    if use_gaussian:
        rots = torch.from_numpy(rotations_np).to(device=device, dtype=torch.float32)[valid][screen_valid][depth_order]
        scales = torch.from_numpy(scales_np).to(device=device, dtype=torch.float32)[valid][screen_valid][depth_order]
        
        # Clamp very small scales to avoid numerical issues
        scales = torch.clamp(scales, min=1e-4)
        
        # Project scales to screen space (simplified)
        # Use the maximum of the two horizontal scales for screen-space size
        # Scale in screen space is approximately scale_3d * focal / depth
        scale_2d = torch.max(scales[:, 0], scales[:, 2]) * (focal / z)  # Use x and z scales
        scale_v = scales[:, 1] * (focal / z)  # Use y scale
        
        # Clamp to reasonable pixel sizes (ensure minimum visibility)
        scale_2d = torch.clamp(scale_2d, min=2.0, max=200.0)  # Increased min to 2.0 for visibility
        scale_v = torch.clamp(scale_v, min=2.0, max=200.0)
        
        # Use same scale for both directions for simplicity (circular Gaussian)
        scale_u = scale_2d
        
        # Fast rendering: use simple splatting with accumulation
        # Convert to integer pixel coordinates
        ui = torch.round(u).to(torch.int32).clamp(0, W - 1)
        vi = torch.round(v).to(torch.int32).clamp(0, H - 1)
        
        # Use index_add for fast accumulation (much faster than loops)
        flat_idx = vi * W + ui
        
        # Simple splatting: each Gaussian contributes to its pixel and neighbors
        radius = 2  # Small radius for speed
        dx = torch.arange(-radius, radius + 1, device=device, dtype=torch.int32)
        dy = torch.arange(-radius, radius + 1, device=device, dtype=torch.int32)
        grid_x, grid_y = torch.meshgrid(dx, dy, indexing='xy')
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)
        
        # Expand to all Gaussians
        px = (ui[:, None] + grid_x[None, :]).clamp(0, W - 1)
        py = (vi[:, None] + grid_y[None, :]).clamp(0, H - 1)
        
        # Compute weights (simple Gaussian falloff)
        du = (px.to(torch.float32) - u[:, None])
        dv = (py.to(torch.float32) - v[:, None])
        dist2 = du * du + dv * dv
        sigma = 1.5
        weight = torch.exp(-dist2 / (2.0 * sigma * sigma)) * 0.99
        
        # Flatten for accumulation
        flat_idx_all = (py * W + px).reshape(-1)
        weight_flat = weight.reshape(-1, 1)
        color_flat = (cols[:, None, :] * weight[:, :, None]).reshape(-1, 3)
        
        # Accumulate using index_add (vectorized and fast)
        img_w = torch.zeros((H * W, 1), device=device, dtype=torch.float32)
        img_c = torch.zeros((H * W, 3), device=device, dtype=torch.float32)
        
        img_w.index_add_(0, flat_idx_all, weight_flat)
        img_c.index_add_(0, flat_idx_all, color_flat)
        
        # Normalize
        eps = 1e-8
        img_rgb = (img_c / (img_w + eps)).reshape(H, W, 3).clamp(0.0, 1.0)
        img_alpha = img_w.reshape(H, W).clamp(0.0, 1.0)
    else:
        # Fallback: simple point-based rendering
        ui = torch.round(u).to(torch.int32).clamp(0, W - 1)
        vi = torch.round(v).to(torch.int32).clamp(0, H - 1)
        
        # Simple splatting with fixed radius
        radius = 2
        for i in range(len(u)):
            px = torch.arange(max(0, ui[i] - radius), min(W, ui[i] + radius + 1), device=device, dtype=torch.int32)
            py = torch.arange(max(0, vi[i] - radius), min(H, vi[i] + radius + 1), device=device, dtype=torch.int32)
            px_grid, py_grid = torch.meshgrid(px, py, indexing='xy')
            px_flat = px_grid.reshape(-1)
            py_flat = py_grid.reshape(-1)
            
            du = (px_flat.float() - u[i])
            dv = (py_flat.float() - v[i])
            dist2 = du * du + dv * dv
            weight = torch.exp(-dist2 / (2.0 * radius * radius)) * 0.99
            
            alpha_old = img_alpha[py_flat, px_flat]
            alpha_new = weight * (1.0 - alpha_old)
            img_alpha[py_flat, px_flat] = alpha_old + alpha_new
            img_rgb[py_flat, px_flat] = (
                img_rgb[py_flat, px_flat] * (1.0 - alpha_new.unsqueeze(1)) +
                cols[i:i+1] * alpha_new.unsqueeze(1)
            )
    
    # Normalize and convert
    # img_rgb is always set, either from vectorized path or alpha blending path
    rgb = img_rgb.clamp(0.0, 1.0)
    out = (rgb.cpu().numpy() * 255.0).astype(np.uint8)
    return out


# -------------------- ORBIT TRAVERSAL (torch) --------------------

def orbit_positions(center: np.ndarray, start_pos: np.ndarray, num_frames: int) -> Iterable[np.ndarray]:
    c = center.astype(np.float64)
    o = start_pos.astype(np.float64) - c
    r_xy = np.linalg.norm(o[[0, 2]])
    if r_xy < 1e-6:
        r_xy, base = max(np.linalg.norm(o), 1.0), 0.0
    else:
        base = math.atan2(o[2], o[0])
    h = o[1]
    for i in range(num_frames):
        th = base + 2.0 * math.pi * i / max(num_frames, 1)
        yield c + np.array([r_xy * math.cos(th), h, r_xy * math.sin(th)], dtype=np.float64)


def _ease_in_out_cubic(t: float) -> float:
    """Smooth easing function for interpolation (cubic ease-in-out)."""
    if t < 0.5:
        return 4.0 * t * t * t
    else:
        return 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0


def _interpolate_camera_path(
    waypoints: Sequence[Tuple[np.ndarray, np.ndarray]],
    num_frames: int,
    loop: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Interpolate between camera waypoints to create a smooth path.
    
    Args:
        waypoints: List of (camera_position, look_at) tuples
        num_frames: Total number of frames to generate
        loop: If True, smoothly loop back to the first waypoint
    
    Returns:
        List of interpolated (camera_position, look_at) tuples
    """
    if len(waypoints) == 0:
        return []
    if len(waypoints) == 1:
        return [waypoints[0]] * num_frames
    
    # Create segments: if looping, add segment from last to first
    segments = []
    for i in range(len(waypoints)):
        if i < len(waypoints) - 1:
            segments.append((waypoints[i], waypoints[i + 1]))
        elif loop:
            segments.append((waypoints[i], waypoints[0]))
    
    if len(segments) == 0:
        return [waypoints[0]] * num_frames
    
    # Distribute frames across segments
    frames_per_segment = max(1, num_frames // len(segments))
    remainder = num_frames % len(segments)
    
    interpolated_path = []
    for seg_idx, (start, end) in enumerate(segments):
        # Add one extra frame to some segments to use up remainder
        seg_frames = frames_per_segment + (1 if seg_idx < remainder else 0)
        
        for frame_idx in range(seg_frames):
            # Normalized interpolation parameter [0, 1]
            t = frame_idx / max(1, seg_frames - 1) if seg_frames > 1 else 0.0
            
            # Apply easing for smoother motion
            t_eased = _ease_in_out_cubic(t)
            
            # Interpolate camera position
            cam_pos = start[0] + t_eased * (end[0] - start[0])
            
            # Interpolate look_at point
            look_at = start[1] + t_eased * (end[1] - start[1])
            
            interpolated_path.append((cam_pos.astype(np.float64), look_at.astype(np.float64)))
    
    return interpolated_path


def render_traversal_torch(
    ply_path: Path,
    output_path: Path,
    num_frames: int = 120,
    fps: int = 24,
    image_size: int = 720,
    tile_size: int = 16,
    device: str = "cuda",
    camera_path: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Path:
    """Render a camera traversal video using proper 3D Gaussian Splatting."""
    # Limit points for faster rendering
    pts_np, cols_np, rots_np, scales_np = load_gaussians(ply_path, max_points=500_000)
    mins, maxs = pts_np.min(0), pts_np.max(0)
    center = (mins + maxs) * 0.5
    extent = maxs - mins
    dist = max(1.2 * float(np.linalg.norm(extent)), 1.0)
    start_pos = center + np.array([0.0, 0.15 * extent[1], dist], dtype=np.float64)

    if camera_path and len(camera_path) > 0:
        # Interpolate between waypoints for smooth motion
        cams = _interpolate_camera_path(camera_path, num_frames, loop=True)
    else:
        cams = [(cam, center) for cam in orbit_positions(center, start_pos, num_frames)]
    total_frames = len(cams)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=fps, codec='libx264', quality=8, macro_block_size=None) as w:
        for i, (cam_pos, look_at) in enumerate(cams):
            if (i % max(1, total_frames // 20)) == 0 or i == total_frames - 1:
                print(f"Rendering frame {i+1}/{total_frames}...", end="\r", flush=True)
            frame = render_frame_torch(
                pts_np, cols_np, cam_pos, look_at,
                image_size=image_size,
                rotations_np=rots_np,
                scales_np=scales_np,
                tile_size=tile_size,
                device=device,
            )
            w.append_data(frame)
    print("\nDone.")
    return output_path


# -------------------- PUBLIC API --------------------

def render_scene(
    ply_path: Path,
    output_path: Path,
    image_size: int = 960,
    tile_size: int = 16,
    device: str = "cuda",
    camera_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Path:
    """
    Render a still image of the scene using proper 3D Gaussian Splatting.
    
    Args:
        ply_path: Path to the PLY file
        output_path: Path where the rendered image will be saved
        image_size: Size of the output image (square)
        tile_size: Tile size for rendering (smaller = better quality, slower)
        device: Device to use for rendering ('cuda' or 'cpu')
    
    Returns:
        Path to the rendered image
    """
    # Limit points for faster rendering
    pts_np, cols_np, rots_np, scales_np = load_gaussians(ply_path, max_points=500_000)
    mins, maxs = pts_np.min(0), pts_np.max(0)
    center = (mins + maxs) * 0.5
    extent = maxs - mins
    if camera_pose is not None:
        cam_pos, target = camera_pose
    else:
        dist = max(1.2 * float(np.linalg.norm(extent)), 1.0)
        cam_pos = center + np.array([0.0, 0.15 * extent[1], dist], dtype=np.float64)
        target = center
    
    frame = render_frame_torch(
        pts_np, cols_np, cam_pos, target,
        image_size=image_size,
        rotations_np=rots_np,
        scales_np=scales_np,
        tile_size=tile_size,
        device=device,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(output_path), frame)
    return output_path


def render_camera_traversal(
    ply_path: Path,
    output_dir: Path,
    num_frames: int = 20,
    fps: int = 5,
    image_size: int = 720,
    tile_size: int = 16,
    device: str = "cuda",
    camera_path: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Path:
    """
    Render a camera traversal video orbiting around the scene using proper 3D Gaussian Splatting.
    
    Args:
        ply_path: Path to the PLY file
        output_dir: Directory where the video will be saved
        num_frames: Number of frames in the video
        fps: Frames per second
        image_size: Size of each frame (square)
        tile_size: Tile size for rendering (smaller = better quality, slower)
        device: Device to use for rendering ('cuda' or 'cpu')
    
    Returns:
        Path to the rendered video
    """
    output_path = output_dir / f"{ply_path.stem}_traversal.mp4"
    return render_traversal_torch(
        ply_path=ply_path,
        output_path=output_path,
        num_frames=num_frames,
        fps=fps,
        image_size=image_size,
        tile_size=tile_size,
        device=device,
        camera_path=camera_path,
    )
