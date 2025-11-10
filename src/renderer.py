"""Renderer for 3D Gaussian Splatting PLY files."""
import math
from pathlib import Path
from typing import Iterable, Tuple

import imageio
import numpy as np
from plyfile import PlyData


def _load_points_and_colors(ply_path: Path, max_points: int = 6_000_000) -> Tuple[np.ndarray, np.ndarray]:
    """Load (N,3) XYZ points and RGB colors in [0,1] from a PLY file."""
    ply_data = PlyData.read(str(ply_path))
    element_names = [e.name for e in ply_data.elements]

    if 'chunk' not in element_names:
        raise ValueError("PLY file does not contain 'chunk' element")

    chunk_data = ply_data['chunk'].data

    # Default to chunk centers so we always have something to look at
    centers_x = (chunk_data['min_x'] + chunk_data['max_x']) / 2.0
    centers_y = (chunk_data['min_y'] + chunk_data['max_y']) / 2.0
    centers_z = (chunk_data['min_z'] + chunk_data['max_z']) / 2.0
    colors_r = (chunk_data['min_r'] + chunk_data['max_r']) / 2.0
    colors_g = (chunk_data['min_g'] + chunk_data['max_g']) / 2.0
    colors_b = (chunk_data['min_b'] + chunk_data['max_b']) / 2.0

    points = np.column_stack([centers_x, centers_y, centers_z])
    colors = np.column_stack([colors_r, colors_g, colors_b])

    if 'vertex' not in element_names:
        return points.astype(np.float64), np.clip(colors, 0, 1).astype(np.float32)

    vertex_data = ply_data['vertex'].data
    estimated_vpc = int(round(len(vertex_data) / max(1, len(chunk_data))))
    if 240 <= estimated_vpc <= 272:
        vertices_per_chunk = 256
    else:
        vertices_per_chunk = max(1, estimated_vpc)

    if len(vertex_data) > max_points:
        step = len(vertex_data) // max_points
        sample_indices = np.arange(0, len(vertex_data), step)[:max_points]
        sampled_vertices = vertex_data[sample_indices]
    else:
        sampled_vertices = vertex_data
        sample_indices = np.arange(len(vertex_data))

    vertex_indices = sample_indices
    chunk_indices = vertex_indices // vertices_per_chunk
    chunk_indices = np.clip(chunk_indices, 0, len(chunk_data) - 1)

    pos_uint = sampled_vertices['packed_position'].astype(np.uint64)
    x_bits = (pos_uint >> 22) & 0x3FF  # 10 bits
    y_bits = (pos_uint >> 11) & 0x7FF  # 11 bits
    z_bits = pos_uint & 0x7FF          # 11 bits

    x_norm = x_bits.astype(np.float64) / 1023.0
    y_norm = y_bits.astype(np.float64) / 2047.0
    z_norm = z_bits.astype(np.float64) / 2047.0

    chunk_mins = np.column_stack([
        chunk_data['min_x'][chunk_indices],
        chunk_data['min_y'][chunk_indices],
        chunk_data['min_z'][chunk_indices],
    ])
    chunk_maxs = np.column_stack([
        chunk_data['max_x'][chunk_indices],
        chunk_data['max_y'][chunk_indices],
        chunk_data['max_z'][chunk_indices],
    ])
    chunk_ranges = chunk_maxs - chunk_mins

    points = chunk_mins + np.column_stack([x_norm, y_norm, z_norm]) * chunk_ranges

    col_uint = sampled_vertices['packed_color'].astype(np.uint32)
    r_bits = (col_uint >> 16) & 0xFF
    g_bits = (col_uint >> 8) & 0xFF
    b_bits = col_uint & 0xFF

    r_norm = r_bits.astype(np.float64) / 255.0
    g_norm = g_bits.astype(np.float64) / 255.0
    b_norm = b_bits.astype(np.float64) / 255.0

    chunk_r_min = chunk_data['min_r'][chunk_indices]
    chunk_r_max = chunk_data['max_r'][chunk_indices]
    chunk_g_min = chunk_data['min_g'][chunk_indices]
    chunk_g_max = chunk_data['max_g'][chunk_indices]
    chunk_b_min = chunk_data['min_b'][chunk_indices]
    chunk_b_max = chunk_data['max_b'][chunk_indices]

    r = chunk_r_min + r_norm * (chunk_r_max - chunk_r_min)
    g = chunk_g_min + g_norm * (chunk_g_max - chunk_g_min)
    b = chunk_b_min + b_norm * (chunk_b_max - chunk_b_min)
    colors = np.column_stack([r, g, b])
    colors = np.clip(colors, 0, 1)

    return points.astype(np.float64), colors.astype(np.float32)


def _camera_basis(camera_pos: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return right, up, forward vectors for the given camera pose."""
    view_dir = target - camera_pos
    view_norm = np.linalg.norm(view_dir)
    if view_norm > 1e-6:
        view_dir = view_dir / view_norm
    else:
        view_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(view_dir, up_hint)
    right_norm = np.linalg.norm(right)
    if right_norm > 1e-6:
        right = right / right_norm
    else:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    up = np.cross(right, view_dir)
    up_norm = np.linalg.norm(up)
    if up_norm > 1e-6:
        up = up / up_norm
    else:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    return right, up, view_dir


def render_frame(
    points: np.ndarray,
    colors: np.ndarray,
    camera_pos: np.ndarray,
    target: np.ndarray,
    *,
    image_size: int = 2048,
    fov_deg: float = 60.0,
) -> np.ndarray:
    """Render a single RGB frame from prepared scene data."""
    points = np.asarray(points)
    colors = np.asarray(colors)
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    scene_size = points.max(axis=0) - points.min(axis=0)
    max_size = max(float(np.max(scene_size)), 1.0)

    right, up, view_dir = _camera_basis(camera_pos, target)

    relative_pos = points - camera_pos
    x_cam = np.sum(relative_pos * right, axis=1)
    y_cam = np.sum(relative_pos * up, axis=1)
    z_cam = np.sum(relative_pos * view_dir, axis=1)

    finite_mask = np.isfinite(x_cam) & np.isfinite(y_cam) & np.isfinite(z_cam)
    pos_depths = z_cam[finite_mask & (z_cam > 0)]
    near = max(1e-4, 0.01 * max_size)
    far = max(near * 80.0, 2.0 * max_size)
    valid_mask = finite_mask & (z_cam > near) & (z_cam < far)
    if not np.any(valid_mask) and pos_depths.size > 0:
        near = max(1e-5, float(np.percentile(pos_depths, 0.5)) * 0.5)
        far = float(np.percentile(pos_depths, 99.5)) * 1.5
        valid_mask = finite_mask & (z_cam > near) & (z_cam < far)
    if not np.any(valid_mask):
        valid_mask = finite_mask & (z_cam > 1e-6)
    if not np.any(valid_mask):
        raise ValueError("No valid points after projection/clipping")

    x_cam = x_cam[valid_mask]
    y_cam = y_cam[valid_mask]
    z_cam = z_cam[valid_mask]
    colors = colors[valid_mask]

    f = 0.5 * image_size / math.tan(math.radians(fov_deg) * 0.5)
    x_img = (f * (x_cam / z_cam)) + (image_size * 0.5)
    y_img = (-f * (y_cam / z_cam)) + (image_size * 0.5)

    x_img = np.clip(np.rint(x_img).astype(np.int32), 0, image_size - 1)
    y_img = np.clip(np.rint(y_img).astype(np.int32), 0, image_size - 1)

    depth_order = np.argsort(z_cam)

    image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    depth_buffer = np.full((image_size, image_size), np.inf)

    num_points = len(x_img)
    if num_points > 3_000_000:
        point_size = 1
    elif num_points > 1_000_000:
        point_size = 2
    else:
        point_size = 3
    half = max(0, point_size // 2)
    alpha = 0.85

    acc = np.zeros_like(image)
    wgt = np.zeros((image_size, image_size, 1), dtype=np.float32)

    for idx in depth_order[::-1]:
        x = x_img[idx]
        y = y_img[idx]
        depth = z_cam[idx]
        if depth >= depth_buffer[y, x]:
            continue
        depth_buffer[y, x] = depth

        c = np.clip(colors[idx], 0.0, 1.0)

        if point_size == 1:
            acc[y, x] += alpha * c
            wgt[y, x, 0] += alpha
        else:
            y0 = max(0, y - half)
            y1 = min(image_size, y + half + 1)
            x0 = max(0, x - half)
            x1 = min(image_size, x + half + 1)
            h = y1 - y0
            w = x1 - x0
            acc[y0:y1, x0:x1] += (alpha * c).reshape(1, 1, 3).repeat(h, axis=0).repeat(w, axis=1)
            wgt[y0:y1, x0:x1, 0] += alpha

    nonzero = wgt[:, :, 0] > 1e-6
    image[nonzero] = acc[nonzero] / wgt[nonzero]

    image_uint8 = (image * 255).astype(np.uint8)
    return image_uint8


def render_scene(ply_path: Path, output_path: Path, *, image_size: int = 2048) -> Path:
    """Render a single frame from the 3D Gaussian Splatting scene."""
    points, colors = _load_points_and_colors(ply_path)
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = (mins + maxs) * 0.5
    extent = maxs - mins

    camera_distance = max(float(extent.max()) * 2.2, 1.0)
    camera_pos = center + np.array([
        camera_distance * 0.6,
        camera_distance * 0.35,
        camera_distance * 1.1,
    ])

    frame = render_frame(points, colors, camera_pos, center, image_size=image_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(output_path), frame)
    return output_path


def _generate_orbit_path(center: np.ndarray, start_pos: np.ndarray, num_frames: int) -> Iterable[np.ndarray]:
    """Yield camera positions that orbit around the scene center."""
    center = np.asarray(center, dtype=np.float64)
    start_pos = np.asarray(start_pos, dtype=np.float64)
    offset = start_pos - center
    horizontal = offset[[0, 2]]
    radius = float(np.linalg.norm(horizontal))
    if radius < 1e-4:
        radius = max(float(np.linalg.norm(offset)), 1.0)
        base_angle = 0.0
    else:
        base_angle = math.atan2(horizontal[1], horizontal[0])
    height = offset[1]

    for idx in range(num_frames):
        theta = base_angle + (2.0 * math.pi * idx / max(num_frames, 1))
        x = radius * math.cos(theta)
        z = radius * math.sin(theta)
        yield center + np.array([x, height, z], dtype=np.float64)


def _write_video_with_fallback(
    camera_positions: Iterable[np.ndarray],
    points: np.ndarray,
    colors: np.ndarray,
    output_path: Path,
    target: np.ndarray,
    *,
    fps: int,
    image_size: int,
) -> Path:
    """Write frames to MP4, falling back to GIF if FFmpeg is unavailable."""
    camera_positions = list(camera_positions)
    num_frames = len(camera_positions)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            str(output_path),
            fps=fps,
            codec='libx264',
            quality=8,
            macro_block_size=None,
        ) as writer:
            for frame_idx, cam_pos in enumerate(camera_positions):
                if frame_idx % max(1, num_frames // 20) == 0 or frame_idx == num_frames - 1:
                    print(f"  Rendering frame {frame_idx + 1}/{num_frames}...", end='\r', flush=True)
                frame = render_frame(points, colors, cam_pos, target, image_size=image_size)
                writer.append_data(frame)
        print(f"  Rendered {num_frames} frames successfully.")
        return output_path
    except Exception as exc:
        if output_path.exists():
            try:
                output_path.unlink()
            except FileNotFoundError:
                pass
        print(f"MP4 export failed ({exc}); falling back to GIF.")
        frames = []
        for frame_idx, cam_pos in enumerate(camera_positions):
            if frame_idx % max(1, num_frames // 20) == 0 or frame_idx == num_frames - 1:
                print(f"  Rendering frame {frame_idx + 1}/{num_frames}...", end='\r', flush=True)
            frames.append(render_frame(points, colors, cam_pos, target, image_size=image_size))
        print(f"  Rendered {num_frames} frames successfully.")
        gif_path = output_path.with_suffix(".gif")
        imageio.mimsave(str(gif_path), frames, duration=1.0 / max(fps, 1))
        return gif_path


def render_camera_traversal(
    ply_path: Path,
    output_dir: Path,
    *,
    num_frames: int = 120,
    fps: int = 24,
    image_size: int = 960,
) -> Path:
    """Render an orbital camera traversal video around the reconstructed scene."""
    num_frames = max(2, num_frames)
    fps = max(1, fps)
    image_size = max(128, image_size)
    print(f"Loading scene from {ply_path.name}...")
    points, colors = _load_points_and_colors(ply_path)
    print(f"  Loaded {len(points):,} points")
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = (mins + maxs) * 0.5
    extent = maxs - mins

    extent_norm = float(np.linalg.norm(extent))
    dist = max(1.2 * extent_norm, 1.0)
    start_pos = center + np.array([
        0.0,
        0.15 * extent[1],
        dist,
    ])

    camera_positions = _generate_orbit_path(center, start_pos, num_frames)
    video_path = output_dir / f"{ply_path.stem}_traversal.mp4"
    print(f"Rendering {num_frames} frames at {image_size}x{image_size}...")
    return _write_video_with_fallback(
        camera_positions,
        points,
        colors,
        video_path,
        center,
        fps=fps,
        image_size=image_size,
    )
