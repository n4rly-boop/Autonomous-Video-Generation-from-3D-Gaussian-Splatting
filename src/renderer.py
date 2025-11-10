"""Renderer for 3D Gaussian Splatting PLY files."""
import numpy as np
from pathlib import Path
from plyfile import PlyData
import imageio


def unpack_position(packed: np.ndarray) -> np.ndarray:
    """Unpack position from uint32 to 3D coordinates.
    
    The packed position is stored as a uint32. We interpret it as a float
    and extract the 3D coordinates. This is a simplified unpacking method.
    """
    # Convert uint32 to float32 view to extract coordinates
    # This assumes the data was packed as floats
    positions = packed['packed_position'].astype(np.float32).view(np.float32)
    
    # If that doesn't work, try interpreting as structured data
    # For now, we'll use a sampling approach from chunks
    return None


def unpack_color(packed: np.ndarray) -> np.ndarray:
    """Unpack color from uint32 to RGB values."""
    # Similar approach - convert to float32 view
    colors = packed['packed_color'].astype(np.float32).view(np.float32)
    return None


def render_scene(ply_path: Path, output_path: Path) -> Path:
    """Render a single frame from the 3D Gaussian Splatting scene.
    
    Args:
        ply_path: Path to the PLY file
        output_path: Path where the rendered image will be saved
        
    Returns:
        Path to the saved image
    """
    # Read PLY file
    ply_data = PlyData.read(str(ply_path))
    
    # Extract chunk data (bounding boxes)
    element_names = [e.name for e in ply_data.elements]
    if 'chunk' in element_names:
        chunk_data = ply_data['chunk'].data
        # Get bounding box centers as representative points
        centers_x = (chunk_data['min_x'] + chunk_data['max_x']) / 2.0
        centers_y = (chunk_data['min_y'] + chunk_data['max_y']) / 2.0
        centers_z = (chunk_data['min_z'] + chunk_data['max_z']) / 2.0
        
        # Get colors from chunks
        colors_r = (chunk_data['min_r'] + chunk_data['max_r']) / 2.0
        colors_g = (chunk_data['min_g'] + chunk_data['max_g']) / 2.0
        colors_b = (chunk_data['min_b'] + chunk_data['max_b']) / 2.0
        
        points = np.column_stack([centers_x, centers_y, centers_z])
        colors = np.column_stack([colors_r, colors_g, colors_b])
    else:
        raise ValueError("PLY file does not contain 'chunk' element")
    
    # Extract vertex data and unpack using chunk-based quantization
    # KEY INSIGHT: Vertices are organized by chunks (256 vertices per chunk)
    # Each vertex's position is quantized relative to its chunk's bounds, not global bounds
    if 'vertex' in element_names:
        vertex_data = ply_data['vertex'].data
        
        # Determine vertices per chunk (typically 256, last chunk may be partial)
        estimated_vpc = int(round(len(vertex_data) / max(1, len(chunk_data))))
        if 240 <= estimated_vpc <= 272:
            vertices_per_chunk = 256
        else:
            vertices_per_chunk = max(1, estimated_vpc)
        
        # Use ALL vertices for maximum detail - the scene needs full point density
        # For very large scenes, we can sample, but try to use as many as possible
        max_points = 6000000  # Use up to 6M points
        if len(vertex_data) > max_points:
            # Use systematic sampling with smaller step for better coverage
            step = len(vertex_data) // max_points
            sample_indices = np.arange(0, len(vertex_data), step)[:max_points]
            sampled_vertices = vertex_data[sample_indices]
        else:
            # Use all vertices
            sampled_vertices = vertex_data
            sample_indices = np.arange(len(vertex_data))
        
        # Unpack positions and colors using chunk-based quantization (vectorized)
        num_samples = len(sampled_vertices)
        
        # Determine which chunk each vertex belongs to
        vertex_indices = sample_indices
        chunk_indices = vertex_indices // vertices_per_chunk
        # Clamp chunk indices to valid range (some vertices might be beyond last chunk)
        chunk_indices = np.clip(chunk_indices, 0, len(chunk_data) - 1)
        
        # Unpack positions (10-11-11 bit layout) - vectorized
        pos_uint = sampled_vertices['packed_position'].astype(np.uint64)
        x_bits = (pos_uint >> 22) & 0x3FF  # 10 bits
        y_bits = (pos_uint >> 11) & 0x7FF  # 11 bits
        z_bits = pos_uint & 0x7FF           # 11 bits
        
        # Normalize to [0, 1]
        x_norm = x_bits.astype(np.float64) / 1023.0
        y_norm = y_bits.astype(np.float64) / 2047.0
        z_norm = z_bits.astype(np.float64) / 2047.0
        
        # Get chunk bounds for each vertex (vectorized)
        chunk_mins = np.column_stack([
            chunk_data['min_x'][chunk_indices],
            chunk_data['min_y'][chunk_indices],
            chunk_data['min_z'][chunk_indices]
        ])
        chunk_maxs = np.column_stack([
            chunk_data['max_x'][chunk_indices],
            chunk_data['max_y'][chunk_indices],
            chunk_data['max_z'][chunk_indices]
        ])
        chunk_ranges = chunk_maxs - chunk_mins
        
        # Map normalized coordinates to chunk bounds
        points = chunk_mins + np.column_stack([x_norm, y_norm, z_norm]) * chunk_ranges
        
        # Unpack colors relative to chunk color ranges (vectorized)
        col_uint = sampled_vertices['packed_color'].astype(np.uint32)
        r_bits = (col_uint >> 16) & 0xFF
        g_bits = (col_uint >> 8) & 0xFF
        b_bits = col_uint & 0xFF
        
        # Normalize to [0, 1]
        r_norm = r_bits.astype(np.float64) / 255.0
        g_norm = g_bits.astype(np.float64) / 255.0
        b_norm = b_bits.astype(np.float64) / 255.0
        
        # Get chunk color ranges
        chunk_r_min = chunk_data['min_r'][chunk_indices]
        chunk_r_max = chunk_data['max_r'][chunk_indices]
        chunk_g_min = chunk_data['min_g'][chunk_indices]
        chunk_g_max = chunk_data['max_g'][chunk_indices]
        chunk_b_min = chunk_data['min_b'][chunk_indices]
        chunk_b_max = chunk_data['max_b'][chunk_indices]
        
        # Map normalized colors to chunk color ranges
        r = chunk_r_min + r_norm * (chunk_r_max - chunk_r_min)
        g = chunk_g_min + g_norm * (chunk_g_max - chunk_g_min)
        b = chunk_b_min + b_norm * (chunk_b_max - chunk_b_min)
        
        colors = np.column_stack([r, g, b])
        
        # Clamp colors to valid range
        colors = np.clip(colors, 0, 1)
    
    # Render the point cloud as a 2D image
    # Choose a camera perspective (looking at the scene from a good angle)
    # Calculate scene center and bounds
    scene_center = points.mean(axis=0)
    scene_size = points.max(axis=0) - points.min(axis=0)
    max_size = scene_size.max()
    
    # Set up camera position for better scene visibility (perspective camera)
    camera_distance = max(max_size * 2.2, 1.0)
    camera_pos = scene_center + np.array([
        camera_distance * 0.6,   # X offset
        camera_distance * 0.35,  # Y offset (elevation)
        camera_distance * 1.1    # Z offset (forward)
    ])
    
    # Build camera direction
    view_dir = scene_center - camera_pos
    view_norm = np.linalg.norm(view_dir)
    if view_norm > 1e-6:
        view_dir = view_dir / view_norm
    else:
        view_dir = np.array([0, 0, -1])  # Default view direction
    
    # Create orthonormal basis for the camera
    up = np.array([0, 1, 0])
    right = np.cross(view_dir, up)
    right_norm = np.linalg.norm(right)
    if right_norm > 1e-6:
        right = right / right_norm
    else:
        right = np.array([1, 0, 0])  # Default right vector
    up = np.cross(right, view_dir)
    up_norm = np.linalg.norm(up)
    if up_norm > 1e-6:
        up = up / up_norm
    else:
        up = np.array([0, 1, 0])  # Default up vector
    
    # Project points into camera space
    relative_pos = points - camera_pos
    x_cam = np.sum(relative_pos * right, axis=1)
    y_cam = np.sum(relative_pos * up, axis=1)
    z_cam = np.sum(relative_pos * view_dir, axis=1)
    
    # Cull invalid values and points behind/too far from the camera
    finite_mask = np.isfinite(x_cam) & np.isfinite(y_cam) & np.isfinite(z_cam)
    pos_depths = z_cam[finite_mask & (z_cam > 0)]
    # Initial near/far guess from scene scale
    near = max(1e-4, 0.01 * max_size)
    far = max(near * 80.0, 2.0 * max_size)
    valid_mask = finite_mask & (z_cam > near) & (z_cam < far)
    # Adaptive fallback if the heuristic window rejected everything
    if not np.any(valid_mask):
        if pos_depths.size > 0:
            near = max(1e-5, float(np.percentile(pos_depths, 0.5)) * 0.5)
            far = float(np.percentile(pos_depths, 99.5)) * 1.5
            valid_mask = finite_mask & (z_cam > near) & (z_cam < far)
    # Last-resort fallback: accept any positive depth
    if not np.any(valid_mask):
        valid_mask = finite_mask & (z_cam > 1e-6)
    if not np.any(valid_mask):
        raise ValueError("No valid points after projection/clipping")
    
    x_cam = x_cam[valid_mask]
    y_cam = y_cam[valid_mask]
    z_cam = z_cam[valid_mask]
    colors = colors[valid_mask]
    
    # Perspective projection
    image_size = 2048
    fov_deg = 60.0
    f = 0.5 * image_size / np.tan(np.deg2rad(fov_deg) * 0.5)
    x_img = (f * (x_cam / z_cam)) + (image_size * 0.5)
    y_img = (-f * (y_cam / z_cam)) + (image_size * 0.5)
    
    x_img = np.clip(np.rint(x_img).astype(np.int32), 0, image_size - 1)
    y_img = np.clip(np.rint(y_img).astype(np.int32), 0, image_size - 1)
    
    # Depth ordering (far to near)
    depth_order = np.argsort(z_cam)
    
    # Black background for contrast
    image = np.zeros((image_size, image_size, 3), dtype=np.float32)
    
    # Depth buffer and alpha blending
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
    
    for idx in depth_order[::-1]:  # iterate far -> near (draw near last)
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
    
    # Normalize blended colors where any contributions exist
    nonzero = wgt[:, :, 0] > 1e-6
    image[nonzero] = acc[nonzero] / wgt[nonzero]
    
    # Convert to uint8 for saving
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Save image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(output_path), image_uint8)
    
    return output_path

