import argparse
import json
import numpy as np
from plyfile import PlyData
import math

def main():
    parser = argparse.ArgumentParser(description="Generate a cinematic camera path for a Gaussian Splatting scene.")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the input .ply file.")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--num_keyframes", type=int, default=10, help="Number of keyframes for the orbit.")
    parser.add_argument("--duration_sec", type=float, default=60.0, help="Duration of the video in seconds.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second.")
    
    args = parser.parse_args()

    print(f"Reading PLY file from {args.ply_path}...")
    try:
        ply_data = PlyData.read(args.ply_path)
        vertex = ply_data['vertex']
        
        # Extract positions
        # Gaussian splat PLYs typically have x, y, z properties
        x = vertex['x']
        y = vertex['y']
        z = vertex['z']
        
        positions = np.vstack([x, y, z]).T
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        return

    print(f"Computing bounding box for {len(positions)} splats...")
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    center = (min_coords + max_coords) / 2.0
    extent = max_coords - min_coords
    
    print(f"Bounding Box Min: {min_coords}")
    print(f"Bounding Box Max: {max_coords}")
    print(f"Center: {center}")
    print(f"Extent: {extent}")

    # Path parameters
    cx, cy, cz = center
    ex, ey, ez = extent
    
    # Radius: slightly larger than the max horizontal extent
    # k ≈ 1.5 as requested
    k = 1.5
    radius = k * max(ex, ez) / 2.0 # radius is half the diameter effectively, but let's interpret 'R = k * max(ex, ez)' as requested? 
    # If ex is full width, R should be relative to half-width usually, but user said "R = k * max(ex, ez)". 
    # If ex=10, max=10. R=15. This puts it 15 units away from center. 
    # If center is 0, bounds are -5 to 5. 15 is good (10 units clearance). 
    # So R = k * max(ex, ez) is safe. Actually max(ex, ez) is the full width. 
    # Let's stick to the formula R = k * max(ex, ez) literally.
    
    radius = k * max(ex, ez)
    
    # Height: center y + 0.2 * extent y
    # Assuming y is up? In standard 3DGS/NerfStudio, Z might be up or Y might be up. 
    # SuperSplat usually expects Y up or Z up depending on export. 
    # Standard PLY often has Y up. 
    # We'll assume Y is up for calculation, but if Z is up, the orbit should be in XY plane?
    # Usually "horizontal plane" implies orbiting around the "up" axis.
    # Let's assume Y is up (common in GL/Three.js).
    # If Z is up (common in some datasets), we would orbit in XY plane.
    # However, Three.js defaults Y up.
    # Let's assume Y is up. Orbit in XZ plane.
    
    camera_height = cy + 0.2 * ey
    
    print(f"Generating path with Radius={radius:.2f}, Height={camera_height:.2f}...")

    num_frames = int(args.duration_sec * args.fps)
    frames = []

    # We generate per-frame directly to ensure smoothness, 
    # but the user prompt mentioned "Expand keyframes into per-frame samples".
    # I will implement the interpolation logic as requested to be strictly compliant.
    
    keyframes = []
    for i in range(args.num_keyframes + 1): # +1 to close the loop
        theta = 2 * math.pi * (i / args.num_keyframes)
        
        # pos(theta) = (cx + R * cos(theta), h, cz + R * sin(theta))
        px = cx + radius * math.cos(theta)
        py = camera_height
        pz = cz + radius * math.sin(theta)
        
        keyframes.append({
            "pos": np.array([px, py, pz]),
            "look_at": center
        })

    # Interpolate
    # We have num_frames total. 
    # We have N segments between N+1 keyframes (where last == first).
    # Wait, if we have K keyframes for 0..2pi, we have K segments.
    
    total_segments = args.num_keyframes
    frames_per_segment = num_frames / total_segments
    
    for f in range(num_frames):
        t = f / num_frames # 0 to 1
        
        # Find segment
        # angle covers 0 to 2pi
        # current angle = t * 2pi
        # segment index
        
        seg_idx = int(t * total_segments)
        seg_t = (t * total_segments) - seg_idx
        
        p0 = keyframes[seg_idx]["pos"]
        p1 = keyframes[seg_idx + 1]["pos"]
        
        # Linear interpolation
        pos = p0 + (p1 - p0) * seg_t
        look_at = center # Fixed look at
        
        frames.append({
            "t": f / args.fps,
            "position": pos.tolist(),
            "look_at": look_at.tolist()
        })

    output_data = {
        "fps": args.fps,
        "duration_sec": args.duration_sec,
        "frames": frames
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Path written to {args.output_json}")

if __name__ == "__main__":
    main()

