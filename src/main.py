"""Entry point tying together exploration, detection, and rendering."""
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from explorer import plan_exploration_path
from path_planner import optimize_path
from renderer import DEFAULT_UP, load_gaussians, render_camera_traversal, render_scene


def _infer_scene_flag(ply_path: Path) -> bool:
    """Heuristic for deciding if a model is a room/scene or a single object."""
    keywords = ("room", "scene", "museum", "theater", "indoor", "outdoor", "street")
    name = ply_path.stem.lower()
    return any(key in name for key in keywords)


def _build_scene_metadata(ply_path: Path, num_cameras: int = 16) -> dict:
    """Assemble metadata consumed by the explorer."""
    seed = abs(hash(ply_path.stem)) % (2**32)
    scene_flag = _infer_scene_flag(ply_path)
    scene_type = "scene" if scene_flag else "object"
    return {
        "ply_path": ply_path,
        "num_cameras": num_cameras,
        "scene_type": scene_type,
        "seed": seed,
        # Keep height modest for scenes, larger for standalone objects
        "camera_height": 0.2 if scene_flag else 0.6,
    }


def _persist_commands(commands: Iterable[str], output_path: Path) -> None:
    """Write camera commands to disk for downstream consumers."""
    lines = list(commands)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_camera_tuple(label: str, command: str) -> np.ndarray:
    match = re.search(rf"{label}=\(([^)]+)\)", command)
    if match is None:
        raise ValueError(f"Command lacks '{label}' tuple.")
    values = [float(v.strip()) for v in match.group(1).split(",")]
    if len(values) != 3:
        raise ValueError(f"{label} must have three components.")
    return np.array(values, dtype=np.float64)


def _parse_camera_command(command: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert serialized command string to numeric pose."""
    pos = _parse_camera_tuple("pos", command)
    look_at = _parse_camera_tuple("look_at", command)
    up = _parse_camera_tuple("up", command)
    return pos, look_at, up


def _commands_to_path(commands: Sequence[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    path: List[Tuple[np.ndarray, np.ndarray]] = []
    for command in commands:
        try:
            pos, look_at, _ = _parse_camera_command(command)
            path.append((pos, look_at))
        except ValueError as exc:
            print(f"[Explorer] Skipping invalid command '{command}': {exc}")
    return path


def _orbit_camera_path(
    ply_path: Path,
    num_samples: int = 120,
    radius_scale: float = 1.25,
    height_ratio: float = 0.15,
    max_points: int = 20000,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate a simple orbiting camera path around the scene centroid."""
    try:
        points, *_ = load_gaussians(ply_path, max_points=max_points)
    except Exception as exc:
        print(f"[Orbit] Failed to load gaussians for {ply_path}: {exc}")
        return []

    if len(points) == 0:
        return []

    center = np.mean(points, axis=0)
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))
    radius = max(radius, 1.0)
    height = radius * height_ratio
    radii = radius * radius_scale

    path: List[Tuple[np.ndarray, np.ndarray]] = []
    angles = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)
    for angle in angles:
        pos = center + np.array(
            [np.cos(angle) * radii, height, np.sin(angle) * radii],
            dtype=np.float64,
        )
        path.append((pos, center.copy()))
    return path


def _format_path_commands(
    path: Sequence[Tuple[np.ndarray, np.ndarray]],
    prefix: str = "orbit",
) -> List[str]:
    commands: List[str] = []
    for idx, (pos, target) in enumerate(path):
        commands.append(
            f"{prefix}_{idx:03d}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) "
            f"look_at=({target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}) "
            f"up=(0.0000, 1.0000, 0.0000)"
        )
    return commands


def _conference_hall_path(
    ply_path: Path,
    num_samples: int = 90,
    side_offset: float = 3.0,
    height_offset: float = 1.5,
    max_points: int = 500_000,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a deliberate walk-through path for a hall that bends 90 degrees.

    Heuristics:
      * Project the splats into a top-down PCA frame to recover the hallway
        footprint (forward axis aligned with the longest extent).
      * Fit an L-shaped polyline that stays within the percentile bounds of the
        footprint, using observed aisle medians near the entrance/exit to
        decide which side to occupy before and after the turn.
      * Keep the camera/focus heights clamped relative to the estimated floor
        so the rig never dips underneath the scene even if the world origin is
        offset.
    """
    try:
        points, *_ = load_gaussians(ply_path, max_points=max_points)
    except Exception as exc:
        print(f"[ConferenceHall] Failed to load gaussians: {exc}")
        return []

    if len(points) == 0:
        return []

    center = np.mean(points, axis=0)
    shifted = points - center
    cov = np.cov(shifted, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]

    major_axis = eigvecs[:, order[0]]
    if np.linalg.norm(major_axis) < 1e-6:
        return []
    major_axis = major_axis / np.linalg.norm(major_axis)

    up = DEFAULT_UP
    lateral = np.cross(up, major_axis)
    if np.linalg.norm(lateral) < 1e-6:
        lateral = eigvecs[:, order[1]]
    if np.linalg.norm(lateral) < 1e-6:
        return []
    lateral = lateral / np.linalg.norm(lateral)

    forward_proj = np.sum(shifted * major_axis, axis=1)
    min_proj = float(np.min(forward_proj))
    max_proj = float(np.max(forward_proj))
    span = max_proj - min_proj
    if span <= 1e-3:
        return []

    lateral_proj = np.sum(shifted * lateral, axis=1)
    v_low, v_high = np.percentile(lateral_proj, [5, 95])
    hall_half_width = max(abs(v_low), abs(v_high), 1.0)

    height_coords = np.sum(points * up, axis=1)
    floor_height = float(np.percentile(height_coords, 5))
    camera_height = floor_height + max(height_offset, 0.5)
    focus_height = floor_height + max(height_offset * 0.5, 0.25)

    u_margin = span * 0.08
    v_margin = hall_half_width * 0.1
    u_start = min_proj + u_margin
    u_end = max_proj - u_margin
    if u_end <= u_start:
        return []

    entry_u = max(min_proj, u_start - span * 0.05)
    exit_u = min(max_proj, u_end + span * 0.03)

    floor_mask = height_coords <= floor_height + 1.0

    def _median_v(mask: np.ndarray, fallback: float) -> float:
        subset = lateral_proj[mask]
        if subset.size < 32:
            return fallback
        return float(np.median(subset))

    start_mask = floor_mask & (forward_proj <= u_start + u_margin)
    end_mask = floor_mask & (forward_proj >= u_end - u_margin)
    start_v_guess = np.clip(-abs(side_offset), v_low + v_margin, v_high - v_margin)
    start_v = _median_v(start_mask, start_v_guess)
    end_v_guess = -start_v_guess if abs(start_v_guess) > 1e-3 else abs(side_offset)
    end_v = _median_v(end_mask, end_v_guess)

    start_v = float(np.clip(start_v, v_low + v_margin, v_high - v_margin))
    end_v = float(np.clip(end_v, v_low + v_margin, v_high - v_margin))

    turn_progress = 0.65
    turn_u = u_start + (u_end - u_start) * turn_progress
    turn_u = float(np.clip(turn_u, u_start + span * 0.05, u_end - span * 0.1))

    polyline = [
        np.array([entry_u, start_v], dtype=np.float64),
        np.array([u_start, start_v], dtype=np.float64),
        np.array([turn_u, start_v], dtype=np.float64),
        np.array([turn_u, end_v], dtype=np.float64),
        np.array([u_end, end_v], dtype=np.float64),
        np.array([exit_u, end_v], dtype=np.float64),
    ]

    segment_lengths = [
        float(np.linalg.norm(polyline[i + 1] - polyline[i]))
        for i in range(len(polyline) - 1)
    ]
    total_length = float(sum(segment_lengths))
    if total_length <= 1e-4:
        return []

    def _uv_at(distance: float) -> np.ndarray:
        remaining = float(np.clip(distance, 0.0, total_length))
        for idx, seg_len in enumerate(segment_lengths):
            if seg_len < 1e-6:
                continue
            if remaining <= seg_len:
                t = remaining / seg_len
                return polyline[idx] * (1.0 - t) + polyline[idx + 1] * t
            remaining -= seg_len
        return polyline[-1].copy()

    lookahead_dist = max(total_length * 0.05, 1.0)
    samples = np.linspace(0.0, total_length, max(2, num_samples))

    path: List[Tuple[np.ndarray, np.ndarray]] = []
    for dist in samples:
        uv = _uv_at(float(dist))
        pos = center + major_axis * uv[0] + lateral * uv[1]
        pos += up * (camera_height - float(np.dot(pos, up)))

        target_uv = _uv_at(float(min(total_length, dist + lookahead_dist)))
        target_point = center + major_axis * target_uv[0] + lateral * target_uv[1]
        target_point += up * (focus_height - float(np.dot(target_point, up)))

        path.append((pos.copy(), target_point.copy()))
    return path


def run_pipeline(scene_root: Path, output_root: Path) -> None:
    """Render a still image and a simple camera traversal video for each scene."""
    ply_files = list(scene_root.glob("*.ply"))
    conf_files = [p for p in ply_files if "conferencehall" in p.stem.lower()]
    if conf_files:
        ply_files = conf_files
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {scene_root}")
    
    for ply_path in ply_files:
        scene_metadata = _build_scene_metadata(ply_path)
        coarse_commands = plan_exploration_path(scene_metadata)
        trajectory_commands: List[str] = list(optimize_path(coarse_commands))
        camera_plan_path = output_root / f"{ply_path.stem}_camera_plan.txt"
        _persist_commands(trajectory_commands, camera_plan_path)
        print(f"Saved exploration plan: {camera_plan_path}")

        camera_path = _commands_to_path(trajectory_commands)
        camera_pose = camera_path[0] if camera_path else None

        conference_path = None
        if "conferencehall" in ply_path.stem.lower():
            conference_path = _conference_hall_path(ply_path)

        orbit_path = _orbit_camera_path(ply_path) if conference_path is None else []
        if orbit_path:
            orbit_commands = _format_path_commands(orbit_path, prefix="orbit")
            orbit_plan_path = output_root / f"{ply_path.stem}_orbit_plan.txt"
            _persist_commands(orbit_commands, orbit_plan_path)
            print(f"Saved orbit plan: {orbit_plan_path}")
            if camera_pose is None:
                camera_pose = orbit_path[0]

        if conference_path:
            conference_commands = _format_path_commands(conference_path, prefix="conference")
            conference_plan_path = output_root / f"{ply_path.stem}_conference_plan.txt"
            _persist_commands(conference_commands, conference_plan_path)
            print(f"Saved conference plan: {conference_plan_path}")
            camera_path = conference_path
            camera_pose = conference_path[0]

        viewer_dir = output_root / f"{ply_path.stem}_viewer"

        viewer_entry = render_scene(
            ply_path=ply_path,
            output_path=viewer_dir,
            camera_pose=camera_pose,
            width=600,
            height=400,
        )
        print(f"Prepared Spark viewer: {viewer_entry}")

        viewer_with_path = render_camera_traversal(
            ply_path=ply_path,
            output_dir=viewer_dir,
            num_frames=56,
            fps=8,
            width=600,
            height=400,
            camera_path=conference_path or orbit_path or camera_path,
        )
        print(f"Viewer now includes interactive path: {viewer_with_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scene_root = project_root / "input-data"
    output_root = project_root / "outputs" / "scene_1"
    output_root.mkdir(parents=True, exist_ok=True)
    run_pipeline(scene_root, output_root)


if __name__ == "__main__":
    main()
