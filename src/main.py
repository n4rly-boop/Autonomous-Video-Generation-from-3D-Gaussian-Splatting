"""Entry point tying together exploration, detection, and rendering."""
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from explorer import plan_exploration_path
from path_planner import optimize_path
from renderer import render_camera_traversal, render_scene


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


def run_pipeline(scene_root: Path, output_root: Path) -> None:
    """Render a still image and a simple camera traversal video for each scene."""
    ply_files = list(scene_root.glob("*.ply"))
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

        image_path = output_root / f"{ply_path.stem}.png"
        video_dir = output_root

        rendered_image = render_scene(
            ply_path=ply_path,
            output_path=image_path,
            camera_pose=camera_pose,
        )
        print(f"Rendered scene: {rendered_image}")

        traversal_video = render_camera_traversal(
            ply_path=ply_path,
            output_dir=video_dir,
            num_frames=150,
            fps=10
            ,
            camera_path=camera_path,
        )
        print(f"Created traversal video: {traversal_video}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scene_root = project_root / "input-data"
    output_root = project_root / "outputs" / "scene_1"
    output_root.mkdir(parents=True, exist_ok=True)
    run_pipeline(scene_root, output_root)


if __name__ == "__main__":
    main()
