"""Entry point tying together exploration, detection, and rendering."""
from __future__ import annotations
from pathlib import Path

from explorer import plan_exploration_path
from detector import detect_objects
from path_planner import optimize_path
from renderer import render_tour


def run_pipeline(scene_root: Path, output_root: Path) -> None:
    """High-level orchestration for one scene."""
    scene_meta = {"root": scene_root}
    coarse_path = plan_exploration_path(scene_meta)
    refined_path = optimize_path(coarse_path)

    panorama_video = render_tour(output_root / "panorama_tour.mp4", refined_path)
    detections = detect_objects([panorama_video.read_bytes()])

    if detections:
        (output_root / "detected_objects.json").write_text("TODO: export detections")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scene_root = project_root / "input-data"
    output_root = project_root / "outputs" / "scene_1"
    output_root.mkdir(parents=True, exist_ok=True)
    run_pipeline(scene_root, output_root)


if __name__ == "__main__":
    main()
