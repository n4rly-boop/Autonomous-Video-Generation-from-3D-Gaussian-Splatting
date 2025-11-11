"""Entry point tying together exploration, detection, and rendering."""
from pathlib import Path

from renderer import render_camera_traversal, render_scene


def run_pipeline(scene_root: Path, output_root: Path) -> None:
    """Render a still image and a simple camera traversal video for each scene."""
    ply_files = list(scene_root.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {scene_root}")
    
    files = ["Museume.ply", "outdoor-street.ply","Theater.ply", "outdoor-drone.ply"]
    # ply_files = [scene_root / file for file in files]

    for ply_path in ply_files:
        image_path = output_root / f"{ply_path.stem}.png"
        video_dir = output_root

        rendered_image = render_scene(
            ply_path=ply_path,
            output_path=image_path,
        )
        print(f"Rendered scene: {rendered_image}")

        traversal_video = render_camera_traversal(
            ply_path=ply_path,
            output_dir=video_dir,
            num_frames=40,
            fps=8
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
