"""Entry point tying together exploration, detection, and rendering."""
from pathlib import Path
from renderer import render_scene


def run_pipeline(scene_root: Path, output_root: Path) -> None:
    """Render a static view of the scene."""
    ply_files = list(scene_root.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {scene_root}")
    
    ply_paths = list(scene_root.glob("*.ply"))
    for ply_path in ply_paths:
        output_image = render_scene(
            ply_path=ply_path,
            output_path=output_root / f"{ply_path.stem}.png"
        )
        print(f"Rendered scene: {output_image}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    scene_root = project_root / "input-data"
    output_root = project_root / "outputs" / "scene_1"
    output_root.mkdir(parents=True, exist_ok=True)
    run_pipeline(scene_root, output_root)


if __name__ == "__main__":
    main()
