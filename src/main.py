"""Entry point tying together exploration, detection, and rendering."""
import argparse
from pathlib import Path
from typing import List

from path_planner import generate_floorplan_path_clean
from renderer import render_camera_traversal

def run_pipeline(ply_path: Path, output_root: Path) -> None:
    """Render a viewer with an interior path for the given scene."""
    
    print(f"Processing: {ply_path.name}")
    
    # Generate Interior Path
    print("Generating interior trajectory...")
    # path_data = generate_interior_path(ply_path, duration=60.0, fps=30)
    path_data = generate_floorplan_path_clean(ply_path, duration=60, fps=30)
    
    
    if not path_data:
        print("Failed to generate path (empty scene?).")
        return

    # Create Viewer
    viewer_dir = output_root / f"{ply_path.stem}_viewer"
    viewer_entry = render_camera_traversal(
        ply_path=ply_path,
        output_dir=viewer_dir,
        path_data=path_data
    )
    
    print(f"Viewer created at: {viewer_entry}")
    print(f"To view, run: python3 -m http.server 8000 --directory {viewer_dir.parent}")
    print(f"Then open: http://localhost:8000/{viewer_dir.name}/index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Video Generation Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input .ply file or directory containing .ply files")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        if input_path.suffix.lower() == ".ply":
            run_pipeline(input_path, output_root)
        else:
            print("Input must be a .ply file.")
    elif input_path.is_dir():
        ply_files = list(input_path.glob("*.ply"))
        if not ply_files:
            print(f"No .ply files found in {input_path}")
            return
        for ply in ply_files:
            run_pipeline(ply, output_root)
    else:
        print(f"Input path not found: {input_path}")

if __name__ == "__main__":
    main()

