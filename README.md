# Autonomous Video Generation from 3D Gaussian Splatting

Generate cinematic camera paths and lightweight interactive viewers directly from 3D Gaussian Splatting reconstructions (`.ply`). The pipeline extracts indoor contours, plans a traversal, and emits a static Spark.js viewer that can autoplay the path or switch to FPS controls.

## Repository Tour
- `src/` – Python pipeline: `main.py` orchestrates exploration/path planning (`explorer.py`, `path_planner.py`) and viewer export (`renderer.py`).
- `spark-viewer/` – self-contained Spark.js/WebGL template copied into every output viewer.
- `scripts/` – utilities such as `spark_ply_renderer.js` (Node-based offline renderer) and inspection helpers.
- `input-data/` – sample `.ply` assets (e.g., `ConferenceHall.ply`).
- `outputs/` – pipeline results (`<scene>_viewer/` folders containing `scene.ply`, `viewer-config.*`, and `path.json`).
- `NoField-seeding-main/` – upstream NoField seeding reference used for experimentation; `Assignment4-ICV.pdf` describes the write-up.

## Prerequisites
| Stack | Notes |
| --- | --- |
| Python 3.10+ | `pip install -r requirements.txt` |
| Node.js 18+ | `npm install` |

## Quick Start
```bash
# 1) Activate your Python environment and install dependencies (see table above)
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2) Generate viewers for a single scene or a folder of .ply files
python src/main.py --input input-data/ConferenceHall.ply --output outputs
# python src/main.py --input input-data --output outputs   # batch mode

# 3) Host the viewers
python -m http.server 8000 --directory outputs
# Open http://localhost:8000/ and choose the scene you want to view
```

Use WASD + mouse FPS controls for interactive mode, or press `P` and wait a bit to play the prerecorded path.

What happens under the hood:
1. `generate_floorplan_path_clean` denoises the point cloud, slices walls, fits a convex interior contour, and distributes poses (falls back to the PCA figure-eight path if needed).
2. `renderer.py` copies the Spark template, writes `scene.ply`, `viewer-config.*`, and persists the computed `path.json`.
3. The viewer autoplays the exported traversal or allows you to control the camera manually with WASD + mouse FPS controls.
