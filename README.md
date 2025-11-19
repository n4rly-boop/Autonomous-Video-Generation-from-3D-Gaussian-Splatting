# Autonomous Video Generation from 3D Gaussian Splatting

## Repository Layout
- `Report.pdf` – project write-up.
- `src/` – core pipeline modules (explorer, detector, path planner, renderer, entry point).
- `configs/` – YAML configs; defaults in `config.yaml`.
- `outputs/scene_1/` – sample output folder for videos, detections, and selections.

## Cinematic Navigation Viewer

This project includes a web-based viewer using **Spark.js** and **Three.js** to render a cinematic fly-through of a 3D Gaussian Splatting scene.

### 1. Generate Camera Path

First, use the Python script to generate a smooth camera orbit path based on your `.ply` scene.

```bash
# Install dependencies
pip install plyfile numpy

# Run the path planner
# Usage: python src/plan_path.py --ply_path <path_to_ply> --output_json <output_path>
python src/plan_path.py --ply_path input-data/ConferenceHall.ply --output_json spark-viewer/path.json
```

This creates `spark-viewer/path.json` containing the animation keyframes.

### 2. Launch the Web Viewer

The viewer is located in the `spark-viewer/` directory. It loads the scene and the generated path.

1.  Ensure your scene file is accessible. By default, the viewer expects a file named `scene.ply` in the `spark-viewer` folder. You can symlink your input data:
    ```bash
    ln -sf $(pwd)/input-data/ConferenceHall.ply spark-viewer/scene.ply
    ```

2.  Start a local HTTP server (required for WASM/CORS):
    ```bash
    cd spark-viewer
    python3 -m http.server 8000
    ```

3.  Open **http://localhost:8000** in your browser.

### Viewer Notes
- The viewer uses a local vendor copy of Three.js (r165) and Spark.js (0.1.10) in `spark-viewer/lib/` to ensure compatibility.
- If the screen is black, check the browser console (F12) for errors.
- Ensure you are running a modern browser with WebGL 2 support.
