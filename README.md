# Autonomous Video Generation from 3D Gaussian Splatting

## Repository Layout
- `Report.pdf` – project write-up.
- `src/` – core pipeline modules (explorer, detector, path planner, renderer, entry point).
- `configs/` – YAML configs; defaults in `config.yaml`.
- `outputs/` – output folder for viewers.

## Usage

### 1. Generate Cinematic Viewer

Run the main pipeline on a single `.ply` file or a directory of `.ply` files.

```bash
# Install dependencies
pip install plyfile numpy

# Run on a single file
python src/main.py --input input-data/ConferenceHall.ply --output outputs

# Run on a directory
python src/main.py --input input-data --output outputs
```

This will:
1.  Analyze the scene geometry using PCA.
2.  Generate a smooth "figure-8" trajectory inside the scene boundaries.
3.  Create a self-contained web viewer in `outputs/<scene_name>_viewer/`.

### 2. Launch the Web Viewer

To view the results, start a local HTTP server in the output directory:

```bash
python3 -m http.server 8000 --directory outputs
```

Then open your browser to **http://localhost:8000**. You will see folders for each scene. Click on a folder to open its viewer.

### Viewer Notes
- The viewer uses a local vendor copy of Three.js (r165) and Spark.js (0.1.10) in `lib/` to ensure compatibility.
- A WebGL 2 compatible browser is required.
