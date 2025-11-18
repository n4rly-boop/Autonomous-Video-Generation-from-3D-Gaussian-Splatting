# Autonomous Video Generation from 3D Gaussian Splatting

## Repository Layout
- `Report.pdf` – project write-up.
- `src/` – core pipeline modules (explorer, detector, path planner, renderer, entry point).
- `configs/` – YAML configs; defaults in `config.yaml`.
- `outputs/scene_1/` – sample output folder for videos, detections, and selections.

## Spark.js point-cloud renderer
Install the Node dependencies once:

```bash
npm install
```

Render an (uncompressed) PLY into a PNG via Spark.js’ canvas APIs:

```bash
node scripts/spark_ply_renderer.js \
  --ply input-data/sample_scene/points.ply \
  --out outputs/spark/sample.png \
  --width 1920 --height 1080 --fov 55
```

Render a camera path (JSON list of `{camera, target, up}` vectors) directly to MP4:

```bash
node scripts/spark_ply_renderer.js \
  --ply input-data/sample_scene/points.ply \
  --video outputs/spark/sample.mp4 \
  --cameraSequence camera_path.json \
  --fps 12
```

Key flags:
- Supports both ASCII and `binary_little_endian` (Gaussian Splatting) PLY layouts.
- `--camera x,y,z` and `--target x,y,z` override the automatically chosen camera.
- `--maxPoints=0` (default) renders every splat; set a positive number to downsample for speed.
- `--pointScale`, `--minPointSize`, `--maxPointSize`, `--minAlpha`, `--maxAlpha` tune splat density and opacity.
- `--background #000000` changes the canvas color.
- `--cameraSequence` expects a JSON array of poses and pairs with `--video`.
- Video mode pipes frames through `ffmpeg`, so ensure it is installed/available on your PATH.
- The Python pipeline automatically injects `NODE_OPTIONS=--max-old-space-size=16384` so Node has enough heap for full-resolution renders.

Run `node scripts/spark_ply_renderer.js --help` for the full option list.
