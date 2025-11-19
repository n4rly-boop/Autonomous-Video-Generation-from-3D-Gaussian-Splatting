"""Spark.js web viewer renderer.

This module prepares a static Spark viewer (HTML + ES modules) for every scene.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Any

import numpy as np
from plyfile import PlyData

DEFAULT_WIDTH = 600
DEFAULT_HEIGHT = 400
DEFAULT_FOV = 60
DEFAULT_BACKGROUND = "#050505"
DEFAULT_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
DEFAULT_MAX_POINTS = 0

VIEWER_STATIC_FILES = ("index.html", "viewer.js")
VIEWER_LIB_DIR = "lib"
CONFIG_JSON = "viewer-config.json"
CONFIG_MODULE = "viewer-config.js"
PATH_JSON = "path.json"

CameraPose = Tuple[np.ndarray, np.ndarray]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _viewer_template_dir() -> Path:
    template = _project_root() / "spark-viewer"
    if not template.exists():
        raise FileNotFoundError(f"Spark viewer template missing: {template}")
    return template


def _viewer_target(path: Path) -> Path:
    return path if path.is_dir() else path.with_suffix("")


def _ensure_static_assets(viewer_dir: Path) -> None:
    template = _viewer_template_dir()
    viewer_dir.mkdir(parents=True, exist_ok=True)
    for filename in VIEWER_STATIC_FILES:
        if (template / filename).exists():
            shutil.copy2(template / filename, viewer_dir / filename)
            
    # Copy lib directory if it exists
    lib_src = template / VIEWER_LIB_DIR
    lib_dst = viewer_dir / VIEWER_LIB_DIR
    if lib_src.exists():
        if lib_dst.exists():
            shutil.rmtree(lib_dst)
        shutil.copytree(lib_src, lib_dst)


def _config_json_path(viewer_dir: Path) -> Path:
    return viewer_dir / CONFIG_JSON


def _config_module_path(viewer_dir: Path) -> Path:
    return viewer_dir / CONFIG_MODULE


def _load_config(viewer_dir: Path) -> dict:
    path = _config_json_path(viewer_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_config_files(viewer_dir: Path, config: dict) -> None:
    json_path = _config_json_path(viewer_dir)
    json_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    module_path = _config_module_path(viewer_dir)
    module_path.write_text(
        "export default " + json.dumps(config, indent=2) + ";\n",
        encoding="utf-8",
    )


def _ensure_camera_pose(pose: CameraPose) -> CameraPose:
    position = np.asarray(pose[0], dtype=np.float64)
    target = np.asarray(pose[1], dtype=np.float64)
    return position, target


def _pose_dict(pose: CameraPose) -> dict:
    position, target = _ensure_camera_pose(pose)
    return {
        "position": [float(v) for v in position.tolist()],
        "target": [float(v) for v in target.tolist()],
        "up": [float(v) for v in DEFAULT_UP.tolist()],
    }


def _infer_vertex_count(ply_path: Path) -> Optional[int]:
    try:
        with ply_path.open("rb") as handle:
            for raw in handle:
                line = raw.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex"):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            return int(parts[2])
                        except ValueError:
                            return None
                if line == "end_header":
                    break
    except OSError:
        return None
    return None


def _scene_center_radius(ply_path: Path, max_points: int = 4096) -> Tuple[np.ndarray, float]:
    points, *_ = load_gaussians(ply_path, max_points=max_points)
    if len(points) == 0:
        return np.zeros(3, dtype=np.float64), 1.0
    center = np.mean(points, axis=0)
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))
    return center, max(radius, 1.0)


def load_gaussians(
    ply_path: Path,
    max_points: int = DEFAULT_MAX_POINTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Utility sampled by explorer/path planner."""
    try:
        ply = PlyData.read(str(ply_path))
    except Exception:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if "vertex" not in ply:
        return np.array([]), np.array([]), np.array([]), np.array([])
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        return np.array([]), np.array([]), np.array([]), np.array([])

    if max_points and max_points > 0 and len(vertex) > max_points:
        step = max(1, len(vertex) // max_points)
        indices = np.arange(0, len(vertex), step, dtype=np.int64)[:max_points]
        vertex = vertex[indices]

    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)

    # Simplified color loading (skip for now as we just need points mostly)
    colors = np.zeros((len(points), 3), dtype=np.float32)
    rotations = np.zeros((len(points), 4), dtype=np.float32)
    scales = np.zeros((len(points), 3), dtype=np.float32)
    return points, colors, rotations, scales


def _default_pose(ply_path: Path, camera_pose: Optional[CameraPose]) -> CameraPose:
    if camera_pose is not None:
        return _ensure_camera_pose(camera_pose)
    return np.array([0.0, 0.0, 3.0], dtype=np.float64), np.zeros(3, dtype=np.float64)


def render_scene(
    *,
    ply_path: Path,
    output_path: Path,
    camera_pose: Optional[CameraPose] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    background: str = DEFAULT_BACKGROUND,
    fov: int = DEFAULT_FOV,
) -> Path:
    viewer_dir = _viewer_target(output_path)
    _ensure_static_assets(viewer_dir)

    dest_model = viewer_dir / "scene.ply"
    # dest_model = viewer_dir / ply_path.name
    # Using fixed name 'scene.ply' to match viewer default
    
    if ply_path.resolve() != dest_model.resolve():
        shutil.copy2(ply_path, dest_model)

    center, radius = _scene_center_radius(ply_path)
    pose = _default_pose(ply_path, camera_pose)
    vertex_count = _infer_vertex_count(ply_path)

    config = {
        "title": ply_path.stem.replace("_", " "),
        "background": background,
        "model": {
            "url": "./scene.ply",
            "fileType": "ply",
            "original": ply_path.name,
        },
        "camera": {
            "fov": fov,
            "pose": _pose_dict(pose),
        },
        "viewport": {"width": width, "height": height},
        "scene": {
            "pointCount": vertex_count,
            "sceneCenter": [float(v) for v in center.tolist()],
            "sceneRadius": radius,
            "source": str(ply_path),
        },
    }

    _write_config_files(viewer_dir, config)
    return viewer_dir / "index.html"


def write_path_json(output_dir: Path, path_data: dict) -> None:
    """Write the path.json expected by the new viewer."""
    viewer_dir = _viewer_target(output_dir)
    path_file = viewer_dir / PATH_JSON
    path_file.write_text(json.dumps(path_data, indent=2), encoding="utf-8")


def render_camera_traversal(
    *,
    ply_path: Path,
    output_dir: Path,
    path_data: Optional[dict] = None,
    **kwargs: Any
) -> Path:
    """
    Sets up the viewer with the given path data.
    """
    viewer_dir = _viewer_target(output_dir)
    _ensure_static_assets(viewer_dir)

    # Ensure scene exists
    render_scene(ply_path=ply_path, output_path=output_dir)

    if path_data:
        write_path_json(output_dir, path_data)
    
    return viewer_dir / "index.html"
