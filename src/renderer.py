"""Spark.js web viewer renderer.

This module no longer rasterizes PNG/MP4 outputs. Instead, it prepares a static
Spark viewer (HTML + ES modules) for every scene, mirroring the public example
at https://sparkjs.dev/examples/hello-world/. The viewer consumes the same PLY
Gaussian splats produced by upstream modules and can be hosted as plain static
files.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from plyfile import PlyData

DEFAULT_WIDTH = 600
DEFAULT_HEIGHT = 400
DEFAULT_FOV = 60
DEFAULT_BACKGROUND = "#050505"
DEFAULT_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
DEFAULT_MAX_POINTS = 0

VIEWER_STATIC_FILES = ("index.html", "viewer.js")
CONFIG_JSON = "viewer-config.json"
CONFIG_MODULE = "viewer-config.js"

_COLOR_FIELDS = (
    ("red", "green", "blue"),
    ("diffuse_red", "diffuse_green", "diffuse_blue"),
    ("r", "g", "b"),
)
_SH_COLOR_FIELDS = ("f_dc_0", "f_dc_1", "f_dc_2")

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
        shutil.copy2(template / filename, viewer_dir / filename)


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
    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError(f"{ply_path} lacks 'vertex' element.")
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("Vertex element missing x/y/z coordinates.")

    if max_points and max_points > 0 and len(vertex) > max_points:
        step = max(1, len(vertex) // max_points)
        indices = np.arange(0, len(vertex), step, dtype=np.int64)[:max_points]
        vertex = vertex[indices]

    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)

    colors = np.ones((len(points), 3), dtype=np.float32)
    color_fields = next((fields for fields in _COLOR_FIELDS if set(fields).issubset(names)), None)
    if color_fields is not None:
        cols = np.stack([vertex[color_fields[0]], vertex[color_fields[1]], vertex[color_fields[2]]], axis=1).astype(np.float32)
        if cols.max() > 1.0 + 1e-3:
            cols /= 255.0
        colors = np.clip(cols, 0.0, 1.0)
    elif set(_SH_COLOR_FIELDS).issubset(names):
        sh = np.stack([vertex[field] for field in _SH_COLOR_FIELDS], axis=1).astype(np.float32)
        colors = 1.0 / (1.0 + np.exp(-sh))

    rotations = np.zeros((len(points), 4), dtype=np.float32)
    rotations[:, 0] = 1.0
    scales = np.ones((len(points), 3), dtype=np.float32) * 0.01
    return points, colors, rotations, scales


def _default_pose(ply_path: Path, camera_pose: Optional[CameraPose]) -> CameraPose:
    if camera_pose is not None:
        return _ensure_camera_pose(camera_pose)
    fallback = _fallback_camera_path(ply_path, num_frames=48)
    if fallback:
        return fallback[0]
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

    dest_model = viewer_dir / ply_path.name
    if ply_path.resolve() != dest_model.resolve():
        shutil.copy2(ply_path, dest_model)

    center, radius = _scene_center_radius(ply_path)
    pose = _default_pose(ply_path, camera_pose)
    vertex_count = _infer_vertex_count(ply_path)

    config = {
        "title": ply_path.stem.replace("_", " "),
        "background": background,
        "model": {
            "url": f"./{dest_model.name}",
            "fileType": ply_path.suffix.lstrip(".") or "ply",
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


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


def _interpolate_camera_path(camera_path: Sequence[CameraPose], num_frames: int) -> List[CameraPose]:
    if not camera_path:
        return []
    if len(camera_path) == 1:
        pose = _ensure_camera_pose(camera_path[0])
        return [pose] * num_frames

    segments = len(camera_path) - 1
    base = num_frames // segments
    remainder = num_frames % segments

    interpolated: List[CameraPose] = []
    for idx in range(segments):
        start = _ensure_camera_pose(camera_path[idx])
        end = _ensure_camera_pose(camera_path[idx + 1])
        steps = max(1, base + (1 if idx < remainder else 0))
        for step in range(steps):
            t = step / steps
            interpolated.append((_lerp(start[0], end[0], t), _lerp(start[1], end[1], t)))
    interpolated.append(_ensure_camera_pose(camera_path[-1]))
    return interpolated[:num_frames]


def _fallback_camera_path(ply_path: Path, num_frames: int) -> List[CameraPose]:
    center, radius = _scene_center_radius(ply_path)
    height = radius * 0.25
    if radius <= 0:
        return []
    poses: List[CameraPose] = []
    for theta in np.linspace(0.0, 2.0 * np.pi, max(num_frames, 12), endpoint=False):
        offset = np.array(
            [
                np.cos(theta) * radius * 1.5,
                height,
                np.sin(theta) * radius * 1.5,
            ],
            dtype=np.float64,
        )
        position = center + offset
        poses.append((position, center.copy()))
    return poses


def _serialize_camera_path(camera_path: Sequence[CameraPose]) -> List[dict]:
    serialized = []
    for pose in camera_path:
        serialized.append(_pose_dict(pose))
    return serialized


def render_camera_traversal(
    *,
    ply_path: Path,
    output_dir: Path,
    num_frames: int = 150,
    fps: int = 10,
    camera_path: Optional[Sequence[CameraPose]] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Path:
    viewer_dir = _viewer_target(output_dir)
    _ensure_static_assets(viewer_dir)

    config = _load_config(viewer_dir)
    if not config:
        # Initialize the viewer with default metadata if render_scene has not been called.
        render_scene(
            ply_path=ply_path,
            output_path=viewer_dir,
            camera_pose=None,
            width=width,
            height=height,
        )
        config = _load_config(viewer_dir)

    path = list(camera_path or [])
    if not path:
        path = _fallback_camera_path(ply_path, max(num_frames, 32))
    interpolated = _interpolate_camera_path(path, num_frames)
    if not interpolated:
        raise RuntimeError("Unable to create camera path for viewer.")

    config["cameraPath"] = {
        "poses": _serialize_camera_path(interpolated),
        "fps": fps,
        "loop": True,
    }
    if "camera" not in config or not config["camera"].get("pose"):
        config.setdefault("camera", {})["pose"] = _pose_dict(interpolated[0])

    _write_config_files(viewer_dir, config)
    return viewer_dir / "index.html"
