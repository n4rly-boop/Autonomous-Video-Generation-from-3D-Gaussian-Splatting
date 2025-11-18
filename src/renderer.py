"""Spark.js-backed rendering utilities.

Instead of running a heavyweight torch renderer, we invoke the Spark.js CLI to
produce stills and traversal videos (internally powered by ffmpeg). The public
functions (`render_scene` and `render_camera_traversal`) keep the signatures that
the rest of the pipeline expects so that `src/main.py` does not need to change.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from plyfile import PlyData

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FOV = 60
DEFAULT_MAX_POINTS = 0
DEFAULT_POINT_SCALE = 1600
DEFAULT_MIN_POINT_SIZE = 1.0
DEFAULT_MAX_POINT_SIZE = 12.0
DEFAULT_BACKGROUND = "#050505"
DEFAULT_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
NODE_HEAP_MB = 16_384

_COLOR_FIELDS = (
    ("red", "green", "blue"),
    ("diffuse_red", "diffuse_green", "diffuse_blue"),
    ("r", "g", "b"),
)
_SH_COLOR_FIELDS = ("f_dc_0", "f_dc_1", "f_dc_2")

CameraPose = Tuple[np.ndarray, np.ndarray]


def _spark_renderer_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "spark_ply_renderer.js"


def _spark_env() -> dict:
    env = os.environ.copy()
    extra = f"--max-old-space-size={NODE_HEAP_MB}"
    existing = env.get("NODE_OPTIONS", "")
    if extra not in existing:
        env["NODE_OPTIONS"] = (existing + " " + extra).strip()
    return env


def _format_vec(vec: np.ndarray) -> str:
    arr = np.asarray(vec, dtype=np.float64)
    return ",".join(f"{float(v):.6f}" for v in arr)


def _ensure_camera_pose(pose: CameraPose) -> CameraPose:
    position = np.asarray(pose[0], dtype=np.float64)
    target = np.asarray(pose[1], dtype=np.float64)
    return position, target


def load_gaussians(
    ply_path: Path,
    max_points: int = DEFAULT_MAX_POINTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Lightweight helper returning point samples used by explorer/path planner.
    """
    ply = PlyData.read(str(ply_path))
    element_names = [elem.name for elem in ply.elements]
    if "vertex" not in element_names:
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
        cols = 1.0 / (1.0 + np.exp(-sh))
        colors = np.clip(cols, 0.0, 1.0)

    rotations = np.zeros((len(points), 4), dtype=np.float32)
    rotations[:, 0] = 1.0

    scales = np.ones((len(points), 3), dtype=np.float32) * 0.01
    return points, colors, rotations, scales


def _call_spark_renderer(
    *,
    ply_path: Path,
    output_path: Path,
    camera_pose: Optional[CameraPose] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fov: int = DEFAULT_FOV,
    background: str = DEFAULT_BACKGROUND,
    max_points: int = DEFAULT_MAX_POINTS,
    point_scale: int = DEFAULT_POINT_SCALE,
    min_point_size: float = DEFAULT_MIN_POINT_SIZE,
    max_point_size: float = DEFAULT_MAX_POINT_SIZE,
) -> Path:
    """Invoke the Spark renderer CLI for a single frame."""
    script = _spark_renderer_path()
    if not script.exists():
        raise FileNotFoundError(f"Spark renderer script not found: {script}")

    cmd = [
        "node",
        str(script),
        "--ply",
        str(ply_path),
        "--out",
        str(output_path),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fov",
        str(fov),
        "--background",
        background,
        "--maxPoints",
        str(max_points),
        "--pointScale",
        str(point_scale),
        "--minPointSize",
        f"{min_point_size}",
        "--maxPointSize",
        f"{max_point_size}",
    ]

    if camera_pose is not None:
        position, target = _ensure_camera_pose(camera_pose)
        cmd.extend(["--camera", _format_vec(position)])
        cmd.extend(["--target", _format_vec(target)])
        cmd.extend(["--up", _format_vec(DEFAULT_UP)])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmd, check=True, env=_spark_env())
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Spark renderer failed for {ply_path} -> {output_path}"
        ) from exc
    return output_path


def render_scene(
    *,
    ply_path: Path,
    output_path: Path,
    camera_pose: Optional[CameraPose] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Path:
    """Render a still image of the scene."""
    return _call_spark_renderer(
        ply_path=ply_path,
        output_path=output_path,
        camera_pose=camera_pose,
        width=width,
        height=height,
    )


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


def _interpolate_camera_path(
    camera_path: Sequence[CameraPose], num_frames: int
) -> List[CameraPose]:
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
        steps = base + (1 if idx < remainder else 0)
        steps = max(1, steps)
        for step in range(steps):
            t = step / steps
            pos = _lerp(start[0], end[0], t)
            look = _lerp(start[1], end[1], t)
            interpolated.append((pos, look))
    interpolated.append(_ensure_camera_pose(camera_path[-1]))
    return interpolated[:num_frames]


def _fallback_camera_path(ply_path: Path, num_frames: int) -> List[CameraPose]:
    pts, _, _, _ = load_gaussians(ply_path, max_points=2048)
    if len(pts) == 0:
        center = np.zeros(3, dtype=np.float64)
        radius = 1.0
    else:
        center = np.mean(pts, axis=0)
        radius = float(np.max(np.linalg.norm(pts - center, axis=1)))
        radius = max(radius, 1.0)
    height = radius * 0.25
    path: List[CameraPose] = []
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
        target = center.copy()
        path.append((position, target))
    return path


def _serialize_camera_sequence(camera_path: Sequence[CameraPose], num_frames: int) -> List[dict]:
    poses = _interpolate_camera_path(camera_path, num_frames)
    if not poses:
        return []
    sequence = []
    for position, target in poses:
        sequence.append(
            {
                "camera": [float(v) for v in position.tolist()],
                "target": [float(v) for v in target.tolist()],
                "up": [float(v) for v in DEFAULT_UP.tolist()],
            }
        )
    return sequence


def _call_spark_video_renderer(
    *,
    ply_path: Path,
    video_path: Path,
    camera_sequence: List[dict],
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fov: int = DEFAULT_FOV,
    fps: int = 10,
    background: str = DEFAULT_BACKGROUND,
    max_points: int = DEFAULT_MAX_POINTS,
    point_scale: int = DEFAULT_POINT_SCALE,
    min_point_size: float = DEFAULT_MIN_POINT_SIZE,
    max_point_size: float = DEFAULT_MAX_POINT_SIZE,
) -> Path:
    if not camera_sequence:
        raise ValueError("Camera sequence must contain at least one pose.")

    script = _spark_renderer_path()
    if not script.exists():
        raise FileNotFoundError(f"Spark renderer script not found: {script}")

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        temp_json = Path(handle.name)
        json.dump(camera_sequence, handle)

    cmd = [
        "node",
        str(script),
        "--ply",
        str(ply_path),
        "--video",
        str(video_path),
        "--cameraSequence",
        str(temp_json),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fov",
        str(fov),
        "--fps",
        str(fps),
        "--background",
        background,
        "--maxPoints",
        str(max_points),
        "--pointScale",
        str(point_scale),
        "--minPointSize",
        f"{min_point_size}",
        "--maxPointSize",
        f"{max_point_size}",
    ]

    try:
        subprocess.run(cmd, check=True, env=_spark_env())
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Spark video renderer failed for {ply_path} -> {video_path}"
        ) from exc
    finally:
        try:
            temp_json.unlink(missing_ok=True)
        except TypeError:
            try:
                temp_json.unlink()
            except FileNotFoundError:
                pass

    return video_path


def render_camera_traversal(
    *,
    ply_path: Path,
    output_dir: Path,
    num_frames: int = 150,
    fps: int = 10,
    camera_path: Optional[Sequence[CameraPose]] = None,
) -> Path:
    """Render a traversal video by sampling camera poses along the planned path."""
    path = list(camera_path or [])
    if not path:
        path = _fallback_camera_path(ply_path, max(num_frames, 32))

    interpolated_path = _interpolate_camera_path(path, num_frames)
    if not interpolated_path:
        raise RuntimeError("Unable to derive any camera poses for traversal video.")

    sequence = _serialize_camera_sequence(interpolated_path, num_frames)
    video_path = output_dir / f"{ply_path.stem}_traversal.mp4"
    return _call_spark_video_renderer(
        ply_path=ply_path,
        video_path=video_path,
        camera_sequence=sequence,
        fps=fps,
    )
