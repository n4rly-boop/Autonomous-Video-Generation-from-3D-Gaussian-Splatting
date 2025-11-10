"""Video rendering orchestrated from 3D Gaussian splats."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence


def render_tour(output_path: Path, trajectory: Sequence[str]) -> Path:
    """Render a video along the provided camera trajectory."""
    # TODO: Interface with Gaussian splatting renderer and video encoder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        output_path.touch()
    return output_path
