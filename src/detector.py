"""Object detection helpers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Detection:
    label: str
    confidence: float
    position: tuple[float, float, float]


def detect_objects(frames: Iterable[bytes]) -> List[Detection]:
    """Run detection over rendered frames and return structured outputs."""
    # TODO: Plug in actual detector (e.g., 2D CNN, 3D point-based model)
    _ = frames
    return []
