from pathlib import Path
import numpy as np
from renderer import load_gaussians
pts, *_ = load_gaussians(Path('input-data/ConferenceHall.ply'), max_points=800000)
print('center', pts.mean(axis=0))
print('min', pts.min(axis=0))
print('max', pts.max(axis=0))
