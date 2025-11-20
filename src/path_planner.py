"""Path planning utilities for smooth camera trajectories."""
from typing import Iterable, Sequence, List, Dict, Any
import numpy as np
from pathlib import Path
from plyfile import PlyData
import scipy.spatial
from sklearn.cluster import DBSCAN
import networkx as nx
import open3d as o3d

def optimize_path(waypoints: Iterable[str]) -> Sequence[str]:
    """Refine coarse waypoints into an executable trajectory."""
    return list(waypoints)

def load_ply_points(ply_path: Path, max_points: int = 100000) -> np.ndarray:
    try:
        ply = PlyData.read(str(ply_path))
        vertex = ply['vertex']
        props = [p.name for p in vertex.properties]
        if 'x' in props and 'y' in props and 'z' in props:
            pts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
        else:
            return np.array([])
            
        # Filter NaNs and Infs
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        
        # Clamp extreme values to avoid covariance overflow
        # A typical scene is within +/- 10000 units
        pts = np.clip(pts, -1e5, 1e5)
        
        if len(pts) > max_points:
             pts = pts[::len(pts)//max_points]
        return pts
    except Exception:
        return np.array([])

def generate_interior_path(ply_path: Path, duration: float = 60.0, fps: int = 30) -> Dict[str, Any]:
    """
    Generates a figure-8 trajectory inside the scene boundaries.
    """
    points = load_ply_points(ply_path)
    if len(points) < 10:
        return {}
    
    # Filter extreme outliers (z-score or percentile)
    # Use 1st-99th percentile clipping to remove fly-away splats
    try:
        p_low, p_high = np.percentile(points, [1, 99], axis=0)
    except Exception:
        return {}

    mask = np.all((points >= p_low) & (points <= p_high), axis=1)
    points = points[mask]
    
    if len(points) < 10:
        return {}

    # PCA for orientation
    center = np.mean(points, axis=0)
    shifted = points - center
    
    # Double check finite
    if not np.isfinite(shifted).all():
        return {}

    try:
        cov = np.cov(shifted, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
    except Exception:
        # Fallback to identity if PCA fails
        vals = np.array([1, 1, 1])
        vecs = np.eye(3)
        
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    
    # Sanitize vectors
    vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Major axes
    axis_1 = vecs[:, 0] # Longest
    axis_2 = vecs[:, 1] # Second longest
    
    # Normalize axes just in case
    norm1 = np.linalg.norm(axis_1)
    if norm1 > 1e-6: axis_1 /= norm1
    else: axis_1 = np.array([1.0, 0.0, 0.0])
        
    norm2 = np.linalg.norm(axis_2)
    if norm2 > 1e-6: axis_2 /= norm2
    else: axis_2 = np.array([0.0, 0.0, 1.0])
    
    # Filter floor height
    y_coords = points[:, 1]
    floor_y = np.percentile(y_coords, 5)
    ceiling_y = np.percentile(y_coords, 95)
    scene_height = ceiling_y - floor_y
    
    # Eye level
    camera_y = floor_y + max(scene_height * 0.35, 1.0)
    
    # Project onto major plane (Axis 1 & 2) to find extent
    proj_1 = np.dot(shifted, axis_1)
    proj_2 = np.dot(shifted, axis_2)
    
    # Handle empty projections
    if len(proj_1) == 0: return {}
    
    p1_min, p1_max = np.percentile(proj_1, [20, 80])
    p2_min, p2_max = np.percentile(proj_2, [20, 80])
    
    scale_1 = (p1_max - p1_min) * 0.45
    scale_2 = (p2_max - p2_min) * 0.45
    
    # Sanity check scales
    if scale_1 <= 0.1 or not np.isfinite(scale_1): scale_1 = 1.0
    if scale_2 <= 0.1 or not np.isfinite(scale_2): scale_2 = 1.0
    
    frames = []
    total_frames = int(duration * fps)
    
    for i in range(total_frames):
        t = (i / total_frames) * 2 * np.pi
        # path as 8 
        u = np.sin(t)
        v = np.sin(2 * t) * 0.5
        
        pos = center + (axis_1 * u * scale_1) + (axis_2 * v * scale_2)


        # # straight forward path 
        # p1 = np.array([x1, y1, z1])   # первая точка
        # p2 = np.array([x2, y2, z2])   # вторая точка
        # pos = (1 - t) * p1 + t * p2

        pos[1] = camera_y
        
        du = np.cos(t)
        dv = np.cos(2 * t)
        tangent = (axis_1 * du * scale_1) + (axis_2 * dv * scale_2)
        
        look_at = pos + tangent
        look_at[1] = camera_y
        
        frames.append({
            "t": i / fps,
            "position": pos.tolist(),
            "look_at": look_at.tolist()
        })
        
    return {
        "fps": fps,
        "duration_sec": duration,
        "frames": frames
    }

def point_in_hull(point, hull):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= 1e-12)
        for eq in hull.equations
    )

def sor_filter(points, nb_neighbors=30, std_ratio=1.5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd_clean, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    return np.asarray(pcd_clean.points)


def generate_floorplan_path_clean(ply_path: Path, duration: int, fps: int, flip_view: bool = True) -> dict:
    """
    Чистая траектория камеры вдоль внутреннего контура помещения.
    - Очистка шума
    - Выделение стен по Y
    - Проекция в XZ
    - ConvexHull + сжатие внутрь
    - Плавная равномерная траектория
    - flip_view=True переворачивает камеру на 180°
    - camera_height и look_up_offset делают камеру выше пола
    """
    points = load_ply_points(ply_path)
    if len(points) < 10:
        return {}
    
    # дополнительная сортировка основываясь на соседних точкахх и радиусе этой точки 
    points=sor_filter(points=points)


    # ================================
    # 1. Строгая фильтрация шума (5-95 процентили по XYZ)
    # ================================
    low, high = np.percentile(points, [25,85], axis=0)
    mask = np.all((points >= low) & (points <= high), axis=1)
    points = points[mask]
    if len(points) < 10:
        return {}

    # ================================
    # 2. Выделяем точки стен (Y)
    # ================================
    y_low, y_high = np.percentile(points[:, 1], [20, 40])
    wall_mask = (points[:, 1] >= y_low) & (points[:, 1] <= y_high)
    wall_points = points[wall_mask]
    if len(wall_points) < 10:
        return {}

    proj_points = wall_points[:, [0, 2]]  # XZ

    # ================================
    # 3. Строим Hull и слегка сжимаем внутрь
    # ================================
    try:
        hull = scipy.spatial.ConvexHull(proj_points)
        contour = proj_points[hull.vertices]
    except Exception:
        return {}
    contour = np.vstack([contour, contour[0]])  # закрываем контур
    center = np.mean(contour[:-1], axis=0)
    contour = center + (contour - center) * 0.7  # сжатие внутрь

    # ================================
    # 4. Расчет сегментов и равномерные расстояния
    # ================================
    segment_lengths = np.linalg.norm(np.diff(contour, axis=0), axis=1)
    cumulative = np.cumsum(segment_lengths)
    total_length = cumulative[-1]
    total_frames = int(fps * duration)
    target_distances = np.linspace(0, total_length, total_frames)

    # ================================
    # 5. Интерполяция точек по сегментам
    # ================================
    def interpolate(distance: float):
        idx = np.searchsorted(cumulative, distance)
        if idx == 0:
            return contour[0]
        if idx >= len(contour) - 1:
            return contour[-1]
        seg_start = contour[idx]
        seg_end = contour[idx + 1]
        seg_len = segment_lengths[idx]
        prev_dist = cumulative[idx - 1]
        frac = (distance - prev_dist) / seg_len
        return (1 - frac) * seg_start + frac * seg_end

    # ================================
    # 6. Высота камеры и взгляд выше пола
    # ================================
    y_coords = points[:, 1]
    floor_y, ceiling_y = np.percentile(y_coords, [5, 95])
    scene_height = ceiling_y - floor_y

    # Камера на уровне глаз (или минимум 1 м)
    camera_height = floor_y + max(scene_height * 0.35, 1)
    # Вектор взгляда немного вверх
    look_up_offset = scene_height * 0.08  # 8% высоты сцены

    # ================================
    # 7. Построение кадров
    # ================================
    frames = []
    min_camera_height = floor_y + 1.6
    for i in range(total_frames):
        dist = target_distances[i]
        pos2d = interpolate(dist)
        next_pos2d = interpolate(dist + 0.1)
        tangent = next_pos2d - pos2d
        norm = np.linalg.norm(tangent)
        if norm > 1e-6:
            tangent /= norm
        else:
            tangent = np.array([1.0, 0.0])

        # Высота камеры: не ниже min_camera_height
        position = np.array([pos2d[0], max(camera_height, min_camera_height), pos2d[1]])

        # Флаг flip_view влияет только на XZ-направление
        look_dir = np.array([tangent[0], look_up_offset, tangent[1]])
        if flip_view:
            look_dir[0] *= -1
            look_dir[2] *= -1

        look_at = position + look_dir
        frames.append({"t": i / fps, "position": position.tolist(), "look_at": look_at.tolist()})

    
    return {"fps": fps, "duration_sec": duration, "frames": frames}


import numpy as np
from pathlib import Path
from plyfile import PlyData
from skimage.morphology import binary_dilation, medial_axis, disk
import scipy.spatial
import networkx as nx
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class PointCloudPathPlanner:
    def __init__(self, ply_path: Path, voxel_size: float = 0.1):
        self.ply_path = ply_path
        self.voxel_size = voxel_size
        self.points = self._load_and_filter_points()
        self.floor_y, self.ceiling_y = self._compute_scene_height()
        # По центру стен: не берем пол и потолок
        self.wall_slice_center = self.floor_y + 0.4 * (self.ceiling_y - self.floor_y)
        self.grid = None
        self.skeleton = None
        self.graph = None

    def _load_and_filter_points(self) -> np.ndarray:
        """Загрузка PLY и удаление шума."""
        try:
            ply = PlyData.read(str(self.ply_path))
            vertex = ply['vertex']
            pts = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
        except Exception as e:
            print(f"Error loading PLY: {e}")
            return np.array([])

        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            return pts

        # Воксельная фильтрация
        quantized = np.round(pts / self.voxel_size).astype(int)
        _, unique_idx = np.unique(quantized, axis=0, return_index=True)
        pts = pts[unique_idx]

        # Фильтрация по перцентилям (1%-99%)
        low, high = np.percentile(pts, [10, 90], axis=0)
        mask = np.all((pts >= low) & (pts <= high), axis=1)
        pts = pts[mask]

        # DBSCAN для удаления шумных точек
        if len(pts) > 50:
            clustering = DBSCAN(eps=self.voxel_size*2, min_samples=5).fit(pts)
            pts = pts[clustering.labels_ != -1]

        return pts

    def _compute_scene_height(self):
        if len(self.points) == 0:
            return 0.0, 0.0
        y_coords = self.points[:, 1]
        floor_y, ceiling_y = np.percentile(y_coords, [5, 95])
        return floor_y, ceiling_y

    def generate_wall_slice_grid(self, safety_radius: float = 0.4, slice_thickness: float = 0.3):
        """
        Строим occupancy grid только для стен на горизонтальном срезе:
        - slice_thickness: толщина слоя по Y
        """
        if len(self.points) == 0:
            print("No points to build grid")
            return None

        min_y = self.wall_slice_center - slice_thickness / 2
        max_y = self.wall_slice_center + slice_thickness / 2
        slice_pts = self.points[(self.points[:,1]>=min_y) & (self.points[:,1]<=max_y)]
        if len(slice_pts) == 0:
            print("No points in wall slice")
            return None

        x_pts = slice_pts[:,0]
        z_pts = slice_pts[:,2]

        margin = 10
        resolution = 0.1
        min_x, max_x = x_pts.min(), x_pts.max()
        min_z, max_z = z_pts.min(), z_pts.max()
        width = int((max_x - min_x)/resolution) + 2*margin
        height = int((max_z - min_z)/resolution) + 2*margin

        grid = np.zeros((height, width), dtype=bool)
        px = ((x_pts - min_x)/resolution).astype(int) + margin
        pz = ((z_pts - min_z)/resolution).astype(int) + margin
        px = np.clip(px, 0, width-1)
        pz = np.clip(pz, 0, height-1)
        grid[pz, px] = True

        # Дилатация стен для безопасной зоны
        dilate_px = int(safety_radius / resolution)
        grid_dilated = binary_dilation(grid, disk(dilate_px))
        self.grid = ~grid_dilated  # свободное пространство
        self._grid_meta = {'min_x': min_x, 'min_z': min_z, 'res': resolution, 'margin': margin}
        return self.grid

    def compute_skeleton_and_graph(self):
        if self.grid is None:
            print("Grid not built yet")
            return None

        skeleton, distance_map = medial_axis(self.grid, return_distance=True)
        self.skeleton = skeleton

        # Преобразуем skeleton в граф
        y_idxs, x_idxs = np.where(skeleton)
        pixels = list(zip(y_idxs, x_idxs))
        pixel_set = set(pixels)
        G = nx.Graph()
        for y, x in pixels:
            G.add_node((y,x))
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dy==0 and dx==0: continue
                    ny, nx_ = y+dy, x+dx
                    if (ny, nx_) in pixel_set:
                        G.add_edge((y,x), (ny,nx_), weight=np.sqrt(dy**2 + dx**2))
        self.graph = G
        return self.skeleton, G

    def show_map(self):
        """Отобразить occupancy grid и скелет."""
        if self.grid is None:
            print("Grid not built")
            return
        plt.figure(figsize=(12,12))
        plt.imshow(~self.grid, cmap='gray', origin='lower')
        if self.skeleton is not None:
            y, x = np.where(self.skeleton)
            plt.plot(x, y, 'r.', markersize=2)
        plt.title("Occupancy Grid (black=obstacle) & Skeleton (red)")
        plt.show()
