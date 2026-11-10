"""
Oriented Bounding Box (OBB) estimation for 3D point clusters.
Calculates minimum-area oriented bounding rectangles and vertical extents.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from autonomy.interfaces.types import DetectedObstacle


@dataclass
class BoxFitterConfig:
    """
    Configuration for bounding box estimation.
    """
    min_dimension: float = 0.20   # Minimum extent along any dimension [m]
    max_dimension: float = 25.0   # Maximum extent [m]
    default_confidence: float = 0.90


class BoxFitter:
    """
    Fits 3D Oriented Bounding Boxes (OBB) to segmented point clusters.
    Uses 2D Convex Hull and rotating edge calipers to minimize projected area in XY.
    """

    def __init__(self, config: Optional[BoxFitterConfig] = None):
        self.config = config or BoxFitterConfig()

    def fit_boxes(
        self,
        clusters: List[np.ndarray],
        timestamp: float = 0.0,
    ) -> List[DetectedObstacle]:
        """
        Fits bounding boxes to all input clusters.

        Args:
            clusters: List of point arrays of shape (K, 3)
            timestamp: Sensor observation timestamp

        Returns:
            List of DetectedObstacle instances
        """
        obstacles: List[DetectedObstacle] = []

        for idx, cluster in enumerate(clusters):
            if len(cluster) < 3:
                continue

            obstacle = self.fit_single_box(cluster, obstacle_id=f"obs_{idx+1}", timestamp=timestamp)
            if obstacle is not None:
                obstacles.append(obstacle)

        return obstacles

    def fit_single_box(
        self,
        cluster: np.ndarray,
        obstacle_id: str = "obs_1",
        timestamp: float = 0.0,
    ) -> Optional[DetectedObstacle]:
        """
        Fits an OBB to a single cluster of 3D points.
        """
        if len(cluster) < 2:
            return None

        # Vertical bounds
        z_min = float(np.min(cluster[:, 2]))
        z_max = float(np.max(cluster[:, 2]))
        height = max(z_max - z_min, self.config.min_dimension)
        z_center = (z_min + z_max) / 2.0

        # Horizontal 2D points (XY)
        pts_2d = cluster[:, :2]

        # Find best orientation via Convex Hull or PCA fallback
        center_x, center_y, length, width, yaw = self._fit_min_area_rectangle(pts_2d)

        # Sanity check dimensions
        if length > self.config.max_dimension or width > self.config.max_dimension:
            return None

        length = max(length, self.config.min_dimension)
        width = max(width, self.config.min_dimension)

        # Confidence based on point count
        confidence = float(np.clip(len(cluster) / 25.0, 0.4, 0.99))

        return DetectedObstacle(
            obstacle_id=obstacle_id,
            x=center_x,
            y=center_y,
            z=z_center,
            length=length,
            width=width,
            height=height,
            yaw=yaw,
            confidence=confidence,
            num_points=len(cluster),
            timestamp=timestamp,
        )

    def _fit_min_area_rectangle(self, pts_2d: np.ndarray) -> tuple[float, float, float, float, float]:
        """
        Fits minimum area rectangle to 2D points.
        Returns: (center_x, center_y, length, width, yaw) where length >= width.
        """
        try:
            hull = ConvexHull(pts_2d)
            hull_points = pts_2d[hull.vertices]
        except QhullError:
            hull_points = pts_2d

        if len(hull_points) < 3:
            # Fallback to axis-aligned bounding box
            min_xy = np.min(pts_2d, axis=0)
            max_xy = np.max(pts_2d, axis=0)
            center = (min_xy + max_xy) / 2.0
            dim_x = max(max_xy[0] - min_xy[0], self.config.min_dimension)
            dim_y = max(max_xy[1] - min_xy[1], self.config.min_dimension)
            if dim_x >= dim_y:
                return float(center[0]), float(center[1]), float(dim_x), float(dim_y), 0.0
            else:
                return float(center[0]), float(center[1]), float(dim_y), float(dim_x), np.pi / 2.0

        # Test edge orientations from the convex hull
        min_area = float("inf")
        best_rect = (0.0, 0.0, 1.0, 1.0, 0.0)

        n_verts = len(hull_points)
        for i in range(n_verts):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % n_verts]
            edge = p2 - p1
            edge_norm = np.linalg.norm(edge)
            if edge_norm < 1e-5:
                continue

            # Unit vectors along and perpendicular to edge
            u = edge / edge_norm
            v = np.array([-u[1], u[0]])

            # Project all hull points onto u and v
            proj_u = np.dot(hull_points, u)
            proj_v = np.dot(hull_points, v)

            min_u, max_u = np.min(proj_u), np.max(proj_u)
            min_v, max_v = np.min(proj_v), np.max(proj_v)

            extent_u = max_u - min_u
            extent_v = max_v - min_v
            area = extent_u * extent_v

            if area < min_area:
                min_area = area
                center_u = (min_u + max_u) / 2.0
                center_v = (min_v + max_v) / 2.0
                center_pt = center_u * u + center_v * v

                heading = float(np.arctan2(u[1], u[0]))
                if extent_u >= extent_v:
                    best_rect = (float(center_pt[0]), float(center_pt[1]), float(extent_u), float(extent_v), heading)
                else:
                    best_rect = (float(center_pt[0]), float(center_pt[1]), float(extent_v), float(extent_u), float((heading + np.pi / 2.0) % np.pi))

        return best_rect
