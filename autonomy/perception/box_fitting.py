"""
Oriented Bounding Box (OBB) estimation for 3D point clusters.
Calculates minimum-area oriented bounding rectangles and vertical extents.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
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

        # Find best orientation via L-shape edge closeness fitting
        center_x, center_y, length, width, yaw = self._fit_l_shape_rectangle(pts_2d)

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

    def _fit_l_shape_rectangle(self, pts_2d: np.ndarray) -> tuple[float, float, float, float, float]:
        """
        Fits an oriented bounding rectangle using L-shape edge closeness optimization.
        Avoids diagonal corner inflation when LiDAR views only two faces of an obstacle.
        Returns: (center_x, center_y, length, width, yaw) where length >= width.
        """
        if len(pts_2d) < 3:
            min_xy = np.min(pts_2d, axis=0)
            max_xy = np.max(pts_2d, axis=0)
            center = (min_xy + max_xy) / 2.0
            dim_x = max(max_xy[0] - min_xy[0], self.config.min_dimension)
            dim_y = max(max_xy[1] - min_xy[1], self.config.min_dimension)
            if dim_x >= dim_y:
                return float(center[0]), float(center[1]), float(dim_x), float(dim_y), 0.0
            else:
                return float(center[0]), float(center[1]), float(dim_y), float(dim_x), np.pi / 2.0

        # Evaluate 45 candidate angles in [0, pi/2) (2-degree angular step)
        angles = np.linspace(0.0, np.pi / 2.0, 45, endpoint=False)
        cos_a = np.cos(angles)  # (A,)
        sin_a = np.sin(angles)  # (A,)

        # Project points onto all candidate axes simultaneously: shape (A, N)
        px = pts_2d[:, 0][np.newaxis, :]  # (1, N)
        py = pts_2d[:, 1][np.newaxis, :]  # (1, N)

        c1 = cos_a[:, np.newaxis] * px + sin_a[:, np.newaxis] * py
        c2 = -sin_a[:, np.newaxis] * px + cos_a[:, np.newaxis] * py

        c1_min = np.min(c1, axis=1, keepdims=True)
        c1_max = np.max(c1, axis=1, keepdims=True)
        c2_min = np.min(c2, axis=1, keepdims=True)
        c2_max = np.max(c2, axis=1, keepdims=True)

        # Closeness criterion: distance of each point to closest rectangle edge
        d1 = np.minimum(c1 - c1_min, c1_max - c1)
        d2 = np.minimum(c2 - c2_min, c2_max - c2)
        closeness_err = np.mean(np.minimum(d1, d2) ** 2, axis=1)

        ext1 = (c1_max - c1_min).ravel()
        ext2 = (c2_max - c2_min).ravel()
        area_tiebreak = 1e-4 * (ext1 * ext2)

        best_idx = int(np.argmin(closeness_err + area_tiebreak))
        theta = float(angles[best_idx])

        u = np.array([float(np.cos(theta)), float(np.sin(theta))])
        v = np.array([-u[1], u[0]])

        min_u = float(c1_min[best_idx, 0])
        max_u = float(c1_max[best_idx, 0])
        min_v = float(c2_min[best_idx, 0])
        max_v = float(c2_max[best_idx, 0])

        extent_u = max_u - min_u
        extent_v = max_v - min_v

        center_u = (min_u + max_u) / 2.0
        center_v = (min_v + max_v) / 2.0
        center_pt = center_u * u + center_v * v

        if extent_u >= extent_v:
            return float(center_pt[0]), float(center_pt[1]), float(extent_u), float(extent_v), theta
        else:
            return float(center_pt[0]), float(center_pt[1]), float(extent_v), float(extent_u), float((theta + np.pi / 2.0) % np.pi)

