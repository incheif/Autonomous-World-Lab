"""
Ground plane removal and point cloud filtering.
Isolates obstacle points from ground/road reflections and sensor artifacts.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from autonomy.interfaces.types import LidarScan, VehicleState


@dataclass
class GroundFilterConfig:
    """
    Configuration parameters for ground removal and point filtering.
    """
    sensor_mount_offset: Tuple[float, float, float] = (1.40, 0.0, 1.60) # Sensor mount relative to rear axle (x, y, z)
    ground_height_threshold: float = 0.20 # Points with world Z <= this are ground [m]
    min_range: float = 0.80               # Min valid range to filter vehicle self-returns [m]
    max_range: float = 65.0               # Max valid range [m]
    max_height: float = 4.0               # Points higher than this (overpasses, birds) filtered out [m]
    # Vehicle footprint in vehicle frame to filter self-reflections
    filter_vehicle_body: bool = True
    vehicle_x_min: float = -1.05
    vehicle_x_max: float = 3.45
    vehicle_y_min: float = -0.95
    vehicle_y_max: float = 0.95
    use_ransac: bool = False              # Fallback to RANSAC plane fitting on uneven terrain
    ransac_distance_threshold: float = 0.15 # Max distance to plane for inlier [m]
    ransac_iterations: int = 50           # Number of RANSAC iterations


class GroundFilter:
    """
    Filters raw LiDAR point clouds to extract non-ground obstacle candidate points.
    Transforms points into world coordinates using vehicle pose and sensor mounting geometry.
    """

    def __init__(self, config: Optional[GroundFilterConfig] = None):
        self.config = config or GroundFilterConfig()

    def filter(
        self,
        scan: LidarScan,
        vehicle_state: VehicleState,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filters a LiDAR scan and separates points into obstacle points and ground points.

        Returns:
            obstacle_points: np.ndarray of shape (N, 3) in world coordinates
            ground_points: np.ndarray of shape (M, 3) in world coordinates
        """
        pts_local = scan.points_local
        if len(pts_local) == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        # Range filter
        ranges = np.linalg.norm(pts_local, axis=1)
        valid_range_mask = (ranges >= self.config.min_range) & (ranges <= self.config.max_range)
        pts_local = pts_local[valid_range_mask]

        if len(pts_local) == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        # Transform from sensor frame to vehicle frame (origin at center of rear axle)
        mx, my, mz = self.config.sensor_mount_offset
        vx = pts_local[:, 0] + mx
        vy = pts_local[:, 1] + my
        vz = pts_local[:, 2] + mz

        # Filter points inside vehicle body
        if self.config.filter_vehicle_body:
            inside_body = (
                (vx >= self.config.vehicle_x_min)
                & (vx <= self.config.vehicle_x_max)
                & (vy >= self.config.vehicle_y_min)
                & (vy <= self.config.vehicle_y_max)
                & (vz >= 0.0)
                & (vz <= 1.80)
            )
            vx = vx[~inside_body]
            vy = vy[~inside_body]
            vz = vz[~inside_body]

        if len(vx) == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        # Transform from vehicle frame to world coordinates (ISO 8855)
        cos_yaw = np.cos(vehicle_state.yaw)
        sin_yaw = np.sin(vehicle_state.yaw)

        wx = vehicle_state.x + vx * cos_yaw - vy * sin_yaw
        wy = vehicle_state.y + vx * sin_yaw + vy * cos_yaw
        wz = vz

        pts_world = np.column_stack((wx, wy, wz))

        # Filter out points above max_height
        height_mask = pts_world[:, 2] <= self.config.max_height
        pts_world = pts_world[height_mask]

        if len(pts_world) == 0:
            return np.empty((0, 3)), np.empty((0, 3))

        if self.config.use_ransac:
            return self._ransac_filter(pts_world)
        else:
            return self._threshold_filter(pts_world)

    def _threshold_filter(self, pts_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Direct height thresholding against ground level."""
        is_ground = pts_world[:, 2] <= self.config.ground_height_threshold
        obstacle_pts = pts_world[~is_ground]
        ground_pts = pts_world[is_ground]
        return obstacle_pts, ground_pts

    def _ransac_filter(self, pts_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """RANSAC plane fitting to detect ground plane even on slight slopes."""
        # Candidate ground points: below 0.5m
        cand_mask = pts_world[:, 2] <= 0.5
        cand_indices = np.where(cand_mask)[0]

        if len(cand_indices) < 10:
            return self._threshold_filter(pts_world)

        best_inliers = None
        max_inliers = -1

        np_random = np.random.RandomState(42)  # Deterministic seed

        for _ in range(self.config.ransac_iterations):
            sample_idx = np_random.choice(cand_indices, 3, replace=False)
            p1, p2, p3 = pts_world[sample_idx]

            # Normal vector
            v1 = p2 - p1
            v2 = p3 - p1
            norm = np.cross(v1, v2)
            norm_len = np.linalg.norm(norm)
            if norm_len < 1e-6:
                continue
            norm = norm / norm_len

            # Normal should point predominantly upward (+Z)
            if abs(norm[2]) < 0.8:
                continue

            d = -np.dot(norm, p1)
            dists = np.abs(np.dot(pts_world, norm) + d)
            inliers = np.where(dists < self.config.ransac_distance_threshold)[0]

            if len(inliers) > max_inliers:
                max_inliers = len(inliers)
                best_inliers = inliers

        if best_inliers is None or max_inliers < 10:
            return self._threshold_filter(pts_world)

        is_ground = np.zeros(len(pts_world), dtype=bool)
        is_ground[best_inliers] = True
        return pts_world[~is_ground], pts_world[is_ground]
