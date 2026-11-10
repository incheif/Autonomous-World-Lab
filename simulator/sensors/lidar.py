"""
32-Beam LiDAR Sensor Simulator for Autonomous-World-Lab.
Implements 3D ray-casting, ray-OBB obstacle intersections, ground plane returns,
occlusion modeling, configurable Gaussian range noise, beam dropouts, and reproducible seeds.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from autonomy.interfaces.types import VehicleState, Obstacle, LidarScan, LidarObservation


@dataclass
class LidarParameters:
    """
    Configuration parameters for simulated 32-beam LiDAR.
    All angular units in radians internally, distances in metres.
    """
    num_channels: int = 32                      # Number of vertical laser rings (e.g. 32-beam)
    v_fov_min_rad: float = math.radians(-20.0)  # Lower vertical angle (-20 deg)
    v_fov_max_rad: float = math.radians(+10.0)  # Upper vertical angle (+10 deg)
    h_fov_rad: float = math.radians(360.0)      # Horizontal field of view (360 deg or forward arc)
    h_resolution_rad: float = math.radians(1.0) # Horizontal angular step (1.0 deg = 360 steps)
    min_range_m: float = 0.5                    # Minimum detection distance [m]
    max_range_m: float = 65.0                   # Maximum laser range [m]
    range_noise_std_m: float = 0.02             # Range measurement Gaussian noise sigma [m] (2 cm)
    dropout_prob: float = 0.015                 # Probability of missing beam / absorbed return
    ground_detection: bool = True               # Whether ground plane (Z=0) returns laser points
    mount_offset_m: Tuple[float, float, float] = (1.40, 0.0, 1.60) # Roof mount relative to rear axle (x, y, z)


class LidarSensor:
    """
    Configurable 3D LiDAR Sensor.
    Performs physical raycasting against obstacles and ground, strictly separating
    Ground Truth observations from Noisy observations.
    """

    def __init__(self, params: LidarParameters = LidarParameters(), seed: Optional[int] = 42):
        self.params: LidarParameters = params
        self.rng: np.random.RandomState = np.random.RandomState(seed)
        self._init_beam_grid()

    def set_seed(self, seed: Optional[int]) -> None:
        """Sets the random seed for reproducible noise and dropouts."""
        self.rng = np.random.RandomState(seed)

    def _init_beam_grid(self) -> None:
        """Precomputes spherical beam unit vectors in sensor coordinate frame."""
        p = self.params

        # Vertical elevation rings
        elevation_angles = np.linspace(p.v_fov_min_rad, p.v_fov_max_rad, p.num_channels)

        # Horizontal azimuth steps
        half_h = p.h_fov_rad / 2.0
        num_h_steps = int(round(p.h_fov_rad / p.h_resolution_rad))
        azimuth_angles = np.linspace(-half_h, half_h, num_h_steps, endpoint=False)

        # 2D Meshgrid of (elevation, azimuth)
        el_grid, az_grid = np.meshgrid(elevation_angles, azimuth_angles, indexing="ij")
        self._beam_elevations = el_grid.flatten()
        self._beam_azimuths = az_grid.flatten()

        # Precompute local ray unit vectors (sensor frame: +X forward, +Y left, +Z up)
        cos_el = np.cos(self._beam_elevations)
        self._local_dirs = np.column_stack((
            cos_el * np.cos(self._beam_azimuths),
            cos_el * np.sin(self._beam_azimuths),
            np.sin(self._beam_elevations),
        ))

    @property
    def total_beams(self) -> int:
        """Total number of rays cast per sweep."""
        return len(self._beam_elevations)

    def scan(
        self,
        vehicle_state: VehicleState,
        obstacles: List[Obstacle],
        road_boundaries: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> LidarObservation:
        """
        Executes a 3D LiDAR sweep against the world geometry.
        Returns a LidarObservation strictly separating noisy data from ground truth.
        """
        p = self.params
        N = self.total_beams

        # 1. Compute LiDAR sensor pose in world frame
        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)
        mx, my, mz = p.mount_offset_m

        sensor_world_x = vehicle_state.x + (mx * cos_yaw - my * sin_yaw)
        sensor_world_y = vehicle_state.y + (mx * sin_yaw + my * cos_yaw)
        sensor_world_z = mz  # Road plane is at Z=0, vehicle body elevation is Z=0

        sensor_origin = np.array([sensor_world_x, sensor_world_y, sensor_world_z], dtype=np.float64)

        # Rotate local ray directions into world frame
        # Vehicle yaw rotates (X, Y) around Z
        R_yaw = np.array([
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw,  cos_yaw, 0.0],
            [0.0,      0.0,     1.0],
        ])
        world_dirs = (R_yaw @ self._local_dirs.T).T  # Shape: (N, 3)

        # Initialize ground-truth ranges to max_range + 1.0 (unhit)
        gt_ranges = np.full(N, p.max_range_m + 1.0, dtype=np.float64)
        gt_intensities = np.zeros(N, dtype=np.float64)

        # 2. Ray-Ground Plane Intersection (Z = 0)
        if p.ground_detection:
            dz = world_dirs[:, 2]
            downward_mask = dz < -1e-4
            if np.any(downward_mask):
                # t = (0 - sensor_z) / dz
                t_ground = -sensor_origin[2] / dz[downward_mask]
                valid_g = (t_ground >= p.min_range_m) & (t_ground <= p.max_range_m)

                indices = np.where(downward_mask)[0][valid_g]
                t_valid = t_ground[valid_g]

                # Update if closer
                closer = t_valid < gt_ranges[indices]
                gt_ranges[indices[closer]] = t_valid[closer]
                gt_intensities[indices[closer]] = 0.35  # Asphalt retro-reflectivity factor

        # 3. Ray-Obstacle Intersections (Oriented Bounding Boxes)
        for obs in obstacles:
            t_hits, hit_mask = self._intersect_obb(sensor_origin, world_dirs, obs)
            if np.any(hit_mask):
                closer = t_hits < gt_ranges[hit_mask]
                hit_indices = np.where(hit_mask)[0][closer]
                gt_ranges[hit_indices] = t_hits[closer]
                gt_intensities[hit_indices] = 0.85  # Solid obstacle reflection

        # 4. Form Ground Truth Scan (Exact, Noise-Free, No Dropouts)
        gt_valid = (gt_ranges >= p.min_range_m) & (gt_ranges <= p.max_range_m)
        gt_scan = LidarScan(
            ranges=gt_ranges.copy(),
            angles_azimuth=self._beam_azimuths.copy(),
            angles_elevation=self._beam_elevations.copy(),
            intensities=gt_intensities.copy(),
            valid_mask=gt_valid,
            timestamp=vehicle_state.timestamp,
        )

        # 5. Form Noisy Observation (Gaussian Range Noise + Stochastic Beam Dropout)
        noisy_ranges = gt_ranges.copy()

        # Add Gaussian range noise to valid hits
        if p.range_noise_std_m > 0.0 and np.any(gt_valid):
            noise = self.rng.normal(0.0, p.range_noise_std_m, size=np.sum(gt_valid))
            noisy_ranges[gt_valid] += noise
            noisy_ranges[gt_valid] = np.maximum(p.min_range_m, noisy_ranges[gt_valid])

        # Apply stochastic beam dropouts
        noisy_valid = gt_valid.copy()
        if p.dropout_prob > 0.0:
            dropout_mask = self.rng.uniform(0.0, 1.0, size=N) < p.dropout_prob
            noisy_valid[dropout_mask] = False
            noisy_ranges[dropout_mask] = p.max_range_m + 1.0

        noisy_scan = LidarScan(
            ranges=noisy_ranges,
            angles_azimuth=self._beam_azimuths.copy(),
            angles_elevation=self._beam_elevations.copy(),
            intensities=gt_intensities.copy(),
            valid_mask=noisy_valid,
            timestamp=vehicle_state.timestamp,
        )

        return LidarObservation(noisy_scan=noisy_scan, ground_truth_scan=gt_scan)

    @staticmethod
    def _intersect_obb(
        ray_origin: np.ndarray,
        ray_dirs: np.ndarray,
        obs: Obstacle,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast vectorized 3D Ray-Oriented Bounding Box intersection (Kay-Kajiya slab method).
        Transforms rays into obstacle's local coordinate frame where it is an axis-aligned box.
        """
        cos_o = math.cos(obs.yaw)
        sin_o = math.sin(obs.yaw)

        # Vector from obstacle center to ray origin
        # Obstacle base is at Z=obs.z, center is at Z=obs.z + height/2
        obs_center = np.array([obs.x, obs.y, obs.z + obs.height / 2.0], dtype=np.float64)
        rel_origin = ray_origin - obs_center

        # Transform origin and directions into obstacle local frame (inverse rotation around Z)
        # local_x = rel_x * cos + rel_y * sin
        # local_y = -rel_x * sin + rel_y * cos
        # local_z = rel_z
        orig_lx = rel_origin[0] * cos_o + rel_origin[1] * sin_o
        orig_ly = -rel_origin[0] * sin_o + rel_origin[1] * cos_o
        orig_lz = rel_origin[2]

        dir_lx = ray_dirs[:, 0] * cos_o + ray_dirs[:, 1] * sin_o
        dir_ly = -ray_dirs[:, 0] * sin_o + ray_dirs[:, 1] * cos_o
        dir_lz = ray_dirs[:, 2]

        # Box half-extents
        hx = obs.length / 2.0
        hy = obs.width / 2.0
        hz = obs.height / 2.0

        def _slab_axis(orig_scalar: float, dir_vec: np.ndarray, half_extent: float):
            eps = 1e-7
            parallel = np.abs(dir_vec) <= eps
            safe_dir = np.where(parallel, 1.0, dir_vec)
            t1 = (-half_extent - orig_scalar) / safe_dir
            t2 = (+half_extent - orig_scalar) / safe_dir
            t_min = np.minimum(t1, t2)
            t_max = np.maximum(t1, t2)

            inside = (-half_extent <= orig_scalar) and (orig_scalar <= half_extent)
            t_min = np.where(parallel, -1e9 if inside else 1e9, t_min)
            t_max = np.where(parallel, 1e9 if inside else -1e9, t_max)
            return t_min, t_max

        t_min_x, t_max_x = _slab_axis(orig_lx, dir_lx, hx)
        t_min_y, t_max_y = _slab_axis(orig_ly, dir_ly, hy)
        t_min_z, t_max_z = _slab_axis(orig_lz, dir_lz, hz)

        t_enter = np.maximum(np.maximum(t_min_x, t_min_y), t_min_z)
        t_exit = np.minimum(np.minimum(t_max_x, t_max_y), t_max_z)

        # Intersection condition: t_enter <= t_exit and t_exit > 0.1
        hit_mask = (t_enter <= t_exit) & (t_exit > 0.1)

        t_hits = np.where(t_enter > 0.1, t_enter, t_exit)
        return t_hits[hit_mask], hit_mask

    def save_scan_npz(self, scan: LidarScan, file_path: str) -> None:
        """Saves a scan to a compressed NPZ file for offline replay."""
        np.savez_compressed(
            file_path,
            ranges=scan.ranges,
            angles_azimuth=scan.angles_azimuth,
            angles_elevation=scan.angles_elevation,
            intensities=scan.intensities,
            valid_mask=scan.valid_mask,
            timestamp=scan.timestamp,
        )

    @staticmethod
    def load_scan_npz(file_path: str) -> LidarScan:
        """Loads a pre-recorded scan from a compressed NPZ file."""
        data = np.load(file_path)
        return LidarScan(
            ranges=data["ranges"],
            angles_azimuth=data["angles_azimuth"],
            angles_elevation=data["angles_elevation"],
            intensities=data["intensities"],
            valid_mask=data["valid_mask"],
            timestamp=float(data["timestamp"]),
        )
