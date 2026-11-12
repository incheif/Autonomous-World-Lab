"""
Data contracts and core typed interfaces for Autonomous-World-Lab.
Enforces explicit SI units: metres (m), seconds (s), radians (rad).
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass(frozen=True)
class VehicleState:
    """
    Ground-truth or estimated vehicle state referenced to center of rear axle.
    """
    x: float             # Longitudinal position [m]
    y: float             # Lateral position [m]
    yaw: float           # Heading angle counter-clockwise from +X [rad]
    velocity: float      # Longitudinal speed [m/s]
    steering_angle: float  # Front wheel steering angle [rad]
    angular_velocity: float = 0.0  # Yaw rate [rad/s]
    acceleration: float = 0.0      # Longitudinal acceleration [m/s^2]
    timestamp: float = 0.0         # Simulation timestamp [s]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "yaw": round(self.yaw, 4),
            "velocity": round(self.velocity, 4),
            "steering_angle": round(self.steering_angle, 4),
            "angular_velocity": round(self.angular_velocity, 4),
            "acceleration": round(self.acceleration, 4),
            "timestamp": round(self.timestamp, 4),
        }


@dataclass(frozen=True)
class ControlCommand:
    """
    Low-level actuator control command.
    """
    acceleration: float = 0.0     # Target acceleration (>0) or braking deceleration (<0) [m/s^2]
    steering: float = 0.0         # Commanded front wheel steering angle [rad]
    brake: float = 0.0            # Normalized braking effort [0.0 to 1.0]
    emergency_stop: bool = False  # Immediate fail-safe stop trigger
    timestamp: float = 0.0        # Command timestamp [s]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "acceleration": round(self.acceleration, 4),
            "steering": round(self.steering, 4),
            "brake": round(self.brake, 4),
            "emergency_stop": self.emergency_stop,
            "timestamp": round(self.timestamp, 4),
        }


@dataclass
class TrajectoryPoint:
    """
    Single waypoint along a planned or reference trajectory.
    """
    x: float
    y: float
    yaw: float = 0.0
    velocity: float = 0.0
    curvature: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "yaw": round(self.yaw, 4),
            "velocity": round(self.velocity, 4),
            "curvature": round(self.curvature, 4),
            "timestamp": round(self.timestamp, 4),
        }


@dataclass
class PlannedTrajectory:
    """
    Complete planned trajectory output from local or global planner.
    """
    points: List[TrajectoryPoint] = field(default_factory=list)
    timestamp: float = 0.0
    valid: bool = True
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "timestamp": round(self.timestamp, 4),
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass
class Obstacle:
    """
    Static or dynamic obstacle in the world.
    """
    obstacle_id: str
    x: float
    y: float
    z: float = 0.0 # Elevation of obstacle base [m]
    length: float = 2.0  # Extent along obstacle heading [m]
    width: float = 2.0   # Extent perpendicular to heading [m]
    height: float = 1.5  # Vertical height [m]
    yaw: float = 0.0
    velocity: float = 0.0
    obstacle_type: str = "box"  # "box", "sedan", "suv", "truck", "barrier"
    color: str = "#e11d48"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.obstacle_id,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "z": round(self.z, 4),
            "length": round(self.length, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "yaw": round(self.yaw, 4),
            "velocity": round(self.velocity, 4),
            "type": self.obstacle_type,
            "color": self.color,
        }


@dataclass
class LidarScan:
    """
    Single LiDAR sweep data packet.
    """
    ranges: np.ndarray             # Measured range array [m]
    angles_azimuth: np.ndarray     # Horizontal azimuth angles [rad]
    angles_elevation: np.ndarray   # Vertical elevation angles [rad]
    intensities: np.ndarray        # Return intensity / reflection factor [0.0 to 1.0]
    valid_mask: np.ndarray         # Boolean mask: True if valid return
    timestamp: float = 0.0         # Timestamp of scan [s]

    @property
    def points_local(self) -> np.ndarray:
        """
        Computes 3D Cartesian coordinates in sensor frame (X forward, Y left, Z up).
        Only returns valid hit points. Shape: (M, 3)
        """
        valid_ranges = self.ranges[self.valid_mask]
        valid_az = self.angles_azimuth[self.valid_mask]
        valid_el = self.angles_elevation[self.valid_mask]

        cos_el = np.cos(valid_el)
        px = valid_ranges * cos_el * np.cos(valid_az)
        py = valid_ranges * cos_el * np.sin(valid_az)
        pz = valid_ranges * np.sin(valid_el)

        return np.column_stack((px, py, pz))

    def to_dict(self, max_points: int = 1500) -> Dict[str, Any]:
        """
        Serializes scan points for WebSocket / Three.js telemetry.
        Decimates points if needed to maintain high network throughput.
        """
        pts = self.points_local
        if len(pts) > max_points:
            indices = np.linspace(0, len(pts) - 1, max_points, dtype=int)
            pts = pts[indices]

        return {
            "timestamp": round(self.timestamp, 4),
            "total_beams": int(len(self.ranges)),
            "valid_returns": int(np.sum(self.valid_mask)),
            "points": [[round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)] for p in pts],
        }


@dataclass
class LidarObservation:
    """
    Dual observation container strictly separating Ground Truth from Noisy Sensor data.
    """
    noisy_scan: LidarScan          # Passed to perception & planning (has noise & dropouts)
    ground_truth_scan: LidarScan   # Reserved strictly for evaluation & metrics


@dataclass
class DetectedObstacle:
    """
    Obstacle identified by the perception subsystem from sensor observations.
    Coordinates are in world frame (East-North-Up / ISO 8855).
    """
    obstacle_id: str
    x: float             # World X center [m]
    y: float             # World Y center [m]
    z: float             # World Z center [m]
    length: float        # Extent along heading [m]
    width: float         # Extent lateral to heading [m]
    height: float        # Vertical height [m]
    yaw: float           # Heading angle [rad]
    confidence: float    # Detection confidence [0.0 to 1.0]
    num_points: int      # Number of point cloud hits in cluster
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.obstacle_id,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "length": round(self.length, 3),
            "width": round(self.width, 3),
            "height": round(self.height, 3),
            "yaw": round(self.yaw, 3),
            "confidence": round(self.confidence, 3),
            "num_points": self.num_points,
            "timestamp": round(self.timestamp, 3),
        }


@dataclass
class OccupancyGridMap:
    """
    2D local bird's-eye-view occupancy grid and safety-inflated cost map.
    Grid values: -1 = unknown, 0 = free, 100 = occupied
    Cost values: 0 = clear, 100 = lethal collision zone
    """
    origin_x: float          # World X coordinate of cell [0, 0] [m]
    origin_y: float          # World Y coordinate of cell [0, 0] [m]
    resolution: float        # Metres per cell [m]
    width_cells: int         # Cells along X axis
    height_cells: int        # Cells along Y axis
    grid: np.ndarray         # Shape (height_cells, width_cells), int8
    cost_map: np.ndarray     # Shape (height_cells, width_cells), float32 [0.0 to 100.0]
    timestamp: float = 0.0

    def world_to_grid(self, wx: float, wy: float) -> Optional[tuple[int, int]]:
        """Converts world coordinates (wx, wy) to grid coordinates (col, row)."""
        col = int(np.floor((wx - self.origin_x) / self.resolution))
        row = int(np.floor((wy - self.origin_y) / self.resolution))
        if 0 <= col < self.width_cells and 0 <= row < self.height_cells:
            return (col, row)
        return None

    def grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        """Converts grid coordinates (col, row) to world coordinates (wx, wy) at cell center."""
        wx = self.origin_x + (col + 0.5) * self.resolution
        wy = self.origin_y + (row + 0.5) * self.resolution
        return (wx, wy)

    def to_dict(self, downsample: int = 2) -> Dict[str, Any]:
        """
        Serialize metadata and downsampled cost map for telemetry HUD.
        """
        ds_cost = self.cost_map[::downsample, ::downsample]
        return {
            "origin_x": round(self.origin_x, 2),
            "origin_y": round(self.origin_y, 2),
            "resolution": round(self.resolution * downsample, 3),
            "width": int(ds_cost.shape[1]),
            "height": int(ds_cost.shape[0]),
            "timestamp": round(self.timestamp, 3),
            # Flatten downsampled cost array as byte/int list (0-100)
            "costs": np.clip(np.round(ds_cost), 0, 100).astype(int).tolist(),
        }



