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
    length: float  # Extent along obstacle heading [m]
    width: float   # Extent perpendicular to heading [m]
    height: float  # Vertical height [m]
    yaw: float = 0.0
    velocity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.obstacle_id,
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "length": round(self.length, 4),
            "width": round(self.width, 4),
            "height": round(self.height, 4),
            "yaw": round(self.yaw, 4),
            "velocity": round(self.velocity, 4),
        }
