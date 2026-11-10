"""
Road and Environment representation for Autonomous-World-Lab.
Maintains road centerline, lane boundaries, start/goal poses, and boundary collision checks.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

from autonomy.interfaces.types import Obstacle


@dataclass
class RoadSegment:
    """
    Waypoints defining the center, left boundary, and right boundary of the road.
    """
    center_x: float
    center_y: float
    left_x: float
    left_y: float
    right_x: float
    right_y: float
    yaw: float = 0.0


class RoadMap:
    """
    Represents the simulated road environment.
    Generates structured road geometry (straight, curve, or S-bend),
    boundary collision checks, and Three.js visual geometry serialization.
    """

    def __init__(
        self,
        road_type: str = "straight",
        road_length_m: float = 120.0,
        lane_width_m: float = 3.75,
        num_lanes: int = 2,
        with_obstacles: bool = True,
        obstacles: Optional[List[Obstacle]] = None,
    ):
        self.road_type: str = road_type
        self.road_length_m: float = float(road_length_m)
        self.lane_width_m: float = float(lane_width_m)
        self.num_lanes: int = int(num_lanes)
        self.total_width_m: float = self.lane_width_m * self.num_lanes
        self.with_obstacles: bool = with_obstacles

        self.segments: List[RoadSegment] = []
        self.obstacles: List[Obstacle] = []

        # Start and Goal poses (x, y, yaw)
        self.start_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.goal_pose: Tuple[float, float, float] = (self.road_length_m - 10.0, 0.0, 0.0)

        self._generate_road()
        if obstacles is not None:
            for obs in obstacles:
                self.add_obstacle(obs)
        elif self.with_obstacles:
            self._add_default_obstacles()

    def _generate_road(self) -> None:
        """Constructs discrete road segments based on configuration."""
        self.segments.clear()
        half_width = self.total_width_m / 2.0
        step_ds = 1.0  # 1 metre segment spacing
        num_steps = int(self.road_length_m / step_ds) + 1

        for i in range(num_steps):
            s = i * step_ds
            if self.road_type == "curve":
                # Gradual gentle curve
                cx = s
                cy = 8.0 * math.sin((s / self.road_length_m) * math.pi)
                # Derivative for heading
                dcy_ds = 8.0 * (math.pi / self.road_length_m) * math.cos((s / self.road_length_m) * math.pi)
                yaw = math.atan2(dcy_ds, 1.0)
            elif self.road_type == "s_curve":
                # S-curve
                cx = s
                cy = 6.0 * math.sin((2.0 * math.pi * s) / self.road_length_m)
                dcy_ds = 6.0 * ((2.0 * math.pi) / self.road_length_m) * math.cos((2.0 * math.pi * s) / self.road_length_m)
                yaw = math.atan2(dcy_ds, 1.0)
            else:
                # Straight road along +X
                cx = s
                cy = 0.0
                yaw = 0.0

            # Normal vector perpendicular to heading (to the left)
            nx = -math.sin(yaw)
            ny = math.cos(yaw)

            lx = cx + half_width * nx
            ly = cy + half_width * ny
            rx = cx - half_width * nx
            ry = cy - half_width * ny

            self.segments.append(
                RoadSegment(
                    center_x=cx,
                    center_y=cy,
                    left_x=lx,
                    left_y=ly,
                    right_x=rx,
                    right_y=ry,
                    yaw=yaw,
                )
            )

        if self.segments:
            start_seg = self.segments[0]
            self.start_pose = (start_seg.center_x, start_seg.center_y, start_seg.yaw)
            goal_idx = min(len(self.segments) - 1, int(num_steps * 0.90))
            goal_seg = self.segments[goal_idx]
            self.goal_pose = (goal_seg.center_x, goal_seg.center_y, goal_seg.yaw)

    def _add_default_obstacles(self) -> None:
        """Places realistic static obstacle boxes along the road."""
        self.obstacles.clear()
        half_lane = self.lane_width_m / 2.0

        if self.road_type == "straight":
            # Obstacle 1: Parked obstacle in right lane at X = 35m
            self.add_obstacle(Obstacle(
                obstacle_id="obs_box_1",
                x=35.0, y=-half_lane, z=0.0,
                length=3.0, width=1.8, height=1.5,
                yaw=0.0,
            ))
            # Obstacle 2: Construction barrier in left lane at X = 70m
            self.add_obstacle(Obstacle(
                obstacle_id="obs_box_2",
                x=70.0, y=half_lane, z=0.0,
                length=2.5, width=1.6, height=1.4,
                yaw=0.08,
            ))
        elif self.road_type in ("curve", "s_curve"):
            # Sample obstacles along the centerline offset
            for idx, frac in enumerate([0.30, 0.65], start=1):
                seg_idx = int(len(self.segments) * frac)
                if seg_idx < len(self.segments):
                    seg = self.segments[seg_idx]
                    # Alternate left/right lane
                    lane_sign = 1.0 if idx % 2 == 1 else -1.0
                    nx = -math.sin(seg.yaw)
                    ny = math.cos(seg.yaw)
                    ox = seg.center_x + (lane_sign * half_lane) * nx
                    oy = seg.center_y + (lane_sign * half_lane) * ny
                    self.add_obstacle(Obstacle(
                        obstacle_id=f"obs_curve_{idx}",
                        x=round(ox, 3), y=round(oy, 3), z=0.0,
                        length=3.2, width=1.8, height=1.5,
                        yaw=round(seg.yaw, 3),
                    ))

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Adds a static obstacle to the road."""
        self.obstacles.append(obstacle)

    def check_obstacle_collision(self, vehicle_corners: List[Tuple[float, float]]) -> bool:
        """
        Checks if vehicle 2D bounding polygon intersects any obstacle bounding box.
        """
        for obs in self.obstacles:
            cos_o = math.cos(obs.yaw)
            sin_o = math.sin(obs.yaw)
            hx = obs.length / 2.0
            hy = obs.width / 2.0

            for vx, vy in vehicle_corners:
                dx = vx - obs.x
                dy = vy - obs.y
                # Transform to obstacle local frame
                lx = dx * cos_o + dy * sin_o
                ly = -dx * sin_o + dy * cos_o

                if abs(lx) <= hx and abs(ly) <= hy:
                    return True
        return False

    def check_boundary_collision(self, vehicle_corners: List[Tuple[float, float]]) -> bool:
        """
        Checks if any corner of the vehicle has exceeded the road shoulders.
        Returns True if vehicle is off-road / colliding with boundary.
        """
        half_width = self.total_width_m / 2.0
        for vx, vy in vehicle_corners:
            # Find closest segment
            min_dist = float("inf")
            closest_seg = None
            for seg in self.segments:
                d = (vx - seg.center_x) ** 2 + (vy - seg.center_y) ** 2
                if d < min_dist:
                    min_dist = d
                    closest_seg = seg

            if closest_seg is not None:
                # Compute signed lateral cross-track distance from centerline
                dx = vx - closest_seg.center_x
                dy = vy - closest_seg.center_y
                # Heading unit vector and normal unit vector
                nx = -math.sin(closest_seg.yaw)
                ny = math.cos(closest_seg.yaw)
                lateral_dist = abs(dx * nx + dy * ny)

                if lateral_dist > half_width:
                    return True  # Corner is outside road boundary
        return False

    def check_goal_reached(self, x: float, y: float, tolerance_m: float = 3.5) -> bool:
        """Checks if vehicle has reached within goal tolerance."""
        gx, gy, _ = self.goal_pose
        dist = math.sqrt((x - gx) ** 2 + (y - gy) ** 2)
        return dist <= tolerance_m

    def to_dict(self) -> Dict[str, Any]:
        """Serializes road geometry and obstacles for Three.js visualizer."""
        return {
            "road_type": self.road_type,
            "road_length_m": self.road_length_m,
            "total_width_m": self.total_width_m,
            "lane_width_m": self.lane_width_m,
            "num_lanes": self.num_lanes,
            "centerline": [[round(s.center_x, 3), round(s.center_y, 3)] for s in self.segments],
            "left_boundary": [[round(s.left_x, 3), round(s.left_y, 3)] for s in self.segments],
            "right_boundary": [[round(s.right_x, 3), round(s.right_y, 3)] for s in self.segments],
            "start_pose": [round(v, 4) for v in self.start_pose],
            "goal_pose": [round(v, 4) for v in self.goal_pose],
            "obstacles": [obs.to_dict() for obs in self.obstacles],
        }

