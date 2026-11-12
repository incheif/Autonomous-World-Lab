"""
Traffic and NPC Vehicle Simulation Engine for Autonomous-World-Lab.
Simulates dynamic ambient traffic, lane following, car-following physics (IDM),
and converts moving vehicles into dynamic 3D obstacles for LiDAR and collision checks.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from autonomy.interfaces.types import Obstacle, VehicleState
from simulator.world.road import RoadMap


CAR_COLORS = [
    "#ef4444",  # Crimson Red
    "#3b82f6",  # Sapphire Blue
    "#f8fafc",  # Pearl White
    "#334155",  # Midnight Charcoal
    "#eab308",  # Cyber Yellow
    "#10b981",  # Emerald Green
    "#f97316",  # Sunset Amber
]

VEHICLE_PROFILES = {
    "sedan": {"length": 4.50, "width": 1.80, "height": 1.45},
    "suv": {"length": 4.80, "width": 1.95, "height": 1.70},
    "truck": {"length": 6.20, "width": 2.20, "height": 2.20},
    "compact": {"length": 3.80, "width": 1.70, "height": 1.40},
}


@dataclass
class NPCVehicle:
    """
    Simulated dynamic traffic vehicle.
    Follows road lane lanes with Intelligent Driver Model (IDM) longitudinal control.
    """
    id: str
    lane: int                  # +1 = Left lane, -1 = Right lane
    s: float                   # Longitudinal arc distance along road [m]
    velocity: float = 6.0      # Current speed [m/s]
    target_speed: float = 7.5  # Desired cruise speed [m/s]
    vehicle_type: str = "sedan"
    color: str = "#ef4444"
    length: float = 4.50
    width: float = 1.80
    height: float = 1.45

    # World pose (updated dynamically)
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def update_pose(self, road: RoadMap):
        """Computes world (x, y, yaw) from arc-length progress s and lane offset."""
        if not road.segments:
            return

        # Clamp/wrap s
        n_segs = len(road.segments)
        max_s = road.road_length_m
        s_norm = max(0.0, min(self.s, max_s))

        # 1-metre segment spacing index
        idx_float = s_norm
        idx0 = int(math.floor(idx_float))
        idx0 = max(0, min(idx0, n_segs - 1))
        idx1 = min(idx0 + 1, n_segs - 1)
        frac = idx_float - idx0

        seg0 = road.segments[idx0]
        seg1 = road.segments[idx1]

        # Interpolate centerline
        cx = seg0.center_x + frac * (seg1.center_x - seg0.center_x)
        cy = seg0.center_y + frac * (seg1.center_y - seg0.center_y)
        # Interpolate yaw (handling circular wrap)
        dyaw = math.atan2(math.sin(seg1.yaw - seg0.yaw), math.cos(seg1.yaw - seg0.yaw))
        yaw = seg0.yaw + frac * dyaw

        # Offset from centerline to lane center (+1 = left, -1 = right)
        half_lane = road.lane_width_m / 2.0
        lane_offset = self.lane * half_lane
        nx = -math.sin(yaw)
        ny = math.cos(yaw)

        self.x = cx + lane_offset * nx
        self.y = cy + lane_offset * ny
        self.yaw = yaw

    def to_obstacle(self) -> Obstacle:
        """Converts NPC state to Obstacle for LiDAR raycasting & collision checks."""
        return Obstacle(
            obstacle_id=self.id,
            x=self.x,
            y=self.y,
            z=0.0,
            length=self.length,
            width=self.width,
            height=self.height,
            yaw=self.yaw,
            velocity=self.velocity,
            obstacle_type=self.vehicle_type,
            color=self.color,
        )


class TrafficManager:
    """
    Manages population of dynamic NPC vehicles across road lanes.
    Updates longitudinal dynamics, safe distance following, and respawning.
    """

    def __init__(self, enabled: bool = True, num_vehicles: int = 4):
        self.enabled: bool = enabled
        self.num_vehicles: int = num_vehicles
        self.vehicles: List[NPCVehicle] = []

    def initialize(self, road: RoadMap):
        """Spawns ambient traffic distributed across lanes."""
        self.vehicles.clear()
        if not self.enabled or not road.segments or self.num_vehicles <= 0:
            return

        road_len = road.road_length_m
        spacing = max(18.0, road_len / (self.num_vehicles + 1))
        types = ["sedan", "suv", "truck", "compact", "sedan"]

        for i in range(self.num_vehicles):
            lane = 1 if (i % 2 == 0) else -1
            # Stagger initial positions ahead of ego start
            s_init = 18.0 + (i * spacing)
            v_type = types[i % len(types)]
            dims = VEHICLE_PROFILES[v_type]
            color = CAR_COLORS[i % len(CAR_COLORS)]
            # Target speeds between 5.5 and 8.0 m/s (~20-30 km/h)
            v_target = 5.5 + (i * 0.7) % 2.5

            npc = NPCVehicle(
                id=f"npc_{i+1}",
                lane=lane,
                s=s_init,
                velocity=v_target * 0.85,
                target_speed=v_target,
                vehicle_type=v_type,
                color=color,
                length=dims["length"],
                width=dims["width"],
                height=dims["height"],
            )
            npc.update_pose(road)
            self.vehicles.append(npc)

    def step(self, dt: float, road: RoadMap, ego_state: Optional[VehicleState] = None):
        """Advances traffic state by dt using Intelligent Driver Model."""
        if not self.enabled or not self.vehicles:
            return

        max_s = road.road_length_m

        for npc in self.vehicles:
            # Find lead obstacle or ego vehicle in the same lane
            min_lead_gap = float("inf")
            lead_speed = npc.target_speed

            # Check other NPC vehicles ahead in same lane
            for other in self.vehicles:
                if other.id == npc.id or other.lane != npc.lane:
                    continue
                gap = other.s - npc.s - other.length
                if 0.0 < gap < min_lead_gap:
                    min_lead_gap = gap
                    lead_speed = other.velocity

            # Check ego vehicle if in front in same lane
            if ego_state is not None:
                # Approximate ego s by projection along road
                dx = ego_state.x - npc.x
                dy = ego_state.y - npc.y
                # Forward distance along heading
                fwd = dx * math.cos(npc.yaw) + dy * math.sin(npc.yaw)
                lat = -dx * math.sin(npc.yaw) + dy * math.cos(npc.yaw)
                if abs(lat) < (road.lane_width_m * 0.65) and 0.0 < fwd < min_lead_gap:
                    min_lead_gap = fwd - 2.8  # vehicle half extents
                    lead_speed = max(0.0, ego_state.velocity)

            # Intelligent Driver Model (IDM) acceleration law
            s0 = 4.0      # Minimum safe jam distance [m]
            T = 1.2       # Safe time headway [s]
            a_max = 2.0   # Comfortable acceleration [m/s^2]
            b = 2.5       # Comfortable deceleration [m/s^2]

            v = max(0.0, npc.velocity)
            v0 = max(1.0, npc.target_speed)

            delta_v = v - lead_speed
            s_star = s0 + max(0.0, v * T + (v * delta_v) / (2.0 * math.sqrt(a_max * b)))

            if min_lead_gap < float("inf"):
                gap = max(min_lead_gap, 0.5)
                accel = a_max * (1.0 - (v / v0) ** 4 - (s_star / gap) ** 2)
            else:
                accel = a_max * (1.0 - (v / v0) ** 4)

            # Clamp acceleration
            accel = max(-5.0, min(a_max, accel))
            npc.velocity = max(0.0, min(15.0, npc.velocity + accel * dt))
            npc.s += npc.velocity * dt

            # Wrap around when reaching end of road
            if npc.s > max_s + 5.0:
                # Find minimum s of current vehicles in this lane
                lane_s = [v.s for v in self.vehicles if v.lane == npc.lane]
                min_s_lane = min(lane_s) if lane_s else 20.0
                npc.s = max(2.0, min_s_lane - 25.0)
                npc.velocity = npc.target_speed * 0.7

            npc.update_pose(road)

    def get_obstacles(self) -> List[Obstacle]:
        """Returns all active NPC vehicles as Obstacle instances."""
        if not self.enabled:
            return []
        return [npc.to_obstacle() for npc in self.vehicles]

    def check_collision_with_ego(self, ego_corners: List[Tuple[float, float]]) -> bool:
        """
        Checks if any point of ego_corners penetrates the bounding box of any NPC vehicle.
        Uses exact point-to-rotated-box projection.
        """
        if not self.enabled or not self.vehicles:
            return False

        for npc in self.vehicles:
            cos_y = math.cos(-npc.yaw)
            sin_y = math.sin(-npc.yaw)
            hx = npc.length / 2.0
            hy = npc.width / 2.0

            for cx, cy in ego_corners:
                dx = cx - npc.x
                dy = cy - npc.y
                lx = dx * cos_y - dy * sin_y
                ly = dx * sin_y + dy * cos_y
                if abs(lx) <= hx and abs(ly) <= hy:
                    return True
        return False

    def to_dict(self) -> List[Dict[str, Any]]:
        """Serializes traffic state for telemetry."""
        return [
            {
                "id": npc.id,
                "x": round(npc.x, 3),
                "y": round(npc.y, 3),
                "yaw": round(npc.yaw, 3),
                "velocity": round(npc.velocity, 2),
                "type": npc.vehicle_type,
                "color": npc.color,
                "length": npc.length,
                "width": npc.width,
                "height": npc.height,
            }
            for npc in self.vehicles
        ]
