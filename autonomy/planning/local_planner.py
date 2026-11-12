"""
Local Obstacle Avoidance Trajectory Planner.
Evaluates smooth lateral-shift candidate trajectories against the perceived Occupancy Grid
and Detected Obstacles (zero access to ground-truth obstacles).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from autonomy.interfaces.types import (
    VehicleState,
    TrajectoryPoint,
    PlannedTrajectory,
    DetectedObstacle,
    OccupancyGridMap,
)


@dataclass
class LocalPlannerConfig:
    """Configuration for Local Cost-Map Trajectory Planner."""
    horizon_m: float = 25.0              # Planning lookahead horizon [m]
    step_ds_m: float = 1.0               # Spacing between local trajectory waypoints [m]
    blend_distance_m: float = 10.0       # Distance over which lane shift blends [m]
    vehicle_width_m: float = 1.80        # Ego vehicle width [m]
    vehicle_length_m: float = 4.50       # Ego vehicle length [m]
    boundary_margin_m: float = 0.25      # Margin from road shoulder edges [m]
    obstacle_margin_m: float = 0.25      # Extra clearance around detected obstacle OBBs [m]
    lethal_cost_threshold: float = 95.0  # Cost map threshold treated as lethal collision
    max_lateral_offset_step_m: float = 0.5 # Resolution of lateral offset candidates [m]
    cruise_speed_mps: float = 8.0        # Target speed on clear straight road [m/s]
    avoidance_speed_mps: float = 5.5     # Reduced speed while actively maneuvering around obstacles [m/s]
    max_lateral_accel_mps2: float = 2.5  # Cornering speed cap [m/s^2]

    # Cost weights
    weight_centerline: float = 1.2       # Prefer staying near global reference path
    weight_hysteresis: float = 0.6       # Prefer consistency with previous chosen offset
    weight_cost_map: float = 0.25        # Penalize entering safety inflation buffer


class LocalAvoidancePlanner:
    """
    Deterministic lattice/Frenet-offset local planner.
    Selects the lowest-cost collision-free trajectory using sensor perception and BEV cost map.
    """

    def __init__(self, config: Optional[LocalPlannerConfig] = None):
        self.config = config or LocalPlannerConfig()
        self.prev_target_offset: float = 0.0

    def reset(self) -> None:
        self.prev_target_offset = 0.0

    def plan(
        self,
        vehicle_state: VehicleState,
        global_trajectory: PlannedTrajectory,
        cost_map: Optional[OccupancyGridMap],
        detected_obstacles: List[DetectedObstacle],
        road_half_width: float,
    ) -> PlannedTrajectory:
        """
        Generates a collision-free local trajectory avoiding detected obstacles.
        Returns PlannedTrajectory(valid=False, reason='NO_SAFE_ROUTE') if road is blocked.
        """
        t_now = vehicle_state.timestamp
        if not global_trajectory.valid or len(global_trajectory.points) < 2:
            return PlannedTrajectory(points=[], timestamp=t_now, valid=False, reason="INVALID_GLOBAL_ROUTE")

        g_pts = global_trajectory.points

        # 1. Find closest point on global trajectory to rear axle
        nearest_idx = self._find_nearest_index(vehicle_state.x, vehicle_state.y, g_pts)
        nearest_pt = g_pts[nearest_idx]

        # Current signed lateral offset from reference path (+ left, - right)
        dx_v = vehicle_state.x - nearest_pt.x
        dy_v = vehicle_state.y - nearest_pt.y
        current_offset = -dx_v * math.sin(nearest_pt.yaw) + dy_v * math.cos(nearest_pt.yaw)

        # 2. Extract local reference slice ahead over horizon_m
        ref_slice: List[Tuple[TrajectoryPoint, float]] = []
        cum_s = 0.0
        ref_slice.append((g_pts[nearest_idx], 0.0))
        for idx in range(nearest_idx + 1, len(g_pts)):
            ds = math.hypot(g_pts[idx].x - g_pts[idx - 1].x, g_pts[idx].y - g_pts[idx - 1].y)
            cum_s += ds
            ref_slice.append((g_pts[idx], cum_s))
            if cum_s >= self.config.horizon_m:
                break

        if len(ref_slice) < 2:
            # Very close to goal
            return PlannedTrajectory(points=g_pts[nearest_idx:], timestamp=t_now, valid=True, reason="AT_GOAL")

        # 3. Generate lateral offset candidates within road corridor
        max_lat = max(0.0, road_half_width - (self.config.vehicle_width_m / 2.0) - self.config.boundary_margin_m)
        step = self.config.max_lateral_offset_step_m
        num_steps = int(math.floor(max_lat / step))
        candidate_offsets = [0.0]
        for k in range(1, num_steps + 1):
            candidate_offsets.append(k * step)
            candidate_offsets.append(-k * step)
        if max_lat > 0.1 and (num_steps * step) < max_lat - 0.05:
            candidate_offsets.append(max_lat)
            candidate_offsets.append(-max_lat)

        best_score = float("inf")
        best_points: Optional[List[TrajectoryPoint]] = None
        best_offset = 0.0

        # 4. Evaluate every candidate lateral offset trajectory
        for target_offset in candidate_offsets:
            cand_pts = self._generate_candidate_points(
                ref_slice=ref_slice,
                start_offset=current_offset,
                target_offset=target_offset,
                timestamp=t_now,
            )

            feasible, map_cost = self._evaluate_candidate(
                points=cand_pts,
                cost_map=cost_map,
                detected_obstacles=detected_obstacles,
            )

            if not feasible:
                continue

            # Compute composite score
            score = (
                self.config.weight_centerline * abs(target_offset)
                + self.config.weight_hysteresis * abs(target_offset - self.prev_target_offset)
                + self.config.weight_cost_map * map_cost
            )

            if score < best_score:
                best_score = score
                best_points = cand_pts
                best_offset = target_offset

        if best_points is None:
            return PlannedTrajectory(
                points=[],
                timestamp=t_now,
                valid=False,
                reason="NO_SAFE_ROUTE",
            )

        self.prev_target_offset = best_offset

        # Adjust target velocity along chosen trajectory based on maneuver & curvature
        is_avoiding = abs(best_offset) > 0.35
        speed_cap = self.config.avoidance_speed_mps if is_avoiding else self.config.cruise_speed_mps
        for pt in best_points:
            v_curv = math.sqrt(self.config.max_lateral_accel_mps2 / max(abs(pt.curvature), 1e-4))
            pt.velocity = float(min(pt.velocity, speed_cap, v_curv))

        reason = "AVOIDING_OBSTACLE" if is_avoiding else "LANE_FOLLOWING"
        return PlannedTrajectory(
            points=best_points,
            timestamp=t_now,
            valid=True,
            reason=reason,
        )

    def _generate_candidate_points(
        self,
        ref_slice: List[Tuple[TrajectoryPoint, float]],
        start_offset: float,
        target_offset: float,
        timestamp: float,
    ) -> List[TrajectoryPoint]:
        """Generates smooth cosine-blended lateral shift path along reference slice."""
        coords = []
        ref_velocities = []
        blend_dist = max(self.config.blend_distance_m, 1.0)

        for pt, s in ref_slice:
            alpha = 0.5 * (1.0 - math.cos(math.pi * min(s / blend_dist, 1.0)))
            lat = start_offset + (target_offset - start_offset) * alpha
            nx = -math.sin(pt.yaw)
            ny = math.cos(pt.yaw)
            coords.append([pt.x + lat * nx, pt.y + lat * ny])
            ref_velocities.append(pt.velocity)

        n = len(coords)
        result: List[TrajectoryPoint] = []
        for i in range(n):
            if i < n - 1:
                dx = coords[i + 1][0] - coords[i][0]
                dy = coords[i + 1][1] - coords[i][1]
            else:
                dx = coords[i][0] - coords[i - 1][0]
                dy = coords[i][1] - coords[i - 1][1]
            yaw = math.atan2(dy, dx)

            curv = 0.0
            if 0 < i < n - 1:
                dx_prev = coords[i][0] - coords[i - 1][0]
                dy_prev = coords[i][1] - coords[i - 1][1]
                yaw_prev = math.atan2(dy_prev, dx_prev)
                ds = max(math.hypot(dx, dy), 1e-3)
                dyaw = math.atan2(math.sin(yaw - yaw_prev), math.cos(yaw - yaw_prev))
                curv = dyaw / ds

            result.append(
                TrajectoryPoint(
                    x=float(coords[i][0]),
                    y=float(coords[i][1]),
                    yaw=float(yaw),
                    velocity=float(ref_velocities[i]),
                    curvature=float(curv),
                    timestamp=timestamp,
                )
            )
        return result

    def _evaluate_candidate(
        self,
        points: List[TrajectoryPoint],
        cost_map: Optional[OccupancyGridMap],
        detected_obstacles: List[DetectedObstacle],
    ) -> Tuple[bool, float]:
        """
        Checks vehicle footprint collision against OccupancyGridMap and DetectedObstacles.
        Returns (is_feasible, average_soft_cost).
        """
        half_w = self.config.vehicle_width_m / 2.0
        check_radius = half_w + self.config.obstacle_margin_m
        # Offsets from rear axle along heading to cover entire 4.5m chassis
        # (rear overhang -0.8m, center +1.4m, front bumper +3.3m)
        body_samples = [0.0, 1.4, 2.8]

        total_cost = 0.0
        cost_samples = 0

        for pt in points:
            cos_y = math.cos(pt.yaw)
            sin_y = math.sin(pt.yaw)

            for offset_x in body_samples:
                sx = pt.x + offset_x * cos_y
                sy = pt.y + offset_x * sin_y

                # 1. Check OccupancyGridMap lethal threshold & accumulate soft cost
                if cost_map is not None:
                    cell = cost_map.world_to_grid(sx, sy)
                    if cell is not None:
                        col, row = cell
                        cell_cost = float(cost_map.cost_map[row, col])
                        if cell_cost >= self.config.lethal_cost_threshold:
                            return False, float("inf")
                        total_cost += cell_cost
                        cost_samples += 1

                # 2. Check Euclidean distance from sample point to detected obstacle OBBs
                for obs in detected_obstacles:
                    dx = sx - obs.x
                    dy = sy - obs.y
                    cos_o = math.cos(obs.yaw)
                    sin_o = math.sin(obs.yaw)
                    lx = dx * cos_o + dy * sin_o
                    ly = -dx * sin_o + dy * cos_o

                    ex = max(0.0, abs(lx) - obs.length / 2.0)
                    ey = max(0.0, abs(ly) - obs.width / 2.0)
                    if math.hypot(ex, ey) <= check_radius:
                        return False, float("inf")

        mean_cost = (total_cost / cost_samples) if cost_samples > 0 else 0.0
        return True, mean_cost

    @staticmethod
    def _find_nearest_index(x: float, y: float, points: List[TrajectoryPoint]) -> int:
        best_idx = 0
        best_d2 = float("inf")
        for i, pt in enumerate(points):
            d2 = (pt.x - x) ** 2 + (pt.y - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        return best_idx

