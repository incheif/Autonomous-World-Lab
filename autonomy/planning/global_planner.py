"""
Global Route and A* Path Planner.
Computes global waypoint sequences along the road network from start to goal
without accessing hidden ground-truth obstacle placements.
"""

import heapq
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
import numpy as np

from autonomy.interfaces.types import TrajectoryPoint, PlannedTrajectory

if TYPE_CHECKING:
    from simulator.world.road import RoadMap


@dataclass
class GlobalPlannerConfig:
    """Configuration for Global A* Planner."""
    grid_resolution_m: float = 1.0      # 2D grid resolution for global A* [m]
    road_margin_m: float = 1.10         # Keep route inside shoulders by vehicle half-width + margin [m]
    centerline_weight: float = 0.35     # Cost penalty per metre away from centerline
    cruise_speed_mps: float = 8.0       # Nominal reference cruise speed [m/s]
    max_lateral_accel_mps2: float = 2.5 # Comfort limit for cornering speed [m/s^2]
    smoothing_iterations: int = 3       # Path smoothing passes


class GlobalAStarPlanner:
    """
    2D Grid-based A* Global Planner over the drivable road corridor.
    Produces a smooth global reference trajectory from start_pose to goal_pose.
    """

    def __init__(self, config: Optional[GlobalPlannerConfig] = None):
        self.config = config or GlobalPlannerConfig()

    def plan(self, road: "RoadMap", timestamp: float = 0.0) -> PlannedTrajectory:
        """
        Plans a global route from road.start_pose to road.goal_pose using A*.
        Does NOT inspect road.obstacles (obstacles are handled by online local planner).
        """
        if not road.segments:
            return PlannedTrajectory(points=[], timestamp=timestamp, valid=False, reason="EMPTY_ROAD")

        res = self.config.grid_resolution_m
        seg_xy = np.array([[s.center_x, s.center_y] for s in road.segments], dtype=float)

        pad = road.total_width_m + 6.0
        min_x = float(np.min(seg_xy[:, 0]) - pad)
        max_x = float(np.max(seg_xy[:, 0]) + pad)
        min_y = float(np.min(seg_xy[:, 1]) - pad)
        max_y = float(np.max(seg_xy[:, 1]) + pad)

        width_cells = int(math.ceil((max_x - min_x) / res))
        height_cells = int(math.ceil((max_y - min_y) / res))

        # Vectorized distance to road centerline for every grid cell
        xs = min_x + (np.arange(width_cells) + 0.5) * res
        ys = min_y + (np.arange(height_cells) + 0.5) * res
        grid_x, grid_y = np.meshgrid(xs, ys)  # shape (H, W)

        # Compute min distance from each grid cell to road centerline segments
        dx = grid_x[..., np.newaxis] - seg_xy[:, 0]
        dy = grid_y[..., np.newaxis] - seg_xy[:, 1]
        dist_to_center = np.sqrt(np.min(dx * dx + dy * dy, axis=-1))

        half_drivable = max(0.5, (road.total_width_m / 2.0) - self.config.road_margin_m)
        traversable = dist_to_center <= half_drivable

        def world_to_cell(wx: float, wy: float) -> Tuple[int, int]:
            c = int(np.clip(math.floor((wx - min_x) / res), 0, width_cells - 1))
            r = int(np.clip(math.floor((wy - min_y) / res), 0, height_cells - 1))
            return (c, r)

        def cell_to_world(col: int, row: int) -> Tuple[float, float]:
            return (min_x + (col + 0.5) * res, min_y + (row + 0.5) * res)

        start_cell = world_to_cell(road.start_pose[0], road.start_pose[1])
        goal_cell = world_to_cell(road.goal_pose[0], road.goal_pose[1])

        # Ensure start and goal cells are marked traversable
        traversable[start_cell[1], start_cell[0]] = True
        traversable[goal_cell[1], goal_cell[0]] = True

        # Run 8-connected A*
        raw_path_cells = self._astar_search(
            traversable=traversable,
            dist_to_center=dist_to_center,
            start=start_cell,
            goal=goal_cell,
            res=res,
        )

        if not raw_path_cells:
            # Fallback to centerline waypoints if grid search fails due to coarse discretization
            return self.plan_centerline(road, timestamp=timestamp)

        # Convert cells to world XY coordinates
        coords = np.array([cell_to_world(c, r) for c, r in raw_path_cells], dtype=float)
        # Pin exact start and goal
        coords[0] = [road.start_pose[0], road.start_pose[1]]
        coords[-1] = [road.goal_pose[0], road.goal_pose[1]]

        # Smooth path coordinates (keeping endpoints fixed)
        coords = self._smooth_path(coords, iterations=self.config.smoothing_iterations)

        points = self._build_trajectory_points(coords, timestamp=timestamp)
        return PlannedTrajectory(points=points, timestamp=timestamp, valid=True, reason="A_STAR_OK")

    def plan_centerline(self, road: "RoadMap", timestamp: float = 0.0) -> PlannedTrajectory:
        """
        Generates a predefined waypoint trajectory directly from road centerline up to goal_pose.
        """
        if not road.segments:
            return PlannedTrajectory(points=[], timestamp=timestamp, valid=False, reason="EMPTY_ROAD")

        gx, gy, _ = road.goal_pose
        coords_list = []
        for seg in road.segments:
            coords_list.append([seg.center_x, seg.center_y])
            if math.hypot(seg.center_x - gx, seg.center_y - gy) <= 1.2:
                break

        coords = np.array(coords_list, dtype=float)
        points = self._build_trajectory_points(coords, timestamp=timestamp)
        return PlannedTrajectory(points=points, timestamp=timestamp, valid=True, reason="CENTERLINE_OK")

    def _astar_search(
        self,
        traversable: np.ndarray,
        dist_to_center: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        res: float,
    ) -> List[Tuple[int, int]]:
        """Deterministic 8-connected A* search."""
        h, w = traversable.shape
        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ]

        def heuristic(c: int, r: int) -> float:
            return math.hypot(goal[0] - c, goal[1] - r) * res

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (heuristic(*start), 0.0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        closed = set()

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed.add(current)
            cx, cy = current

            for dc, dr, step_mult in neighbors:
                nc, nr = cx + dc, cy + dr
                if not (0 <= nc < w and 0 <= nr < h):
                    continue
                if not traversable[nr, nc]:
                    continue
                nxt = (nc, nr)
                if nxt in closed:
                    continue

                center_penalty = self.config.centerline_weight * float(dist_to_center[nr, nc])
                tentative_g = current_g + (step_mult * res) + center_penalty

                if tentative_g < g_score.get(nxt, float("inf")):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    f = tentative_g + heuristic(nc, nr)
                    heapq.heappush(open_heap, (f, tentative_g, nxt))

        return []

    @staticmethod
    def _smooth_path(coords: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Applies weighted moving-average smoothing while preserving start and goal points."""
        if len(coords) < 3:
            return coords
        smoothed = coords.copy()
        for _ in range(iterations):
            updated = smoothed.copy()
            for i in range(1, len(smoothed) - 1):
                updated[i] = 0.25 * smoothed[i - 1] + 0.50 * smoothed[i] + 0.25 * smoothed[i + 1]
            smoothed = updated
        return smoothed

    def _build_trajectory_points(self, coords: np.ndarray, timestamp: float) -> List[TrajectoryPoint]:
        """Computes yaw, curvature, and curvature-aware velocity profile along path."""
        n = len(coords)
        points: List[TrajectoryPoint] = []
        if n == 0:
            return points

        yaws = np.zeros(n)
        for i in range(n):
            if i < n - 1:
                dx = coords[i + 1, 0] - coords[i, 0]
                dy = coords[i + 1, 1] - coords[i, 1]
            else:
                dx = coords[i, 0] - coords[i - 1, 0] if n > 1 else 1.0
                dy = coords[i, 1] - coords[i - 1, 1] if n > 1 else 0.0
            yaws[i] = math.atan2(dy, dx)

        # Compute discrete curvature and remaining distance to goal
        curvatures = np.zeros(n)
        dist_remaining = np.zeros(n)
        for i in range(n - 2, -1, -1):
            ds = math.hypot(coords[i + 1, 0] - coords[i, 0], coords[i + 1, 1] - coords[i, 1])
            dist_remaining[i] = dist_remaining[i + 1] + ds

        for i in range(1, n - 1):
            ds1 = math.hypot(coords[i, 0] - coords[i - 1, 0], coords[i, 1] - coords[i - 1, 1])
            ds2 = math.hypot(coords[i + 1, 0] - coords[i, 0], coords[i + 1, 1] - coords[i, 1])
            ds = max((ds1 + ds2) / 2.0, 1e-3)
            dyaw = math.atan2(math.sin(yaws[i + 1] - yaws[i - 1]), math.cos(yaws[i + 1] - yaws[i - 1]))
            curvatures[i] = dyaw / (2.0 * ds)

        for i in range(n):
            k = abs(curvatures[i])
            v_curv = math.sqrt(self.config.max_lateral_accel_mps2 / max(k, 1e-4))
            # Decelerate smoothly near goal (assuming comfortable decel ~ 2.0 m/s^2)
            v_goal = math.sqrt(2.0 * 1.8 * max(dist_remaining[i], 0.0))
            v_target = min(self.config.cruise_speed_mps, v_curv, max(v_goal, 1.5 if dist_remaining[i] > 1.0 else 0.0))

            points.append(
                TrajectoryPoint(
                    x=float(coords[i, 0]),
                    y=float(coords[i, 1]),
                    yaw=float(yaws[i]),
                    velocity=float(v_target),
                    curvature=float(curvatures[i]),
                    timestamp=timestamp,
                )
            )

        return points
