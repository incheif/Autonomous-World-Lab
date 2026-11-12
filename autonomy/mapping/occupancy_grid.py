"""
2D Occupancy Grid and Safety-Inflated Cost Map.
Integrates LiDAR observations and detected obstacles into free, occupied, and safety-buffered zones.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
from scipy.ndimage import distance_transform_edt

from autonomy.interfaces.types import VehicleState, DetectedObstacle, OccupancyGridMap


# Grid cell state definitions
UNKNOWN: int = -1
FREE: int = 0
OCCUPIED: int = 100


@dataclass
class OccupancyGridConfig:
    """
    Configuration parameters for occupancy grid and cost map generation.
    """
    resolution: float = 0.35               # Cell size [m]
    width_m: float = 60.0                  # Map horizontal extent [m]
    height_m: float = 60.0                 # Map vertical extent [m]
    vehicle_length: float = 4.50           # Vehicle footprint length [m]
    vehicle_width: float = 1.80            # Vehicle footprint width [m]
    safety_margin: float = 0.50            # Additional clearance margin [m]
    max_range: float = 50.0                # Max raycast integration range [m]
    unknown_cost: float = 25.0             # Cost assigned to unobserved / unknown cells


class OccupancyGrid:
    """
    2D Bird's-Eye-View Rolling Occupancy Grid and Cost Map.
    Supports ray-traced free space updates, obstacle cell insertion,
    and distance-transform obstacle inflation based on vehicle dimensions.
    """

    def __init__(self, config: Optional[OccupancyGridConfig] = None):
        self.config = config or OccupancyGridConfig()
        self.width_cells = int(np.ceil(self.config.width_m / self.config.resolution))
        self.height_cells = int(np.ceil(self.config.height_m / self.config.resolution))

        # Half extents for safety inflation
        # Inscribed radius (guaranteed collision) + circumscribed radius (potential collision)
        self.lethal_radius = (self.config.vehicle_width / 2.0) + 0.10
        self.inflation_radius = (
            np.hypot(self.config.vehicle_length / 2.0, self.config.vehicle_width / 2.0)
            + self.config.safety_margin
        )

        # Storage
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.grid = np.full((self.height_cells, self.width_cells), UNKNOWN, dtype=np.int8)
        self.cost_map = np.zeros((self.height_cells, self.width_cells), dtype=np.float32)

    def reset(self, center_x: float = 0.0, center_y: float = 0.0):
        """
        Resets and re-centers the grid around a given world point.
        """
        self.origin_x = center_x - (self.config.width_m / 2.0)
        self.origin_y = center_y - (self.config.height_m / 2.0)
        self.grid.fill(UNKNOWN)
        self.cost_map.fill(self.config.unknown_cost)

    def world_to_grid(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        """Converts world coordinates (wx, wy) to grid indices (col, row)."""
        col = int(np.floor((wx - self.origin_x) / self.config.resolution))
        row = int(np.floor((wy - self.origin_y) / self.config.resolution))
        if 0 <= col < self.width_cells and 0 <= row < self.height_cells:
            return (col, row)
        return None

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Converts grid indices (col, row) to world coordinates (wx, wy)."""
        wx = self.origin_x + (col + 0.5) * self.config.resolution
        wy = self.origin_y + (row + 0.5) * self.config.resolution
        return (wx, wy)

    def update_from_scan(
        self,
        vehicle_state: VehicleState,
        obstacle_points: np.ndarray,
        ground_points: Optional[np.ndarray] = None,
        recompute_cost: bool = True,
    ):
        """
        Updates the occupancy grid using ray-tracing from vehicle position to obstacle/ground points.
        """
        # Re-center grid window if vehicle has shifted significantly (> 10m from center)
        center_x = self.origin_x + self.config.width_m / 2.0
        center_y = self.origin_y + self.config.height_m / 2.0
        if np.hypot(vehicle_state.x - center_x, vehicle_state.y - center_y) > 10.0:
            self.reset(vehicle_state.x, vehicle_state.y)

        # Vehicle cell
        v_cell = self.world_to_grid(vehicle_state.x, vehicle_state.y)
        if v_cell is None:
            self.reset(vehicle_state.x, vehicle_state.y)
            v_cell = self.world_to_grid(vehicle_state.x, vehicle_state.y)

        # Mark ground points as free space (line-of-sight confirmed clear of elevated obstacles)
        if ground_points is not None and len(ground_points) > 0:
            sub_ground = ground_points[::8, :2]
            cols = np.floor((sub_ground[:, 0] - self.origin_x) / self.config.resolution).astype(int)
            rows = np.floor((sub_ground[:, 1] - self.origin_y) / self.config.resolution).astype(int)
            valid = (cols >= 0) & (cols < self.width_cells) & (rows >= 0) & (rows < self.height_cells)
            valid_c = cols[valid]
            valid_r = rows[valid]
            not_occ = self.grid[valid_r, valid_c] != OCCUPIED
            self.grid[valid_r[not_occ], valid_c[not_occ]] = FREE

        # Ray-trace obstacle returns: free line of sight leading up to occupied hit
        if len(obstacle_points) > 0:
            v_col, v_row = v_cell if v_cell else (self.width_cells // 2, self.height_cells // 2)

            dists = np.hypot(obstacle_points[:, 0] - vehicle_state.x, obstacle_points[:, 1] - vehicle_state.y)
            in_range = dists <= self.config.max_range
            pts_in_range = obstacle_points[in_range]

            if len(pts_in_range) > 0:
                cols = np.floor((pts_in_range[:, 0] - self.origin_x) / self.config.resolution).astype(int)
                rows = np.floor((pts_in_range[:, 1] - self.origin_y) / self.config.resolution).astype(int)
                valid = (cols >= 0) & (cols < self.width_cells) & (rows >= 0) & (rows < self.height_cells)
                valid_cells = np.column_stack((cols[valid], rows[valid]))

                if len(valid_cells) > 0:
                    unique_cells = np.unique(valid_cells, axis=0)

                    for h_col, h_row in unique_cells:
                        free_cells = self._bresenham_line(v_col, v_row, int(h_col), int(h_row))
                        for c, r in free_cells[:-1]:
                            if 0 <= c < self.width_cells and 0 <= r < self.height_cells:
                                if self.grid[r, c] != OCCUPIED:
                                    self.grid[r, c] = FREE
                        self.grid[int(h_row), int(h_col)] = OCCUPIED

        # Compute safety inflation cost map
        if recompute_cost:
            self._compute_cost_map()

    def update_from_detected_obstacles(
        self,
        detected_obstacles: List[DetectedObstacle],
        recompute_cost: bool = True,
    ):
        """
        Directly burns detected bounding box footprints into the occupancy grid.
        """
        for obs in detected_obstacles:
            # Box half-extents
            hl = obs.length / 2.0
            hw = obs.width / 2.0
            cos_y = np.cos(obs.yaw)
            sin_y = np.sin(obs.yaw)

            # Sample cells inside box
            steps_l = max(3, int(np.ceil(obs.length / self.config.resolution)))
            steps_w = max(3, int(np.ceil(obs.width / self.config.resolution)))
            dl = np.linspace(-hl, hl, steps_l)
            dw = np.linspace(-hw, hw, steps_w)
            mesh_l, mesh_w = np.meshgrid(dl, dw)

            wx = obs.x + mesh_l * cos_y - mesh_w * sin_y
            wy = obs.y + mesh_l * sin_y + mesh_w * cos_y

            for x_val, y_val in zip(wx.flat, wy.flat):
                cell = self.world_to_grid(float(x_val), float(y_val))
                if cell:
                    self.grid[cell[1], cell[0]] = OCCUPIED

        if recompute_cost:
            self._compute_cost_map()

    def _compute_cost_map(self):
        """
        Calculates exact Euclidean distance transform around occupied cells and computes
        safety inflation costs with vehicle footprint buffer.
        """
        is_occupied = (self.grid == OCCUPIED)

        if not np.any(is_occupied):
            self.cost_map.fill(0.0)
            # Retain unknown cost
            self.cost_map[self.grid == UNKNOWN] = self.config.unknown_cost
            return

        # Distance to closest occupied cell in metres
        dist_cells = distance_transform_edt(~is_occupied)
        dist_m = dist_cells * self.config.resolution

        # Lethal zone: d <= lethal_radius
        lethal_mask = dist_m <= self.lethal_radius

        # Buffer zone: lethal_radius < d <= inflation_radius
        buffer_mask = (dist_m > self.lethal_radius) & (dist_m <= self.inflation_radius)

        # Clear zone: d > inflation_radius
        clear_mask = dist_m > self.inflation_radius

        self.cost_map[lethal_mask] = 100.0

        # Exponential decay in safety buffer zone
        decay_factor = (self.inflation_radius - dist_m[buffer_mask]) / max(
            self.inflation_radius - self.lethal_radius, 1e-4
        )
        self.cost_map[buffer_mask] = np.clip(100.0 * np.power(decay_factor, 1.5), 0.0, 99.0)

        self.cost_map[clear_mask] = 0.0

        # Unknown penalty
        unknown_mask = (self.grid == UNKNOWN) & clear_mask
        self.cost_map[unknown_mask] = self.config.unknown_cost

    def get_map_snapshot(self, timestamp: float = 0.0) -> OccupancyGridMap:
        """
        Exports an immutable OccupancyGridMap snapshot.
        """
        return OccupancyGridMap(
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            resolution=self.config.resolution,
            width_cells=self.width_cells,
            height_cells=self.height_cells,
            grid=self.grid.copy(),
            cost_map=self.cost_map.copy(),
            timestamp=timestamp,
        )

    @staticmethod
    def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """Standard Bresenham ray-traversal algorithm."""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        points.append((x1, y1))
        return points
