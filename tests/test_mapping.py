"""
Unit tests for Mapping Subsystem: 2D Occupancy Grid, Ray-tracing, and Vehicle Footprint Inflation.
"""

import numpy as np
import pytest

from autonomy.interfaces.types import VehicleState, DetectedObstacle
from autonomy.mapping import OccupancyGrid, OccupancyGridConfig, UNKNOWN, FREE, OCCUPIED


def test_occupancy_grid_initialization():
    """Verifies that occupancy grid initializes to unknown state and correct spatial dimensions."""
    config = OccupancyGridConfig(resolution=0.5, width_m=20.0, height_m=20.0)
    og = OccupancyGrid(config)
    og.reset(0.0, 0.0)

    assert og.width_cells == 40
    assert og.height_cells == 40
    assert np.all(og.grid == UNKNOWN)
    assert pytest.approx(og.origin_x) == -10.0
    assert pytest.approx(og.origin_y) == -10.0


def test_occupancy_grid_raycast_free_and_occupied():
    """Verifies ray-casting marks line of sight as FREE and endpoint as OCCUPIED."""
    config = OccupancyGridConfig(resolution=1.0, width_m=30.0, height_m=30.0)
    og = OccupancyGrid(config)
    og.reset(0.0, 0.0)

    vehicle_state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, steering_angle=0.0)

    # Obstacle hit at x=10.0, y=0.0
    obs_pts = np.array([[10.0, 0.0, 1.0]])
    og.update_from_scan(vehicle_state, obstacle_points=obs_pts)

    # Cell at endpoint should be OCCUPIED
    hit_cell = og.world_to_grid(10.0, 0.0)
    assert hit_cell is not None
    assert og.grid[hit_cell[1], hit_cell[0]] == OCCUPIED

    # Cells along line of sight (e.g., x=5.0, y=0.0) should be FREE
    mid_cell = og.world_to_grid(5.0, 0.0)
    assert mid_cell is not None
    assert og.grid[mid_cell[1], mid_cell[0]] == FREE


def test_vehicle_footprint_safety_inflation():
    """
    Verifies that cost map applies lethal cost within lethal radius
    and tapers monotonically down to zero outside inflation radius.
    """
    config = OccupancyGridConfig(
        resolution=0.25,
        width_m=20.0,
        height_m=20.0,
        vehicle_length=4.50,
        vehicle_width=1.80,
        safety_margin=0.50,
    )
    og = OccupancyGrid(config)
    og.reset(0.0, 0.0)

    # Burn an obstacle at center (0, 0)
    obs = DetectedObstacle(
        obstacle_id="center_box",
        x=0.0,
        y=0.0,
        z=0.5,
        length=1.0,
        width=1.0,
        height=1.0,
        yaw=0.0,
        confidence=1.0,
        num_points=50,
    )
    og.update_from_detected_obstacles([obs])

    # Check center cost is lethal (100)
    c_center = og.world_to_grid(0.0, 0.0)
    assert c_center is not None
    assert og.cost_map[c_center[1], c_center[0]] == 100.0

    # Within lethal radius (e.g. 0.8m) cost should be 100
    c_near = og.world_to_grid(0.8, 0.0)
    assert c_near is not None
    assert og.cost_map[c_near[1], c_near[0]] == 100.0

    # Between lethal radius and inflation radius, cost should decrease monotonically
    cost_1 = og.cost_map[og.world_to_grid(1.5, 0.0)[1], og.world_to_grid(1.5, 0.0)[0]]
    cost_2 = og.cost_map[og.world_to_grid(2.2, 0.0)[1], og.world_to_grid(2.2, 0.0)[0]]
    assert cost_1 > cost_2

    # Far away (> inflation_radius ~ 3.5m) cost should be 0.0 (or unknown)
    c_far = og.world_to_grid(8.0, 0.0)
    assert c_far is not None
    # If not unknown, cost is 0
    assert og.cost_map[c_far[1], c_far[0]] in (0.0, config.unknown_cost)


def test_occupancy_grid_snapshot_to_dict():
    """Verifies that OccupancyGridMap serializes correctly for telemetry."""
    config = OccupancyGridConfig(resolution=0.5, width_m=10.0, height_m=10.0)
    og = OccupancyGrid(config)
    og.reset(5.0, 5.0)

    snapshot = og.get_map_snapshot(timestamp=1.23)
    data = snapshot.to_dict(downsample=2)

    assert "origin_x" in data
    assert "origin_y" in data
    assert "resolution" in data
    assert "costs" in data
    assert data["timestamp"] == 1.23
    assert len(data["costs"]) > 0

