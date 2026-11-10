"""
Unit tests for RoadMap and boundary collision checks.
"""

import math
from simulator.world.road import RoadMap
from simulator.vehicle.kinematic_bicycle import KinematicBicycleModel


def test_straight_road_generation():
    road = RoadMap(road_type="straight", road_length_m=100.0, lane_width_m=3.5, num_lanes=2)
    assert len(road.segments) > 0
    assert road.total_width_m == 7.0

    # Start pose should be at origin
    assert road.start_pose == (0.0, 0.0, 0.0)
    # Centerline Y coordinates should all be 0.0
    for s in road.segments:
        assert abs(s.center_y) < 1e-4
        assert abs(s.left_y - 3.5) < 1e-4
        assert abs(s.right_y - (-3.5)) < 1e-4


def test_curve_road_generation():
    road = RoadMap(road_type="s_curve", road_length_m=120.0)
    assert len(road.segments) > 0
    # Centerline should exhibit non-zero Y excursions
    y_vals = [s.center_y for s in road.segments]
    assert max(y_vals) > 2.0
    assert min(y_vals) < -2.0


def test_boundary_collision_detection():
    road = RoadMap(road_type="straight", road_length_m=100.0, lane_width_m=3.5, num_lanes=2)
    model = KinematicBicycleModel()

    # Vehicle on centerline -> no collision
    model.reset(x=10.0, y=0.0, yaw=0.0)
    corners = model.get_corners()
    assert not road.check_boundary_collision(corners)

    # Vehicle driven far off-road (Y = 15m) -> collision detected
    model.reset(x=10.0, y=15.0, yaw=0.0)
    corners_offroad = model.get_corners()
    assert road.check_boundary_collision(corners_offroad)


def test_goal_detection():
    road = RoadMap(road_type="straight", road_length_m=100.0)
    gx, gy, _ = road.goal_pose

    # Far from goal
    assert not road.check_goal_reached(0.0, 0.0)
    # At goal
    assert road.check_goal_reached(gx, gy, tolerance_m=2.0)

