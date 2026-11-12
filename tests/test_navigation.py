"""
Unit and end-to-end headless scenario tests for Phase 4 Autonomous Navigation:
Global A*, Local Obstacle Avoidance, Pure Pursuit Tracking, Speed Control, and Safe Stop.
"""

import math
import pytest

from autonomy.interfaces.types import VehicleState, TrajectoryPoint, PlannedTrajectory
from autonomy.planning import GlobalAStarPlanner
from autonomy.control import PurePursuitController
from simulator.simulation import Simulation
from simulator.world.road import RoadMap
from simulator.scenarios import get_scenario


def test_global_astar_planner_curves():
    """Verifies GlobalAStarPlanner finds valid routes along straight, curve, and s_curve roads."""
    planner = GlobalAStarPlanner()
    for road_type in ["straight", "curve", "s_curve"]:
        road = RoadMap(road_type=road_type, road_length_m=100.0, with_obstacles=False)
        traj = planner.plan(road)
        assert traj.valid
        assert len(traj.points) > 20
        last_pt = traj.points[-1]
        gx, gy, _ = road.goal_pose
        assert math.hypot(last_pt.x - gx, last_pt.y - gy) <= 1.5


def test_pure_pursuit_steering_sign():
    """Verifies Pure Pursuit steers left (+delta) for +Y targets and right (-delta) for -Y targets."""
    pp = PurePursuitController()
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=5.0, steering_angle=0.0)

    left_traj = PlannedTrajectory(
        points=[
            TrajectoryPoint(x=0.0, y=0.0),
            TrajectoryPoint(x=6.0, y=2.5),
            TrajectoryPoint(x=12.0, y=2.5),
        ],
        valid=True,
    )
    steer_left, _ = pp.compute_steering(state, left_traj)
    assert steer_left > 0.05

    right_traj = PlannedTrajectory(
        points=[
            TrajectoryPoint(x=0.0, y=0.0),
            TrajectoryPoint(x=6.0, y=-2.5),
            TrajectoryPoint(x=12.0, y=-2.5),
        ],
        valid=True,
    )
    steer_right, _ = pp.compute_steering(state, right_traj)
    assert steer_right < -0.05


def test_autopilot_straight_clear_reaches_goal():
    """Acceptance Test 1: Vehicle autonomously reaches goal in obstacle-free scenario."""
    spec = get_scenario("straight_clear")
    sim = Simulation(dt_sec=0.02, road=spec.builder(), autopilot=True)
    summary = sim.run_headless(duration_sec=spec.max_duration_sec)

    assert summary["goal_reached"] is True
    assert summary["collision_occurred"] is False
    assert summary["failure_reason"] is None


def test_autopilot_avoids_static_obstacle():
    """Acceptance Test 2: Vehicle detects and avoids a static obstacle using LiDAR perception only."""
    spec = get_scenario("straight_static_obstacle")
    sim = Simulation(dt_sec=0.02, road=spec.builder(), autopilot=True)
    summary = sim.run_headless(duration_sec=spec.max_duration_sec)

    assert summary["collision_occurred"] is False
    assert summary["goal_reached"] is True


def test_autopilot_double_obstacle_slalom():
    """Acceptance Test 3: Vehicle weaves past two staggered obstacles without colliding."""
    spec = get_scenario("straight_double_obstacle")
    sim = Simulation(dt_sec=0.02, road=spec.builder(), autopilot=True)
    summary = sim.run_headless(duration_sec=spec.max_duration_sec)

    assert summary["collision_occurred"] is False
    assert summary["goal_reached"] is True


def test_autopilot_s_curve_avoidance():
    """Acceptance Test 4: Closed-loop autonomous driving and obstacle avoidance on curved S-bend."""
    spec = get_scenario("s_curve_avoidance")
    sim = Simulation(dt_sec=0.02, road=spec.builder(), autopilot=True)
    summary = sim.run_headless(duration_sec=spec.max_duration_sec)

    assert summary["collision_occurred"] is False
    assert summary["goal_reached"] is True


def test_autopilot_blocked_road_safe_stop():
    """Acceptance Test 5: Vehicle detects blocked road, triggers safe stop, and logs failure reason."""
    spec = get_scenario("blocked_road")
    sim = Simulation(dt_sec=0.02, road=spec.builder(), autopilot=True)
    summary = sim.run_headless(duration_sec=spec.max_duration_sec)

    assert summary["collision_occurred"] is False
    assert summary["goal_reached"] is False
    assert abs(sim.vehicle.state.velocity) < 0.05
    assert summary["failure_reason"] == "NO_SAFE_ROUTE"
    assert summary["emergency_stop_steps"] > 0

