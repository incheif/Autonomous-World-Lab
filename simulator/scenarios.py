"""
Deterministic Scenario Definitions for Autonomous-World-Lab.
Provides standardized, repeatable test worlds for autonomous navigation verification.
"""

from dataclasses import dataclass
from typing import Dict, Callable
from autonomy.interfaces.types import Obstacle
from simulator.world.road import RoadMap


@dataclass
class ScenarioSpec:
    """Metadata and builder for a deterministic simulation scenario."""
    scenario_id: str
    name: str
    description: str
    seed: int
    max_duration_sec: float
    expect_goal_reached: bool
    expect_safe_stop: bool
    builder: Callable[[], RoadMap]


def build_straight_clear() -> RoadMap:
    return RoadMap(
        road_type="straight",
        road_length_m=100.0,
        lane_width_m=3.75,
        num_lanes=2,
        with_obstacles=False,
    )


def build_straight_static_obstacle() -> RoadMap:
    return RoadMap(
        road_type="straight",
        road_length_m=110.0,
        lane_width_m=3.75,
        num_lanes=2,
        obstacles=[
            Obstacle(
                obstacle_id="static_box_center",
                x=35.0,
                y=-0.6,
                z=0.0,
                length=3.0,
                width=2.0,
                height=1.5,
                yaw=0.0,
            )
        ],
    )


def build_straight_double_obstacle() -> RoadMap:
    return RoadMap(
        road_type="straight",
        road_length_m=120.0,
        lane_width_m=3.75,
        num_lanes=2,
        obstacles=[
            Obstacle(
                obstacle_id="obs_right_lane",
                x=35.0,
                y=-1.4,
                z=0.0,
                length=3.0,
                width=1.8,
                height=1.5,
                yaw=0.0,
            ),
            Obstacle(
                obstacle_id="obs_left_lane",
                x=70.0,
                y=1.4,
                z=0.0,
                length=3.0,
                width=1.8,
                height=1.5,
                yaw=0.0,
            ),
        ],
    )


def build_s_curve_avoidance() -> RoadMap:
    return RoadMap(
        road_type="s_curve",
        road_length_m=140.0,
        lane_width_m=3.75,
        num_lanes=2,
        with_obstacles=True,
    )


def build_blocked_road() -> RoadMap:
    return RoadMap(
        road_type="straight",
        road_length_m=100.0,
        lane_width_m=3.75,
        num_lanes=2,
        obstacles=[
            Obstacle(
                obstacle_id="full_road_barrier",
                x=40.0,
                y=0.0,
                z=0.0,
                length=2.0,
                width=7.4,
                height=1.6,
                yaw=0.0,
            )
        ],
    )


SCENARIOS: Dict[str, ScenarioSpec] = {
    "straight_clear": ScenarioSpec(
        scenario_id="straight_clear",
        name="Straight Clear Road",
        description="Obstacle-free 100m straight corridor baseline.",
        seed=42,
        max_duration_sec=20.0,
        expect_goal_reached=True,
        expect_safe_stop=False,
        builder=build_straight_clear,
    ),
    "straight_static_obstacle": ScenarioSpec(
        scenario_id="straight_static_obstacle",
        name="Static Obstacle Avoidance",
        description="Straight road with a single blocking obstacle box at X=35m.",
        seed=42,
        max_duration_sec=25.0,
        expect_goal_reached=True,
        expect_safe_stop=False,
        builder=build_straight_static_obstacle,
    ),
    "straight_double_obstacle": ScenarioSpec(
        scenario_id="straight_double_obstacle",
        name="Double Staggered Slalom",
        description="Two staggered obstacles in alternating lanes at X=35m and X=70m.",
        seed=42,
        max_duration_sec=30.0,
        expect_goal_reached=True,
        expect_safe_stop=False,
        builder=build_straight_double_obstacle,
    ),
    "s_curve_avoidance": ScenarioSpec(
        scenario_id="s_curve_avoidance",
        name="S-Curve Obstacle Avoidance",
        description="Curved 140m S-bend with static obstacles in alternating lanes.",
        seed=42,
        max_duration_sec=35.0,
        expect_goal_reached=True,
        expect_safe_stop=False,
        builder=build_s_curve_avoidance,
    ),
    "blocked_road": ScenarioSpec(
        scenario_id="blocked_road",
        name="Blocked Road Fail-Safe",
        description="Full-width barricade spanning both lanes; requires safe emergency stop.",
        seed=42,
        max_duration_sec=15.0,
        expect_goal_reached=False,
        expect_safe_stop=True,
        builder=build_blocked_road,
    ),
}


def get_scenario(scenario_id: str) -> ScenarioSpec:
    if scenario_id not in SCENARIOS:
        raise ValueError(f"Unknown scenario_id '{scenario_id}'. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[scenario_id]

