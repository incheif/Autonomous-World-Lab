"""
Closed-Loop Autonomous Navigation Stack.
Connects Perception -> Mapping -> Global A* Planning -> Local Avoidance -> Pure Pursuit Control.
Strictly consumes sensor-derived perception & cost maps (no ground-truth obstacle access).
"""

from typing import Optional, TYPE_CHECKING
from autonomy.interfaces.types import (
    VehicleState,
    ControlCommand,
    PlannedTrajectory,
    OccupancyGridMap,
)
from autonomy.perception import PerceptionResult
from autonomy.planning import GlobalAStarPlanner, GlobalPlannerConfig, LocalAvoidancePlanner, LocalPlannerConfig
from autonomy.control import VehicleController, PurePursuitConfig, SpeedControllerConfig

if TYPE_CHECKING:
    from simulator.world.road import RoadMap


class NavigationStack:
    """
    End-to-end autonomy stack integrating route planning, obstacle avoidance, and path tracking.
    """

    def __init__(
        self,
        global_config: Optional[GlobalPlannerConfig] = None,
        local_config: Optional[LocalPlannerConfig] = None,
        pursuit_config: Optional[PurePursuitConfig] = None,
        speed_config: Optional[SpeedControllerConfig] = None,
    ):
        self.global_planner = GlobalAStarPlanner(global_config)
        self.local_planner = LocalAvoidancePlanner(local_config)
        self.controller = VehicleController(pursuit_config, speed_config)

        self.global_trajectory: Optional[PlannedTrajectory] = None
        self.local_trajectory: Optional[PlannedTrajectory] = None
        self.planner_status: str = "IDLE"
        self.failure_reason: Optional[str] = None

    def reset(self, road: "RoadMap", timestamp: float = 0.0) -> None:
        """
        Plans initial global A* route along the road network and resets planner state.
        """
        self.local_planner.reset()
        self.global_trajectory = self.global_planner.plan(road, timestamp=timestamp)
        self.local_trajectory = None
        self.planner_status = "READY" if self.global_trajectory.valid else "GLOBAL_PLAN_FAILED"
        self.failure_reason = None if self.global_trajectory.valid else self.global_trajectory.reason

    def step(
        self,
        estimated_state: VehicleState,
        perception: Optional[PerceptionResult],
        cost_map: Optional[OccupancyGridMap],
        road_half_width: float,
        goal_pose: tuple[float, float, float],
        replan: bool = True,
    ) -> ControlCommand:
        """
        Executes one Plan + Control cycle using ONLY sensor-derived perception/mapping
        and road corridor geometry.
        """
        if self.global_trajectory is None or not self.global_trajectory.valid:
            self.planner_status = "NO_GLOBAL_ROUTE"
            self.failure_reason = "NO_GLOBAL_ROUTE"
            return ControlCommand(acceleration=0.0, steering=0.0, brake=1.0, emergency_stop=True, timestamp=estimated_state.timestamp)

        detected_obstacles = perception.detected_obstacles if perception else []

        # 1. Local Obstacle Avoidance Plan (replan at sensor rate or if no active local trajectory)
        if replan or self.local_trajectory is None:
            self.local_trajectory = self.local_planner.plan(
                vehicle_state=estimated_state,
                global_trajectory=self.global_trajectory,
                cost_map=cost_map,
                detected_obstacles=detected_obstacles,
                road_half_width=road_half_width,
            )

        # 2. Pure Pursuit + Longitudinal Control
        cmd = self.controller.compute_command(
            state=estimated_state,
            trajectory=self.local_trajectory,
            goal_pose=goal_pose,
        )

        if not self.local_trajectory.valid:
            self.planner_status = "EMERGENCY_STOP"
            self.failure_reason = self.local_trajectory.reason or "NO_SAFE_ROUTE"
        elif cmd.emergency_stop:
            self.planner_status = "GOAL_REACHED"
            self.failure_reason = None
        else:
            self.planner_status = self.local_trajectory.reason or "ACTIVE"
            self.failure_reason = None

        return cmd
