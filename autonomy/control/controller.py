"""
Combined Lateral + Longitudinal Vehicle Controller.
Executes Pure Pursuit path tracking, speed regulation, and emergency stop fail-safe.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from autonomy.interfaces.types import VehicleState, ControlCommand, PlannedTrajectory
from .pure_pursuit import PurePursuitController, PurePursuitConfig


@dataclass
class SpeedControllerConfig:
    """Configuration for longitudinal speed control."""
    kp_speed: float = 1.6             # Proportional speed gain [1/s]
    max_accel_mps2: float = 3.5       # Max throttle acceleration [m/s^2]
    max_decel_mps2: float = 6.0       # Max braking deceleration [m/s^2]
    goal_tolerance_m: float = 3.0     # Stop threshold at goal [m]
    default_target_speed_mps: float = 7.5


class VehicleController:
    """
    Combines Pure Pursuit lateral steering and proportional speed regulation
    with hard emergency stop override when no valid trajectory exists.
    """

    def __init__(
        self,
        pursuit_config: Optional[PurePursuitConfig] = None,
        speed_config: Optional[SpeedControllerConfig] = None,
    ):
        self.pursuit = PurePursuitController(pursuit_config)
        self.speed_config = speed_config or SpeedControllerConfig()

    @property
    def last_lookahead_point(self) -> Optional[Tuple[float, float]]:
        return self.pursuit.last_lookahead_point

    def compute_command(
        self,
        state: VehicleState,
        trajectory: PlannedTrajectory,
        goal_pose: Optional[Tuple[float, float, float]] = None,
    ) -> ControlCommand:
        """
        Computes actuator ControlCommand from current VehicleState and PlannedTrajectory.
        Issues emergency stop immediately if trajectory is invalid or blocked.
        """
        t_now = state.timestamp

        # 1. Fail-safe stop if planner reports invalid / blocked path
        if not trajectory.valid or not trajectory.points:
            return ControlCommand(
                acceleration=0.0,
                steering=0.0,
                brake=1.0,
                emergency_stop=True,
                timestamp=t_now,
            )

        # 2. Check if already at goal
        if goal_pose is not None:
            dist_to_goal = math.hypot(goal_pose[0] - state.x, goal_pose[1] - state.y)
            if dist_to_goal <= self.speed_config.goal_tolerance_m:
                return ControlCommand(
                    acceleration=0.0,
                    steering=0.0,
                    brake=1.0,
                    emergency_stop=True,
                    timestamp=t_now,
                )

        # 3. Lateral steering control via Pure Pursuit
        steer_cmd, _ = self.pursuit.compute_steering(state, trajectory)

        # 4. Target velocity from nearest/lookahead point on trajectory
        target_speed = self.speed_config.default_target_speed_mps
        if trajectory.points:
            # Find nearest point to fetch reference speed
            nearest_pt = min(
                trajectory.points,
                key=lambda p: (p.x - state.x) ** 2 + (p.y - state.y) ** 2,
            )
            target_speed = nearest_pt.velocity

            # Slow down further for large steering angles
            steer_ratio = min(abs(steer_cmd) / self.pursuit.config.max_steer_rad, 1.0)
            target_speed = min(target_speed, max(3.0, self.speed_config.default_target_speed_mps * (1.0 - 0.55 * steer_ratio)))

        # Longitudinal error -> accel or brake
        speed_error = target_speed - state.velocity
        raw_accel = self.speed_config.kp_speed * speed_error

        if raw_accel >= -0.05:
            accel_cmd = max(0.0, min(self.speed_config.max_accel_mps2, raw_accel))
            brake_cmd = 0.0
        else:
            accel_cmd = 0.0
            brake_cmd = min(1.0, abs(raw_accel) / self.speed_config.max_decel_mps2)

        return ControlCommand(
            acceleration=float(accel_cmd),
            steering=float(steer_cmd),
            brake=float(brake_cmd),
            emergency_stop=False,
            timestamp=t_now,
        )

