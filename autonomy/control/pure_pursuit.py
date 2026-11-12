"""
Pure Pursuit Geometric Path Tracking Controller.
Computes front-wheel steering angle to smoothly converge onto a planned trajectory.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from autonomy.interfaces.types import VehicleState, PlannedTrajectory, TrajectoryPoint


@dataclass
class PurePursuitConfig:
    """Configuration parameters for Pure Pursuit steering."""
    wheelbase_m: float = 2.80          # Distance between rear and front axles [m]
    min_lookahead_m: float = 4.0       # Minimum lookahead distance [m]
    max_lookahead_m: float = 16.0      # Maximum lookahead distance [m]
    lookahead_gain_s: float = 0.65     # Velocity-proportional lookahead gain [s]
    max_steer_rad: float = math.radians(35.0)


class PurePursuitController:
    """
    Geometric Pure Pursuit path tracker referenced to the rear axle.
    """

    def __init__(self, config: Optional[PurePursuitConfig] = None):
        self.config = config or PurePursuitConfig()
        self.last_lookahead_point: Optional[Tuple[float, float]] = None

    def compute_steering(
        self,
        state: VehicleState,
        trajectory: PlannedTrajectory,
    ) -> Tuple[float, Optional[Tuple[float, float]]]:
        """
        Computes steering angle [rad] and target lookahead point (x, y).
        """
        if not trajectory.valid or not trajectory.points:
            self.last_lookahead_point = None
            return 0.0, None

        points = trajectory.points
        v = max(abs(state.velocity), 0.0)
        ld = max(
            self.config.min_lookahead_m,
            min(self.config.max_lookahead_m, self.config.min_lookahead_m + self.config.lookahead_gain_s * v),
        )

        # Find nearest point on trajectory first so we only look forward along the path
        nearest_idx = 0
        min_d2 = float("inf")
        for i, pt in enumerate(points):
            d2 = (pt.x - state.x) ** 2 + (pt.y - state.y) ** 2
            if d2 < min_d2:
                min_d2 = d2
                nearest_idx = i

        # Search forward along trajectory for point at distance >= ld
        target_pt = points[-1]
        for i in range(nearest_idx, len(points)):
            dist = math.hypot(points[i].x - state.x, points[i].y - state.y)
            if dist >= ld:
                if i > nearest_idx:
                    # Linearly interpolate along segment for smooth steering
                    p_prev = points[i - 1]
                    p_curr = points[i]
                    d_prev = math.hypot(p_prev.x - state.x, p_prev.y - state.y)
                    denom = max(dist - d_prev, 1e-6)
                    frac = max(0.0, min(1.0, (ld - d_prev) / denom))
                    ix = p_prev.x + frac * (p_curr.x - p_prev.x)
                    iy = p_prev.y + frac * (p_curr.y - p_prev.y)
                    target_pt = TrajectoryPoint(x=ix, y=iy, velocity=p_curr.velocity)
                else:
                    target_pt = points[i]
                break

        # Actual effective lookahead distance to chosen target point
        eff_ld = max(math.hypot(target_pt.x - state.x, target_pt.y - state.y), 0.5)

        # Transform target point angle to vehicle heading frame
        angle_to_target = math.atan2(target_pt.y - state.y, target_pt.x - state.x)
        alpha = self._normalize_angle(angle_to_target - state.yaw)

        # Pure Pursuit steering law: delta = atan2(2 * L * sin(alpha), Ld)
        steer = math.atan2(2.0 * self.config.wheelbase_m * math.sin(alpha), eff_ld)
        steer = max(-self.config.max_steer_rad, min(self.config.max_steer_rad, steer))

        lookahead_xy = (float(target_pt.x), float(target_pt.y))
        self.last_lookahead_point = lookahead_xy
        return steer, lookahead_xy

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

