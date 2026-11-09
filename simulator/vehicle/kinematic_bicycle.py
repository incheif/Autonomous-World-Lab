"""
Kinematic Bicycle Model for autonomous ground vehicles.
Referenced to the center of the rear axle using ISO 8855 conventions.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from autonomy.interfaces.types import VehicleState, ControlCommand


@dataclass
class VehicleParameters:
    """
    Physical dimensions and actuator constraints of the simulated vehicle.
    All parameters are strictly defined in SI units.
    """
    # Wheelbase & geometry
    wheelbase_m: float = 2.80       # Distance between front and rear axle [m]
    track_width_m: float = 1.60     # Lateral distance between left and right wheels [m]
    length_m: float = 4.50          # Total vehicle body length [m]
    width_m: float = 1.80           # Total vehicle body width [m]
    height_m: float = 1.45          # Vehicle body height [m]
    rear_overhang_m: float = 0.85   # Rear axle to rear bumper distance [m]
    front_overhang_m: float = 0.85  # Front axle to front bumper distance [m]
    tire_radius_m: float = 0.33     # Nominal tire rolling radius [m]

    # Actuator limits
    max_steer_rad: float = math.radians(35.0)    # Maximum steering angle [rad] (~0.61 rad)
    max_steer_rate_rps: float = math.radians(60.0) # Maximum steering angular rate [rad/s]
    max_accel_mps2: float = 3.5                  # Maximum positive longitudinal acceleration [m/s^2]
    max_decel_mps2: float = 6.0                  # Maximum braking deceleration [m/s^2] (positive value)
    max_speed_mps: float = 25.0                  # Forward speed limit [m/s] (~90 km/h)
    min_speed_mps: float = -3.5                  # Reverse speed limit [m/s] (~12 km/h)
    drag_coeff: float = 0.05                     # Resistance damping term [1/s]


class KinematicBicycleModel:
    """
    Discrete-time Kinematic Bicycle Model.
    
    Equations of motion (rear axle reference):
        x_dot = v * cos(psi)
        y_dot = v * sin(psi)
        psi_dot = (v / L) * tan(delta)
        v_dot = a - c_drag * v
        delta_dot = clamp((delta_target - delta) / dt, -rate_max, +rate_max)
    """

    def __init__(
        self,
        params: VehicleParameters = VehicleParameters(),
        initial_state: VehicleState = VehicleState(
            x=0.0, y=0.0, yaw=0.0, velocity=0.0, steering_angle=0.0, timestamp=0.0
        ),
    ):
        self.params: VehicleParameters = params
        self._state: VehicleState = initial_state

    @property
    def state(self) -> VehicleState:
        """Current ground-truth vehicle state."""
        return self._state

    def reset(
        self,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
        velocity: float = 0.0,
        steering_angle: float = 0.0,
        timestamp: float = 0.0,
    ) -> VehicleState:
        """Reset vehicle state to specified initial conditions."""
        self._state = VehicleState(
            x=float(x),
            y=float(y),
            yaw=self._normalize_angle(float(yaw)),
            velocity=float(velocity),
            steering_angle=float(steering_angle),
            angular_velocity=0.0,
            acceleration=0.0,
            timestamp=float(timestamp),
        )
        return self._state

    def step(self, command: ControlCommand, dt: float) -> VehicleState:
        """
        Integrate the kinematic equations forward by one fixed timestep dt.
        Returns the updated VehicleState.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        s = self._state

        # 1. Handle emergency stop override
        if command.emergency_stop:
            target_accel = -self.params.max_decel_mps2
            target_steer = 0.0
        else:
            # Handle throttle vs brake blending
            if command.brake > 0.0:
                brake_effort = max(0.0, min(1.0, command.brake))
                target_accel = -brake_effort * self.params.max_decel_mps2
            else:
                target_accel = max(
                    -self.params.max_decel_mps2,
                    min(self.params.max_accel_mps2, command.acceleration),
                )
            target_steer = max(
                -self.params.max_steer_rad,
                min(self.params.max_steer_rad, command.steering),
            )

        # 2. Rate-limit steering actuator
        steer_diff = target_steer - s.steering_angle
        max_steer_step = self.params.max_steer_rate_rps * dt
        actual_steer_delta = max(-max_steer_step, min(max_steer_step, steer_diff))
        new_steering = s.steering_angle + actual_steer_delta
        new_steering = max(-self.params.max_steer_rad, min(self.params.max_steer_rad, new_steering))

        # 3. 4th-Order Runge-Kutta (RK4) integration for kinematic state
        L = self.params.wheelbase_m
        c_drag = self.params.drag_coeff

        def derivatives(state_vec: np.ndarray, accel: float, steer: float) -> np.ndarray:
            # state_vec = [x, y, yaw, v]
            _, _, psi, v = state_vec
            dx = v * math.cos(psi)
            dy = v * math.sin(psi)
            dpsi = (v / L) * math.tan(steer)
            dv = accel - c_drag * v
            return np.array([dx, dy, dpsi, dv])

        current_vec = np.array([s.x, s.y, s.yaw, s.velocity], dtype=np.float64)

        k1 = derivatives(current_vec, target_accel, new_steering)
        k2 = derivatives(current_vec + 0.5 * dt * k1, target_accel, new_steering)
        k3 = derivatives(current_vec + 0.5 * dt * k2, target_accel, new_steering)
        k4 = derivatives(current_vec + dt * k3, target_accel, new_steering)

        new_vec = current_vec + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        new_x = float(new_vec[0])
        new_y = float(new_vec[1])
        new_yaw = self._normalize_angle(float(new_vec[2]))
        new_v = float(new_vec[3])

        # Clamp velocity to envelope
        new_v = max(self.params.min_speed_mps, min(self.params.max_speed_mps, new_v))

        # Prevent braking / emergency stop from causing reverse acceleration
        if command.emergency_stop or command.brake > 0.0:
            if s.velocity >= 0.0 and new_v <= 0.0:
                new_v = 0.0
            elif s.velocity <= 0.0 and new_v >= 0.0:
                new_v = 0.0
        elif target_accel < 0.0 and s.velocity > 0.0 and new_v < 0.0:
            new_v = 0.0

        # Near-zero speed clamping to avoid numerical drift when stopped
        if abs(new_v) < 1e-4 and abs(target_accel) < 1e-3:
            new_v = 0.0

        yaw_rate = (new_v / L) * math.tan(new_steering)
        actual_accel = (new_v - s.velocity) / dt

        self._state = VehicleState(
            x=new_x,
            y=new_y,
            yaw=new_yaw,
            velocity=new_v,
            steering_angle=new_steering,
            angular_velocity=yaw_rate,
            acceleration=actual_accel,
            timestamp=s.timestamp + dt,
        )
        return self._state

    def get_corners(self) -> List[Tuple[float, float]]:
        """
        Returns the 4 bounding box corner coordinates in world frame [(x, y)]:
        [Front-Left, Front-Right, Rear-Right, Rear-Left]
        """
        s = self._state
        p = self.params

        half_w = p.width_m / 2.0
        front_x = p.wheelbase_m + p.front_overhang_m
        rear_x = -p.rear_overhang_m

        local_corners = [
            (front_x, half_w),   # Front Left
            (front_x, -half_w),  # Front Right
            (rear_x, -half_w),   # Rear Right
            (rear_x, half_w),    # Rear Left
        ]

        cos_yaw = math.cos(s.yaw)
        sin_yaw = math.sin(s.yaw)

        world_corners = []
        for lx, ly in local_corners:
            wx = s.x + (lx * cos_yaw - ly * sin_yaw)
            wy = s.y + (lx * sin_yaw + ly * cos_yaw)
            world_corners.append((wx, wy))

        return world_corners

    @staticmethod
    def _normalize_angle(angle_rad: float) -> float:
        """Normalizes an angle to [-pi, pi]."""
        while angle_rad > math.pi:
            angle_rad -= 2.0 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2.0 * math.pi
        return angle_rad
