"""
Unit tests for KinematicBicycleModel.
Verifies straight line motion, turning circle radius R = L / tan(delta),
steering rate limits, braking, and deterministic replay.
"""

import math
import pytest
from autonomy.interfaces.types import ControlCommand
from simulator.vehicle.kinematic_bicycle import KinematicBicycleModel, VehicleParameters


def test_straight_line_acceleration():
    model = KinematicBicycleModel(params=VehicleParameters(drag_coeff=0.0))
    model.reset(x=0.0, y=0.0, yaw=0.0, velocity=0.0)

    cmd = ControlCommand(acceleration=2.0, steering=0.0)
    dt = 0.02
    # 50 steps = 1.0 second
    for _ in range(50):
        model.step(cmd, dt)

    s = model.state
    # v = a * t = 2.0 * 1.0 = 2.0 m/s
    assert abs(s.velocity - 2.0) < 0.05
    # x approx 0.5 * a * t^2 = 1.0 m
    assert abs(s.x - 1.0) < 0.05
    assert abs(s.y) < 1e-4
    assert abs(s.yaw) < 1e-4


def test_turning_circle_radius():
    """
    For a kinematic bicycle with rear axle reference:
    Yaw rate omega = (v / L) * tan(delta)
    Turning radius R = L / tan(delta) = v / omega
    """
    params = VehicleParameters(wheelbase_m=2.8, drag_coeff=0.0)
    model = KinematicBicycleModel(params=params)

    # Constant forward speed 5.0 m/s
    model.reset(x=0.0, y=0.0, yaw=0.0, velocity=5.0)

    steer_deg = 15.0
    steer_rad = math.radians(steer_deg)
    cmd = ControlCommand(acceleration=0.0, steering=steer_rad)

    dt = 0.02
    # Run until steering reaches steady-state
    for _ in range(100):
        model.step(cmd, dt)

    s = model.state
    expected_radius = params.wheelbase_m / math.tan(steer_rad)
    measured_radius = s.velocity / s.angular_velocity

    assert abs(measured_radius - expected_radius) < 0.05


def test_steering_rate_and_saturation_limits():
    params = VehicleParameters(
        max_steer_rad=math.radians(35.0),
        max_steer_rate_rps=math.radians(60.0),
    )
    model = KinematicBicycleModel(params=params)
    model.reset()

    # Request extreme steering beyond saturation (e.g. 50 deg)
    cmd = ControlCommand(steering=math.radians(50.0))
    dt = 0.1  # 0.1s

    # After 1 step (0.1s), steer angle should not exceed 60 deg/s * 0.1s = 6 deg
    s1 = model.step(cmd, dt)
    assert abs(s1.steering_angle - math.radians(6.0)) < 1e-3

    # Run for 2.0s, steer angle must strictly saturate at max_steer_rad (35 deg)
    for _ in range(20):
        model.step(cmd, dt)
    assert abs(model.state.steering_angle - params.max_steer_rad) < 1e-4


def test_emergency_stop():
    model = KinematicBicycleModel()
    model.reset(velocity=15.0)

    # Trigger emergency stop
    cmd = ControlCommand(emergency_stop=True)
    dt = 0.02
    # Should decelerate rapidly to 0 and clamp
    for _ in range(300):
        model.step(cmd, dt)

    assert model.state.velocity == 0.0


def test_deterministic_replay():
    """
    Verifies that running two identical command sequences on separate instances
    produces identical state vectors to within floating point precision.
    """
    m1 = KinematicBicycleModel()
    m2 = KinematicBicycleModel()

    m1.reset(x=5.0, y=2.0, yaw=0.5, velocity=3.0)
    m2.reset(x=5.0, y=2.0, yaw=0.5, velocity=3.0)

    commands = [
        ControlCommand(acceleration=1.5, steering=0.2),
        ControlCommand(acceleration=2.0, steering=-0.1),
        ControlCommand(acceleration=-1.0, steering=0.3),
        ControlCommand(acceleration=0.0, steering=-0.2),
    ]

    dt = 0.02
    for cmd in commands:
        for _ in range(50):
            s1 = m1.step(cmd, dt)
            s2 = m2.step(cmd, dt)
            assert s1.x == s2.x
            assert s1.y == s2.y
            assert s1.yaw == s2.yaw
            assert s1.velocity == s2.velocity
            assert s1.steering_angle == s2.steering_angle
