"""
Unit tests for 32-Beam LiDAR simulation.
Verifies ray-OBB geometry, occlusion, ground returns, noise/dropout reproducibility,
and scan serialization.
"""

import math
from pathlib import Path
import numpy as np
import pytest

from autonomy.interfaces.types import VehicleState, Obstacle
from simulator.sensors.lidar import LidarSensor, LidarParameters


def test_lidar_initialization():
    params = LidarParameters(
        num_channels=32,
        v_fov_min_rad=math.radians(-20.0),
        v_fov_max_rad=math.radians(10.0),
        h_resolution_rad=math.radians(1.0),
    )
    sensor = LidarSensor(params=params)

    assert params.num_channels == 32
    # 360 azimuth steps * 32 elevation channels = 11,520 beams
    assert sensor.total_beams == 360 * 32


def test_lidar_known_geometry_distance():
    """
    Vehicle at origin (0, 0, 0), yaw = 0.
    LiDAR mounted at (1.4, 0.0, 1.6).
    Obstacle box centered at (x=15.0, y=0.0, z=0.0) with length=2.0 (half-length=1.0).
    Front surface is at x = 14.0 m.
    Horizontal ray at azimuth = 0, elevation = 0 must hit front surface at range = 14.0 - 1.4 = 12.6 m.
    """
    params = LidarParameters(
        num_channels=33,
        v_fov_min_rad=math.radians(-16.0),
        v_fov_max_rad=math.radians(16.0),
        h_resolution_rad=math.radians(1.0),
        range_noise_std_m=0.0,
        dropout_prob=0.0,
    )
    sensor = LidarSensor(params=params)

    vehicle = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, steering_angle=0.0)
    obs = Obstacle(
        obstacle_id="target_box",
        x=15.0, y=0.0, z=0.0,
        length=2.0, width=2.0, height=3.0,
        yaw=0.0,
    )

    obs_result = sensor.scan(vehicle, [obs])
    gt = obs_result.ground_truth_scan

    # Find the beam closest to (azimuth=0, elevation=0)
    az_err = np.abs(gt.angles_azimuth)
    el_err = np.abs(gt.angles_elevation)
    center_beam_idx = np.argmin(az_err + el_err)

    assert gt.valid_mask[center_beam_idx]
    measured_range = gt.ranges[center_beam_idx]
    expected_range = 14.0 - 1.40  # 12.60m
    assert abs(measured_range - expected_range) < 0.05


def test_lidar_occlusion():
    """
    Nearer obstacle at x=15m must completely occlude farther obstacle at x=30m.
    No returns should penetrate through to x=30m.
    """
    params = LidarParameters(
        num_channels=32,
        range_noise_std_m=0.0,
        dropout_prob=0.0,
        ground_detection=False, # Disable ground to isolate obstacles
    )
    sensor = LidarSensor(params=params)

    vehicle = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, steering_angle=0.0)
    near_obs = Obstacle("near_box", x=15.0, y=0.0, z=0.0, length=2.0, width=4.0, height=4.0)
    far_obs = Obstacle("far_box", x=30.0, y=0.0, z=0.0, length=2.0, width=4.0, height=4.0)

    obs_result = sensor.scan(vehicle, [near_obs, far_obs])
    gt = obs_result.ground_truth_scan

    # For beams pointing near front (azimuth in [-5 deg, +5 deg], elevation in [-5 deg, +5 deg])
    forward_mask = (np.abs(gt.angles_azimuth) < math.radians(5.0)) & (np.abs(gt.angles_elevation) < math.radians(5.0))
    valid_forward_ranges = gt.ranges[forward_mask & gt.valid_mask]

    assert len(valid_forward_ranges) > 0
    # All forward hits must be on the near box (range ~ 12-14m), none on the far box (> 25m)
    assert np.all(valid_forward_ranges < 20.0)


def test_lidar_noise_and_dropout_reproducibility():
    """
    Verify that with the same seed, noise and dropouts are 100% reproducible.
    With different seeds, noise values differ.
    """
    params = LidarParameters(
        num_channels=16,
        range_noise_std_m=0.05,
        dropout_prob=0.10,
    )

    s1 = LidarSensor(params=params, seed=1234)
    s2 = LidarSensor(params=params, seed=1234)
    s3 = LidarSensor(params=params, seed=9999)

    vehicle = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, steering_angle=0.0)
    obs = Obstacle("box", x=10.0, y=0.0, z=0.0, length=2.0, width=2.0, height=2.0)

    res1 = s1.scan(vehicle, [obs])
    res2 = s2.scan(vehicle, [obs])
    res3 = s3.scan(vehicle, [obs])

    # Replicas with identical seed must match identically
    np.testing.assert_array_equal(res1.noisy_scan.ranges, res2.noisy_scan.ranges)
    np.testing.assert_array_equal(res1.noisy_scan.valid_mask, res2.noisy_scan.valid_mask)

    # Different seeds must produce different noise / dropouts
    assert not np.array_equal(res1.noisy_scan.ranges, res3.noisy_scan.ranges)


def test_lidar_save_and_replay(tmp_path: Path):
    """
    Verify save_scan_npz and load_scan_npz roundtrip.
    """
    sensor = LidarSensor(params=LidarParameters(num_channels=16))
    vehicle = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, steering_angle=0.0)
    obs = Obstacle("box", x=12.0, y=1.0, z=0.0, length=2.0, width=2.0, height=2.0)

    res = sensor.scan(vehicle, [obs])
    scan = res.noisy_scan

    file_path = str(tmp_path / "test_scan.npz")
    sensor.save_scan_npz(scan, file_path)
    loaded_scan = LidarSensor.load_scan_npz(file_path)

    np.testing.assert_allclose(scan.ranges, loaded_scan.ranges)
    np.testing.assert_allclose(scan.angles_azimuth, loaded_scan.angles_azimuth)
    np.testing.assert_allclose(scan.angles_elevation, loaded_scan.angles_elevation)
    np.testing.assert_array_equal(scan.valid_mask, loaded_scan.valid_mask)
    assert scan.timestamp == loaded_scan.timestamp

