"""
Unit tests for Perception Subsystem: Ground Filtering, Euclidean Clustering, and OBB Fitting.
"""

import numpy as np
import pytest

from autonomy.interfaces.types import VehicleState, LidarScan, Obstacle
from autonomy.perception import (
    GroundFilter,
    GroundFilterConfig,
    EuclideanClusterer,
    EuclideanClustererConfig,
    BoxFitter,
    BoxFitterConfig,
    PerceptionPipeline,
)
from simulator.simulation import Simulation
from simulator.world.road import RoadMap


def test_ground_filter_removes_road_returns():
    """Verifies ground plane points are isolated from elevated obstacle points."""
    filter_mod = GroundFilter(GroundFilterConfig(
        sensor_mount_offset=(0.0, 0.0, 1.80),
        ground_height_threshold=0.20,
        filter_vehicle_body=False,
    ))
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, velocity=0.0, steering_angle=0.0)

    # Synthetic local points: ground returns have local z = -1.8 (world z = 0.0)
    # Obstacle points have local z = 0.0 (world z = 1.8)
    n_pts = 100
    ground_local = np.column_stack((
        np.linspace(5, 25, n_pts),
        np.zeros(n_pts),
        np.full(n_pts, -1.80),
    ))
    obs_local = np.column_stack((
        np.linspace(10, 14, 20),
        np.full(20, 2.0),
        np.full(20, -0.5),  # world z = 1.30m
    ))

    all_local = np.vstack((ground_local, obs_local))
    ranges = np.linalg.norm(all_local, axis=1)
    az = np.arctan2(all_local[:, 1], all_local[:, 0])
    el = np.arcsin(all_local[:, 2] / ranges)

    scan = LidarScan(
        ranges=ranges,
        angles_azimuth=az,
        angles_elevation=el,
        intensities=np.ones(len(ranges)),
        valid_mask=np.ones(len(ranges), dtype=bool),
        timestamp=0.0,
    )

    obstacle_pts, ground_pts = filter_mod.filter(scan, state)

    assert len(ground_pts) == n_pts
    assert len(obstacle_pts) == 20
    assert np.all(obstacle_pts[:, 2] > 0.20)
    assert np.all(ground_pts[:, 2] <= 0.20)


def test_euclidean_clustering_segments_two_objects():
    """Verifies that two spatially separated point groups are segmented into 2 distinct clusters."""
    clusterer = EuclideanClusterer(EuclideanClustererConfig(eps=1.0, min_points=5))

    # Cluster 1: around (5, 0, 1)
    np_rand = np.random.RandomState(42)
    c1 = np_rand.uniform(-0.4, 0.4, size=(30, 3)) + np.array([5.0, 0.0, 1.0])

    # Cluster 2: around (15, 3, 1) - 10m away
    c2 = np_rand.uniform(-0.4, 0.4, size=(25, 3)) + np.array([15.0, 3.0, 1.0])

    # Noise points: isolated single points
    noise = np.array([[30.0, 30.0, 1.0], [50.0, -10.0, 1.0]])

    all_points = np.vstack((c1, c2, noise))
    clusters = clusterer.cluster(all_points)

    assert len(clusters) == 2
    # Verify sizes
    sizes = sorted([len(c) for c in clusters])
    assert sizes == [25, 30]


def test_box_fitting_known_rectangle():
    """Verifies that BoxFitter recovers the center and dimensions of a known rectangle."""
    fitter = BoxFitter(BoxFitterConfig())

    # Points around box: center=(10, 5), length=4.0, width=2.0, height=1.5, yaw=0.0
    x = np.linspace(8.0, 12.0, 20)
    y = np.linspace(4.0, 6.0, 15)
    z = np.linspace(0.2, 1.7, 10)

    # Perimeter / surface points
    pts_list = []
    for xi in x:
        for yi in [4.0, 6.0]:
            for zi in [0.2, 1.7]:
                pts_list.append([xi, yi, zi])
    for yi in y:
        for xi in [8.0, 12.0]:
            for zi in [0.2, 1.7]:
                pts_list.append([xi, yi, zi])

    cluster = np.array(pts_list)
    obs = fitter.fit_single_box(cluster, obstacle_id="test_box", timestamp=1.0)

    assert obs is not None
    assert obs.obstacle_id == "test_box"
    assert pytest.approx(obs.x, abs=0.2) == 10.0
    assert pytest.approx(obs.y, abs=0.2) == 5.0
    assert pytest.approx(obs.length, abs=0.2) == 4.0
    assert pytest.approx(obs.width, abs=0.2) == 2.0
    assert pytest.approx(obs.height, abs=0.1) == 1.5


def test_end_to_end_perception_with_simulation():
    """Tests that simulation detects world obstacles via perception pipeline without ground-truth cheating."""
    road = RoadMap(
        road_type="straight",
        road_length_m=80.0,
        obstacles=[
            Obstacle(obstacle_id="box_ahead", x=15.0, y=0.0, z=0.0, length=2.5, width=2.0, height=1.5, yaw=0.0)
        ]
    )
    sim = Simulation(road=road)
    state, status = sim.step()

    assert sim.last_perception is not None
    detected = sim.last_perception.detected_obstacles

    assert len(detected) >= 1
    d_obs = detected[0]
    # Check estimated obstacle is near x=15.0, y=0.0
    assert pytest.approx(d_obs.x, abs=1.5) == 15.0
    assert pytest.approx(d_obs.y, abs=1.0) == 0.0
    assert d_obs.confidence > 0.5
