"""
Perception pipeline orchestrating ground removal, point clustering, and 3D bounding box detection.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from autonomy.interfaces.types import LidarScan, VehicleState, DetectedObstacle
from .ground_filter import GroundFilter, GroundFilterConfig
from .clustering import EuclideanClusterer, EuclideanClustererConfig
from .box_fitting import BoxFitter, BoxFitterConfig


@dataclass
class PerceptionResult:
    """
    Perception output for a single LiDAR observation.
    """
    obstacle_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    ground_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    clusters: List[np.ndarray] = field(default_factory=list)
    detected_obstacles: List[DetectedObstacle] = field(default_factory=list)
    timestamp: float = 0.0


class PerceptionPipeline:
    """
    Autonomous perception subsystem.
    Consumes raw/noisy LiDAR scans and estimated vehicle pose.
    Does NOT access ground-truth world obstacles.
    """

    def __init__(
        self,
        ground_config: Optional[GroundFilterConfig] = None,
        cluster_config: Optional[EuclideanClustererConfig] = None,
        box_config: Optional[BoxFitterConfig] = None,
    ):
        self.ground_filter = GroundFilter(ground_config)
        self.clusterer = EuclideanClusterer(cluster_config)
        self.box_fitter = BoxFitter(box_config)

    def process(
        self,
        scan: LidarScan,
        vehicle_state: VehicleState,
    ) -> PerceptionResult:
        """
        Executes end-to-end perception on a single LiDAR scan.
        """
        # 1. Ground removal & coordinate transformation to world frame
        obs_points, ground_points = self.ground_filter.filter(scan, vehicle_state)

        if len(obs_points) == 0:
            return PerceptionResult(
                obstacle_points=obs_points,
                ground_points=ground_points,
                clusters=[],
                detected_obstacles=[],
                timestamp=scan.timestamp,
            )

        # 2. Euclidean spatial clustering
        clusters = self.clusterer.cluster(obs_points)

        # 3. Oriented bounding box fitting
        detected_obstacles = self.box_fitter.fit_boxes(clusters, timestamp=scan.timestamp)

        return PerceptionResult(
            obstacle_points=obs_points,
            ground_points=ground_points,
            clusters=clusters,
            detected_obstacles=detected_obstacles,
            timestamp=scan.timestamp,
        )

