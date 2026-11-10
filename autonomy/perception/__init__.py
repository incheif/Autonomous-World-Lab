"""
Perception package for point cloud processing, clustering, and obstacle detection.
"""

from .ground_filter import GroundFilter, GroundFilterConfig
from .clustering import EuclideanClusterer, EuclideanClustererConfig
from .box_fitting import BoxFitter, BoxFitterConfig
from .pipeline import PerceptionPipeline, PerceptionResult

__all__ = [
    "GroundFilter",
    "GroundFilterConfig",
    "EuclideanClusterer",
    "EuclideanClustererConfig",
    "BoxFitter",
    "BoxFitterConfig",
    "PerceptionPipeline",
    "PerceptionResult",
]
