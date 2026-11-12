"""
Euclidean Point Cloud Clustering.
Groups filtered obstacle returns into discrete object clusters using spatial KD-Trees.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class EuclideanClustererConfig:
    """
    Configuration parameters for Euclidean clustering.
    """
    eps: float = 1.20           # Maximum Euclidean distance between connected points [m]
    min_points: int = 4         # Minimum points to form a valid obstacle cluster
    max_points: int = 4000      # Maximum points in a single cluster
    use_2d_distance: bool = True # Use 2D (XY) Euclidean distance for ground vehicles


class EuclideanClusterer:
    """
    Segments 3D point clouds into discrete object clusters using cKDTree nearest-neighbor queries.
    """

    def __init__(self, config: Optional[EuclideanClustererConfig] = None):
        self.config = config or EuclideanClustererConfig()

    def cluster(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Segments input points into a list of point clusters.

        Args:
            points: np.ndarray of shape (N, 3)

        Returns:
            clusters: List of np.ndarray, each of shape (K, 3)
        """
        if len(points) < self.config.min_points:
            return []

        # For ground vehicles, horizontal spatial proximity (XY) is the primary clustering metric
        query_points = points[:, :2] if self.config.use_2d_distance else points
        tree = cKDTree(query_points)

        n_points = len(points)
        visited = np.zeros(n_points, dtype=bool)
        clusters: List[np.ndarray] = []

        # Find neighbors for all points within eps
        neighbors_list = tree.query_ball_point(query_points, r=self.config.eps)

        for i in range(n_points):
            if visited[i]:
                continue

            # Breadth-first search / region growing for connected component
            cluster_indices = []
            queue = [i]
            visited[i] = True

            head = 0
            while head < len(queue):
                curr = queue[head]
                head += 1
                cluster_indices.append(curr)

                if len(cluster_indices) >= self.config.max_points:
                    break

                for neighbor in neighbors_list[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

            if len(cluster_indices) >= self.config.min_points:
                clusters.append(points[cluster_indices])

        return clusters

