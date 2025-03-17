import numpy as np
from typing import Union, List

class Plane:
    """Represents a plane in 3D space using point-normal form."""
    
    def __init__(self, point: np.ndarray, normal: np.ndarray):
        """Initialize plane with point and normal vector.
        
        Args:
            point (np.ndarray): Point on plane (shape: (3,))
            normal (np.ndarray): Normal vector (shape: (3,))
        """
        self.point = np.asarray(point, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        # Normalize the normal vector
        self.normal = self.normal / np.linalg.norm(self.normal)

    def calculate_above_plane(self, points: np.ndarray) -> np.ndarray:
        """Determine which points are above the plane.
        
        Args:
            points (np.ndarray): Array of points to test (shape: (N, 3))
            
        Returns:
            np.ndarray: Boolean array indicating which points are above plane
        """
        # Calculate signed distance from points to plane
        distances = self.signed_distance(points)
        return distances > 0

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Calculate signed distance from points to plane.
        
        Args:
            points (np.ndarray): Array of points (shape: (N, 3))
            
        Returns:
            np.ndarray: Signed distances (positive above plane)
        """
        return np.dot(points - self.point, self.normal)

    def calculate_intersection(self, edge_vertices: List[np.ndarray]) -> Union[np.ndarray, None]:
        """Calculate intersection point between line segment and plane.
        
        Args:
            edge_vertices (List[np.ndarray]): Two vertices defining line segment
            
        Returns:
            np.ndarray: Intersection point if it exists, None otherwise
        """
        p1, p2 = edge_vertices
        # Convert to numpy arrays if not already
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        
        # Calculate intersection parameter
        d = np.dot(p2 - p1, self.normal)
        if abs(d) < 1e-10:  # Line is parallel to plane
            return None
            
        t = np.dot(self.point - p1, self.normal) / d
        
        # Check if intersection is within line segment
        if 0 <= t <= 1:
            return p1 + t * (p2 - p1)
        return None