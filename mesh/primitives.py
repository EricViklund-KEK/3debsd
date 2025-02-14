import numpy as np
from typing import List, Tuple, Optional, Set





class Plane:
    """Represents a plane in 3D space using point-normal form."""
    
    def __init__(self, point: np.ndarray, normal: np.ndarray):
        """Initialize plane with a point and normal vector.
        
        Args:
            point (np.ndarray): A point on the plane (shape: (3,))
            normal (np.ndarray): Normal vector to the plane (shape: (3,))
        """
        self.point = np.asarray(point, dtype=float)
        self.normal = np.asarray(normal, dtype=float)
        # Normalize the normal vector
        self.normal /= np.linalg.norm(self.normal)
        
    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Calculate signed distance from points to the plane.
        
        Args:
            points (np.ndarray): Points to calculate distance to (shape: (N, 3))
            
        Returns:
            np.ndarray: Signed distances (shape: (N,))
        """
        return np.dot(points - self.point, self.normal)

class Edge:
    """Represents a line segment in 3D space."""
    
    def __init__(self, start: np.ndarray, end: np.ndarray):
        """Initialize line segment with start and end points.
        
        Args:
            start (np.ndarray): Start point (shape: (3,))
            end (np.ndarray): End point (shape: (3,))
        """
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        self.direction = self.end - self.start
        self.length = np.linalg.norm(self.direction)
        
    def point_at(self, t: float) -> np.ndarray:
        """Get point along line at parameter t.
        
        Args:
            t (float): Parameter in range [0,1]
            
        Returns:
            np.ndarray: Point on line (shape: (3,))
        """
        return self.start + t * self.direction
    
    def intersect_plane(self, plane: Plane) -> Tuple['Edge', Tuple[Optional[np.ndarray], Optional[float]], 'Edge']:
        """Calculate intersection point between line segment and plane.
        
        Args:
            plane (Plane): Plane to intersect with
            
        Returns:
            tuple: (edge_before, (intersection_point, parameter), edge_after)
                  where edge_before is the part before intersection,
                  edge_after is the part after intersection,
                  and intersection_point/parameter are None if no intersection
        """
        direction = self.end - self.start
        denom = np.dot(direction, plane.normal)
        
        # Check if line is parallel to plane
        if abs(denom) < 1e-10:
            return self, (None, None), None
            
        t = np.dot(plane.point - self.start, plane.normal) / denom
        
        # Return intersection point and parameter if within segment
        if 0 <= t <= 1:
            intersection_point = self.point_at(t)
            edge_before = Edge(self.start, intersection_point)
            edge_after = Edge(intersection_point, self.end)
            return edge_before, (intersection_point, t), edge_after
        
        return self, (None, None), None

class Face:
    """Represents a face in 3D space."""
    
    def __init__(self, edges: List[Edge]):
        """Initialize face with edges.
        
        Args:
            edges (List[Edge]): List of edges forming the face boundary
        """
        assert len(edges) >= 3, "Face must have at least 3 edges"
        self.edges = edges
        self._compute_normal()
        
    def _compute_normal(self):
        """Compute face normal using first three edges."""
        if len(self.edges) >= 3:
            # Get three non-collinear points from edges
            p0 = self.edges[0].start
            p1 = self.edges[0].end
            p2 = None
            
            # Find a third point not collinear with p0 and p1
            for edge in self.edges[1:]:
                test_point = edge.end
                v1 = p1 - p0
                v2 = test_point - p0
                cross = np.cross(v1, v2)
                if np.linalg.norm(cross) > 1e-10:
                    p2 = test_point
                    break
            
            if p2 is not None:
                v1 = p1 - p0
                v2 = p2 - p0
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    self.normal = normal / norm
                    return
                    
        self.normal = None

    def intersect_plane(self, plane: Plane) -> Tuple['Face', Edge, List['Face']]:
        """Calculate the intersection edge between the face and a plane.
        
        Args:
            plane (Plane): Plane to intersect with
            
        Returns:
            tuple: (modified_face, intersection_edge, new_faces) where modified_face is
                  this face after intersection, intersection_edge is the edge created by
                  the intersection, and new_faces are any additional faces created by splitting
        """
        intersection_points = []
        intersection_edges = []
        modified_edges = []
        split_edges = []

        # Calculate the intersection of each edge
        for edge in self.edges:
            edge_before, (point, t), edge_after = edge.intersect_plane(plane)
            if point is not None:
                intersection_points.append(point)
                if edge_after is not None:
                    split_edges.extend([edge_before, edge_after])
                    modified_edges.append(edge_before)
                else:
                    modified_edges.append(edge)

        # If fewer than 2 intersection points, no valid intersection
        if len(intersection_points) < 2:
            return self, None, []

        # Create new edge from intersection points
        intersection_edge = Edge(intersection_points[0], intersection_points[1])
        
        # Create new faces if needed (face is split by plane)
        new_faces = []
        if len(intersection_points) > 2:
            # Sort intersection points to form a valid polygon
            center = np.mean(intersection_points, axis=0)
            normal = np.cross(intersection_points[1] - intersection_points[0],
                            intersection_points[2] - intersection_points[0])
            angles = np.arctan2(
                np.dot(np.cross(intersection_points - center, normal[:, None].T),
                      intersection_points[0] - center),
                np.dot(intersection_points - center, intersection_points[0] - center))
            sorted_indices = np.argsort(angles)
            sorted_points = intersection_points[sorted_indices]
            
            # Create edges for the new faces
            new_edges = [Edge(sorted_points[i], sorted_points[(i+1)%len(sorted_points)])
                        for i in range(len(sorted_points))]
            
            # TODO: Handle disconnected edge loops in the future for concave face support.
            
            # Create the new faces
            new_faces = [Face(new_edges)]

        # Update this face with modified edges
        self.edges = modified_edges
        self._compute_normal()

        return self, intersection_edge, new_faces

class Domain:
    """Represents a convex volume in 3D space composed of Faces"""

    def __init__(self, faces: List[Face]):
        """Initialize Domain with faces.
        
        Args:
            faces (List[Face]): List of faces forming the domain boundary
        """
        assert len(faces) >= 4, "Domain must have at least 4 faces"
        self.faces = faces
    
    def intersect_plane(self, plane: Plane) -> Tuple['Domain', Face, List['Domain']]:
        """Calculate intersection face between the domain and a plane.
        
        Args:
            plane (Plane): Plane to intersect with
            
        Returns:
            tuple: (modified_domain, intersection_face, new_domains) where modified_domain
                  is this domain after intersection, intersection_face is the face created
                  by the intersection, and new_domains are any additional domains created
                  by splitting
        """
        intersection_edges = []
        modified_faces = []
        new_faces = []

        # Get intersections for each face
        for face in self.faces:
            face, edge, faces = face.intersect_plane(plane)
            if edge:
                intersection_edges.append(edge)
                new_faces.extend(faces)

        # Create intersection face if we have edges
        intersection_face = None
        if intersection_edges:
            # Order edges to form a valid face
            vertices = []
            current_edge = intersection_edges[0]
            ordered_edges = [current_edge]
            
            while len(ordered_edges) < len(intersection_edges):
                # Find next edge that connects to current edge
                found = False
                for edge in intersection_edges:
                    if edge not in ordered_edges:
                        if np.allclose(edge.start, current_edge.end):
                            vertices.append(current_edge.start)
                            current_edge = edge
                            ordered_edges.append(edge)
                            found = True
                            break
                if not found:
                    break
            
            # Add final vertices
            if ordered_edges:
                vertices.extend([ordered_edges[-1].start, ordered_edges[-1].end])
            
            if len(vertices) >= 3:
                intersection_face = Face(ordered_edges)

        # Create new domains if volume is split
        new_domains = []
        if intersection_face and new_faces:
            # Group faces into new domains based on which side of the plane they're on
            pos_faces = []
            neg_faces = []
            for face in new_faces:
                if np.dot(face.normal, plane.normal) > 0:
                    pos_faces.append(face)
                else:
                    neg_faces.append(face)
            
            if len(pos_faces) >= 4:
                self.faces = pos_faces
            if len(neg_faces) >= 4:
                new_domains.append(Domain(neg_faces))

        return self, intersection_face, new_domains
