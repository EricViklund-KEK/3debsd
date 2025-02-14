import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from mesh.mesh3d import Mesh3D
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

class MeshModifier:
    """Class for performing geometric operations and modifications on Mesh3D objects.
    
    TODO: Implement remove methods.
    TODO: Add overload support for create/delete methods for single object and list of object input.
    """
    
    def __init__(self, mesh: Mesh3D):
        """Initialize with a Mesh3D object.
        
        Args:
            mesh (Mesh3D): The mesh to modify
        """
        self.mesh = mesh
        self._new_vertex_index = mesh.num_vertices
        self._new_edge_index = mesh.num_edges
        self._new_face_index = mesh.num_faces
        self._new_domain_index = mesh.num_domains
        
    def add_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Add new vertices to the mesh.
        
        Args:
            vertices (np.ndarray): Vertices to add (shape: (N, 3))
            
        Returns:
            np.ndarray: Indices of added vertices
        """
        vertices = np.asarray(vertices, dtype=float)
        n_new = len(vertices)
        
        # Add vertices to mesh
        self.mesh.vertices = np.vstack([self.mesh.vertices, vertices])

        # Add rows to T_VE
        self.mesh.T_VE = vstack((self.mesh.T_VE,csr_matrix((n_new,self.mesh.T_VE.shape[1]))))
        
        # Return indices of new vertices
        new_indices = np.arange(self._new_vertex_index, 
                              self._new_vertex_index + n_new)
        self._new_vertex_index += n_new

        edit_dict = {'added_vertices': new_indices}
        
        # Invalidate KD-tree since we modified vertices
        self.mesh._kdtree = None
        
        return edit_dict
    
    def delete_vertices(self, vertex_indices: List[int]) -> List[int]:
        """Removes vertices at the given indices and updates the mesh.
        
        Args:
            vertex_indices (List[int]): Indices of vertices to remove
            
        Returns:
            List[int]: Indices of affected edges
        """
        # Find edges that use these vertices
        affected_edges = set()
        for col in range(self.mesh.T_VE.shape[1]):
            vert_indices = self.mesh.T_VE[:,col].nonzero()[0]
            if any(v in vertex_indices for v in vert_indices):
                affected_edges.add(col)
        
        # Remove vertices from vertex array
        mask = np.ones(self.mesh.num_vertices, dtype=bool)
        mask[vertex_indices] = False
        self.mesh.vertices = self.mesh.vertices[mask]
        
        # Update vertex indices in T_VE
        new_indices = np.cumsum(mask) - 1
        for col in range(self.mesh.T_VE.shape[1]):
            if col not in affected_edges:
                vert_indices = self.mesh.T_VE[:,col].nonzero()[0]
                self.mesh.T_VE[new_indices[vert_indices], col] = True
                
        if affected_edges:
            edit_dict = self.remove_edges(list(affected_edges))
        else:
            edit_dict = {}

        edit_dict['removed_vertices'] = vert_indices

        return edit_dict
        
    def create_edge(self, edge: Edge) -> dict:
        """Create a new edge between vertices.
        
        Args:
            edge (Edge): Edge to add to the mesh
            
        Returns:
            dict: edit dictionary
        """
        # Add vertices if they don't exist
        edit_dict = self.add_vertices(np.vstack((edge.start,edge.end)))

        # Create new column for T_VE
        new_col = csr_matrix((self.mesh.num_vertices, 1), dtype=bool)
        new_col[edit_dict['added_vertices']] = True
        
        if self.mesh.T_VE is None:
            self.mesh.T_VE = new_col
        else:
            self.mesh.T_VE = hstack([self.mesh.T_VE, new_col])

        # Create new row for T_EF
        new_row = csr_matrix((1,self.mesh.num_faces))

        if self.mesh.T_EF is None:
            self.mesh.T_EF = new_row
        else:
            self.mesh.T_EF = vstack([self.mesh.T_EF, new_row])
        
        edge_idx = self._new_edge_index
        self._new_edge_index += 1
        
        # Create edit dictionary
        edit_dict['added_edges'] = edge_idx
        
        return edit_dict
    
    def remove_edges(self, edge_idx: List[int]) -> List[int]:
        """Remove edges and update the mesh.
        
        Args:
            edge_idx (List[int]): Indices of edges to remove
            
        Returns:
            List[int]: Indices of affected faces
        """
        # Find faces that use this edge
        affected_faces = set()
        edges = np.zeros((self.mesh.T_EF.shape[0]), dtype='bool')
        edges[edge_idx] = True
        for col in range(self.mesh.T_EF.shape[1]):
            if np.any(self.mesh.T_EF[edges, col].toarray()):
                affected_faces.add(col)
        
        # Remove edges from T_VE
        mask = np.ones(self.mesh.num_edges, dtype=bool)
        mask[edge_idx] = False
        self.mesh.T_VE = self.mesh.T_VE[:,mask]
        
        # Update edge indices in T_EF
        new_indices = np.cumsum(mask) - 1
        for col in range(self.mesh.T_EF.shape[1]):
            if col not in affected_faces:
                edge_idxs = self.mesh.T_EF[:,col].nonzero()[0]
                self.mesh.T_EF[new_indices[edge_idxs], col] = True

        if affected_faces:
            edit_dict = self.remove_faces(list(affected_faces))
        else:
            edit_dict = {}

        edit_dict['affected_faces'] = affected_faces
                
        return edit_dict
        
    def create_face(self, face: Face) -> int:
        """Create a new face from edges.
        
        Args:
            face (Face): Face to add to the mesh
            
        Returns:
            int: Index of new face
        """
        # Create edges if they don't exist
        edge_indices = [self.create_edge(edge) for edge in face.edges]

        # Create new column for T_EF
        new_col = csr_matrix((self.mesh.num_edges, 1), dtype=bool)
        for edge_idx in edge_indices:
            new_col[edge_idx] = True
            
        if self.mesh.T_EF is None:
            self.mesh.T_EF = new_col
        else:
            self.mesh.T_EF = hstack([self.mesh.T_EF, new_col])


        # Create new row for T_FD
        new_row = csr_matrix((1, self.mesh.num_domains), dtype=bool)
        
        if self.mesh.T_FD is None:
            self.mesh.T_FD = new_row
        else:
            self.mesh.T_FD = vstack([self.mesh.T_FD, new_row])
            
        face_idx = self._new_face_index
        self._new_face_index += 1
        return face_idx
    
    def remove_faces(self, face_indices: List[int]) -> List[int]:
        """Remove faces and update the mesh.
        
        Args:
            face_indices (List[int]): Indices of faces to remove
            
        Returns:
            List[int]: Indices of affected domains
        """
        # Check for valid indices
        assert face_indices
        assert min(face_indices) >= 0
        assert max(face_indices) < self.mesh.num_faces

        # Find domains that use these faces
        affected_domains = set()
        for face_idx in face_indices:
            for col in range(self.mesh.T_FD.shape[1]):
                if self.mesh.T_FD[face_idx, col]:
                    affected_domains.add(col)
        
        # Remove faces from T_EF
        mask = np.ones(self.mesh.num_faces, dtype=bool)
        mask[face_indices] = False
        self.mesh.T_EF = self.mesh.T_EF[:,mask]
        
        # Update face indices in T_FD
        new_indices = np.cumsum(mask) - 1
        for col in range(self.mesh.T_FD.shape[1]):
            if col not in affected_domains:
                face_idxs = self.mesh.T_FD[:,col].nonzero()[0]
                self.mesh.T_FD[new_indices[face_idxs], col] = True
                
        if affected_domains:
            self.remove_domains(list(affected_domains))

        return list(affected_domains)
        
    def create_domain(self, domain: Domain) -> int:
        """Create a new domain from faces.
        
        Args:
            domain (Domain): Domain to add to the mesh
            
        Returns:
            int: Index of new domain
        """
        # Create faces if they don't exist
        face_indices = [self.create_face(face) for face in domain.faces]
        
        # Create new column for T_FD
        new_col = csr_matrix((self.mesh.num_faces, 1), dtype=bool)
        for face_idx in face_indices:
            new_col[face_idx] = True
            
        if self.mesh.T_FD is None:
            self.mesh.T_FD = new_col
        else:
            self.mesh.T_FD = hstack([self.mesh.T_FD, new_col])
            
        domain_idx = self._new_domain_index
        self._new_domain_index += 1
        return domain_idx
    def remove_domains(self, domain_indices: List[int]) -> Set[int]:
        """Remove domains and update the mesh.
        
        Args:
            domain_indices (List[int]): Indices of domains to remove
            
        Returns:
            Set[int]: Set of removed face indices
        """
        
        # Remove domains from T_FD
        mask = np.ones(self.mesh.num_domains, dtype=bool)
        mask[domain_indices] = False
        self.mesh.T_FD = self.mesh.T_FD[:,mask]

    def intersect_with_plane(self, plane: Plane) -> dict:
        """Calculate intersection between mesh and plane.
        
        Args:
            plane (Plane): Plane to intersect with
            
        Returns:
            dict: Dictionary containing intersection information
        """
        intersection_points = []
        intersection_edges = []
        intersection_faces = []
        modified_edges = []
        modified_faces = []
        modified_domains = []
        new_edges = []
        new_faces = []
        new_domains = []

        # Check each domain in the mesh for intersection with the plane
        for domain_idx in range(self.mesh.num_domains):
            # Get faces for this domain
            domain_faces = []
            face_indices = self.mesh.T_FD[:,domain_idx].nonzero()[0]
            
            for face_idx in face_indices:
                # Get edges for this face
                face_edges = []
                edge_indices = self.mesh.T_EF[:,face_idx].nonzero()[0]
                
                for edge_idx in edge_indices:
                    # Get vertices for this edge
                    vert_indices = self.mesh.T_VE[:,edge_idx].nonzero()[0]
                    if len(vert_indices) == 2:
                        v1 = self.mesh.vertices[vert_indices[0]]
                        v2 = self.mesh.vertices[vert_indices[1]]
                        face_edges.append(Edge(v1, v2))
                
                if face_edges:
                    # Create face object
                    domain_faces.append(Face(face_edges))
            
            if domain_faces:
                # Create domain and intersect with plane
                domain = Domain(domain_faces)
                modified_domain, intersection_face, new_domain_list = domain.intersect_plane(plane)
                
                # Update mesh with modified domain
                self.create_domain(modified_domain)
                
                if intersection_face:
                    # Create intersection face in mesh
                    face_idx = self.create_face(intersection_face)
                    intersection_faces.append(intersection_face)

                    # Add intersection results
                    for edge in intersection_face.edges:
                        edge_idx = self.create_edge(edge)
                        intersection_points.extend([edge.start, edge.end])
                        intersection_edges.append(edge)
                        new_edges.append(edge_idx)
                    
                    modified_domains.append((domain_idx, modified_domain))
                    
                    # Create any new domains
                    for new_domain in new_domain_list:
                        domain_idx = self.create_domain(new_domain)
                        new_domains.append(domain_idx)

        return {
            'intersection_points': np.array(intersection_points), #new intersection vertices created
            'intersection_edges': intersection_edges, #new intersection edges created
            'intersection_faces': intersection_faces, #new intersection faces created
            'modified_edges': modified_edges, #existing edges modified
            'modified_faces': modified_faces, #existing faces modified
            'modified_domains': modified_domains, #existing domains modified
            'new_edges': new_edges, #new edges created due to edge splitting
            'new_faces': new_faces, #new faces created due to face splitting            
            'new_domains': new_domains #new domains created due to domain splitting
        }
    
    def remove_outside(self, plane: Plane) -> dict:
        """Remove all parts of the mesh that lie outside the given plane.
        The plane's normal points towards the region to be kept.
        
        Args:
            plane (Plane): Plane defining the cut boundary. Points in the direction
                         of the normal will be kept, points in the opposite direction
                         will be removed.
            
        Returns:
            dict: Information about the operation including:
                - intersection_info: Results from intersect_with_plane
                - removed_vertices: Indices of removed vertices
                - removed_edges: Indices of removed edges
                - removed_faces: Indices of removed faces
                - removed_domains: Indices of removed domains
        """
        # First intersect with plane to create new geometry at intersection
        intersection_info = self.intersect_with_plane(plane)
        
        # Find vertices that lie outside the plane (positive signed distance)
        signed_distances = plane.signed_distance(self.mesh.vertices)
        outside_vertex_indices = np.where(signed_distances > 0)[0]
        
        # Remove outside vertices and track affected elements
        if len(outside_vertex_indices) > 0:
            affected_edges = self.delete_vertices(outside_vertex_indices.tolist())
        else:
            affected_edges = []
            
        # Combine results
        result = {
            'intersection_info': intersection_info,
            'removed_vertices': outside_vertex_indices,
            'removed_edges': affected_edges,
            'removed_faces': intersection_info['modified_faces'],
            'removed_domains': [idx for idx, _ in intersection_info['modified_domains']]
        }
        
        return result