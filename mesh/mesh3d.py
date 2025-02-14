import numpy as np
from scipy.sparse import csr_array, csr_matrix
from scipy.spatial import KDTree
from .option import Option
from typing import Optional, Tuple, Any

class MeshValidationError(Exception):
    """Exception raised for mesh validation errors."""
    pass

class Mesh3D:
    """A class representing a 3D mesh structure defined by vertices, edges, and faces.
    
    The mesh is represented using vertices and sparse connectivity matrices. Connection matrices
    follow the naming convention T_XY where X and Y are object types (V=vertices, E=edges, 
    F=faces, D=domains) and by convention the lower dimension object is on the first axis.
    
    Attributes:
        vertices (Option[np.ndarray]): Array of shape (N, 3) containing the coordinates of the mesh vertices
        T_VE (Option[csr_matrix]): Sparse matrix of shape (num_vertices, num_edges) defining vertex-edge connectivity
        T_EF (Option[csr_matrix]): Sparse matrix of shape (num_edges, num_faces) defining edge-face connectivity
        T_FD (Option[csr_matrix]): Sparse matrix of shape (num_faces, num_domains) defining face-domain connectivity
        _kdtree (Option[KDTree]): Internal KDTree structure for efficient nearest neighbor queries
    """
    
    def __init__(self, vertices=None, T_VE=None, T_EF=None, T_FD=None):
        """Initialize the 3D mesh.
        
        Args:
            vertices (np.ndarray): Array of shape (N, 3) containing vertex coordinates
            T_VE (np.ndarray or csr_matrix, optional): Vertex-edge connectivity matrix
            T_EF (np.ndarray or csr_matrix, optional): Edge-face connectivity matrix
            T_FD (np.ndarray or csr_matrix, optional): Face-domain connectivity matrix
        """
        # Wrap all attributes in Option
        self.vertices = Option(np.asarray(vertices, dtype=float) if vertices is not None else None)
        self.T_VE = Option(csr_matrix(T_VE) if T_VE is not None else None)
        self.T_EF = Option(csr_matrix(T_EF) if T_EF is not None else None)
        self.T_FD = Option(csr_matrix(T_FD) if T_FD is not None else None)
        self._kdtree = Option.none()
    
    def build_kdtree(self) -> None:
        """Build KD-tree for efficient nearest neighbor queries."""
        self._kdtree = self.vertices.map(KDTree)
    
    def compute_distances(self, query_points: np.ndarray) -> Option[np.ndarray]:
        """Calculate the unsigned distance from query points to the mesh.
        
        Args:
            query_points (np.ndarray): Array of shape (M, 3) containing query point coordinates
            
        Returns:
            Option[np.ndarray]: Array of shape (M,) containing distances to mesh
        """
        if self._kdtree.is_none():
            self.build_kdtree()
        return self._kdtree.map(lambda tree: tree.query(query_points)[0])
    
    def find_vertex(self, vertex: np.ndarray, tolerance: float = 1e-10) -> Option[int]:
        """Find the index of an existing vertex within tolerance."""
        return self.vertices.map(lambda verts: 
            next((i for i, v in enumerate(verts) 
                if np.allclose(v, vertex, atol=tolerance)), None))
    
    def find_edge(self, start_idx: int, end_idx: int) -> Option[int]:
        """Find the index of an existing edge between two vertices."""
        return self.T_VE.map(lambda t_ve: 
            next((i for i in range(t_ve.shape[1]) 
                if t_ve[start_idx, i] and t_ve[end_idx, i]), None))
    
    @property
    def num_vertices(self) -> int:
        """Return the number of vertices in the mesh."""
        return self.vertices.map(lambda v: v.shape[0]).get_or_else(0)
    
    @property
    def num_edges(self) -> int:
        """Return the number of edges in the mesh."""
        return self.T_VE.map(lambda m: m.shape[1]).get_or_else(0)
    
    @property
    def num_faces(self) -> int:
        """Return the number of faces in the mesh."""
        return self.T_EF.map(lambda m: m.shape[1]).get_or_else(0)
    
    @property
    def num_domains(self) -> int:
        """Return the number of domains in the mesh."""
        return self.T_FD.map(lambda m: m.shape[1]).get_or_else(0)
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of the mesh (vertices, edges, faces, domains)."""
        return (self.num_vertices, self.num_edges, self.num_faces, self.num_domains)
    
    def validate_connectivity(self) -> Option[bool]:
        """Check if all required connectivity matrices exist."""
        return (
            self.T_VE.flat_map(lambda t_ve: 
            self.T_EF.flat_map(lambda t_ef:
            self.T_FD.map(lambda t_fd: True)))
        ).or_else(Option.some(False))
    
    def is_valid(self) -> bool:
        """Check if the mesh is valid using monadic operations.
        
        A valid mesh requires:
        1. Water tight domains - each edge within a domain is shared by exactly 2 faces
        2. No dangling elements
        3. Connected components
        
        Returns:
            bool: True if the mesh is valid, False otherwise
        """
        # First check if we have all required matrices
        if not self.validate_connectivity().get_or_else(False):
            return False
            
        try:
            # Now we can safely unwrap the matrices since we validated they exist
            t_ve = self.T_VE.value
            t_ef = self.T_EF.value
            t_fd = self.T_FD.value
            
            # Check vertex-edge connectivity
            vertex_edge_counts = np.array(t_ve.sum(axis=1)).flatten()
            if np.any(vertex_edge_counts == 0):
                return False
                
            # Check edge-face connectivity
            edge_face_counts = np.array(t_ef.sum(axis=1)).flatten()
            if np.any(edge_face_counts != 2):
                return False
                
            # Check face-domain connectivity
            face_domain_counts = np.array(t_fd.sum(axis=1)).flatten()
            if np.any(face_domain_counts == 0):
                return False
                
            # Check water-tightness and connectivity of each domain
            T_ED = t_ef @ t_fd
            
            for domain_idx in range(self.num_domains):
                domain_faces = t_fd[:, domain_idx].nonzero()[0]
                domain_edges = set()
                
                for face_idx in domain_faces:
                    face_edges = t_ef[:, face_idx].nonzero()[0]
                    domain_edges.update(face_edges)
                    
                for edge_idx in domain_edges:
                    edge_faces = t_ef[edge_idx, :].nonzero()[1]
                    domain_edge_faces = [f for f in edge_faces if f in domain_faces]
                    if len(domain_edge_faces) != 2:
                        return False
                        
                # Check domain connectivity
                visited = set()
                to_visit = {domain_faces[0]}
                
                while to_visit:
                    face = to_visit.pop()
                    visited.add(face)
                    face_edges = t_ef[:, face].nonzero()[0]
                    
                    for edge in face_edges:
                        adjacent_faces = set(t_ef[edge, :].nonzero()[1])
                        adjacent_faces = {f for f in adjacent_faces if f in domain_faces}
                        to_visit.update(adjacent_faces - visited)
                        
                if len(visited) != len(domain_faces):
                    return False
                    
            return True
            
        except Exception:
            return False
    
    def __repr__(self):
        return f"Mesh3D(vertices: {self.num_vertices}, edges: {self.num_edges}, faces: {self.num_faces}, domains: {self.num_domains})"