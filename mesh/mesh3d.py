import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple

class MeshValidationError(Exception):
    """Exception raised for mesh validation errors."""
    pass

class Mesh3D:
    """A class representing a 3D mesh structure defined by vertices, edges, and faces.
    
    The mesh is represented using vertices and sparse connectivity matrices. Connection matrices
    follow the naming convention T_XY where X and Y are object types (V=vertices, E=edges, 
    F=faces, D=domains) and by convention the lower dimension object is on the first axis.
    
    Attributes:
        vertices (np.ndarray): Array of shape (N, 3) containing the coordinates of the mesh vertices
        T_VE (csr_matrix): Sparse matrix of shape (num_vertices, num_edges) defining vertex-edge connectivity
        T_EF (csr_matrix): Sparse matrix of shape (num_edges, num_faces) defining edge-face connectivity
        T_FD (csr_matrix): Sparse matrix of shape (num_faces, num_domains) defining face-domain connectivity
        _kdtree (KDTree): Internal KDTree structure for efficient nearest neighbor queries
    """
    
    def __init__(self, vertices=None, T_VE=None, T_EF=None, T_FD=None):
        """Initialize the 3D mesh.
        
        Args:
            vertices (np.ndarray): Array of shape (N, 3) containing vertex coordinates
            T_VE (np.ndarray or csr_matrix, optional): Vertex-edge connectivity matrix
            T_EF (np.ndarray or csr_matrix, optional): Edge-face connectivity matrix
            T_FD (np.ndarray or csr_matrix, optional): Face-domain connectivity matrix
        """
        # Initialize with empty arrays/matrices if None
        self.vertices = np.zeros((0, 3)) if vertices is None else np.asarray(vertices, dtype=float)
        self.T_VE = csr_matrix((self.num_vertices, 0)) if T_VE is None else csr_matrix(T_VE)
        self.T_EF = csr_matrix((self.num_edges, 0)) if T_EF is None else csr_matrix(T_EF)
        self.T_FD = csr_matrix((self.num_faces, 0)) if T_FD is None else csr_matrix(T_FD)
    
    @property
    def num_vertices(self) -> int:
        """Return the number of vertices in the mesh."""
        return self.vertices.shape[0]
    
    @property
    def num_edges(self) -> int:
        """Return the number of edges in the mesh."""
        return self.T_VE.shape[1]
    
    @property
    def num_faces(self) -> int:
        """Return the number of faces in the mesh."""
        return self.T_EF.shape[1]
    
    @property
    def num_domains(self) -> int:
        """Return the number of domains in the mesh."""
        return self.T_FD.shape[1]
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of the mesh (vertices, edges, faces, domains)."""
        return (self.num_vertices, self.num_edges, self.num_faces, self.num_domains)
    
    def __repr__(self):
        return f"Mesh3D(vertices: {self.num_vertices}, edges: {self.num_edges}, faces: {self.num_faces}, domains: {self.num_domains})"
    
    def get_edge_vertices(self, edge_idx: int) -> tuple[int, int]:
        """Get vertex indices for an edge."""
        verts = self.T_VE[:, edge_idx].nonzero()[0]
        return verts[0], verts[1]

    def get_face_edges(self, face_idx: int) -> list[int]:
        """Get edge indices for a face."""
        return self.T_EF[:, face_idx].nonzero()[0].tolist()