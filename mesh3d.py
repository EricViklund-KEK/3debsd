import numpy as np
from scipy.sparse import csr_array, csr_matrix
from scipy.spatial import KDTree

class Mesh3D:
    """A class representing a 3D mesh structure defined by vertices, edges, and faces.
    
    The mesh is represented using vertices and sparse connectivity matrices. Connection matrices
    follow the naming convention T_XY where X and Y are object types (V=vertices, E=edges, 
    F=faces, D=domains) and by convention the lower dimension object is on the first axis.
    
    Attributes:
        vertices (np.ndarray): Array of shape (N, 3) containing the coordinates of the mesh vertices
        T_VE (csr_matrix): Sparse matrix of shape (num_vertices, num_edges) defining vertex-edge connectivity
        T_EF (csr_matrix): Sparse matrix of shape (num_edges, num_faces) defining edge-face connectivity
        T_FD (csr_matrix, optional): Sparse matrix of shape (num_faces, num_domains) defining face-domain connectivity
        _kdtree (KDTree): Internal KDTree structure for efficient nearest neighbor queries
    """
    
    def __init__(self, vertices, T_VE=None, T_EF=None, T_FD=None):
        """Initialize the 3D mesh.
        
        Args:
            vertices (np.ndarray): Array of shape (N, 3) containing vertex coordinates
            T_VE (np.ndarray or csr_matrix, optional): Vertex-edge connectivity matrix
            T_EF (np.ndarray or csr_matrix, optional): Edge-face connectivity matrix
            T_FD (np.ndarray or csr_matrix, optional): Face-domain connectivity matrix
        """
        self.vertices = np.asarray(vertices, dtype=float)
        
        # Convert vertex-edge connectivity to sparse matrix if provided
        if T_VE is not None:
            self.T_VE = csr_matrix(T_VE) if not isinstance(T_VE, csr_matrix) else T_VE
        else:
            self.T_VE = None
            
        # Convert edge-face connectivity to sparse matrix if provided
        if T_EF is not None:
            self.T_EF = csr_matrix(T_EF) if not isinstance(T_EF, csr_matrix) else T_EF
        else:
            self.T_EF = None
            
        # Convert face-domain connectivity to sparse matrix if provided
        if T_FD is not None:
            self.T_FD = csr_matrix(T_FD) if not isinstance(T_FD, csr_matrix) else T_FD
        else:
            self.T_FD = None
            
        self._kdtree = None
    
    def build_kdtree(self):
        """Build KD-tree for efficient nearest neighbor queries."""
        self._kdtree = KDTree(self.vertices)
    
    def compute_distances(self, query_points):
        """Calculate the unsigned distance from query points to the mesh.
        
        Args:
            query_points (np.ndarray): Array of shape (M, 3) containing query point coordinates
            
        Returns:
            np.ndarray: Array of shape (M,) containing distances to mesh
        """
        if self._kdtree is None:
            self.build_kdtree()
            
        # For now, just compute distances to nearest vertices
        # TODO: Implement more sophisticated distance calculations to edges and faces
        distances, _ = self._kdtree.query(query_points)
        return distances
    
    @property
    def num_vertices(self):
        """Return the number of vertices in the mesh."""
        return self.vertices.shape[0]
    
    @property
    def num_edges(self):
        """Return the number of edges in the mesh."""
        return self.T_VE.shape[1] if self.T_VE is not None else 0
    
    @property
    def num_faces(self):
        """Return the number of faces in the mesh."""
        return self.T_EF.shape[1] if self.T_EF is not None else 0
    
    @property
    def num_domains(self):
        """Return the number of domains in the mesh."""
        return self.T_FD.shape[1] if self.T_FD is not None else 0
    
    def __repr__(self):
        return f"Mesh3D(vertices: {self.num_vertices}, edges: {self.num_edges}, faces: {self.num_faces}, domains: {self.num_domains})" 