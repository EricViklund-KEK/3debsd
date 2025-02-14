import numpy as np
from mesh.mesh3d import Mesh3D

class EBSD3D(Mesh3D):
    """A class representing a 3D EBSD (Electron Backscatter Diffraction) dataset.
    
    This class extends the Mesh3D class to include crystallographic orientation data
    associated with each domain in the mesh. Orientations are stored as Euler angles
    in the Bunge convention (φ1, Φ, φ2).
    
    Attributes:
        vertices (np.ndarray): Array of shape (N, 3) containing the coordinates of the mesh vertices
        T_VE (csr_matrix): Sparse matrix of shape (num_vertices, num_edges) defining vertex-edge connectivity
        T_EF (csr_matrix): Sparse matrix of shape (num_edges, num_faces) defining edge-face connectivity
        T_FD (csr_matrix): Sparse matrix of shape (num_faces, num_domains) defining face-domain connectivity
        euler_angles (np.ndarray): Array of shape (num_domains, 3) containing Euler angles (φ1, Φ, φ2)
            for each domain in radians
        phase_ids (np.ndarray): Array of shape (num_domains,) containing phase identifiers
            for each domain
        confidence_indices (np.ndarray): Array of shape (num_domains,) containing confidence
            indices for each orientation measurement
        _kdtree (KDTree): Internal KDTree structure for efficient nearest neighbor queries
    """
    
    def __init__(self, vertices, T_VE=None, T_EF=None, T_FD=None, 
                 euler_angles=None, phase_ids=None, confidence_indices=None):
        """Initialize the 3D EBSD dataset.
        
        Args:
            vertices (np.ndarray): Array of shape (N, 3) containing vertex coordinates
            T_VE (np.ndarray or csr_matrix, optional): Vertex-edge connectivity matrix
            T_EF (np.ndarray or csr_matrix, optional): Edge-face connectivity matrix
            T_FD (np.ndarray or csr_matrix, optional): Face-domain connectivity matrix
            euler_angles (np.ndarray, optional): Array of shape (num_domains, 3) containing
                Euler angles in radians
            phase_ids (np.ndarray, optional): Array of shape (num_domains,) containing
                phase identifiers
        """
        super().__init__(vertices, T_VE, T_EF, T_FD)
        
        # Initialize EBSD-specific attributes
        if euler_angles is not None:
            self.euler_angles = np.asarray(euler_angles, dtype=float)
            if self.euler_angles.shape[0] != self.num_domains:
                raise ValueError(f"Number of euler angles ({self.euler_angles.shape[0]}) must match number of domains ({self.num_domains})")
        else:
            self.euler_angles = None
            
        if phase_ids is not None:
            self.phase_ids = np.asarray(phase_ids, dtype=int)
            if self.phase_ids.shape[0] != self.num_domains:
                raise ValueError(f"Number of phase IDs ({self.phase_ids.shape[0]}) must match number of domains ({self.num_domains})")
        else:
            self.phase_ids = None
    
    def __repr__(self):
        base_repr = super().__repr__()
        return f"EBSD{base_repr[4:]}"  # Replace "Mesh" with "EBSD" 