import logging
import numpy as np
from mesh.mesh3d import Mesh3D
from scipy.spatial.transform import Rotation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_tree
from scipy.spatial import Delaunay


logger = logging.getLogger(__name__)

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
    
    def __init__(self, points, vertices, PDmap, T_VE=None, T_EF=None, T_FD=None, 
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
        super().__init__(points, vertices, PDmap, T_VE, T_EF, T_FD)


        
        # Initialize EBSD-specific attributes
        if euler_angles is not None:
            self.euler_angles = np.asarray(euler_angles, dtype=float)
            # if self.euler_angles.shape[0] != self.num_domains:
            #     raise ValueError(f"Number of euler angles ({self.euler_angles.shape[0]}) must match number of domains ({self.num_domains})")
        else:
            self.euler_angles = None
            
        if phase_ids is not None:
            self.phase_ids = np.asarray(phase_ids, dtype=int)
            # if self.phase_ids.shape[0] != self.num_domains:
            #     raise ValueError(f"Number of phase IDs ({self.phase_ids.shape[0]}) must match number of domains ({self.num_domains})")
        else:
            self.phase_ids = None

        # self.A_DGB = self._calculate_A_DGB()
        self.A_DGB = self._calculate_GBs()
        self.T_DG = self._find_grains()
        self.T_FG = self._find_GB_faces()


    def _calculate_GBs(self, tol=5):
        """Calculate grain boundaries based on misorientation angle.
        
        Args:
            tol (float): Tolerance angle in degrees for identifying grain boundaries
        
        Returns:
            np.ndarray: Array of shape (num_edges,) containing grain boundary IDs
        """
        # Calculate misorientation angles between adjacent domains
        rotations = Rotation.from_euler('XZX',self.euler_angles)
        rotations = rotations[self.PDmap]

        # domain_connectivity = self.T_FD.T @ self.T_FD
        # domain_pairs = domain_connectivity.tocoo().nonzero()
        # misorientations = rotations[domain_pairs[0]].inv() * rotations[domain_pairs[1]]
        # misorientations = np.linalg.norm(misorientations.as_rotvec(),axis=-1)


        domain_pairs = []
        for face_id in range(self.T_FD.shape[0]):
            face_domains = self.T_FD[face_id].nonzero()[1]
            if len(face_domains) == 2:
                domain_pairs.append(face_domains)
            else:
                # TODO: Figure out why this is happening
                raise ValueError(f"Face {face_id} has {len(face_domains)} domains, expected 2")

        domain_pairs = np.array(domain_pairs).T

        misorientations = rotations[domain_pairs[0]].inv() * rotations[domain_pairs[1]]
        misorientations = np.linalg.norm(misorientations.as_rotvec(),axis=-1)
        
        # Identify grain boundaries based on misorientation angle
        is_gb = misorientations > np.radians(tol)
        
        T_FDGB = self.T_FD.multiply(is_gb[:,None])
        T_FDGB.eliminate_zeros()

        # # Create grain boundary IDs
        # gb_ids = np.zeros(self.num_faces, dtype=bool)
        # gb_ids[is_gb] = True
        
        # return gb_ids
    
        return T_FDGB.T @ T_FDGB

    def _calculate_A_DGB(self):
        gb_ids = self._calculate_GBs()

        T_FDGB = self.T_FD.multiply(gb_ids[:,None])
        T_FDGB.eliminate_zeros()

        A_DGB = T_FDGB.T @ T_FDGB
        A_DGB.eliminate_zeros()

        return A_DGB

    def _find_grains(self):
        """Find grains based on domain connectivity.
        
        Returns:
            np.ndarray: Array of shape (num_domains,) containing grain IDs
        """


        A_D = self.T_FD.T @ self.T_FD
        A_D.eliminate_zeros()
        A_DnonGB = A_D - self.A_DGB
        A_DnonGB.eliminate_zeros()


        remaining_regions = np.arange(self.num_domains)
        grains = []

        while remaining_regions.shape[0] > 0:
            csr_grain = breadth_first_tree(A_DnonGB,remaining_regions[0],directed=False)

            csr_grain = csr_grain.tocoo()

            grain_regions = np.unique(np.concatenate((csr_grain.coords[0],csr_grain.coords[1],remaining_regions[0,None])))

            remaining_regions = np.setdiff1d(remaining_regions,grain_regions,assume_unique=True)

            grains.append(grain_regions)

        T_DG = csr_matrix((self.num_domains,len(grains)),dtype='bool')

        grain_ind, domain_ind = np.array([[i,domain] for i, grain in enumerate(grains) for domain in grain]).T

        T_DG[domain_ind,grain_ind] = True

        return T_DG
    
    def _find_GB_faces(self):
        """Find faces that represent grain boundaries."""
        B_D = self.T_DG @ self.T_DG.T
        B_D.eliminate_zeros()

        T_FG = (self.T_FD@(B_D.multiply(self.A_DGB))@self.T_DG).multiply(self.T_FD@self.T_DG)
        T_FG.eliminate_zeros()
        
        # Log the number of grain boundary faces
        num_gb_faces = T_FG.nnz
        logger.info(f"Found {num_gb_faces} grain boundary faces")
        
        return T_FG
    
    def _triangulate_grain(self, grain_id):
        """Triangulate a grain's boundary faces using fan triangulation."""
        grain_faces = self.T_FG[:,grain_id].nonzero()[0]
        
        # If no grain boundary faces, return empty list
        if len(grain_faces) == 0:
            logger.debug(f"No boundary faces found for grain {grain_id}")
            return []
        
        logger.debug(f"Triangulating {len(grain_faces)} faces for grain {grain_id}")
        
        # Create a list of triangles from the grain faces
        T_VF = self.T_VE @ self.T_EF

        triangles = []
        for face_id in grain_faces:
            face_vertices = T_VF[:,face_id].nonzero()[0]
            
            if len(face_vertices) < 3:
                logger.warning(f"Face {face_id} has only {len(face_vertices)} vertices, skipping")
                continue
                
            # For triangular faces, just add them directly
            if len(face_vertices) == 3:
                triangles.append(face_vertices)
                continue
                
            # For faces with more than 3 vertices, use fan triangulation
            # Fan triangulation creates triangles by connecting the first vertex to all others in sequence
            for i in range(1, len(face_vertices) - 1):
                triangles.append([face_vertices[0], face_vertices[i], face_vertices[i+1]])
        
        logger.debug(f"Created {len(triangles)} triangles for grain {grain_id}")
        return triangles

    def GB_mesh(self, id: int) -> list[list[int, int, int]]:
        """Return the mesh of a grain boundary."""
        return self._triangulate_grain(id)


    def __repr__(self):
        base_repr = super().__repr__()
        return f"EBSD{base_repr[4:]}"  # Replace "Mesh" with "EBSD"