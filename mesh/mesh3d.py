from collections import defaultdict
import numpy as np
from scipy.sparse.csgraph import breadth_first_tree
from scipy.sparse import csr_array
from scipy.spatial.transform import Rotation
from scipy.spatial import Voronoi
from typing import Callable, Tuple

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def explode_dict(d):
    key_ids = []
    value_ids = []
    for key, value in d.items():
        for v in value:
            key_ids.append(key)
            value_ids.append(v)

    return key_ids, value_ids


def find_grains(domains, A_DnonGB):
    """Find grains in the mesh."""

    remaining_domains = domains.copy()
    grains = []
    while remaining_domains.shape[0] > 0:
        csr_grain = breadth_first_tree(A_DnonGB,remaining_domains[0],directed=False)

        csr_grain = csr_grain.tocoo()

        grain_regions = np.unique(np.concatenate((csr_grain.coords[0],csr_grain.coords[1],remaining_domains[0,None])))

        remaining_domains = np.setdiff1d(remaining_domains,grain_regions,assume_unique=True)

        grains.append(grain_regions)

        logger.info(f"Found grain with {len(grain_regions)} domains")

    logger.info(f"Found {len(grains)} grains")

    return grains
    
def calculate_GB_connectivity(mesh: 'Mesh3D') -> tuple[csr_array, csr_array]:
    """Calculate grain boundary connectivity in the mesh."""
    logger.info("Calculating grain boundary connectivity")

    rotations = Rotation.from_euler('XZX', mesh.euler_angles)

    A_D = mesh.T_FD.T @ mesh.T_FD
    A_D.eliminate_zeros() 

    domain_pairs = np.array(A_D.tocoo().nonzero()).T
    domain_pairs = domain_pairs[domain_pairs[:, 0] < domain_pairs[:, 1]]  # Keep only unique pairs

    domain_to_point = np.zeros(mesh.num_domains, dtype=int)
    for point_id, domain_id in enumerate(mesh.PD_map):
        domain_to_point[domain_id] = point_id    

    point_pairs = domain_to_point[domain_pairs]

    misorientations = rotations[point_pairs[:,0]] * rotations[point_pairs[:,1]].inv()
    misorientations = np.linalg.norm(misorientations.as_rotvec(),axis=-1)

    A_DGB = csr_array((mesh.T_FD.shape[1],mesh.T_FD.shape[1]))
    T_FDGB = csr_array(mesh.T_FD.shape)

    T_FDGB = mesh.T_FD * (misorientations > 0.05)[:,None]
    T_FDGB.eliminate_zeros()

    A_DGB = T_FDGB.T @ T_FDGB
    A_DGB.eliminate_zeros()

    A_DnonGB = A_D - A_DGB
    A_DnonGB.eliminate_zeros() 

    return A_DGB, A_DnonGB

def calculate_GBs(mesh: 'Mesh3D') -> csr_array:
    """Calculate grain boundaries in the mesh."""
    # Placeholder for actual GB calculation logic
    logger.info("Calculating grain boundaries")

    A_DGB, A_DnonGB = calculate_GB_connectivity(mesh)

    grains = find_grains(np.arange(mesh.num_domains), A_DnonGB)

    T_DG = csr_array((mesh.num_domains,len(grains)),dtype='bool')
    grain_ind, domain_ind = np.array([[i,domain] for i, grain in enumerate(grains) for domain in grain]).T
    T_DG[domain_ind,grain_ind] = True

    B_D = T_DG @ T_DG.T
    B_D.eliminate_zeros()

    T_FG = mesh.T_FD@(B_D * A_DGB)@T_DG * (mesh.T_FD@T_DG)
    T_FG.eliminate_zeros()

    return T_FG

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
        T_VE (csr_array): Sparse matrix of shape (num_vertices, num_edges) defining vertex-edge connectivity
        T_EF (csr_array): Sparse matrix of shape (num_edges, num_faces) defining edge-face connectivity
        T_FD (csr_array): Sparse matrix of shape (num_faces, num_domains) defining face-domain connectivity
        _kdtree (KDTree): Internal KDTree structure for efficient nearest neighbor queries
    """

    _mesh: 'Mesh3D' = None
    
    def __init__(self, point_coordinates, euler_angles, PD_map, vertices, T_VE, T_EF, T_FD):
        """Initialize the 3D mesh.
        
        Args:
            vertices (np.ndarray): Array of shape (N, 3) containing vertex coordinates
            T_VE (np.ndarray or csr_array, optional): Vertex-edge connectivity matrix
            T_EF (np.ndarray or csr_array, optional): Edge-face connectivity matrix
            T_FD (np.ndarray or csr_array, optional): Face-domain connectivity matrix
        """
        self.point_coordinates = point_coordinates
        self.euler_angles = euler_angles
        self.vertices = np.asarray(vertices, dtype=float)
        self.PD_map = PD_map
        self.T_VE = csr_array(T_VE)
        self.T_EF = csr_array(T_EF)
        self.T_FD = csr_array(T_FD)
    
        self.T_FG = calculate_GBs(self)



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
    
    @classmethod
    def from_voronoi_tessellation(cls, data) -> 'Mesh3D':
        """Create a Mesh3D instance from Voronoi tessellation points.
        
        Args:
            points (np.ndarray): Array of shape (N, 3) containing the coordinates of the Voronoi points
        returns:
            Mesh3D: A new instance of Mesh3D created from Voronoi tessellation points
        """
        points = data['point_coordinates']
        euler_angles = data['euler_angles']

        vor = Voronoi(points)
        vertices = vor.vertices
        vertices = np.concatenate((vertices, np.array([[100,100,100]])), axis=0)  # Add a point at infinity

        logger.debug(f"regions: {vor.regions}")
        logger.debug(f"point_region: {vor.point_region}")
        logger.debug(f"num_points: {len(vor.point_region)}")

        # Create point_map, replacing -1 with the last domain index (num_domains - 1)
        num_domains = len(vor.regions)

        num_vertices = len(vertices)
        num_edges = 0
        num_faces = len(vor.ridge_vertices)

        # Create edge list and ridge-to-edge mapping
        edge_ind = 0
        VE_map = defaultdict(list)
        EF_map = defaultdict(list)
        FD_map = defaultdict(list)

        for ridge_ind, ridge in enumerate(vor.ridge_vertices):
            for vert1, vert2 in zip(ridge, ridge[1:] + ridge[:1]):
                if vert1 == -1:
                    vert1 = num_vertices - 1
                if vert2 == -1:
                    vert2 = num_vertices - 1
                EF_map[edge_ind].append(ridge_ind)
                VE_map[vert1].append(edge_ind)
                VE_map[vert2].append(edge_ind)
                edge_ind += 1

        num_edges = edge_ind + 1

        for ridge_ind, (point1, point2) in enumerate(vor.ridge_points):
            domain1 = vor.point_region[point1]
            domain2 = vor.point_region[point2]
            FD_map[ridge_ind].append(domain1)
            FD_map[ridge_ind].append(domain2)

        # Create sparse matrices
        VE_map = explode_dict(VE_map)
        EF_map = explode_dict(EF_map)
        FD_map = explode_dict(FD_map)


        T_VE = csr_array((np.ones(len(VE_map[0]), dtype=bool), VE_map), shape=(num_vertices, num_edges))
        T_EF = csr_array((np.ones(len(EF_map[0]), dtype=bool), EF_map), shape=(num_edges, num_faces))
        T_FD = csr_array((np.ones(len(FD_map[0]), dtype=bool), FD_map), shape=(num_faces, num_domains))

        cls._mesh = cls(points, euler_angles, vor.point_region, vertices, T_VE, T_EF, T_FD)

        return cls._mesh
    

                