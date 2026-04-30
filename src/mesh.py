"""
This module provides the `Mesh3D` class for Voronoi-based grain boundary analysis.
"""



import numpy as np
from pyvista import PolyData
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.sparse.csgraph import breadth_first_tree
from scipy.spatial import Voronoi
from scipy.sparse import csr_array

__all__ = ["Mesh3D"]


def _mirror_points_across_plane(points, plane_point, plane_normal):
    """
    Reflects 3D points across a plane.

    Args:
        points: np.ndarray of shape (N, 3) - the input points.
        plane_point: np.ndarray or list of shape (3,) - a point on the plane.
        plane_normal: np.ndarray or list of shape (3,) - the normal vector of the plane.

    Returns:
        np.ndarray of shape (N, 3) - the mirrored points.
    """
    points = np.asarray(points)
    plane_point = np.asarray(plane_point)
    plane_normal = np.asarray(plane_normal)

    if points.shape[1] != 3 or plane_point.shape != (3,) or plane_normal.shape != (3,):
        raise ValueError("Input shapes are invalid. Points must be (N, 3), plane_point and plane_normal must be (3,)")

    # Normalize the normal vector
    n = plane_normal / np.linalg.norm(plane_normal)

    # Vector from plane point to each input point
    v = points - plane_point

    # Distance from point to plane along the normal
    d = np.dot(v, n)  # shape: (N,)

    # Reflect the points
    mirrored = points - 2 * np.outer(d, n)

    return mirrored

def _is_outside_bounds(points, bounds):
    """
    Checks whether the given points are outside the specified bounds.

    Args:
        points (np.ndarray): The points to check (shape: N, 3).
        bounds (np.ndarray): The bounds to check against (shape: 3, 2), where each column defines the min and max of each axis.

    Returns:
        np.ndarray: A boolean array indicating which points are outside the bounds (shape: N,).
    """
    bounds = np.array(bounds)
    min, max = bounds.T
    return np.any(np.greater(points, max[None,:]), axis=1) | np.any(np.less(points, min[None,:]), axis=1)

class Mesh3D:
    """
    A class for processing 3D meshes and grain boundary data.

    This class handles the generation of Voronoi diagrams from point clouds, 
    manages geometric transformations, and extracts topological relationships 
    between grains and phases.
    """
    def __init__(self, points: np.ndarray, point_data: dict, bounds: np.ndarray):
        """
        Initialize the Mesh3D object and compute the Voronoi topology.

        Parameters
        ----------
        points : numpy.ndarray
            An array of 3D coordinates with shape $(N, 3)$.
        point_data : dict
            Dictionary containing metadata associated with each point:
            
            * ``'euler'`` : numpy.ndarray
                Euler angles with shape $(N, 3)$.
            * ``'phase'`` : numpy.ndarray
                Phase identifiers with shape $(N,)$.
        bounds : numpy.ndarray
            Mesh boundaries defining the min/max limits, with shape $(3, 2)$.

        Notes
        -----
        The initialization process triggers the computation of the Voronoi 
        tessellation based on the provided `points` and `bounds`.

        Examples
        --------
        >>> import numpy as np
        >>> pts = np.array([[0.1, 0.1, 0.1], 
        >>>                 [0.15, 0.11, 0.11], 
        >>>                 [0.2, 0.1, 0.1], 
        >>>                 [0.5, 0.55, 0.5], 
        >>>                 [0.5, 0.5, 0.55]])  
        >>> data = {'euler': np.array([[0, 0, 0], 
        >>>                            [0, 0, 0], 
        >>>                            [0, 0, 0], 
        >>>                            [10, 0, 0], 
        >>>                            [10, 0, 0]]), 
        >>>         'phase': np.array([1, 1, 1, 1, 1])}
        >>> bnds = np.array([[0, 1], [0, 1], [0, 1]])
        >>> mesh = Mesh3D(pts, data, bnds)
        >>> type(mesh)
        <class 'Mesh3D'>
        """









        coordinates = points
        euler_points = point_data["euler"]
        phase_flat = point_data["phase"]

        normals = np.array(
            ((1.0,0.0,0.0),
            (0.0,1.0,0.0),
            (0.0,0.0,1.0))
            )

        out_of_bounds = _is_outside_bounds(coordinates,bounds)
        coordinates = coordinates[~out_of_bounds]
        euler_points = euler_points[~out_of_bounds]

        #EBSD_points = np.append(coordinates,euler_points,axis=-1)

        # nonzero = list(not(np.array_equal(euler_points[i,:],np.array((0.0,0.0,0.0)))) for i in range(euler_points.shape[0]))
        nonzero = phase_flat != 0
        coordinates = coordinates[nonzero]
        euler_points = euler_points[nonzero]
        phase_flat = phase_flat[nonzero]



        #coordinates = np.append(coordinates,[[100,100,100]],axis=0)
        np.random.seed(0)
        subsample = np.random.randint(0,coordinates.shape[0],size=50000)

        euler_points = euler_points[subsample]
        coordinates = coordinates[subsample]
        phase_flat = phase_flat[subsample]

        new_coordinates = [coordinates]
        new_euler_points = [euler_points]
        new_phase_flat = [phase_flat]

        for axis, normal in zip(bounds,normals):
            for bound in axis:
                point = bound * normal
                mirrored_coordinates = _mirror_points_across_plane(coordinates, point, normal)
                new_coordinates.append(mirrored_coordinates)
                new_euler_points.append(euler_points)
                new_phase_flat.append(phase_flat)

        coordinates = np.concatenate(new_coordinates, axis=0)
        euler_points = np.concatenate(new_euler_points, axis=0)
        phase_flat = np.concatenate(new_phase_flat, axis=0)

        outer_bounds = ((bounds[0, 0] - 1.0, bounds[0, 1] + 1.0),
                        (bounds[1, 0] - 1.0, bounds[1, 1] + 1.0),
                        (bounds[2, 0] - 1.0, bounds[2, 1] + 1.0))

        is_inside_outer = ~_is_outside_bounds(coordinates,outer_bounds)

        coordinates = coordinates[is_inside_outer]
        euler_points = euler_points[is_inside_outer]
        phase_flat = phase_flat[is_inside_outer]

        rotations = Rotation.from_euler('XZX',euler_points)

        del euler_points   


        vor = Voronoi(coordinates)

        # Scipy's Voronoi implementation does not define the edges the make up the ridges, so we need to create them ourselves. 
        # We also need to add vertices for the infinite ridges, which we will place far away from the centroid of the points in the direction of the ridge.
        vertices = vor.vertices.copy()
        ridge_vertices = vor.ridge_vertices.copy()

        centroid = np.average(coordinates, axis=0)

        edges = []
        ridge_edges = []
        i = 0

        for ridge in ridge_vertices:
            new_ridge = []
            for i, (vert1, vert2) in enumerate(zip(ridge,ridge[1:]+ridge[:1])):
                if vert1 == -1:
                    new_vertex = 1e1 * (vertices[vert2] - centroid)
                    vertices = np.concatenate((vertices,new_vertex[None,:]),axis=0)
                    vert1 = vertices.shape[0] - 1
                    ridge[i] = vert1
                if vert2 == -1:
                    new_vertex = 1e1 * (vertices[vert1] - centroid)
                    vertices = np.concatenate((vertices,new_vertex[None,:]),axis=0)
                    vert2 = vertices.shape[0] - 1
                    ridge[i] = vert2
                """create new edge"""
                new_edge = [vert1,vert2].copy()
                edges.append(new_edge)

                """add edge to ridge"""
                new_ridge.append(i)
                i += 1

            ridge_edges.append(new_ridge)

        edges, inverse = np.unique(np.array(edges),axis=0,return_inverse=True)
        ridge_edges = [[inverse[j] for j in ridge_edges[i]] for i in range(len(ridge_edges))]


        region_ridges = [[] for _ in range(len(vor.regions))]

        for ridge_ind, ridge in enumerate(vor.point_region[vor.ridge_points]):
            for region_ind in ridge:
                region_ridges[region_ind].append(ridge_ind)



        # Create sparse matrices for vertex-edge, edge-face, and face-domain relationships
        T_VE = csr_array((vertices.shape[0],edges.shape[0]),dtype='bool')
        T_EF = csr_array((edges.shape[0],len(ridge_edges)),dtype='bool')
        T_FD = csr_array((len(ridge_edges),len(vor.regions)),dtype='bool')

        X,Y = np.indices(edges.shape)

        T_VE[edges.flatten(),X.flatten()] = True

        ridge_ind = np.repeat(np.arange(len(ridge_edges)),[len(ridge_edges[i]) for i in range(len(ridge_edges))])
        edge_ind = [edge for ridge in ridge_edges for edge in ridge]

        T_EF[edge_ind,ridge_ind] = True

        ridge_ind = [ridge for region in region_ridges for ridge in region]
        region_ind = np.repeat(np.arange(len(region_ridges)),[len(region_ridges[i]) for i in range(len(region_ridges))])

        T_FD[ridge_ind,region_ind] = True



        # Map data points to Voronoi domains
        DP_map = [0]*len(vor.regions)
        for point, domain_ind in enumerate(vor.point_region):
            DP_map[domain_ind] = point

        # Identify faces that create the bounds
        outside_domains = _is_outside_bounds(coordinates[DP_map], bounds)
        is_out = np.zeros(T_FD.shape[1], dtype=bool)
        is_out[outside_domains] = True

        T_FD = T_FD.tocsr()
        T_FDout = T_FD[:, is_out]
        T_FDin = T_FD[:, ~is_out]
        boundary_faces = T_FDout.max(axis=1) * T_FDin.max(axis=1)


        # Identify grain boundaries, treat bounds as grain boundaries for visualization]
        misorientations = rotations[vor.ridge_points[:,0]] * rotations[vor.ridge_points[:,1]].inv()
        misorientations = np.linalg.norm(misorientations.as_rotvec(),axis=-1)

        phase_boundaries = phase_flat[vor.ridge_points[:,0]] != phase_flat[vor.ridge_points[:,1]]

        GBs = ((misorientations > 0.05) + phase_boundaries) * T_FDin.max(axis=1)
        GBs = GBs + boundary_faces

        # Compute the adjacency matrices
        A_DGB = csr_array((T_FD.shape[1],T_FD.shape[1]))
        T_FDGB = csr_array(T_FD.shape)


        # Multiply T_FD by GBs to get T_FDGB, which will have True for faces that are grain boundaries and False otherwise
        T_FDGB = T_FD * GBs[:,None]
        T_FDGB.eliminate_zeros()

        A_DGB = T_FDGB.T @ T_FDGB
        A_DGB.eliminate_zeros()
        A_D = T_FD.T @ T_FD
        A_D.eliminate_zeros()
        A_DnonGB = A_D - A_DGB
        A_DnonGB.eliminate_zeros()




        # Find the grains by finding connected components in the non-grain boundary adjacency matrix using breadth first search. 
        remaining_regions = np.arange(T_FD.shape[1])[~outside_domains]
        grains = []

        while remaining_regions.shape[0] > 0:
            csr_grain = breadth_first_tree(A_DnonGB,remaining_regions[0],directed=False)

            csr_grain = csr_grain.tocoo()

            grain_regions = np.unique(np.concatenate((csr_grain.coords[0],csr_grain.coords[1],remaining_regions[0,None])))

            remaining_regions = np.setdiff1d(remaining_regions,grain_regions,assume_unique=True)

            grains.append(grain_regions)

        # Create a sparse matrix to represent the point-domain relationships
        T_PD = csr_array((len(vor.point_region),len(vor.regions)),dtype='bool')
        point_ind = np.arange((len(vor.point_region)))
        domain_ind = vor.point_region
        T_PD[point_ind,domain_ind] = True

        # Map grains to their phases
        grain_phase = []
        for grain_indices in grains:
            point_ind = T_PD[:,grain_indices].nonzero()[0]
            grain_phase.append(phase_flat[point_ind])

        # Create a sparse matrix to represent the domain-grain relationships
        T_DG = csr_array((len(vor.regions),len(grains)),dtype='bool')
        grain_ind, domain_ind = np.array([[i,domain] for i, grain in enumerate(grains) for domain in grain]).T
        T_DG[domain_ind,grain_ind] = True

        # Compute the face-grain relationship
        B_D = T_DG @ T_DG.T
        B_D.eliminate_zeros()
        subgrain_matrix = A_DGB * B_D
        gb_matrix = A_DGB - subgrain_matrix
        T_FG = (T_FD @ gb_matrix @ T_DG) * (T_FD @ T_DG)


        self._vor = vor
        self._T_FG = T_FG
        self._vertices = vertices
        self._grains = grains
        self._grain_phase = grain_phase






    def plot_grain(self, id: int) -> PolyData:
        """
        Generates a 3D visualization of a specific grain boundary.

        Parameters
        ----------
        id : int
            The unique integer identifier of the grain to be visualized.

        Returns
        -------
        pv.PolyData
            A PyVista PolyData object representing the grain geometry, 
            ready for interactive plotting or file export.

        Examples
        --------
        >>> # Assuming 'mesh' is an initialized Mesh3D object
        >>> grain_poly = mesh.plot_grain(id=0)
        >>> grain_poly.plot(color='gold', show_edges=True) # doctest: +SKIP
        
        You can also save the resulting geometry to a file:
        
        >>> grain_poly.save("grain_0.vtk") # doctest: +SKIP
        """
        GB_vertices = [self._vor.ridge_vertices[face] for face in self._T_FG[:,[id]].tocoo().coords[0]]

        # First, identify which vertices are actually used in GB_vertices
        used_vertices = set()
        for triangle in GB_vertices:
            for idx in triangle:
                used_vertices.add(idx)

        # Create a mapping from old indices to new indices
        old_to_new = {}
        new_vertices = []

        for i, idx in enumerate(sorted(used_vertices)):
            old_to_new[idx] = i
            new_vertices.append(self._vertices[idx])

        # Convert vertices to numpy array
        new_vertices = np.array(new_vertices)

        # Update the indices in GB_vertices
        new_GB_vertices = []
        for triangle in GB_vertices:
            new_triangle = [old_to_new[idx] for idx in triangle]
            new_GB_vertices.append(new_triangle)

        grain_mesh = pv.PolyData\
            .from_irregular_faces(new_vertices, new_GB_vertices)\
            .connectivity('largest')\
            .triangulate()\
            .clean()\
            .compute_normals(auto_orient_normals=True)

        polydata = grain_mesh

        return polydata
