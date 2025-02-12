import numpy as np
from scipy.spatial import Voronoi
from scipy.sparse import csr_array, csr_matrix, coo_array, coo_matrix
from mesh3d import Mesh3D

def create_voronoi_mesh(points, infinite_point=None):
    """Create a 3D mesh from points using Voronoi tessellation.
    
    Args:
        points (np.ndarray): Array of shape (N, 3) containing point coordinates
        infinite_point (np.ndarray, optional): Coordinates for handling infinite ridges.
            Defaults to [100, 100, 100] if not specified.
            
    Returns:
        Mesh3D: A mesh object representing the Voronoi diagram
    """
    # Generate Voronoi diagram
    vor = Voronoi(points)
    vertices = vor.vertices
    
    # Handle infinite ridges by adding an "infinite" point
    if infinite_point is None:
        infinite_point = np.array([100.0, 100.0, 100.0])
    vertices = np.concatenate((vertices, infinite_point[None,:]), axis=0)
    
    # Create edge list and ridge-to-edge mapping
    edges = []
    ridge_edges = []
    edge_counter = 0
    
    # Process each ridge to create edges
    for ridge in vor.ridge_vertices:
        new_ridge = []
        # Create edges for each pair of vertices in the ridge
        for vert1, vert2 in zip(ridge, ridge[1:] + ridge[:1]):
            new_edge = [vert1, vert2]
            edges.append(new_edge)
            new_ridge.append(edge_counter)
            edge_counter += 1
        ridge_edges.append(new_ridge)
    
    # Remove duplicate edges and update ridge references
    edges = np.array(edges)
    edges[edges == -1] = edges.max() + 1
    edges, inverse = np.unique(edges, axis=0, return_inverse=True)
    ridge_edges = [[inverse[j] for j in ridge_edges[i]] for i in range(len(ridge_edges))]

    # Create mapping from ridges to regions
    region_ridges = [[] for _ in range(points.shape[0])]

    for ridge_ind, ridge in enumerate(vor.point_region[vor.ridge_points]):
        for region_ind in ridge:
            region_ridges[region_ind - 1].append(ridge_ind)
    
    # Create connectivity matrices
    #T_EF = csr_array((edges.shape[0],len(ridge_edges)),dtype='bool')
    #T_FD = csr_array((len(ridge_edges),len(vor.regions)),dtype='bool')

    
    # Create vertex-edge connectivity matrix
    X,Y = np.indices(edges.shape)
    T_VE = coo_matrix((np.ones(edges.flatten().shape[0],dtype='bool'), (edges.flatten(),X.flatten())),shape=(vertices.shape[0],edges.shape[0]))
    T_VE = T_VE.tocsr()
    
    # Create edge-face connectivity matrix
    ridge_ind = np.repeat(np.arange(len(ridge_edges)),[len(ridge_edges[i]) for i in range(len(ridge_edges))])
    edge_ind = [edge for ridge in ridge_edges for edge in ridge]
    
    T_EF = coo_matrix((np.ones(len(edge_ind),dtype='bool'), (edge_ind,ridge_ind)),shape=(edges.shape[0],len(ridge_edges)))
    T_EF = T_EF.tocsr()
    #T_EF[edge_ind,ridge_ind] = True
    
    ridge_ind = [ridge for region in region_ridges for ridge in region]
    region_ind = np.repeat(np.arange(len(region_ridges)),[len(region_ridges[i]) for i in range(len(region_ridges))])
    
    T_FD = coo_matrix((np.ones(len(ridge_ind),dtype='bool'), (ridge_ind,region_ind)),shape=(len(ridge_edges),points.shape[0]))
    T_FD = T_FD.tocsr()
    #T_FD[ridge_ind,region_ind] = True

    # Create and return the mesh object
    return Mesh3D(vertices, T_VE, T_EF, T_FD)

def create_bounded_voronoi_mesh(points, bounds):
    """Create a 3D mesh from points using Voronoi tessellation with bounded region.
    
    Args:
        points (np.ndarray): Array of shape (N, 3) containing point coordinates
        bounds (tuple): Tuple of (min_coords, max_coords) defining the bounding box
            Example: (np.array([0,0,0]), np.array([1,1,1])) for unit cube
            
    Returns:
        Mesh3D: A mesh object representing the bounded Voronoi diagram
    """
    # Add points at the corners of the bounding box
    min_coords, max_coords = bounds
    corner_points = np.array([
        [min_coords[0], min_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]]
    ])
    
    # Combine original points with corner points
    all_points = np.concatenate([points, corner_points])
    
    # Create the mesh
    return create_voronoi_mesh(all_points) 