import numpy as np
from scipy.spatial import Voronoi
from scipy.sparse import csr_array, csr_matrix, coo_array, coo_matrix
from .mesh3d import Mesh3D

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

def compute_line_box_intersection(p1, p2, bounds):
    """Compute intersection of a line segment with a bounding box.
    
    Args:
        p1 (np.ndarray): Start point of line segment
        p2 (np.ndarray): End point of line segment
        bounds (tuple): (min_coords, max_coords) defining the bounding box
        
    Returns:
        np.ndarray or None: Intersection point if it exists, None otherwise
    """
    min_coords, max_coords = bounds
    
    # Compute direction vector
    dir_vec = p2 - p1
    
    # Check intersection with each face of the box
    t_values = []
    for i in range(3):  # For each dimension (x,y,z)
        if dir_vec[i] != 0:
            t1 = (min_coords[i] - p1[i]) / dir_vec[i]
            t2 = (max_coords[i] - p1[i]) / dir_vec[i]
            t_values.extend([t1, t2])
    
    # Filter valid intersections
    valid_t = []
    for t in t_values:
        if t >= 0:  # Only consider forward direction
            point = p1 + t * dir_vec
            # Check if intersection point is within bounds
            if np.all(point >= min_coords) and np.all(point <= max_coords):
                valid_t.append(t)
    
    if valid_t:
        # Return the closest intersection point
        t = min(valid_t)
        return p1 + t * dir_vec
    return None

def create_bounded_voronoi_mesh(points, voronoi_bounds):
    """Create a 3D mesh from points using Voronoi tessellation with bounded region.
    
    Args:
        points (np.ndarray): Array of shape (N, 3) containing point coordinates
        voronoi_bounds (tuple): Tuple of (min_coords, max_coords) defining the bounding box
            Example: (np.array([0,0,0]), np.array([1,1,1])) for unit cube
            
    Returns:
        Mesh3D: A mesh object representing the bounded Voronoi diagram
    """
    # Generate Voronoi diagram
    vor = Voronoi(points)
    vertices = list(vor.vertices)
    
    # Process infinite vertices
    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 in simplex:
            # Get indices of the points forming the ridge
            i = simplex.index(-1)
            finite_idx = simplex[1-i]
            
            # Get the finite vertex of the ridge
            finite_vertex = vor.vertices[finite_idx]
            
            # Compute direction vector
            direction = vor.vertices[finite_idx] - center
            direction = direction / np.linalg.norm(direction)
            
            # Find intersection with bounding box
            far_point = finite_vertex + direction * 100  # Use a point far in the direction
            intersection = compute_line_box_intersection(finite_vertex, far_point, voronoi_bounds)
            
            if intersection is not None:
                vertices.append(intersection)
                # Update ridge vertex index to point to new vertex
                simplex[i] = len(vertices) - 1
    
    vertices = np.array(vertices)
    
    # Create edge list and ridge-to-edge mapping
    edges = []
    ridge_edges = []
    edge_counter = 0
    
    # Process each ridge to create edges
    for ridge in vor.ridge_vertices:
        new_ridge = []
        # Create edges for each pair of vertices in the ridge
        for vert1, vert2 in zip(ridge, ridge[1:] + ridge[:1]):
            if vert1 >= 0 and vert2 >= 0:  # Only create edges between valid vertices
                new_edge = [vert1, vert2]
                edges.append(new_edge)
                new_ridge.append(edge_counter)
                edge_counter += 1
        if new_ridge:  # Only add ridges that have edges
            ridge_edges.append(new_ridge)
    
    edges = np.array(edges)

    # Create mapping from ridges to regions
    region_ridges = [[] for _ in range(points.shape[0])]
    for ridge_ind, ridge_points in enumerate(vor.ridge_points):
        for point_ind in ridge_points:
            region_ridges[point_ind].append(ridge_ind)
    
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