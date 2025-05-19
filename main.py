import numpy as np
from scipy.spatial import Voronoi
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from scipy.sparse.csgraph import breadth_first_tree



import logging
logging.basicConfig(level=logging.INFO)



def connectivity_matrices_from_voronoi(vor: Voronoi):
    vertices: np.ndarray = vor.vertices

    centroid = np.average(vertices, axis = 0)

    edges = []
    ridge_edges = []
    i = 0

    for ridge in vor.ridge_vertices:
        new_ridge = []
        for (vert1, vert2) in zip(ridge,ridge[1:]+ridge[:1]):
            """check if vert == -1"""
            if vert1 == -1:
                new_vertex = 1e6 * (vertices[vert2] - centroid)
                vertices = np.concat((vertices,new_vertex[None,:]), axis=0)
                vert1 = vertices.shape[0] - 1
            if vert2 == -1:
                new_vertex = 1e6 * (vertices[vert1] - centroid)
                vertices = np.concat((vertices,new_vertex[None,:]), axis=0)
                vert2 = vertices.shape[0] - 1

            """create new edge"""
            new_edge = [vert1,vert2]
            edges.append(new_edge)

            """add edge to ridge"""
            new_ridge.append(i)
            i += 1

        ridge_edges.append(new_ridge)

    edges, inverse = np.unique(np.array(edges),axis=0,return_inverse=True)
    ridge_edges = [[inverse[j] for j in ridge_edges[i]] for i in range(len(ridge_edges))]

    X,Y = np.indices(edges.shape)

    T_VE = csr_matrix((np.ones(edges.flatten().shape[0], dtype='bool'), 
                       (edges.flatten(), X.flatten())),
                      shape=(vertices.shape[0], edges.shape[0]))


    
    ridge_ind = np.repeat(np.arange(len(ridge_edges)),[len(ridge_edges[i]) for i in range(len(ridge_edges))])
    edge_ind = [edge for ridge in ridge_edges for edge in ridge]

    T_EF = csr_matrix((np.ones(len(edge_ind),dtype='bool'),
                       (edge_ind,ridge_ind)),
                       shape=(edges.shape[0],len(ridge_edges)))
    
    
    region_ridges = [[] for _ in range(len(vor.regions))]

    for ridge_ind, ridge in enumerate(vor.point_region[vor.ridge_points]):
        for region_ind in ridge:
            region_ridges[region_ind].append(ridge_ind)

    ridge_ind = [ridge for region in region_ridges for ridge in region]
    region_ind = np.repeat(np.arange(len(region_ridges)),[len(region_ridges[i]) for i in range(len(region_ridges))])
    
    T_FD = csr_matrix((np.ones(len(ridge_ind),dtype='bool'),
                       (ridge_ind,region_ind)),
                       shape=(len(ridge_edges),len(vor.regions)))



    return T_VE, T_EF, T_FD

def find_grains(misorientations, T_FD):

    GB_threshold = 5*np.pi/180
    T_FDGB = T_FD * (misorientations > GB_threshold)[:,None]
    T_FDGB.eliminate_zeros()


    A_DGB = T_FDGB.T @ T_FDGB
    A_DGB.eliminate_zeros()
    A_D = T_FD.T @ T_FD
    A_D.eliminate_zeros()
    A_DnonGB = A_D - A_DGB
    A_DnonGB.eliminate_zeros()

    remaining_regions = np.arange(T_FD.shape[1])
    grains = []
    while remaining_regions.shape[0] > 0:
        grain = breadth_first_tree(A_DnonGB,remaining_regions[0],directed=False)
        grain_regions = np.unique(np.concatenate((grain.nonzero()[0],grain.nonzero()[1],remaining_regions[0,None])))
        
        remaining_regions = np.setdiff1d(remaining_regions,grain_regions,assume_unique=True)

        grains.append(grain_regions)

    return grains

def main():
    import numpy as np

    euler = np.load('./output/euler_flat.npy')
    points = np.load('./output/points.npy')
    Nb = np.load('./output/Nb_flat.npy')
    Sn = np.load('./output/Sn_flat.npy')


    nonzero = list(not(np.array_equal(euler[i,:],np.array((0.0,0.0,0.0)))) for i in range(euler.shape[0]))

    points = points[nonzero]

    np.random.seed(0)
    subsample = np.random.randint(0,len(points),size=50000)

    points = points[subsample]

    vor = Voronoi(points)

    T_VE, T_EF, T_FD = connectivity_matrices_from_voronoi(vor)

    logging.info(f"T_VE {T_VE.shape}, T_EF {T_EF.shape}, T_FD {T_FD.shape}")

    


    rotations = Rotation.from_euler('XZX',euler)
    del euler

    misorientations = rotations[vor.ridge_points[:,0]] * rotations[vor.ridge_points[:,1]].inv()
    misorientations = np.linalg.norm(misorientations.as_rotvec(),axis=-1)

    grains = find_grains(misorientations, T_FD)

    logging.info(f"grains {len(grains)}")




if __name__ == "__main__":
    main()