import numpy as np
from scipy.spatial import Voronoi
from scipy.sparse import csr_matrix


def connectivity_matrices_from_voronoi(vor: Voronoi):
    vertices = vor.vertices

    edges = []
    ridge_edges = []
    i = 0

    for ridge in vor.ridge_vertices:
        new_ridge = []
        for (vert1, vert2) in zip(ridge,ridge[1:]+ridge[:1]):
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

    T_VE = csr_matrix((np.ones(edges.flatten().shape[0], dtype='bool'), (edges.flatten(),X.flatten())), shape=())


def main():
    import numpy as np

    euler = np.load('./output/euler_flat.npy')
    points = np.load('./output/points.npy')
    Nb = np.load('./output/Nb_flat.npy')
    Sn = np.load('./output/Sn_flat.npy')


    nonzero = list(not(np.array_equal(euler[i,:],np.array((0.0,0.0,0.0)))) for i in range(euler.shape[0]))

    np.random.seed(0)
    subsample = np.random.randint(0,len(nonzero),size=50000)

    vor = Voronoi(points[nonzero][subsample])





if __name__ == "__main__":
    main()