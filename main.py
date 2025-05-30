import numpy as np
from scipy.spatial import Voronoi
from scipy.sparse import csr_array
from scipy.spatial.transform import Rotation
from scipy.sparse.csgraph import breadth_first_tree
import pyvista as pv



import logging
logging.basicConfig(level=logging.INFO)



def connectivity_matrices_from_voronoi(vor: Voronoi, centroid):
    vertices: np.ndarray = vor.vertices

    # edges = []
    # ridge_edges = []
    # i = 0

    # for ridge in vor.ridge_vertices:
    #     new_ridge = []
    #     for (vert1, vert2) in zip(ridge,ridge[1:]+ridge[:1]):
    #         """check if vert == -1"""
    #         if vert1 == -1:
    #             new_vertex = 1e1 * (vertices[vert2] - centroid)
    #             vertices = np.concat((vertices,new_vertex[None,:]), axis=0)
    #             vert1 = vertices.shape[0] - 1
    #         if vert2 == -1:
    #             new_vertex = 1e1 * (vertices[vert1] - centroid)
    #             vertices = np.concat((vertices,new_vertex[None,:]), axis=0)
    #             vert2 = vertices.shape[0] - 1

    #         """create new edge"""
    #         new_edge = [vert1,vert2]
    #         edges.append(new_edge)

    #         """add edge to ridge"""
    #         new_ridge.append(i)
    #         i += 1

    #     ridge_edges.append(new_ridge)

    # edges, inverse = np.unique(np.array(edges),axis=0,return_inverse=True)
    # ridge_edges = [[inverse[j] for j in ridge_edges[i]] for i in range(len(ridge_edges))]

    # for ridge in vor.ridge_vertices:
    #     for i, vertex in enumerate(ridge):
    #         if vertex == -1:
    #             new_vertex = 1e0 * (vertices[ridge[i-1]] - centroid)
    #             vertices = np.concat((vertices,new_vertex[None,:]), axis=0)
    #             ridge[i] = vertices.shape[0] - 1



    # num_vertices = vertices.shape[0]
    # # num_edges = edges.shape[0]
    # num_faces = len(vor.ridge_vertices)
    # num_domains = len(vor.regions)

    # # X,Y = np.indices(edges.shape)

    # # T_VE = csr_array((np.ones(edges.flatten().shape[0], dtype='bool'), 
    # #                    (edges.flatten(), X.flatten())),
    # #                   shape=(num_vertices, num_edges))


    
    # # ridge_ind = np.repeat(np.arange(len(ridge_edges)),[len(ridge_edges[i]) for i in range(len(ridge_edges))])
    # # edge_ind = [edge for ridge in ridge_edges for edge in ridge]

    # # T_EF = csr_array((np.ones(len(edge_ind),dtype='bool'),
    # #                    (edge_ind,ridge_ind)),
    # #                    shape=(num_edges,num_faces))

    # vertex_ind = []
    # face_ind = []
    # for i, ridge in enumerate(vor.ridge_vertices):
    #     for vertex in ridge:
    #         vertex_ind.append(vertex)
    #         face_ind.append(i)

    # T_VF = csr_array((np.ones(len(vertex_ind), dtype='bool'),
    #                   (vertex_ind, face_ind)),
    #                   shape=(num_vertices, num_faces))
    
    
    # region_ridges = [[] for _ in range(len(vor.regions))]

    # for ridge_ind, ridge in enumerate(vor.point_region[vor.ridge_points]):
    #     for region_ind in ridge:
    #         region_ridges[region_ind].append(ridge_ind)

    # ridge_ind = [ridge for region in region_ridges for ridge in region]
    # region_ind = np.repeat(np.arange(len(region_ridges)),[len(region_ridges[i]) for i in range(len(region_ridges))])
    
    # T_FD = csr_array((np.ones(len(ridge_ind),dtype='bool'),
    #                    (ridge_ind,region_ind)),
    #                    shape=(num_faces,num_domains))

    infinite_point = np.array((100.0,100.0,100.0))
    vertices = np.concatenate((vertices,infinite_point[None,:]),axis=0)

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

    region_ridges = [[] for _ in range(len(vor.regions))]

    for ridge_ind, ridge in enumerate(vor.point_region[vor.ridge_points]):
        for region_ind in ridge:
            region_ridges[region_ind].append(ridge_ind)

    T_VE = csr_array((vor.vertices.shape[0],edges.shape[0]),dtype='bool')
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

    T_VF = T_VE@T_EF


    return vertices, T_VF, T_FD

def save_obj(filepath, verts, faces):
    file = open(filepath, 'w')
    for vertex in verts:
        file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

    for face in faces:
        file.write(f"f ")
        for ind in face:
            file.write(f"{ind+1} ")
        file.write("\n")  

    file.close()

def find_grains(misorientations: np.ndarray, T_FD: csr_array, gb_threshold = 1*np.pi/180):

    logging.info(f"misorientations.shape = {misorientations.shape}")
    logging.info(f"T_FD.shape = {T_FD.shape}")
    T_FDgb = T_FD * (misorientations > gb_threshold)[:,None]
    T_FDgb.eliminate_zeros()


    A_Dgb = T_FDgb.T @ T_FDgb
    A_Dgb.eliminate_zeros()
    A_D = T_FD.T @ T_FD
    A_D.eliminate_zeros()
    A_Dnongb = A_D - A_Dgb
    A_Dnongb.eliminate_zeros()

    remaining_regions = np.arange(T_FD.shape[1])
    grains = []
    while remaining_regions.shape[0] > 0:
        grain = breadth_first_tree(A_Dnongb,remaining_regions[0],directed=False).tocoo()
        grain_regions = np.unique(np.concatenate((grain.coords[0],grain.coords[1],remaining_regions[0,None])))
        
        remaining_regions = np.setdiff1d(remaining_regions,grain_regions,assume_unique=True)

        grains.append(grain_regions)

    grain_ind, domain_ind = np.array([[i,domain] for i, grain in enumerate(grains) for domain in grain]).T

    T_DG = csr_array((np.ones(grain_ind.shape[0], dtype='bool'),
                      (domain_ind, grain_ind)),
                      shape=(T_FD.shape[1],len(grains)))
    
    B_D = T_DG @ T_DG.T
    B_D.eliminate_zeros()
    T_FGgb = T_FD@(B_D * A_Dgb)@T_DG * (T_FD@T_DG)
    T_FGgb.eliminate_zeros()
    
    return T_DG, T_FGgb

def save_grain(id, vertices, ridge_vertices, T_FG):
    GB_vertices = [ridge_vertices[face] for face in T_FG[:,[id]].tocoo().coords[0]]

    # Create a new list without out-of-bounds vertices
    filtered_GB_vertices = []

    for face in GB_vertices:          
        # Check if any vertex is out of bounds
        verts_coords = vertices[face]
        if verts_coords.max() > 10.0 or verts_coords.min() < 0.0:
            continue
            
        # If we get here, the face is valid
        filtered_GB_vertices.append(face)

    print(f"Filtered out {len(GB_vertices) - len(filtered_GB_vertices)} faces with out-of-bounds vertices")

    # Replace the original list with the filtered one
    GB_vertices = filtered_GB_vertices

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
        new_vertices.append(vertices[idx])

    # Convert vertices to numpy array
    new_vertices = np.array(new_vertices)

    # Update the indices in GB_vertices
    new_GB_vertices = []
    for triangle in GB_vertices:
        new_triangle = [old_to_new[idx] for idx in triangle]
        new_GB_vertices.append(new_triangle)

    print(f"Reduced vertices from {len(vertices)} to {len(new_vertices)}")


    save_obj(f'output/grain{id}.obj',new_vertices,new_GB_vertices)

def plot_mesh(vertices, T_VF, T_FD):
    faces_vertices = []
    for domain in range(T_FD.shape[1]):
        for face in T_FD[:,domain].nonzero()[0]:
            vert_ind = T_VF[:,face].nonzero()[0]
            faces_vertices.append(vert_ind)

    return pv.PolyData.from_irregular_faces(vertices, faces_vertices)

def plot_grain(id, vertices, ridge_vertices, T_FG):
    GB_vertices = [ridge_vertices[face] for face in T_FG[:,id].tocoo().coords[0]]

    # Create a new list without out-of-bounds vertices
    filtered_GB_vertices = []

    for face in GB_vertices:
        # Check if any vertex is -1 (indicating infinity in Voronoi diagrams)
        if -1 in face:
            continue
            
        # # Check if any vertex is out of bounds
        # verts_coords = vertices[face]
        # if verts_coords.max() > 10.0 or verts_coords.min() < 0.0:
        #     continue
            
        # If we get here, the face is valid
        filtered_GB_vertices.append(face)

    print(f"Filtered out {len(GB_vertices) - len(filtered_GB_vertices)} faces with out-of-bounds vertices")

    # Replace the original list with the filtered one
    GB_vertices = filtered_GB_vertices

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
        new_vertices.append(vertices[idx])

    # Convert vertices to numpy array
    new_vertices = np.array(new_vertices)

    # Update the indices in GB_vertices
    new_GB_vertices = []
    for triangle in GB_vertices:
        new_triangle = [old_to_new[idx] for idx in triangle]
        new_GB_vertices.append(new_triangle)

    print(f"Reduced vertices from {len(vertices)} to {len(new_vertices)}")

    if len(new_vertices) == 0:
        return

    grain_mesh = pv.PolyData.from_irregular_faces(new_vertices, new_GB_vertices).compute_normals().triangulate()  
    polydata = grain_mesh

    return polydata
    

def main():
    import numpy as np

    euler = np.load('./output/euler_flat.npy')
    points = np.load('./output/points.npy')
    Nb = np.load('./output/Nb_flat.npy')
    Sn = np.load('./output/Sn_flat.npy')


    nonzero = list(not(np.array_equal(euler[i,:],np.array((0.0,0.0,0.0)))) for i in range(euler.shape[0]))

    points = points[nonzero]

    np.random.seed(0)
    subsample = np.random.randint(0,len(points),size=20000)

    points = points[subsample]

    vor = Voronoi(points)

    centroid = np.average(points, axis=0)

    vertices, T_VF, T_FD = connectivity_matrices_from_voronoi(vor, centroid)

    logging.info(f"T_VF {T_VF.shape}, T_FD {T_FD.shape}")

    logging.info("Calculating ridge vertices")
    T_VF = T_VF.tocsc()
    ridge_vertices = [T_VF[:, face].coords[0].tolist() for face in range(T_VF.shape[1])]
    logging.info(f"len(ridge_vertices) = {len(ridge_vertices)}")

    logging.info("Writing mesh to output")
    save_obj("output/mesh.obj", vertices, [ridge_vertices[face] for face in T_FD[:,0].coords[0].tolist()])
    


    rotations = Rotation.from_euler('XZX',euler)
    del euler

    misorientations = rotations[vor.ridge_points[:,0]] * rotations[vor.ridge_points[:,1]].inv()
    misorientations = np.linalg.norm(misorientations.as_rotvec(),axis=-1)

    T_DG, T_FGgb = find_grains(misorientations,T_FD)

    logging.info(f"T_DG.shape = {T_DG.shape}")
    logging.info(f"T_FG.shape = {T_FGgb.shape}")



    logging.info("Writing grains to output")
    plotter = pv.Plotter()
    colors = np.random.rand(T_FGgb.shape[1],3)

    # plotter.add_mesh(plot_mesh(vertices, T_VF, T_FD))
    for i in range(T_FGgb.shape[1]):
        mesh = plot_grain(i, vertices, ridge_vertices, T_FGgb)
        if mesh is not None:
            plotter.add_mesh(mesh, color=colors[i])


    plotter.show()




if __name__ == "__main__":
    main()