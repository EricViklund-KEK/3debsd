from mesh.modification.create import create_domains, create_faces, create_edges, add_vertices
from mesh.modification.remove import remove_domains, remove_faces, remove_edges, delete_vertices
from mesh.modification.selection import star, closure, link, intersection, difference
from mesh.mesh3d import Mesh3D
from mesh.geometry.plane import Plane
import numpy as np
from scipy.sparse import csr_matrix





def intersect_with_plane(mesh: Mesh3D, plane: Plane) -> tuple[Mesh3D, Mesh3D]:
    """Calculate intersection between mesh and plane.
    
    Args:
        mesh: The mesh to intersect
        plane: Plane to intersect with
        
    Returns:
        tuple[Mesh3D, Mesh3D]: The mesh above the plane and the mesh below the plane
    """
    vertices = mesh.vertices

    # Get the vertices above and below the plane
    above_mask = plane.calculate_above_plane(vertices)
    above_verts = vertices[above_mask]
    below_verts = vertices[~above_mask]
    
    # Convert vertex indices to selection dicts
    above_select = {'vertices': np.where(above_mask)[0].tolist()}
    below_select = {'vertices': np.where(~above_mask)[0].tolist()}

    # Find elements that intersect the plane
    intersected = intersection([
        star(mesh, link(mesh, above_select)),
        star(mesh, above_select)
    ])
    
    # Get elements strictly above/below plane
    above_elements = difference(star(mesh, above_select), intersected)
    below_elements = difference(star(mesh, below_select), intersected)

    # Create new meshes
    above_mesh = delete_vertices(mesh, below_select['vertices'])
    below_mesh = delete_vertices(mesh, above_select['vertices'])

    # # Create new meshes
    # T_VE = mesh.T_VE[above_elements['vertices'],:][:,above_elements['edges']]
    # T_EF = mesh.T_EF[above_elements['edges'],:][:,above_elements['faces']]
    # T_FD = mesh.T_FD[above_elements['faces'],:][:,above_elements['domains']]
    # above_mesh = Mesh3D(
    #     vertices=above_verts, 
    #     T_VE=T_VE,
    #     T_EF=T_EF,
    #     T_FD=T_FD
    #     )
    
    # T_VE = mesh.T_VE[below_elements['vertices'],:][:,below_elements['edges']]
    # T_EF = mesh.T_EF[below_elements['edges'],:][:,below_elements['faces']]
    # T_FD = mesh.T_FD[below_elements['faces'],:][:,below_elements['domains']]
    # below_mesh = Mesh3D(
    #     vertices=below_verts, 
    #     T_VE=T_VE,
    #     T_EF=T_EF,
    #     T_FD=T_FD
    #     )
    
    return above_mesh, below_mesh



