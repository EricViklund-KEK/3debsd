import numpy as np
from scipy.sparse import csr_matrix
from mesh.mesh3d import Mesh3D
from typing import List, Dict, Set

def delete_vertices(mesh: Mesh3D, vertex_indices: List[int]) -> Dict:
    """Removes vertices at the given indices and updates the mesh.
    
    Args:
        mesh (Mesh3D): The mesh to modify
        vertex_indices (List[int]): Indices of vertices to remove
        
    Returns:
        Dict: Edit dictionary containing indices of affected edges and removed vertices
    """
    # Find edges that use these vertices
    affected_edges = set()
    for col in range(mesh.T_VE.shape[1]):
        vert_indices = mesh.T_VE[:,col].nonzero()[0]
        if any(v in vertex_indices for v in vert_indices):
            affected_edges.add(col)
    
    # Remove vertices from vertex array
    mask = np.ones(mesh.num_vertices, dtype=bool)
    mask[vertex_indices] = False
    mesh.vertices = mesh.vertices[mask]
    
    # Update vertex indices in T_VE
    new_indices = np.cumsum(mask) - 1
    for col in range(mesh.T_VE.shape[1]):
        if col not in affected_edges:
            vert_indices = mesh.T_VE[:,col].nonzero()[0]
            mesh.T_VE[new_indices[vert_indices], col] = True
            
    if affected_edges:
        edit_dict = remove_edges(mesh, list(affected_edges))
    else:
        edit_dict = {}

    edit_dict['removed_vertices'] = vertex_indices

    return edit_dict

def remove_edges(mesh: Mesh3D, edge_indices: List[int]) -> Dict:
    """Remove edges and update the mesh.
    
    Args:
        mesh (Mesh3D): The mesh to modify
        edge_indices (List[int]): Indices of edges to remove
        
    Returns:
        Dict: Edit dictionary containing indices of affected faces
    """
    # Find faces that use this edge
    affected_faces = set()
    edges = np.zeros((mesh.T_EF.shape[0]), dtype='bool')
    edges[edge_indices] = True
    for col in range(mesh.T_EF.shape[1]):
        if np.any(mesh.T_EF[edges, col].toarray()):
            affected_faces.add(col)
    
    # Remove edges from T_VE
    mask = np.ones(mesh.num_edges, dtype=bool)
    mask[edge_indices] = False
    mesh.T_VE = mesh.T_VE[:,mask]
    
    # Update edge indices in T_EF
    new_indices = np.cumsum(mask) - 1
    for col in range(mesh.T_EF.shape[1]):
        if col not in affected_faces:
            edge_idxs = mesh.T_EF[:,col].nonzero()[0]
            mesh.T_EF[new_indices[edge_idxs], col] = True

    if affected_faces:
        edit_dict = remove_faces(mesh, list(affected_faces))
    else:
        edit_dict = {}

    edit_dict['affected_faces'] = affected_faces
            
    return edit_dict

def remove_faces(mesh: Mesh3D, face_indices: List[int]) -> List[int]:
    """Remove faces and update the mesh.
    
    Args:
        mesh (Mesh3D): The mesh to modify
        face_indices (List[int]): Indices of faces to remove
        
    Returns:
        List[int]: Indices of affected domains
    """
    # Check for valid indices
    assert face_indices
    assert min(face_indices) >= 0
    assert max(face_indices) < mesh.num_faces

    # Find domains that use these faces
    affected_domains = set()
    for face_idx in face_indices:
        for col in range(mesh.T_FD.shape[1]):
            if mesh.T_FD[face_idx, col]:
                affected_domains.add(col)
    
    # Remove faces from T_EF
    mask = np.ones(mesh.num_faces, dtype=bool)
    mask[face_indices] = False
    mesh.T_EF = mesh.T_EF[:,mask]
    
    # Update face indices in T_FD
    new_indices = np.cumsum(mask) - 1
    for col in range(mesh.T_FD.shape[1]):
        if col not in affected_domains:
            face_idxs = mesh.T_FD[:,col].nonzero()[0]
            mesh.T_FD[new_indices[face_idxs], col] = True
            
    if affected_domains:
        remove_domains(mesh, list(affected_domains))

    return list(affected_domains)

def remove_domains(mesh: Mesh3D, domain_indices: List[int]) -> None:
    """Remove domains and update the mesh.
    
    Args:
        mesh (Mesh3D): The mesh to modify
        domain_indices (List[int]): Indices of domains to remove
    """
    # Remove domains from T_FD
    mask = np.ones(mesh.num_domains, dtype=bool)
    mask[domain_indices] = False
    mesh.T_FD = mesh.T_FD[:,mask]