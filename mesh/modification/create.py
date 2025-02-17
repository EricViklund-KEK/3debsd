import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from mesh.mesh3d import Mesh3D
from mesh.primitives import Edge, Face, Domain
from typing import List, Dict, Set, Tuple

def mesh_union(meshes: List[Mesh3D], remove_duplicates: bool = False) -> Mesh3D:
    """Union of multiple meshes.
    
    Args:
        meshes (List[Mesh3D]): List of meshes to union
        remove_duplicates (bool): Whether to remove duplicates after union
        
    Returns:
        Mesh3D: The union of the meshes
    """
    if not meshes:
        return Mesh3D()
    
    if remove_duplicates:
        raise NotImplementedError("Removing duplicates is not implemented yet")
    
    # Concatenate all vertices
    all_vertices = np.vstack([mesh.vertices for mesh in meshes])
    
    # Calculate vertex offsets for each mesh
    vert_offsets = np.cumsum([0] + [mesh.num_vertices for mesh in meshes[:-1]])
    edge_offsets = np.cumsum([0] + [mesh.num_edges for mesh in meshes[:-1]])
    face_offsets = np.cumsum([0] + [mesh.num_faces for mesh in meshes[:-1]])
    doma_offsets = np.cumsum([0] + [mesh.num_domains for mesh in meshes[:-1]])
    
    # Initialize empty matrices with final sizes
    total_vertices = sum(mesh.num_vertices for mesh in meshes)
    total_edges = sum(mesh.num_edges for mesh in meshes)
    total_faces = sum(mesh.num_faces for mesh in meshes)
    total_domains = sum(mesh.num_domains for mesh in meshes)
    
    T_VE = csr_matrix((total_vertices, total_edges))
    T_EF = csr_matrix((total_edges, total_faces))
    T_FD = csr_matrix((total_faces, total_domains))
    
    # Fill matrices with offset indices
    for i, mesh in enumerate(meshes):
        # Get row and column ranges for this mesh's submatrices
        v_start, e_start = vert_offsets[i], edge_offsets[i]
        e_end = e_start + mesh.num_edges
        f_start, f_end = face_offsets[i], face_offsets[i] + mesh.num_faces
        d_start = doma_offsets[i]
        
        # Extract data and indices from original matrices
        T_VE_data = mesh.T_VE.data
        T_VE_indices = mesh.T_VE.indices
        T_VE_indptr = mesh.T_VE.indptr
        
        T_EF_data = mesh.T_EF.data
        T_EF_indices = mesh.T_EF.indices
        T_EF_indptr = mesh.T_EF.indptr
        
        T_FD_data = mesh.T_FD.data
        T_FD_indices = mesh.T_FD.indices
        T_FD_indptr = mesh.T_FD.indptr
        
        # Create new submatrices with offset indices
        T_VE_sub = csr_matrix(
            (T_VE_data, T_VE_indices + e_start, T_VE_indptr),
            shape=(mesh.num_vertices, total_edges)
        )
        T_EF_sub = csr_matrix(
            (T_EF_data, T_EF_indices + f_start, T_EF_indptr),
            shape=(mesh.num_edges, total_faces)
        )
        T_FD_sub = csr_matrix(
            (T_FD_data, T_FD_indices + d_start, T_FD_indptr),
            shape=(mesh.num_faces, total_domains)
        )
        
        # Add submatrices to the corresponding rows in the final matrices
        T_VE[v_start:v_start + mesh.num_vertices] = T_VE_sub
        T_EF[e_start:e_end] = T_EF_sub
        T_FD[f_start:f_end] = T_FD_sub
    
    # Create new mesh with combined data
    return Mesh3D(vertices=all_vertices, T_VE=T_VE, T_EF=T_EF, T_FD=T_FD)

def add_vertices(mesh: Mesh3D, vertices: np.ndarray) -> Mesh3D:
    """Add new vertices to the mesh.
    
    Args:
        mesh (Mesh3D): The mesh to modify
        vertices (np.ndarray): Vertices to add (shape: (N, 3))
        
    Returns:
        Tuple[Mesh3D, List[int]]: The modified mesh and indices of all vertices (both existing and new)
    """
    # Concatenate new vertices with existing ones
    new_vertices = np.vstack([mesh.vertices, vertices])
    
    # Create new T_VE matrix with additional rows for new vertices
    new_T_VE = vstack([mesh.T_VE, csr_matrix((len(vertices), mesh.num_edges))])
    
    # Create new mesh with updated vertices and matrices
    new_mesh = Mesh3D(vertices=new_vertices, T_VE=new_T_VE, T_EF=mesh.T_EF, T_FD=mesh.T_FD)
    
    return new_mesh

def _calculate_sparse_components(indices_list: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Helper function to calculate components for CSR matrix construction.
    
    Args:
        indices_list (List[List[int]]): List of lists containing indices for each new item
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: data, row indices, and column indices for CSR matrix
    """
    # Calculate total number of entries
    total_entries = sum(len(indices) for indices in indices_list)
    
    # Pre-allocate arrays
    data = np.ones(total_entries)
    row_ind = np.zeros(total_entries, dtype=int)
    col_ind = np.zeros(total_entries, dtype=int)
    
    # Fill arrays
    idx = 0
    for new_item_idx, item_indices in enumerate(indices_list):
        for index in item_indices:
            row_ind[idx] = index
            col_ind[idx] = new_item_idx
            idx += 1
            
    return data, row_ind, col_ind

def create_edges(mesh: Mesh3D, vert_indices: List[List[int]]) -> Mesh3D:
    """Create new edges between vertices."""

    num_edges = len(vert_indices)

    if len(vert_indices) != num_edges or not all(len(verts) == 2 for verts in vert_indices):
        raise ValueError("Invalid vertex indices for edges")
    
    # Create new columns in T_VE
    data, row_ind, col_ind = _calculate_sparse_components(vert_indices)
    new_T_VE = hstack([
        mesh.T_VE,
        csr_matrix((data, (row_ind, col_ind)), shape=(mesh.num_vertices, num_edges))
    ])
    
    # Add new rows to T_EF
    new_T_EF = vstack([
        mesh.T_EF,
        csr_matrix((num_edges, mesh.num_faces))
    ])
    
    # Create new mesh with updated matrices
    new_mesh = Mesh3D(vertices=mesh.vertices, T_VE=new_T_VE, T_EF=new_T_EF, T_FD=mesh.T_FD)
    
    # Return the new edge indices
    return new_mesh

def create_faces(mesh: Mesh3D, edge_indices: List[List[int]]) -> Mesh3D:
    """Create new faces from edges."""

    num_faces = len(edge_indices)

    if len(edge_indices) != num_faces:
        raise ValueError("Number of edge lists doesn't match num_faces")
    
    # Create new columns in T_EF
    data, row_ind, col_ind = _calculate_sparse_components(edge_indices)
    new_T_EF = hstack([
        mesh.T_EF,
        csr_matrix((data, (row_ind, col_ind)), shape=(mesh.num_edges, num_faces))
    ])
    
    # Add new rows to T_FD
    new_T_FD = vstack([
        mesh.T_FD,
        csr_matrix((num_faces, mesh.num_domains))
    ])
    
    # Create new mesh with updated matrices
    new_mesh = Mesh3D(vertices=mesh.vertices, T_VE=mesh.T_VE, T_EF=new_T_EF, T_FD=new_T_FD)
    
    # Return the new face indices
    return new_mesh

def create_domains(mesh: Mesh3D, face_indices: List[List[int]]) -> Mesh3D:
    """Create new domains from faces."""

    num_domains = len(face_indices)
    
    # Create new columns in T_FD
    data, row_ind, col_ind = _calculate_sparse_components(face_indices)
    new_T_FD = hstack([
        mesh.T_FD,
        csr_matrix((data, (row_ind, col_ind)), shape=(mesh.num_faces, num_domains))
    ])
    
    # Create new mesh with updated matrices
    new_mesh = Mesh3D(vertices=mesh.vertices, T_VE=mesh.T_VE, T_EF=mesh.T_EF, T_FD=new_T_FD)
    
    # Return the new domain indices
    return new_mesh