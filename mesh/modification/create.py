import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from mesh.mesh3d import Mesh3D
from mesh.primitives import Edge, Face, Domain
from mesh.option import Option
from typing import List, Dict, Set, Optional, Tuple

def add_vertices(mesh: Mesh3D, vertices: np.ndarray) -> Tuple[Mesh3D, List[int]]:
    """Add new vertices to the mesh.
    
    Args:
        mesh (Mesh3D): The mesh to modify
        vertices (np.ndarray): Vertices to add (shape: (N, 3))
        
    Returns:
        Tuple[Mesh3D, List[int]]: The modified mesh and indices of all vertices (both existing and new)
    """
    vertices = np.asarray(vertices, dtype=float)
    
    # If mesh has no vertices, simply set them
    if mesh.vertices.is_none():
        mesh.vertices = Option.some(vertices)
        mesh.T_VE = Option.some(csr_matrix((len(vertices), 0)))
        return mesh, list(range(len(vertices)))
    
    # Get existing vertices
    existing = mesh.vertices.value
    
    # Process vertices and track indices
    all_indices = []
    new_vertices = []
    
    for vertex in vertices:
        # Check if vertex exists using np.allclose
        found = False
        for i, existing_vertex in enumerate(existing):
            if np.allclose(vertex, existing_vertex):
                all_indices.append(i)
                found = True
                break
        
        if not found:
            new_vertices.append(vertex)
            all_indices.append(len(existing) + len(new_vertices) - 1)
    
    if not new_vertices:
        return mesh, all_indices
        
    new_vertices = np.array(new_vertices)
    n_new = len(new_vertices)
    
    # Add vertices to mesh using Option
    mesh.vertices = mesh.vertices.map(
        lambda v: np.vstack([v, new_vertices])
    )

    # Add new row to T_VE
    mesh.T_VE = mesh.T_VE.map(
        lambda t_ve: vstack([t_ve, csr_matrix((1, t_ve.shape[1]))])
    ).or_else(Option.some(csr_matrix((n_new, 0))))
    
    # Invalidate KD-tree since we modified vertices
    mesh._kdtree = Option.none()
    
    return mesh, all_indices

def create_edge(mesh: Mesh3D, edge: Edge) -> Tuple[Mesh3D, int]:
    """Create a new edge between vertices.
    
    Args:
        mesh (Mesh3D): The mesh to modify
        edge (Edge): Edge to add to the mesh
        
    Returns:
        Tuple[Mesh3D, int]: The modified mesh and the index of the created edge
    """
    # Add vertices and get their indices
    vertices = np.vstack([edge.start, edge.end])
    mesh, indices = add_vertices(mesh, vertices)
    
    # Get vertex indices
    start_idx = indices[0] if indices else 0
    end_idx = indices[1] if len(indices) > 1 else 1

    # Check if edge already exists
    if not mesh.T_VE.is_none():
        t_ve = mesh.T_VE.value
        n_vertices = t_ve.shape[0]
        
        # Create edge column vector
        new_col = csr_matrix(([1, 1], ([start_idx, end_idx], [0, 0])), 
                            shape=(n_vertices, 1))
        
        # Check each existing edge
        for i in range(t_ve.shape[1]):
            existing_edge = t_ve.getcol(i)
            # Compare the nonzero indices
            if (set(existing_edge.nonzero()[0]) == set([start_idx, end_idx])):
                return mesh, i
    
    # Edge doesn't exist, create it
    edge_idx = mesh.T_VE.value.shape[1] if not mesh.T_VE.is_none() else 0
    
    # Add new column to T_VE
    new_col = csr_matrix(([1, 1], ([start_idx, end_idx], [0, 0])), 
                        shape=(mesh.vertices.value.shape[0], 1))
    mesh.T_VE = mesh.T_VE.map(
        lambda t_ve: hstack([t_ve, new_col])
    ).or_else(Option.some(new_col))

    # Add new row to T_EF
    mesh.T_EF = mesh.T_EF.map(
        lambda t_ef: vstack([t_ef, csr_matrix((1, t_ef.shape[1]))])
    ).or_else(Option.some(csr_matrix((1, 0))))

    return mesh, edge_idx

def create_face(mesh: Mesh3D, face: Face) -> Tuple[Mesh3D, int]:
    """Create a new face from edges.
    
    Args:
        mesh (Mesh3D): The mesh to modify
        face (Face): Face to add to the mesh
        
    Returns:
        Tuple[Mesh3D, int]: The modified mesh and the index of the created face
    """
    edge_indices = []
    
    # Add all edges and collect their indices
    for edge in face.edges:
        mesh, edge_idx = create_edge(mesh, edge)
        edge_indices.append(edge_idx)
    
    # Check if face already exists
    if not mesh.T_EF.is_none():
        t_ef = mesh.T_EF.value
        n_edges = t_ef.shape[0]
        
        # Create face column vector
        new_col = csr_matrix(([1] * len(edge_indices), (edge_indices, [0] * len(edge_indices))),
                            shape=(n_edges, 1))
        
        # Check each existing face
        for i in range(t_ef.shape[1]):
            existing_face = t_ef.getcol(i)
            # Compare the nonzero indices
            if (set(existing_face.nonzero()[0]) == set(edge_indices)):
                return mesh, i
    
    # Face doesn't exist, create it
    face_idx = mesh.T_EF.value.shape[1] if not mesh.T_EF.is_none() else 0
    
    # Create new column for T_EF
    new_col = csr_matrix(([1] * len(edge_indices), (edge_indices, [0] * len(edge_indices))),
                        shape=(mesh.T_EF.value.shape[0] if not mesh.T_EF.is_none() else len(edge_indices), 1))
    mesh.T_EF = mesh.T_EF.map(
        lambda t_ef: hstack([t_ef, new_col])
    ).or_else(Option.some(new_col))

    # Create new row for T_FD
    mesh.T_FD = mesh.T_FD.map(
        lambda t_fd: vstack([t_fd, csr_matrix((1, t_fd.shape[1]))])
    ).or_else(Option.some(csr_matrix((1, 0))))
            
    return mesh, face_idx

def create_domain(mesh: Mesh3D, domain: Domain) -> Tuple[Mesh3D, int]:
    """Create a new domain from faces.
    
    Args:
        mesh (Mesh3D): The mesh to modify
        domain (Domain): Domain to add to the mesh
        
    Returns:
        Tuple[Mesh3D, int]: The modified mesh and the index of the created domain
    """
    face_indices = []
    
    # Add all faces and collect their indices
    for face in domain.faces:
        mesh, face_idx = create_face(mesh, face)
        face_indices.append(face_idx)
    
    # Check if domain already exists
    if not mesh.T_FD.is_none():
        t_fd = mesh.T_FD.value
        n_faces = t_fd.shape[0]
        
        # Create domain column vector
        new_col = csr_matrix(([1] * len(face_indices), (face_indices, [0] * len(face_indices))),
                            shape=(n_faces, 1))
        
        # Check each existing domain
        for i in range(t_fd.shape[1]):
            existing_domain = t_fd.getcol(i)
            # Compare the nonzero indices
            if (set(existing_domain.nonzero()[0]) == set(face_indices)):
                return mesh, i
    
    # Domain doesn't exist, create it
    domain_idx = mesh.T_FD.value.shape[1] if not mesh.T_FD.is_none() else 0
    
    # Create new column for T_FD
    new_col = csr_matrix(([1] * len(face_indices), (face_indices, [0] * len(face_indices))),
                        shape=(mesh.T_FD.value.shape[0] if not mesh.T_FD.is_none() else len(face_indices), 1))
    mesh.T_FD = mesh.T_FD.map(
        lambda t_fd: hstack([t_fd, new_col])
    ).or_else(Option.some(new_col))
        
    return mesh, domain_idx