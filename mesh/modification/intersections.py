from mesh.primitives import Plane, Edge, Face, Domain
from mesh.modification import create, remove
from mesh import Mesh3D

import numpy as np





def intersect_with_plane(mesh, plane: Plane) -> dict:
    """Calculate intersection between mesh and plane.
    
    Args:
        plane (Plane): Plane to intersect with
        
    Returns:
        dict: Dictionary containing intersection information
    """
    intersection_points = []
    intersection_edges = []
    intersection_faces = []
    modified_edges = []
    modified_faces = []
    modified_domains = []
    new_edges = []
    new_faces = []
    new_domains = []

    # Check each domain in the mesh for intersection with the plane
    for domain_idx in range(mesh.num_domains):
        # Get faces for this domain
        domain_faces = []
        face_indices = mesh.T_FD[:,domain_idx].nonzero()[0]
        
        for face_idx in face_indices:
            # Get edges for this face
            face_edges = []
            edge_indices = mesh.T_EF[:,face_idx].nonzero()[0]
            
            for edge_idx in edge_indices:
                # Get vertices for this edge
                vert_indices = mesh.T_VE[:,edge_idx].nonzero()[0]
                if len(vert_indices) == 2:
                    v1 = mesh.vertices[vert_indices[0]]
                    v2 = mesh.vertices[vert_indices[1]]
                    face_edges.append(Edge(v1, v2))
            
            if face_edges:
                # Create face object
                domain_faces.append(Face(face_edges))
        
        if domain_faces:
            # Create domain and intersect with plane
            domain = Domain(domain_faces)
            modified_domain, intersection_face, new_domain_list = domain.intersect_plane(plane)
            
            # Update mesh with modified domain
            create_domain(mesh, modified_domain)
            
            if intersection_face:
                # Create intersection face in mesh
                face_idx = create_face(mesh, intersection_face)
                intersection_faces.append(intersection_face)

                # Add intersection results
                for edge in intersection_face.edges:
                    edge_idx = create_edge(mesh, edge)
                    intersection_points.extend([edge.start, edge.end])
                    intersection_edges.append(edge)
                    new_edges.append(edge_idx)
                
                modified_domains.append((domain_idx, modified_domain))
                
                # Create any new domains
                for new_domain in new_domain_list:
                    domain_idx = create_domain(mesh, new_domain)
                    new_domains.append(domain_idx)

    return {
        'intersection_points': np.array(intersection_points), #new intersection vertices created
        'intersection_edges': intersection_edges, #new intersection edges created
        'intersection_faces': intersection_faces, #new intersection faces created
        'modified_edges': modified_edges, #existing edges modified
        'modified_faces': modified_faces, #existing faces modified
        'modified_domains': modified_domains, #existing domains modified
        'new_edges': new_edges, #new edges created due to edge splitting
        'new_faces': new_faces, #new faces created due to face splitting            
        'new_domains': new_domains #new domains created due to domain splitting
    }

def remove_outside(mesh, plane: Plane) -> dict:
    """Remove all parts of the mesh that lie outside the given plane.
    The plane's normal points towards the region to be kept.
    
    Args:
        plane (Plane): Plane defining the cut boundary. Points in the direction
                        of the normal will be kept, points in the opposite direction
                        will be removed.
        
    Returns:
        dict: Information about the operation including:
            - intersection_info: Results from intersect_with_plane
            - removed_vertices: Indices of removed vertices
            - removed_edges: Indices of removed edges
            - removed_faces: Indices of removed faces
            - removed_domains: Indices of removed domains
    """
    # First intersect with plane to create new geometry at intersection
    intersection_info = intersect_with_plane(mesh, plane)
    
    # Find vertices that lie outside the plane (positive signed distance)
    signed_distances = plane.signed_distance(mesh.vertices)
    outside_vertex_indices = np.where(signed_distances > 0)[0]
    
    # Remove outside vertices and track affected elements
    if len(outside_vertex_indices) > 0:
        affected_edges = delete_vertices(mesh, outside_vertex_indices.tolist())
    else:
        affected_edges = []
        
    # Combine results
    result = {
        'intersection_info': intersection_info,
        'removed_vertices': outside_vertex_indices,
        'removed_edges': affected_edges,
        'removed_faces': intersection_info['modified_faces'],
        'removed_domains': [idx for idx, _ in intersection_info['modified_domains']]
    }
    
    return result