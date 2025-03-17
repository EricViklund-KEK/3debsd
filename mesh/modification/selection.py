from mesh.mesh3d import Mesh3D
from scipy.sparse import csr_matrix

def closure(mesh: Mesh3D, index_dict: dict) -> dict:
    """Finds the closure of the objects indicated by their index in the mesh.
    
    The closure includes all specified simplices and their faces (subsimplices).
    
    Args:
        mesh: The input mesh
        index_dict: Dictionary with keys 'vertices', 'edges', 'faces', and/or 'domains'
                   containing lists of indices for each type
    
    Returns:
        dict: Dictionary containing lists of indices for each type of simplex in the closure
    """
    # Initialize empty sets for each dimension
    v_set = set(index_dict.get('vertices', []))
    e_set = set(index_dict.get('edges', []))
    f_set = set(index_dict.get('faces', []))
    d_set = set(index_dict.get('domains', []))
    
    # Add faces of domains
    for d in d_set:
        f_set.update(mesh.T_FD[:, d].nonzero()[0])
    
    # Add faces of faces
    for f in f_set:
        e_set.update(mesh.T_EF[:, f].nonzero()[0])
    
    # Add faces of edges
    for e in e_set:
        v_set.update(mesh.T_VE[:, e].nonzero()[0])
    
    return {
        'vertices': sorted(v_set),
        'edges': sorted(e_set),
        'faces': sorted(f_set),
        'domains': sorted(d_set)
    }

def star(mesh: Mesh3D, index_dict: dict) -> dict:
    """Finds the star of the objects indicated by their index in the mesh.
    
    The star includes all specified simplices and all simplices that contain them.
    
    Args:
        mesh: The input mesh
        index_dict: Dictionary with keys 'vertices', 'edges', 'faces', and/or 'domains'
                   containing lists of indices for each type
    
    Returns:
        dict: Dictionary containing lists of indices for each type of simplex in the star
    """
    # Initialize sets with given indices
    v_set = set(index_dict.get('vertices', []))
    e_set = set(index_dict.get('edges', []))
    f_set = set(index_dict.get('faces', []))
    d_set = set(index_dict.get('domains', []))
    
    # Find cofaces of vertices
    for v in v_set:
        e_set.update(mesh.T_VE[v].nonzero()[1])
    
    # Find cofaces of edges
    for e in e_set:
        f_set.update(mesh.T_EF[e].nonzero()[1])
    
    # Find cofaces of faces
    for f in f_set:
        d_set.update(mesh.T_FD[f].nonzero()[1])
    
    # Get all vertices needed for the selected elements
    for e in e_set:
        v_set.update(mesh.T_VE[:, e].nonzero()[0])
    
    return {
        'vertices': sorted(v_set),
        'edges': sorted(e_set),
        'faces': sorted(f_set),
        'domains': sorted(d_set)
    }

def link(mesh: Mesh3D, index_dict: dict) -> dict:
    """Finds the link of the objects indicated by their index in the mesh.
    
    The link is the closure of the star minus the star of the closure.
    
    Args:
        mesh: The input mesh
        index_dict: Dictionary with keys 'vertices', 'edges', 'faces', and/or 'domains'
                   containing lists of indices for each type
    
    Returns:
        dict: Dictionary containing lists of indices for each type of simplex in the link
    """
    # Get the star and closure
    star_indices = star(mesh, index_dict)
    closure_indices = closure(mesh, index_dict)

    # Get the closure of the star
    closure_of_star = closure(mesh, star_indices)
    
    # Get the star of the closure
    star_of_closure = star(mesh, closure_indices)

    return difference(closure_of_star, star_of_closure)

def intersection(index_dicts: list[dict]) -> dict:
    """Calculate the intersection of a list of index dicts.
    
    Args:
        index_dicts: List of dictionaries, each with keys 'vertices', 'edges', 'faces', 
                    and/or 'domains' containing lists of indices
    
    Returns:
        dict: Dictionary containing the intersection of indices for each type
    """
    if not index_dicts:
        return {'vertices': [], 'edges': [], 'faces': [], 'domains': []}
    
    result = {
        'vertices': set(index_dicts[0].get('vertices', [])),
        'edges': set(index_dicts[0].get('edges', [])),
        'faces': set(index_dicts[0].get('faces', [])),
        'domains': set(index_dicts[0].get('domains', []))
    }
    
    for d in index_dicts[1:]:
        result['vertices'] &= set(d.get('vertices', []))
        result['edges'] &= set(d.get('edges', []))
        result['faces'] &= set(d.get('faces', []))
        result['domains'] &= set(d.get('domains', []))
    
    return {k: sorted(v) for k, v in result.items()}

def union(index_dicts: list[dict]) -> dict:
    """Calculate the union of a list of index dicts.
    
    Args:
        index_dicts: List of dictionaries, each with keys 'vertices', 'edges', 'faces', 
                    and/or 'domains' containing lists of indices
    
    Returns:
        dict: Dictionary containing the union of indices for each type
    """
    result = {
        'vertices': set(),
        'edges': set(),
        'faces': set(),
        'domains': set()
    }
    
    for d in index_dicts:
        result['vertices'].update(d.get('vertices', []))
        result['edges'].update(d.get('edges', []))
        result['faces'].update(d.get('faces', []))
        result['domains'].update(d.get('domains', []))
    
    return {k: sorted(v) for k, v in result.items()}

def difference(index_dict1: dict, index_dict2: dict) -> dict:
    """Calculate the set difference between two index dicts (index_dict1 - index_dict2).
    
    Args:
        index_dict1: First dictionary with indices
        index_dict2: Second dictionary with indices to subtract
    
    Returns:
        dict: Dictionary containing the difference of indices for each type
    """
    result = {
        'vertices': set(index_dict1.get('vertices', [])) - set(index_dict2.get('vertices', [])),
        'edges': set(index_dict1.get('edges', [])) - set(index_dict2.get('edges', [])),
        'faces': set(index_dict1.get('faces', [])) - set(index_dict2.get('faces', [])),
        'domains': set(index_dict1.get('domains', [])) - set(index_dict2.get('domains', []))
    }
    
    return {k: sorted(v) for k, v in result.items()}