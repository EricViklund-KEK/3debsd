import sys

sys.path.insert(0,'.')

import numpy as np
from mesh.mesh3d import Mesh3D
from mesh.primitives import Edge, Face, Domain
from mesh.modification.create import add_vertices, create_edge, create_face, create_domain

def test_add_vertices():
    mesh = Mesh3D()
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    
    # Add vertices
    modified_mesh, indices = add_vertices(mesh, vertices)
    
    assert modified_mesh.num_vertices == 3
    assert modified_mesh.vertices.map(lambda v: np.array_equal(v, vertices)).get_or_else(False)
    assert modified_mesh.T_VE.map(lambda m: m.shape == (3, 0)).get_or_else(False)
    assert len(indices) == 3
    assert indices == [0, 1, 2]

    # Add another vertex
    new_vertex = np.array([[0, 0, 1]])
    modified_mesh, new_indices = add_vertices(modified_mesh, new_vertex)
    
    assert modified_mesh.num_vertices == 4
    assert modified_mesh.vertices.map(lambda v: np.array_equal(v[-1], new_vertex[0])).get_or_else(False)
    assert modified_mesh.T_VE.map(lambda m: m.shape == (4, 0)).get_or_else(False)
    assert len(new_indices) == 1
    assert new_indices == [3]

def test_create_edge():
    mesh = Mesh3D()
    edge = Edge(np.array([0, 0, 0]), np.array([1, 0, 0]))

    # Create edge
    modified_mesh, edge_idx = create_edge(mesh, edge)
    
    assert modified_mesh.num_vertices == 2
    assert modified_mesh.num_edges == 1
    assert modified_mesh.T_VE.map(lambda m: m.shape == (2, 1)).get_or_else(False)
    assert modified_mesh.T_EF.map(lambda m: m.shape == (1, 0)).get_or_else(False)
    assert edge_idx == 0

    # Create another edge
    edge2 = Edge(np.array([1, 0, 0]), np.array([1, 1, 0]))
    modified_mesh, edge2_idx = create_edge(modified_mesh, edge2)
    
    assert modified_mesh.num_vertices == 3
    assert modified_mesh.num_edges == 2
    assert modified_mesh.T_VE.map(lambda m: m.shape == (3, 2)).get_or_else(False)
    assert modified_mesh.T_EF.map(lambda m: m.shape == (2, 0)).get_or_else(False)
    assert edge2_idx == 1

    # Add a vertex
    new_vertex = np.array([[0, 0, 1]])
    modified_mesh, new_indices = add_vertices(modified_mesh, new_vertex)
    
    assert modified_mesh.num_vertices == 4
    assert modified_mesh.num_edges == 2
    assert modified_mesh.T_VE.map(lambda m: m.shape == (4, 2)).get_or_else(False)
    assert len(new_indices) == 1
    assert new_indices == [3]

def test_create_face():
    mesh = Mesh3D()
    # Create a triangular face
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 0, 0])
    v3 = np.array([0, 1, 0])
    edges = [
        Edge(v1, v2),
        Edge(v2, v3),
        Edge(v3, v1)
    ]
    face = Face(edges)
    
    # Create face
    modified_mesh, face_idx = create_face(mesh, face)
    
    assert modified_mesh.num_vertices == 3
    assert modified_mesh.num_edges == 3
    assert modified_mesh.num_faces == 1
    assert modified_mesh.T_VE.map(lambda m: m.shape == (3, 3)).get_or_else(False)
    assert modified_mesh.T_EF.map(lambda m: m.shape == (3, 1)).get_or_else(False)
    assert modified_mesh.T_FD.map(lambda m: m.shape == (1, 0)).get_or_else(False)
    assert face_idx == 0

    # Create another face
    v4 = np.array([1, 1, 0])
    edges2 = [
        Edge(v2, v3),
        Edge(v3, v4),
        Edge(v4, v2)
    ]
    face2 = Face(edges2)
    modified_mesh, face2_idx = create_face(modified_mesh, face2)
    
    assert modified_mesh.num_vertices == 4
    assert modified_mesh.num_edges == 5  # One edge is shared
    assert modified_mesh.num_faces == 2
    assert modified_mesh.T_VE.map(lambda m: m.shape == (4, 5)).get_or_else(False)
    assert modified_mesh.T_EF.map(lambda m: m.shape == (5, 2)).get_or_else(False)
    assert face2_idx == 1

    # Add an edge
    edge = Edge(v1, v4)
    modified_mesh, edge_idx = create_edge(modified_mesh, edge)
    
    assert modified_mesh.num_edges == 6
    assert modified_mesh.T_VE.map(lambda m: m.shape == (4, 6)).get_or_else(False)
    assert modified_mesh.T_EF.map(lambda m: m.shape == (6, 2)).get_or_else(False)
    assert edge_idx == 5

    # Add a vertex
    new_vertex = np.array([[0, 0, 1]])
    modified_mesh, new_indices = add_vertices(modified_mesh, new_vertex)
    
    assert modified_mesh.num_vertices == 5
    assert modified_mesh.T_VE.map(lambda m: m.shape == (5, 6)).get_or_else(False)
    assert len(new_indices) == 1
    assert new_indices == [4]

def test_create_domain():
    mesh = Mesh3D()
    # Create a tetrahedron
    v1 = np.array([0, 0, 0])
    v2 = np.array([1, 0, 0])
    v3 = np.array([0, 1, 0])
    v4 = np.array([0, 0, 1])
    
    # Create four triangular faces
    faces = []
    faces.append(Face([
        Edge(v1, v2),
        Edge(v2, v3),
        Edge(v3, v1)
    ]))
    faces.append(Face([
        Edge(v1, v2),
        Edge(v2, v4),
        Edge(v4, v1)
    ]))
    faces.append(Face([
        Edge(v2, v3),
        Edge(v3, v4),
        Edge(v4, v2)
    ]))
    faces.append(Face([
        Edge(v3, v1),
        Edge(v1, v4),
        Edge(v4, v3)
    ]))
    
    domain = Domain(faces)
    
    # Create domain
    modified_mesh, domain_idx = create_domain(mesh, domain)
    
    assert modified_mesh.num_vertices == 4
    assert modified_mesh.num_edges == 6
    assert modified_mesh.num_faces == 4
    assert modified_mesh.num_domains == 1
    assert modified_mesh.T_VE.map(lambda m: m.shape == (4, 6)).get_or_else(False)
    assert modified_mesh.T_EF.map(lambda m: m.shape == (6, 4)).get_or_else(False)
    assert modified_mesh.T_FD.map(lambda m: m.shape == (4, 1)).get_or_else(False)
    assert domain_idx == 0

    # Create another domain (a pyramid on top of one face)
    v5 = np.array([0.5, 0.5, 1])
    pyramid_faces = []
    # Base face is already created (one of the tetrahedron faces)
    pyramid_faces.append(faces[0])
    # Add four triangular faces for the sides
    pyramid_faces.append(Face([
        Edge(v1, v2),
        Edge(v2, v5),
        Edge(v5, v1)
    ]))
    pyramid_faces.append(Face([
        Edge(v2, v3),
        Edge(v3, v5),
        Edge(v5, v2)
    ]))
    pyramid_faces.append(Face([
        Edge(v3, v1),
        Edge(v1, v5),
        Edge(v5, v3)
    ]))
    
    pyramid_domain = Domain(pyramid_faces)
    modified_mesh, domain2_idx = create_domain(modified_mesh, pyramid_domain)
    
    assert modified_mesh.num_vertices == 5
    assert modified_mesh.num_edges == 9
    assert modified_mesh.num_faces == 7
    assert modified_mesh.num_domains == 2
    assert modified_mesh.T_VE.map(lambda m: m.shape == (5, 9)).get_or_else(False)
    assert modified_mesh.T_EF.map(lambda m: m.shape == (9, 7)).get_or_else(False)
    assert modified_mesh.T_FD.map(lambda m: m.shape == (7, 2)).get_or_else(False)
    assert domain2_idx == 1