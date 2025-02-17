import unittest
import numpy as np
from mesh.mesh3d import Mesh3D
from mesh.modification.create import (
    mesh_union, add_vertices, create_edges,
    create_faces, create_domains
)

class TestMeshCreation(unittest.TestCase):
    def setUp(self):
        # Create a simple cube mesh for testing
        vertices = np.array([
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ])
        self.cube = Mesh3D(vertices=vertices)
        
        # Create edges (12 edges of a cube)
        edge_verts = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
        ]
        self.cube = create_edges(self.cube, edge_verts)
        
        # Create faces (6 faces of a cube)
        face_edges = [
            [0, 1, 2, 3],      # bottom
            [4, 5, 6, 7],      # top
            [0, 9, 4, 8],      # front
            [1, 10, 5, 9],     # right
            [2, 11, 6, 10],    # back
            [3, 8, 7, 11],     # left
        ]
        self.cube = create_faces(self.cube, face_edges)
        
        # Create a single domain (the cube volume)
        self.cube = create_domains(self.cube, [[0, 1, 2, 3, 4, 5]])

    def test_mesh_union(self):
        # Create two translated cubes
        vertices2 = self.cube.vertices + np.array([2, 0, 0])
        cube2 = Mesh3D(vertices=vertices2, T_VE=self.cube.T_VE, 
                      T_EF=self.cube.T_EF, T_FD=self.cube.T_FD)
        
        # Union the meshes
        union = mesh_union([self.cube, cube2])
        
        # Verify the results
        self.assertEqual(union.num_vertices, 16)  # 8 vertices per cube
        self.assertEqual(union.num_edges, 24)     # 12 edges per cube
        self.assertEqual(union.num_faces, 12)     # 6 faces per cube
        self.assertEqual(union.num_domains, 2)    # 1 domain per cube
        
        # Check that vertices were properly combined
        np.testing.assert_array_equal(union.vertices[:8], self.cube.vertices)
        np.testing.assert_array_equal(union.vertices[8:], vertices2)

    def test_add_vertices(self):
        new_vertices = np.array([[2, 2, 2], [3, 3, 3]])
        mesh = add_vertices(self.cube, new_vertices)
        
        self.assertEqual(mesh.num_vertices, 10)  # 8 original + 2 new
        np.testing.assert_array_equal(mesh.vertices[-2:], new_vertices)

    def test_create_edges(self):
        # Add two new vertices
        mesh = add_vertices(self.cube, np.array([[2, 2, 2], [3, 3, 3]]))
        
        # Create a new edge between the new vertices
        new_edges = [[8, 9]]
        mesh = create_edges(mesh, new_edges)
        
        self.assertEqual(mesh.num_edges, 13)  # 12 original + 1 new
        
        # Verify the new edge connectivity
        edge_vertices = mesh.T_VE[:, 12].nonzero()[0]
        self.assertEqual(set(edge_vertices), {8, 9})

    def test_create_faces(self):
        # Create a new face using existing edges
        new_faces = [[0, 1, 2, 3]]  # bottom face of cube
        mesh = create_faces(self.cube, new_faces)
        
        self.assertEqual(mesh.num_faces, 7)  # 6 original + 1 new
        
        # Verify the new face connectivity
        face_edges = mesh.T_EF[:, 6].nonzero()[0]
        self.assertEqual(set(face_edges), {0, 1, 2, 3})

    def test_create_domains(self):
        # Create a new domain using existing faces
        new_domains = [[0, 1, 2, 3, 4, 5]]  # all faces of cube
        mesh = create_domains(self.cube, new_domains)
        
        self.assertEqual(mesh.num_domains, 2)  # 1 original + 1 new
        
        # Verify the new domain connectivity
        domain_faces = mesh.T_FD[:, 1].nonzero()[0]
        self.assertEqual(set(domain_faces), {0, 1, 2, 3, 4, 5})

if __name__ == '__main__':
    unittest.main() 