import unittest
import numpy as np
from mesh.mesh3d import Mesh3D

from mesh.modification.create import (
    mesh_union, add_vertices, create_edges,
    create_faces, create_domains
)
from mesh.modification.selection import (
    closure, star, link, intersection, union, difference
)

class TestSelection(unittest.TestCase):
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
        
        # Create edges
        edge_verts = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # top
            [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
        ]
        self.cube = create_edges(self.cube, edge_verts)
        
        # Create faces
        face_edges = [
            [0, 1, 2, 3],      # bottom
            [4, 5, 6, 7],      # top 
            [0, 9, 4, 8],      # front
            [1, 10, 5, 9],     # right
            [2, 11, 6, 10],    # back
            [3, 8, 7, 11],     # left
        ]
        self.cube = create_faces(self.cube, face_edges)
        
        # Create domain
        self.cube = create_domains(self.cube, [[0, 1, 2, 3, 4, 5]])

    def test_closure(self):
        # Test vertex closure
        vert_closure = closure(self.cube, {'vertices': [0]})
        self.assertEqual(len(vert_closure['vertices']), 1)
        self.assertEqual(vert_closure['vertices'], [0])
        self.assertEqual(len(vert_closure['edges']), 0)
        
        # Test edge closure
        edge_closure = closure(self.cube, {'edges': [0]})
        self.assertEqual(len(edge_closure['vertices']), 2)
        self.assertEqual(len(edge_closure['edges']), 1)
        
        # Test face closure
        face_closure = closure(self.cube, {'faces': [0]})
        self.assertEqual(len(face_closure['vertices']), 4)
        self.assertEqual(len(face_closure['edges']), 4)
        self.assertEqual(len(face_closure['faces']), 1)
        
        # Test domain closure
        domain_closure = closure(self.cube, {'domains': [0]})
        self.assertEqual(len(domain_closure['vertices']), 8)
        self.assertEqual(len(domain_closure['edges']), 12)
        self.assertEqual(len(domain_closure['faces']), 6)
        self.assertEqual(len(domain_closure['domains']), 1)

    def test_star(self):
        # Test vertex star
        vert_star = star(self.cube, {'vertices': [0]})
        self.assertTrue(len(vert_star['edges']) >= 3)
        self.assertTrue(len(vert_star['faces']) >= 3)
        self.assertTrue(len(vert_star['domains']) >= 1)
        
        # Test edge star
        edge_star = star(self.cube, {'edges': [0]})
        self.assertTrue(len(edge_star['faces']) >= 2)
        self.assertTrue(len(edge_star['domains']) >= 1)
        
        # Test face star
        face_star = star(self.cube, {'faces': [0]})
        self.assertTrue(len(face_star['domains']) >= 1)

    def test_link(self):
        # Test vertex link
        vert_link = link(self.cube, {'vertices': [0]})
        self.assertTrue(len(vert_link['vertices']) > 0)
        self.assertTrue(len(vert_link['edges']) > 0)
        
        # Test edge link
        edge_link = link(self.cube, {'edges': [0]})
        self.assertTrue(len(edge_link['vertices']) > 0)

    def test_set_operations(self):
        # Test intersection
        dict1 = {'vertices': [0, 1], 'edges': [0, 1]}
        dict2 = {'vertices': [1, 2], 'edges': [1, 2]}
        result = intersection([dict1, dict2])
        self.assertEqual(result['vertices'], [1])
        self.assertEqual(result['edges'], [1])
        
        # Test union
        result = union([dict1, dict2])
        self.assertEqual(set(result['vertices']), {0, 1, 2})
        self.assertEqual(set(result['edges']), {0, 1, 2})
        
        # Test difference
        result = difference(dict1, dict2)
        self.assertEqual(result['vertices'], [0])
        self.assertEqual(result['edges'], [0])

if __name__ == '__main__':
    unittest.main()