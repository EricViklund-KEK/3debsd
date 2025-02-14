
import sys

sys.path.insert(0,'.')

import unittest
import numpy as np
from scipy.sparse import csr_matrix
from mesh.mesh_modifier import Plane, Edge, Face, Domain, MeshModifier
from mesh.mesh3d import Mesh3D
from mesh.voronoi3d import create_voronoi_mesh
from gui.main_window import MainWindow
from PySide6.QtWidgets import QApplication

class TestMeshIntersection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize Qt Application for visualization
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.main_window = MainWindow()
        
    def setUp(self):
        # Create a test mesh from a cube from (0,0,0) to (1,1,1)
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # Top face
        ])

        # Create T_VE - 12 edges connecting vertices
        T_VE = csr_matrix((np.ones(24, dtype=bool),
                          ([0,1,2,3,4,5,6,7,0,1,2,3,1,2,3,0,5,6,7,4,4,5,6,7],  # Row indices
                           [0,1,2,3,4,5,6,7,8,9,10,11,0,1,2,3,4,5,6,7,8,9,10,11])), # Col indices
                         shape=(8, 12))
                
        # Create T_EF - 6 faces each using 4 edges
        T_EF = csr_matrix((np.ones(24, dtype=bool),
                          ([0,1,2,3,4,5,6,7,8,9,10,11,0,1,2,3,4,5,6,7,8,9,10,11],  # Row indices
                           [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5])), # Col indices
                         shape=(12, 6))

        # Create T_FD - 6 faces forming 1 domain
        T_FD = csr_matrix((np.ones(6, dtype=bool),
                          ([0,1,2,3,4,5],  # Row indices
                           [0,0,0,0,0,0])), # Col indices
                         shape=(6, 1))

        # Create mesh and modifier
        self.mesh = Mesh3D(vertices, T_VE=T_VE, T_EF=T_EF, T_FD=T_FD)
        self.modifier = MeshModifier(self.mesh)
        
    def test_diagonal_plane_intersection(self):
        """Test intersection with a plane cutting diagonally through the cube"""
        # Create a plane that cuts through the cube diagonally
        plane = Plane(np.array([0.5, 0.5, 0.5]), np.array([1, 1, 1]))
        
        # Perform intersection
        result = self.modifier.intersect_with_plane(plane)
        
        # Verify intersection results
        self.assertGreater(len(result['intersection_points']), 0)
        self.assertGreater(len(result['intersection_edges']), 0)
        self.assertGreater(len(result['intersection_faces']), 0)
        
        # Verify specific intersection properties
        intersection_points = result['intersection_points']
        self.assertEqual(len(intersection_points), 6)  # Should create 6 intersection points
        
        # Check that intersection points lie on the plane
        distances = plane.signed_distance(intersection_points)
        np.testing.assert_array_almost_equal(distances, np.zeros_like(distances))
        
        # Visualize the result
        self.main_window.plot_ebsd_mesh(self.mesh)
        self.main_window.show()
        self.app.processEvents()
        
    def test_remove_outside_half(self):
        """Test removing half of the cube using remove_outside"""
        # Create a plane that cuts the cube in half
        plane = Plane(np.array([0.5, 0, 0]), np.array([1, 0, 0]))
        
        # Remove everything to the right of the plane
        result = self.modifier.remove_outside(plane)
        
        # Verify removal results
        self.assertGreater(len(result['removed_vertices']), 0)
        self.assertGreater(len(result['removed_edges']), 0)
        self.assertGreater(len(result['removed_faces']), 0)
        
        # Verify specific removal properties
        self.assertEqual(len(result['removed_vertices']), 4)  # Should remove 4 vertices
        self.assertEqual(len(result['removed_edges']), 6)     # Should remove 6 edges
        self.assertEqual(len(result['removed_faces']), 3)     # Should remove 3 faces
        
        # Verify remaining mesh structure
        self.assertEqual(self.mesh.num_vertices, 4)
        self.assertEqual(self.mesh.num_edges, 6)
        self.assertEqual(self.mesh.num_faces, 3)
        
        # Verify mesh consistency
        self.assertTrue(self.mesh.is_valid())
        
        # Visualize the result
        self.main_window.plot_ebsd_mesh(self.mesh)
        self.main_window.show()
        self.app.processEvents()
        
    def test_complex_intersection(self):
        """Test intersection with multiple planes creating complex cuts"""
        # Create three intersecting planes
        planes = [
            Plane(np.array([0.5, 0, 0]), np.array([1, 0, 0])),    # YZ plane
            Plane(np.array([0, 0.5, 0]), np.array([0, 1, 0])),    # XZ plane
            Plane(np.array([0, 0, 0.5]), np.array([0, 0, 1]))     # XY plane
        ]
        
        # Store initial state for comparison
        initial_vertices = self.mesh.num_vertices
        initial_edges = self.mesh.num_edges
        initial_faces = self.mesh.num_faces
        
        # Perform sequential intersections
        results = []
        for i, plane in enumerate(planes):
            result = self.modifier.intersect_with_plane(plane)
            results.append(result)
            
            # Verify that each intersection creates new geometry
            self.assertGreater(self.mesh.num_vertices, initial_vertices + i*2)
            self.assertGreater(self.mesh.num_edges, initial_edges + i*3)
            self.assertGreater(self.mesh.num_faces, initial_faces + i)
            
            # Verify mesh consistency after each cut
            self.assertTrue(self.mesh.is_valid())
            
            # Visualize after each intersection
            self.main_window.plot_ebsd_mesh(self.mesh)
            self.main_window.show()
            self.app.processEvents()
            
        # Verify final state
        self.assertTrue(self.mesh.is_valid())
        
        # Verify that we created new geometry at each step
        for result in results:
            self.assertGreater(len(result['intersection_points']), 0)
            self.assertGreater(len(result['intersection_edges']), 0)
            self.assertGreater(len(result['intersection_faces']), 0)

if __name__ == '__main__':
    unittest.main()