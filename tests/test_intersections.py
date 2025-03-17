import unittest
import numpy as np
from mesh.mesh3d import Mesh3D
from mesh.geometry.plane import Plane
from mesh.modification.create import create_edges, create_faces, create_domains
from mesh.modification.intersections import intersect_with_plane

class TestIntersections(unittest.TestCase):
    def setUp(self):
        # Create a 4x4x4 cube mesh
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        z = np.linspace(0, 4, 5)
        
        # Create vertex grid
        X, Y, Z = np.meshgrid(x, y, z)
        vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        self.mesh = Mesh3D(vertices=vertices)
        
        # Create edges with connectivity tracking
        edge_verts = []
        n = 5  # vertices per side
        edge_map = {}  # Map vertex pairs to edge indices
        
        # Add x-direction edges
        for k in range(n):
            for j in range(n):
                for i in range(n-1):
                    idx = k*n*n + j*n + i
                    edge_verts.append([idx, idx+1])
                    edge_map[tuple(sorted([idx, idx+1]))] = len(edge_verts) - 1
        
        # Add y-direction edges
        for k in range(n):
            for j in range(n-1):
                for i in range(n):
                    idx = k*n*n + j*n + i
                    edge_verts.append([idx, idx+n])
                    edge_map[tuple(sorted([idx, idx+n]))] = len(edge_verts) - 1
        
        # Add z-direction edges
        for k in range(n-1):
            for j in range(n):
                for i in range(n):
                    idx = k*n*n + j*n + i
                    edge_verts.append([idx, idx+n*n])
                    edge_map[tuple(sorted([idx, idx+n*n]))] = len(edge_verts) - 1
        
        self.mesh = create_edges(self.mesh, edge_verts)
        
        # Create faces
        face_edges = []
        # Add xy-plane faces
        for k in range(n):
            for j in range(n-1):
                for i in range(n-1):
                    idx = k*n*n + j*n + i
                    e1 = edge_map[tuple(sorted([idx, idx+1]))]
                    e2 = edge_map[tuple(sorted([idx+1, idx+n+1]))]
                    e3 = edge_map[tuple(sorted([idx+n, idx+n+1]))]
                    e4 = edge_map[tuple(sorted([idx, idx+n]))]
                    face_edges.append([e1, e2, e3, e4])
        
        self.mesh = create_faces(self.mesh, face_edges)
        
        # Create domains
        domain_faces = []
        for k in range(n-1):
            for j in range(n-1):
                for i in range(n-1):
                    # Get the six faces of each cube
                    faces = []
                    idx = k*n*n + j*n + i
                    # Add xy-plane faces (top and bottom)
                    faces.append(k*(n-1)*(n-1) + j*(n-1) + i)  # bottom
                    faces.append((k+1)*(n-1)*(n-1) + j*(n-1) + i)  # top
                    # Add remaining faces
                    domain_faces.append(faces)
                    
        self.mesh = create_domains(self.mesh, domain_faces)

    def test_diagonal_plane_intersection(self):
        point = np.array([2, 2, 2])  # Center of the cube
        normal = np.array([1, 1, 1])
        plane = Plane(point, normal)
        
        # Perform intersection
        above_mesh, below_mesh = intersect_with_plane(self.mesh, plane)
        
        # Verify results
        self.assertGreater(above_mesh.num_vertices, 0)
        self.assertGreater(below_mesh.num_vertices, 0)
        
        # Verify all vertices are on correct side of plane
        for v in above_mesh.vertices:
            self.assertGreaterEqual(np.dot(v - point, normal), -1e-10)
            
        for v in below_mesh.vertices:
            self.assertLessEqual(np.dot(v - point, normal), 1e-10)

    def test_horizontal_plane_intersection(self):
        point = np.array([0, 0, 1.5])
        normal = np.array([0, 0, 1])
        plane = Plane(point, normal)
        
        # Perform intersection
        above_mesh, below_mesh = intersect_with_plane(self.mesh, plane)
        
        # Verify results
        # The horizontal plane cuts the cube in half.
        # z = 0, 1 for below and 2,3,4 for above
        self.assertEqual(above_mesh.num_vertices, 5*5*3) 
        self.assertEqual(below_mesh.num_vertices, 5*5*2)

        # Check the number of domains
        self.assertEqual(above_mesh.num_domains, 4*4*2)
        self.assertEqual(below_mesh.num_domains, 4*4*1)

        # Verify z-coordinates
        for v in above_mesh.vertices:
            self.assertGreaterEqual(v[2], 1.5 - 1e-10)
            
        for v in below_mesh.vertices:
            self.assertLessEqual(v[2], 1.5 + 1e-10)
            
    def test_no_intersection(self):
        point = np.array([0, 0, 5])  # Plane above the cube
        normal = np.array([0, 0, 1])
        plane = Plane(point, normal)
        
        # Perform intersection
        above_mesh, below_mesh = intersect_with_plane(self.mesh, plane)
        
        # Verify results
        self.assertEqual(above_mesh.num_vertices, 0)
        self.assertEqual(below_mesh.num_vertices, self.mesh.num_vertices)
        self.assertEqual(below_mesh.num_edges, self.mesh.num_edges)
        self.assertEqual(below_mesh.num_faces, self.mesh.num_faces)
        self.assertEqual(below_mesh.num_domains, self.mesh.num_domains)

if __name__ == '__main__':
    unittest.main()