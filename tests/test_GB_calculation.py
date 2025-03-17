import logging
import sys
import numpy as np
from scipy.sparse import csr_matrix
import unittest
from mesh.ebsd3d import EBSD3D
from mesh.voronoi3d import create_voronoi_mesh

# Test class to group EBSD grain boundary calculations tests
class TestEBSDGrainBoundaries(unittest.TestCase):
    """Test suite for EBSD grain boundary calculations."""
    def setUp(self):
        """Set up the test fixture with common data."""

        # Initialize logging
        self.logger = logging.getLogger('TestEBSDGrainBoundaries')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)




        # Create points for Voronoi tessellation
        points = np.random.rand(100, 3)
        # Create Voronoi mesh from points
        mesh = create_voronoi_mesh(points)
        # Create Euler angles for the domains
        euler_angles = np.random.rand(100, 3) * np.pi
        # Create the EBSD3D object using the Voronoi mesh and our euler angles
        self.ebsd = EBSD3D(
            vertices=mesh.vertices,
            T_VE=mesh.T_VE,
            T_EF=mesh.T_EF,
            T_FD=mesh.T_FD,
            euler_angles=euler_angles
        )
    
    def test_calculate_GBs(self):
        """Test the grain boundary calculation function correctly identifies boundaries based on misorientation."""
        # Create points for Voronoi tessellation
        # Using points that will create adjacent cells (domains)
        points = np.random.rand(100, 3)
        
        # Create Voronoi mesh from points
        mesh = create_voronoi_mesh(points)
        
        # Create Euler angles for the domains
        euler_angles = np.random.rand(100, 3) * np.pi
        
        # Create the EBSD3D object using the Voronoi mesh and our euler angles
        ebsd = EBSD3D(
            vertices=mesh.vertices,
            T_VE=mesh.T_VE,
            T_EF=mesh.T_EF,
            T_FD=mesh.T_FD,
            euler_angles=euler_angles
        )
        
        # Calculate grain boundaries with a 5-degree tolerance
        gb_ids = ebsd._calculate_GBs(tol=5)
        
        # Since domains have different orientations with some above and some below
        # our tolerance, we should have at least one grain boundary
        self.assertTrue(len(gb_ids) > 0, "Should identify at least one grain boundary")
        
        # Get count of boundaries that were detected
        num_boundaries_detected = len(gb_ids)

        
        # Test with a very large tolerance
        gb_ids_very_large_tol = ebsd._calculate_GBs(tol=181)
        
        # With 50-degree tolerance, no edges should be grain boundaries
        self.assertFalse(np.any(gb_ids_very_large_tol), 
                        "No boundaries should be detected with 181-degree tolerance")
    
    def test_calculate_grains(self):
        """Test the grain calculation function correctly identifies grains based on misorientation."""
        # Create points for Voronoi tessellation
        # Using points that will create adjacent cells (domains)
        points = np.random.rand(100, 3)
        
        # Create Voronoi mesh from points
        mesh = create_voronoi_mesh(points)
        
        # Create Euler angles for the domains
        euler_angles = np.random.rand(100, 3) * 0.08*np.pi
        
        # Create the EBSD3D object using the Voronoi mesh and our euler angles
        ebsd = EBSD3D(
            vertices=mesh.vertices,
            T_VE=mesh.T_VE,
            T_EF=mesh.T_EF,
            T_FD=mesh.T_FD,
            euler_angles=euler_angles
        )
        
        # Calculate grains with a 5-degree tolerance
        T_DG = ebsd._find_grains()
        
        # Since domains have different orientations with some above and some below
        # our tolerance, we should have at least one grain
        self.assertTrue(T_DG.shape[1] > 0, "Should identify at least one grain")

        self.logger.info(f"Number of grains: {T_DG.shape[1]}")
        for i in range(T_DG.shape[1]):
            self.logger.info(f"Grain {i} has {np.sum(T_DG[:,i])} domains")
        

    def test_grain_triangulation(self):
        """Test the grain triangulation function correctly identifies grain boundaries."""
        
        tri_mesh = self.ebsd._triangulate_grain(0)

        self.logger.info(f"Number of triangles: {len(tri_mesh)}")


if __name__ == "__main__":
    unittest.main()
