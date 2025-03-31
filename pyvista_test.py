import logging
import pyvista as pv
import numpy as np

from mesh.ebsd3d import EBSD3D
from mesh.voronoi3d import create_voronoi_mesh, create_bounded_voronoi_mesh

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)





def load_data(output_folder):
    """Load processed EBSD data from numpy arrays."""
    logging.info(f"Loading data from {output_folder}")
    points = np.load(f'{output_folder}/points.npy')
    euler_angles = np.load(f'{output_folder}/euler_flat.npy')
    phase_ids = np.load(f'{output_folder}/phase_flat.npy')
    mad_values = np.load(f'{output_folder}/mad_flat.npy')  # confidence measure
    
    logging.info(f"Loaded {len(points)} data points")
    return points, euler_angles, phase_ids, mad_values

def create_ebsd_mesh(points, euler_angles, phase_ids, mad_values, voronoi_bounds=None):
    """Create EBSD3D object from data using Voronoi tessellation."""
    logging.info("Creating Voronoi mesh")

    # Create base mesh using Voronoi tessellation
    mesh = create_voronoi_mesh(points)
    
    logging.info(f"Created mesh with {len(mesh.vertices)} vertices, {mesh.T_VE.shape[1]} edges, " 
                f"{mesh.T_EF.shape[0]} faces, and {mesh.T_FD.shape[1]} domains")
    
    # Create EBSD3D object with crystallographic data
    logging.info("Creating EBSD3D object")
    ebsd_mesh = EBSD3D(
        vertices=mesh.vertices,
        T_VE=mesh.T_VE,
        T_EF=mesh.T_EF,
        T_FD=mesh.T_FD,
        euler_angles=euler_angles,
        phase_ids=phase_ids,
        confidence_indices=mad_values
    )
    
    # logging.info(f"Created EBSD mesh with {ebsd_mesh.T_DG.shape[1]} grains")
    
    return ebsd_mesh

def main():


    # Set the output folder path where numpy arrays are stored
    output_folder = "output"  # Adjust this path as needed
    
    # Load processed data
    points, euler_angles, phase_ids, mad_values = load_data(output_folder)

    # Subsample points for performance but ensure we have enough for visualization
    logging.info(f"Original data size: {len(points)} points")
    
    # Use smaller subsampling value for testing
    # subsample_rate = 50  # Use a smaller value like 100 or 200 for testing
    # logging.info(f"Subsampling points (1/{subsample_rate})")
    
    np.random.seed(0)
    subsample = np.random.randint(0,points.shape[0],size=10000)

    # points = points[::subsample_rate]
    # euler_angles = euler_angles[::subsample_rate]
    # phase_ids = phase_ids[::subsample_rate]
    # mad_values = mad_values[::subsample_rate]
    
    points = points[subsample]
    euler_angles = euler_angles[subsample]
    phase_ids = phase_ids[subsample]
    mad_values = mad_values[subsample]

    logging.info(f"Subsampled to {len(points)} points")
    
    # Calculate bounds as min/max in each dimension
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    bounds = (min_bounds, max_bounds)
    logging.info(f"Mesh bounds: min={min_bounds}, max={max_bounds}")

    # Create EBSD mesh
    ebsd_mesh = create_ebsd_mesh(points, euler_angles, phase_ids, mad_values, voronoi_bounds=bounds)   
    
    # if ebsd_mesh.T_DG.shape[1] == 0:
    #     logging.error("No grains found in the mesh! Check the data and parameters.")
    #     return
    
    T_VF = ebsd_mesh.T_VE @ ebsd_mesh.T_EF
    faces = []
    for face in range(T_VF.shape[1]):
        verts = T_VF[:, face].nonzero()[0]
        faces.append(verts)
        
        

    plot_faces = []
    for face in faces:
        face_verts = ebsd_mesh.vertices[face]
        if (face_verts < min_bounds).any() or (face_verts > max_bounds).any():
            logger.debug(f"Face {face_verts} is outside bounds")
        else:
            plot_faces.append(face)

    unique_indices = set()
    for face in plot_faces:
        for vert in face:
            unique_indices.add(vert)

    new_indices = {old: new for new, old in enumerate(unique_indices)}
    new_faces = []
    for face in plot_faces:
        new_faces.append([new_indices[old] for old in face])

    vertices = ebsd_mesh.vertices[list(unique_indices)]


    # plot_mesh = pv.PolyData.from_regular_faces(ebsd_mesh.vertices, [plot_faces[0]])
    plot_mesh = pv.PolyData.from_irregular_faces(vertices, new_faces)
    plot_mesh.plot(cpos='xy', show_edges=True)

if __name__ == "__main__":
    main()