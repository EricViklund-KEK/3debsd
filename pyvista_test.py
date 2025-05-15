import logging
import pyvista as pv
import numpy as np
from scipy.sparse.csgraph import breadth_first_tree
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from mesh.ebsd3d import EBSD3D
from mesh.voronoi3d import create_voronoi_mesh, create_bounded_voronoi_mesh
from mesh.mesh3d import Mesh3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)





def load_data(output_folder):
    """Load processed EBSD data from numpy arrays."""
    logging.info(f"Loading data from {output_folder}")
    points = np.load(f'{output_folder}/points.npy')
    euler_angles = np.load(f'{output_folder}/euler_flat.npy')
    phase_ids = np.load(f'{output_folder}/phase_flat.npy')
    mad_values = np.load(f'{output_folder}/mad_flat.npy')  # confidence measure
    
    logging.info(f"Loaded {len(points)} data points")
    return points, euler_angles, phase_ids, mad_values


def unstructured_grid_from_mesh(mesh, bounds):
    """Convert mesh to PyVista UnstructuredGrid."""
    logging.info("Converting mesh to PyVista UnstructuredGrid")
    
    T_VD = mesh.T_VE @ mesh.T_EF @ mesh.T_FD
    cell_types = [pv.CellType.CONVEX_POINT_SET] * mesh.num_domains
    
    used_vertices = set()
    cell_point_ids = []

    for domain in range(mesh.num_domains):
        domain_vertices = T_VD[:,domain].nonzero()[0]
        used_vertices.update(domain_vertices)

        cell_point_ids.append(domain_vertices)


    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_vertices))}

    cell_array = []
    for cell in cell_point_ids:
        new_cell = [index_map[pt] for pt in cell]
        cell_array.append(len(new_cell))
        cell_array.extend(new_cell)

    cells = cell_array
    points = mesh.vertices[sorted(used_vertices)]

    
    try:
        grid = pv.UnstructuredGrid(cells, cell_types, points)
    except Exception as e:
        logging.error(f"Error creating UnstructuredGrid: {e}")
        raise

    if bounds is not None:
        roi = pv.Box(bounds=bounds)
        grid = grid.clip_box([-1,1,-1,1,-1,1], invert=False)

    logging.info(f"UnstructuredGrid created with {grid.n_cells} cells and {grid.n_points} points")

    return grid

def polydata_from_mesh(mesh: Mesh3D, bounds):
    """Convert mesh to PyVista PolyData."""
    logging.info("Converting mesh to PyVista PolyData")

    id = 0
    
    T_VF = mesh.T_VE @ mesh.T_EF

    used_vertices = set()
    for face in mesh.T_FG[:,[id]].tocoo().coords[0]:
        face_vertices = T_VF[:,[face]].nonzero()[0]
        used_vertices.update(face_vertices)

    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_vertices))}

    faces = []
    for face in mesh.T_FG[:,[id]].tocoo().coords[0]:
        face_vertices = T_VF[:,[face]].nonzero()[0]
        new_face = [index_map[pt] for pt in face_vertices]
        faces.append(new_face)

    points = mesh.vertices[sorted(used_vertices)]

    polydata = pv.PolyData.from_irregular_faces(points, faces)

    if bounds is not None:
        roi = pv.Box(bounds=bounds)
        polydata = polydata.clip_box(roi, invert=False)

    return polydata

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
    subsample = np.random.randint(0,points.shape[0],size=50)

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

    point_data = {
        'point_coordinates': points,
        'euler_angles': euler_angles,
    }
    mesh = Mesh3D.from_voronoi_tessellation(point_data)
    logging.info(f"Mesh: {mesh}")

    bounds = np.array([min_bounds[0], max_bounds[0], min_bounds[1], max_bounds[1], min_bounds[2], max_bounds[2]])
    logging.info(f"Bounds for clipping: {bounds}")
    
    polydata = polydata_from_mesh(mesh, bounds)
    logging.info(f"PolyData: {polydata}")

    plotter = pv.Plotter()
    plotter.add_mesh(polydata, show_edges=True, color='red')
    plotter.show()




if __name__ == "__main__":
    main()