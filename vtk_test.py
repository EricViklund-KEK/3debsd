#!/usr/bin/env python3
import logging
import numpy as np
import os
import collections

# Import VTK modules
import vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkUnstructuredGrid,
    vtkCellArray,
    vtkConvexPointSet,
    vtkPlane,
    vtkCellTypes
)
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTableBasedClipDataSet
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor
)

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


def unstructured_grid_from_mesh(mesh):
    """Convert mesh to VTK UnstructuredGrid using direct VTK API."""
    logging.info("Converting mesh to VTK UnstructuredGrid")
    
    # Get vertices connected to each domain
    T_VD = mesh.T_VE @ mesh.T_EF @ mesh.T_FD
    
    # Create a new VTK points object
    points_vtk = vtkPoints()
    used_vertices = set()
    cell_point_ids = []

    # Find vertices for each domain/cell
    for domain in range(mesh.num_domains):
        domain_vertices = T_VD[:,domain].nonzero()[0]
        used_vertices.update(domain_vertices)
        cell_point_ids.append(domain_vertices)

    # Create index mapping from old to new indices
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(used_vertices))}
    
    # Insert points into VTK points object
    points_array = mesh.vertices[sorted(used_vertices)]
    for i, point in enumerate(points_array):
        points_vtk.InsertNextPoint(point[0], point[1], point[2])
    
    # Create VTK unstructured grid
    grid = vtkUnstructuredGrid()
    grid.SetPoints(points_vtk)
    
    # Create cells (domains)
    for cell_points in cell_point_ids:
        # Map old vertex indices to new indices
        new_cell_points = [index_map[pt] for pt in cell_points]
        
        # Create a convex point set for each domain
        convex_point_set = vtkConvexPointSet()
        
        # Add points to the convex point set
        for pt_id in new_cell_points:
            convex_point_set.GetPointIds().InsertNextId(pt_id)
        
        # Add the cell to the grid
        grid.InsertNextCell(convex_point_set.GetCellType(), 
                           convex_point_set.GetPointIds())

    logging.info(f"UnstructuredGrid created with {grid.GetNumberOfCells()} cells and {grid.GetNumberOfPoints()} points")
    logging.debug(f"Grid: {grid}")
    
    return grid


def visualize_with_plane_clipper(grid, clip_normal=[-1.0, -1.0, 1.0]):
    """Visualize the VTK unstructured grid with a plane clipper."""
    logging.info("Setting up visualization with plane clipper")
    
    # Create a clip plane
    logging.info(f"Creating clip plane with normal: {clip_normal}")
    clip_plane = vtkPlane()
    clip_plane.SetOrigin(0.0, 0.0, 0.0)
    clip_plane.SetNormal(-1.0, 0.0, 0.0)
    
    # Create clipper
    logging.info("Creating plane clipper")
    clipper = vtkTableBasedClipDataSet()
    clipper.SetClipFunction(clip_plane)
    clipper.SetInputData(grid)
    clipper.SetValue(0.0)
    clipper.GenerateClippedOutputOn()
    clipper.Update()
    
    # Create colors
    colors = vtk.vtkNamedColors()
    
    # Create renderer and render window
    logging.info("Creating renderer and render window")
    renderer = vtkRenderer()
    renderer.SetBackground(colors.GetColor3d('Wheat'))
    renderer.UseHiddenLineRemovalOn()
    
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1024, 768)
    render_window.SetWindowName("3D EBSD Visualization with Plane Clipper")
    
    # Setup for inside part (yellow)
    logging.info("Setting up inside part visualization")
    inside_mapper = vtkDataSetMapper()
    inside_mapper.SetInputData(clipper.GetOutput())
    inside_mapper.ScalarVisibilityOff()
    
    inside_actor = vtkActor()
    inside_actor.SetMapper(inside_mapper)
    inside_actor.GetProperty().SetDiffuseColor(colors.GetColor3d('Banana'))
    inside_actor.GetProperty().SetAmbient(0.3)
    inside_actor.GetProperty().EdgeVisibilityOn()
    
    # Add actors to renderer
    renderer.AddActor(inside_actor)
    
    # Set up interactor
    logging.info("Setting up interactor")
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Initialize and start the visualization
    logging.info("Initializing and starting the visualization")
    renderer.ResetCamera()
    renderer.GetActiveCamera().Dolly(1.4)
    renderer.ResetCameraClippingRange()
    render_window.Render()
    interactor.Initialize()
    interactor.Start()
    
    # Generate a report on cell types
    logging.info("Generating cell type report")
    num_inside_cells = clipper.GetOutput().GetNumberOfCells()
    logging.info(f"The inside dataset contains {num_inside_cells} cells")
    
    inside_cell_map = {}
    for i in range(num_inside_cells):
        cell_type = clipper.GetOutput().GetCellType(i)
        inside_cell_map[cell_type] = inside_cell_map.get(cell_type, 0) + 1
    
    for k, v in collections.OrderedDict(sorted(inside_cell_map.items())).items():
        logging.info(f"\tCell type {vtkCellTypes.GetClassNameFromTypeId(k)} occurs {v} times")
    
    num_clipped_cells = clipper.GetClippedOutput().GetNumberOfCells()
    logging.info(f"The clipped dataset contains {num_clipped_cells} cells")
    
    clipped_cell_map = {}
    for i in range(num_clipped_cells):
        cell_type = clipper.GetClippedOutput().GetCellType(i)
        clipped_cell_map[cell_type] = clipped_cell_map.get(cell_type, 0) + 1
    
    for k, v in collections.OrderedDict(sorted(clipped_cell_map.items())).items():
        logging.info(f"\tCell type {vtkCellTypes.GetClassNameFromTypeId(k)} occurs {v} times")


def main():
    # Set the output folder path where numpy arrays are stored
    output_folder = "output"  # Adjust this path as needed
    
    # Load processed data
    points, euler_angles, phase_ids, mad_values = load_data(output_folder)

    # Subsample points for performance but ensure we have enough for visualization
    logging.info(f"Original data size: {len(points)} points")
    
    # Sample random points for better performance
    np.random.seed(0)
    subsample = np.random.randint(0, points.shape[0], size=10000)
    
    points = points[subsample]
    euler_angles = euler_angles[subsample]
    phase_ids = phase_ids[subsample]
    mad_values = mad_values[subsample]

    logging.info(f"Subsampled to {len(points)} points")
    
    # Calculate bounds as min/max in each dimension
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    logging.info(f"Mesh bounds: min={min_bounds}, max={max_bounds}")

    # Create mesh using Voronoi tessellation
    mesh = Mesh3D.from_voronoi_tessellation(points)
    logging.info(f"Mesh: {mesh}")
    
    # Convert to VTK unstructured grid without clipping
    grid = unstructured_grid_from_mesh(mesh)
    
    # Define the clip plane normal (can be adjusted as needed)
    clip_normal = [-1.0, -1.0, 1.0]
    
    # Visualize using the plane clipper
    visualize_with_plane_clipper(grid, clip_normal)


if __name__ == "__main__":
    main()