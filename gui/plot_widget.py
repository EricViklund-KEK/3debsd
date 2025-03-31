from PySide6.QtWidgets import QWidget, QVBoxLayout
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.all import vtkRenderer, vtkAxesActor, vtkCellArray, vtkPolyData, vtkPoints, vtkPolyDataMapper, vtkActor
from vtkmodules.all import vtkTriangle, vtkUnsignedCharArray
from vtkmodules.util import numpy_support

import numpy as np
import logging


class EBSDPlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # Store current mesh
        self._current_mesh = None
        
        # Create the layout
        layout = QVBoxLayout(self)
        
        # Create VTK widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtkWidget)
        
        # Create renderer and add it to the widget
        self.renderer = vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        
        # Initialize interactor and set background color
        self.iren.Initialize()
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        
        # Add axes
        self.add_axes()
        
        logging.info("VTK plot widget initialized")
    
    def add_axes(self):
        """Add coordinate axes to the scene"""
        axes = vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        self.renderer.AddActor(axes)
    
    def clear_plot(self):
        """Clear the current plot"""
        self.renderer.RemoveAllViewProps()
        self.add_axes()  # Re-add the axes
        logging.debug("Plot cleared")
    
    def draw_bounding_box(self, bounds):
        """Draw the bounding box based on the provided bounds."""
        # Unpack the bounds tuple into min and max coordinates
        min_coords, max_coords = bounds
        
        # Create points for the corners of the bounding box
        corners = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],  # min x, min y, min z
            [max_coords[0], min_coords[1], min_coords[2]],  # max x, min y, min z
            [max_coords[0], max_coords[1], min_coords[2]],  # max x, max y, min z
            [min_coords[0], max_coords[1], min_coords[2]],  # min x, max y, min z
            [min_coords[0], min_coords[1], max_coords[2]],  # min x, min y, max z
            [max_coords[0], min_coords[1], max_coords[2]],  # max x, min y, max z
            [max_coords[0], max_coords[1], max_coords[2]],  # max x, max y, max z
            [min_coords[0], max_coords[1], max_coords[2]],  # min x, max y, max z
        ])
        
        # Create lines to represent the edges of the bounding box
        lines = vtkCellArray()
        for i in range(4):
            lines.InsertNextCell(2)
            lines.InsertCellPoint(i)       # Bottom face
            lines.InsertCellPoint((i + 1) % 4)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(i + 4)   # Top face
            lines.InsertCellPoint((i + 1) % 4 + 4)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(i)       # Vertical edges
            lines.InsertCellPoint(i + 4)
        
        # Create a polydata object
        poly_data = vtkPolyData()
        points = vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(corners))
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)
        
        # Create a mapper and actor
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        box_actor = vtkActor()
        box_actor.SetMapper(mapper)
        box_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red color for the bounding box
        box_actor.GetProperty().SetLineWidth(2.0)  # Set line width
        
        # Add the bounding box actor to the renderer
        self.renderer.AddActor(box_actor)
    
    def plot_grain_boundaries(self):
        """Plot grain boundaries from the current EBSD mesh."""
        if self._current_mesh is None or not hasattr(self._current_mesh, "T_DG"):
            logging.warning("No EBSD mesh with grain boundaries available")
            return
        
        logging.info("Plotting grain boundaries")
        
        # Create a polydata object for the grain boundaries
        polydata = vtkPolyData()
        points = vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(self._current_mesh.vertices))
        polydata.SetPoints(points)
        
        # Create cells for triangles
        cells = vtkCellArray()
        num_grains = self._current_mesh.T_DG.shape[1]
        
        # Create color array
        # colors = vtkUnsignedCharArray()
        # colors.SetNumberOfComponents(3)
        # # colors.SetName("Colors")
        
        # Track the total number of triangles for color mapping
        total_triangles = 0
        filtered_triangles = 0
        
        # Get bounds if available
        bounds = None
        if hasattr(self, '_bounds') and self._bounds is not None:
            bounds = self._bounds
            min_coords, max_coords = bounds
        
        # For each grain, get its boundary triangles
        for grain_id in range(num_grains):
            triangles = self._current_mesh.GB_mesh(grain_id)
            
            if not triangles:
                continue
                
            # Random color for this grain
            grain_color = np.random.randint(0, 255, size=3)
            
            # Add triangles to cell array
            for triangle in triangles:
                if len(triangle) != 3:
                    logging.warning(f"Expected triangle with 3 vertices, got {len(triangle)}")
                    continue
                
                # Check if triangle is inside bounds
                if bounds is not None:
                    triangle_coords = self._current_mesh.vertices[triangle]
                    
                    # Skip if triangle is outside bounds
                    if (triangle_coords < min_coords).any() or (triangle_coords > max_coords).any():
                        filtered_triangles += 1
                        continue
                
                vtk_triangle = vtkTriangle()
                for i in range(3):
                    vtk_triangle.GetPointIds().SetId(i, triangle[i])
                cells.InsertNextCell(vtk_triangle)
                
                # Add color for this triangle
                # colors.InsertNextTuple3(grain_color[0], grain_color[1], grain_color[2])
                
                total_triangles += 1
        
        polydata.SetPolys(cells)
        # polydata.GetCellData().SetScalars(colors)
        
        # Create mapper and actor
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(2)
        
        # Add actor to renderer
        self.renderer.AddActor(actor)
        self.vtkWidget.GetRenderWindow().Render()
        logging.info(f"Plotted {total_triangles} triangles (filtered {filtered_triangles}) for {num_grains} grain boundaries")
    
    def plot_ebsd_mesh(self, ebsd_mesh, voronoi_bounds=None):
        """Plot the EBSD mesh in 3D."""
        self.clear_plot()
        self._current_mesh = ebsd_mesh
        self._bounds = voronoi_bounds
        
        # Draw bounding box if provided
        if voronoi_bounds is not None:
            self.draw_bounding_box(voronoi_bounds)
        
        # Plot grain boundaries
        self.plot_grain_boundaries()
        
        # Reset camera and render
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
    
    def closeEvent(self, event):
        """Clean up VTK objects when widget is closed"""
        self.vtkWidget.Finalize()
        super().closeEvent(event)