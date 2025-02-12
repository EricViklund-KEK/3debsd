from PySide6.QtWidgets import QWidget, QVBoxLayout
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.all as vtk
import numpy as np
import logging
from vtk.util import numpy_support

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
        self.renderer = vtk.vtkRenderer()
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
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        self.renderer.AddActor(axes)
    
    def clear_plot(self):
        """Clear the current plot"""
        self.renderer.RemoveAllViewProps()
        self.add_axes()  # Re-add the axes
        logging.debug("Plot cleared")
    
    def plot_ebsd_mesh(self, ebsd_mesh):
        """Store the mesh for later plotting"""
        self._current_mesh = ebsd_mesh
        self.clear_plot()
    
    def update_plot(self, selected_domains, selected_faces, selected_edges, selected_vertices):
        """Update the plot based on selected components"""
        if not self._current_mesh:
            return
            
        try:
            self.clear_plot()
            logging.info("Updating plot with selected components")
            
            # Collect all vertices to be displayed
            display_vertices = set()
            
            # Add selected vertices
            if selected_vertices:
                display_vertices.update(selected_vertices)
            
            # Add vertices from selected faces
            if selected_faces:
                for face_idx in selected_faces:
                    face_edges = self._current_mesh.T_EF[:, face_idx].nonzero()[0]
                    for edge_idx in face_edges:
                        edge_vertices = self._current_mesh.T_VE[:, edge_idx].nonzero()[0]
                        display_vertices.update(edge_vertices)
            
            # If no vertices selected, use all except infinity point
            if not display_vertices:
                display_vertices = set(range(len(self._current_mesh.vertices) - 1))
            
            # Remove infinity point if present
            display_vertices.discard(len(self._current_mesh.vertices) - 1)
            
            # Convert to list and get coordinates
            display_vertices = list(display_vertices)
            vertex_coords = self._current_mesh.vertices[display_vertices]
            
            # Create points
            points = vtk.vtkPoints()
            points.SetData(numpy_support.numpy_to_vtk(vertex_coords))
            
            # Create polydata
            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(points)
            
            if selected_faces:
                # Create cells only for selected faces
                cells = vtk.vtkCellArray()
                id_list = []
                
                for face_idx in selected_faces:
                    # Get vertices for this face
                    face_edges = self._current_mesh.T_EF[:, face_idx].nonzero()[0]
                    face_vertices = set()
                    for edge_idx in face_edges:
                        edge_vertices = self._current_mesh.T_VE[:, edge_idx].nonzero()[0]
                        face_vertices.update(edge_vertices)
                    
                    # Skip faces connected to infinity point
                    if len(self._current_mesh.vertices) - 1 not in face_vertices:
                        # Convert global vertex indices to local indices in display_vertices
                        local_vertices = [display_vertices.index(v) for v in face_vertices]
                        id_list.append(len(local_vertices))
                        id_list.extend(local_vertices)
                
                if id_list:
                    cells.SetCells(len(selected_faces), 
                                 numpy_support.numpy_to_vtkIdTypeArray(np.array(id_list)))
                    poly_data.SetPolys(cells)
            
            # Create mesh mapper and actor
            mesh_mapper = vtk.vtkPolyDataMapper()
            mesh_mapper.SetInputData(poly_data)
            
            mesh_actor = vtk.vtkActor()
            mesh_actor.SetMapper(mesh_mapper)
            mesh_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
            # mesh_actor.GetProperty().SetOpacity(0.7)
            mesh_actor.GetProperty().SetEdgeVisibility(True)
            mesh_actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
            
            # Add actor to renderer
            self.renderer.AddActor(mesh_actor)
            
            # Center camera on selected components
            if len(vertex_coords) > 0:
                # Calculate bounding box
                bounds = np.array([
                    np.min(vertex_coords[:, 0]), np.max(vertex_coords[:, 0]),
                    np.min(vertex_coords[:, 1]), np.max(vertex_coords[:, 1]),
                    np.min(vertex_coords[:, 2]), np.max(vertex_coords[:, 2])
                ])
                
                # Calculate center and size of bounding box
                center = np.array([
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    (bounds[4] + bounds[5]) / 2
                ])
                
                size = np.array([
                    bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    bounds[5] - bounds[4]
                ])
                
                # Set camera position
                max_size = np.max(size)
                camera = self.renderer.GetActiveCamera()
                camera.SetFocalPoint(center[0], center[1], center[2])
                camera.SetPosition(center[0], center[1], center[2] + max_size * 2)
                camera.SetViewUp(0, 1, 0)
                
                self.renderer.ResetCameraClippingRange()
            
            # Render
            self.vtkWidget.GetRenderWindow().Render()
            
            logging.info("Plot update completed")
            
        except Exception as e:
            logging.error(f"Error updating plot: {str(e)}", exc_info=True)
            raise
    
    def closeEvent(self, event):
        """Clean up VTK objects when widget is closed"""
        self.vtkWidget.Finalize()
        super().closeEvent(event) 