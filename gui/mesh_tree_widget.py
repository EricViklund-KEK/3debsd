from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QMenu
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, Signal
import numpy as np

class MeshTreeWidget(QTreeWidget):
    # Signal emitted when selection changes with lists of selected components
    selectionChanged = Signal(list, list, list, list)  # domains, faces, edges, vertices
    grainBoundariesToggled = Signal(bool)  # Signal for toggling grain boundaries
    
    def __init__(self):
        super().__init__()
        self.setHeaderLabel("Scene Hierarchy")
        self.setMinimumWidth(250)
        self.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        
        # Cache for lazy loading
        self._ebsd_mesh = None
        self._connectivity = None
        
        # Connect signals
        self.itemExpanded.connect(self._on_item_expanded)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.customContextMenuRequested.connect(self._show_context_menu)
    
    def _precompute_connectivity(self, ebsd_mesh):
        """Precompute all connectivity relationships"""
        T_FD = ebsd_mesh.T_FD.tocoo()
        T_EF = ebsd_mesh.T_EF.tocoo()
        T_VE = ebsd_mesh.T_VE.tocoo()

        # Convert sparse matrices to lists of indices for faster lookup
        domain_to_faces = [T_FD.row[T_FD.col == i] for i in range(T_FD.shape[1])]
        face_to_edges = [T_EF.row[T_EF.col == i] for i in range(T_EF.shape[1])]
        edge_to_vertices = [T_VE.row[T_VE.col == i] for i in range(T_VE.shape[1])]
        
        return {
            'domain_to_faces': domain_to_faces,
            'face_to_edges': face_to_edges,
            'edge_to_vertices': edge_to_vertices
        }
    
    def update_mesh(self, ebsd_mesh):
        """Update the tree widget with mesh information"""
        self.clear()
        
        # Store mesh and precompute connectivity
        self._ebsd_mesh = ebsd_mesh
        self._connectivity = self._precompute_connectivity(ebsd_mesh)
        
        # Create root item for the mesh
        mesh_item = QTreeWidgetItem(["EBSD Mesh"])
        self.addTopLevelItem(mesh_item)
        
        # Add visualization options item
        vis_options = QTreeWidgetItem(["Visualization Options"])
        mesh_item.addChild(vis_options)
        
        # Add grain boundaries option
        gb_option = QTreeWidgetItem(["Grain Boundaries (Off)"])
        gb_option.setData(0, Qt.UserRole, ('vis_option', 'grain_boundaries'))
        gb_option.setCheckState(0, Qt.Unchecked)
        vis_options.addChild(gb_option)
        
        # Add domains top level
        num_domains = ebsd_mesh.T_FD.shape[1]
        domains_item = QTreeWidgetItem([f"3D Domains ({num_domains})"])
        mesh_item.addChild(domains_item)
        
        # If grains are available, add grain section
        if hasattr(ebsd_mesh, 'T_DG'):
            num_grains = ebsd_mesh.T_DG.shape[1]
            grains_item = QTreeWidgetItem([f"Grains ({num_grains})"])
            mesh_item.addChild(grains_item)
            
            # Add grain items
            for grain_idx in range(num_grains):
                domains_in_grain = ebsd_mesh.T_DG[:, grain_idx].nonzero()[0]
                grain_item = QTreeWidgetItem([f"Grain {grain_idx} ({len(domains_in_grain)} domains)"])
                grain_item.setData(0, Qt.UserRole, ('grain', grain_idx))
                
                # Add a dummy child to show expansion arrow if grain has domains
                if len(domains_in_grain) > 0:
                    grain_item.addChild(QTreeWidgetItem([""]))
                
                grains_item.addChild(grain_item)
        
        # Add domain items with minimal info - children will be added on expansion
        for domain_idx in range(num_domains):
            domain_info = f"Domain {domain_idx}"
            if ebsd_mesh.euler_angles is not None:
                phi1, Phi, phi2 = np.degrees(ebsd_mesh.euler_angles[domain_idx])
                domain_info += f" - Euler angles: ({phi1:.1f}°, {Phi:.1f}°, {phi2:.1f}°)"
            if ebsd_mesh.phase_ids is not None:
                domain_info += f" - Phase: {ebsd_mesh.phase_ids[domain_idx]}"
            
            domain_item = QTreeWidgetItem([domain_info])
            domain_item.setData(0, Qt.UserRole, ('domain', domain_idx))
            
            # Add a dummy child to show expansion arrow
            if len(self._connectivity['domain_to_faces'][domain_idx]) > 0:
                domain_item.addChild(QTreeWidgetItem([""]))
            
            domains_item.addChild(domain_item)
        
        # Expand the mesh item by default
        mesh_item.setExpanded(True)
        vis_options.setExpanded(True)
        
        # Connect to item changed signal to handle checkboxes
        self.itemChanged.connect(self._on_item_changed)
    
    def _on_item_changed(self, item, column):
        """Handle checkbox state changes"""
        data = item.data(0, Qt.UserRole)
        if not data:
            return
            
        item_type, option = data
        
        if item_type == 'vis_option' and option == 'grain_boundaries':
            show_gb = item.checkState(0) == Qt.Checked
            item.setText(0, f"Grain Boundaries ({'On' if show_gb else 'Off'})")
            self.grainBoundariesToggled.emit(show_gb)
    
    def _on_item_expanded(self, item):
        """Lazy load children when item is expanded"""
        # Get item type and index
        data = item.data(0, Qt.UserRole)
        if not data:
            return
        
        item_type, idx = data
        
        # Remove dummy child if it exists
        if item.childCount() == 1 and item.child(0).text(0) == "":
            item.removeChild(item.child(0))
        
        # Add children based on item type
        if item_type == 'grain':
            domains = self._ebsd_mesh.T_DG[:, idx].nonzero()[0]
            domains_item = QTreeWidgetItem([f"Domains ({len(domains)})"])
            item.addChild(domains_item)
            
            for domain_idx in domains:
                domain_info = f"Domain {domain_idx}"
                if self._ebsd_mesh.euler_angles is not None:
                    phi1, Phi, phi2 = np.degrees(self._ebsd_mesh.euler_angles[domain_idx])
                    domain_info += f" - Euler angles: ({phi1:.1f}°, {Phi:.1f}°, {phi2:.1f}°)"
                
                domain_item = QTreeWidgetItem([domain_info])
                domain_item.setData(0, Qt.UserRole, ('domain', domain_idx))
                
                if len(self._connectivity['domain_to_faces'][domain_idx]) > 0:
                    domain_item.addChild(QTreeWidgetItem([""]))
                
                domains_item.addChild(domain_item)
                
        elif item_type == 'domain':
            faces = self._connectivity['domain_to_faces'][idx]
            faces_item = QTreeWidgetItem([f"Faces ({len(faces)})"])
            item.addChild(faces_item)
            
            for face_idx in faces:
                face_item = QTreeWidgetItem([f"Face {face_idx}"])
                face_item.setData(0, Qt.UserRole, ('face', face_idx))
                if len(self._connectivity['face_to_edges'][face_idx]) > 0:
                    face_item.addChild(QTreeWidgetItem([""]))
                faces_item.addChild(face_item)
                
        elif item_type == 'face':
            edges = self._connectivity['face_to_edges'][idx]
            edges_item = QTreeWidgetItem([f"Edges ({len(edges)})"])
            item.addChild(edges_item)
            
            for edge_idx in edges:
                edge_item = QTreeWidgetItem([f"Edge {edge_idx}"])
                edge_item.setData(0, Qt.UserRole, ('edge', edge_idx))
                if len(self._connectivity['edge_to_vertices'][edge_idx]) > 0:
                    edge_item.addChild(QTreeWidgetItem([""]))
                edges_item.addChild(edge_item)
                
        elif item_type == 'edge':
            vertices = self._connectivity['edge_to_vertices'][idx]
            vertices_item = QTreeWidgetItem([f"Vertices ({len(vertices)})"])
            item.addChild(vertices_item)
            
            for vertex_idx in vertices:
                vertex = self._ebsd_mesh.vertices[vertex_idx]
                vertex_info = f"Vertex {vertex_idx}: ({vertex[0]:.1f}, {vertex[1]:.1f}, {vertex[2]:.1f})"
                vertex_item = QTreeWidgetItem([vertex_info])
                vertices_item.addChild(vertex_item)
    
    def _show_context_menu(self, position):
        """Show context menu for tree items"""
        item = self.itemAt(position)
        if not item:
            return
            
        data = item.data(0, Qt.UserRole)
        if not data:
            return
            
        item_type, idx = data
        
        menu = QMenu()
        
        if item_type == 'grain':
            # Add action to visualize grain boundaries
            action = QAction("Visualize Grain Boundaries", self)
            action.triggered.connect(lambda: self._visualize_grain_boundaries(idx))
            menu.addAction(action)
        
        if menu.actions():
            menu.exec(self.mapToGlobal(position))
    
    def _visualize_grain_boundaries(self, grain_idx):
        """Visualize boundaries for a specific grain"""
        # Find grain boundary option and check it
        for i in range(self.topLevelItemCount()):
            top_item = self.topLevelItem(i)
            for j in range(top_item.childCount()):
                vis_item = top_item.child(j)
                if vis_item.text(0).startswith("Visualization Options"):
                    for k in range(vis_item.childCount()):
                        gb_item = vis_item.child(k)
                        data = gb_item.data(0, Qt.UserRole)
                        if data and data[0] == 'vis_option' and data[1] == 'grain_boundaries':
                            gb_item.setCheckState(0, Qt.Checked)
                            break
        
        # Select the grain to visualize its boundaries
        self.clearSelection()
        self._on_selection_changed()  # Update plot with grain boundaries enabled
    
    def _on_selection_changed(self):
        """Handle selection changes and emit signal with selected components"""
        selected_domains = set()
        selected_faces = set()
        selected_edges = set()
        selected_vertices = set()
        
        # Process all selected items
        for item in self.selectedItems():
            data = item.data(0, Qt.UserRole)
            if not data:
                continue
                
            item_type, idx = data
            
            # Add directly selected component
            if item_type == 'domain':
                selected_domains.add(idx)
                # Add all child faces
                selected_faces.update(self._connectivity['domain_to_faces'][idx])
            elif item_type == 'face':
                selected_faces.add(idx)
                # Add all child edges
                selected_edges.update(self._connectivity['face_to_edges'][idx])
            elif item_type == 'edge':
                selected_edges.add(idx)
                # Add all child vertices
                selected_vertices.update(self._connectivity['edge_to_vertices'][idx])
            elif item_type == 'vertex':
                selected_vertices.add(idx)
        
        # Emit signal with selected components
        self.selectionChanged.emit(
            list(selected_domains),
            list(selected_faces),
            list(selected_edges),
            list(selected_vertices)
        )