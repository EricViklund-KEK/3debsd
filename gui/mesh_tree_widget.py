from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem
from PySide6.QtCore import Signal

class MeshTreeWidget(QTreeWidget):
    selectionChanged = Signal(list)
    
    def __init__(self):
        super().__init__()
        self.setHeaderLabel("EBSD Data")
        self.itemSelectionChanged.connect(self._on_selection_changed)
        
    def update_mesh(self, ebsd_mesh):
        """Update the tree with mesh information"""
        self.clear()
        
        # Create root item
        root = QTreeWidgetItem(self, ["EBSD Mesh"])
        self.addTopLevelItem(root)
        
        # Add grain boundaries item
        gb_item = QTreeWidgetItem(root, ["Grain Boundaries"])
        
        # Add information about grains if available
        if hasattr(ebsd_mesh, "T_DG"):
            num_grains = ebsd_mesh.T_DG.shape[1]
            grains_item = QTreeWidgetItem(root, [f"Grains ({num_grains})"])
            
            # Add individual grains
            for i in range(min(num_grains, 20)):  # Limit to first 20 for performance
                grain_item = QTreeWidgetItem(grains_item, [f"Grain {i+1}"])
        
        # Expand root
        root.setExpanded(True)
    
    def _on_selection_changed(self):
        """Handle selection changes in the tree widget"""
        selected_items = []
        for item in self.selectedItems():
            selected_items.append(item.text(0))
        
        self.selectionChanged.emit(selected_items)
