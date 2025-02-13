from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QSplitter
from gui.plot_widget import EBSDPlotWidget
from gui.mesh_tree_widget import MeshTreeWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EBSD 3D Viewer")
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Create splitter for tree and plot
        splitter = QSplitter()
        layout.addWidget(splitter)
        
        # Create tree widget
        self.tree_widget = MeshTreeWidget()
        splitter.addWidget(self.tree_widget)
        
        # Create plot widget
        self.plot_widget = EBSDPlotWidget()
        splitter.addWidget(self.plot_widget)
        
        # Set window size
        self.resize(900, 600)
        
        # Connect tree selection to plot updates
        self.tree_widget.selectionChanged.connect(self.plot_widget.update_plot)
    
    def plot_ebsd_mesh(self, ebsd_mesh, voronoi_bounds=None):
        """Plot the EBSD mesh in the plot widget and update the tree"""
        self.plot_widget.plot_ebsd_mesh(ebsd_mesh, voronoi_bounds=voronoi_bounds)
        self.tree_widget.update_mesh(ebsd_mesh) 