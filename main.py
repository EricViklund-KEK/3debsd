import numpy as np
from ebsd3d import EBSD3D
from voronoi3d import create_voronoi_mesh
from PySide6.QtWidgets import QApplication
import sys
import logging
from gui.main_window import MainWindow
import traceback

# Configure logging at the start of the program
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Print to console
        logging.FileHandler('debug.log')    # Also save to file
    ]
)

def exception_hook(exctype, value, tb):
    """Global exception handler to log unhandled exceptions"""
    logging.error("Uncaught exception:", exc_info=(exctype, value, tb))
    sys.__excepthook__(exctype, value, tb)  # Call the default handler as well

sys.excepthook = exception_hook

def load_data(output_folder):
    """Load processed EBSD data from numpy arrays."""
    logging.info(f"Loading data from {output_folder}")
    points = np.load(f'{output_folder}/points.npy')
    euler_angles = np.load(f'{output_folder}/euler_flat.npy')
    phase_ids = np.load(f'{output_folder}/phase_flat.npy')
    mad_values = np.load(f'{output_folder}/mad_flat.npy')  # confidence measure
    
    logging.info(f"Loaded {len(points)} data points")
    return points, euler_angles, phase_ids, mad_values

def create_ebsd_mesh(points, euler_angles, phase_ids, mad_values):
    """Create EBSD3D object from data using Voronoi tessellation."""
    logging.info("Creating Voronoi mesh")
    # Create base mesh using Voronoi tessellation
    mesh = create_voronoi_mesh(points)
    
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
    
    return ebsd_mesh

def main():
    # Create Qt Application
    app = QApplication(sys.argv)
    
    # Create main window
    window = MainWindow()

    # Show the window
    window.show()
    logging.info("GUI application started")    
    
    # Set the output folder path where numpy arrays are stored
    output_folder = "output"  # Adjust this path as needed
    
    # Load processed data
    points, euler_angles, phase_ids, mad_values = load_data(output_folder)

    # Subsample points
    logging.info("Subsampling points (1/100)")
    points = points[::100]
    euler_angles = euler_angles[::100]
    phase_ids = phase_ids[::100]
    mad_values = mad_values[::100]
    
    # Create EBSD mesh
    ebsd_mesh = create_ebsd_mesh(points, euler_angles, phase_ids, mad_values)
    
    # Plot the mesh in the GUI
    window.plot_ebsd_mesh(ebsd_mesh)
    

    
    # Start the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 