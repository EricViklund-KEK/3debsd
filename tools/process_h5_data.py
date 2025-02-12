import h5py
import numpy as np
import os
from skimage.registration import phase_cross_correlation
from pathlib import Path

def load_and_process_h5_files(data_folder, output_folder):
    """
    Load h5 files, process the data and save as numpy arrays
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get sorted list of h5 files
    h5_files = os.listdir(data_folder)
    sorted_files = sorted(h5_files, key=lambda x: int(x.split(' ')[4]))
    
    # Get dimensions from first file
    with h5py.File(os.path.join(data_folder, sorted_files[0]), 'r') as f:
        image_resolution = (f['1/EDS/Header/X Cells'][0], 
                          f['1/EDS/Header/Y Cells'][0])
        
    # Define crop region
    crop = slice(15, 65), slice(15, 105)
    
    # Initialize data cubes
    shape = (crop[0].stop - crop[0].start, 
            crop[1].stop - crop[1].start, 
            len(sorted_files))
    Sn_cube = np.zeros(shape)
    Nb_cube = np.zeros(shape)
    euler_cube = np.zeros(shape + (3,))
    phase_cube = np.zeros(shape, dtype=np.int32)
    mad_cube = np.zeros(shape)
    
    
    # Get reference image for alignment
    with h5py.File(os.path.join(data_folder, sorted_files[0]), 'r') as f:
        reference_image = np.array(f['1/EDS/Data/Window Integral/Sn Lα1']).reshape(image_resolution[1], 
                                                                                  image_resolution[0])
    
    # Process each file
    for i, file in enumerate(sorted_files):
        with h5py.File(os.path.join(data_folder, file), 'r') as f:
            # Load data
            euler_data = np.array(f['1/EBSD/Data/Euler']).reshape(image_resolution[1], 
                                                                 image_resolution[0], 3)
            Sn_data = np.array(f['1/EDS/Data/Window Integral/Sn Lα1']).reshape(image_resolution[1], 
                                                                              image_resolution[0])
            Nb_data = np.array(f['1/EDS/Data/Window Integral/Nb Lα1']).reshape(image_resolution[1], 
                                                                              image_resolution[0])
            phase_data = np.array(f['1/EBSD/Data/Phase']).reshape(image_resolution[1],
                                                                 image_resolution[0])
            mad_data = np.array(f['1/EBSD/Data/Mean Angular Deviation']).reshape(image_resolution[1],
                                                                               image_resolution[0])
            
            # Calculate shift using phase cross correlation
            shift, _, _ = phase_cross_correlation(reference_image, Sn_data, normalization=None)
            shift = shift.astype('int')
            
            # Apply shift to all data
            euler_data = np.roll(euler_data, shift, axis=(0,1))
            Sn_data = np.roll(Sn_data, shift, axis=(0,1))
            Nb_data = np.roll(Nb_data, shift, axis=(0,1))
            phase_data = np.roll(phase_data, shift, axis=(0,1))
            mad_data = np.roll(mad_data, shift, axis=(0,1))
            
            # Crop data
            euler_data = euler_data[crop]
            Sn_data = Sn_data[crop]
            Nb_data = Nb_data[crop]
            phase_data = phase_data[crop]
            mad_data = mad_data[crop]
            
            # Scale Euler angles
            euler_data[:,:,0] = euler_data[:,:,0]/(2*np.pi)
            euler_data[:,:,1] = euler_data[:,:,1]/np.pi
            euler_data[:,:,2] = euler_data[:,:,2]/np.pi
            
            # Store in data cubes
            Sn_cube[:,:,i] = Sn_data
            Nb_cube[:,:,i] = Nb_data
            euler_cube[:,:,i,:] = euler_data
            phase_cube[:,:,i] = phase_data
            mad_cube[:,:,i] = mad_data
    
    # Create coordinate grid (in μm)
    ind = np.indices(Sn_cube.shape)
    coord_cube = np.stack((ind[0]*0.1, ind[1]*0.1, ind[2]*0.1), axis=-1)
    
    # Save processed data
    np.save(os.path.join(output_folder, 'coord_cube.npy'), coord_cube)
    np.save(os.path.join(output_folder, 'Sn_cube.npy'), Sn_cube)
    np.save(os.path.join(output_folder, 'Nb_cube.npy'), Nb_cube)
    np.save(os.path.join(output_folder, 'euler_cube.npy'), euler_cube)
    np.save(os.path.join(output_folder, 'phase_cube.npy'), phase_cube)
    np.save(os.path.join(output_folder, 'mad_cube.npy'), mad_cube)
    
    # Create and save flattened versions
    points = coord_cube.reshape(-1, 3)  # Reshape to (n_points, 3)
    Sn_flat = Sn_cube.flatten()  # Reshape to (n_points,)
    Nb_flat = Nb_cube.flatten()  # Reshape to (n_points,)
    euler_flat = euler_cube.reshape(-1, 3)  # Reshape to (n_points, 3)
    phase_flat = phase_cube.flatten()  # Reshape to (n_points,)
    mad_flat = mad_cube.flatten()  # Reshape to (n_points,)
    
    # Save flattened arrays
    np.save(os.path.join(output_folder, 'points.npy'), points)
    np.save(os.path.join(output_folder, 'Sn_flat.npy'), Sn_flat)
    np.save(os.path.join(output_folder, 'Nb_flat.npy'), Nb_flat)
    np.save(os.path.join(output_folder, 'euler_flat.npy'), euler_flat)
    np.save(os.path.join(output_folder, 'phase_flat.npy'), phase_flat)
    np.save(os.path.join(output_folder, 'mad_flat.npy'), mad_flat)
    
    return coord_cube, Sn_cube, Nb_cube, euler_cube, phase_cube, mad_cube

if __name__ == "__main__":
    # Define input/output paths
    data_folder = "./data/8-5-24/h5oina/"
    output_folder = "./output"
    
    # Process the data
    coord_cube, Sn_cube, Nb_cube, euler_cube, phase_cube, mad_cube = load_and_process_h5_files(data_folder, output_folder)
    print(f"Processing complete. Data saved to {output_folder}")
    print(f"Data shape: {Sn_cube.shape}")
    print(f"Flattened data shape: {(Sn_cube.size, )}") 