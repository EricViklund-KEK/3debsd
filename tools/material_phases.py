

import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes
from scipy.signal import find_peaks


def interface_from_grid(grid_3d, grid_spacing: tuple[float, float, float], level, pad_width: int = 10) -> pv.PolyData:

    verts, faces, normals, values = marching_cubes(grid_3d, level, spacing=grid_spacing)
    verts = verts - np.array(pad_width) * grid_spacing

    interface_mesh = pv.PolyData.from_regular_faces(verts, faces)
    return interface_mesh


def calibrate_raw_count(count: np.ndarray, reference_count: np.ndarray, bins: int = 100) -> np.ndarray:

    counts, bin_edges = np.histogram(count / (reference_count + 1e-6), bins=bins)
    peaks, _ = find_peaks(counts, height=4000)
    background_peak = bin_edges[peaks][0]
    upper_peak = bin_edges[peaks][-1]

    calibrated_count = (count / (reference_count + 1e-6) - background_peak) / (upper_peak - background_peak)
    return calibrated_count