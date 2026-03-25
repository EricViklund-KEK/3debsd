This repository contains a Python implementation of the MTEX[^1] grain boundary reconstruction algorithm, a three dimensional EBSD/EDS dataset of a Nb3Sn thin film on Nb substrate, and visualizations of this dataset created using the algorithm. This repository exists alongside our paper "Microstructural Characterization of Nb3Sn Thin Films Using FIB Tomography" [^2] to share the experimental data presented and the software used to analyze it. We hope that sharing this code will encourage wider adoption of three dimensional EBSD in the scientific community.

# Running the Code

Clone this repository and install dependencies using:

```bash
pip install -r requirements.txt
```

The data is stored in a h5 file and needs to be converted to Numpy arrays before it can be displayed. We do not store these intermediary data arrays in the repository. To generate the numpy arrays, run the command:

```bash
python3 ./tools/process_h5_data.py
```

once. 

Once the data has been converted to Numpy arrays you do not need to do it again unless the raw data changes.

Open any of the interactive Python notebooks and run all cells. For example:

```bash
example.ipynb
```

# Using the Algorithm in Your Own Project

The algorithm itself is implemented using only scipy and numpy as dependencies. A plotting utility is also included which requires the pyvista library to run. The algorithm exists in the Mesh3D class. It takes three arguments:

* A point array of shape [3, N] where N is the number of data points in the data set
* A Python dictionary containing two arrays:

  * "euler", a [3, N] array of the euler angles measured at each data point
  * "phase", a [1, N] array of integers where 0 corresponds to unidentified points, 1 and 2 are material phases, and 3 is a point in vacuum (4+ can be used for additional phases)
* A [3, 2] array of lower and upper bounds on the x, y, and z axes used to clip the grain boundaries

Once the mesh is constructed, the plot_grain method can be used to generate pyvista meshes for each reconstructed grain by passing an integer id.

# Figures

These are some animated figures which could not be included in the paper.

<figure>
    <video src="./movie_4.mp4" width="100%" controls>
        <p>Your browser doesn't support HTML5 video. You can download the animated figures directly from this repository.</p>
    </video>
    <figcaption>
        A horizontal slice of a Nb3Sn thin film on a Nb substrate. The slice plane is colored based on the Sn concentration.
    </figcaption>
</figure>

# References

[^1]: F. Bachmann, R. Hielscher, and H. Schaeben, “Grain detection from 2d and 3d EBSD data—Specification of the MTEX algorithm,” Ultramicroscopy, vol. 111, no. 12, pp. 1720–1733, Dec. 2011, doi: 10.1016/j.ultramic.2011.08.002.

[^2]: E. Viklund, D. N. Seidman, and S. Posen, “Microstructural Characterization of Nb3Sn Thin Films Using FIB Tomography,” Mar. 11, 2026, arXiv: arXiv:2603.10472. doi: 10.48550/arXiv.2603.10472.
