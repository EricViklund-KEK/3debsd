This repository contains a Python implementation of the MTEX[^1] grain boundary reconstruction algorithm, a three dimensional EBSD/EDS dataset of a Nb3Sn thin film on Nb substrate, and visualizations of this dataset created using the algorithm. This repository exists alongside our paper "Microstructural Characterization of Nb3Sn Thin Films Using FIB Tomography" [^2] to share the experimental data presented and the software used to analyze it. We hope that sharing this code will encourage wider adoption of three dimensional EBSD in the scientific community.

# Algorithm

This algorithm is based on the mathematics of discrete geometry and matrix algebra. The geometry of the microstructure is represented using discrete vertices, edges, faces, and volumes. Volumes are bounded by faces, faces by edges, and edges by vertices. This structural relationship is encoded into matrices called incidence matrices, which contain a boolean value for each pair of $n$-polygon and $(n + 1)$-polygon, true if they are connected and false otherwise. I.E. if the $i$-th edge is a boundary of the $j$-th face in the geometry, then the value of $I_{ij}$ is true. We denote the incidence matrix, $\mathbf{I}$, that relates $n$-polygons to $m$-polygons using using the superscripts $n$ and $m$, denoted as $\mathbf{I}^{n,m}$. Since we primarily care about three dimensional geometry, we label $n = 0 \to V$ for vertices, $n = 1 \to E$ for edges, $n = 2 \to F$ for faces, and $n = 3 \to D$ for volumes/domains. Later we will find incidence matrices involving grains, which we can think about as higher dimensional collections of volumes which corresponds to $n = 4 \to G$ for grains.



The adjacency matrix encodes the relationship between all pairs of $n$-polygons and its values are true if two $n$-polygons are connected by an $(n + 1)$-polygon. I.E. if point $i$ and point $j$ are connected by an edge then $A_{ij}$ is true. The adjacency matrix of the $n$-polygons is equal to the matrix product of the corresponding incidence matrix and the transpose of its incidence matrix:
$$\mathbf{A}^n = \mathbf{I}^{n,n+1} (\mathbf{I}^{n,n+1})^T$$
We can also calculate incidence matrices between $n$-polygons and $(n + 2)$-polygons, which represent the $n$-polygons which are contained in the boundary of each $(n + 2)$-polygon, by multiplying the $n$-polygon incidence matrix with the $(n + 1)$-polygon incidence matrix:
$$\mathbf{I}^{n,n+2} = \mathbf{I}^{n,n+1} \mathbf{I}^{n+1,n+2}$$
This can be extended to any $(n + j)$-polygon incidence matrix. For example, to find the incidence matrix that encodes all vertices that make up each volume you would calculate the triple product of the vertex, edge, and face incidence matrices:
$$\mathbf{I}^{VD} = \mathbf{I}^{VE} \mathbf{I}^{EF} \mathbf{I}^{FD}$$
By filtering, slicing, and multiplying different incidence matrices and their transposes it is possible to easily construct complex subsets of the geometry.

To construct this discrete geometry, the first step is to convert the microstructural data into a flat list of points $P = \{p_1, p_2, \dots, p_n\}$, where each point $p_i \in \mathbb{R}^3$ is associated with a position vector $\mathbf{x}_i$, an orientation $g_i \in SO(3)$, and a phase label $\phi_i$. The algorithm does not require a structured grid, which greatly simplifies working with unstructured, incomplete, or missing data. 



The points are triangulated based on their position using **Delaunay triangulation**, $DT(P)$, which is defined such that no point $p_i \in P$ is inside the circumsphere of any tetrahedron in the triangulation. This has the desirable mathematical property of creating edges between every pair of nearest neighbor points. The mathematical dual of this triangulation is known as the **Voronoi diagram**, $\mathcal{V}(P)$, which divides space into regions (cells) $V_i$ defined as:

$$V_i = \{ x \in \mathbb{R}^3 \mid \|x - p_i\| \leq \|x - p_j\| \text{ for all } j \neq i \}$$

These regions are the volumes closest to each point, having boundaries $\partial V_i$ that are guaranteed to be convex polyhedra.

Grain boundaries are approximated by calculating the misorientation angle $\Delta\theta$ between two neighboring points and assigning it to the face which is shared by their corresponding region and then selecting only the faces above a threshold misorientation angle $\theta_{crit}$. By first approximation, if the misorientation of two nearest neighbor points is large then there is a grain boundary exactly equidistant and perpendicular to those two points, which is exactly the boundary between the two Voronoi regions the points occupy. This selection decomposes the adjacency matrix into regions that share a grain boundary face and those that don't.

To reconstruct the grains we represent the microstructure as a graph $G = (V, E)$, where the set of vertices $V$ corresponds to the points of the Delaunay triangulation, and the set of edges $E$ represents the connections between nearest neighbors.

The reconstruction of grains is a partitioning of the graph based on the misorientation criteria. We define a subset of "blocked" edges $E_{GB} \subset E$ such that an edge $e_{ij}$ between points $p_i$ and $p_j$ belongs to $E_{GB}$ if the misorientation angle $\Delta\theta_{ij}$ exceeds the critical threshold $\theta_{crit}$.

The reduced graph $G' = (V, E \setminus E_{GB})$ contains only the edges that do not cross a grain boundary.

A **grain** $\mathcal{G}_k$ is mathematically defined as a **connected component** of the graph $G'$. Two points $p_i$ and $p_j$ belong to the same grain if there exists a path $P = (v_0, v_1, \dots, v_m)$ in $G'$ such that $v_0 = p_i$ and $v_m = p_j$.



We can also define a **sub-grain boundary** at an edge $e_{ij} \in E_{GB}$ if, despite the edge being blocked, the vertices $p_i$ and $p_j$ still belong to the same connected component $\mathcal{G}_k$. Formally:
$$e_{ij} \text{ is a sub-grain boundary if } e_{ij} \in E_{GB} \text{ and } \exists \text{ path } P \subseteq G' \text{ connecting } p_i, p_j$$

A standard breadth-first search (BFS) or depth-first search (DFS) is used to find all maximal connected subgraphs. The result is a partition of the vertices:
$$V = \bigcup_{k=1}^N \mathcal{G}_k, \quad \text{where } \mathcal{G}_a \cap \mathcal{G}_b = \emptyset \text{ for } a \neq b$$

This partitioning is encoded into the **Domain-Grain incidence matrix** $\mathbf{I}^{D,G}$, where each row represents a volume (domain $D_i$ centered at point $p_i$) and each column represents a reconstructed grain $G_j$:

$$\mathbf{I}^{D,G}_{ij} = 
\begin{cases} 
1 & \text{if } p_i \in \mathcal{G}_j \\
0 & \text{otherwise}
\end{cases}$$

<figure>
    <image src="./output/algo_diagram.png" width="100%" controls>
    </image>
    <figcaption>
        Reconstructing grain boundaries from unstructured point measurements of grain orientation using Voronoi decomposition and graph search.
    </figcaption>
</figure>

All of the pieces required to reconstruct the grain boundaries are now present. The last step is to calculate which of the mesh faces correspond to which grain's boundary. We could compose our grain-volume and volume-face incidence matrices to find the faces which belong to each grain, but this would include all faces—both interior and boundary faces—which make up the volume of each grain. 

We only want to select the boundary faces. This selection is accomplished by including the grain boundary incidence matrix in the composition, selecting only the regions which touch a grain boundary. Let $\mathbf{I}^{F,D}$ be the face-domain incidence matrix and $\mathbf{I}^{D,G}$ be the domain-grain incidence matrix. If $\mathbf{A}_{bound}$ represents the adjacency matrix filtered for grain boundaries, the final grain-to-grain boundary face incidence matrix $\mathbf{I}^{F,G}_{bound}$ is calculated as:

$$\mathbf{I}^{F,G}_{bound} = (\mathbf{I}^{F,D} \mathbf{A}_{bound} \mathbf{I}^{D,G}) \odot (\mathbf{I}^{F,D} \mathbf{I}^{D,G})$$

In this expression, $\odot$ denotes the Hadamard (pointwise) product. The first term, $(\mathbf{I}^{F,D} \mathbf{A}_{bound} \mathbf{I}^{D,G})$, identifies faces associated with regions that touch a grain, while the second term, $(\mathbf{I}^{F,D} \mathbf{I}^{D,G})$, filters out faces that are not part of the grain. This final incidence matrix is what we use to find the mesh vertices, edges, and faces that will be displayed graphically.

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
