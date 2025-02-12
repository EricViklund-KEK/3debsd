import matplotlib.pyplot as plt




def Scatter3D(points, color = None, labels = None, range = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(points[:,0],points[:,1],points[:,2],c=color)

    if labels is not None:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

    if range is not None:
        ax.set_xlim(range[0:2])
        ax.set_ylim(range[2:4])
        ax.set_zlim(range[4:6])

    plt.show()

    return fig, ax

