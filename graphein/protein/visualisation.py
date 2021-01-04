"""Functions for plotting protein graphs and meshes"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein


from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.ops import sample_points_from_meshes


def plot_pointcloud(mesh: Meshes, title: str = "") -> None:
    """
    Plots pytorch3d Meshes object as pointcloud
    :param mesh: Meshes object to plot
    :param title: Title of plot
    :return:
    """
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()


if __name__ == "__main__":
    from graphein.protein.meshes import (
        convert_verts_and_face_to_mesh,
        create_mesh,
    )

    v, f, a = create_mesh(pdb_code="3eiy")
    m = convert_verts_and_face_to_mesh(v, f)

    plot_pointcloud(m, "Test")
