import dgl
from graphein.construct_graphs import ProteinGraph
import matplotlib.pyplot as plt

# import plotly.plotly as py
# import plotly.graph_objs as go

import networkx as nx
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, Tuple, List, Optional, Union


def network_plot_3d(
    g: dgl.DGLGraph,
    angle: int,
    out_path: str = None,
    figsize: Tuple[int, int] = (10, 7),
    node_alpha: float = 0.7,
    node_size_min: float = 20.0,
    node_size_multiplier: float = 20.0,
    bg_col: str = "white",
    out_format: str = ".png",
    colour_by: str = "degree",
    edge_col: str = "black",
    edge_alpha: float = 0.5,
) -> None:
    """
    Plots Protein Graphs as flattened 3D projections. Requires 3D coordinates to be present in the graph object
    :param G: DGLGraph
    :param angle: angle of view
    :param out_path:
    :return:
    """

    # Todo assertions
    if type(g) is dgl.DGLGraph:
        g = g.to_networkx(node_attrs=["coords"])

    # Get node positions
    pos = nx.get_node_attributes(g, "coords")
    # print(pos)

    # Get number of nodes
    n = g.number_of_nodes()
    # Get the maximum number of edges adjacent to a single node
    edge_max = max([g.degree(i) for i in range(n)])
    # Define color range proportional to number of edges adjacent to a single node
    colors = [plt.cm.plasma(g.degree(i) / edge_max) for i in range(n)]
    # 3D network plot
    with plt.style.context(("ggplot")):

        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(
                xi,
                yi,
                zi,
                color=colors[key],
                s=node_size_min + node_size_multiplier * g.degree(key),
                edgecolors="k",
                alpha=node_alpha,
            )

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(g.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c=edge_col, alpha=edge_alpha)

    # Set the initial view
    ax.view_init(30, angle)
    # Hide the axes
    ax.set_axis_off()
    if out_path is not None:
        plt.savefig(out_path + str(angle).zfill(3) + out_format)
        plt.close("all")
    else:
        plt.show()

    return


def plot_protein(
    pdb_code: Optional[str] = None, pdb_file: Optional[str] = None
) -> None:
    raise NotImplementedError


def plot_protein_graph_2d(g):

    if type(g) is nx.Graph or nx.DiGraph:
        nx.draw_networkx(g)

        return

    elif type(g) is dgl.DGLGraph:
        # Convert DGL graph to NX for plotting
        next

    raise NotImplementedError


def plot_protein_graph_3d(g):

    raise NotImplementedError


def plot_interactive_protein_graph(g):

    raise NotImplementedError


if __name__ == "__main__":

    pg = ProteinGraph(
        granularity="CA",
        insertions=False,
        keep_hets=False,
        intramolecular_interactions=None,
        get_contacts_path="/Users/arianjamasb/github/getcontacts",
        pdb_dir="../examples/pdbs/",
        contacts_dir="../examples/contacts/",
        exclude_waters=True,
        covalent_bonds=False,
        include_ss=True,
        include_ligand=False,
        verbose=True,
        long_interaction_threshold=5,
        edge_distance_cutoff=10,
        edge_featuriser=None,
        node_featuriser="meiler",
    )

    g = pg.dgl_graph_from_pdb_code(
        "3eiy",
        chain_selection="all",
        edge_construction=["distance", "delaunay"],  # , 'delaunay', 'k_nn'],
        encoding=False,
        k_nn=None,
    )

    network_plot_3d(g, angle=30, out_path=None, figsize=(10, 7))
