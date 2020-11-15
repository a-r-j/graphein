from typing import Any, Dict, List, Optional, Tuple, Union

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#from graphein.construct_graphs import ProteinGraph

# import plotly.plotly as py
# import plotly.graph_objs as go

### MODIFIED FROM ORIGINAL - NEEDS LOTS OF WORK

def protein_graph_plot_3d(
    g: nx.Graph,
    angle: int,
    out_path: str = None,
    figsize: Tuple[int, int] = (10, 7),
    node_alpha: float = 0.7,
    node_size_min: float = 20.0,
    node_size_multiplier: float = 20.0,
    bg_col: str = "white",
    out_format: str = ".png",
    colour_by: str = "seq_position",
    edge_col: str = "black",
    edge_alpha: float = 0.5,
) -> None:
    """
    Plots Protein Graphs as flattened 3D projections. Requires 3D coordinates to be present in the graph object
    :param g:
    :type g: dgl.DGLGraph
    :param angle:
    :type angle: int
    :param out_path:
    :param figsize:
    :param node_alpha:
    :param node_size_min:
    :param node_size_multiplier:
    :param bg_col:
    :param out_format:
    :param colour_by: Indicates way to colour graph: {"degree", "seq_position"}
    :param edge_col: Colour to draw edges with
    :param edge_alpha: Alpha (transparancy for edges)
    :type edge_alpha: float
    :return:
    """

    # Todo assertions
    assert (
        out_format is ".png" or ".svg"
    ), "We require either '.png' or '.svg' for saving plots"

    if type(g) is dgl.DGLGraph:
        node_attrs = ["coords"]
        edge_attrs = ["contacts"]
        g = g.to_networkx(node_attrs=node_attrs, edge_attrs=edge_attrs)
    else:
        assert nx.get_node_attributes(
            g, "coords"
        ), "We require coordinate features to draw a 3D plot"

    # Get node positions
    pos = nx.get_node_attributes(g, "coords")

    # Get number of nodes
    n = g.number_of_nodes()
    # Get the maximum number of edges adjacent to a single node

    #edge_max = max([g.degree(i) for i in range(n)])

    # Define color range proportional to number of edges adjacent to a single node
    if colour_by == "degree":
        colors = [plt.cm.plasma(g.degree(i) / edge_max) for i in range(n)]

    if colour_by == "seq_position":
        colors = [plt.cm.plasma(i / n) for i in range(n)]

    if colour_by == "residue_type":
        raise NotImplementedError

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
                #color=colors[key],
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
    ax.set_facecolor("white")
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
        edge_construction=["contacts"],  # , 'delaunay', 'k_nn'],
        encoding=False,
        k_nn=None,
    )

    protein_graph_plot_3d(
        g, angle=30, out_path=None, figsize=(10, 7), colour_by="seq_position"
    )
