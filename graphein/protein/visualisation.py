"""Functions for plotting protein graphs and meshes"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from itertools import count
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

from graphein.utils.utils import import_message

try:
    from pytorch3d.ops import sample_points_from_meshes
except ImportError:
    import_message(
        submodule="graphein.protein.visualisation",
        package="pytorch3d",
        conda_channel="pytorch3d",
    )


def plot_pointcloud(mesh: Meshes, title: str = "") -> Axes3D:
    """
    Plots pytorch3d Meshes object as pointcloud

    :param mesh: Meshes object to plot
    :type mesh: pytorch3d.structures.meshes.Meshes
    :param title: Title of plot
    :type title: str
    :return: returns Axes3D containing plot
    :rtype: Axes3D
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
    return ax


def colour_nodes(
    G: nx.Graph, colour_map: matplotlib.colors.ListedColormap, colour_by: str
) -> List[Tuple[float, float, float, float]]:
    """
    Computes node colours based on "degree", "seq_position" or node attributes

    :param G: Graph to compute node colours for
    :type G: nx.Graph
    :param colour_map:  Colourmap to use.
    :type colour_map: matplotlib.colors.ListedColormap
    :param colour_by: Manner in which to colour nodes. If not "degree" or "seq_position", this must correspond to a node feature
    :type colour_by: str
    :return: List of node colours
    :rtype: List[Tuple[float, float, float, float]]
    """
    # get number of nodes
    n = G.number_of_nodes()

    # Define color range proportional to number of edges adjacent to a single node
    if colour_by == "degree":
        # Get max number of edges connected to a single node
        edge_max = max([G.degree[i] for i in G.nodes()])
        colors = [colour_map(G.degree[i] / edge_max) for i in G.nodes()]
    elif colour_by == "seq_position":
        colors = [colour_map(i / n) for i in range(n)]
    elif colour_by == "chain":
        chains = G.graph["chain_ids"]
        chain_colours = dict(
            zip(chains, list(colour_map(1 / len(chains), 1, len(chains))))
        )
        colors = [chain_colours[d["chain_id"]] for n, d in G.nodes(data=True)]
    else:
        node_types = set(nx.get_node_attributes(G, colour_by).values())
        mapping = dict(zip(sorted(node_types), count()))
        colors = [
            colour_map(mapping[d[colour_by]] / len(node_types))
            for n, d in G.nodes(data=True)
        ]

    return colors


def colour_edges(
    G: nx.Graph,
    colour_map: matplotlib.colors.ListedColormap,
    colour_by: str = "kind",
) -> List[Tuple[float, float, float, float]]:
    """
    Computes edge colours based on the kind of bond/interaction.

    :param G: nx.Graph protein structure graph to compute edge colours from
    :type G: nx.Graph
    :param colour_map: Colourmap to use
    :type colour_map: matplotlib.colors.ListedColormap
    :param colour_by: Edge attribute to colour by. Currently only "kind" is supported
    :type colour_by: str
    :return: List of edge colours
    :rtype: List[Tuple[float, float, float, float]]
    """
    if colour_by == "kind":
        edge_types = set(
            frozenset(a) for a in nx.get_edge_attributes(G, "kind").values()
        )
        mapping = dict(zip(sorted(edge_types), count()))
        colors = [
            colour_map(
                mapping[frozenset(G.edges[i]["kind"])]
                / (len(edge_types) + 1)  # avoid division by zero error
            )
            for i in G.edges()
        ]
    else:
        raise NotImplementedError(
            "Other edge colouring methods not implemented."
        )
    return colors


def plotly_protein_structure_graph(
    G: nx.Graph,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (620, 650),
    node_alpha: float = 0.7,
    node_size_min: float = 20.0,
    node_size_multiplier: float = 20.0,
    label_node_ids: bool = True,
    node_colour_map=plt.cm.plasma,
    edge_color_map=plt.cm.plasma,
    colour_nodes_by: str = "degree",
    colour_edges_by: str = "kind",
) -> go.Figure:
    """
    Plots protein structure graph using plotly.

    :param G:  nx.Graph Protein Structure graph to plot
    :type G: nx.Graph
    :param plot_title: Title of plot, defaults to None
    :type plot_title: str, optional
    :param figsize: Size of figure, defaults to (620, 650)
    :type figsize: Tuple[int, int]
    :param node_alpha: Controls node transparency, defaults to 0.7
    :type node_alpha: float
    :param node_size_min: Specifies node minimum size
    :type node_size_min: float
    :param node_size_multiplier: Scales node size by a constant. Node sizes reflect degree.
    :type node_size_multiplier: float
    :param label_node_ids: bool indicating whether or not to plot node_id labels
    :type label_node_ids: bool
    :param node_colour_map: colour map to use for nodes
    :type node_colour_map: plt.cm
    :param edge_color_map: colour map to use for edges
    :type edge_color_map: plt.cm
    :param colour_nodes_by: Specifies how to colour nodes. "degree", "seq_position" or a node feature
    :type colour_edges_by: str
    :param colour_edges_by: Specifies how to colour edges. Currently only "kind" is supported
    :type colour_nodes_by: str
    :returns: Plotly Graph Objects plot
    :rtype: go.Figure
    """

    # Get Node Attributes
    pos = nx.get_node_attributes(G, "coords")

    # Get node colours
    node_colors = colour_nodes(
        G, colour_map=node_colour_map, colour_by=colour_nodes_by
    )
    edge_colors = colour_edges(
        G, colour_map=edge_color_map, colour_by=colour_edges_by
    )

    # 3D network plot
    x_nodes = []
    y_nodes = []
    z_nodes = []
    node_sizes = []
    node_labels = []

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node
    for i, (key, value) in enumerate(pos.items()):
        x_nodes.append(value[0])
        y_nodes.append(value[1])
        z_nodes.append(value[2])
        node_sizes.append(node_size_min + node_size_multiplier * G.degree[key])

        if label_node_ids:
            node_labels.append(list(G.nodes())[i])

    nodes = go.Scatter3d(
        x=x_nodes,
        y=y_nodes,
        z=z_nodes,
        mode="markers",
        marker={
            "symbol": "circle",
            "color": node_colors,
            "size": node_sizes,
            "opacity": node_alpha,
        },
        text=list(G.nodes()),
        hoverinfo="text+x+y+z",
    )

    # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
    # Those two points are the extrema of the line to be plotted
    x_edges = []
    y_edges = []
    z_edges = []

    for node_a, node_b in G.edges(data=False):
        x_edges.extend([pos[node_a][0], pos[node_b][0], None])
        y_edges.extend([pos[node_a][1], pos[node_b][1], None])
        z_edges.extend([pos[node_a][2], pos[node_b][2], None])

    axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title="",
    )

    edges = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line={"color": edge_colors, "width": 10},
        text=[
            " / ".join(list(edge_type))
            for edge_type in nx.get_edge_attributes(G, "kind").values()
        ],
        hoverinfo="text",
    )

    fig = go.Figure(
        data=[nodes, edges],
        layout=go.Layout(
            title=plot_title,
            width=figsize[0],
            height=figsize[1],
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
            margin=dict(t=100),
        ),
    )

    return fig


def plot_protein_structure_graph(
    G: nx.Graph,
    angle: int = 30,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7),
    node_alpha: float = 0.7,
    node_size_min: float = 20.0,
    node_size_multiplier: float = 20.0,
    label_node_ids: bool = True,
    node_colour_map=plt.cm.plasma,
    edge_color_map=plt.cm.plasma,
    colour_nodes_by: str = "degree",
    colour_edges_by: str = "kind",
    edge_alpha: float = 0.5,
    plot_style: str = "ggplot",
    out_path: Optional[str] = None,
    out_format: str = ".png",
) -> Axes3D:
    """
    Plots protein structure graph in Axes3D.

    :param G:  nx.Graph Protein Structure graph to plot
    :type G: nx.Graph
    :param angle:  View angle
    :type angle: int
    :param plot_title: Title of plot. Defaults to None
    :type plot_title: str, optional
    :param figsize: Size of figure, defaults to (10, 7)
    :type figsize: Tuple[int, int]
    :param node_alpha: Controls node transparency, defaults to 0.7
    :type node_alpha: float
    :param node_size_min: Specifies node minimum size, defaults to 20
    :type node_size_min: float
    :param node_size_multiplier: Scales node size by a constant. Node sizes reflect degree.
    :type node_size_multiplier: float
    :param label_node_ids: bool indicating whether or not to plot node_id labels
    :type label_node_ids: bool
    :param node_colour_map: colour map to use for nodes
    :type node_colour_map: plt.cm
    :param edge_color_map: colour map to use for edges
    :type edge_color_map: plt.cm
    :param colour_nodes_by: Specifies how to colour nodes. "degree", "seq_position" or a node feature
    :type colour_nodes_by: str
    :param colour_edges_by: Specifies how to colour edges. Currently only "kind" is supported
    :param edge_alpha: Controls edge transparency
    :param plot_style: matplotlib style sheet to use
    :param out_path: If not none, writes plot to this location
    :param out_format: Fileformat to use for plot
    :return:
    """

    # Get Node Attributes
    pos = nx.get_node_attributes(G, "coords")

    # Get node colours
    node_colors = colour_nodes(
        G, colour_map=node_colour_map, colour_by=colour_nodes_by
    )
    edge_colors = colour_edges(
        G, colour_map=edge_color_map, colour_by=colour_edges_by
    )

    # 3D network plot
    with plt.style.context(plot_style):

        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig, auto_add_to_figure=True)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for i, (key, value) in enumerate(pos.items()):
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(
                xi,
                yi,
                zi,
                color=node_colors[i],
                s=node_size_min + node_size_multiplier * G.degree[key],
                edgecolors="k",
                alpha=node_alpha,
            )
            if label_node_ids:
                label = list(G.nodes())[i]
                ax.text(xi, yi, zi, label)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(G.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c=edge_colors[i], alpha=edge_alpha)

    # Set title
    ax.set_title(plot_title)
    # Set the initial view
    ax.view_init(30, angle)
    # Hide the axes
    ax.set_axis_off()
    if out_path is not None:
        plt.savefig(out_path + str(angle).zfill(3) + out_format)
        plt.close("all")

    return ax


if __name__ == "__main__":
    # TODO: Move the block here into tests.
    from graphein.protein.config import ProteinGraphConfig
    from graphein.protein.edges.atomic import (
        add_atomic_edges,
        add_bond_order,
        add_ring_status,
    )
    from graphein.protein.features.nodes.amino_acid import (
        expasy_protein_scale,
        meiler_embedding,
    )
    from graphein.protein.graphs import construct_graph

    # Test Point cloud plotting
    # v, f, a = create_mesh(pdb_code="3eiy")
    # m = convert_verts_and_face_to_mesh(v, f)
    # plot_pointcloud(m, "Test")
    # TEST PROTEIN STRUCTURE GRAPH PLOTTING
    configs = {
        "granularity": "atom",
        "keep_hets": False,
        "deprotonate": True,
        "insertions": False,
        "verbose": False,
    }

    config = ProteinGraphConfig(**configs)
    config.edge_construction_functions = [
        add_atomic_edges,
        add_ring_status,
        add_bond_order,
    ]

    config.node_metadata_functions = [meiler_embedding, expasy_protein_scale]

    g = construct_graph(
        config=config, pdb_path="../examples/pdbs/3eiy.pdb", pdb_code="3eiy"
    )

    p = plotly_protein_structure_graph(
        g,
        30,
        (1000, 2000),
        colour_nodes_by="element_symbol",
        colour_edges_by="kind",
        label_node_ids=False,
    )

    p.show()
