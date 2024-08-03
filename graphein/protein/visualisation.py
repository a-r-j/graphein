"""Functions for plotting protein graphs and meshes."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import re
from itertools import count
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.colors as co
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from loguru import logger as log
from mpl_toolkits.mplot3d import Axes3D

from graphein.protein.subgraphs import extract_k_hop_subgraph
from graphein.utils.dependencies import import_message

try:
    from pytorch3d.ops import sample_points_from_meshes
except ImportError:
    message = import_message(
        submodule="graphein.protein.visualisation",
        package="pytorch3d",
        conda_channel="pytorch3d",
    )
    log.debug(message)

try:
    from mpl_chord_diagram import chord_diagram
except ImportError:
    import_message(
        submodule="graphein.protein.visualisation",
        package="mpl_chord_diagram",
        pip_install=True,
        extras=True,
    )
    log.debug(message)


def plot_pointcloud(mesh: Meshes, title: str = "") -> Axes3D:
    """
    Plots pytorch3d Meshes object as pointcloud.

    :param mesh: Meshes object to plot.
    :type mesh: pytorch3d.structures.meshes.Meshes
    :param title: Title of plot.
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
    G: nx.Graph,
    colour_by: str,
    colour_map: matplotlib.colors.ListedColormap = plt.cm.plasma,
) -> List[Tuple[float, float, float, float]]:
    """
    Computes node colours based on ``"degree"``, ``"seq_position"`` or node
    attributes.

    :param G: Graph to compute node colours for
    :type G: nx.Graph
    :param colour_map:  Colourmap to use.
    :type colour_map: matplotlib.colors.ListedColormap
    :param colour_by: Manner in which to colour nodes. If not ``"degree"`` or
        ``"seq_position"``, this must correspond to a node feature.
    :type colour_by: str
    :return: List of node colours
    :rtype: List[Tuple[float, float, float, float]]
    """
    # get number of nodes
    n = G.number_of_nodes()

    # Define color range proportional to number of edges adjacent to a single
    # node
    if colour_by == "degree":
        # Get max number of edges connected to a single node
        edge_max = max(G.degree[i] for i in G.nodes())
        colors = [colour_map(G.degree[i] / (edge_max + 1)) for i in G.nodes()]
    elif colour_by == "seq_position":
        colors = [colour_map(i / n) for i in range(n)]
    elif colour_by == "chain":
        chains = G.graph["chain_ids"]
        chain_colours = dict(
            zip(chains, list(colour_map(1 / len(chains), 1, len(chains))))
        )
        colors = [chain_colours[d["chain_id"]] for _, d in G.nodes(data=True)]
    elif colour_by == "plddt":
        levels: List[str] = ["Very High", "Confident", "Low", "Very Low"]
        mapping = dict(zip(sorted(levels), count()))
        colors = []
        for _, d in G.nodes(data=True):
            if d["b_factor"] > 90:
                colors.append((27 / 256, 86 / 256, 206 / 256, 1))
            elif d["b_factor"] > 70:
                colors.append((126 / 256, 202 / 256, 239 / 256, 1))
            elif d["b_factor"] > 50:
                colors.append((250 / 256, 218 / 256, 77 / 256, 1))
            else:
                colors.append((239 / 256, 131 / 256, 83 / 256, 1))
        # colors = [colour_map(mapping[c] / len(levels)) for c in colors]
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
    colour_by: Optional[str] = "kind",
    set_alpha: float = 1.0,
    return_as_rgba: bool = False,
) -> List[Tuple[float, float, float, float]] or List[str]:
    """
    Computes edge colours based on the kind of bond/interaction.

    :param G: nx.Graph protein structure graph to compute edge colours from.
    :type G: nx.Graph
    :param colour_map: Colourmap to use.
    :type colour_map: matplotlib.colors.ListedColormap
    :param colour_by: Edge attribute to colour by. Currently only ``"kind"`` is
        supported.
    :type colour_by: Optional[str]
    :param set_alpha: Sets a given alpha value between 0.0 and 1.0 for all the
        edge colours.
    :type set_alpha: float
    :param return_as_rgba: Returns a list of rgba strings instead of tuples.
    :type return_as_rgba: bool
    :return: List of edge colours.
    :rtype: List[Tuple[float, float, float, float]] or List[str]
    """
    if colour_by == "kind":
        edge_types = {
            frozenset(a) for a in nx.get_edge_attributes(G, "kind").values()
        }
        mapping = dict(zip(sorted(edge_types), count()))
        colors = [
            colour_map(
                mapping[frozenset(G.edges[i]["kind"])]
                / (len(edge_types) + 1)  # avoid division by zero error
            )
            for i in G.edges()
        ]
    elif colour_by is None:
        colors = [(0.0, 0.0, 0.0, 1.0) for _ in G.edges()]
    else:
        edge_types = set(nx.get_edge_attributes(G, colour_by).values())
        mapping = dict(zip(sorted(edge_types), count()))
        colors = [
            colour_map(mapping[d[colour_by]] / (len(edge_types) + 1))
            for _, _, d in G.edges(data=True)
        ]

    assert (
        0.0 <= set_alpha <= 1.0
    ), f"Alpha value {set_alpha} must be between 0.0 and 1.0"
    colors = [c[:3] + (set_alpha,) for c in colors]
    if return_as_rgba:
        return [
            f"rgba{tuple(list(co.convert_to_RGB_255(c[:3])) + [c[3]])}"
            for c in colors
        ]
    return colors


def plotly_protein_structure_graph(
    G: nx.Graph,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (620, 650),
    node_alpha: float = 0.7,
    node_size_min: float = 20.0,
    node_size_multiplier: float = 1.0,
    node_size_feature: str = "degree",
    label_node_ids: bool = True,
    node_colour_map=plt.cm.plasma,
    edge_color_map=plt.cm.plasma,
    colour_nodes_by: str = "degree",
    colour_edges_by: Optional[str] = "kind",
) -> go.Figure:
    """
    Plots protein structure graph using plotly.

    :param G:  nx.Graph Protein Structure graph to plot
    :type G: nx.Graph
    :param plot_title: Title of plot, defaults to ``None``.
    :type plot_title: str, optional
    :param figsize: Size of figure, defaults to ``(620, 650)``.
    :type figsize: Tuple[int, int]
    :param node_alpha: Controls node transparency, defaults to ``0.7``.
    :type node_alpha: float
    :param node_size_min: Specifies node minimum size. Defaults to ``20.0``.
    :type node_size_min: float
    :param node_size_multiplier: Scales node size by a constant. Node sizes
        reflect degree. Defaults to ``1.0``.
    :type node_size_multiplier: float
    :param node_size_feature: Which feature to scale the node size by. Defaults
        to ``degree``.
    :type node_size_feature: str
    :param label_node_ids: bool indicating whether or not to plot ``node_id``
        labels. Defaults to ``True``.
    :type label_node_ids: bool
    :param node_colour_map: colour map to use for nodes. Defaults to
        ``plt.cm.plasma``.
    :type node_colour_map: plt.cm
    :param edge_color_map: colour map to use for edges. Defaults to
        ``plt.cm.plasma``.
    :type edge_color_map: plt.cm
    :param colour_nodes_by: Specifies how to colour nodes. ``"degree"``,
        ``"seq_position"`` or a node feature.
    :type colour_nodes_by: str
    :param colour_edges_by: Specifies how to colour edges. Currently only
        ``"kind"`` or ``None`` are supported.
    :type colour_edges_by: Optional[str]
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

    # Get node size
    def node_scale_by(G: nx.Graph, feature: str):
        if feature == "degree":
            return lambda k: node_size_min + node_size_multiplier * G.degree[k]
        elif feature == "rsa":
            return (
                lambda k: node_size_min
                + node_size_multiplier * G.nodes(data=True)[k]["rsa"]
            )

        # Meiler embedding dimension
        p = re.compile("meiler-([1-7])")
        dim = p.search(feature)[1]
        if dim:
            return lambda k: node_size_min + node_size_multiplier * max(
                0, G.nodes(data=True)[k]["meiler"][f"dim_{dim}"]
            )  # Meiler values may be negative
        else:
            raise ValueError(f"Cannot size nodes by feature '{feature}'")

    get_node_size = node_scale_by(G, node_size_feature)

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
        node_sizes.append(get_node_size(key))

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

    # Loop on the list of edges to get the x,y,z, coordinates of the connected
    # nodes. Those two points are the extrema of the line to be plotted
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

    repeated_edge_colours = []
    for (
        edge_col
    ) in (
        edge_colors
    ):  # Repeat as each line segment is ({x,y,z}_start, {x,y,z}_end, None)
        repeated_edge_colours.extend((edge_col, edge_col, edge_col))

    edge_colors = repeated_edge_colours

    edge_text = [
        " / ".join(list(edge_type))
        for edge_type in nx.get_edge_attributes(G, "kind").values()
    ]
    edge_text = np.repeat(
        edge_text, 3
    )  # Repeat as each line segment is ({x,y,z}_start, {x,y,z}_end, None)

    edges = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line={"color": edge_colors, "width": 10},
        text=edge_text,
        hoverinfo="text",
    )

    return go.Figure(
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


def plot_protein_structure_graph(
    G: nx.Graph,
    angle: int = 30,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7),
    node_alpha: float = 0.7,
    node_size_min: float = 20.0,
    node_size_multiplier: float = 1.0,
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
    Plots protein structure graph in ``Axes3D``.

    :param G:  nx.Graph Protein Structure graph to plot.
    :type G: nx.Graph
    :param angle:  View angle. Defaults to ``30``.
    :type angle: int
    :param plot_title: Title of plot. Defaults to ``None``.
    :type plot_title: str, optional
    :param figsize: Size of figure, defaults to ``(10, 7)``.
    :type figsize: Tuple[int, int]
    :param node_alpha: Controls node transparency, defaults to ``0.7``.
    :type node_alpha: float
    :param node_size_min: Specifies node minimum size, defaults to ``20.0``.
    :type node_size_min: float
    :param node_size_multiplier: Scales node size by a constant. Node sizes
        reflect degree. Defaults to ``1.0``.
    :type node_size_multiplier: float
    :param label_node_ids: bool indicating whether or not to plot ``node_id``
        labels. Defaults to ``True``.
    :type label_node_ids: bool
    :param node_colour_map: colour map to use for nodes. Defaults to
        ``plt.cm.plasma``.
    :type node_colour_map: plt.cm
    :param edge_color_map: colour map to use for edges. Defaults to
        ``plt.cm.plasma``.
    :type edge_color_map: plt.cm
    :param colour_nodes_by: Specifies how to colour nodes. ``"degree"``,
        ``"seq_position"`` or a node feature.
    :type colour_nodes_by: str
    :param colour_edges_by: Specifies how to colour edges. Currently only
        ``"kind"`` is supported.
    :type colour_edges_by: str
    :param edge_alpha: Controls edge transparency. Defaults to ``0.5``.
    :type edge_alpha: float
    :param plot_style: matplotlib style sheet to use. Defaults to ``"ggplot"``.
    :type plot_style: str
    :param out_path: If not none, writes plot to this location. Defaults to
        ``None`` (does not save).
    :type out_path: str, optional
    :param out_format: File format to use for plot
    :type out_format: str
    :return: matplotlib Axes3D object.
    :rtype: Axes3D
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
        ax = Axes3D(fig)
        fig.add_axes(ax)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each
        # node
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

        # Loop on the list of edges to get the x,y,z, coordinates of the
        # connected nodes. Those two points are the extrema of the line to be
        # plotted
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


def add_vector_to_plot(
    g: nx.Graph,
    fig,
    vector: str = "sidechain_vector",
    scale: float = 5,
    colour: str = "red",
    width: int = 10,
) -> go.Figure:
    """Adds representations of vector features to the protein graph.

    Requires all nodes have a vector feature (1 x 3 array).

    :param g: Protein graph containing vector features
    :type g: nx.Graph
    :param fig: 3D plotly figure to add vectors to.
    :type fig: go.Figure
    :param vector: Name of node vector feature to add, defaults to
        ``"sidechain_vector"``.
    :type vector: str, optional
    :param scale: How much to scale the vectors by, defaults to 5
    :type scale: float, optional
    :param colour: Colours for vectors, defaults to ``"red"``.
    :type colour: str, optional
    :return: 3D Plotly plot with vectors added.
    :rtype: go.Figure
    """
    # Compute line segment positions
    x_edges = []
    y_edges = []
    z_edges = []
    edge_text = []
    for _, d in g.nodes(data=True):
        x_edges.extend(
            [d["coords"][0], d["coords"][0] + d[vector][0] * scale, None]
        )
        y_edges.extend(
            [d["coords"][1], d["coords"][1] + d[vector][1] * scale, None]
        )
        z_edges.extend(
            [d["coords"][2], d["coords"][2] + d[vector][2] * scale, None]
        )
        edge_text.extend([None, f"{vector}", None])

    edge_trace = go.Scatter3d(
        x=x_edges,
        y=y_edges,
        z=z_edges,
        mode="lines",
        line={"color": colour, "width": width},
        text=3 * [f"{vector}" for _ in range(len(g))],
        hoverinfo="text",
    )
    # Compute cone positions.
    arrow_tip_ratio = 0.1
    arrow_starting_ratio = 0.98
    x = []
    y = []
    z = []
    u = []
    v = []
    w = []
    for _, d in g.nodes(data=True):
        x.extend(
            [d["coords"][0] + d[vector][0] * scale * arrow_starting_ratio]
        )
        y.extend(
            [d["coords"][1] + d[vector][1] * scale * arrow_starting_ratio]
        )
        z.extend(
            [d["coords"][2] + d[vector][2] * scale * arrow_starting_ratio]
        )
        u.extend([d[vector][0]])  # * arrow_tip_ratio])
        v.extend([d[vector][1]])  # * arrow_tip_ratio])
        w.extend([d[vector][2]])  # * arrow_tip_ratio])

    if colour == "red":
        colour = [[0, "rgb(255,0,0)"], [1, "rgb(255,0,0)"]]
    elif colour == "blue":
        colour = [[0, "rgb(0,0,255)"], [1, "rgb(0,0,255)"]]
    elif colour == "green":
        colour = [[0, "rgb(0,255,0)"], [1, "rgb(0,255,0)"]]

    cone_trace = go.Cone(
        x=x,
        y=y,
        z=z,
        u=u,
        v=v,
        w=w,
        text=[f"{vector}" for _ in range(len(g.nodes()))],
        hoverinfo="u+v+w+text",
        colorscale=colour,
        showlegend=False,
        showscale=False,
        sizemode="absolute",
    )
    fig.add_trace(edge_trace)
    fig.add_trace(cone_trace)
    return fig


def plot_distance_matrix(
    g: Optional[nx.Graph],
    dist_mat: Optional[np.ndarray] = None,
    use_plotly: bool = True,
    title: Optional[str] = None,
    show_residue_labels: bool = True,
) -> go.Figure:
    """Plots a distance matrix of the graph.

    :param g: NetworkX graph containing a distance matrix as a graph attribute
        (``g.graph['dist_mat']``).
    :type g: nx.Graph, optional
    :param dist_mat: Distance matrix to plot. If not provided, the distance
        matrix is taken from the graph. Defaults to ``None``.
    :type dist_mat: np.ndarray, optional
    :param use_plotly: Whether to use ``plotly`` or ``seaborn`` for plotting.
        Defaults to ``True``.
    :type use_plotly: bool
    :param title: Title of the plot.Defaults to ``None``.
    :type title: str, optional
    :show_residue_labels: Whether to show residue labels on the plot. Defaults
        to ``True``.
    :type show_residue_labels: bool
    :raises: ValueError if neither a graph ``g`` or a ``dist_mat`` are provided.
    :return: Plotly figure.
    :rtype: px.Figure
    """
    if g is None and dist_mat is None:
        raise ValueError("Must provide either a graph or a distance matrix.")

    if dist_mat is None:
        dist_mat = g.graph["dist_mat"]  # type: ignore
    if g is not None:
        x_range = list(g.nodes)
        y_range = list(g.nodes)
        if not title:
            title = g.graph["name"] + " - Distance Matrix"
    else:
        x_range = list(range(dist_mat.shape[0]))  # type: ignore
        y_range = list(range(dist_mat.shape[1]))  # type: ignore
        if not title:
            title = "Distance matrix"

    if use_plotly:
        return px.imshow(
            dist_mat,
            x=x_range,
            y=y_range,
            labels=dict(color="Distance"),
            title=title,
        )
    tick_labels = x_range if show_residue_labels else []
    return sns.heatmap(
        dist_mat, xticklabels=tick_labels, yticklabels=tick_labels
    ).set(title=title)


def plot_distance_landscape(
    g: Optional[nx.Graph] = None,
    dist_mat: Optional[np.ndarray] = None,
    add_contour: bool = True,
    title: Optional[str] = None,
    width: int = 500,
    height: int = 500,
    autosize: bool = False,
) -> go.Figure:
    """Plots a distance landscape of the graph.

    :param g: Graph to plot (must contain a distance matrix in
        ``g.graph["dist_mat"]``).
    :type g: nx.Graph
    :param add_contour: Whether or not to show the contour, defaults to
        ``True``.
    :type add_contour: bool, optional
    :param width: Plot width, defaults to ``500``.
    :type width: int, optional
    :param height: Plot height, defaults to ``500``.
    :type height: int, optional
    :param autosize: Whether or not to autosize the plot, defaults to ``False``.
    :type autosize: bool, optional
    :return: Plotly figure of distance landscape.
    :rtype: go.Figure
    """
    if g:
        dist_mat = g.graph["dist_mat"]
        if not title:
            title = g.graph["name"] + " - Distance Landscape"
        tick_labels = list(g.nodes)
    else:
        if not title:
            title = "Distance landscape"
        tick_labels = list(range(dist_mat.shape[0]))

    fig = go.Figure(data=[go.Surface(z=dist_mat)])

    if add_contour:
        fig.update_traces(
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True,
            )
        )

    fig.update_layout(
        title=title,
        autosize=autosize,
        width=width,
        height=height,
        scene=dict(
            zaxis_title="Distance",
            xaxis=dict(
                ticktext=tick_labels,
                tickvals=list(range(len(tick_labels))),
                nticks=10,
                showticklabels=False,
            ),
            yaxis=dict(
                ticktext=tick_labels,
                tickvals=list(range(len(tick_labels))),
                nticks=10,
                showticklabels=False,
            ),
        ),
    )

    return fig


def asteroid_plot(
    g: nx.Graph,
    node_id: str,
    k: int = 2,
    colour_nodes_by: str = "shell",  # residue_name
    colour_edges_by: str = "kind",
    edge_colour_map: plt.cm.Colormap = plt.cm.plasma,
    edge_alpha: float = 1.0,
    show_labels: bool = True,
    title: Optional[str] = None,
    width: int = 600,
    height: int = 500,
    use_plotly: bool = True,
    show_edges: bool = False,
    show_legend: bool = True,
    node_size_multiplier: float = 10.0,
) -> Union[plotly.graph_objects.Figure, matplotlib.figure.Figure]:
    # sourcery skip: remove-unnecessary-else, swap-if-else-branches
    """Plots a k-hop subgraph around a node as concentric shells.

    Radius of each point is proportional to the degree of the node
        (modified by node_size_multiplier).

    :param g: NetworkX graph to plot.
    :type g: nx.Graph
    :param node_id: Node to centre the plot around.
    :type node_id: str
    :param k: Number of hops to plot. Defaults to ``2``.
    :type k: int
    :param colour_nodes_by: Colour the nodes by this attribute. Currently only
        ``"shell"`` is supported.
    :type colour_nodes_by: str
    :param colour_edges_by: Colour the edges by this attribute. Currently only
        ``"kind"`` is supported.
    :type colour_edges_by: str
    :param edge_colour_map: Colour map for edges. Defaults to ``plt.cm.plasma``.
    :type edge_colour_map: plt.cm.Colormap
    :param edge_alpha: Sets a given alpha value between 0.0 and 1.0 for all the
        edge colours.
    :type edge_alpha: float
    :param title: Title of the plot. Defaults to ``None``.
    :type title: str
    :param width: Width of the plot. Defaults to ``600``.
    :type width: int
    :param height: Height of the plot. Defaults to ``500``.
    :type height: int
    :param use_plotly: Use plotly to render the graph. Defaults to ``True``.
    :type use_plotly: bool
    :param show_edges: Whether to show edges in the plot. Defaults to ``False``.
    :type show_edges: bool
    :param show_legend: Whether to show the legend of the edges. Defaults to
        `True``.
    :type show_legend: bool
    :param node_size_multiplier: Multiplier for the size of the nodes. Defaults
        to ``10.0``.
    :type node_size_multiplier: float
    :returns: Plotly figure or matplotlib figure.
    :rtpye: Union[plotly.graph_objects.Figure, matplotlib.figure.Figure]
    """
    assert node_id in g.nodes(), f"Node {node_id} not in graph"

    nodes: Dict[int, List[str]] = {0: [node_id]}
    node_list: List[str] = [node_id]
    # Iterate over the number of hops and extract nodes in each shell
    for i in range(1, k):
        subgraph = extract_k_hop_subgraph(g, node_id, k=i)
        candidate_nodes = subgraph.nodes()
        # Check we've not already found nodes in the previous shells
        nodes[i] = [n for n in candidate_nodes if n not in node_list]
        node_list += candidate_nodes
    shells = [nodes[i] for i in range(k)]
    log.debug(f"Plotting shells: {shells}")

    if use_plotly:
        # Get shell layout and set as node attributes.
        pos = nx.shell_layout(subgraph, shells)
        nx.set_node_attributes(subgraph, pos, "pos")

        if show_edges:
            edge_colors = colour_edges(
                subgraph,
                colour_map=edge_colour_map,
                colour_by=colour_edges_by,
                set_alpha=edge_alpha,
                return_as_rgba=True,
            )
            show_legend_bools = [
                x not in edge_colors[:i] for i, x in enumerate(edge_colors)
            ]
            edge_trace = []
            for i, (u, v) in enumerate(subgraph.edges()):
                x0, y0 = subgraph.nodes[u]["pos"]
                x1, y1 = subgraph.nodes[v]["pos"]
                bond_kind = " / ".join(list(subgraph[u][v]["kind"]))
                tr = go.Scatter(
                    x=(x0, x1),
                    y=(y0, y1),
                    mode="lines",
                    line=dict(width=1, color=edge_colors[i]),
                    hoverinfo="text",
                    text=[bond_kind],
                    name=bond_kind,
                    legendgroup=bond_kind,
                    showlegend=show_legend_bools[i],
                )
                edge_trace.append(tr)

        node_x: List[str] = []
        node_y: List[str] = []
        for node in subgraph.nodes():
            x, y = subgraph.nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)

        degrees = [
            subgraph.degree(n) * node_size_multiplier for n in subgraph.nodes()
        ]

        if colour_nodes_by == "shell":
            node_colours = []
            for n in subgraph.nodes():
                for k, v in nodes.items():
                    if n in v:
                        node_colours.append(k)
        else:
            raise NotImplementedError(
                f"Colour by {colour_nodes_by} not implemented."
            )
            # TODO colour by AA type
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=list(subgraph.nodes()),
            mode="markers+text" if show_labels else "markers",
            hoverinfo="text",
            textposition="bottom center",
            showlegend=False,
            marker=dict(
                colorscale="YlGnBu",
                reversescale=True,
                color=node_colours,
                size=degrees,
                colorbar=dict(
                    thickness=15,
                    title="Shell",
                    tickvals=list(range(k)),
                    xanchor="left",
                    titleside="right",
                ),
                line_width=2,
            ),
        )

        data = edge_trace + [node_trace] if show_edges else [node_trace]
        return go.Figure(
            data=data,
            layout=go.Layout(
                title=title or f'Asteroid Plot - {g.graph["name"]}',
                width=width,
                height=height,
                titlefont_size=16,
                legend=dict(yanchor="top", y=1, xanchor="left", x=1.10),
                showlegend=show_legend,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False
                ),
                yaxis=dict(
                    showgrid=False, zeroline=False, showticklabels=False
                ),
            ),
        )
    else:
        nx.draw_shell(subgraph, nlist=shells, with_labels=show_labels)


def plot_chord_diagram(
    g: nx.Graph,
    show_names: bool = True,
    order: Optional[List] = None,
    width: float = 0.1,
    pad: float = 2.0,
    gap: float = 0.03,
    chordwidth: float = 0.7,
    ax=None,
    colors=None,
    cmap=None,
    alpha=0.7,
    use_gradient: bool = False,
    chord_colors=None,
    show: bool = False,
    **kwargs,
):
    """
    Plot a chord diagram.


    Based on Tanguy Fardet's implementation:
    https://github.com/tfardet/mpl_chord_diagram

    :param g: NetworkX graph to plot
        Flux data, mat[i, j] is the flux from i to j (adjacency matrix)
    :type g: nx.Graph
    :param show_names: Whether to show the names of the nodes
    :type show_names: bool
    :param order: list, optional (default: order of the matrix entries)
        Order in which the arcs should be placed around the trigonometric
        circle.
    :param width: float, optional (default: 0.1)
        Width/thickness of the ideogram arc.
    :type width: float
    :param pad: float, optional (default: 2)
        Distance between two neighboring ideogram arcs. Unit: degree.
    :type pad: float
    :param gap: float, optional (default: 0)
        Distance between the arc and the beginning of the cord.
    :type gap: float
    :param chordwidth: float, optional (default: 0.7)
        Position of the control points for the chords, controlling their shape.
    :param ax: matplotlib axis, optional (default: new axis)
        Matplotlib axis where the plot should be drawn.
    :param colors: list, optional (default: from `cmap`)
        List of user defined colors or floats.
    :param cmap: str or colormap object (default: viridis)
        Colormap that will be used to color the arcs and chords by default.
        See `chord_colors` to use different colors for chords.
    :param alpha: float in [0, 1], optional (default: 0.7)
        Opacity of the chord diagram.
    :param use_gradient: bool, optional (default: False)
        Whether a gradient should be use so that chord extremities have the
        same color as the arc they belong to.
    :type use_gradient: bool
    :param chord_colors: str, or list of colors, optional (default: None)
        Specify color(s) to fill the chords differently from the arcs.
        When the keyword is not used, chord colors default to the colomap given
        by `colors`.
        Possible values for `chord_colors` are:

        * a single color (do not use an RGB tuple, use hex format instead),
        e.g. "red" or "#ff0000"; all chords will have this color
        * a list of colors, e.g. ``["red", "green", "blue"]``, one per node
        (in this case, RGB tuples are accepted as entries to the list).
        Each chord will get its color from its associated source node, or
        from both nodes if `use_gradient` is True.
    :param show: bool, optional (default: False)
        Whether the plot should be displayed immediately via an automatic call
        to ``plt.show()``.
    :param kwargs: keyword arguments
        Available kwargs are:

        ================  ==================  ===============================
            Name               Type           Purpose and possible values
        ================  ==================  ===============================
        fontcolor         str or list         Color of the names
        fontsize          int                 Size of the font for names
        rotate_names      (list of) bool(s)   Rotate names by 90°
        sort              str                 Either "size" or "distance"
        zero_entry_size   float               Size of zero-weight reciprocal
        ================  ==================  ===============================
    :type kwargs: Dict[str, Any]
    """
    mat = nx.adjacency_matrix(g).todense()
    names = list(g.nodes)
    if show_names:
        if g.graph["node_type"] == "chain":
            names = [f"Chain {n}" for n in names]
        elif g.graph["node_type"] != "secondary_structure":
            raise ValueError()
    else:
        names = None

    a = chord_diagram(
        mat,
        names=names,
        order=order,
        width=width,
        pad=pad,
        gap=gap,
        chordwidth=chordwidth,
        ax=ax,
        colors=colors,
        cmap=cmap,
        alpha=alpha,
        use_gradient=use_gradient,
        chord_colors=chord_colors,
        show=show,
        **kwargs,
    )
