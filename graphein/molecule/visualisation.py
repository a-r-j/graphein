"""Functions for featurising Small Molecule Graphs.

Plotting functions for molecules wrap the methods defined on protein graphs and provide sane defaults.
"""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from graphein.protein.visualisation import (
    plot_protein_structure_graph,
    plotly_protein_structure_graph,
)


def plot_molecular_graph(
    G: nx.Graph,
    angle: int = 30,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 7),
    node_alpha: float = 0.7,
    node_size_min: float = 20.0,
    node_size_multiplier: float = 1,
    label_node_ids: bool = True,
    node_colour_map: plt.cm = plt.cm.plasma,
    edge_color_map: plt.cm = plt.cm.plasma,
    colour_nodes_by: str = "element",
    colour_edges_by: str = "kind",
    edge_alpha: float = 0.5,
    plot_style: str = "ggplot",
    out_path: Optional[str] = None,
    out_format: str = ".png",
) -> Axes3D:
    """
    Plots molecular graph in ``Axes3D``.

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
    :param node_size_min: Specifies node minimum size, defaults to ``20``.
    :type node_size_min: float
    :param node_size_multiplier: Scales node size by a constant. Node sizes reflect degree. Defaults to ``20``.
    :type node_size_multiplier: float
    :param label_node_ids: bool indicating whether or not to plot ``node_id`` labels. Defaults to ``True``.
    :type label_node_ids: bool
    :param node_colour_map: colour map to use for nodes. Defaults to ``plt.cm.plasma``.
    :type node_colour_map: plt.cm
    :param edge_color_map: colour map to use for edges. Defaults to ``plt.cm.plasma``.
    :type edge_color_map: plt.cm
    :param colour_nodes_by: Specifies how to colour nodes. ``"degree"``, ``"seq_position"`` or a node feature.
    :type colour_nodes_by: str
    :param colour_edges_by: Specifies how to colour edges. Currently only ``"kind"`` is supported.
    :type colour_edges_by: str
    :param edge_alpha: Controls edge transparency. Defaults to ``0.5``.
    :type edge_alpha: float
    :param plot_style: matplotlib style sheet to use. Defaults to ``"ggplot"``.
    :type plot_style: str
    :param out_path: If not none, writes plot to this location. Defaults to ``None`` (does not save).
    :type out_path: str, optional
    :param out_format: Fileformat to use for plot
    :type out_format: str
    :return: matplotlib Axes3D object.
    :rtype: Axes3D
    """

    return plot_protein_structure_graph(
        G,
        angle=angle,
        plot_title=plot_title,
        figsize=figsize,
        node_alpha=node_alpha,
        node_size_min=node_size_min,
        node_size_multiplier=node_size_multiplier,
        label_node_ids=label_node_ids,
        node_colour_map=node_colour_map,
        edge_color_map=edge_color_map,
        colour_nodes_by=colour_nodes_by,
        colour_edges_by=colour_edges_by,
        edge_alpha=edge_alpha,
        plot_style=plot_style,
        out_path=out_path,
        out_format=out_format,
    )


def plotly_molecular_graph(
    g: nx.Graph,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (620, 650),
    node_alpha: float = 0.7,
    node_size_min: float = 20,
    node_size_multiplier: float = 1.0,
    label_node_ids: bool = True,
    node_color_map: plt.cm = plt.cm.plasma,
    edge_color_map: plt.cm = plt.cm.plasma,
    colour_nodes_by: str = "element",
    colour_edges_by: str = "kind",
) -> go.Figure:
    """
    Plots molecular graph using plotly.

    :param G:  nx.Graph Molecular graph to plot
    :type G: nx.Graph
    :param plot_title: Title of plot, defaults to ``None``.
    :type plot_title: str, optional
    :param figsize: Size of figure, defaults to ``(620, 650)``.
    :type figsize: Tuple[int, int]
    :param node_alpha: Controls node transparency, defaults to ``0.7``.
    :type node_alpha: float
    :param node_size_min: Specifies node minimum size. Defaults to ``20.0``.
    :type node_size_min: float
    :param node_size_multiplier: Scales node size by a constant. Node sizes reflect degree. Defaults to ``1.0``.
    :type node_size_multiplier: float
    :param label_node_ids: bool indicating whether or not to plot ``node_id`` labels. Defaults to ``True``.
    :type label_node_ids: bool
    :param node_colour_map: colour map to use for nodes. Defaults to ``plt.cm.plasma``.
    :type node_colour_map: plt.cm
    :param edge_color_map: colour map to use for edges. Defaults to ``plt.cm.plasma``.
    :type edge_color_map: plt.cm
    :param colour_nodes_by: Specifies how to colour nodes. ``"degree"``, or a node feature. Defaults to ``"element"``.
    :type colour_edges_by: str
    :param colour_edges_by: Specifies how to colour edges. Currently only ``"kind"`` is supported.
    :type colour_nodes_by: str
    :returns: Plotly Graph Objects plot
    :rtype: go.Figure
    """
    return plotly_protein_structure_graph(
        g,
        plot_title=plot_title,
        figsize=figsize,
        node_alpha=node_alpha,
        node_size_min=node_size_min,
        node_size_multiplier=node_size_multiplier,
        label_node_ids=label_node_ids,
        node_colour_map=node_color_map,
        edge_color_map=edge_color_map,
        colour_nodes_by=colour_nodes_by,
        colour_edges_by=colour_edges_by,
    )
