"""Visualisation utils for ML."""
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

from graphein.utils.utils import import_message

from ..protein.visualisation import plotly_protein_structure_graph
from .conversion import GraphFormatConvertor

try:
    from torch_geometric.data import Data
except ImportError:
    import_message(
        submodule="graphein.ml.conversion",
        package="torch_geometric",
        pip_install=True,
        conda_channel="rusty1s",
    )

try:
    import torch
except ImportError:
    import_message(
        submodule="graphein.ml.visualisation",
        package="torch",
        pip_install=True,
        conda_channel="pytorch",
    )


def plot_pyg_data(
    x: Data,
    node_colour_tensor: Optional[torch.Tensor] = None,
    edge_colour_tensor: Optional[torch.Tensor] = None,
    plot_title: Optional[str] = None,
    figsize: Tuple[int, int] = (620, 650),
    node_alpha: float = 0.7,
    node_size_min: float = 20.0,
    node_size_multiplier: float = 20.0,
    label_node_ids: bool = True,
    node_colour_map=plt.cm.plasma,
    edge_color_map=plt.cm.plasma,
    colour_nodes_by: str = "residue_name",
    colour_edges_by: Optional[str] = None,
) -> go.Figure:
    """
    Plots protein structure graph from ``torch_geometric.data.Data`` using plotly.

    This function can be used for logging proteins to e.g. wandb.

    :param x: ``torch_geometric.data.Data`` Protein Structure graph to plot.
    :type x: torch_geometric.data.Data
    :param node_colour_tensor: Tensor of node colours (must match length of nodes in graph).
        If ``None``, ``colour_nodes_by`` will be used. Default is ``None``.
    :type node_colour_tensor: torch.Tensor, optional
    :parm edge_colour_tensor: Tensor of edge colours (must match length of edges in graph).
        If ``None``, ``colour_edges_by`` will be used. Default is ``None``.
    :type edge_colour_tensor: torch.Tensor, optional
    :param plot_title: Title of plot, defaults to ``None``.
    :type plot_title: str, optional
    :param figsize: Size of figure, defaults to ``(620, 650)``.
    :type figsize: Tuple[int, int]
    :param node_alpha: Controls node transparency, defaults to ``0.7``.
    :type node_alpha: float
    :param node_size_min: Specifies node minimum size. Defaults to ``20.0``.
    :type node_size_min: float
    :param node_size_multiplier: Scales node size by a constant. Node sizes reflect degree. Defaults to ``20.0``.
    :type node_size_multiplier: float
    :param label_node_ids: bool indicating whether or not to plot ``node_id`` labels. Defaults to ``True``.
    :type label_node_ids: bool
    :param node_colour_map: colour map to use for nodes. Defaults to ``plt.cm.plasma``.
    :type node_colour_map: plt.cm
    :param edge_color_map: colour map to use for edges. Defaults to ``plt.cm.plasma``.
    :type edge_color_map: plt.cm
    :param colour_nodes_by: Specifies how to colour nodes. ``"degree"``, ``"seq_position"`` or a node feature.
    :type colour_nodes_by: str
    :param colour_edges_by: Specifies how to colour edges. Currently only ``"kind"`` or ``None`` are supported.
    :type colour_edges_by: Optional[str]
    :returns: Plotly Graph Objects plot
    :rtype: go.Figure
    """
    convertor = GraphFormatConvertor(src_format="pyg", dst_format="nx")
    nx_graph = convertor(x)
    node_map = dict(enumerate(x.node_id))
    nx_graph = nx.relabel_nodes(nx_graph, node_map)

    # Add metadata
    nx_graph.name = x.name
    nx_graph.graph["coords"] = x.coords[0]
    nx_graph.graph["dist_mat"] = x.dist_mat

    # Assign coords and seq info to nodes
    for i, (_, d) in enumerate(nx_graph.nodes(data=True)):
        d["chain_id"] = x.node_id[i].split(":")[0]
        d["residue_name"] = x.node_id[i].split(":")[1]
        d["seq_position"] = x.node_id[i].split(":")[2]
        d["coords"] = x.coords[0][i]
        if node_colour_tensor is not None:
            d["colour"] = float(node_colour_tensor[i])

    if edge_colour_tensor is not None:
        # TODO add edge types
        for i, (_, _, d) in enumerate(nx_graph.edges(data=True)):
            d["colour"] = float(edge_colour_tensor[i])

    return plotly_protein_structure_graph(
        nx_graph,
        plot_title,
        figsize,
        node_alpha,
        node_size_min,
        node_size_multiplier,
        label_node_ids,
        node_colour_map,
        edge_color_map,
        colour_nodes_by if node_colour_tensor is None else "colour",
        colour_edges_by if edge_colour_tensor is None else "colour",
    )
