"""Visualisation utilities for RNA Secondary Structure Graphs."""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from collections import defaultdict
from itertools import chain
from typing import Dict, List

import networkx as nx

from graphein.rna.graphs import RNA_BASE_COLORS


def plot_rna_graph(
    g: nx.Graph,
    label_base_type: bool = True,
    label_base_position: bool = False,
    label_dotbracket_symbol: bool = False,
    **kwargs,
):
    """Plots a RNA Secondary Structure Graph. Colours edges by kind.

    :param g: NetworkX graph of RNA secondary structure graph.
    :type g: nx.Graph
    :param label_base_type: Whether to label the base type of each base.
    :type label_base_type: bool
    :param label_base_position: Whether to label the base position of each base.
    :type label_base_position: bool
    :param label_dotbracket_symbol: Whether to label the dotbracket symbol of each base.
    """
    edge_colors = nx.get_edge_attributes(g, "color").values()
    node_colors = nx.get_node_attributes(g, "color").values()

    # Construct node labelling scheme
    node_label_dicts: List[Dict[int, str]] = []
    if label_base_type:
        node_label_dicts.append(
            {n: d["nucleotide"] for n, d in g.nodes(data=True)}
        )
    if label_base_position:
        node_label_dicts.append({n: str(n) for n in g.nodes()})
    elif label_dotbracket_symbol:
        node_label_dicts.append(
            {n: d["dotbracket_symbol"] for n, d in g.nodes(data=True)}
        )

    node_labels = defaultdict(str)
    for key, value in chain.from_iterable(map(dict.items, node_label_dicts)):
        node_labels[key] += value

    nx.draw(
        g,
        node_color=node_colors,
        edge_color=edge_colors,
        with_labels=True,
        labels=node_labels,
        **kwargs,
    )
