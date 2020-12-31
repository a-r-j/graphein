"""Functions for working with RNA Secondary Structure Graphs"""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Emmanuele Rossi
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import logging
from typing import Callable, List, Optional

import networkx as nx

log = logging.getLogger(__name__)

RNA_BASES = ["A", "U", "G", "C", "I"]

RNA_BASE_COLORS = {
    "A": "r",
    "U": "b",
    "G": "g",
    "C": "y",
    "I": "m",
}

SUPPORTED_DOTBRACKET_NOTATION = ["(", ".", ")"]

# Todo Pseudoknots: Some secondary structure databases include other characters ( [] , {}, <>, a, etc...)
#  to represent pairing in pseudoknots.

# Todo checking of valid base-parings


def annotate_node_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Annotates nodes with metadata
    :param G: RNA Secondary structure graph to add node metadata to
    :type G: nx.Graph
    :param funcs: List of node metadata annotation functions
    :type funcs: List[Callable]
    :return: RNA secondary structure graph with node metadata added
    :rtype: nx.Graph
    """
    for func in funcs:
        for n, d in G.nodes(data=True):
            func(n, d)
    return G


def annotate_graph_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Annotates graph with graph-level metadata
    :param G: RNA Secondary structure graph to add graph-level metadata to
    :type G: nx.Graph
    :param funcs: List of graph metadata annotation functions
    :type funcs: List[Callable]
    :return: RNA secondary structure graph with node metadata added
    :rtype: nx.Graph
    """
    for func in funcs:
        func(G)
    return G


def annotate_edge_metadata(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Annotates RNA Secondary Structure graph edges with edge metadata
    :param G: RNA Secondary structure graph to add edge metadata to
    :type G: nx.Graph
    :param funcs: List of edge metadata annotation functions
    :type funcs: List[Callable]
    :return: RNA secondary structure graph with edge metadata added
    :rtype: nx.Graph
    """
    for func in funcs:
        for u, v, d in G.edges(data=True):
            func(u, v, d)
    return G


def compute_edges(G: nx.Graph, funcs: List[Callable]) -> nx.Graph:
    """
    Computes edges for an RNA Secondary structure graph from a list of edge construction functions
    :param G: RNA Secondary Structure graph to add features to
    :type G: nx.Graph
    :param funcs: List of edge construction functions
    :type funcs: List[Callable]
    :return: RNA Secondary structure graph with edges added
    :rtype: nx.Graph
    """
    for func in funcs:
        func(G)
    return G


def construct_rna_graph(
    dotbracket: Optional[str],
    sequence: Optional[str],
    edge_construction_funcs: [List[Callable]],
    edge_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    graph_annotation_funcs: Optional[List[Callable]] = None,
) -> nx.Graph:
    """
    Constructs an RNA secondary structure graph from dotbracket notation
    :param dotbracket: Dotbracket notation representation of secondary structure
    :type dotbracket: str
    :param sequence: Corresponding sequence RNA bases
    :type sequence: Optional[str]
    :param edge_construction_funcs: List of edge construction functions
    :type edge_construction_funcs: [List[Callable]]
    :param edge_annotation_funcs: List of edge metadata annotation functions
    :type edge_annotation_funcs: Optional[List[Callable]], default = None
    :param node_annotation_funcs: List of node metadata annotation functions
    :type node_annotation_funcs: Optional[List[Callable]], default = None
    :param graph_annotation_funcs: List of graph metadata annotation functions
    :type graph_annotation_funcs: Optiona[List[Callable]], default = None
    :return: nx.Graph of RNA secondary structure
    :rtype: nx.Graph
    """
    G = nx.Graph()

    # Check sequence and dotbracket lengths match
    if dotbracket and sequence:
        assert len(dotbracket) == len(
            sequence
        ), "Sequence and dotbracket lengths must match"

    # Assign dotbracket as graph metadata
    if dotbracket:
        # Perform substitution on dotbracket sequence
        dotbracket = [
            i if i in SUPPORTED_DOTBRACKET_NOTATION else "."
            for i in dotbracket
        ]
        G.graph["dotbracket"] = dotbracket
        node_ids = [i for i in range(len(dotbracket))]

    # Assign sequence as graph metadata
    if sequence:
        # Check sequence contains valid characters
        assert [
            i in RNA_BASES for i in sequence
        ], "Sequence must contain valid RNA bases: {A, U, G, C, I}"
        G.graph["sequence"] = sequence
        node_ids = [i for i in range(len(sequence))]

    # add nodes
    G.add_nodes_from(node_ids)
    log.debug(f"Added {len(node_ids)} nodes")

    # Add dotbracket symbol if dotbracket is provided
    if dotbracket:
        nx.set_node_attributes(
            G,
            dict(zip(node_ids, [i for i in dotbracket])),
            "dotbracket_symbol",
        )

    # Add nucleotide base info if sequence is provided
    if sequence:
        nx.set_node_attributes(
            G, dict(zip(node_ids, [i for i in sequence])), "nucleotide"
        )
        nx.set_node_attributes(
            G,
            dict(zip(node_ids, [RNA_BASE_COLORS[i] for i in sequence])),
            "color",
        )

    # Annotate additional graph metadata
    if graph_annotation_funcs is not None:
        G = annotate_graph_metadata(G, graph_annotation_funcs)

    # Annotate additional node metadata
    if node_annotation_funcs is not None:
        G = annotate_node_metadata(G, node_annotation_funcs)

    # Add edges
    G = compute_edges(G, edge_construction_funcs)

    # Annotate additional edge metadata
    if edge_annotation_funcs is not None:
        G = annotate_edge_metadata(G, edge_annotation_funcs)

    return G


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from graphein.rna.edges import (
        add_all_dotbracket_edges,
        add_base_pairing_interactions,
        add_phosphodiester_bonds,
    )

    edge_funcs_1 = [add_base_pairing_interactions, add_phosphodiester_bonds]
    edge_funcs_2 = [add_all_dotbracket_edges]

    g = construct_rna_graph(
        "((((....))))..(())",
        "AUGAUGAUGAUGCICIAU",
        edge_construction_funcs=edge_funcs_1,
    )
    h = construct_rna_graph(
        "((((....))))..(())",
        "AUGAUGAUGAUGCICIAU",
        edge_construction_funcs=edge_funcs_2,
    )

    assert g.edges() == h.edges()

    nx.info(g)

    edge_colors = nx.get_edge_attributes(g, "color").values()
    node_colors = nx.get_node_attributes(g, "color").values()

    nx.draw(
        g, edge_color=edge_colors, node_color=node_colors, with_labels=True
    )
    plt.show()
