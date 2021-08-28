"""Functions for working with RNA Secondary Structure Graphs"""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Emmanuele Rossi, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import logging
from typing import Callable, Dict, List, Optional

import networkx as nx

from graphein.utils.utils import (
    annotate_edge_metadata,
    annotate_graph_metadata,
    annotate_node_metadata,
    compute_edges,
)

log = logging.getLogger(__name__)

RNA_BASES: List[str] = ["A", "U", "G", "C", "I"]

RNA_BASE_COLORS: Dict[str, str] = {
    "A": "r",
    "U": "b",
    "G": "g",
    "C": "y",
    "I": "m",
}

CANONICAL_BASE_PAIRINGS: Dict[str, str] = {
    "A": ["U"],
    "U": ["A"],
    "G": ["C"],
    "C": ["G"],
}

WOBBLE_BASE_PAIRINGS: Dict[str, str] = {
    "A": ["I"],
    "U": ["G", "I"],
    "G": ["U"],
    "C": ["I"],
    "I": ["A", "C", "U"],
}

VALID_BASE_PAIRINGS = {
    key: CANONICAL_BASE_PAIRINGS.get(key, [])
    + WOBBLE_BASE_PAIRINGS.get(key, [])
    for key in set(
        list(CANONICAL_BASE_PAIRINGS.keys())
        + list(WOBBLE_BASE_PAIRINGS.keys())
    )
}

SIMPLE_DOTBRACKET_NOTATION = ["(", ".", ")"]
SUPPORTED_PSEUDOKNOT_NOTATION = ["[", "]", "{", "}", "<", ">"]
SUPPORTED_DOTBRACKET_NOTATION = (
    SIMPLE_DOTBRACKET_NOTATION + SUPPORTED_PSEUDOKNOT_NOTATION
)


def validate_rna_sequence(s: str) -> None:
    """
    Validate RNA sequence. This ensures that it only containts supported bases. Supported bases are: "A", "U", "G", "C", "I".
    Supported bases can be accessed in graphein.rna.graphs.RNA_BASES

    :param s: Sequence to validate
    :type s: str
    :raises ValueError: Raises ValueError if the sequence contains an unsupported base character
    """
    letters_used = set(s)
    if not letters_used.issubset(RNA_BASES):
        offending_letter = letters_used.difference(RNA_BASES)
        position = s.index(offending_letter)
        raise ValueError(
            f"Invalid letter {offending_letter} found at position {position} in the sequence {s}."
        )


def validate_lengths(db: str, seq: str) -> None:
    """
    Check lengths of dotbracket and sequence match

    :param db: Dotbracket string to check
    :type db: str
    :param seq: RNA nucleotide sequence to check.
    :type seq: str
    :raises ValueError: Raises ValueError if lengths of dotbracket and sequence do not match.
    """
    if len(db) != len(seq):
        raise ValueError(
            f"Length of dotbracket ({len(db)}) does not match length of sequence ({len(seq)})."
        )


def validate_dotbracket(db: str) -> str:
    """
    Sanitize dotbracket string. This ensures that it only has supported symbols.
    SIMPLE_DOTBRACKET_NOTATION = ["(", ".", ")"]
    SUPPORTED_PSEUDOKNOT_NOTATION = ["[", "]", "{", "}", "<", ">"]
    SUPPORTED_DOTBRACKET_NOTATION = (
        SIMPLE_DOTBRACKET_NOTATION + SUPPORTED_PSEUDOKNOT_NOTATION
    )

    :param db: Dotbrack notation string
    :type db: str
    :raises ValueError: Raises ValueError if dotbracket notation contains unsupported symbols
    """
    chars_used = set(db)
    if not chars_used.issubset(SUPPORTED_DOTBRACKET_NOTATION):
        offending_letter = chars_used.difference(SUPPORTED_DOTBRACKET_NOTATION)
        position = db.index(offending_letter)
        raise ValueError(
            f"Invalid letter {offending_letter} found at position {position} in the sequence {db}."
        )


def construct_rna_graph(
    dotbracket: Optional[str],
    sequence: Optional[str],
    edge_construction_funcs: List[Callable],
    edge_annotation_funcs: Optional[List[Callable]] = None,
    node_annotation_funcs: Optional[List[Callable]] = None,
    graph_annotation_funcs: Optional[List[Callable]] = None,
) -> nx.Graph:
    """
    Constructs an RNA secondary structure graph from dotbracket notation

    :param dotbracket: Dotbracket notation representation of secondary structure
    :type dotbracket: str, optional
    :param sequence: Corresponding sequence RNA bases
    :type sequence: str, optional
    :param edge_construction_funcs: List of edge construction functions. Defaults to None.
    :type edge_construction_funcs: List[Callable], optional
    :param edge_annotation_funcs: List of edge metadata annotation functions. Defaults to None.
    :type edge_annotation_funcs: List[Callable], optional
    :param node_annotation_funcs: List of node metadata annotation functions. Defaults to None.
    :type node_annotation_funcs: List[Callable], optional
    :param graph_annotation_funcs: List of graph metadata annotation functions. Defaults to None
    :type graph_annotation_funcs: List[Callable], optional
    :return: nx.Graph of RNA secondary structure
    :rtype: nx.Graph
    """
    G = nx.Graph()

    # Build node IDs first.
    node_ids = (
        list(range(len(sequence)))
        if sequence
        else list(range(len(dotbracket)))
    )

    # Check sequence and dotbracket lengths match
    if dotbracket and sequence:
        validate_lengths(dotbracket, sequence)

    # add nodes
    G.add_nodes_from(node_ids)
    log.debug(f"Added {len(node_ids)} nodes")

    # Add dotbracket symbol if dotbracket is provided
    if dotbracket:
        validate_dotbracket(dotbracket)
        G.graph["dotbracket"] = dotbracket

        nx.set_node_attributes(
            G,
            dict(zip(node_ids, dotbracket)),
            "dotbracket_symbol",
        )

    # Add nucleotide base info if sequence is provided
    if sequence:
        validate_rna_sequence(sequence)
        G.graph["sequence"] = sequence
        nx.set_node_attributes(G, dict(zip(node_ids, sequence)), "nucleotide")
        colors = [RNA_BASE_COLORS[i] for i in sequence]
        nx.set_node_attributes(G, dict(zip(node_ids, colors)), "color")

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
        add_pseudoknots,
    )

    edge_funcs_1 = [
        add_base_pairing_interactions,
        add_phosphodiester_bonds,
        add_pseudoknots,
    ]
    edge_funcs_2 = [add_all_dotbracket_edges]

    # g = construct_rna_graph(
    #    "((((....))))..(())",
    #    "AUGAUGAUGAUGCICIAU",
    #    edge_construction_funcs=edge_funcs_1,
    # )

    g = construct_rna_graph(
        "......((((((......[[[))))))......]]]....",
        sequence=None,
        edge_construction_funcs=edge_funcs_1,
    )

    """
    h = construct_rna_graph(
        "((((....))))..(())",
        "AUGAUGAUGAUGCICIAU",
        edge_construction_funcs=edge_funcs_2,
    )
    """

    # assert g.edges() == h.edges()

    nx.info(g)

    edge_colors = nx.get_edge_attributes(g, "color").values()
    node_colors = nx.get_node_attributes(g, "color").values()

    nx.draw(
        g, edge_color=edge_colors  # , node_color=node_colors, with_labels=True
    )
    plt.show()
