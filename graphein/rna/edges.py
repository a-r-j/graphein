"""Functions to compute edges for an RNA secondary structure graph."""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Emmanuele Rossi, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import logging

import networkx as nx

from .graphs import CANONICAL_BASE_PAIRINGS, WOBBLE_BASE_PAIRINGS

log = logging.getLogger(__name__)


def check_base_pairing_type(base_1: str, base_2: str) -> str:
    """
    Checks type and validity of base pairing interactions.

    :param base_1: str RNA Base letter for base 1
    :type base_1: str
    :param base_2: str RNA base letter for base 2
    :type base_2: str
    :return: string referencing the type of base pairing:
    :rtype: str
    """
    try:
        if base_2 in CANONICAL_BASE_PAIRINGS[base_1]:
            return "canonical"
        elif base_2 in WOBBLE_BASE_PAIRINGS[base_1]:
            return "wobble"
    except:
        return "invalid"


def add_phosphodiester_bonds(G: nx.Graph) -> nx.Graph:
    """
    Adds phosphodiester bonds between adjacent nucleotides to an RNA secondary structure graph.

    :param G: RNA Graph to add edges to
    :type G: nx.Graph
    :return: RNA graph with phosphodiester_bond edges added
    :rtype: nx.Graph
    """
    # Iterate over dotbracket to build connectivity
    bases = []
    for i, c in enumerate(G.graph["dotbracket"]):
        # Add adjacent edges (phosphodiester_bonds)
        if i > 0:
            G.add_edge(i, i - 1, attr="phosphodiester_bond", color="b")
    log.debug("Added phosphodiester bonds as edges")
    return G


def add_base_pairing_interactions(G: nx.Graph) -> nx.Graph:
    """
    Adds base_pairing interactions between nucleotides to an RNA secondary structure graph.

    :param G: RNA Graph to add edges to
    :type G: nx.Graph
    :return: RNA graph with base_pairing edges added
    :rtype: nx.Graph
    """
    # Check sequence is used
    check_base_pairing = "sequence" in G.graph.keys()
    # Iterate over dotbracket to build connectivity
    bases = []
    for i, c in enumerate(G.graph["dotbracket"]):
        # Add base_pairing interactions
        if c == "(":
            bases.append(i)
        elif c == ")":
            neighbor = bases.pop()

            if check_base_pairing:
                pairing_type = check_base_pairing_type(
                    G.nodes[i]["nucleotide"], G.nodes[neighbor]["nucleotide"]
                )
            else:
                pairing_type = "unknown"

            G.add_edge(
                i,
                neighbor,
                attr="base_pairing",
                pairing_type=pairing_type,
                color="r",
            )
        elif c in [".", "[", "]", "{", "}", "<", ">"]:
            continue
        else:
            raise ValueError("Input is not in dot-bracket notation!")
        log.debug("Added base_pairing interactions as edges")
    return G


def add_pseudoknots(G: nx.Graph) -> nx.Graph:
    """
    Adds pseudoknots nucleotides to an RNA secondary structure graph.

    :param G: RNA Graph to add edges to
    :type G: nx.Graph
    :return: RNA graph with pseudoknot edges added
    :rtype: nx.Graph
    """
    # Check sequence is used
    check_base_pairing = "sequence" in G.graph.keys()
    # Iterate over dotbracket to build connectivity
    knot_bases_1 = []  # for [[[]]] knots
    knot_bases_2 = []  # for {{{}}} knots
    knot_bases_3 = []  # for <<<>>> knots

    for i, c in enumerate(G.graph["dotbracket"]):
        if c in ["[", "{", "<"]:
            if c == "<":
                knot_bases_3.append(i)
            elif c == "[":
                knot_bases_1.append(i)
            elif c == "{":
                knot_bases_2.append(i)
        elif c in ["]", "}", ">"]:
            if c == ">":
                neighbor = knot_bases_3.pop()

            elif c == "]":
                neighbor = knot_bases_1.pop()
            elif c == "}":
                neighbor = knot_bases_2.pop()
            if check_base_pairing:
                pairing_type = check_base_pairing_type(
                    G.nodes[i]["nucleotide"], G.nodes[neighbor]["nucleotide"]
                )
            else:
                pairing_type = "unknown"
            G.add_edge(
                i,
                neighbor,
                attr="pseudoknot",
                pairing_type=pairing_type,
                color="g",
            )
        elif c in ["(", ")", "."]:
            continue
        else:
            raise ValueError("Input is not in dot-bracket notation!")
        log.debug("Added pseudoknot interactions as edges")
    return G


def add_all_dotbracket_edges(G: nx.Graph) -> nx.Graph:
    """
    Adds phosphodiester bonds between adjacent nucleotides and base_pairing interactions to an RNA secondary structure graph.

    :param G: RNA Graph to add edges to
    :type G: nx.Graph
    :return: RNA graph with phosphodiester_bond and base_pairing edges added
    :rtype: nx.Graph
    """
    # Iterate over dotbracket to build connectivity
    G = add_phosphodiester_bonds(G)
    G = add_base_pairing_interactions(G)
    G = add_pseudoknots(G)
    return G
