"""Functions to compute edges for an RNA secondary structure graph"""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Emmanuele Rossi
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import logging

import networkx as nx

log = logging.getLogger(__name__)


def add_phosphodiester_bonds(G: nx.Graph) -> nx.Graph:
    """
    Adds phosphodiester bonds between adjacent nucleotides to an RNA secondary structure graph
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
    Adds base_pairing interactions between nucleotides to an RNA secondary structure graph
    :param G: RNA Graph to add edges to
    :type G: nx.Graph
    :return: RNA graph with base_pairing edges added
    :rtype: nx.Graph
    """
    # Iterate over dotbracket to build connectivity
    bases = []
    for i, c in enumerate(G.graph["dotbracket"]):
        # Add base_pairing interactions
        if c == "(":
            bases.append(i)
        elif c == ")":
            neighbor = bases.pop()
            G.add_edge(i, neighbor, attr="base_pairing", color="r")
        elif c == ".":
            continue
        else:
            raise ValueError("Input is not in dot-bracket notation!")
        log.debug("Added base_pairing interactions as edges")
    return G


def add_all_dotbracket_edges(G: nx.Graph) -> nx.Graph:
    """
    Adds phosphodiester bonds between adjacent nucleotides and base_pairing interactions to an RNA secondary structure graph
    :param G: RNA Graph to add edges to
    :type G: nx.Graph
    :return: RNA graph with phosphodiester_bond and base_pairing edges added
    :rtype: nx.Graph
    """
    # Iterate over dotbracket to build connectivity
    G = add_phosphodiester_bonds(G)
    G = add_base_pairing_interactions(G)
    return G
