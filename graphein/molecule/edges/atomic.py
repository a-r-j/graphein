"""Functions for computing atomic structure of molecules."""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import logging

import networkx as nx

log = logging.getLogger(__name__)


def add_atom_bonds(G: nx.Graph) -> nx.Graph:
    """Adds atomic bonds to a molecular graph.

    :param G: Molecular graph to add atomic bond edges to.
    :type G: nx.Graph
    :return: Molecular graph with atomic bonds added.
    :rtype: nx.Graph
    """
    for bond in G.graph["rdmol"].GetBonds():
        n1, n2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if G.has_edge(n1, n2):
            G.edges[n1, n2]["kind"].add("bond")
        else:
            G.add_edge(n1, n2, kind={"bond"})
    return G