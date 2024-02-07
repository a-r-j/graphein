"""Functions for computing atomic structure of molecules."""

import networkx as nx

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from loguru import logger as log


def add_atom_bonds(G: nx.Graph) -> nx.Graph:
    """Adds atomic bonds to a molecular graph.

    :param G: Molecular graph to add atomic bond edges to.
    :type G: nx.Graph
    :return: Molecular graph with atomic bonds added.
    :rtype: nx.Graph
    """
    for bond in G.graph["rdmol"].GetBonds():
        n1, n2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        sym1, sym2 = (
            G.graph["rdmol"].GetAtoms()[n1].GetSymbol(),
            G.graph["rdmol"].GetAtoms()[n2].GetSymbol(),
        )
        n1 = f"{sym1}:{str(n1)}"
        n2 = f"{sym2}:{str(n2)}"
        if G.has_edge(n1, n2):
            G.edges[n1, n2]["kind"].add("bond")
            G.edges[n1, n2]["bond"] = bond
        else:
            G.add_edge(n1, n2, kind={"bond"}, bond=bond)
    return G
