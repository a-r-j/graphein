"""Functions for computing atomic structure of proteins."""
import logging

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Any, Dict

import networkx as nx
import numpy as np
import pandas as pd


log = logging.getLogger(__name__)

def add_atom_bonds(G: nx.Graph) -> nx.Graph:
    for bond in G.graph["rdmol"].GetBonds():
        n1, n2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if G.has_edge(n1, n2):
            G.edges[n1, n2]["kind"].add("bond")
        else:
            G.add_edge(n1, n2, kind={"bond"})
    return G
    

