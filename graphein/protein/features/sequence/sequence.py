"""Functions for graph-level featurization of the sequence of a protein. This submodule is focussed on physicochemical
proporties of the sequence."""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from functools import partial

import networkx as nx
from Bio import SeqUtils
from multipledispatch import dispatch

from graphein.protein.features.sequence.utils import (
    aggregate_feature_over_chains,
    compute_feature_over_chains,
)

# from graphein.protein.features.utils import aggregate_graph_feature_over_chains


@dispatch(str, str)
def molecular_weight(protein: str, seq_type: str = "protein"):
    func = partial(SeqUtils.molecular_weight, seq_type=seq_type)

    return func(protein)


@dispatch(nx.Graph, str)
def molecular_weight(protein: nx.Graph, seq_type: str = "protein"):
    func = partial(SeqUtils.molecular_weight, seq_type=seq_type)

    G = compute_feature_over_chains(
        protein, func, feature_name="molecular_weight"
    )
    return G
