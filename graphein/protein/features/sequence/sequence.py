# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from functools import partial
from typing import List, Optional

import networkx as nx
import pandas as pd

from graphein.protein.features.sequence.utils import (
    aggregate_feature_over_chains,
    compute_feature_over_chains,
)

# from graphein.protein.features.utils import aggregate_graph_feature_over_chains


"""
def molecular_weight(
    G: nx.Graph, aggregation_type: Optional[List[str]]
) -> nx.Graph:
    # Calculate MW for each chain
    func = partial(SeqUtils.molecular_weight, seq_type="protein")
    G = compute_feature_over_chains(G, func, feature_name="molecular_weight")

    # Sum MW for all chains
    if aggregation_type is not None:
        G = aggregate_graph_feature_over_chains(
            G,
            feature_name="molecular_weight",
            aggregation_type=aggregation_type,
        )

    return G
"""


def molecular_weight(input, data: Optional = None, seq_type="protein"):
    from Bio import SeqUtils

    func = partial(SeqUtils.molecular_weight, seq_type=seq_type)

    # If a graph is provided, e.g. from a protein graph we compute the function over the chains
    if isinstance(input, nx.Graph):
        G = compute_feature_over_chains(
            input, func, feature_name="molecular_weight"
        )
        return G

    # If a node is provided, e.g. from a PPI graph we extract the sequence and compute the weight
    elif type(input) == str:
        for id in data["uniprot_ids"]:
            print(data)
            data[f"molecular_weight_{id}"] = func(data[f"sequence_{id}"])
        return


def aaindex2(sequence: str, feature_type) -> pd.Series:
    pass


if __name__ == "__main__":
    print(molecular_weight(input="MYTGV"))
