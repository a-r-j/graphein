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

#from graphein.protein.features.utils import aggregate_graph_feature_over_chains


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


def molecular_weight(sequence: str, seq_type="protein") -> pd.Series:
    from Bio import SeqUtils
    return pd.Series(SeqUtils.molecular_weight(sequence, seq_type=seq_type), name="molecular_weight")


def aaindex2(sequence: str, feature_type) -> pd.Series:
    pass


if __name__ == "__main__":
    print(molecular_weight(sequence="MYTGV"))
