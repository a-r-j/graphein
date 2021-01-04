from typing import Callable, Optional, List

import networkx as nx
import numpy as np


def compute_feature_over_chains(
    G: nx.Graph, func: Callable, feature_name: str
) -> nx.Graph:
    for c in G.graph["chain_ids"]:
        G.graph[f"{feature_name}_{c}"] = func(G.graph[f"sequence_{c}"])
        """
        feat = func(G.graph[f"sequence_{c}"])

        if out_type == "series":
            feat = pd.Series(feat)
        elif out_type == "np":
            raise NotImplementedError

        G.graph[f"{feature_name}_{c}"] = feat
        """
    return G


def aggregate_feature_over_chains(G: nx.Graph, feature_name: str, aggregation_type: str) -> nx.Graph:

    if aggregation_type == "max":
        func = np.max
    elif aggregation_type == "min":
        func = np.min
    elif aggregation_type == "mean":
        func = np.mean
    elif aggregation_type == "sum":
        func = np.sum
    else:
        raise ValueError(f"Unsupported aggregator: {aggregation_type}. Please use min, max, mean, sum")

    G.graph[f"{feature_name}_{aggregation_type}"] = func([G.graph[f"{feature_name}_{c}"]] for c in G.graph["chain_ids"])
    return G


def sequence_to_ngram(sequence: str, N: int) -> List[str]:
    return [sequence[i:i+N] for i in range(len(sequence)-N+1)]

