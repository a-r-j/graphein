from typing import Any, Callable, List

import networkx as nx
import numpy as np


def compute_feature_over_chains(
    G: nx.Graph, func: Callable, feature_name: str
) -> nx.Graph:
    """
    Computes a sequence featurisation function over the chains in a graph
    :param G: nx.Graph protein structure graph to featurise
    :param func: Sequence featurisation function
    :param feature_name: name of added feature
    :return: Graph with added features of the form G.graph[f"{feature_name}_{chain_id}"]
    """
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


def aggregate_feature_over_chains(
    G: nx.Graph, feature_name: str, aggregation_type: str
) -> nx.Graph:
    """
    Performs aggregation of a given feature over chains in a graph to produce an aggregated value
    :param G: nx.Graph protein structure graph
    :param feature_name: Name of feature to aggregate
    :param aggregation_type: Type of aggregation to perform (min/max/mean/sum)
    :return: Graph with new aggregated feature
    """

    if aggregation_type == "max":
        func = np.max
    elif aggregation_type == "min":
        func = np.min
    elif aggregation_type == "mean":
        func = np.mean
    elif aggregation_type == "sum":
        func = np.sum
    else:
        raise ValueError(
            f"Unsupported aggregator: {aggregation_type}. Please use min, max, mean, sum"
        )

    G.graph[f"{feature_name}_{aggregation_type}"] = func(
        [G.graph[f"{feature_name}_{c}"] for c in G.graph["chain_ids"]]
    )
    return G


def sequence_to_ngram(sequence: str, N: int) -> List[str]:
    """
    Chops a sequence into overlapping N-grams (substrings of length N)
    :param sequence: str Sequence to convert to N-garm
    :param N: Length ofN-grams (int)
    :return: List of n-grams
    """
    return [sequence[i : i + N] for i in range(len(sequence) - N + 1)]


def subset_by_node_feature_value(
    G: nx.Graph, feature_name: str, feature_value: Any
) -> nx.Graph:
    """
    Extracts a subgraph from a protein structure graph based on nodes with a certain feature value
    :param G: nx.Graph protein structure graph to extract a subgraph from
    :param feature_name: Name of feature to base subgraph extraction from
    :param feature_value: Value of feature to select
    :return: Subgraph of G based on nodes with a given feature value
    """
    node_list = []
    for n, d in G.nodes(data=True):

        if d[feature_name] == feature_value:
            node_list.append(n)

    return G.subgraph(node_list)
