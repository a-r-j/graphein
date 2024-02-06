"""Utility functions to work with graph-level features."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import networkx as nx
import numpy as np
import pandas as pd


def convert_graph_dict_feat_to_series(
    G: nx.Graph, feature_name: str
) -> nx.Graph:
    """
    Takes in a graph and a graph-level ``feature_name``. Converts this feature to a ``pd.Series``.
    This is useful as some features are output as dictionaries and we wish to standardise this.

    :param G:  nx.Graph containing ``G.graph[f"{feature_name}"]`` (``Dict[Any, Any]``).
    :type G: nx.Graph
    :param feature_name: Name of feature to convert to dictionary.
    :type feature_name: str
    :return: nx.Graph containing ``G.graph[f"{feature_name}"]: pd.Series``.
    :rtype: nx.Graph
    """
    G.graph[feature_name] = pd.Series(G.graph[feature_name])
    return G


def aggregate_graph_feature_over_chains(
    G: nx.Graph, feature_name: str, aggregation_type: str
) -> nx.Graph:
    """
    Performs aggregation of a feature over the chains. E.g. sums/averages/min/max molecular weights for each chain.

    :param G: nx.Graph of protein containing chain-specific features.
    :type G: nx.Graph
    :param feature_name: Name of features to aggregate.
    :type feature_name: str
    :param aggregation_type: Type of aggregation to perform (``"min"`, ``"max"``, ``"sum"``, ``"mean"``).
    :type aggregation_type: str
    :raises NameError: If ``aggregation_type`` is not one of ``"min"`, ``"max"``, ``"sum"``, ``"mean"``.
    :return: nx.Graph of protein with a new aggregated feature ``G.graph[f"{feature_name}_{aggregation_type}"]``.
    :rtype: nx.Graph
    """
    if aggregation_type == "mean":
        func = np.mean
    elif aggregation_type == "max":
        func = np.max
    elif aggregation_type == "sum":
        func = np.sum
    elif aggregation_type == "min":
        func = np.min
    else:
        raise NameError(
            "Unsupported aggregation type. Please use mean, max, sum or min."
        )

    G.graph[f"{feature_name}_{aggregation_type}"] = func(
        [G.graph[f"{feature_name}_{c}"] for c in G.graph["chain_ids"]]
    )

    return G
