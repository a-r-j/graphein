import networkx as nx
import numpy as np
import pandas as pd


def convert_graph_dict_feat_to_series(
    G: nx.Graph, feature_name: str
) -> nx.Graph:
    """
    Takes in a graph and a graph-level feature_name. Converts this feature to a pd.Series.
    This is useful as some features are output as dictionaries and we wish to standardise this.
    :param G:  nx.Graph containing G.graph[f"{feature_name}"]: Dict[Any, Any]
    :param feature_name: Name of feature to convert to dictionary
    :return: nx.Graph containing G.graph[f"{feature_name}"]: pd.Series
    """
    G.graph[feature_name] = pd.Series(G.graph[feature_name])
    return G


def aggregate_graph_feature_over_chains(
    G: nx.Graph, feature_name: str, aggregation_type: str
) -> nx.Graph:
    """
    Performs aggregation of a feature over the chains. E.g. sums/averages/min/max molecular weights for each chain
    :param G: nx.Graph of protein containing chain-specific features
    :param feature_name: name of features to aggregate
    :param aggregation_type: Type of aggregation to perform (min/max/sum/mean)
    :return: nx.Graph of protein with a new aggregated feature G.graph[f"{feature_name}_{aggregation_type}"]
    """
    if aggregation_type == "mean":
        func = np.mean
    elif aggregation_type == "max":
        func = np.max
    elif aggregation_type == "sum":
        func = np.sum
    elif aggregation_type == "min":
        func = np.sum
    else:
        raise NameError(
            "Unsupported aggregation type. Please use mean, max, sum or min"
        )

    G.graph[f"{feature_name}_{aggregation_type}"] = func(
        [G.graph[f"{feature_name}_{c}"] for c in G.graph["chain_ids"]]
    )

    return G
