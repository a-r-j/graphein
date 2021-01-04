import pandas as pd
import networkx as nx
import numpy as np


def convert_graph_dict_feat_to_series(
    G: nx.Graph, feature_name: str
) -> nx.Graph:
    G.graph[feature_name] = pd.Series(G.graph[feature_name])
    return G


def aggregate_graph_feature_over_chains(
    G: nx.Graph, feature_name: str, aggregation_type: str
) -> nx.Graph:
    if aggregation_type == "mean":
        G.graph[f"{feature_name}_mean"] = np.mean(
            [G.graph[f"{feature_name}_{c}"] for c in G.graph["chain_ids"]]
        )

    if aggregation_type == "max":
        G.graph[f"{feature_name}_max"] = np.max(
            [G.graph[f"{feature_name}_{c}"] for c in G.graph["chain_ids"]]
        )

    if aggregation_type == "sum":
        G.graph[f"{feature_name}_sum"] = np.sum(
            [G.graph[f"{feature_name}_{c}"] for c in G.graph["chain_ids"]]
        )

    return G
