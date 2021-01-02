"""Functions for working with Protein Structure Graphs"""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd
import torch


@lru_cache()
def _load_esm_model():
    import torch

    return torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")


def compute_esm_embedding(sequence: str, representation: str):
    model, alphabet = _load_esm_model()
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    if representation == "residue":
        return token_representations.numpy()
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


def convert_graph_dict_feat_to_series(G: nx.Graph, feature_name: str) -> nx.Graph:
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


if __name__ == "__main__":
    print(compute_esm_embedding("MYGVYMK", representation="sequence"))
