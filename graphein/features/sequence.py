# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import os
import networkx as nx
from pathlib import Path
from functools import lru_cache
from graphein.features.utils import compute_esm_embedding
from Bio import SeqUtils


def esm_sequence_embedding(G: nx.Graph) -> nx.Graph:
    for c in G.graph["chain_ids"]:
        G.graph[f"esm_embedding_{c}"] = compute_esm_embedding(
            G.graph[f"sequence_{c}"], representation="sequence"
        )
    return G


def esm_residue_embedding(G: nx.Graph) -> nx.Graph:
    # todo iterate over sequences
    for c in G.graph["chain_id"]:
        embedding = compute_esm_embedding(
            G.graph["sequence"], representation="residue"
        )
        print(embedding)
        for i, n, d in enumerate(G.nodes(data=True)):
            n["esm_embedding"] = embedding[i]
    return G


@lru_cache()
def _load_biovec_model():
    """Loads pretrained ProtVec Model

    Source: ProtVec: A Continuous Distributed Representation of Biological Sequences
    Paper: http://arxiv.org/pdf/1503.05140v1.pdf
    """
    import biovec

    return biovec.models.load_protvec(
        os.fspath(
            Path(__file__).parent
            / "pretrained_models"
            / "swissprot-reviewed-protvec.model"
        )
    )


def biovec_sequence_embedding(G: nx.Graph) -> nx.Graph:
    pv = _load_biovec_model()
    for c in G.graph["chain_ids"]:
        G.graph[f"biovec_embedding_{c}"] = pv.to_vecs(G.graph[f"sequence_{c}"])

    return G


def molecular_weight(G: nx.Graph, total: bool = True) -> nx.Graph:
    # Calculate MW for each chain
    for c in G.graph["chain_ids"]:
        G.graph[f"molecular_weight_{c}"] = SeqUtils.molecular_weight(
            G.graph[f"sequence_{c}"], "protein"
        )

    # Sum MW for all chains
    if total:
        G.graph["molecular_weight_total"] = sum(
            G.graph[f"molecular_weight_{c}"] for c in G.graph["chain_ids"]
        )
    return G


if __name__ == "__main__":
    print(len(biovec_sequence_embedding(nx.Graph())))
