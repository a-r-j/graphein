import os
from functools import lru_cache, partial
from pathlib import Path

import networkx as nx
import torch

from graphein.protein.features.sequence.utils import (
    compute_feature_over_chains,
)


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


def subset(G, feature_name, feature_value):
    node_list = []
    for n, d in G.nodes(data=True):
        if d["feature_name"] = feature_value:
             node_list.append(n)
    return G.subgraph(node_list)

def esm_residue_embedding(
    G: nx.Graph,
    model_name: str = "esm1b_t33_650M_UR50S",
    output_layer: int = 33,
) -> nx.Graph:

    for chain in G.graph["chain_id"]:
        embedding = compute_esm_embedding(
            G.graph[f"sequence_{chain}"],
            representation="residue",
            model_name=model_name,
            output_layer=output_layer,
        )

        subgraph = subset(G, "chain_id", chain)
        
        for i, n, d in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["esm_embedding"] = embedding[i]

    return G


def esm_sequence_embedding(G: nx.Graph) -> nx.Graph:
    func = partial(compute_esm_embedding, representation="sequence")
    G = compute_feature_over_chains(G, func, feature_name="esm_embedding")

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
            Path(__file__).parent.parent
            / "pretrained_models"
            / "swissprot-reviewed-protvec.model"
        )
    )


def biovec_sequence_embedding(G: nx.Graph) -> nx.Graph:
    pv = _load_biovec_model()
    func = pv.to_vecs
    G = compute_feature_over_chains(G, func, feature_name="biovec_embedding")
    return G
