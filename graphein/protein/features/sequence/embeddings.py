"""Functions to add embeddings from pre-trained language models protein
structure graphs."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import os
from functools import lru_cache, partial
from pathlib import Path

import networkx as nx
import numpy as np
from loguru import logger as log

from graphein.protein.features.sequence.utils import (
    compute_feature_over_chains,
    subset_by_node_feature_value,
)
from graphein.utils.dependencies import import_message

try:
    import torch
except ImportError:
    message = import_message(
        submodule="graphein.protein.features.sequence.embeddings",
        package="torch",
        pip_install=True,
        conda_channel="pytorch",
    )
    log.debug(message)

try:
    import biovec
except ImportError:
    message = import_message(
        submodule="graphein.protein.features.sequence.embeddings",
        package="biovec",
        pip_install=True,
        extras=True,
    )
    log.debug(message)


@lru_cache()
def _load_esm_model(model_name: str = "esm1b_t33_650M_UR50S"):
    """
    Loads pre-trained FAIR ESM model from torch hub.

        *Biological Structure and Function Emerge from Scaling Unsupervised*
        *Learning to 250 Million Protein Sequences* (2019)
        Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth
        and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick,
        C. Lawrence and Ma, Jerry and Fergus, Rob


        *Transformer protein language models are unsupervised structure learners*
        2020 Rao, Roshan M and Meier, Joshua and Sercu, Tom and Ovchinnikov,
        Sergey and Rives, Alexander

    Pre-trained models:
    Full Name layers params Dataset Embedding Dim Model URL
    ========= ====== ====== ======= ============= =========
    ESM-1b   esm1b_t33_650M_UR50S 33 650M UR50/S 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
    ESM1-main esm1_t34_670M_UR50S34 670M UR50/S 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50S.pt
    esm1_t34_670M_UR50D 34 670M UR50/D 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50D.pt
    esm1_t34_670M_UR100 34 670M UR100 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR100.pt
    esm1_t12_85M_UR50S 12 85M UR50/S 768 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t12_85M_UR50S.pt
    esm1_t6_43M_UR50S 6 43M UR50/S 768 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t6_43M_UR50S.pt

    :param model_name: Name of pre-trained model to load
    :type model_name: str
    :return: loaded pre-trained model
    """

    return torch.hub.load("facebookresearch/esm", model_name)


def compute_esm_embedding(
    sequence: str,
    representation: str,
    model_name: str = "esm1b_t33_650M_UR50S",
    output_layer: int = 33,
) -> np.ndarray:
    """
    Computes sequence embedding using Pre-trained ESM model from FAIR

        *Biological Structure and Function Emerge from Scaling Unsupervised*
        *Learning to 250 Million Protein Sequences* (2019)
        Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth
        and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick,
        C. Lawrence and Ma, Jerry and Fergus, Rob

        *Transformer protein language models are unsupervised structure learners*
        2020 Rao, Roshan M and Meier, Joshua and Sercu, Tom and Ovchinnikov,
            Sergey and Rives, Alexander

    Pre-trained models:

    Full Name layers params Dataset Embedding Dim Model URL
    ========= ====== ====== ======= ============= =========
    ESM-1b esm1b_t33_650M_UR50S 33 650M UR50/S 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
    ESM1-main esm1_t34_670M_UR50S 34 670M UR50/S 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50S.pt
    esm1_t34_670M_UR50D 34 670M UR50/D 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50D.pt
    esm1_t34_670M_UR100 34 670M UR100 1280 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR100.pt
    esm1_t12_85M_UR50S 12 85M UR50/S 768 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t12_85M_UR50S.pt
    esm1_t6_43M_UR50S 6 43M UR50/S 768 https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t6_43M_UR50S.pt

    :param sequence: Protein sequence to embed (str)
    :type sequence: str
    :param representation: Type of embedding to extract. ``"residue"`` or
        ``"sequence"``. Sequence-level embeddings are averaged residue
        embeddings
    :type representation: str
    :param model_name: Name of pre-trained model to use
    :type model_name: str
    :param output_layer: integer indicating which layer the output should be
        taken from.
    :type output_layer: int
    :return: embedding (``np.ndarray``)
    :rtype: np.ndarray
    """
    model, alphabet = _load_esm_model(model_name)
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(
            batch_tokens, repr_layers=[output_layer], return_contacts=True
        )
    token_representations = results["representations"][output_layer]

    if representation == "residue":
        return token_representations.numpy()

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first
    # residue is token 1.
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


def esm_residue_embedding(
    G: nx.Graph,
    model_name: str = "esm1b_t33_650M_UR50S",
    output_layer: int = 33,
) -> nx.Graph:
    """
    Computes ESM residue embeddings from a protein sequence and adds the to the
    graph.

        *Biological Structure and Function Emerge from Scaling Unsupervised*
        *Learning to 250 Million Protein Sequences* (2019)
        Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth
        and Lin, Zeming and Liu, Jason and Guo,
        Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus,
        Rob


        *Transformer protein language models are unsupervised structure learners*
        (2020) Rao, Roshan M and Meier, Joshua and Sercu, Tom and Ovchinnikov,
        Sergey and Rives, Alexander

    **Pre-trained models**

    =========                     ====== ====== ======= ============= =========
    Full Name                     layers params Dataset Embedding Dim Model URL
    =========                     ====== ====== ======= ============= =========
    ESM-1b esm1b_t33_650M_UR50S   33    650M   UR50/S     1280        https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
    ESM1-main esm1_t34_670M_UR50S 34    670M   UR50/S     1280        https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50S.pt
    esm1_t34_670M_UR50D           34    670M   UR50/D     1280        https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR50D.pt
    esm1_t34_670M_UR100           34    670M   UR100      1280        https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t34_670M_UR100.pt
    esm1_t12_85M_UR50S            12    85M    UR50/S     768         https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t12_85M_UR50S.pt
    esm1_t6_43M_UR50S             6     43M    UR50/S     768         https://dl.fbaipublicfiles.com/fair-esm/models/esm1_t6_43M_UR50S.pt
    =========                     ====== ====== ======= ============= =========

    :param G: ``nx.Graph`` to add esm embedding to.
    :type G: nx.Graph
    :param model_name: Name of pre-trained model to use.
    :type model_name: str
    :param output_layer: index of output layer in pre-trained model.
    :type output_layer: int
    :return: ``nx.Graph`` with esm embedding feature added to nodes.
    :rtype: nx.Graph
    """

    for chain in G.graph["chain_ids"]:
        embedding = compute_esm_embedding(
            G.graph[f"sequence_{chain}"],
            representation="residue",
            model_name=model_name,
            output_layer=output_layer,
        )
        # remove start and end tokens from per-token residue embeddings
        embedding = embedding[0, 1:-1]
        subgraph = subset_by_node_feature_value(G, "chain_id", chain)

        for i, (n, d) in enumerate(subgraph.nodes(data=True)):
            G.nodes[n]["esm_embedding"] = embedding[i]

    return G


def esm_sequence_embedding(G: nx.Graph) -> nx.Graph:
    """
    Computes ESM sequence embedding feature over chains in a graph.

    :param G: nx.Graph protein structure graph.
    :type G: nx.Graph
    :return: nx.Graph protein structure graph with esm embedding features added
        eg. ``G.graph["esm_embedding_A"]`` for chain A.
    :rtype: nx.Graph
    """
    func = partial(compute_esm_embedding, representation="sequence")
    G = compute_feature_over_chains(G, func, feature_name="esm_embedding")

    return G


@lru_cache()
def _load_biovec_model():
    """Loads pretrained ProtVec Model.

    **Source**

        ProtVec: A Continuous Distributed Representation of Biological Sequences

    Paper: http://arxiv.org/pdf/1503.05140v1.pdf
    """

    return biovec.models.load_protvec(
        os.fspath(
            Path(__file__).parent.parent
            / "pretrained_models"
            / "swissprot-reviewed-protvec.model"
        )
    )


def biovec_sequence_embedding(G: nx.Graph) -> nx.Graph:
    """
    Adds BioVec sequence embedding feature to the graph. Computed over chains.

    **Source**
        ProtVec: A Continuous Distributed Representation of Biological Sequences

    Paper: http://arxiv.org/pdf/1503.05140v1.pdf

    :param G: nx.Graph protein structure graph.
    :type G: nx.Graph
    :return: nx.Graph protein structure graph with biovec embedding added. e.g.
        ``G.graph["biovec_embedding_A"]`` for chain ``A``.
    :rtype: nx.Graph
    """
    pv = _load_biovec_model()
    func = pv.to_vecs
    G = compute_feature_over_chains(G, func, feature_name="biovec_embedding")
    return G
