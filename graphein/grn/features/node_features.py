"""Node featurisation utilities for Gene Regulatory Networks."""

from typing import Any

from graphein.ppi.features.node_features import add_uniprot_sequences


def add_sequence_to_nodes(n: str, d: dict[str, Any]):
    """
    Maps UniProt ACC to UniProt ID. Retrieves sequence from UniProt and adds
    it to the node

    :param n: Graph node. Unused (retained for a consistent function signature).
    :type n: str
    :param d: Graph attribute dictionary.
    :type d: Dict[str, Any]
    """
    add_uniprot_sequences(d, d["gene_id"])
