"""Node featurisation utilities for Gene Regulatory Networks."""

from typing import Any, Dict

from bioservices import HGNC, UniProt
from loguru import logger as log


def add_sequence_to_nodes(n: str, d: Dict[str, Any]):
    """
    Maps UniProt ACC to UniProt ID. Retrieves sequence from UniProt and adds
    it to the node

    :param n: Graph node. Unused (retained for a consistent function signature).
    :type n: str
    :param d: Graph attribute dictionary.
    :type d: Dict[str, Any]
    """

    h = HGNC(verbose=False)
    u = UniProt(verbose=False)

    d["uniprot_ids"] = h.fetch("symbol", d["gene_id"])["response"]["docs"][0][
        "uniprot_ids"
    ]

    # Todo these API calls should probably be batched
    # Todo mapping with bioservices to support other protein IDs?

    for id in d["uniprot_ids"]:
        d[f"sequence_{id}"] = u.search(id, columns="sequence").split("\n")[1]
