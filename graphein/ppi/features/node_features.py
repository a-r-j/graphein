"""Functions for adding nodes features to a PPI Graph."""

# %%
# Graphein
# Author: Ramon Vinas, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Any, Dict

from bioservices import HGNC, UniProt
from loguru import logger as log


def add_sequence_to_nodes(n: str, d: Dict[str, Any]):
    """
    Maps UniProt ACC to UniProt ID. Retrieves sequence from UniProt and adds it to the node as a feature

    :param n: Graph node.
    :type n: str
    :param d: Graph attribute dictionary.
    :type d: Dict[str, Any]
    """
    h = HGNC(verbose=False)
    u = UniProt(verbose=False)

    d["uniprot_ids"] = h.fetch("symbol", d["protein_id"])["response"]["docs"][
        0
    ]["uniprot_ids"]

    # Todo these API calls should probably be batched
    # Todo mapping with bioservices to support other protein IDs?

    for id in d["uniprot_ids"]:
        d[f"sequence_{id}"] = u.search(id, columns="sequence").split("\n")[1]
