"""Functions for adding nodes features to a PPI Graph."""

# %%
# Graphein
# Author: Ramon Vinas, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Any
from urllib.parse import quote

import requests

HGNC_FETCH_URL = "https://rest.genenames.org/fetch/symbol/"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/"


def add_uniprot_sequences(d: dict[str, Any], gene_symbol: str):
    """Looks up the UniProt accessions of an HGNC gene symbol and adds them
    (``d["uniprot_ids"]``) and their sequences (``d["sequence_<accession>"]``)
    to ``d``.

    :param d: Graph attribute dictionary.
    :type d: Dict[str, Any]
    :param gene_symbol: HGNC gene symbol, e.g. ``"ZNF3"``.
    :type gene_symbol: str
    :raises requests.HTTPError: If HGNC or UniProt return an error.
    """
    response = requests.get(
        HGNC_FETCH_URL + quote(gene_symbol, safe=""),
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    d["uniprot_ids"] = response.json()["response"]["docs"][0]["uniprot_ids"]

    # Todo these API calls should probably be batched
    for acc in d["uniprot_ids"]:
        response = requests.get(
            UNIPROT_ENTRY_URL + quote(acc, safe=""),
            params={"fields": "sequence", "format": "tsv"},
            timeout=30,
        )
        response.raise_for_status()
        d[f"sequence_{acc}"] = response.text.split("\n")[1]


def add_sequence_to_nodes(n: str, d: dict[str, Any]):
    """
    Maps UniProt ACC to UniProt ID. Retrieves sequence from UniProt and adds it to the node as a feature

    :param n: Graph node.
    :type n: str
    :param d: Graph attribute dictionary.
    :type d: Dict[str, Any]
    """
    add_uniprot_sequences(d, d["protein_id"])
