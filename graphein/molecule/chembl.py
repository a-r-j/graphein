"""Functions for retrieving molecular data from ChEMBL."""

from typing import Any
from urllib.parse import quote

import networkx as nx
import requests

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

CHEMBL_MOLECULE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/"


def get_smiles_from_chembl(chembl_id: str) -> str:
    """Retrieves a SMILE string from a ChEMBL ID.

    :param chembl_id: ChEMBL ID, e.g., ``'CHEMBL1234'``
    :type smiles: str
    :returns: A SMILE string, e.g.,
        ``'COc1cccc(c1)NC(=O)c2cccnc2'``
    :rtype: str
    """
    data = get_chembl_metadata(chembl_id)
    return data["molecule_structures"]["canonical_smiles"]


def get_chembl_id_from_smiles(smiles: str) -> str:
    """Retrieves a ChEMBL ID from a SMILE string.

    :param smiles: A valid SMILE string, e.g.,
        ``'COc1cccc(c1)NC(=O)c2cccnc2'``
    :type smiles: str
    :return: ChEMBL ID, e.g., ``'CHEMBL1234'``
    :rtype: str
    """
    return get_chembl_metadata(smiles)["molecule_chembl_id"]


def get_chembl_metadata(query: str) -> dict[str, Any]:
    """Retrieves metadata from ChEMBL for a SMILE string or ChEMBL ID.

    :param query: A valid SMILE string or ChEMBL ID, e.g.
        ``'CHEMBL1234'`` or ``'COc1cccc(c1)NC(=O)c2cccnc2'``
    :type query: str
    :raises requests.HTTPError: If ChEMBL has no molecule for the query.
    :return: A dictionary of metadata.
    :rtype: Dict[str, Any]
    """
    # Encode the query: SMILES contain URL characters such as # (C#N)
    response = requests.get(
        CHEMBL_MOLECULE_URL + quote(query, safe=""),
        params={"format": "json"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def add_chembl_metadata(g: nx.Graph) -> nx.Graph:
    """
    Add ChEMBL metadata to the graph.

    :param g: The graph to add metadata to.
    :type g: nx.Graph
    :return: Graph with ChEMBL metadata added.
    :rtype: nx.Graph
    """
    # Get the SMILE
    smiles = g.graph["smiles"]
    g.graph["chembl_metadata"] = get_chembl_metadata(smiles)
    return g
