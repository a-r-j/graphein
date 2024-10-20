"""Functions for retrieving molecular data from ChEMBL."""

from typing import Any, Dict

import networkx as nx

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from bioservices import ChEMBL

from graphein.utils.dependencies import requires_python_libs


@requires_python_libs("bioservices")
def get_smiles_from_chembl(chembl_id: str) -> str:
    """Retrieves a SMILE string from a ChEMBL ID.

    :param chembl_id: ChEMBL ID, e.g., ``'CHEMBL1234'``
    :type smiles: str
    :returns: A SMILE string, e.g.,
        ``'COc1cccc(c1)NC(=O)c2cccnc2'``
    :rtype: str
    """
    chembl = ChEMBL()
    data = chembl.get_molecule(chembl_id)
    return data["molecule_structures"]["canonical_smiles"]


@requires_python_libs("bioservices")
def get_chembl_id_from_smiles(smiles: str) -> str:
    """Retrieves a ChEMBL ID from a SMILE string.

    :param smiles: A valid SMILE string, e.g.,
        ``'COc1cccc(c1)NC(=O)c2cccnc2'``
    :type smiles: str
    :return: ChEMBL ID, e.g., ``'CHEMBL1234'``
    :rtype: str
    """
    chembl = ChEMBL()
    data = chembl.get_molecule(smiles)
    return data["molecule_chembl_id"]


def get_chembl_metadata(query: str) -> Dict[str, Any]:
    """Retrieves metadata from ChEMBL for a SMILE string or ChEMBL ID.

    :param query: A valid SMILE string or ChEMBL ID, e.g.
        ``'CHEMBL1234'`` or ``'COc1cccc(c1)NC(=O)c2cccnc2'``
    :type query: str
    :return: A dictionary of metadata.
    :rtype: Dict[str, Any]
    """
    chembl = ChEMBL()
    return chembl.get_molecule(query)


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
