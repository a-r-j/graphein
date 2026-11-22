"""Tests for graphein.ppi.features.node_features (queries HGNC and UniProt)."""

from graphein.grn.features.node_features import (
    add_sequence_to_nodes as add_grn_sequence,
)
from graphein.ppi.features.node_features import add_sequence_to_nodes

# UniProt P17036 (ZNF3_HUMAN), 446 residues
ZNF3_START = "METQADLVSQEPQALLDSALPSKVPAFSDKDSLGDEMLAAALLKAKSQELVTFEDVAVYFIR"


def test_add_sequence_to_nodes():
    d = {"protein_id": "ZNF3"}
    add_sequence_to_nodes("ZNF3", d)
    assert d["uniprot_ids"] == ["P17036"]
    assert d["sequence_P17036"].startswith(ZNF3_START)
    assert len(d["sequence_P17036"]) == 446


def test_grn_add_sequence_to_nodes():
    d = {"gene_id": "ZNF3"}
    add_grn_sequence("ZNF3", d)
    assert d["uniprot_ids"] == ["P17036"]
    assert len(d["sequence_P17036"]) == 446
