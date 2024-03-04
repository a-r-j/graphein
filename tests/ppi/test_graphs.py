"""Tests for graphein.ppi.graphs"""

import bioservices
import pytest

from graphein.ppi.config import PPIGraphConfig
from graphein.ppi.edges import add_biogrid_edges, add_string_edges
from graphein.ppi.features.node_features import add_sequence_to_nodes
from graphein.ppi.graphs import compute_ppi_graph

PROTEIN_LIST = [
    "CDC42",
    "CDK1",
    "KIF23",
    "PLK1",
    "RAC2",
    "RACGAP1",
    "RHOA",
    "RHOB",
]


@pytest.mark.skip("not a test")
def hgnc_available():
    try:
        a = bioservices.HGNC()
        return True
    except TypeError:
        return False


HGNC_AVAILABLE = hgnc_available()


# Test Graph Construction
@pytest.mark.skipif(not HGNC_AVAILABLE, reason="HGNC not available")
def test_construct_graph():
    config = PPIGraphConfig()

    g = compute_ppi_graph(
        protein_list=PROTEIN_LIST,
        edge_construction_funcs=[add_biogrid_edges, add_string_edges],
        node_annotation_funcs=[add_sequence_to_nodes],
        config=config,
    )

    # Check nodes and edges
    assert len(g.nodes()) == 8
    assert len(g.edges()) == 21

    # Check edge types are from string/biogrid
    # Check nodes are in our list
    for u, v, d in g.edges(data=True):
        assert d["kind"].issubset({"string", "biogrid"})
        assert u in PROTEIN_LIST
        assert v in PROTEIN_LIST

    # Check sequence is defined if UniProt ID found
    for _, d in g.nodes(data=True):
        assert d["protein_id"] in PROTEIN_LIST
        if d["uniprot_ids"] is not None:
            for id in d["uniprot_ids"]:
                assert d[f"sequence_{id}"] is not None
