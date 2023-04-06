from pathlib import Path

import networkx as nx
import pytest
from hypothesis import given
from hypothesis.strategies import text

from graphein.rna.edges import (
    add_all_dotbracket_edges,
    add_base_pairing_interactions,
    add_phosphodiester_bonds,
)
from graphein.rna.graphs import (
    RNA_BASES,
    SUPPORTED_DOTBRACKET_NOTATION,
    construct_graph,
)

TEST_SEQUENCE = "UUGGAGUACACAACCUGUACACUCUUUC"
TEST_DOTBRACKET = "..(((((..(((...)))..)))))..."


def test_construct_rna_graph():
    g = construct_graph(
        dotbracket=TEST_DOTBRACKET,
        sequence=TEST_SEQUENCE,
        edge_construction_funcs=[
            add_base_pairing_interactions,
            add_phosphodiester_bonds,
        ],
    )

    h = construct_graph(
        dotbracket=TEST_DOTBRACKET,
        sequence=TEST_SEQUENCE,
        edge_construction_funcs=[add_all_dotbracket_edges],
    )

    assert g.edges() == h.edges()
    assert g.nodes() == h.nodes()

    # Check number of nodes and edges
    assert len(g.nodes()) == len(TEST_SEQUENCE)
    assert len(g.nodes()) == len(TEST_DOTBRACKET)

    # Check node features are in alphabet
    for n, d in g.nodes(data=True):
        assert d["dotbracket_symbol"] in SUPPORTED_DOTBRACKET_NOTATION
        assert d["nucleotide"] in RNA_BASES

    # Count edges for each type
    phosphodiesters = 0
    bp = 0
    for u, v, d in g.edges(data=True):
        if d["attr"] == "phosphodiester_bond":
            phosphodiesters += 1
        elif d["attr"] == "base_pairing":
            bp += 1

    assert phosphodiesters == len(TEST_SEQUENCE) - 1
    assert bp == TEST_DOTBRACKET.count("(")


def test_pdb_rna_graph():
    g = construct_graph(pdb_code="2jyf")
    assert isinstance(g, nx.Graph)

    for n, d in g.nodes(data=True):
        assert isinstance(n, str)
        assert "coords" in d.keys()
        assert "chain_id" in d.keys()
        assert "residue_name" in d.keys()
        assert "residue_number" in d.keys()
        assert "atom_type" in d.keys()
        assert "b_factor" in d.keys()

    for _, _, d in g.edges(data=True):
        assert "kind" in d.keys()
        assert "bond_length" in d.keys()

    assert "name" in g.graph.keys()
    assert "pdb_code" in g.graph.keys()
    assert "path" in g.graph.keys()
    assert "chain_ids" in g.graph.keys()
    assert "pdb_df" in g.graph.keys()
    assert "raw_pdb_df" in g.graph.keys()
    assert "coords" in g.graph.keys()


def test_construct_graph():
    """Example-based test that graph construction works correctly.
    Uses 4hhb PDB file as an example test case.
    """
    file_path = Path(__file__).parent / "test_data/2jyf.pdb"
    g = construct_graph(path=str(file_path))

    for n, d in g.nodes(data=True):
        assert isinstance(n, str)
        assert "coords" in d.keys()
        assert "chain_id" in d.keys()
        assert "residue_name" in d.keys()
        assert "residue_number" in d.keys()
        assert "atom_type" in d.keys()
        assert "b_factor" in d.keys()

    for _, _, d in g.edges(data=True):
        assert "kind" in d.keys()
        assert "bond_length" in d.keys()

    assert "name" in g.graph.keys()
    assert "pdb_code" in g.graph.keys()
    assert "path" in g.graph.keys()
    assert "chain_ids" in g.graph.keys()
    assert "pdb_df" in g.graph.keys()
    assert "raw_pdb_df" in g.graph.keys()
    assert "coords" in g.graph.keys()
