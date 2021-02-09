"""Tests for graphein.protein.graphs"""

from pathlib import Path

import networkx as nx

from graphein.protein.graphs import construct_graph


def test_construct_graph():
    """Example-based test that graph construction works correctly.

    Uses 4hhb PDB file as an example test case.
    """
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    G = construct_graph(pdb_path=str(file_path))
    assert isinstance(G, nx.Graph)
    assert len(G) == 574

    # Check number of peptide bonds
    peptide_bond_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d["kind"] == {"peptide_bond"}
    ]
    assert len(peptide_bond_edges) == 570
