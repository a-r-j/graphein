"""Tests for graphein.protein_ligand.graphs"""

from functools import partial
from pathlib import Path

import networkx as nx
import pytest

from graphein.protein_ligand.config import DSSPConfig, ProteinLigandGraphConfig
from graphein.protein_ligand.graphs import (
    construct_graph,
    read_pdb_to_dataframe,
)

DATA_PATH = Path(__file__).resolve().parent / "test_data" / "3v1w.pdb"

# Example-based Graph Construction test
def test_construct_graph():
    """Example-based test that graph construction works correctly.

    Uses 3v1w PDB file as an example test case.
    """
    file_path = Path(__file__).parent / "test_data" / "3v1w.pdb"
    G = construct_graph(pdb_path=str(file_path))
    print (G)
    assert isinstance(G, nx.Graph)
    assert len(G) == 574

    # Check number of peptide bonds
    peptide_bond_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d["kind"] == {"peptide_bond"}
    ]
    assert len(peptide_bond_edges) == 570

test_construct_graph()