"""Tests for graphein.cli"""

import pickle
import tempfile
from pathlib import Path

import networkx as nx
from click.testing import CliRunner

from graphein.cli import main

DATA_PATH = Path(__file__).resolve().parent


def test_cli():
    """Example-based test that graph construction works correctly.

    Uses 4hhb PDB file as an example test case.
    """
    file_path = DATA_PATH.parent / "protein/test_data/4hhb.pdb"
    temp_dir = tempfile.TemporaryDirectory()

    runner = CliRunner()
    result = runner.invoke(main, ["-p", file_path, "-o", Path(temp_dir.name)])
    assert result.exit_code == 0

    with open(Path(temp_dir.name) / f"{file_path.stem}.gpickle", "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.Graph)
    assert len(G) == 574
    # Check number of peptide bonds
    peptide_bond_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d["kind"] == {"peptide_bond"}
    ]
    assert len(peptide_bond_edges) == 570
    temp_dir.cleanup()
