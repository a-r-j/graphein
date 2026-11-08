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


PDB = DATA_PATH.parent / "protein/test_data/4hhb.pdb"


def test_cli_accepts_builtin_config_name(tmp_path):
    """-c accepts a built-in name, and its config is applied."""
    result = CliRunner().invoke(
        main, ["-c", "ca_distance_10", "-p", str(PDB), "-o", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    with open(tmp_path / "4hhb.gpickle", "rb") as f:
        G = pickle.load(f)
    assert any(
        "distance_threshold" in d["kind"] for _, _, d in G.edges(data=True)
    )


def test_cli_requires_path_and_output(tmp_path):
    """Missing -p or -o is a usage error, not a traceback."""
    for args in (["-o", str(tmp_path)], ["-p", str(PDB)]):
        result = CliRunner().invoke(main, ["-c", "ca_distance_10", *args])
        assert result.exit_code == 2, result.output


def test_cli_rejects_empty_config(tmp_path):
    """An empty -c (e.g. an unset shell variable) fails, not defaults."""
    result = CliRunner().invoke(
        main, ["-c", "", "-p", str(PDB), "-o", str(tmp_path)]
    )
    assert result.exit_code == 2, result.output
