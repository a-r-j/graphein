"""Tests for graphein.protein.folding_utils"""

import pytest

pytest.importorskip("torch")

from graphein.protein import folding_utils  # noqa: E402


@pytest.fixture
def runs(monkeypatch):
    """Record subprocess.run calls instead of running the tools."""
    calls = []
    monkeypatch.setattr(folding_utils, "is_tool", lambda *a, **k: True)
    monkeypatch.setattr(
        folding_utils.subprocess,
        "run",
        lambda cmd, **kwargs: calls.append((cmd, kwargs)),
    )
    return calls


def test_esmfold_fasta_passes_paths_as_arguments(tmp_path, runs):
    """Paths with shell metacharacters reach esm-fold as one argument."""
    fasta = tmp_path / "seqs; touch pwned.fasta"
    fasta.write_text(">a\nAAAA\n")
    out_dir = tmp_path / "out dir"
    folding_utils.esmfold_fasta(
        str(fasta), str(out_dir), cpu_only=True, cpu_offload=True
    )
    [(cmd, kwargs)] = runs
    assert cmd == [
        "esm-fold",
        "--fasta",
        str(fasta),
        "-o",
        str(out_dir),
        "--num-recycles",
        "4",
        "--cpu-only",
        "--cpu-offload",
    ]
    assert not kwargs.get("shell")


def test_esm_embed_fasta_passes_paths_as_arguments(tmp_path, runs):
    fasta = tmp_path / "seqs; touch pwned.fasta"
    fasta.write_text(">a\nAAAA\n")
    folding_utils.esm_embed_fasta(str(fasta), str(tmp_path / "out dir"))
    [(cmd, kwargs)] = runs
    assert cmd[:4] == [
        "esm-extract",
        "esm2_t33_650M_UR50D",
        str(fasta),
        str(tmp_path / "out dir"),
    ]
    assert cmd[4:] == [
        "--repr_layers",
        "0",
        "32",
        "33",
        "--include",
        "mean",
        "per_tok",
    ]
    assert not kwargs.get("shell")


def test_foldcompress_database_passes_paths_as_arguments(tmp_path, runs):
    db_dir = tmp_path / "pdbs; touch pwned"
    db_dir.mkdir()
    folding_utils.foldcompress_database("my db", str(db_dir), ["a", "b"])
    (compress, _), (subdb, _) = runs
    assert compress == ["foldcomp", "compress", str(db_dir), "my db", "--db"]
    assert subdb[-3:] == [
        str(db_dir / "my dbid_list.txt"),
        "my db",
        "my db_selection",
    ]
