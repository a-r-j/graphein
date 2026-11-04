"""Tests for graphein.protein.tensor.dataset."""

import shutil
from pathlib import Path

import pytest

try:
    import torch

    from graphein.protein.tensor import dataset
    from graphein.protein.tensor.data import Protein

    PYG_AVAIL = True
except ImportError:
    PYG_AVAIL = False

PDB_PATH = Path(__file__).resolve().parent.parent / "test_data" / "1lds.pdb"


@pytest.mark.skipif(not PYG_AVAIL, reason="PyTorch Geometric not available")
def test_process_and_get(tmp_path, monkeypatch, capsys):
    pdb_dir = tmp_path / "pdb"
    pdb_dir.mkdir()
    shutil.copy(PDB_PATH, pdb_dir)
    # Avoid multiprocessing; we only test the dataset bookkeeping
    monkeypatch.setattr(
        dataset,
        "to_protein_mp",
        lambda paths, **kwargs: [
            Protein(coords=torch.rand(4, 37, 3)) for _ in paths
        ],
    )
    ds = dataset.ProteinDataset(
        root=str(tmp_path),
        pdb_dir=str(pdb_dir),
        out_dir=str(tmp_path / "processed"),
        paths=[str(pdb_dir / "1lds.pdb")],
        num_cores=1,
    )
    assert str(pdb_dir) not in capsys.readouterr().out

    assert ds.get("1lds").coords.shape == (4, 37, 3)
    with pytest.raises(FileNotFoundError, match="1lds.pt"):
        ds.get("1ldx")
