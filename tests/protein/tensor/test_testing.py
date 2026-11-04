"""Tests for graphein.protein.tensor.testing."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from pathlib import Path

import pytest

from graphein.protein.graphs import read_pdb_to_dataframe
from graphein.protein.resi_atoms import ATOM_NUMBERING
from graphein.protein.tensor.io import protein_df_to_tensor
from graphein.protein.tensor.sequence import (
    get_atom_indices,
    infer_residue_types,
)
from graphein.protein.tensor.testing import (
    has_complete_backbone,
    has_complete_residue,
    is_complete_structure,
)

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False

PDB_PATH = Path(__file__).resolve().parent.parent / "test_data" / "1lds.pdb"


def _residue(residue: str) -> "torch.Tensor":
    x = torch.full((37, 3), 1e-5)
    for i in get_atom_indices()[residue]:
        x[i] = torch.rand(3) + 1
    return x


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_has_complete_residue():
    x = _residue("SER")
    assert has_complete_residue(x, "SER")
    assert has_complete_residue(x, "S")
    assert not has_complete_residue(x, "ALA")
    x[get_atom_indices()["SER"][-1]] = 1e-5
    assert not has_complete_residue(x, "SER")


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_has_complete_backbone_and_structure():
    x = torch.stack([_residue("ALA"), _residue("GLY")])
    assert has_complete_backbone(x) is True
    assert is_complete_structure(x, "AG")
    assert not is_complete_structure(x, ["ALA", "ALA"])
    x[1, 3] = 1e-5  # Remove O
    assert has_complete_backbone(x) is False


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_complete_structure_and_residue():
    df = read_pdb_to_dataframe(PDB_PATH)
    x = protein_df_to_tensor(df[df["record_name"] == "ATOM"])
    sequence = infer_residue_types(x)
    residues = infer_residue_types(x, return_sequence=False)

    assert is_complete_structure(x, sequence)
    assert is_complete_structure(x, residues)
    assert all(
        has_complete_residue(x[i], res) for i, res in enumerate(sequence)
    )

    # Remove the CA atom of residue 3
    x[3, ATOM_NUMBERING["CA"]] = 1e-5
    assert not has_complete_residue(x[3], sequence[3])
    assert has_complete_residue(x[4], residues[4])
    assert not is_complete_structure(x, sequence)
