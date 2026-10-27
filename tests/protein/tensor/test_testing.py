"""Tests for graphein.protein.tensor.testing."""

import pytest

from graphein.protein.tensor.sequence import get_atom_indices
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
