"""Tests for graphein.protein.tensor.data."""

import pytest

from graphein.protein.tensor.data import Protein

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_save_and_load_protein():
    a = Protein().from_pdb_code("4hhb")
    torch.save(a, "4hhb.pt")
    b = torch.load("4hhb.pt", weights_only=False)
    assert a == b
