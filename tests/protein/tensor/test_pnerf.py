"""Tests for graphein.protein.tensor.pnerf."""

import pytest

from graphein.protein.tensor.angles import dihedrals
from graphein.protein.tensor.pnerf import (
    dihedrals_to_coords,
    reconstruct_dihedrals,
)

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_embedded_and_radian_dihedrals_give_same_coords():
    torch.manual_seed(0)
    x = torch.rand(8, 4, 3) * 3
    rad = dihedrals(x, embed=False)
    embedded = dihedrals(x, embed=True)
    assert embedded.shape == (8, 6)

    assert torch.allclose(
        dihedrals_to_coords(embedded, None, None),
        dihedrals_to_coords(rad, None, None),
        atol=1e-5,
    )
    assert torch.allclose(
        reconstruct_dihedrals(embedded, x),
        reconstruct_dihedrals(rad, x),
        atol=1e-5,
    )
