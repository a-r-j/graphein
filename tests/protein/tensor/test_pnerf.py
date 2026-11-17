"""Tests for graphein.protein.tensor.pnerf."""

import pytest

from graphein.protein.tensor.angles import dihedrals
from graphein.protein.tensor.pnerf import (
    dihedrals_to_coords,
    reconstruct_dihedrals,
    sn_nerf,
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


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_sn_nerf_batched_matches_single():
    torch.manual_seed(0)
    a, b, c = (torch.randn(5, 3) * 3 for _ in range(3))
    chi = torch.randn(5)
    l_cd, theta = torch.tensor(1.5), torch.tensor(1.9)
    l_bc = (c - b).norm(dim=-1, keepdim=True)
    out = sn_nerf(a, b, c, l_cd, theta, chi, l_bc)
    assert out.shape == (5, 3)
    single = torch.stack(
        [
            sn_nerf(a[i], b[i], c[i], l_cd, theta, chi[i], l_bc[i])
            for i in range(5)
        ]
    )
    assert torch.allclose(out, single, atol=1e-6)
    # d is l_cd from c, at angle theta to b
    assert torch.allclose((out - c).norm(dim=-1), torch.full((5,), 1.5))
    with pytest.raises(ValueError):
        sn_nerf(a, b, c, l_cd, torch.tensor(4.0), chi, l_bc)
