"""Tests for graphein.protein.tensor.reconstruction."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import pytest

from graphein.protein.resi_atoms import BB_BUILD_INFO
from graphein.protein.tensor.geometry import kabsch
from graphein.protein.tensor.reconstruction import (
    _get_build_params,
    build_sidechain,
    dist_mat_to_coords,
    place_o_coords,
)

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_dist_mat_to_coords():
    # Test that the distance matrix is recovered from the coordinates, with a
    # small error.
    for _ in range(10):
        coords = torch.rand((10, 3))
        d = torch.cdist(coords, coords)
        X = dist_mat_to_coords(d)
        assert torch.allclose(d, torch.cdist(X, X), atol=1e-4)
        # A distance matrix only determines coordinates up to a reflection
        X_aligned = kabsch(X, coords, allow_reflections=True)
        assert torch.allclose(coords, X_aligned, atol=1e-4)


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_build_sidechain_accepts_sin_cos_torsions():
    torch.manual_seed(0)
    residues = ["SER", "ARG", "PHE", "LYS"]
    bb = torch.rand(len(residues), 4, 3) * 3
    rad = (torch.rand(len(residues), 4) * 2 - 1) * torch.pi
    # (cos(chi1), sin(chi1), cos(chi2), sin(chi2), ...)
    sin_cos = torch.stack([rad.cos(), rad.sin()], dim=-1).flatten(1)

    expected = build_sidechain(bb, residues, rad)
    out = build_sidechain(bb, residues, sin_cos)
    assert out.shape == (len(residues), 37, 3)
    assert torch.allclose(out, expected, atol=1e-4)


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_get_build_params():
    assert _get_build_params("ARG") is _get_build_params("ARG")
    bonds, pbonds, _, _, _ = zip(*_get_build_params("ARG"))
    assert pbonds[0].item() == pytest.approx(BB_BUILD_INFO["BONDLENS"]["n-ca"])
    assert pbonds[1:] == bonds[:-1]


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_place_o_coords():
    # A three-residue chain (a size-3 dimension used to confuse torch.cross)
    coords = torch.rand(3, 3, 3) * 3
    out = place_o_coords(coords)
    assert out.shape == (3, 4, 3)
    assert torch.equal(out[:, :3], coords)
    # O is placed 1.231 A from C
    assert torch.allclose(
        (out[:, 3] - coords[:, 2]).norm(dim=-1), torch.full((3,), 1.231)
    )
