"""Tests for graphein.protein.tensor.edges."""

import pytest

import graphein.protein.tensor as gpt
from graphein.protein.tensor.edges import compute_edges

try:
    import torch
    import torch_geometric  # noqa: F401

    PYG_AVAIL = True
except ImportError:
    PYG_AVAIL = False


@pytest.mark.skipif(not PYG_AVAIL, reason="PyTorch Geometric not available")
def test_compute_edges():
    x = torch.rand(20, 3)
    assert compute_edges(x, "knn_4").shape == (2, 20 * 4)
    radius = compute_edges(x, "EPS_0.5")
    assert (torch.cdist(x, x)[radius[0], radius[1]] <= 0.5).all()
    with pytest.raises(ValueError):
        compute_edges(x, "foo_3")


@pytest.mark.skipif(not PYG_AVAIL, reason="PyTorch Geometric not available")
def test_docstring_examples():
    x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    batch = torch.tensor([0, 0, 0, 0])
    # Each point has two neighbours at distance 2 (< 2.5)
    radius = gpt.edges.radius_edges(x, radius=2.5, batch=batch, loop=False)
    assert radius.shape == (2, 8)
    knn = gpt.edges.knn_edges(x, k=2, batch=batch, loop=False)
    assert knn.shape == (2, 8)

    protein = gpt.Protein(coords=torch.rand(10, 37, 3))
    assert protein.edges("knn_3").shape == (2, 30)
    protein.edges("knn_3", x=protein.full_atom_coords()[0], cache="atomic")
    assert protein.atomic.shape == (2, 370 * 3)
