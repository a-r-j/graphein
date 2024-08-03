"""Tests for graphein.protein.tensor.reconstruction."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import pytest

from graphein.protein.tensor.geometry import kabsch
from graphein.protein.tensor.reconstruction import dist_mat_to_coords

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
        X_aligned = kabsch(X, coords)
        assert torch.allclose(coords, X_aligned, atol=1e-4)
