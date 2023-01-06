"""Tests for graphein.protein.tensor.geometry."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import numpy as np
import pytest

from graphein.protein.tensor.geometry import (
    quaternion_to_matrix,
    whole_protein_kabsch,
)

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_whole_protein_kabsch():
    # Compute the RMSD between the original and aligned coordinates
    def rmsd(x, y):
        return torch.sqrt(torch.mean((x - y) ** 2))

    # Test 2D
    A = torch.rand(10, 2)
    R0 = torch.tensor(
        [[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]],
        dtype=torch.float,
    )
    B = (R0.mm(A.T)).T
    t0 = torch.tensor([3.0, 3.0])
    B += t0
    R, t = whole_protein_kabsch(A, B, return_rot=True)
    A_aligned = (R.mm(A.T)).T + t
    error = rmsd(A_aligned, B)
    assert error < 1e-6, "RMSD too high after alignment"

    # Test 3D
    coords = torch.rand(10, 3)
    # Create a random rotation matrix
    quat = torch.rand(4)
    rot = quaternion_to_matrix(quat)
    # Apply a random rotation and translation to coords
    coords_distorted = rot.mm(coords.T).T + torch.rand(3) * 4
    r, t = whole_protein_kabsch(coords_distorted, coords, return_rot=True)
    coords_aligned = r.mm(coords_distorted.T).T + t

    assert rmsd(coords_aligned, coords) < 1e-6, "RMSD too high after alignment"
    assert torch.allclose(
        coords_aligned, coords, atol=1e-6
    ), "Coords differ too much after alignment"
