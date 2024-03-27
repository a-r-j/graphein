"""Tests for graphein.protein.tensor.geometry."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import numpy as np
import pytest

import graphein.protein.tensor as gpt
from graphein.protein.tensor.angles import (
    get_backbone_bond_angles,
    get_backbone_bond_lengths,
)
from graphein.protein.tensor.geometry import (
    IDEAL_BB_BOND_ANGLES,
    IDEAL_BB_BOND_LENGTHS,
    center_protein,
    kabsch,
    quaternion_to_matrix,
)

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False


def test_center_protein():
    x = torch.randn((5, 3))
    x_centered = center_protein(x)
    assert torch.allclose(x_centered.mean(dim=0), torch.zeros(3), atol=1e-5)


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
    R, t = kabsch(A, B, return_transformed=False)
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
    r, t = kabsch(coords_distorted, coords, return_transformed=False)
    coords_aligned = r.mm(coords_distorted.T).T + t

    assert rmsd(coords_aligned, coords) < 1e-6, "RMSD too high after alignment"
    assert torch.allclose(
        coords_aligned, coords, atol=1e-6
    ), "Coords differ too much after alignment"


def test_idealise_backbone():
    protein = gpt.Protein().from_pdb_code(pdb_code="5caj", chain_selection="A")
    protein.coords = protein.coords[:20, :, :]

    ideal = protein.idealize_backbone(n_iter=100)

    IDEAL_BL = torch.tensor(
        IDEAL_BB_BOND_LENGTHS, device=protein.coords.device
    )
    IDEAL_BA = torch.tensor(IDEAL_BB_BOND_ANGLES, device=protein.coords.device)

    ideal_bl = IDEAL_BL - get_backbone_bond_lengths(ideal[:, :4, :])
    true_bl = IDEAL_BL - get_backbone_bond_lengths(protein.coords[:, :4, :])

    ideal_ba = IDEAL_BA - get_backbone_bond_angles(ideal[:, :4, :])
    true_ba = IDEAL_BA - get_backbone_bond_angles(protein.coords[:, :4, :])

    assert torch.mean(torch.abs(ideal_ba)) < torch.mean(
        torch.abs(true_ba)
    ), "Bond angles not more idealised"
    assert torch.mean(torch.abs(ideal_bl)) < torch.mean(
        torch.abs(true_bl)
    ), "Bond lengths not more idealised"
