"""Tests for graphein.protein.tensor.geometry."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import numpy as np
import torch

from graphein.protein.tensor.geometry import (
    angle_to_unit_circle,
    dihedrals_to_rad,
    quaternion_to_matrix,
    torsion_to_rad,
    whole_protein_kabsch,
)


def test_torsion_to_rad():
    # Create dummy torsion angles in radians
    angles = (torch.rand(10, 4) - 0.5) * 2 * 2 * np.pi
    angles_emb = angle_to_unit_circle(angles)
    angles_rads = torsion_to_rad(angles_emb)

    delta = torch.abs(angles_rads - angles)

    delta[delta.nonzero()] = torch.abs(delta[torch.nonzero(delta)] - 2 * np.pi)

    delta = ((delta + 2 * np.pi) / np.pi) % 2
    np.testing.assert_allclose(delta, torch.zeros_like(delta), atol=1e-5)


def test_dihedrals_to_rad():
    # Create dummy dihedral angles in radians
    angles = (torch.rand(10, 3) - 0.5) * 2 * 2 * np.pi
    angles_emb = angle_to_unit_circle(angles)
    angles_rads = dihedrals_to_rad(angles_emb, concat=True)

    delta = torch.abs(angles_rads - angles)

    delta[delta.nonzero()] = torch.abs(delta[torch.nonzero(delta)] - 2 * np.pi)

    delta = ((delta + 2 * np.pi) / np.pi) % 2
    np.testing.assert_allclose(delta, torch.zeros_like(delta), atol=1e-5)


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


test_whole_protein_kabsch()
