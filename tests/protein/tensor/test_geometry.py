import numpy as np
import torch

from graphein.protein.tensor.geometry import (
    angle_to_unit_circle,
    dihedrals_to_rad,
    torsion_to_rad,
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


test_torsion_to_rad()
test_dihedrals_to_rad()
