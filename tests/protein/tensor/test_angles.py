"""Tests for graphein.protein.tensor.angles."""

import math

import numpy as np
import pytest

from graphein.protein.tensor.angles import (
    _dihedral_angle,
    angle_to_unit_circle,
    dihedrals_to_rad,
    get_backbone_bond_angles,
    sidechain_torsion,
    to_ang,
    torsion_to_rad,
)
from graphein.protein.tensor.io import protein_to_pyg

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_to_ang():
    # Test 1: Test angle between two perpendicular vectors
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    c = torch.tensor([0.0, 1.0, 0.0])
    expected_output = torch.tensor(math.pi / 4)
    assert torch.isclose(to_ang(a, b, c), expected_output).all()

    # Test 2: Test angle between two collinear vectors
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    c = torch.tensor([2.0, 0.0, 0.0])
    expected_output = torch.tensor(math.pi)
    assert torch.isclose(to_ang(a, b, c), expected_output).all()

    # Test 3: Test angle between two non-collinear, non-perpendicular vectors
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    c = torch.tensor([1.0, 1.0, 0.0])
    expected_output = torch.tensor(math.pi / 2)
    assert torch.isclose(to_ang(a, b, c), expected_output).all()

    # Test 4: Test angle between three-dimensional vectors
    a = torch.tensor([0.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    c = torch.tensor([1.0, 1.0, 1.0])
    expected_output = torch.tensor(
        math.pi / 2
    )  # torch.tensor(math.atan(math.sqrt(2)))
    assert torch.isclose(to_ang(a, b, c), expected_output).all()


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_angle_to_unit_circle():
    # Test 1: Test encoding of a single angle
    x = torch.tensor([math.pi / 2])
    expected_output = torch.tensor([[0.0, 1.0]])
    assert torch.allclose(angle_to_unit_circle(x), expected_output, atol=1e-7)

    # Test 2: Test encoding of multiple angles
    x = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    print(x.unsqueeze(0).shape)
    expected_output = torch.tensor(
        [[1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]]
    )
    assert torch.allclose(angle_to_unit_circle(x), expected_output, atol=1e-7)


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_torsion_to_rad():
    # Create dummy torsion angles in radians
    angles = (torch.rand(10, 4) - 0.5) * 2 * 2 * np.pi
    angles_emb = angle_to_unit_circle(angles)
    angles_rads = torsion_to_rad(angles_emb)

    delta = torch.abs(angles_rads - angles)

    delta[delta.nonzero()] = torch.abs(delta[torch.nonzero(delta)] - 2 * np.pi)

    delta = ((delta + 2 * np.pi) / np.pi) % 2
    np.testing.assert_allclose(
        delta, torch.zeros_like(delta), atol=1e-3, rtol=1e-3
    )


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_dihedral_angle():
    # Test 1: Test angle between two perpendicular planes
    a = torch.tensor([[1, 0, 0]], dtype=torch.float32)
    b = torch.tensor([[0, 0, 0]], dtype=torch.float32)
    c = torch.tensor([[0, 1, 0]], dtype=torch.float32)
    d = torch.tensor([[0, 0, 1]], dtype=torch.float32)
    expected_output = -torch.tensor([math.pi / 2])

    assert torch.isclose(
        _dihedral_angle(a, b, c, d), expected_output, rtol=1e-4, atol=1e-4
    ).all()

    # Test 2: Test angle between two parallel planes
    a = torch.tensor([[1, 0, 0]], dtype=torch.float32)
    b = torch.tensor([[0, 0, 0]], dtype=torch.float32)
    c = torch.tensor([[0, 0, 0]], dtype=torch.float32)
    d = torch.tensor([[1, 0, 0]], dtype=torch.float32)
    expected_output = torch.tensor([0.0])
    assert torch.isclose(
        _dihedral_angle(a, b, c, d), expected_output, rtol=1e-4, atol=1e-4
    ).all()


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_dihedrals_to_rad():
    # Create dummy dihedral angles in radians
    angles = (torch.rand(10, 3) - 0.5) * 2 * 2 * np.pi
    angles_emb = angle_to_unit_circle(angles)
    angles_rads = dihedrals_to_rad(angles_emb, concat=True)

    delta = torch.abs(angles_rads - angles)

    delta[delta.nonzero()] = torch.abs(delta[torch.nonzero(delta)] - 2 * np.pi)

    delta = ((delta + 2 * np.pi) / np.pi) % 2
    np.testing.assert_allclose(
        delta, torch.zeros_like(delta), atol=1e-4, rtol=1e-4
    )


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_pyl_torsion_angle():
    p = protein_to_pyg(pdb_code="1nth")

    pyl_index = p.residue_id.index("A:PYL:202:")
    sc_angles = sidechain_torsion(p.coords, p.residues)

    torch.testing.assert_close(
        torch.zeros_like(sc_angles[pyl_index]),
        sc_angles[pyl_index],
        rtol=1e-5,
        atol=1e-5,
    )
