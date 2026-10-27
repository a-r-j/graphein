"""Tests for graphein.protein.tensor.angles."""

import math
import warnings

import numpy as np
import pytest

from graphein.protein.tensor.angles import (
    _dihedral_angle,
    angle_to_unit_circle,
    dihedrals,
    dihedrals_to_rad,
    get_backbone_bond_angles,
    get_backbone_bond_lengths,
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


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_batched_dihedrals_match_single():
    # Three proteins of length 4: torch.cross without dim crossed along the
    # first size-3 dimension (the batch) instead of the coordinates
    torch.manual_seed(0)
    coords = torch.rand(12, 4, 3) * 3
    batch = torch.arange(3).repeat_interleave(4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batched = dihedrals(coords, batch=batch, embed=False)
    single = torch.cat([dihedrals(c, embed=False) for c in coords.split(4)])
    assert torch.allclose(batched, single, atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_backbone_geometry_nan_fill():
    x = torch.rand(3, 4, 3) * 3
    x[1, 0] = float("nan")  # missing N
    lengths = get_backbone_bond_lengths(x)
    # [Ca-C, C-N, N-Ca]; N-Ca is filled with the ideal N-Ca bond length
    assert lengths[1, 2].item() == pytest.approx(1.458)
    assert lengths[0, 1].item() == pytest.approx(1.320)
    assert not lengths.isnan().any()
    angles = get_backbone_bond_angles(x)
    # [C-N-Ca, N-Ca-C, Ca-C-N]
    assert angles[1, 1].item() == pytest.approx(1.941)
    assert not angles.isnan().any()


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
@pytest.mark.parametrize("embed", [False, True])
def test_sidechain_torsion_pads_trailing_residues(embed):
    torch.manual_seed(0)
    coords = torch.rand(4, 37, 3)
    residues = ["SER", "GLY", "ALA", "SEC"]
    angles, mask = sidechain_torsion(
        coords, residues, embed=embed, return_mask=True
    )
    width = 8 if embed else 4
    assert angles.shape == mask.shape == (4, width)
    # GLY, ALA and SEC (no Se atoms with 37 atom types) are padded
    pad = [1.0, 0.0] * 4 if embed else [0.0] * 4
    assert angles[1:].tolist() == [pad] * 3
    assert not mask[1:].any()
    assert mask[0, : 2 if embed else 1].all()
