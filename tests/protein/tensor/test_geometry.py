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
    assert torch.allclose(coords_aligned, coords, atol=1e-6), (
        "Coords differ too much after alignment"
    )


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

    assert torch.mean(torch.abs(ideal_ba)) < torch.mean(torch.abs(true_ba)), (
        "Bond angles not more idealised"
    )
    assert torch.mean(torch.abs(ideal_bl)) < torch.mean(torch.abs(true_bl)), (
        "Bond lengths not more idealised"
    )


MIRROR = torch.diag(torch.tensor([1.0, 1.0, -1.0])) if TORCH_AVAIL else None


def _random_rotation() -> "torch.Tensor":
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    return q * torch.det(q)


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_kabsch_no_reflection():
    torch.manual_seed(0)
    A = torch.randn(20, 3)
    B = A @ MIRROR + 0.1 * torch.randn(20, 3)

    R, t = kabsch(A, B, return_transformed=False)
    assert torch.isclose(torch.det(R), torch.tensor(1.0), atol=1e-5)

    # The returned rotation is at least as good as any sampled rotation
    def rmsd(R):
        return (((R @ (A - A.mean(0)).T).T + B.mean(0) - B) ** 2).sum()

    best_sampled = min(rmsd(_random_rotation()) for _ in range(2000))
    assert rmsd(R) <= best_sampled + 1e-4

    # Reflections are allowed (and so exact) when requested
    R, _ = kabsch(
        A, A @ MIRROR, return_transformed=False, allow_reflections=True
    )
    assert torch.isclose(torch.det(R), torch.tensor(-1.0), atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_kabsch_residue_wise_matches_whole():
    torch.manual_seed(0)

    def centred(x):
        # Put CA (index 1) and the mean at the origin so residue-wise
        # (CA-centred) and whole (mean-centred) alignments are the same problem
        x[:, 1] = 0
        x[:, -1] = -x[:, :-1].sum(dim=1)
        return x

    A = centred(torch.randn(10, 6, 3))
    B = centred(A @ MIRROR + 0.3 * torch.randn(10, 6, 3))

    out = kabsch(A, B, residue_wise=True)
    R, _ = kabsch(A, B, residue_wise=True, return_transformed=False)
    assert torch.allclose(torch.det(R), torch.ones(10), atol=1e-5)
    for i in range(10):
        assert torch.allclose(out[i], kabsch(A[i], B[i]), atol=1e-4)

    R, _ = kabsch(
        A,
        A @ MIRROR,
        residue_wise=True,
        return_transformed=False,
        allow_reflections=True,
    )
    assert torch.allclose(torch.det(R), -torch.ones(10), atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_idealize_backbone_anchors_to_input(monkeypatch):
    import graphein.protein.tensor.geometry as geometry

    # Replace the bond geometry with coordinates so the ideal loss pulls them
    # strongly towards ~2; only the MSE anchor keeps them near the input.
    monkeypatch.setattr(
        geometry, "get_backbone_bond_lengths", lambda x: x[:, :3, 0]
    )
    monkeypatch.setattr(
        geometry, "get_backbone_bond_angles", lambda x: x[:, :3, 1]
    )
    x = torch.full((1, 4, 3), 50.0)
    out = geometry.idealize_backbone(x, lr=1.0, n_iter=300)
    assert torch.equal(x, torch.full((1, 4, 3), 50.0)), "Input was modified"
    assert out[0, :3, :2].min() > 40


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_idealize_backbone_cache():
    protein = gpt.Protein(coords=torch.rand(5, 4, 3) * 5)
    out = protein.idealize_backbone(n_iter=2, cache="ideal")
    assert protein.ideal is out


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_center_protein_keeps_fill_values():
    x = torch.rand(5, 37, 3) + 1
    x[:, 5:] = 1e-5
    centred = center_protein(x)
    assert (centred[:, 5:] == 1e-5).all()
    assert torch.allclose(centred[:, 1].mean(0), torch.zeros(3), atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_apply_structural_noise():
    x = torch.zeros(10, 3)
    noise = gpt.geometry.apply_structural_noise(
        x, magnitude=0.5, gaussian=False, return_transformed=False
    )
    assert noise.shape == x.shape and noise.abs().max() <= 0.5
    assert torch.equal(
        gpt.geometry.apply_structural_noise(x, magnitude=0.0), x
    )
