"""Utilities for manipulating protein geometry."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from multipledispatch import dispatch
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from ..resi_atoms import CHI_ANGLES_ATOMS
from .representation import get_c_alpha, get_full_atom_coords
from .types import (
    AtomTensor,
    CoordTensor,
    DihedralTensor,
    QuaternionTensor,
    RotationTensor,
    TorsionTensor,
)
from .utils import has_nan


def to_ang(a: CoordTensor, b: CoordTensor, c: CoordTensor) -> torch.Tensor:
    """Compute the angle between vectors ``ab`` and ``bc``.

    :param a: Coordinates of point ``a`` ``(L x 3)``.
    :type a: graphein.protein.tensor.types.CoordTensor
    :param b: Coordinates of point ``b`` ``(L x 3)``.
    :type b: graphein.protein.tensor.types.CoordTensor
    :param c: Coordinates of point ``c`` ``(L x 3)``.
    :type c: graphein.protein.tensor.types.CoordTensor
    :return: Angle between vectors ab and bc in radians.
    :rtype: torch.Tensor
    """
    ba = b - a
    bc = b - c
    return torch.acos(
        (ba * bc).sum(dim=1) / (torch.norm(ba, dim=1) * torch.norm(bc, dim=1))
    )


# @torch.jit.script
def get_backbone_bond_lengths(x: AtomTensor) -> torch.Tensor:
    """Compute the bond lengths between atoms."""
    n, a, c = x[:, 0, :], x[:, 1, :], x[:, 2, :]

    n_a = torch.norm(n - a, dim=1)
    a_c = torch.norm(a - c, dim=1)
    c_n = torch.norm(c[:-1, :] - n[1:, :], dim=1)
    # c_n = torch.cat([c_n, torch.tensor([1.3287])])
    c_n = torch.cat([c_n, torch.tensor([1.32], device=x.device)])

    if has_nan(a_c):
        a_c = torch.nan_to_num(a_c, nan=1.523)
    if has_nan(c_n):
        c_n = torch.nan_to_num(c_n, nan=1.320)
    if has_nan(n_a):
        n_a = torch.nan_to_num(n_a, nan=1.329)

    # return torch.stack([n_a, a_c, c_n], dim=1)
    # return torch.stack([n_a, a_c, c_n], dim=1)
    # return torch.stack([c_n, a_c, n_a], dim=1)
    return torch.stack([a_c, c_n, n_a], dim=1)


# @torch.jit.script
def get_backbone_bond_angles(x: AtomTensor) -> torch.Tensor:
    """Compute the bond angles between atoms."""
    n, a, c = x[:, 0, :], x[:, 1, :], x[:, 2, :]

    n_a_c = to_ang(n, a, c)
    a_c_n = to_ang(a[:-1, :], c[:-1, :], n[1:, :])
    c_n_a = to_ang(c[:-1, :], n[1:, :], a[1:, :])

    a_c_n = torch.cat([a_c_n, torch.tensor([2.0], device=x.device)])
    c_n_a = torch.cat([c_n_a, torch.tensor([2.1], device=x.device)])

    if has_nan(c_n_a):
        c_n_a = torch.nan_to_num(c_n_a, nan=2.124)
    if has_nan(n_a_c):
        n_a_c = torch.nan_to_num(n_a_c, nan=1.941)
    if has_nan(a_c_n):
        a_c_n = torch.nan_to_num(a_c_n, nan=2.028)
    # This (below line) works!!, Orig AlQ order
    return torch.stack([c_n_a, n_a_c, a_c_n], dim=1)


def angle_to_unit_circle(x: torch.Tensor) -> torch.Tensor:
    """Encodes an angle in radians to a unit circle.

    I.e. returns a tensor or the form:
    ``[cos(x1), sin(x1), cos(x2), sin(x2), ...]``.

    :param x: Tensor of angles in radians.
    :type x: torch.Tensor
    :return: Tensor of angles encoded on a unit circle.
    :rtype: torch.Tensor
    """
    cosines = torch.cos(x)
    sines = torch.sin(x)
    return rearrange([cosines, sines], "t h w -> h (w t)")


def _dihedral_angle(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """computes dihedral angle between 4 points.

    :param a: First point. Shape: ``(L x 3)``
    :type a: torch.Tensor
    :param b: Second point. Shape: ``(L x 3)``
    :type b: torch.Tensor
    :param c: Third point. Shape: ``(L x 3)``
    :type c: torch.Tensor
    :param d: Fourth point. Shape: ``(L x 3)``
    :type d: torch.Tensor
    :param eps: Jitter value, defaults to 1e-7
    :type eps: float, optional
    :return: Tensor of dihedral angles in radians.
    :rtype: torch.Tensor
    """
    eps = torch.tensor(eps, device=a.device)  # type: ignore

    bc = F.normalize(b - c, dim=2)
    n1 = torch.cross(F.normalize(a - b, dim=2), bc)
    n2 = torch.cross(bc, F.normalize(c - d, dim=2))
    x = (n1 * n2).sum(dim=2)
    x = torch.clamp(x, -1 + eps, 1 - eps)
    x[x.abs() < eps] = eps

    y = (torch.cross(n1, bc) * n2).sum(dim=2)
    return torch.atan2(y, x)


def dihedrals(
    coords: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
    rad: bool = True,
    sparse: bool = True,
    embed: bool = True,
    n_idx: int = 0,
    ca_idx: int = 1,
    c_idx: int = 2,
) -> DihedralTensor:

    length = coords.shape[0]

    if batch is None:
        batch = torch.zeros(length, device=coords.device).long()

    X, mask = to_dense_batch(coords, batch)

    C_curr = X[:, :-1, c_idx, :]
    N_curr = X[:, :-1, n_idx, :]
    Ca_curr = X[:, :-1, ca_idx, :]
    N_next = X[:, 1:, n_idx, :]
    Ca_next = X[:, 1:, ca_idx, :]
    C_next = X[:, 1:, c_idx, :]

    phi = torch.zeros_like(X[:, :, 0, 0], device=coords.device)
    psi = torch.zeros_like(X[:, :, 0, 0], device=coords.device)
    omg = torch.zeros_like(X[:, :, 0, 0], device=coords.device)

    phi[:, 1:] = _dihedral_angle(C_curr, N_next, Ca_next, C_next)
    psi[:, :-1] = _dihedral_angle(N_curr, Ca_curr, C_curr, N_next)
    omg[:, :-1] = _dihedral_angle(Ca_curr, C_curr, N_next, Ca_next)

    angles = torch.stack([phi, psi, omg], dim=2)

    if rad:
        angles = angles * 180 / np.pi

    if sparse:
        angles = angles[mask]

    if embed:
        angles = angle_to_unit_circle(angles)

    return angles


def dihedrals_to_rad(
    x_dihedrals: DihedralTensor,
    concat: bool = False,
) -> Union[DihedralTensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Converts dihedrals to radians.
    """
    phi = torch.atan2(x_dihedrals[:, 1], x_dihedrals[:, 0])
    psi = torch.atan2(x_dihedrals[:, 3], x_dihedrals[:, 2])
    omg = torch.atan2(x_dihedrals[:, 5], x_dihedrals[:, 4])

    return torch.stack([phi, psi, omg], dim=1) if concat else (phi, psi, omg)


def torsion_to_rad(
    x_torsion: TorsionTensor,
    concat: bool = True,
) -> Union[
    TorsionTensor,
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Converts sidechain torsions in
    ``(sin(chi1), sin(ch1), cos(chi2), sin(chi2), ...)`` format to radians.

    :param x_torsion: Torsion tensor of shape ``(L, 8)``.
    :type x_torsion: graphein.protein.tensor.types.TorsionTensor
    :param concat: Whether to concatenate the torsions into a single tensor.
        If ``False``, this function returns a tuple of tensors for chi1-4
        , defaults to ``True``.
    :type concat: bool
    :return: Torsion tensor of shape ``(L, 4)`` if ``concat=True``, otherwise
        a tuple of tensors for chi1-4 in radians.
    :rtype: Union[graphein.protein.tensor.types.TorsionTensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    """
    ch1 = torch.atan2(x_torsion[:, 1], x_torsion[:, 0])
    chi2 = torch.atan2(x_torsion[:, 3], x_torsion[:, 2])
    chi3 = torch.atan2(x_torsion[:, 5], x_torsion[:, 4])
    chi4 = torch.atan2(x_torsion[:, 7], x_torsion[:, 6])

    if concat:
        return torch.stack([ch1, chi2, chi3, chi4], dim=1)

    return ch1, chi2, chi3, chi4


def whole_protein_kabsch(
    A: Union[AtomTensor, CoordTensor],
    B: Union[AtomTensor, CoordTensor],
    ca_only: bool = True,
    fill_value: float = 1e-5,
    return_rot: bool = False,
) -> Union[CoordTensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes registration between two (2D or 3D) point clouds with known
    correspondences using Kabsch algorithm.

    Registration occurs in the zero centered coordinate system, and then
    must be transported back.

    See: https://en.wikipedia.org/wiki/Kabsch_algorithm

    Based on implementation by Guillaume Bouvier (@bougui505):
        https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8

    :param A: Torch tensor of shape ``(N,D)`` -- Point Cloud to Align (source)
    :param B: Torch tensor of shape ``(N,D)`` -- Reference Point Cloud (target)
    :param ca_only: Whether to use only C-alpha atoms for alignment, defaults to
        ``True``. If ``False``, all atoms are used.
    :param fill_value: Value to fill in for missing atoms, defaults to ``1e-5``.
        Only relevant if ``ca_only=False``.
    :return: Torch tensor of shape ``(N,D)`` -- Aligned Point Cloud or Optimal
        rotation and translation. Rotation matrix is of shape ``(D,D)`` and for
        multiplication from the right.
    :rtype: Union[graphein.protein.tensor.types.CoordTensor, Tuple[torch.Tensor, torch.Tensor]]
    """
    if ca_only:
        A = get_c_alpha(A)
        B = get_c_alpha(B)
    else:
        A = get_full_atom_coords(A, fill_value=fill_value)
        B = get_full_atom_coords(B, fill_value=fill_value)
    # Get center of mass
    a_mean = A.mean(dim=0)
    b_mean = B.mean(dim=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)

    # try:
    U, _, V = torch.linalg.svd(H)
    V = V.T

    # Flip
    with torch.no_grad():
        flip = (torch.det(U) * torch.det(V.T)) < 0

    V = V.clone()
    V[flip, -1] *= -1

    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T

    return (R, t.squeeze()) if return_rot else R.mm(A.T).T + t.squeeze()


def extract_torsion_coords(batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a (L*?) x 4 x 3 tensor of the coordinates of the atoms for each
    sidechain torsion angle.

    Also returns a (L*?) x 1 indexing tensor to map back to each residue
    (this is because we have variable numbers of torsion angles per residue).
    """
    if isinstance(batch, Batch):
        res_types = [item for sublist in batch.node_id for item in sublist]
        res_types: List[str] = [
            res.split(":")[1] for res in res_types
        ]  # This is the only ugly part if we're not storing string node IDs
    else:
        res_types: List[str] = [
            res.split(":")[1] for res in batch.node_id
        ]  # This is the only ugly part if we're not storing string node IDs
    res_atoms = []
    idxs = []

    # Iterate over residues and grab indices of the atoms for each Chi angle
    for i, res in enumerate(res_types):
        res_coords = []
        for angle_coord_set in CHI_ANGLES_ATOMS[res]:
            res_coords.append([ATOM_NUMBERING[i] for i in angle_coord_set])
            idxs.append(i)
        res_atoms.append(torch.tensor(res_coords, device=batch.coords.device))

    idxs = torch.tensor(idxs, device=batch.coords.device).long()
    res_atoms = torch.cat(res_atoms).long()  # TODO torch.stack instead of cat

    # Select the coordinates for each chi angle
    coords = torch.take_along_dim(
        batch.atom_tensor[idxs, :, :], dim=1, indices=res_atoms.unsqueeze(-1)
    )
    return idxs, coords


# @torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: RotationTensor) -> QuaternionTensor:
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quaternion_to_matrix(quaternions: QuaternionTensor) -> RotationTensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
