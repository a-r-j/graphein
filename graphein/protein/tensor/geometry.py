"""Utilities for manipulating protein geometry."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import List, Tuple, Union

from loguru import logger as log

from graphein.utils.utils import import_message

from ..resi_atoms import CHI_ANGLES_ATOMS
from .representation import get_c_alpha, get_full_atom_coords
from .types import AtomTensor, CoordTensor, QuaternionTensor, RotationTensor

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    message = import_message(
        "graphein.protein.tensor.geometry",
        package="torch",
        conda_channel="pytorch",
        pip_install=True,
    )
    log.warning(message)

try:
    from torch_geometric.data import Batch
except ImportError:
    message = import_message(
        "graphein.protein.tensor.geometry",
        package="torch_geometric",
        conda_channel="pyg",
        pip_install=True,
    )
    log.warning(message)


def get_center(
    x: Union[AtomTensor, CoordTensor],
    ca_only: bool = True,
    fill_value: float = 1e-5,
) -> CoordTensor:
    """
    Returns the center of a protein.

    :param x: Point Cloud to Center. Torch tensor of shape ``(Length , 3)`` or
        ``(Length, num atoms, 3)``.
    :param ca_only: If ``True``, only the C-alpha atoms will be used to compute
        the center. Only relevant with AtomTensor inputs. Default is ``False``.
    :type ca_only: bool
    :param fill_value: Value used to denote missing atoms. Default is 1e-5.
    :type fill_value: float
    :return: Torch tensor of shape ``(N,D)`` -- Center of Point Cloud
    :rtype: Union[graphein.protein.tensor.types.AtomTensor, graphein.protein.tensor.types.CoordTensor]
    """
    if x.ndim != 3:
        return x.mean(dim=0)
    if ca_only:
        return get_c_alpha(x).mean(dim=0)

    x_flat, _, _ = get_full_atom_coords(x, fill_value=fill_value)
    return x_flat.mean(dim=0)


def center_protein(
    x: Union[AtomTensor, CoordTensor], ca_only: bool = True, fill_value=1e-5
) -> Union[AtomTensor, CoordTensor]:
    """
    Centers a protein in the coordinate system.

    :param x: Point Cloud to Center. Torch tensor of shape ``(Length , 3)`` or
        ``(Length, num atoms, 3)``.
    :param ca_only: If ``True``, only the C-alpha atoms will be used to compute
        the center. Only relevant with AtomTensor inputs. Default is ``False``.
    :type ca_only: bool
    :param fill_value: Value used to denote missing atoms. Default is 1e-5.
    :type fill_value: float
    :return: Centered Point Cloud of same shape as input.
    :rtype: Union[graphein.protein.tensor.types.AtomTensor, graphein.protein.tensor.types.CoordTensor]
    """
    center = get_center(x, ca_only=ca_only, fill_value=fill_value)
    # Mask missing atoms
    fill_mask = torch.where(
        x == fill_value, torch.tensor(1.0), torch.tensor(0.0)
    )
    centered = x - center
    # Restore fill values
    centered[fill_mask] = fill_value
    return centered


def whole_protein_kabsch(
    A: Union[AtomTensor, CoordTensor],
    B: Union[AtomTensor, CoordTensor],
    ca_only: bool = True,
    fill_value: float = 1e-5,
    return_transformed: bool = True,
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
    # Get center of mass
    centroid_A = get_center(A, ca_only=ca_only, fill_value=fill_value)
    centroid_B = get_center(B, ca_only=ca_only, fill_value=fill_value)

    AA = A - centroid_A
    BB = B - centroid_B

    # Covariance matrix
    H = AA.T @ BB
    U, _, Vt = torch.svd(H)

    # if (torch.det(U) * torch.det(Vt.T)) < 0.0:
    #    print("Flipping!")
    # Vt[:,-1] *= -1

    R = Vt @ U.T
    t = centroid_B - R @ centroid_A

    return R.mm(A.T).T + t.T if return_transformed else (R, t)


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
