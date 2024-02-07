"""
pNeRF algorithm for parallelized conversion from torsion (dihedral) angles to
Cartesian coordinates implemented with PyTorch.

Reference implementation in tensorflow by Mohammed AlQuraishi:
    https://github.com/aqlaboratory/pnerf/blob/master/pnerf.py
Paper (preprint) by Mohammed AlQuraishi:
    https://www.biorxiv.org/content/early/2018/08/06/385450

PyTorch implementation modifield from the implementation by Felix Opolka:
    https://github.com/FelixOpolka/pnerf-pytorch
"""

import collections
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .angles import (
    dihedrals_to_rad,
    get_backbone_bond_angles,
    get_backbone_bond_lengths,
)
from .types import AtomTensor, CoordTensor, DihedralTensor

# Constants
NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3

BOND_LENGTHS = np.array([1.458, 1.523, 1.320], dtype=np.float32)
BOND_ANGLES = np.array([2.124, 1.941, 2.028], dtype=np.float32)


def reconstruct_dihedrals(
    dihedrals: DihedralTensor, ground_truth: AtomTensor
) -> AtomTensor:
    """
    Reconstructs backbone from dihedral using bond lengths and angles from
    ground truth.

    """
    if dihedrals.shape[-1] == 6:
        phi, psi, omg = dihedrals_to_rad(dihedrals)
        # dihedrals = torch.stack([omg, phi, psi], dim=-1)  # This works
        dihedrals = torch.stack([phi, psi, omg], dim=-1)

    bond_lengths = get_backbone_bond_lengths(ground_truth)
    bond_angles = get_backbone_bond_angles(ground_truth)
    bond_angles = torch.index_select(
        bond_angles, 1, torch.tensor([1, 2, 0], device=dihedrals.device)
    )
    return dihedrals_to_coords(dihedrals, bond_lengths, bond_angles)


def dihedrals_to_coords(
    dihedral: torch.Tensor,
    bond_lengths: Optional[torch.Tensor],
    bond_angles: Optional[torch.Tensor],
) -> AtomTensor:
    if dihedral.shape[-1] == 6:
        phi, psi, omg = dihedrals_to_rad(dihedral)
        # dihedrals = torch.stack([omg, phi, psi], dim=-1)  # This works
        dihedral = torch.stack([phi, psi, omg], dim=-1)

    if bond_lengths is None:
        pts = dihedral_to_point_fixed(dihedral.unsqueeze(1))
    else:
        pts = dihedral_to_point_variable(
            dihedral.unsqueeze(1), bond_lengths, bond_angles
        )

    return point_to_coordinate(pts)


def dihedral_to_point_fixed(
    dihedral: torch.Tensor,
    bond_lengths: np.ndarray = BOND_LENGTHS,
    bond_angles: np.ndarray = BOND_ANGLES,
) -> torch.Tensor:
    """
    Takes triplets of dihedral angles (phi, psi, omega) and returns 3D points
    ready for use in reconstruction of coordinates. Bond lengths and angles
    are based on idealized averages.

    :param dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    :return: Tensor containing points of the protein's backbone atoms.
        Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """
    num_steps = dihedral.shape[0]
    batch_size = dihedral.shape[1]

    r_cos_theta = torch.tensor(
        bond_lengths * np.cos(np.pi - bond_angles), device=dihedral.device
    )
    r_sin_theta = torch.tensor(
        bond_lengths * np.sin(np.pi - bond_angles), device=dihedral.device
    )

    # point_x = r_cos_theta.view(1, 1, -1).repeat(num_steps, batch_size, 1)
    point_x = r_cos_theta.unsqueeze(1)
    point_y = torch.cos(dihedral) * r_sin_theta
    point_z = torch.sin(dihedral) * r_sin_theta

    # print(point_x.shape)
    point = torch.stack([point_x, point_y, point_z])
    # print(point.shape)
    point_perm = point.permute(1, 3, 2, 0)
    return point_perm.contiguous().view(
        num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS
    )


def dihedral_to_point_variable(
    dihedral: torch.Tensor,
    bond_lengths: torch.Tensor,
    bond_angles: torch.Tensor,
) -> torch.Tensor:
    """
    Takes triplets of dihedral angles (phi, psi, omega) and returns 3D points
    ready for use in reconstruction of coordinates. Bond lengths and angles
    are based on idealized averages.

    :param dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    :return: Tensor containing points of the protein's backbone atoms.
        Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """
    num_steps = dihedral.shape[0]
    batch_size = dihedral.shape[1]

    r_cos_theta = bond_lengths * torch.cos(np.pi - bond_angles)
    r_sin_theta = bond_lengths * torch.sin(np.pi - bond_angles)

    point_x = r_cos_theta.unsqueeze(1)
    point_y = (torch.cos(dihedral.squeeze(1)) * r_sin_theta).unsqueeze(1)
    point_z = (torch.sin(dihedral.squeeze(1)) * r_sin_theta).unsqueeze(1)

    point = torch.stack([point_x, point_y, point_z])
    point_perm = point.permute(1, 3, 2, 0)
    return point_perm.contiguous().view(
        num_steps * NUM_DIHEDRALS, batch_size, NUM_DIMENSIONS
    )


def point_to_coordinate(
    points: torch.Tensor, num_fragments: int = 6
) -> CoordTensor:
    """
    Takes points from dihedral_to_point and sequentially converts them into
    coordinates of a 3D structure.

    Reconstruction is done in parallel by independently reconstructing
    num_fragments and the reconstituting the chain at the end in reverse order.
    The core reconstruction algorithm is NeRF, based on
    DOI: 10.1002/jcc.20237 by Parsons et al. 2005.
    The parallelized version is described in
    https://www.biorxiv.org/content/early/2018/08/06/385450.

    :param points: Tensor containing points as returned by `dihedral_to_point`.
        Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    :param num_fragments: Number of fragments in which the sequence is split
        to perform parallel computation.
    :return: Tensor containing correctly transformed atom coordinates.
        Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """

    # Compute optimal number of fragments if needed
    total_num_angles = points.shape[0]  # NUM_STEPS x NUM_DIHEDRALS
    if num_fragments is None:
        num_fragments = int(math.sqrt(total_num_angles))

    # Initial three coordinates (specifically chosen to eliminate need for
    # extraneous matmul)
    Triplet = collections.namedtuple("Triplet", "a, b, c")
    batch_size = points.shape[1]
    init_matrix = np.array(
        [
            [-np.sqrt(1.0 / 2.0), np.sqrt(3.0 / 2.0), 0],
            [-np.sqrt(2.0), 0, 0],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    init_matrix = torch.from_numpy(init_matrix).to(points.device)
    init_coords = [
        row.repeat([num_fragments * batch_size, 1]).view(
            num_fragments, batch_size, NUM_DIMENSIONS
        )
        for row in init_matrix
    ]
    init_coords = Triplet(
        *init_coords
    )  # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    # Pad points to yield equal-sized fragments
    padding = (
        num_fragments - (total_num_angles % num_fragments)
    ) % num_fragments  # (NUM_FRAGS x FRAG_SIZE) - (NUM_STEPS x NUM_DIHEDRALS)
    points = F.pad(
        # points, (0, 0, 0, 0, 0, padding)
        points,
        (0, 0, 0, 0, 0, padding),
    )  # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
    points = points.view(
        num_fragments, -1, batch_size, NUM_DIMENSIONS
    )  # [NUM_FRAGS, FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
    points = points.permute(
        1, 0, 2, 3
    )  # [FRAG_SIZE, NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    # Extension function used for single atom reconstruction and whole fragment
    # alignment
    def extend(prev_three_coords, point, multi_m):
        """
        Aligns an atom or an entire fragment depending on value of `multi_m`
        with the preceding three atoms.

        :param prev_three_coords: Named tuple storing the last three atom
            coordinates ("a", "b", "c") where "c" is the current end of the
            structure (i.e. closest to the atom/ fragment that will be added now).
            Shape NUM_DIHEDRALS x [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMENSIONS].
            First rank depends on value of `multi_m`.
        :param point: Point describing the atom that is added to the structure.
            Shape [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
            First rank depends on value of `multi_m`.
        :param multi_m: If True, a single atom is added to the chain for
            multiple fragments in parallel. If False, an single fragment is added.
            Note the different parameter dimensions.
        :return: Coordinates of the atom/ fragment.
        """

        bc = F.normalize(prev_three_coords.c - prev_three_coords.b, dim=-1)
        # bc = F.normalize(prev_three_coords.b - prev_three_coords.c, dim=-1)
        n = F.normalize(
            torch.cross(prev_three_coords.b - prev_three_coords.a, bc), dim=-1
        )
        if multi_m:  # multiple fragments, one atom at a time
            m = torch.stack([bc, torch.cross(n, bc), n]).permute(1, 2, 3, 0)
            m = m.to(point.device)  ###
        else:  # single fragment, reconstructed entirely at once.
            s = point.shape + (3,)
            m = torch.stack([bc, torch.cross(n, bc), n]).permute(1, 2, 0)
            m = m.repeat(s[0], 1, 1).view(s)
            m = m.to(point.device)  ###
        coord = (
            torch.squeeze(torch.matmul(m, point.unsqueeze(3)), dim=3)
            + prev_three_coords.c
        )
        return coord

    # Loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially
    # generating the coordinates for each fragment across all batches
    coords_list = [None] * points.shape[
        0
    ]  # FRAG_SIZE x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]
    prev_three_coords = init_coords

    for i in range(points.shape[0]):  # Iterate over FRAG_SIZE
        coord = extend(prev_three_coords, points[i], True)
        coords_list[i] = coord
        prev_three_coords = Triplet(
            prev_three_coords.b, prev_three_coords.c, coord
        )  # b, c, coord

    coords_pretrans = torch.stack(coords_list).permute(1, 0, 2, 3)

    # Loop backwards over NUM_FRAGS to align the individual fragments. For each
    # next fragment, we transform the fragments we have already iterated over
    # (coords_trans) to be aligned with the next fragment
    coords_trans = coords_pretrans[-1]
    for i in reversed(range(coords_pretrans.shape[0] - 1)):
        # Transform the fragments that we have already iterated over to be
        # aligned with the next fragment `coords_trans`
        transformed_coords = extend(
            Triplet(*[di[i] for di in prev_three_coords]), coords_trans, False
        )
        coords_trans = torch.cat([coords_pretrans[i], transformed_coords], 0)

    # coords = F.pad(
    #    coords_trans[: total_num_angles - 1], (0, 0, 0, 0, 1, 0)
    # )  # original

    # Pad and set first Ca to origin
    coords = F.pad(coords_trans[: total_num_angles - 2], (0, 0, 0, 0, 2, 0))
    # Set first N to canonical position
    coords[0, 0] = torch.tensor([[-1.4584, 0, 0]])
    return coords


def sn_nerf(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    l_cd: torch.Tensor,
    theta: torch.Tensor,
    chi: torch.Tensor,
    l_bc: torch.Tensor,
):
    """Return coordinates for point d given previous points & parameters. Optimized NeRF.
    This function has been optimized from the original nerf to be about 20% faster. It
    contains fewer normalization steps and total calculations than the original
    formulation. See https://doi.org/10.1002/jcc.20237 for details.


    Args:
        a (torch.float32 tensor): (3 x 1) tensor describing point a.
        b (torch.float32 tensor): (3 x 1) tensor describing point b.
        c (torch.float32 tensor): (3 x 1) tensor describing point c.
        l_cd (torch.float32 tensor): (1) tensor describing the length between points
            c & d.
        theta (torch.float32 tensor): (1) tensor describing angle between points b, c,
            and d.
        chi (torch.float32 tensor): (1) tensor describing dihedral angle between points
            a, b, c, and d.
        l_bc (torch.float32 tensor): (1) tensor describing length between points b and c.
    Raises:
        ValueError: Raises ValueError when value of theta is not in [-pi, pi].
    Returns:
        torch.float32 tensor: (3 x 1) tensor describing coordinates of point c after
        placement using points a, b, c, and several parameters.
    """
    if not (-np.pi <= theta <= np.pi):
        raise ValueError(
            f"theta must be in radians and in [-pi, pi]. theta = {theta}"
        )
    AB = b - a
    BC = c - b
    bc = BC / l_bc
    n = F.normalize(torch.cross(AB, bc), dim=0)
    n_x_bc = torch.cross(n, bc)

    M = torch.stack([bc, n_x_bc, n], dim=1).to(chi.device)

    l_cd = l_cd.to(chi.device)
    theta = theta.to(chi.device)
    c = c.to(chi.device)
    D2 = (
        torch.stack(
            [
                -l_cd * torch.cos(theta),
                l_cd * torch.sin(theta) * torch.cos(chi),
                l_cd * torch.sin(theta) * torch.sin(chi),
            ]
        )
        .to(torch.float32)
        .unsqueeze(1)
    )

    D = torch.mm(M, D2).squeeze()

    return D + c
