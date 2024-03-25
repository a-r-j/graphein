"""Utilities for computing various protein angles."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import copy
from typing import List, Optional, Tuple, Union

import numpy as np
from loguru import logger as log

from graphein.utils.dependencies import import_message

from ..resi_atoms import ATOM_NUMBERING, CHI_ANGLES_ATOMS
from .testing import has_nan
from .types import AtomTensor, CoordTensor, DihedralTensor, TorsionTensor

try:
    from einops import rearrange
except ImportError:
    message = import_message(
        "graphein.protein.tensor.angles",
        "einops",
        pip_install=True,
        extras=True,
    )

try:
    from torch_geometric.utils import to_dense_batch
except ImportError:
    message = import_message(
        "graphein.protein.tensor.angles",
        "torch_geometric",
        "pyg",
        pip_install=True,
    )

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    message = import_message(
        "graphein.protein.tensor.angles",
        "torch",
        conda_channel="pytorch",
        pip_install=True,
    )
    log.warning(message)


def _extract_torsion_coords(
    coords: AtomTensor, res_types: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a ``(L*?) x 4 x 3`` tensor of the coordinates of the atoms for
    each sidechain torsion angle. The first dimension will be larger than the
    input as we flatten the array of torsion angles per residue.

    Also returns a ``(L*?) x 1`` indexing tensor to map back to each residue
    (this is because we have variable numbers of torsion angles per residue).

    :param coords: AtomTensor of shape ``(L*?, 37, 3)``
    :type coords: AtomTensor
    :param res_types: List of 3-letter residue types
    :type res_types: List[str]
    :return: Coordinates for computing each sidechain torsion angle
        and indexing tensor.
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    res_atoms = []
    idxs = []

    # Whether or not the protein contains selenocysteine
    selenium = coords.shape[1] == 38

    # Iterate over residues and grab indices of the atoms for each Chi angle
    for i, res in enumerate(res_types):
        res_coords = []

        try:
            angle_groups = CHI_ANGLES_ATOMS[res]
        except KeyError:
            log.warning(
                f"Can't determine chi angle groups for non-standard residue: {res}. These will be set to 0"
            )
            angle_groups = []
        if (not selenium and res == "SEC") or res == "PYL":
            angle_groups = []

        for angle_coord_set in angle_groups:
            res_coords.append([ATOM_NUMBERING[i] for i in angle_coord_set])
            idxs.append(i)
        res_atoms.append(torch.tensor(res_coords, device=coords.device))

    idxs = torch.tensor(idxs, device=coords.device).long()
    res_atoms = torch.cat(res_atoms).long()  # TODO torch.stack instead of cat

    # Select the coordinates for each chi angle
    coords = torch.take_along_dim(
        coords[idxs, :, :], dim=1, indices=res_atoms.unsqueeze(-1)
    )
    return idxs, coords


def sidechain_torsion(
    coords: AtomTensor,
    res_types: List[str],
    rad: bool = True,
    embed: bool = False,
    return_mask: bool = False,
) -> Union[TorsionTensor, Tuple[TorsionTensor, torch.Tensor]]:
    """
    Computes sidechain torsion angles for a protein.

    Atom groups used to define each chi angle are specified in
    :ref:`graphein.protein.resi_atoms.CHI_ANGLES_ATOMS`.

    :param coords: Tensor of shape ``(L, N, 3)`` where ``L`` is the number of
        residues, ``N`` is the number of atoms per residue (Default is 37;
        problems may occur with differing selections).
    :type coords: AtomTensor
    :param res_types: List of three-letter residue types.
    :type res_types: List[str]
    :param rad: If ``True``, returns angles in radians, else in degrees.
        Defaults to ``True``.
    :type rad: bool, optional
    :param return_mask: _description_, defaults to False
    :type return_mask: bool, optional
    :return: _description_
    :rtype: Union[TorsionTensor, Tuple[TorsionTensor, torch.Tensor]]
    """
    # Whether or not the protein contains selenocysteine
    selenium = coords.shape[1] == 38

    idxs, coords = _extract_torsion_coords(coords, res_types)
    angles = _dihedral_angle(
        coords[:, 0, :].unsqueeze(1),
        coords[:, 1, :].unsqueeze(1),
        coords[:, 2, :].unsqueeze(1),
        coords[:, 3, :].unsqueeze(1),
    )

    if embed and not rad:
        raise ValueError("Cannot embed torsion angles in degrees.")

    if not rad:
        angles = angles * 180 / torch.pi

    angles, mask = to_dense_batch(angles, idxs)
    angles = angles.squeeze(-1)

    # Interleave sin and cos transformed tensors
    if embed:
        angles = rearrange(
            [torch.cos(angles), torch.sin(angles)], "t h w-> h (w t)"
        )
        mask = rearrange([mask, mask], "t h w -> h (w t)")

    # Pad if last residues are a run of ALA, GLY or UNK
    post_pad_len = 0
    res_types = copy.deepcopy(res_types)
    res_types.reverse()
    PAD_RESIDUES = ["ALA", "GLY", "UNK"]
    # If we have selenocysteine but no Se atoms,
    # add it to the list of residues to pad since
    if not selenium:
        PAD_RESIDUES.append("SEC")
    for res in res_types:
        if res in PAD_RESIDUES:
            post_pad_len += 1
        else:
            break

    if post_pad_len != 0:
        if embed:
            msk = torch.tensor(
                [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], device=coords.device
            ).repeat(post_pad_len, 1)
            mask_msk = torch.tensor([False] * 8, device=coords.device).repeat(
                post_pad_len, 1
            )
        else:
            msk = torch.zeros(post_pad_len, 4, device=coords.device)
            mask_msk = torch.zeros(
                post_pad_len, 4, device=coords.device, dtype=bool
            )
        angles = torch.vstack([angles, msk])
        mask = torch.vstack([mask, mask_msk])

    return (angles, mask) if return_mask else angles


def chi1():
    raise NotImplementedError


def chi2():
    raise NotImplementedError


def chi3():
    raise NotImplementedError


def chi4():
    raise NotImplementedError


def kappa(
    x: Union[AtomTensor, CoordTensor],
    batch: Optional[torch.Tensor] = None,
    rad: bool = True,
    sparse: bool = True,
    ca_idx: int = 1,
    embed: bool = False,
) -> torch.Tensor:
    """
    Computes virtual bond angle (bend angle) defined by three Ca atoms of
    residues ``i-2``, ``i`` and ``i+2``. The first and last two angles are zero
    padded.

    .. seealso::
        :meth:`graphein.protein.tensor.angles.alpha`
        :meth:`graphein.protein.tensor.angles.dihedrals`

    :param x: Tensor of atomic positions or tensor of CA positions.
    :type x: Union[AtomTensor, CoordTensor]
    :param ca_idx: If ``x`` is an AtomTensor, this is the index of the CA atoms
        in dimension 1, defaults to 1
    :type ca_idx: int, optional
    :return: Tensor of bend angles
    :rtype: torch.Tensor
    """
    if not rad and embed:
        raise ValueError("Cannot embed kappa angles in degrees.")

    if x.ndim == 3:
        x = x[:, ca_idx, :]

    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

    x, mask = to_dense_batch(x, batch)

    ca_prev = x[:, :-4, :]
    ca = x[:, 2:-2, :]
    ca_next = x[:, 4:, :]

    angles = np.pi - to_ang(
        # ca_next.view(-1, 3), ca.view(-1, 3), ca_prev.view(-1, 3)
        ca_next,
        ca,
        ca_prev,
    )
    # Zero pad first two and last two angles
    angles = F.pad(angles, (2, 2))

    if not rad:
        angles = torch.rad2deg(angles)

    if sparse:
        angles = angles[mask]

    if embed:
        angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    return angles


def alpha(
    x: Union[AtomTensor, CoordTensor],
    batch: Optional[torch.Tensor] = None,
    sparse: bool = True,
    rad: bool = True,
    ca_idx: int = 1,
    embed: bool = True,
) -> torch.Tensor:
    """
    Computes virtual bond dihedral angle defined by four Ca atoms of residues
    ``i-1``, ``i``, ``i+1``, ``i+2``. The first angle and last two angles are
    zero padded.

    .. seealso::
        :meth:`graphein.protein.tensor.angles.kappa`
        :meth:`graphein.protein.tensor.angles.dihedrals`

    :param x: Tensor of atomic positions or tensor of CA positions.
    :type x: Union[AtomTensor, CoordTensor]
    :param batch: Tensor of batch indices, defaults to ``None``.
    :type batch: Optional[torch.Tensor], optional
    :param sparse: Whether or not to return the output in sparse or dense format
        , defaults to ``True``.
    :type sparse: bool, optional
    :param rad: _description_, defaults to ``True``.
    :type rad: bool, optional
    :param ca_idx: If ``x`` is an AtomTensor, this argument specifies the index
        of the CA atoms in dimension 1, defaults to ``1``.
    :type ca_idx: int, optional
    :param embed: Whether or not to embed the angles on the unit circle,
        defaults to ``True``.
    :type embed: bool, optional
    :return: Tensor of dihedral angles
    :rtype: torch.Tensor
    """
    if not rad and embed:
        raise ValueError(
            "Cannot embed angles on unit circle if not in radians."
        )

    if x.ndim == 3:
        x = x[:, ca_idx, :]

    if batch is None:
        batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

    x, mask = to_dense_batch(x, batch)
    ca_prev = x[:, :-3, :]
    ca = x[:, 1:-2, :]
    ca_next = x[:, 2:-1, :]
    ca_next_next = x[:, 3:, :]

    angles = _dihedral_angle(ca_prev, ca, ca_next, ca_next_next)
    angles = F.pad(angles, (1, 2))

    if not rad:
        angles = torch.rad2deg(angles)
    if sparse:
        angles = angles[mask]

    if embed:
        angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    return angles


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
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if b.ndim == 1:
        b = b.unsqueeze(0)
    if c.ndim == 1:
        c = c.unsqueeze(0)

    ba = b - a
    bc = b - c
    return torch.acos(
        (ba * bc).sum(dim=-1)
        / (torch.norm(ba, dim=-1) * torch.norm(bc, dim=-1))
    )


# @torch.jit.script
def get_backbone_bond_lengths(x: AtomTensor) -> torch.Tensor:
    """
    Compute the bond lengths between backbone atoms.

    Returns the a tensor of shape ``(N, 3)`` where ``N`` is the number of
    residues in protein. Dimension 1 contains the bond lengths between Ca and
    C, C and N, and N and Ca.

    :param x: Tensor of atomic positions.
    :type x: AtomTensor
    :return: Tensor of backbone bond lengths.
    :rtype: torch.Tensor
    """
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
    """Compute the bond angles between backbone atoms:
        ``[C-N-Ca, N-Ca-C, Ca-C,N]``.

    .. seealso:: :meth:`graphein.protein.tensor.angles.to_ang`

    :param x: Tensor of atomic positions.
    :type x: AtomTensor
    :return: Tensor of backbone bond angles ``[L x 3]``.
    :rtype: torch.Tensor
    """
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
    if x.ndim == 1:
        x = x.unsqueeze(0)
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

    # bc = F.normalize(b - c, dim=2)
    bc = F.normalize(b - c, dim=-1)
    # n1 = torch.cross(F.normalize(a - b, dim=2), bc)
    n1 = torch.cross(F.normalize(a - b, dim=-1), bc)
    # n2 = torch.cross(bc, F.normalize(c - d, dim=2))
    n2 = torch.cross(bc, F.normalize(c - d, dim=-1))
    # x = (n1 * n2).sum(dim=2)
    x = (n1 * n2).sum(dim=-1)
    x = torch.clamp(x, -1 + eps, 1 - eps)
    x[x.abs() < eps] = eps

    y = (torch.cross(n1, bc) * n2).sum(dim=-1)
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

    if embed and not rad:
        raise ValueError(
            "Cannot embed angles in degrees. Use embed=True and rad=True."
        )

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

    if not rad:
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

    :param x_dihedrals: Dihedral tensor of shape ``(L, 6)``.
    :type x_dihedrals: DihedralTensor
    :param concat: Whether to concatenate the angles into a single tensor, or
        to return a tuple of tensors (``phi``, ``psi``, ``omega``), defaults to
        ``False``.
    :type concat: bool, optional
    :return: Dihedral tensor of shape ``(L, 3)`` or a tuple of tensors
    :rtype: Union[DihedralTensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
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
    ``(sin(chi1), sin(chi1), cos(chi2), sin(chi2), ...)`` format to radians.

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
    chi1 = torch.atan2(x_torsion[:, 1], x_torsion[:, 0])
    chi2 = torch.atan2(x_torsion[:, 3], x_torsion[:, 2])
    chi3 = torch.atan2(x_torsion[:, 5], x_torsion[:, 4])
    chi4 = torch.atan2(x_torsion[:, 7], x_torsion[:, 6])

    if concat:
        return torch.stack([chi1, chi2, chi3, chi4], dim=1)

    return chi1, chi2, chi3, chi4
