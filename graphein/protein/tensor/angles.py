"""Utilities for computing various protein angles."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from functools import lru_cache

import numpy as np
from loguru import logger as log

from graphein.utils.dependencies import import_message

from ..resi_atoms import ATOM_NUMBERING_MODIFIED, CHI_ANGLES_ATOMS
from .types import AtomTensor, CoordTensor, DihedralTensor, TorsionTensor

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

# Row of each residue type in the chi atom table; the extra last row is for
# residues without chi angles
_CHI_ROWS = {res: i for i, res in enumerate(CHI_ANGLES_ATOMS)}
_NO_CHI = len(_CHI_ROWS)


@lru_cache(maxsize=None)
def _chi_atom_table(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the atom indices of the (up to) four chi angles of each residue
    type (``(num types + 1) x 4 x 4``) and a mask of which chi angles exist
    (``(num types + 1) x 4``). Rows follow ``_CHI_ROWS``.
    """
    index = torch.zeros(_NO_CHI + 1, 4, 4, dtype=torch.long)
    mask = torch.zeros(_NO_CHI + 1, 4, dtype=torch.bool)
    for res, row in _CHI_ROWS.items():
        for k, group in enumerate(CHI_ANGLES_ATOMS[res]):
            index[row, k] = torch.tensor(
                [ATOM_NUMBERING_MODIFIED[a] for a in group]
            )
            mask[row, k] = True
    return index.to(device), mask.to(device)


def sidechain_torsion(
    coords: AtomTensor,
    res_types: list[str],
    rad: bool = True,
    embed: bool = False,
    return_mask: bool = False,
) -> TorsionTensor | tuple[TorsionTensor, torch.Tensor]:
    """
    Computes sidechain torsion angles for a protein.

    Atom groups used to define each chi angle are specified in
    :ref:`graphein.protein.resi_atoms.CHI_ANGLES_ATOMS`. Missing chi angles
    are ``0`` (``[1, 0]`` if ``embed=True``) and ``False`` in the mask.

    :param coords: Tensor of shape ``(L, N, 3)`` where ``L`` is the number of
        residues, ``N`` is the number of atoms per residue (Default is 37;
        problems may occur with differing selections). With ``N = 38``,
        selenocysteine (``SEC``) has a chi angle (``SE`` at index 37).
    :type coords: AtomTensor
    :param res_types: List of three-letter residue types.
    :type res_types: List[str]
    :param rad: If ``True``, returns angles in radians, else in degrees.
        Defaults to ``True``.
    :type rad: bool, optional
    :param embed: If ``True``, returns ``[cos(chi1), sin(chi1), ...]``
        (``L x 8``). Defaults to ``False``.
    :type embed: bool, optional
    :param return_mask: Whether to also return a boolean mask of the chi angles
        that exist, defaults to ``False``.
    :type return_mask: bool, optional
    :return: Torsion tensor of shape ``(L, 4)`` (``(L, 8)`` if ``embed=True``)
        and, if ``return_mask=True``, the mask of the same shape.
    :rtype: Union[TorsionTensor, Tuple[TorsionTensor, torch.Tensor]]
    """
    if embed and not rad:
        raise ValueError("Cannot embed torsion angles in degrees.")

    for res in sorted(set(res_types) - _CHI_ROWS.keys()):
        log.warning(
            f"Can't determine chi angle groups for non-standard residue: {res}. These will be set to 0"
        )
    # Without Se atoms (38th atom type), SEC has no chi angles
    no_chi = {"PYL"} if coords.shape[1] == 38 else {"PYL", "SEC"}
    rows = torch.tensor(
        [
            _NO_CHI if res in no_chi else _CHI_ROWS.get(res, _NO_CHI)
            for res in res_types
        ],
        device=coords.device,
    )
    index, mask = _chi_atom_table(coords.device)
    index, mask = index[rows], mask[rows]

    # (L, 4 chi angles, 4 atoms, 3)
    chi_coords = coords[
        torch.arange(len(rows), device=coords.device).view(-1, 1, 1), index
    ]
    angles = _dihedral_angle(*chi_coords.unbind(2)).masked_fill(~mask, 0.0)

    if not rad:
        angles = angles * 180 / torch.pi

    if embed:
        angles = angle_to_unit_circle(angles)
        mask = mask.repeat_interleave(2, dim=1)

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
    x: AtomTensor | CoordTensor,
    batch: torch.Tensor | None = None,
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
    x: AtomTensor | CoordTensor,
    batch: torch.Tensor | None = None,
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
    c_n = torch.cat([c_n, torch.tensor([1.32], device=x.device)])

    # Fill missing values with ideal bond lengths
    a_c = torch.nan_to_num(a_c, nan=1.523)
    c_n = torch.nan_to_num(c_n, nan=1.320)
    n_a = torch.nan_to_num(n_a, nan=1.458)
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

    # Fill missing values with ideal bond angles
    c_n_a = torch.nan_to_num(c_n_a, nan=2.124)
    n_a_c = torch.nan_to_num(n_a_c, nan=1.941)
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
    return torch.stack([torch.cos(x), torch.sin(x)], dim=-1).flatten(-2)


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
    bc = F.normalize(b - c, dim=-1)
    n1 = torch.linalg.cross(F.normalize(a - b, dim=-1), bc)
    n2 = torch.linalg.cross(bc, F.normalize(c - d, dim=-1))
    x = (n1 * n2).sum(dim=-1)
    x = torch.clamp(x, -1 + eps, 1 - eps)
    x[x.abs() < eps] = eps

    y = (torch.linalg.cross(n1, bc) * n2).sum(dim=-1)
    return torch.atan2(y, x)


def dihedrals(
    coords: torch.Tensor,
    batch: torch.Tensor | None = None,
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

    # phi, psi, omega; the first phi and last psi/omega are zero
    angles = X.new_zeros(*X.shape[:2], 3)
    angles[:, 1:, 0] = _dihedral_angle(C_curr, N_next, Ca_next, C_next)
    angles[:, :-1, 1] = _dihedral_angle(N_curr, Ca_curr, C_curr, N_next)
    angles[:, :-1, 2] = _dihedral_angle(Ca_curr, C_curr, N_next, Ca_next)

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
) -> DihedralTensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    angles = torch.atan2(x_dihedrals[:, 1::2], x_dihedrals[:, 0::2])
    return angles if concat else angles.unbind(1)


def torsion_to_rad(
    x_torsion: TorsionTensor,
    concat: bool = True,
) -> (
    TorsionTensor
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
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
    angles = torch.atan2(x_torsion[:, 1::2], x_torsion[:, 0::2])
    return angles if concat else angles.unbind(1)
