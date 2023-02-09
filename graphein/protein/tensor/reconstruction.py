"""Utilities for reconstructing protein structures."""

from typing import Iterable, List, Tuple, Union

import numpy as np
import torch

from ..resi_atoms import ATOM_NUMBERING, BB_BUILD_INFO, SC_BUILD_INFO
from .angles import torsion_to_rad
from .pnerf import sn_nerf
from .types import AtomTensor, CoordTensor, TorsionTensor


def place_fourth_atom(
    a_coord: torch.Tensor,
    b_coord: torch.Tensor,
    c_coord: torch.Tensor,
    length: torch.Tensor,
    planar: torch.Tensor,
    dihedral: torch.Tensor,
) -> torch.Tensor:
    """
    Given 3 coords + a length + a planar angle + a dihedral angle, compute a
    fourth coord

    Adapted from the IgFold Implementation (which is under the John's
    Hopkins License):
    https://github.com/Graylab/IgFold/blob/main/igfold/utils/coordinates.py

    :param a_coord: First coordinate
    :type a_coord: torch.Tensor
    :param b_coord: Second coordinate
    :type b_coord: torch.Tensor
    :param c_coord: Third coordinate
    :type c_coord: torch.Tensor
    :param length: Length of the bond
    :type length: torch.Tensor
    :param planar: Planar angle
    :type planar: torch.Tensor
    :param dihedral: Dihedral angle
    :type dihedral: torch.Tensor
    :return: Fourth coordinate
    :rtype: torch.Tensor
    """
    bc_vec = b_coord - c_coord
    bc_vec = bc_vec / bc_vec.norm(dim=-1, keepdim=True)

    n_vec = (b_coord - a_coord).expand(bc_vec.shape).cross(bc_vec)
    n_vec = n_vec / n_vec.norm(dim=-1, keepdim=True)

    m_vec = [bc_vec, n_vec.cross(bc_vec), n_vec]
    d_vec = [
        length * torch.cos(planar),
        length * torch.sin(planar) * torch.cos(dihedral),
        -length * torch.sin(planar) * torch.sin(dihedral),
    ]

    return c_coord + sum(m * d for m, d in zip(m_vec, d_vec))


def get_ideal_backbone_coords(
    n: int = 1, ca_center: bool = True, device: torch.device = "cpu"
) -> torch.Tensor:
    """
    Get idealized backbone ``(N, CA, C, CB)`` coordinates.

    Adapted from the IgFold Implementation (which is under the John's
    Hopkins License):
    https://github.com/Graylab/IgFold/blob/main/igfold/utils/coordinates.py

    :param center: if ``True`` the reference residue is centered on CA,
        otherwise, the center of mass is used. Defaults to ``True``.
    :type CA_centered: bool, optional
    :return: Idealized backbone coordinates
    :rtype: torch.Tensor
    """
    N = torch.tensor([[0, 0, -1.458]], dtype=torch.float, device=device)
    A = torch.tensor([[0, 0, 0]], dtype=torch.float, device=device)
    B = torch.tensor([[0, 1.426, 0.531]], dtype=torch.float, device=device)
    C = place_fourth_atom(
        B,
        A,
        N,
        torch.tensor(2.460),
        torch.tensor(0.615),
        torch.tensor(-2.143),
    )

    coords = torch.cat([N, A, C, B]).float()

    if not ca_center:
        coords -= coords.mean(
            dim=0,
            keepdim=True,
        )

    return coords.unsqueeze(0).repeat(n, 1, 1)


def place_o_coords(coords: AtomTensor) -> AtomTensor:
    """
    Adapted from the IgFold Implementation (which is under the John's
    Hopkins License):
    https://github.com/Graylab/IgFold/blob/main/igfold/utils/coordinates.py

    :param coords: _description_
    :type coords: _type_
    :return: _description_
    :rtype: _type_
    """
    N = coords[:, 0, :]
    A = coords[:, 1, :]
    C = coords[:, 2, :]

    o_coords = place_fourth_atom(
        torch.roll(N, shifts=-1, dims=0),
        A,
        C,
        torch.tensor(1.231),
        torch.tensor(2.108),
        torch.tensor(-3.142),
    ).unsqueeze(2)

    coords = torch.cat(
        [coords, o_coords.view(-1, 1, 3)],
        dim=1,
    )

    return coords


def dist_mat_to_coords(d: torch.Tensor, k: int = 3) -> CoordTensor:
    """
    Computes a set of coordinates that satisfy the (valid) Euclidean
    distance matrix ``d``.

    See: https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix

    :param d: Valid euclidean distance matrix
    :type d: torch.Tensor
    :param k: Number of dimensions of coordinate output, defaults to ``3``.
    :type k: int, optional
    :return: Set of coordinates that satisfy the distance matrix ``d``.
    :rtype: graphein.protein.tensor.types.CoordTensor
    """
    D_ij_2 = torch.pow(d, 2)
    D_1i_2 = torch.pow(d[0, :], 2).unsqueeze(1)
    D_1j_2 = torch.pow(d[:, 0], 2).unsqueeze(0)
    Mij = (D_1i_2 + D_1j_2 - D_ij_2) / 2

    # try:
    U, S, _ = torch.linalg.svd(Mij)
    # except Exception:  # torch.svd may have convergence issues for GPU and CPU.
    #    U, S, _ = torch.linalg.svd(
    #        Mij + 1e-4 * Mij.mean() * torch.rand_like(Mij)
    #    )
    X = U * torch.sqrt(S)
    X = X[:, :k]
    return X


def build_sidechain(
    bb: AtomTensor,
    residue_types: List[str],
    torsion_angles: TorsionTensor,
) -> AtomTensor:  # sourcery skip: assign-if-exp, lift-duplicated-conditional
    if torsion_angles.shape[-1] == 8:
        torsion_angles = torsion_to_rad(torsion_angles)
        torsion_angles = torch.stack(torsion_angles, dim=1)
    # Initialise new atom tensor [L x 37 x 3] and fill it with backbone atoms.
    # pos = torch.ones_like(x.atom_tensor) * 1e-5
    pos = torch.ones((bb.shape[0], 37, 3), device=bb.device) * 1e-5

    if bb.shape[1] == 5:
        pos[:, :5, :] = bb
        cb_skip = True
    elif bb.shape[1] == 4:
        pos[:, :4, :] = bb
        cb_skip = False

    # Get the residues in the batch (list of 3-letter AA codes)
    # batch_seq = list(chain.from_iterable(x.node_id))
    # nodes = [n.split(":")[1] for n in batch_seq]

    # Iterate over residues
    # for i, res in enumerate(nodes):
    for i, res in enumerate(residue_types):
        start = i == 0
        next_res = pos[i, :] if i == len(residue_types) - 1 else None

        build_vals = _get_build_params(res)
        last_torsion = None
        # Iterate over atoms
        for j, (bond, pbond, angle, torsion, atom_name) in enumerate(
            build_vals
        ):
            # Skip CB as we place it in BB construction
            if atom_name[-1] == "CB" and cb_skip:
                continue
            if start and next_res is not None:
                a, b, c = pos[i + 1, 0], pos[i, 1], pos[i, 2]  # N+, CA, C
            elif start:
                a, b, c = pos[i - 1, 2], pos[i, 0], pos[i, 1]  # C-, N, CA
            else:
                idxs = [ATOM_NUMBERING[atom] for atom in atom_name[:-1]]
                a, b, c = (
                    pos[i, idxs[0]],
                    pos[i, idxs[1]],
                    pos[i, idxs[2]],
                )  # N, CA, C

            # Select correct torsion angle
            if type(torsion) is str and torsion == "p":
                if j == 5 and res == "ARG":
                    torsion = torch.tensor(0.0)
                else:
                    torsion = torsion_angles[i, j - 1]
            elif (
                type(torsion) is str
                and torsion == "i"
                and last_torsion is not None
            ):
                torsion = last_torsion - np.pi

            # Get coordinates of new point
            new_pt = sn_nerf(a, b, c, bond, angle, torsion, pbond)
            # Assign new point to atom tensor
            pos[i, ATOM_NUMBERING[atom_name[-1]]] = new_pt
            last_torsion = torsion

    return pos


def _get_build_params(
    res: str,
) -> Iterable[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Union[torch.Tensor, str],
        List[str],
    ]
]:
    """
    Returns iterable of sidechain construction params for a residue.

    Derived from OpenFold:
    https://github.com/aqlaboratory/openfold/blob/main/openfold/np/residue_constants.py

    :param res: Residue name (3-letter code)
    :type res: str
    :return: Iterable of sidechain construction params
    :rtype: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Union[torch.Tensor, str], List[str]]]
    """
    build_params = SC_BUILD_INFO[res]
    bond_vals = [
        torch.tensor(b, dtype=torch.float32)
        for b in build_params["bonds-vals"]
    ]
    pbond_vals = [
        torch.tensor(BB_BUILD_INFO["BONDLENS"]["n-ca"], dtype=torch.float32)
    ] + [
        torch.tensor(b, dtype=torch.float32)
        for b in build_params["bonds-vals"]
    ][
        :-1
    ]
    angle_vals = [
        torch.tensor(a, dtype=torch.float32)
        for a in build_params["angles-vals"]
    ]
    torsion_vals = [
        torch.tensor(t, dtype=torch.float32) if t not in ["p", "i"] else t
        for t in build_params["torsion-vals"]
    ]
    atom_names = [t.split("-") for t in build_params["torsion-names"]]

    return zip(bond_vals, pbond_vals, angle_vals, torsion_vals, atom_names)
