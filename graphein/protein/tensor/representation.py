"""Utility functions for converting between representations of protein structures."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import List, Tuple

from loguru import logger as log

from graphein.utils.utils import import_message

from .types import AtomTensor, CoordTensor

try:
    import torch
except ImportError:
    message = import_message(
        "graphein.protein.tensor.representation",
        "torch",
        conda_channel="pytorch",
        pip_install=True,
    )
    log.warning(message)


def get_full_atom_coords(
    atom_tensor: AtomTensor, fill_value: float = 1e-5
) -> Tuple[CoordTensor, torch.Tensor, torch.Tensor]:
    """Converts an ``AtomTensor`` to a full atom representation.

    Return tuple of coords ``(N_atoms x 3)``, residue_index ``(N_atoms)``,
    atom_type ``(N_atoms x [0-36])``


    :param atom_tensor: AtomTensor of shape
        ``(N_residues, N_atoms (default is 37), 3)``
    :type atom_tensor: graphein.protein.tensor.AtomTensor
    :param fill_value: Value used to fill missing values. Defaults to ``1e-5``.
    :return: Tuple of coords, residue_index, atom_type
    :rtype: Tuple[CoordTensor, torch.Tensor, torch.Tensor]
    """
    # Get number of atoms per residue
    filled = atom_tensor[:, :, 0] != fill_value
    nz = filled.nonzero()
    residue_index = nz[:, 0]
    atom_type = nz[:, 1]
    coords = atom_tensor.reshape(-1, 3)
    coords = coords[coords != fill_value].reshape(-1, 3)
    return coords, residue_index, atom_type


def get_c_alpha(x: AtomTensor, index: int = 1) -> CoordTensor:
    """Returns tensor of C-alpha atoms: ``(L x 3)``

    :param x: Tensor of atom positions of shape:
        ``(N_residues, N_atoms (default=37), 3)``
    :type x: graphein.protein.tensor.types.AtomTensor
    :param index: Index of C-alpha atom in dimension 1 of the AtomTensor.
    :type index: int
    """
    return x if x.ndim == 2 else x[:, index, :]


def get_backbone(x: AtomTensor) -> AtomTensor:  # TODO
    """Returns tensor of backbone atoms: ``(L x 4 x 3)``

    :param x: _description_
    :type x: AtomTensor
    :return: _description_
    :rtype: AtomTensor
    """
    raise NotImplementedError


def coarsen_sidechain(
    x: AtomTensor,
    backbone_indices: List[int] = [0, 1, 2, 3],
    aggregation: str = "mean",
) -> CoordTensor:
    """Returns tensor of sidechain centroids: ``(L x 3``

    :param x: Tensor of atom positions of shape
        ``(N_residues, N_atoms (default=37), 3)``
    :type x: graphein.protein.tensor.AtomTensor
    :param backbone_indices: List of indices in dimension 1 of the AtomTensor
        that correspond to backbone atoms (N, Ca, C, O). Defaults to
        ``[0, 1, 2, 3]``.
    :type backbone_indices: List[int]
    :param aggregation: Aggregation method to use. Defaults to ``"mean"``.
    :type aggregation: str
    :return: Tensor of sidechain centroids of shape ``(N_residues, 3)``
    :rtype: graphein.protein.tensor.CoordTensor
    :raises NotImplementedError: If aggregation method is not implemented.
    """
    # sourcery skip: remove-unnecessary-else, swap-if-else-branches
    # Select indices
    sidechain_indices = [
        a for a in range(x.shape[1]) if a not in backbone_indices
    ]
    sidechain_points = x[:, sidechain_indices, :]

    # Compute mean sidechain position
    if aggregation == "mean":
        sidechain_points = torch.mean(sidechain_points, dim=1)
        # TODO mask filled positions
    else:
        raise NotImplementedError(
            f"Aggregation method {aggregation} not implemented."
        )

    return sidechain_points