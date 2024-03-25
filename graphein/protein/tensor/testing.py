"""Utilities for sanity checking protein tensors."""

from functools import partial
from typing import List, Tuple, Union

from loguru import logger as log

from graphein.utils.dependencies import import_message

from ..resi_atoms import (
    ATOM_NUMBERING_MODIFIED,
    STANDARD_AMINO_ACID_MAPPING_1_TO_3,
)
from .sequence import get_atom_indices
from .types import AtomTensor, BackboneTensor, CoordTensor, ResidueTensor

try:
    import torch
except ImportError:
    message = import_message(
        "graphein.protein.tensor.testing",
        "torch",
        conda_channel="pytorch",
        pip_install=True,
    )
    log.warning(message)


def has_nan(x: torch.Tensor) -> bool:
    """Checks a tensor for ``NaN`` values.

    :param x: Tensor to check for ``NaN`` values
    :type x: torch.Tensor
    :return: Boolean indicating whether the tensor contains NaN values.
    :rtype: bool
    """
    return x.isnan().any()


assert_tensor_equal = partial(torch.testing.assert_close, rtol=0, atol=0)
is_tensor_equal = partial(torch.allclose, rtol=0, atol=0)


def has_complete_backbone(
    x: Union[AtomTensor, BackboneTensor],
    fill_value: float = 1e-5,
    backbone_indices: List[int] = [0, 1, 2, 3],
) -> bool:
    """
    Checks a protein coordinate tensor (``L x Num Atoms (default 37) x 3``) for
    a complete backbone assignment.

    If ``fill_value`` entries are found in the ``backbone_indices`` (dim 1) then
    the backbone is not considered complete and we return ``False``. If no
    entries with the ``fill_value`` are found, we return ``True``.

    :param x: Atom Tensor to check backbone completeness.
    :type x: graphein.protein.tensor.types.AtomTensor
    :param fill_value: Fill value used in the ``AtomTensor``. Default is ``1e-5``.
    :type fill_value: Float
    :param backbone_indices: List of indices in dimension 1 of the AtomTensor
        to be checked. Defaults to ``[0, 1, 2, 3]`` for ``[N, Ca, C, O]`` in the
        default assignment.
    :type backbone_indices: List[int].
    :return: Boolean indicating whether the backbone is complete.
    :rtype: bool

    .. seealso::

        :func:`graphein.protein.tensor.testing.has_complete_residue`
        :func:`graphein.protein.tensor.testing.is_complete_structure`


    """
    indices = torch.tensor(backbone_indices)
    return torch.sum(x[:, indices] == fill_value) == 0


def has_complete_residue(
    x: ResidueTensor, residue_type: str, fill_value: float = 1e-5
) -> bool:
    """Checks whether a residue has all of the requisite atoms.

    :param x: Tensor of shape ``37 x 3`` of atom coordinates.
    :type x: torch.Tensor
    :param residue_type: 1 or 3-letter code of residue type.
    :type residue_type: str
    :param fill_value: Fill value used to denote the absence of an atom,
        defaults to ``1e-5``.
    :type fill_value: float, optional
    :return: Boolean indicating whether the residue has all of the requisite
        atoms.
    :rtype: bool

    .. seealso::

        :func:`graphein.protein.tensor.testing.has_complete_residue`
        :func:`graphein.protein.tensor.testing.is_complete_structure`
    """
    if len(residue_type) == 1:
        residue_type = STANDARD_AMINO_ACID_MAPPING_1_TO_3[residue_type]
    true_residue_indices = get_atom_indices()[residue_type]

    def _get_index(y: torch.Tensor) -> Tuple[int, ...]:
        indices = (
            torch.nonzero(torch.sum(y != fill_value, dim=1))
            .squeeze(1)
            .tolist()
        )
        # remove oxt if present
        oxt_index = ATOM_NUMBERING_MODIFIED["OXT"]
        indices = tuple(i for i in indices if i != oxt_index)
        return indices

    present_indices = _get_index(x)
    return present_indices == true_residue_indices


def is_complete_structure(
    x: AtomTensor, residues: Union[List[str], str], fill_value: float = 1e-5
) -> bool:
    """Checks whether a protein structure has all of the requisite atoms.

    :param x: AtomTensor to check for completeness.
    :type x: graphein.protein.tensor.types.AtomTensor
    :param residues: List of 1 or 3-letter codes of residue types.
    :type residues: Union[List[str], str]
    :param fill_value: Fill value used to denote the absence of an atom,
        defaults to ``1e-5``.
    :type fill_value: float, optional
    :return: Boolean indicating whether the structure has all of the requisite
        atoms.
    :rtype: bool

    .. seealso::

        :func:`graphein.protein.tensor.sequence.infer_residue_types`
    """
    length = x.shape[0]
    assert (
        len(residues) == length
    ), "Length of sequence must match coordinate tensor in dimension 0."

    if isinstance(residues, str):
        residues = list(residues)

    return all(
        has_complete_residue(x[i], residues[i], fill_value=fill_value)
        for i in range(length)
    )


def random_atom_tensor(length: int = 64) -> AtomTensor:
    """Returns a random tensor of shape ``length x 37 x 3``.


    .. seealso::

        :func:`random_coord_tensor`
        :func:`graphein.protein.tensor.data.get_random_protein`
        :func:`graphein.protein.tensor.data.get_random_batch`

    :param length: Length of random protein, defaults to ``64``
    :type length: int, optional
    :return: Random tensor of shape ``length x 37 x 3``
    :rtype: AtomTensor
    """
    return torch.rand((length, 37, 3))


def random_coord_tensor(length: int = 64) -> CoordTensor:
    """Returns a random tensor of shape ``length x 3``.

    .. seealso::
        :func:`random_atom_tensor`
        :func:`graphein.protein.tensor.data.get_random_protein`
        :func:`graphein.protein.tensor.data.get_random_batch`

    :param length: Length of random coordinates, defaults to ``64``
    :type length: int, optional
    :return: Random tensor of shape ``length x 3``
    :rtype: CoordTensor
    """
    return torch.rand((length, 3))


def fill_missing_o_coords(x: AtomTensor):
    # TODO
    raise NotImplementedError


def fill_missing_backbone(
    x: Union[AtomTensor, CoordTensor], fill_value: float = 1e-5
) -> Union[AtomTensor, BackboneTensor]:
    if has_complete_backbone(x, fill_value):
        return x
    # TODO
    raise NotImplementedError
