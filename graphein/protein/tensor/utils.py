"""Utilities for working protein Protein Tensors."""
import torch

from .types import AtomTensor


def has_complete_backbone(
    x: AtomTensor,
    fill_value: float = 1e-5,
    backbone_indices: List[int] = [0, 1, 2, 3],
) -> bool:
    """
    Checks a protein coordinate tensor (``L x Num Atoms (default 37) x 3``) for
    a complete backbone assignment.

    If ``fill_value`` entries are found in the ``backbone_indices`` (dim 1) then
    the backbone is not considered complete and we return False. If no entries with
    the ``fill_value`` are found, we return true.

    :param x: Atom Tensor to check
    :type x: graphein.protein.tensor.types.AtomTensor
    :param fill_value: Fill value used in the ``AtomTensor``. Default is ``1e-5``.
    :type fill_value: Float
    :param backbone_indices: List of indices in dimension 1 of the AtomTensor
        to be checked. Defaults to ``[0, 1, 2, 3]`` for ``N, Ca, C, O`` in the
        default assignment.
    type backbone_indices: List[int].
    """
    indices = torch.tensor(backbone_indices)
    if torch.sum(x[:, indices] == fill_value) != 0:
        return False
    else:
        return True
