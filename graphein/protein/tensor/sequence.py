"""Utilities for working with protein sequences."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from loguru import logger as log

from graphein.utils.dependencies import import_message

from ..resi_atoms import (
    ATOM_NUMBERING_MODIFIED,
    RESI_THREE_TO_1,
    STANDARD_AMINO_ACIDS,
    STANDARD_RESIDUE_ATOMS,
)
from .types import AtomTensor

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    message = import_message(
        "graphein.protein.tensor.sequence",
        "torch",
        conda_channel="pytorch",
        pip_install=True,
    )
    log.debug(message)


def get_sequence(
    df: pd.DataFrame,
    chains: Union[str, List[str]] = "all",
    insertions: bool = False,
    list_of_three: bool = False,
    three_to_one_map: Optional[str] = None,
    per_atom: bool = False,
) -> Union[str, List[str]]:
    """
    Retrieves the amino acid sequence from a DataFrame of a protein structure.

    :param df: DataFrame of protein structure.
    :type df: pd.DataFrame
    :param chains: List of chains to retain, defaults to ``None`` (all chains).
    :type chains: Optional[List[str]], optional
    :param insertions: Whether or not to keep insertions, defaults to ``False``
    :type insertions: bool, optional
    :param list_of_three: Whether or not to return a list of three letter codes.
        If ``False``, returns the sequence as a one-letter code string.
        Defaults to ``False``.
    :type list_of_three: bool, optional
    :return: Amino acid sequence; either as list of three-letter codes
        (``["ALA", "GLY", "TRP"...]``; ``list_of_three=True``) or string
        (``AGW..``; ``list_of_three=False``).
    :rtype: Union[str, List[str]]
    """
    # Select chains
    if chains != "all":
        if isinstance(chains, str):
            chains = [chains]
        df = df.loc[df.chain_id.isin(chains)]

    # Assign residues IDs
    if "residue_id" not in df.columns:
        df["residue_id"] = (
            df.chain_id
            + ":"
            + df.residue_name
            + ":"
            + df.residue_number.astype(str)
        )
        if insertions:
            df["residue_id"] = df.residue_id + ":" + df.insertion

    # Get residue from unique IDs
    if per_atom:
        sequence = [res.split(":")[1] for res in df.residue_id]
    else:
        sequence = [res.split(":")[1] for res in df.residue_id.unique()]

    # Convert to one letter code
    if list_of_three:
        # if three_to_one_map is not None:
        # map = RESI_THREE_TO_1
        # sequence = [three_to_one_map[res] for res in sequence]
        return sequence
    else:
        return "".join([RESI_THREE_TO_1[res] for res in sequence])


def get_residue_id(
    df: pd.DataFrame, insertions: bool = True, unique: bool = True
) -> List[str]:
    """
    Returns a list of residue IDs from a DataFrame of a protein structure.

    Residue IDs are of the form: ``[chain_id:residue_name:residue_number]`` or
    ``[chain_id:residue_name:residue_number:insertion_code]` if
    ``insertions=True``

    E.g.

    ``["A:SER:1", "A:GLY:2", ...]`` or ``["A:SER:1:A", "A:GLY:2:", ...]``

    :param df: DataFrame of protein structure to extract residue IDs from.
    :type df: pd.DataFrame
    :param insertions: Whether or not to include insertion codes in the residue
        ID.
    :param unique: Whether or not to return only unique residue IDs. If
        ``False``, it returns a (repeated) ID for each atom in the protein. If
        ``True`` we return the unique set of residue IDs. Default is ``True``.
    :type unique: bool, optional
    :return: List of residue IDs.
    :rtype: List[str]
    """
    if "residue_id" not in df.columns:
        df["residue_id"] = (
            df.chain_id
            + ":"
            + df.residue_name
            + ":"
            + df.residue_number.astype(str)
        )
        if insertions:
            df["residue_id"] = df.residue_id + ":" + df.insertion.astype(str)
    return list(df.residue_id.unique()) if unique else list(df.residue_id)


def residue_type_tensor(
    df: pd.DataFrame,
    vocabulary: List[str] = STANDARD_AMINO_ACIDS,
    three_to_one_mapping: Dict[str, str] = RESI_THREE_TO_1,
    one_hot: bool = False,
    insertions: bool = False,
    dtype: torch.dtype = torch.long,
    device: torch.device = torch.device("cpu"),
    per_atom: bool = False,
) -> torch.Tensor:
    """Returns a tensor of the residue types in a protein structure.

    :param df: DataFrame of protein structure.
    :type df: pd.DataFrame
    :param vocabulary: List of allowable residue types, defaults to
        :ref:`graphein.protein.resi_atoms.STANDARD_AMINO_ACIDS`
    :type vocabulary: List[str], optional
    :param three_to_one_mapping: Mapping from three letter to codes to one
        letter amino acid codes, defaults to
        :ref:`graphein.protein.resi_atoms.RESI_THREE_TO_1`
    :type three_to_one_mapping: Dict[str, str], optional
    :param one_hot: Whether to return a tensor of integers denoting residue
        type or whether to one-hot encode residue types, defaults to ``False``.
    :type one_hot: bool, optional
    :param insertions: Whether or not to include insertions, defaults to
        ``False``.
    :type insertions: bool, optional
    :param dtype: torch.dtype of tensor, defaults to ``torch.long``
    :type dtype: torch.dtype, optional
    :param device: Device to create tensor on, defaults to
        ``torch.device("cpu")``
    :type device: torch.device, optional
    :return: Tensor of residue types.
    :rtype: torch.Tensor
    """
    residues = get_sequence(
        df, insertions=insertions, list_of_three=True, per_atom=per_atom
    )

    # Convert to one letter code
    residues_one = []
    for res in residues:
        res = three_to_one_mapping[res]
        if res in vocabulary:
            residues_one.append(res)
        else:
            residues_one.append("X")

    residues = torch.tensor(
        [vocabulary.index(res) for res in residues_one],
        dtype=dtype,
        device=device,
    )

    if one_hot:
        residues = F.one_hot(residues, num_classes=len(vocabulary))
    return residues


@lru_cache(maxsize=1)
def get_atom_indices(
    invert: bool = False,
) -> Union[Dict[str, Tuple[int, ...]], Dict[Tuple[int, ...], str]]:
    """
    Generates a dictionary mapping residue types to atom indices.

    :param invert: If ``True``, inverts the dictionary so that the keys are the
        atom indices and the values are the residue types, defaults to ``False``
    :type invert: bool, optional
    :return: Dictionary mapping residue types to atom indices or vice versa
    :rtype: Union[Dict[str, Tuple[int, ...]], Dict[Tuple[int, ...], str]]
    """
    index_map = {
        k: tuple(sorted([ATOM_NUMBERING_MODIFIED[i] for i in v]))
        for k, v in STANDARD_RESIDUE_ATOMS.items()
    }
    if invert:
        index_map = {v: k for k, v in index_map.items()}
    return index_map


def infer_residue_types(
    x: AtomTensor, fill_value: float = 1e-5, return_sequence: bool = True
) -> Union[str, List[str]]:
    """
    Infers residue types from atom tensor based on non-filled residues.

    .. note:: This function is not robust to structures with missing atoms.

    :param x: Tensor of shape ``(N, Num Atoms, 3)`` where ``N`` is the number
        of residues in the protein, ``Num Atoms`` is the number of atoms
        selected (default is ``37``,
        see: :ref:`graphein.protein.resi_atoms.ATOM_NUMBERING`)
    :type x: AtomTensor
    :param fill_value: Fill value used to denote the absence of an atom in ``x``
        , defaults to ``1e-5``.
    :type fill_value: float, optional
    :param return_sequence: If ``True``, returns the sequence. If ``False``,
        returns a list of three-letter residue codes, defaults to ``True``.
    :type return_sequence: bool, optional
    :return: Sequence or list of three-letter residue codes
    :rtype: Union[str, List[str]]
    """
    rmap: Dict[Tuple[int, ...], str] = get_atom_indices(invert=True)

    def _get_index(y: torch.Tensor) -> str:
        indices = (
            torch.nonzero(torch.sum(y != fill_value, dim=1))
            .squeeze(1)
            .tolist()
        )
        # remove oxt if present
        oxt_index = ATOM_NUMBERING_MODIFIED["OXT"]
        indices = tuple(i for i in indices if i != oxt_index)
        try:
            identity = rmap[indices]
        except KeyError:
            identity = "UNK"
        return identity

    seq = [_get_index(x[i, :, :]) for i in range(x.shape[0])]

    if return_sequence:
        seq = "".join([RESI_THREE_TO_1[res] for res in seq])

    return seq
