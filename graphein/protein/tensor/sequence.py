"""Utilities for working with protein sequences."""
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
import torch.nn.functional as F

from graphein.protein.resi_atoms import RESI_THREE_TO_1, STANDARD_AMINO_ACIDS


def get_sequence(
    df: pd.DataFrame,
    chains: Optional[List[str]] = None,
    insertions: bool = False,
    list_of_three: bool = False,
    three_to_one_map: Optional[str] = None,
    per_atom: bool = False,
) -> Union[str, List[str]]:
    """Retrieves the amino acid sequence from a DataFrame of a protein structure.

    :param df: DataFrame of protein structure.
    :type df: pd.DataFrame
    :param chains: List of chains to retain, defaults to ``None`` (all chains).
    :type chains: Optional[List[str]], optional
    :param insertions: Whether or not to keep insertions, defaults to ``False``
    :type insertions: bool, optional
    :param list_of_three: Whether or not to return a list of three letter codes. If ``False``,
        returns the sequence as a one-letter code string. Defaults to ``False``.
    :type list_of_three: bool, optional
    :return: Amino acid sequence; either as list of three-letter codes
        (``["ALA", "GLY", "TRP"...]``; ``list_of_three=True``) or string
        (``AGY..``; ``list_of_three=False``).
    :rtype: Union[str, List[str]]
    """
    # Select chains
    if chains is not None:
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
    sequence = [res.split(":")[1] for res in df.residue_id.unique()]

    # Convert to one letter code
    if list_of_three:
        # if three_to_one_map is not None:
        # map = RESI_THREE_TO_1
        # sequence = [three_to_one_map[res] for res in sequence]
        return sequence
    else:
        return "".join([RESI_THREE_TO_1[res] for res in sequence])


def get_residue_id(df: pd.DataFrame, insertions: bool = False) -> List[str]:
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
    return list(df.residue_id.unique())


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
    :param vocabulary: List of allowable residue types, defaults to graphein.protein.resi_atoms.STANDARD_AMINO_ACIDS
    :type vocabulary: List[str], optional
    :param three_to_one_mapping: Mapping from three letter to codes to one letter amino acid codes,
        defaults to graphein.protein.RESI_THREE_TO_1
    :type three_to_one_mapping: Dict[str, str], optional
    :param one_hot: Whether to return a tensor of integers denoting residue type or whether to one-hot encode residue types,
        defaults to ``False``.
    :type one_hot: bool, optional
    :param insertions: Whether or not to include insertions, defaults to ``False``.
    :type insertions: bool, optional
    :param dtype: torch.dtype of tensor, defaults to ``torch.long``
    :type dtype: torch.dtype, optional
    :param device: Device to create tensor on, defaults to ``torch.device("cpu")``
    :type device: torch.device, optional
    :return: Tensor of residue types.
    :rtype: torch.Tensor
    """
    residues = get_sequence(df, insertions=insertions, list_of_three=True)

    # Convert to one letter code
    residues = [three_to_one_mapping[res] for res in residues]

    residues = torch.tensor(
        [vocabulary.index(res) for res in residues],
        dtype=dtype,
        device=device,
    )

    if one_hot:
        residues = F.one_hot(residues, num_classes=len(vocabulary))
    return residues
