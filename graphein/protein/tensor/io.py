"""Utilities for parsing proteins into and out of tensors."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import List, Optional

import pandas as pd
from loguru import logger as log

from graphein.utils.utils import import_message

from ..graphs import (
    deprotonate_structure,
    filter_hetatms,
    read_pdb_to_dataframe,
    remove_insertions,
    select_chains,
    sort_dataframe,
)
from ..resi_atoms import PROTEIN_ATOMS
from .sequence import get_residue_id
from .types import AtomTensor, CoordTensor

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
except ImportError:
    message = import_message(
        submodule="graphein.protein.tensor",
        package="torch",
        conda_channel="pytorch",
        pip_install=True,
    )
    log.warning(message)


def get_protein_length(df: pd.DataFrame, insertions: bool = False) -> int:
    """Return the number of unique amino acids in the protein.

    Note for future development: this function may return incorrect results
    on raw PDB files containing insertions/altlocs. This should not be the case
    for proteins processed by the Graphein pre-processing workflow. Caveat emptor.

    :param df: Protein DataFrame to get length of
    :type df: pd.DataFrame
    :param insertions: Whether or not to include insertions in the length
    :type insertions: bool
    :return: Number of unique residues
    :rtype: int
    """
    # Get unique residues:
    if "residue_id" not in df.columns:
        ids = (
            df["chain_id"]
            + ":"
            + df["residue_type"]
            + ":"
            + df["residue_number"]
        )
        if insertions:
            df["residue_id"] = df.residue_id + ":" + df.insertion
    else:
        ids = df["residue_id"].values

    return len(set(ids))


def protein_to_pyg(
    pdb_path: Optional[str] = None,
    pdb_code: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    chain_selection: str = "all",
    deprotonate: bool = True,
    keep_insertions: bool = False,
    keep_hets: List[str] = [],
    model_index: int = 1,
    atom_types: List[str] = PROTEIN_ATOMS,
) -> Data:
    df = read_pdb_to_dataframe(
        pdb_path=pdb_path,
        pdb_code=pdb_code,
        uniprot_id=uniprot_id,
        model_index=model_index,
    )
    df = select_chains(df, chain_selection)
    if deprotonate:
        df = deprotonate_structure(df)
    if not keep_insertions:
        df = remove_insertions(df)
    # Remove hetatms
    hets = filter_hetatms(df, keep_hets=keep_hets)
    df = df.loc[df.record_name == "ATOM"]
    df = pd.concat([df] + hets)
    df = sort_dataframe(df)

    df["residue_id"] = (
        df["chain_id"]
        + ":"
        + df["residue_name"]
        + ":"
        + df["residue_number"].astype(str)
    )
    if keep_insertions:
        df["residue_id"] = df.residue_id + ":" + df.insertion

    from graphein.protein.tensor.sequence import residue_type_tensor

    return Data(
        coords=protein_df_to_tensor(df, atoms_to_keep=atom_types),
        residue_id=get_residue_id(df),
        residue_type=residue_type_tensor(df),
        chains=protein_df_to_chain_tensor(df),
    )


def protein_df_to_chain_tensor(
    df: pd.DataFrame,
    chains_to_keep: Optional[List[str]] = None,
    insertions: bool = False,
    one_hot: bool = False,
    dtype: torch.dtype = torch.int64,
    device: torch.device = torch.device("cpu"),
    per_atom: bool = False,
) -> torch.Tensor:
    """Returns a tensor of chain IDs for a protein structure.

    :param df: DataFrame of protein structure. Must have a column called
        ``"chain_id"`` (and ``insertion`` if the ``insertions=True``).
    :type df: pd.DataFrame
    :param chains_to_keep: List of chains to retain, defaults to ``None``
        (all chains).
    :type chains_to_keep: Optional[List[str]], optional
    :param insertions: Whether or not to keep insertions, defaults to ``False``
    :type insertions: bool, optional
    :param one_hot: Whether or not to return a one-hot encoded tensor
        (``L x num_chains``). If ``False`` an integer tensor is returned.
        Defaults to ``False``.
    :type one_hot: bool, optional
    :return: Onehot encoded or integer tensor indicating chain membership for
        each residue.
    :rtype: torch.Tensor
    """
    if "residue_id" not in df.columns:
        get_residue_id(df, insertions=insertions)

    # Select chains to keep from user input
    if chains_to_keep is not None:
        df = df.loc[df.chain_id.isin(chains_to_keep)]

    # Keep or remove insertions
    if insertions:
        df = df.loc[df.insertion.isin(["", " "])]

    if not per_atom:
        chains = pd.Series(df.residue_id.unique())
        chains = chains.str.split(":").str[0]
    else:
        chains = df.chain_id

    # One-hot encode chain IDs
    chains = pd.get_dummies(chains)
    chains = torch.tensor(chains.values, dtype=dtype, device=device)

    # Integers instead of one-hot
    if not one_hot:
        chains = torch.argmax(chains, dim=1)

    return chains


def protein_df_to_tensor(
    df: pd.DataFrame,
    atoms_to_keep: List[str] = PROTEIN_ATOMS,
    insertions: bool = False,
    fill_value: float = 1e-5,
) -> AtomTensor:
    """
    Transforms a DataFrame of a protein structure into a
    ``Length x Num_Atoms (default 37) x 3`` tensor.

    :param df: DataFrame of protein structure.
    :type df: pd.DataFrame
    :param atoms_to_keep: List of atom types to retain in the tensor.
        Defaults to :const:`~graphein.protein.resi_atoms.PROTEIN_ATOMS`
    :type atoms_to_keep: List[str]
    :param insertions: Whether or not to keep insertions. Defaults to ``False``.
    :type insertions: bool
    :param fill_value: Value to fill missing entries with. Defaults to ``1e-5``.
    :type fill_value: float
    :returns: ``Length x Num_Atoms (default 37) x 3`` tensor.
    :rtype: graphein.protein.tensor.types.AtomTensor
    """
    # Assign a residue ID.
    # if "residue_id" not in df.columns:
    #    df["residue_id"] = (
    #        df["chain_id"]
    #        + ":"
    #        + df["residue_name"]
    #        + ":"
    #        + str(df["residue_number"])
    #    )
    # Add insertion code if including insertions
    #    if insertions:
    #        df["residue_id"] = (
    #            df["residue_id"] + ":" + df["insertion"].astype(str)
    #        )
    res_ids = get_residue_id(df, insertions=insertions)
    num_residues = len(df["residue_id"].unique())
    df = df.loc[df["atom_name"].isin(atoms_to_keep)]
    residue_indices = pd.factorize(df["residue_id"])[0]
    atom_indices = df["atom_name"].map(lambda x: atoms_to_keep.index(x)).values

    positions = torch.ones((num_residues, len(atoms_to_keep), 3)) * fill_value
    positions[residue_indices, atom_indices] = torch.tensor(
        df[["x_coord", "y_coord", "z_coord"]].values
    ).float()
    return positions
