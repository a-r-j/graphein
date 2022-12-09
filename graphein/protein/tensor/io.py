"""Utilities for parsing proteins into and out of tensors."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import List

import pandas as pd
from loguru import logging as log

from graphein.utils import import_message

from ..resi_atoms import PROTEIN_ATOMS
from .types import AtomTensor

try:
    import torch
except ImportError:
    message = import_message(
        submodules="graphein.protein.tensor",
        package="torch",
        conda_channel="pytorch",
        pip_install=True
    )

def protein_df_to_tensor(
    df: pd.DataFrame,
    atoms_to_keep: List[str] = PROTEIN_ATOMS,
    insertions: bool = False
    fill_value: float = 1e-5
    ) -> AtomTensor:
    """
    Transforms a dataframe of a protein structure into a ``Length x Num_Atoms (default 37) x 3`` tensor.

    :param df: DataFrame of protein structure.
    :type df: pd.DataFrame
    :param atoms_to_keep: List of atomtypes to retain in the tensor. Defaults to :const:`~graphein.protein.resi_atoms.PROTEIN_ATOMS`
    :type atoms_to_keep: List[str]
    :param insertions: Whether or not to keep insertions. Defaults to ``False``.
    :type insertions: bool
    :param fill_value: Value to fill missing entries with. Defaults to ``1e-5``.
    :type fill_value: float
    :returns: ``Length x Num_Atoms (default 37) x 3`` tensor.
    :rtype: graphein.protein.tensor.types.AtomTensor
    """
    # Assign a residue ID.
    if "residue_id" not in df.columns:
        df["residue_id"] = df["chain_id"] + ":" + df["residue_name"] + ":" + str(df["residue_number"])
        # Add insertion code if including insertions
        if insertions:
            df["residue_id"] = df["residue_id"] + ":" + df["insertion"].astype(str)
    num_residues = len(df["residue_id"].unique())
    df = df.loc[df["atom_name"].isin(atoms_to_keep)]
    residue_indices = pd.factorize(df["residue_id"])[0]
    atom_indices = df["atom_name"].map(lambda x: atoms_to_keep.index(x)).values

    positions = torch.ones((num_residues, len(atoms_to_keep), 3)) * fill_value
    positions[residue_indices, atom_indices] = torch.tensor(df[["x_coord", "y_coord", "z_coord"]].values).float()
    return positions
