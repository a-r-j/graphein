"""Tests for graphein.protein.tensor.io"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from biopandas.pdb import PandasPdb

from graphein.protein.tensor.io import (
    protein_df_to_chain_tensor,
    protein_df_to_tensor,
)
from graphein.protein.tensor.sequence import get_residue_id


def get_example_df():
    p = PandasPdb().fetch_pdb("3EIY")
    return p.df["ATOM"]


def test_protein_df_to_chain_tensor():
    df = get_example_df()
    num_chains = len(df.chain_id.unique())

    num_residues = len(get_residue_id(df))

    chains = protein_df_to_chain_tensor(df, one_hot=True)
    print(chains)
    assert (
        chains.shape[0] == num_residues
    ), "Number of residues and chain IDs do not match."
    assert (
        chains.shape[1] == num_chains
    ), "Number of chains do not match dimension"
    assert chains.max() == 1, "Chain IDs are not one hot."
    if num_chains > 1:
        assert chains.min() == 0, "Chain IDs are not one hot."

    chain = protein_df_to_chain_tensor(df, one_hot=False)
    assert (
        chain.shape[0] == num_residues
    ), "Number of residues and chain IDs do not match."
    assert chain.max() == num_chains - 1, "Chain IDs are not zero-indexed."
    assert chain.min() == 0, "Chain IDs are not zero-indexed."


def test_protein_df_to_tensor():  # sourcery skip: extract-duplicate-method
    df = get_example_df()

    num_residues = len(get_residue_id(df))
    print(get_residue_id(df))
    assert num_residues == 174, "Incorrect number of residues."

    positions = protein_df_to_tensor(df)
    assert positions.shape[0] == num_residues, "Incorrect number of residues."
    assert positions.shape[1] == 37, "Incorrect number of atoms."
    assert positions.shape[2] == 3, "Incorrect number of coordinates."

    # Backbone only
    atoms_to_keep = ["N", "CA", "C", "O"]
    positions = protein_df_to_tensor(df, atoms_to_keep=atoms_to_keep)
    assert positions.shape[0] == num_residues, "Incorrect number of residues."
    assert positions.shape[1] == 4, "Incorrect number of atoms."
    assert positions.shape[2] == 3, "Incorrect number of coordinates."
