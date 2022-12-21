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


def get_example_df():
    p = PandasPdb().fetch_pdb("3ED8")
    return p.df["ATOM"]


def test_protein_df_to_chain_tensor():
    df = get_example_df()
    num_chains = len(df.chain_id.unique())

    chains = protein_df_to_chain_tensor(df, one_hot=True)
    assert chains.shape[0] == len(
        df
    ), "Number of residues and chain IDs do not match."
    assert (
        chains.shape[1] == num_chains
    ), "Number of chains do not match dimension"
    assert chains.max() == num_chains - 1, "Chain IDs are not zero-indexed."
    assert chains.min() == 0, "Chain IDs are not zero-indexed."

    chain = protein_df_to_chain_tensor(df, one_hot=False)
    assert chain.shape[0] == len(
        df
    ), "Number of residues and chain IDs do not match."
    assert chain.max() == num_chains - 1, "Chain IDs are not zero-indexed."
    assert chain.min() == 0, "Chain IDs are not zero-indexed."


def test_protein_df_to_tensor():  # sourcery skip: extract-duplicate-method
    df = get_example_df()

    positions = protein_df_to_tensor(df)
    assert positions.shape[0] == len(df), "Incorrect number of residues."
    assert positions.shape[1] == 37, "Incorrect number of atoms."
    assert positions.shape[2] == 3, "Incorrect number of coordinates."

    # Backbone only
    atoms_to_keep = ["N", "CA", "C", "O"]
    assert positions.shape[0] == len(df), "Incorrect number of residues."
    assert positions.shape[1] == 4, "Incorrect number of atoms."
    assert positions.shape[2] == 3, "Incorrect number of coordinates."
