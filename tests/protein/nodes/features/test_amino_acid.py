"""Tests for graphein.protein.features.nodes.amino_acids"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from functools import partial

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.features.nodes.amino_acid import (
    amino_acid_one_hot,
    expasy_protein_scale,
    hydrogen_bond_acceptor,
    hydrogen_bond_donor,
    load_expasy_scales,
)
from graphein.protein.graphs import construct_graph
from graphein.protein.resi_atoms import RESI_THREE_TO_1


def test_load_expasy_scale():
    """Example-based test for `load_expasy_scales`."""
    scales = load_expasy_scales()
    scale = expasy_protein_scale(n="A13LEU", d={"residue_name": "LEU"})

    assert_series_equal(scale, scales["LEU"])


def test_load_meilier_embeddings():
    """Example-based test for `load_meiler_embeddings`."""
    # The test implemented here should test that something about the meiler_embeddings csv file is true.


# An execution test is one that simply tests that the function executes.
# In other words, the _only_ thing we are guaranteeing here
# is that the function will execute without erroring out.
# We are not guaranteeing the correctness of the output.
# This can be modified.
def test_expasy_protein_scale():
    """Execution test for `expasy_protein_scale` function."""
    d = {"residue_name": "LEU"}
    n = "DUMMY"
    expasy_protein_scale(n, d)


def test_amino_acid_one_hot_execution():
    """Execution test for `amino_acid_one_hot` function."""
    d = {"residue_name": "LEU"}
    n = "DUMMY"
    amino_acid_one_hot(n, d)


def test_amino_acid_one_hot_example():
    """Example-based test on 4hhb for `amino_acid_onehot`."""

    # Test np array
    config = ProteinGraphConfig(node_metadata_functions=[amino_acid_one_hot])
    g = construct_graph(pdb_code="4hhb", config=config)

    for n, d in g.nodes(data=True):
        assert sum(d["amino_acid_one_hot"]) == 1

    # Test pd.Series
    config = ProteinGraphConfig(
        node_metadata_functions=[
            partial(amino_acid_one_hot, return_array=False)
        ]
    )
    g = construct_graph(pdb_code="4hhb", config=config)

    for n, d in g.nodes(data=True):
        assert sum(d["amino_acid_one_hot"]) == 1
        assert (
            d["amino_acid_one_hot"].idxmax()
            == RESI_THREE_TO_1[d["residue_name"]]
        )


def test_hydrogen_bond_acceptor():
    config = ProteinGraphConfig(
        node_metadata_functions=[hydrogen_bond_acceptor]
    )
    g = construct_graph(pdb_code="4hhb", config=config)

    for _, d in g.nodes(data=True):
        assert "hbond_acceptors" in d.keys()
        assert isinstance(d["hbond_acceptors"], pd.Series)

    config = ProteinGraphConfig(
        node_metadata_functions=[
            partial(hydrogen_bond_acceptor, return_array=True)
        ]
    )
    g = construct_graph(pdb_code="4hhb", config=config)

    for _, d in g.nodes(data=True):
        assert "hbond_acceptors" in d.keys()
        assert isinstance(d["hbond_acceptors"], np.ndarray)

    config = ProteinGraphConfig(
        node_metadata_functions=[
            partial(
                hydrogen_bond_acceptor, return_array=True, sum_features=False
            )
        ]
    )
    g = construct_graph(pdb_code="4hhb", config=config)
    for _, d in g.nodes(data=True):
        assert "hbond_acceptors" in d.keys()
        assert isinstance(d["hbond_acceptors"], np.ndarray)
        assert d["hbond_acceptors"] < 2

    config = ProteinGraphConfig(
        node_metadata_functions=[hydrogen_bond_acceptor], granularity="atom"
    )
    g = construct_graph(pdb_code="4hhb", config=config)

    for _, d in g.nodes(data=True):
        assert "hbond_acceptors" in d.keys()
        assert isinstance(d["hbond_acceptors"], pd.Series)

    config = ProteinGraphConfig(
        node_metadata_functions=[
            partial(hydrogen_bond_acceptor, return_array=True)
        ],
        granularity="atom",
    )
    g = construct_graph(pdb_code="4hhb", config=config)

    for _, d in g.nodes(data=True):
        assert "hbond_acceptors" in d.keys()
        assert isinstance(d["hbond_acceptors"], np.ndarray)

    config = ProteinGraphConfig(
        node_metadata_functions=[
            partial(
                hydrogen_bond_acceptor, return_array=True, sum_features=False
            )
        ],
        granularity="atom",
    )
    g = construct_graph(pdb_code="4hhb", config=config)
    for _, d in g.nodes(data=True):
        assert "hbond_acceptors" in d.keys()
        assert isinstance(d["hbond_acceptors"], np.ndarray)
        assert d["hbond_acceptors"] < 2


def test_hydrogen_bond_donor():
    config = ProteinGraphConfig(node_metadata_functions=[hydrogen_bond_donor])
    g = construct_graph(pdb_code="4hhb", config=config)

    for _, d in g.nodes(data=True):
        assert "hbond_donors" in d.keys()
        assert isinstance(d["hbond_donors"], pd.Series)

    config = ProteinGraphConfig(
        node_metadata_functions=[
            partial(hydrogen_bond_donor, return_array=True)
        ]
    )
    g = construct_graph(pdb_code="4hhb", config=config)

    for _, d in g.nodes(data=True):
        assert "hbond_donors" in d.keys()
        assert isinstance(d["hbond_donors"], np.ndarray)

    config = ProteinGraphConfig(
        node_metadata_functions=[
            partial(hydrogen_bond_donor, return_array=True, sum_features=False)
        ]
    )
    g = construct_graph(pdb_code="4hhb", config=config)
    for _, d in g.nodes(data=True):
        assert "hbond_donors" in d.keys()
        assert isinstance(d["hbond_donors"], np.ndarray)
        assert d["hbond_donors"] < 2

    # Atom graphs
    config = ProteinGraphConfig(
        node_metadata_functions=[hydrogen_bond_donor], granularity="atom"
    )
    g = construct_graph(pdb_code="4hhb", config=config)

    for _, d in g.nodes(data=True):
        assert "hbond_donors" in d.keys()
        assert isinstance(d["hbond_donors"], pd.Series)

    config = ProteinGraphConfig(
        node_metadata_functions=[
            partial(hydrogen_bond_donor, return_array=True)
        ],
        granularity="atom",
    )
    g = construct_graph(pdb_code="4hhb", config=config)

    for _, d in g.nodes(data=True):
        assert "hbond_donors" in d.keys()
        assert isinstance(d["hbond_donors"], np.ndarray)

    config = ProteinGraphConfig(
        node_metadata_functions=[
            partial(hydrogen_bond_donor, return_array=True, sum_features=False)
        ],
        granularity="atom",
    )
    g = construct_graph(pdb_code="4hhb", config=config)
    for _, d in g.nodes(data=True):
        assert "hbond_donors" in d.keys()
        assert isinstance(d["hbond_donors"], np.ndarray)
        assert d["hbond_donors"] < 2


# def test_aaindex_1_feat():
#     """Execution test for `aaindex_1_feat`."""
#     d = {"residue_name": "LEU"}
#     n = "DUMMY"
#     aaindex_1_feat(n, d, feature_name="KRIW790103")
