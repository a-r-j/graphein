"""Tests for graphein.protein.features.nodes.geometry"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import operator
from functools import partial

import numpy as np
import pytest
from loguru import logger

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.features.nodes.geometry import (
    add_beta_carbon_vector,
    add_sequence_neighbour_vector,
    add_sidechain_vector,
    add_virtual_beta_carbon_vector,
)
from graphein.protein.graphs import construct_graph


def test_add_beta_carbon_vector(caplog):
    config = ProteinGraphConfig(
        edge_construction_functions=[
            partial(add_beta_carbon_vector, scale=True)
        ],
    )
    g = construct_graph(pdb_code="1lds", config=config)

    raw_pdb = g.graph["raw_pdb_df"]
    for n, d in g.nodes(data=True):
        # Check that the node has the correct attributes
        assert "c_beta_vector" in d.keys()
        # Check the vector is of the correct dimensionality
        assert d["c_beta_vector"].shape == (3,)

        # check glycines are zero
        if d["residue_name"] == "GLY":
            np.testing.assert_equal(
                d["c_beta_vector"], np.array([0.0, 0.0, 0.0])
            )
        else:
            # Check scaled vector has norm close 1
            np.testing.assert_almost_equal(
                np.linalg.norm(d["c_beta_vector"]), 1.0
            )

    # Test unscaled vector
    config = ProteinGraphConfig(
        edge_construction_functions=[
            partial(add_beta_carbon_vector, scale=False)
        ],
    )
    g = construct_graph(pdb_code="1lds", config=config)

    for n, d in g.nodes(data=True):
        # check glycines are zero
        if d["residue_name"] == "GLY":
            np.testing.assert_equal(
                d["c_beta_vector"], np.array([0.0, 0.0, 0.0])
            )
        else:
            # Check the vector is pointing in the correct direction
            cb_true = np.array(
                raw_pdb.loc[
                    (raw_pdb.node_id == n) & (raw_pdb.atom_name == "CB")
                ][["x_coord", "y_coord", "z_coord"]]
            ).T.squeeze()
            np.testing.assert_almost_equal(
                cb_true, d["coords"] + d["c_beta_vector"]
            )
    # Test altloc handling
    g = construct_graph(config=config, pdb_code="6rew")
    for n, d in g.nodes(data=True):
        assert d["c_beta_vector"].shape == (3,)

    g = construct_graph(config=config, pdb_code="7w9w")
    for n, d in g.nodes(data=True):
        assert d["c_beta_vector"].shape == (3,)

    # Test handling of missing beta-carbons
    g = construct_graph(config=config, pdb_code="3se8")

    assert "H:CYS:104" in g.nodes
    for n, d in g.nodes(data=True):
        assert d["c_beta_vector"].shape == (3,)
        if n == "H:CYS:104":
            np.testing.assert_equal(
                d["c_beta_vector"], np.array([0.0, 0.0, 0.0])
            )


def test_add_sidechain_vector():
    config = ProteinGraphConfig(
        edge_construction_functions=[
            partial(add_sidechain_vector, scale=True)
        ],
    )
    g = construct_graph(pdb_code="1lds", config=config)

    for n, d in g.nodes(data=True):
        # Check that the node has the correct attributes
        assert "sidechain_vector" in d.keys()
        # Check the vector is of the correct dimensionality
        assert d["sidechain_vector"].shape == (3,)

        # check glycines are zero
        if d["residue_name"] == "GLY":
            np.testing.assert_equal(
                d["sidechain_vector"], np.array([0.0, 0.0, 0.0])
            )
        else:
            # Check scaled vector has norm close 1
            np.testing.assert_almost_equal(
                np.linalg.norm(d["sidechain_vector"]), 1.0
            )

    # Test unscaled vector
    config = ProteinGraphConfig(
        edge_construction_functions=[
            partial(add_sidechain_vector, scale=False)
        ],
    )
    g = construct_graph(pdb_code="1lds", config=config)

    for n, d in g.nodes(data=True):
        # check glycines are zero
        if d["residue_name"] == "GLY":
            np.testing.assert_equal(
                d["sidechain_vector"], np.array([0.0, 0.0, 0.0])
            )
        else:
            # Check the vector is pointing in the correct direction
            sc_true = np.array(
                g.graph["rgroup_df"]
                .groupby("node_id")
                .mean(numeric_only=True)
                .loc[n][["x_coord", "y_coord", "z_coord"]]
            )
            np.testing.assert_almost_equal(
                sc_true, d["coords"] + d["sidechain_vector"]
            )


def test_add_virtual_beta_carbon_vector():
    config = ProteinGraphConfig(
        edge_construction_functions=[
            partial(add_virtual_beta_carbon_vector, scale=True)
        ],
    )
    g = construct_graph(pdb_code="1lds", config=config)

    raw_pdb = g.graph["raw_pdb_df"]
    for n, d in g.nodes(data=True):
        # Check that the node has the correct attributes
        assert "virtual_c_beta_vector" in d.keys()
        # Check the vector is of the correct dimensionality
        assert d["virtual_c_beta_vector"].shape == (3,)

        # check glycines are not zero
        if d["residue_name"] == "GLY":
            np.testing.assert_array_compare(
                operator.__ne__,
                d["virtual_c_beta_vector"],
                np.array([0.0, 0.0, 0.0]),
            )
        else:
            # Check scaled vector has norm close 1
            np.testing.assert_almost_equal(
                np.linalg.norm(d["virtual_c_beta_vector"]), 1.0
            )

    # Test unscaled vector
    config = ProteinGraphConfig(
        edge_construction_functions=[
            partial(add_virtual_beta_carbon_vector, scale=False)
        ],
    )
    g = construct_graph(pdb_code="1lds", config=config)

    for n, d in g.nodes(data=True):
        # check glycines are not zero
        if d["residue_name"] == "GLY":
            np.testing.assert_array_compare(
                operator.__ne__,
                d["virtual_c_beta_vector"],
                np.array([0.0, 0.0, 0.0]),
            )

    # Test altloc handling
    g = construct_graph(config=config, pdb_code="6rew")
    for n, d in g.nodes(data=True):
        assert d["virtual_c_beta_vector"].shape == (3,)

    g = construct_graph(config=config, pdb_code="7w9w")
    for n, d in g.nodes(data=True):
        assert d["virtual_c_beta_vector"].shape == (3,)


@pytest.mark.parametrize("n_to_c", [True, False])
def test_add_sequence_neighbour_vector(n_to_c):
    config = ProteinGraphConfig(edge_construction_functions=[])
    g = construct_graph(pdb_code="1igt", config=config)
    add_sequence_neighbour_vector(g, n_to_c=n_to_c)

    key = "sequence_neighbour_vector_" + ("n_to_c" if n_to_c else "c_to_n")
    for n, d in g.nodes(data=True):
        # Check that the node has the correct attributes
        assert key in d.keys()
        # Check the vector is of the correct dimensionality
        assert d[key].shape == (3,)

        # check A insertions have non-zero backward vectors
        print(n, n_to_c, d[key])
        if n.endswith(":A") and not n_to_c:
            assert np.any(np.not_equal(d[key], [0.0, 0.0, 0.0]))
