"""Tests for graphein.protein.subgraphs"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import pickle
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pytest

from graphein.protein.config import DSSPConfig, ProteinGraphConfig
from graphein.protein.edges.distance import (
    add_ionic_interactions,
    add_peptide_bonds,
)
from graphein.protein.features.nodes import rsa, secondary_structure
from graphein.protein.graphs import construct_graph
from graphein.protein.subgraphs import (
    extract_k_hop_subgraph,
    extract_subgraph,
    extract_subgraph_by_bond_type,
    extract_subgraph_by_sequence_position,
    extract_subgraph_from_atom_types,
    extract_subgraph_from_chains,
    extract_subgraph_from_node_list,
    extract_subgraph_from_point,
    extract_subgraph_from_residue_types,
    extract_subgraph_from_secondary_structure,
    extract_surface_subgraph,
)
from graphein.utils.dependencies import is_tool

DSSP_AVAILABLE = is_tool("mkdssp")


def test_node_list_subgraphing():
    """Tests subgraph extraction from a list of nodes."""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    NODE_LIST = ["C:ALA:28", "C:ARG:31", "D:LEU:75", "A:THR:38"]

    G = construct_graph(path=str(file_path))

    g = extract_subgraph_from_node_list(G, NODE_LIST, filter_dataframe=True)

    # Check we get back a graph and it contains the correct nodes
    assert isinstance(g, nx.Graph)
    assert len(g) == len(NODE_LIST)
    for n in g.nodes():
        assert n in NODE_LIST
    assert (
        g.graph["pdb_df"]["node_id"]
        .str.contains("|".join(NODE_LIST), case=True)
        .all()
    )

    # Check the list of nodes is the same as the list of nodes in the original graph
    returned_node_list = extract_subgraph_from_node_list(
        G, NODE_LIST, return_node_list=True
    )
    assert all(elem in NODE_LIST for elem in returned_node_list)

    # Check there is no overlap when we inverse the selection
    g = extract_subgraph_from_node_list(
        G, NODE_LIST, inverse=True, filter_dataframe=True
    )
    assert len(g) == len(G) - len(NODE_LIST)
    for n in g.nodes():
        assert n not in NODE_LIST

    assert not (
        g.graph["pdb_df"]["node_id"]
        .str.contains("|".join(NODE_LIST), case=True)
        .any()
    )

    returned_node_list = extract_subgraph_from_node_list(
        G, NODE_LIST, inverse=True, return_node_list=True
    )

    assert all(elem not in NODE_LIST for elem in returned_node_list)


def test_extract_subgraph_from_atom_types():
    """Tests subgraph extraction from a list of allowed atom types"""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    G = construct_graph(path=str(file_path))

    ATOM_TYPES = ["CA"]
    g = extract_subgraph_from_atom_types(G, ATOM_TYPES, filter_dataframe=True)
    assert isinstance(g, nx.Graph)
    assert len(g) == len(G)


def test_extract_subgraph_from_residue_types():
    """Tests subgraph extraction from a list of nodes."""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    RESIDUE_TYPES = ["ALA", "SER", "GLY"]
    ALANINES = 72
    SERINES = 32
    GLYCINES = 40

    G = construct_graph(path=str(file_path))

    g = extract_subgraph_from_residue_types(
        G, RESIDUE_TYPES, filter_dataframe=True
    )

    # Check we get back a graph and it contains the correct nodes
    assert isinstance(g, nx.Graph)
    assert len(g) == ALANINES + SERINES + GLYCINES
    for n, d in g.nodes(data=True):
        assert d["residue_name"] in RESIDUE_TYPES
    assert (
        g.graph["pdb_df"]["residue_name"]
        .str.contains("|".join(RESIDUE_TYPES), case=True)
        .all()
    )

    assert (
        len([n for n, d in g.nodes(data=True) if d["residue_name"] == "ALA"])
        == ALANINES
    )
    assert (
        len([n for n, d in g.nodes(data=True) if d["residue_name"] == "GLY"])
        == GLYCINES
    )
    assert (
        len([n for n, d in g.nodes(data=True) if d["residue_name"] == "SER"])
        == SERINES
    )

    # Check the list of nodes is the same as the list of nodes in the original graph
    returned_node_list = extract_subgraph_from_node_list(
        G, RESIDUE_TYPES, return_node_list=True
    )
    assert all(elem in RESIDUE_TYPES for elem in returned_node_list)

    # Check there is no overlap when we inverse the selection
    g = extract_subgraph_from_residue_types(
        G, RESIDUE_TYPES, inverse=True, filter_dataframe=True
    )

    # assert len(g) == (len(G) - GLYCINES - ALANINES - SERINES)
    for n in g.nodes():
        assert n not in RESIDUE_TYPES

    assert not (
        g.graph["pdb_df"]["residue_name"]
        .str.contains("|".join(RESIDUE_TYPES), case=True)
        .any()
    )

    returned_node_list = extract_subgraph_from_residue_types(
        G, RESIDUE_TYPES, inverse=True, return_node_list=True
    )

    assert all(elem not in RESIDUE_TYPES for elem in returned_node_list)


def test_extract_subgraph_from_point():
    """Tests subgraph extraction from a spherical selection."""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    G = construct_graph(path=str(file_path))

    POINT = np.array([0.0, 0.0, 0.0])
    RADIUS = 10
    s_g = extract_subgraph_from_point(G, POINT, RADIUS, filter_dataframe=True)

    # Check all nodes are within the sphere
    for n, d in s_g.nodes(data=True):
        assert np.linalg.norm(d["coords"] - POINT) < RADIUS

    # Check we have extracted all the nodes
    for n, d in G.nodes(data=True):
        if np.linalg.norm(d["coords"] - POINT) < RADIUS:
            assert n in s_g.nodes()

    s_g = extract_subgraph_from_point(
        G, POINT, RADIUS, filter_dataframe=True, inverse=True
    )

    # Check all nodes are not within the sphere
    for n, d in s_g.nodes(data=True):
        assert np.linalg.norm(d["coords"] - POINT) > RADIUS

    # Check we have extracted all the nodes
    for n, d in G.nodes(data=True):
        if np.linalg.norm(d["coords"] - POINT) > RADIUS:
            assert n in s_g.nodes()


def test_extract_subgraph_from_chains():
    """Tests subgraph extraction from chains."""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    G = construct_graph(path=str(file_path))

    CHAINS = ["A", "C"]
    s_g = extract_subgraph_from_chains(G, CHAINS, filter_dataframe=True)

    # Test we only selected the correct chains
    for n, d in s_g.nodes(data=True):
        assert d["chain_id"] in CHAINS

    # Test we have extracted all the nodes
    for n, d in G.nodes(data=True):
        if d["chain_id"] in CHAINS:
            assert n in s_g.nodes()

    # Test the dataframe is correct
    assert s_g.graph["pdb_df"]["chain_id"].isin(CHAINS).all()

    s_g = extract_subgraph_from_chains(
        G, CHAINS, filter_dataframe=True, inverse=True
    )

    # Test we only selected the correct chains
    for n, d in s_g.nodes(data=True):
        assert d["chain_id"] not in CHAINS

    # Test we have extracted all the nodes
    for n, d in G.nodes(data=True):
        if d["chain_id"] in CHAINS:
            assert n not in s_g.nodes()


# @pytest.mark.skip(reason="TODO")
def test_extract_subgraph_from_sequence_position():
    """Tests subgraph extraction from sequence position."""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    G = construct_graph(path=str(file_path))

    SEQ_POS = list(range(1, 50, 2))

    s_g = extract_subgraph_by_sequence_position(
        G,
        SEQ_POS,
        filter_dataframe=True,
    )
    # Test we only selected the correct chains
    for n, d in s_g.nodes(data=True):
        assert d["residue_number"] in SEQ_POS

    # Test we have extracted all the nodes
    for n, d in G.nodes(data=True):
        if d["residue_number"] in SEQ_POS:
            assert n in s_g.nodes()

    # Test the dataframe is correct
    assert s_g.graph["pdb_df"]["residue_number"].isin(SEQ_POS).all()

    s_g = extract_subgraph_by_sequence_position(
        G, SEQ_POS, filter_dataframe=True, inverse=True
    )
    # Test we only selected the correct chains
    for n, d in s_g.nodes(data=True):
        assert d["residue_number"] not in SEQ_POS

    # Test we have extracted all the nodes
    for n, d in G.nodes(data=True):
        if d["residue_number"] in SEQ_POS:
            assert n not in s_g.nodes()


def test_extract_subgraph_from_bond_type():
    """Tests subgraph extraction from bond type"""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    config = ProteinGraphConfig(
        edge_construction_functions=[add_peptide_bonds, add_ionic_interactions]
    )
    G = construct_graph(path=str(file_path))  # , config=config)

    BOND_TYPES = ["ionic"]

    s_g = extract_subgraph_by_bond_type(G, BOND_TYPES, filter_dataframe=True)

    for u, v, d in G.edges(data=True):
        if d["kind"] in BOND_TYPES:
            assert u in s_g.nodes()
            assert v in s_g.nodes()
            assert (u, v) in s_g.edges()

    for u, v, d in s_g.edges(data=True):
        for bond in list(d["kind"]):
            assert bond in BOND_TYPES

    s_g = extract_subgraph_by_bond_type(
        G, BOND_TYPES, filter_dataframe=True, inverse=True
    )

    for u, v, d in G.edges(data=True):
        if d["kind"] in BOND_TYPES:
            assert (u, v) not in s_g.edges()

    for u, v, d in s_g.edges(data=True):
        for bond in list(d["kind"]):
            assert bond not in BOND_TYPES


def test_extract_k_hop_subgraph():
    """Tests k-hop subgraph extraction."""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    G = construct_graph(path=str(file_path))

    CENTRAL_NODE = "B:SER:49"
    K = 1
    s_g = extract_k_hop_subgraph(G, CENTRAL_NODE, K, filter_dataframe=True)

    for n in s_g.nodes():
        if n != CENTRAL_NODE:
            assert n in list(G.neighbors(CENTRAL_NODE))

    for n in list(G.neighbors(CENTRAL_NODE)):
        assert n in s_g.nodes()


@pytest.mark.skipif(not DSSP_AVAILABLE, reason="DSSP not installed.")
def test_surface_subgraph():
    """Tests surface subgraph extraction."""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    config = ProteinGraphConfig(
        graph_metadata_functions=[rsa], dssp_config=DSSPConfig()
    )
    G = construct_graph(path=str(file_path), config=config)

    RSA_THRESHOLD: float = 0.2
    s_g = extract_surface_subgraph(G, RSA_THRESHOLD, filter_dataframe=True)

    for n, d in s_g.nodes(data=True):
        assert d["rsa"] >= RSA_THRESHOLD

    for n, d in G.nodes(data=True):
        if d["rsa"] >= RSA_THRESHOLD:
            assert n in s_g.nodes(), print(n, d)


@pytest.mark.skipif(not DSSP_AVAILABLE, reason="DSSP not installed.")
def test_secondary_structure_subgraph():
    """Tests secondary subgraph extraction."""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    config = ProteinGraphConfig(
        graph_metadata_functions=[secondary_structure],
        dssp_config=DSSPConfig(),
    )
    G = construct_graph(path=str(file_path), config=config)

    SS_ELEMENTS: List[str] = ["H"]
    s_g = extract_subgraph_from_secondary_structure(
        G, SS_ELEMENTS, filter_dataframe=True
    )

    for _, d in s_g.nodes(data=True):
        assert d["ss"] in SS_ELEMENTS

    for n, d in G.nodes(data=True):
        if d["ss"] in SS_ELEMENTS:
            assert n in s_g.nodes()


@pytest.mark.skipif(not DSSP_AVAILABLE, reason="DSSP not installed.")
def test_successful_pickle():
    """Tests subgraphs can be successfully pickled and unpickled"""
    file_path = Path(__file__).parent / "test_data/4hhb.pdb"
    config = ProteinGraphConfig(
        graph_metadata_functions=[secondary_structure],
        dssp_config=DSSPConfig(),
    )
    G = construct_graph(path=str(file_path), config=config)
    s_g = extract_subgraph_from_residue_types(
        G,
        residue_types=["ALA", "SER", "MET"],
        update_coords=True,
        filter_dataframe=True,
        recompute_distmat=True,
    )

    with open("/tmp/test_graph.p", "wb") as f:
        pickle.dump(s_g, f)

    with open("/tmp/test_graph.p", "rb") as f:
        loaded_graph = pickle.load(f)

    assert nx.is_isomorphic(s_g, loaded_graph)
