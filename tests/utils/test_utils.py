"""Tests for graphein.utils.utils"""

import networkx as nx

from graphein.protein.graphs import construct_graph
from graphein.utils.utils import (
    get_edge_attribute_names,
    get_graph_attribute_names,
    get_node_attribute_names,
)


def test_get_graph_attribute_names():
    g = construct_graph(pdb_code="3eiy")
    DEFAULT_ATTRS = [
        "name",
        "pdb_code",
        "path",
        "chain_ids",
        "pdb_df",
        "raw_pdb_df",
        "rgroup_df",
        "coords",
        "node_type",
        "sequence_A",
        "config",
        "dist_mat",
    ]
    graph_attrs = get_graph_attribute_names(g)
    assert set(graph_attrs) == set(DEFAULT_ATTRS), (
        "Graph attributes do not match expected attributes."
    )


def test_get_node_attribute_names():
    g = construct_graph(pdb_code="3eiy")
    DEFAULT_ATTRS = [
        "chain_id",
        "residue_name",
        "residue_number",
        "atom_type",
        "element_symbol",
        "coords",
        "b_factor",
        "meiler",
    ]
    node_attrs = get_node_attribute_names(g)
    assert set(node_attrs) == set(DEFAULT_ATTRS), (
        "Node attributes do not match expected attributes."
    )


def test_get_edge_attribute_names():
    g = construct_graph(pdb_code="3eiy")
    DEFAULT_ATTRS = ["kind", "distance"]
    edge_attrs = get_edge_attribute_names(g)
    assert set(edge_attrs) == set(DEFAULT_ATTRS), (
        "Edge attributes do not match expected attributes."
    )


def test_attribute_names_with_differing_attributes():
    # Nodes and edges often carry different attributes, e.g. edges from
    # different edge functions.
    g = nx.Graph()
    g.add_node(1, a=1)
    g.add_node(2, a=1, b=2)
    g.add_edge(1, 2, kind={"peptide_bond"})
    g.add_edge(2, 3, kind={"hbond"}, distance=3.0)
    assert sorted(get_node_attribute_names(g)) == ["a", "b"]
    assert sorted(get_edge_attribute_names(g)) == ["distance", "kind"]
