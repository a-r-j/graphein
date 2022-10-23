"""Tests for graphein.utils.utils"""

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
        "pdb_path",
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
    assert set(graph_attrs) == set(
        DEFAULT_ATTRS
    ), "Graph attributes do not match expected attributes."


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
    assert set(node_attrs) == set(
        DEFAULT_ATTRS
    ), "Node attributes do not match expected attributes."


def test_get_edge_attribute_names():
    g = construct_graph(pdb_code="3eiy")
    DEFAULT_ATTRS = ["kind", "distance"]
    edge_attrs = get_edge_attribute_names(g)
    assert set(edge_attrs) == set(
        DEFAULT_ATTRS
    ), "Edge attributes do not match expected attributes."
