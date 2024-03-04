"""Tests for graphein.testing.utils"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from graphein.protein import construct_graph
from graphein.testing.utils import (
    compare_approximate,
    compare_exact,
    edge_data_equal,
    edges_equal,
    graphs_equal,
    graphs_isomorphic,
    node_data_equal,
    nodes_equal,
)


def test_graph_isomorphism():
    """Test that the graph isomorphism is working."""
    file_path = Path(__file__).parent.parent / "protein" / "test_data/4hhb.pdb"
    G_1 = construct_graph(path=str(file_path))
    G_2 = construct_graph(path=str(file_path))
    assert graphs_isomorphic(G_1, G_2), "Graphs are not isomorphic."
    assert graphs_equal(G_1, G_2), "Graphs are not equal."
    assert nodes_equal(G_1, G_2), "Graphs do not contain the same nodes."
    assert node_data_equal(G_1, G_2), "Node features differ."
    assert edges_equal(G_1, G_2), "Graphs do not contain the same edges."
    assert edge_data_equal(G_1, G_2), "Edge features differ."

    file_path = Path(__file__).parent.parent / "protein" / "test_data/1lds.pdb"
    G_2 = construct_graph(path=str(file_path))
    assert not graphs_isomorphic(G_1, G_2), "Graphs not isomorphic."
    assert not graphs_equal(G_1, G_2), "Graphs are equal."
    assert not nodes_equal(G_1, G_2), "Graphs contain the same nodes."
    assert not node_data_equal(G_1, G_2), "Node features do not differ."
    assert not edges_equal(G_1, G_2), "Graphs contain the same edges."
    assert not edge_data_equal(G_1, G_2), "Edge features do not differ."


def test_dictionaries_equal():
    dict_1 = {
        "a": 1,
        "b": 2,
        "c": np.array([0, 1, 2]),
        "d": pd.DataFrame({"a": [1, 2, 3]}),
    }
    dict_2 = {
        "a": 1,
        "b": 2,
        "c": np.array([0, 1, 3]),
        "d": pd.DataFrame({"a": [1, 2, 3]}),
    }
    dict_3 = {
        "a": 1,
        "b": 2,
        "c": np.array([0, 1, 2]),
        "d": pd.DataFrame({"a": [1, 2, 4]}),
    }

    assert compare_exact(dict_1, dict_1), "Dictionaries are not equal."
    assert compare_approximate(dict_1, dict_1), "Dictionaries are not equal."

    assert not compare_exact(dict_1, dict_2), "Dictionaries are equal."
    assert not compare_exact(dict_1, dict_3), "Dictionaries are equal."
    assert not compare_exact(dict_2, dict_3), "Dictionaries are equal."

    assert not compare_approximate(dict_1, dict_2), "Dictionaries are equal."
    assert not compare_approximate(dict_1, dict_3), "Dictionaries are equal."
    assert not compare_approximate(dict_2, dict_3), "Dictionaries are equal."
