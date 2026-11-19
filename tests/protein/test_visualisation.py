"""Tests for graphein.protein.visualisation node sizing and colouring."""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from graphein.protein.features.nodes.amino_acid import hydrophobicity
from graphein.protein.graphs import construct_graph
from graphein.protein.resi_atoms import HYDROPHOBICITY_SCALES
from graphein.protein.visualisation import (
    _node_feature_func,
    _node_size_func,
    asteroid_plot,
    plotly_protein_structure_graph,
)

PDB = Path(__file__).parent / "test_data" / "4hhb.pdb"
KD = HYDROPHOBICITY_SCALES["kd"]


@pytest.fixture(scope="module")
def g():
    return construct_graph(path=PDB, verbose=False)


def _arg(g):
    return next(n for n, d in g.nodes(data=True) if d["residue_name"] == "ARG")


@pytest.mark.parametrize(
    "feature", ["kd-hydrophobicity", "hydrophobicity", "meiler-2"]
)
def test_node_sizes_are_not_negative(g, feature):
    size = _node_size_func(g, feature, min=20, multiplier=10)
    assert min(size(n) for n in g) >= 0
    # plotly rejects negative marker sizes
    plotly_protein_structure_graph(
        g, node_size_feature=feature, node_size_multiplier=10
    )


@pytest.mark.parametrize(
    "feature", ["hydrophobicity_ww", "ww-hydrophobicity", "wwhydrophobicity"]
)
def test_hydrophobicity_scale_prefix_or_suffix(g, feature):
    # hydrophobicity(mapping="ww") stores "hydrophobicity_ww"
    assert (
        _node_feature_func(g, feature)(_arg(g))
        == (HYDROPHOBICITY_SCALES["ww"]["ARG"])
    )


def test_hydrophobicity_uses_stored_node_feature(g):
    h = g.copy()
    nx.set_node_attributes(h, 1.5, "hydrophobicity_kd")
    assert _node_feature_func(h, "hydrophobicity_kd")(_arg(h)) == 1.5


def test_hydrophobicity_of_non_standard_residues():
    h = nx.Graph()
    h.add_node("A:MSE:1", residue_name="MSE")
    h.add_node("A:UNK:2", residue_name="UNK")
    f = _node_feature_func(h, "hydrophobicity")
    assert f("A:MSE:1") == KD["MET"]  # parent residue
    assert f("A:UNK:2") == 0.0


def test_invalid_hydrophobicity_scale(g):
    with pytest.raises(KeyError, match="'xx' not a valid hydrophobicity"):
        _node_feature_func(g, "xx-hydrophobicity")


@pytest.mark.parametrize("feature", ["meiler-0", "meiler-8", "meiler-12"])
def test_invalid_meiler_dimension(g, feature):
    with pytest.raises(ValueError, match="dimensions 1-7"):
        _node_feature_func(g, feature)


def test_asteroid_plot_colour_by_distance(g):
    node = _arg(g)
    fig = asteroid_plot(g, node, colour_nodes_by="distance")
    nodes = fig.data[-1]
    colours = dict(zip(nodes.text, nodes.marker.color))
    assert colours[node] == 0
    assert all(c > 0 for n, c in colours.items() if n != node)


def test_asteroid_plot_colorbar_ticks(g):
    node = _arg(g)
    fig = asteroid_plot(g, node, k=3)
    assert fig.data[-1].marker.colorbar.tickvals == (0, 1, 2)
    fig = asteroid_plot(g, node, k=3, colour_nodes_by="bfactor")
    assert fig.data[-1].marker.colorbar.tickvals is None


@pytest.mark.parametrize("residue", ["ARG", "MSE", "UNK"])
def test_hydrophobicity_feature_has_one_shape(residue):
    d = {"residue_name": residue}
    out = hydrophobicity("A:X:1", d, mapping="ww")
    assert isinstance(out, np.ndarray) and out.shape == (1,)
    assert d["hydrophobicity_ww"] is out
    series = hydrophobicity("A:X:1", d, return_array=False)
    assert isinstance(series, pd.Series) and len(series) == 1


def test_hydrophobicity_feature_values_and_errors():
    d = {"residue_name": "MSE"}
    assert hydrophobicity("A:MSE:1", d)[0] == KD["MET"]
    with pytest.raises(ValueError, match="Unsupported mapping"):
        hydrophobicity("A:MSE:1", d, mapping="xx")
