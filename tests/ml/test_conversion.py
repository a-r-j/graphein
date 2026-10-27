"""Tests for graph format conversion procedures."""

from functools import partial
from pathlib import Path

import pytest
import torch

from graphein.ml import GraphFormatConvertor
from graphein.ml.conversion import convert_nx_to_pyg_data
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import (
    add_distance_to_edges,
    add_k_nn_edges,
    add_peptide_bonds,
)
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.graphs import construct_graph

try:
    import torch_geometric
    from torch_geometric.utils import is_undirected

    PYG_AVAIL = True
except ImportError:
    PYG_AVAIL = False

try:
    import dgl  # noqa: F401

    DGL_AVAIL = True
except (ImportError, FileNotFoundError):
    DGL_AVAIL = False

PDB_PATH = (
    Path(__file__).resolve().parent.parent
    / "protein"
    / "test_data"
    / "4hhb.pdb"
)


@pytest.mark.skipif(not PYG_AVAIL, reason="PyG not installed")
@pytest.mark.parametrize("pdb_code", ["10gs", "1bui", "1cw3"])
def test_nx_to_pyg(pdb_code):
    # Construct graph of a multimer protein complex
    edge_funcs = {
        "edge_construction_functions": [
            partial(
                add_k_nn_edges,
                k=1,
                long_interaction_threshold=0,
                exclude_edges=["inter"],
                kind_name="intra",
            ),
            partial(
                add_k_nn_edges,
                k=1,
                long_interaction_threshold=0,
                exclude_edges=["intra"],
                kind_name="inter",
            ),
        ]
    }
    node_feature_funcs = {"node_metadata_functions": [amino_acid_one_hot]}
    config = ProteinGraphConfig(**edge_funcs, **node_feature_funcs)
    g = construct_graph(config=config, pdb_code=pdb_code)

    # Convert to PyG
    convertor = GraphFormatConvertor(
        src_format="nx",
        dst_format="pyg",
        columns=[
            "coords",
            "node_id",
            "amino_acid_one_hot",
            "edge_index_inter",
            "edge_index_intra",
            "edge_index",
        ],
    )
    data = convertor(g)

    # Test
    # Nodes
    assert len(data.node_id) == data.num_nodes

    # Coordinates
    assert isinstance(data.coords, torch.Tensor)
    assert data.coords.shape == torch.Size([data.num_nodes, 3])

    # Features
    assert isinstance(data.amino_acid_one_hot, torch.Tensor)
    assert data.amino_acid_one_hot.shape == torch.Size([data.num_nodes, 20])

    # Edges
    assert isinstance(data.edge_index, torch.Tensor)
    assert data.edge_index.shape[0] == 2
    assert isinstance(data.edge_index_inter, torch.Tensor)
    assert data.edge_index_inter.shape[0] == 2
    assert isinstance(data.edge_index_intra, torch.Tensor)
    assert data.edge_index_intra.shape[0] == 2
    assert (
        data.edge_index.shape[1]
        == data.edge_index_inter.shape[1] + data.edge_index_intra.shape[1]
    )

    # Directed/undirected consistency
    assert g.is_directed() is not is_undirected(data.edge_index)


def _multi_kind_graph():
    """4HHB with peptide bond and kNN edges (some edges have both kinds)."""
    config = ProteinGraphConfig(
        edge_construction_functions=[
            add_peptide_bonds,
            partial(add_k_nn_edges, k=3),
        ]
    )
    g = construct_graph(config=config, path=str(PDB_PATH))
    return add_distance_to_edges(g)


@pytest.mark.skipif(not PYG_AVAIL, reason="PyG not installed")
@pytest.mark.parametrize("directed", [False, True])
def test_nx_to_pyg_edge_features_aligned(directed):
    """Edge features (incl. non-tensor ``kind``) follow ``edge_index``.

    Regression test for #392: ``to_undirected`` doubled and reordered
    ``edge_index`` but not ``kind``.
    """
    g = _multi_kind_graph()
    if directed:
        g = g.to_directed()
    data = GraphFormatConvertor(
        "nx", "pyg", columns=["edge_index", "kind", "distance", "node_id"]
    )(g)

    n_edges = data.edge_index.shape[1]
    assert n_edges == (1 if directed else 2) * g.number_of_edges()
    assert len(data.kind) == n_edges
    assert data.distance.shape == torch.Size([n_edges])
    for k, (u, v) in enumerate(data.edge_index.t().tolist()):
        edge = g.edges[data.node_id[u], data.node_id[v]]
        assert set(data.kind[k]) == edge["kind"]
        assert data.distance[k].item() == pytest.approx(edge["distance"])


@pytest.mark.skipif(not PYG_AVAIL, reason="PyG not installed")
def test_nx_to_pyg_self_loop_and_missing_kind():
    """Self-loop features aren't doubled when symmetrizing, and edges
    without a ``kind`` don't break the conversion."""
    g = _multi_kind_graph()
    n = next(iter(g.nodes))
    g.add_edge(n, n, kind={"self"}, distance=1.5)
    u, v = next((u, v) for u, v in g.edges if u != v)
    del g.edges[u, v]["kind"]
    data = GraphFormatConvertor(
        "nx", "pyg", columns=["edge_index", "kind", "distance", "node_id"]
    )(g)
    loops = (data.edge_index[0] == data.edge_index[1]).nonzero().flatten()
    assert loops.tolist() and all(
        data.distance[i].item() == pytest.approx(1.5) for i in loops
    )
    assert sum(k is None for k in data.kind) == 2  # both directions of (u, v)


class _ReverseSet(set):
    """A set that iterates in reverse-sorted order, to make set iteration
    order (normally hash-seed dependent) deterministic in tests."""

    def __iter__(self):
        return iter(sorted(super().__iter__(), reverse=True))


@pytest.mark.skipif(not PYG_AVAIL, reason="PyG not installed")
def test_nx_to_pyg_edge_index_by_kind():
    """``edge_index_<kind>`` keys use the sorted kinds, so they don't depend
    on set iteration order."""
    g = _multi_kind_graph()
    for _, _, d in g.edges(data=True):
        d["kind"] = _ReverseSet(d["kind"])
    kinds = {"_".join(sorted(d["kind"])) for _, _, d in g.edges(data=True)}
    assert "knn_peptide_bond" in kinds
    data = GraphFormatConvertor(
        "nx",
        "pyg",
        columns=["edge_index", "kind"] + [f"edge_index_{k}" for k in kinds],
    )(g)
    for kind in kinds:
        split = getattr(data, f"edge_index_{kind}")
        expected = sum(
            "_".join(sorted(d["kind"])) == kind
            for _, _, d in g.edges(data=True)
        )
        assert split.shape == torch.Size([2, 2 * expected])


@pytest.mark.skipif(not PYG_AVAIL, reason="PyG not installed")
def test_convert_nx_to_pyg_data_edge_features():
    """One entry per edge, not flattened (#217); numeric features work."""
    g = _multi_kind_graph()
    data = convert_nx_to_pyg_data(g)
    edges = list(g.edges(data=True))
    assert data.edge_index.shape == torch.Size([2, len(edges)])
    assert len(data.kind) == len(edges)
    for k, (_, _, d) in enumerate(edges):
        assert data.kind[k] == d["kind"]
        assert float(data.distance[k]) == pytest.approx(d["distance"])


@pytest.mark.skipif(not DGL_AVAIL, reason="DGL not installed")
@pytest.mark.parametrize("verbose", ["gnn", "default", "all_info"])
def test_nx_to_dgl(verbose):
    g = construct_graph(path=str(PDB_PATH))
    data = GraphFormatConvertor("nx", "dgl", verbose=verbose)(g)

    assert data.num_nodes() == g.number_of_nodes()
    assert data.num_edges() == g.number_of_edges()
    assert data.ndata["coords"].shape == torch.Size([data.num_nodes(), 3])
    # Node features must not be attached to edges
    assert not data.edata


@pytest.mark.skipif(not DGL_AVAIL, reason="DGL not installed")
def test_nx_to_dgl_edge_features():
    g = construct_graph(path=str(PDB_PATH))
    for i, (_, _, d) in enumerate(g.edges(data=True)):
        d["b_factor"] = float(i)
    data = GraphFormatConvertor(
        "nx", "dgl", columns=["coords", "b_factor", "distance", "kind"]
    )(g)

    assert data.edata["b_factor"].tolist() == [
        float(i) for i in range(g.number_of_edges())
    ]
    assert data.edata["distance"].tolist() == pytest.approx(
        [d["distance"] for _, _, d in g.edges(data=True)]
    )
    assert data.ndata["b_factor"].shape == torch.Size([data.num_nodes()])
