"""Tests for graph format conversion procedures."""

from functools import partial

import pytest
import torch

from graphein.ml import GraphFormatConvertor
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import add_k_nn_edges
from graphein.protein.features.nodes.amino_acid import amino_acid_one_hot
from graphein.protein.graphs import construct_graph

try:
    import torch_geometric
    from torch_geometric.utils import is_undirected

    PYG_AVAIL = True
except ImportError:
    PYG_AVAIL = False


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
