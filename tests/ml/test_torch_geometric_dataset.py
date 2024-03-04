"""Tests for PyTorch Geometric Dataset constructors."""

import os
import shutil
from pathlib import Path

import pytest
from numpy.testing import assert_array_equal

import graphein.protein as gp
from graphein.ml import GraphFormatConvertor

try:
    import torch_geometric

    PYG_AVAIL = True
except ImportError:
    PYG_AVAIL = False

try:
    from graphein.ml import (
        InMemoryProteinGraphDataset,
        ProteinGraphDataset,
        ProteinGraphListDataset,
    )
except (NameError, ImportError):
    pass

ROOT_DIR = Path(__file__).parent


@pytest.mark.skipif(not PYG_AVAIL, reason="PyG not installed")
def test_list_dataset():
    # Construct graphs
    graphs = gp.construct_graphs_mp(
        pdb_code_it=["3eiy", "4hhb", "1lds", "2ll6"], return_dict=False
    )

    # do some transformation
    graphs = [gp.extract_subgraph_from_chains(g, ["A"]) for g in graphs]

    # Convert to PyG Data format
    convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg")
    graphs = [convertor(g) for g in graphs]

    # Create dataset
    ds = ProteinGraphListDataset(
        root=ROOT_DIR, data_list=graphs, name="list_test"
    )

    assert len(ds) == len(graphs)
    assert os.path.exists(ROOT_DIR / "processed" / "data_list_test.pt")

    for i, d in enumerate(ds):
        assert_array_equal(d.edge_index, graphs[i].edge_index)
        assert d.node_id == graphs[i].node_id
        assert_array_equal(d.coords[0], graphs[i].coords[0])
        assert d.name == graphs[i].name
        assert d.num_nodes == graphs[i].num_nodes
    # Clean up
    shutil.rmtree(ROOT_DIR / "processed")


@pytest.mark.skipif(not PYG_AVAIL, reason="PyG not installed")
def test_in_memory_dataset():
    pdb_list = ["7vxg", "7dys", "7whv", "7eek"]
    uniprots = ["P10513", "B1VC86", "P13948", "P17998"]

    ds = InMemoryProteinGraphDataset(
        root=ROOT_DIR,
        name="in_memory_test",
        pdb_codes=pdb_list,
        uniprot_ids=uniprots,
    )
    assert len(ds) == len(pdb_list + uniprots)

    # Check raw files exist
    for pdb in pdb_list + uniprots:
        assert f"{pdb}.pdb" in os.listdir(ROOT_DIR / "raw")

    # Check processed data exists
    assert os.path.exists(ROOT_DIR / "processed" / "data_in_memory_test.pt")

    # Clean up
    shutil.rmtree(ROOT_DIR / "raw")
    shutil.rmtree(ROOT_DIR / "processed")


@pytest.mark.skipif(not PYG_AVAIL, reason="PyG not installed")
def test_protein_graph_dataset():
    pdb_list = ["4rty", "4r01", "5e08", "6f15"]
    uniprots = ["A0A0A1EI90", "A0A0B4JCS5", "A0A0B4JCZ3", "A0A0B4JCZ0"]

    ds = ProteinGraphDataset(
        root=ROOT_DIR,
        pdb_codes=pdb_list,
        uniprot_ids=uniprots,
    )
    assert len(ds) == len(pdb_list + uniprots)

    # Check raw files and processed data exist
    for pdb in pdb_list + uniprots:
        assert f"{pdb}.pdb" in os.listdir(ROOT_DIR / "raw")
        assert f"{pdb}.pt" in os.listdir(ROOT_DIR / "processed")

    # Clean up
    shutil.rmtree(ROOT_DIR / "raw")
    shutil.rmtree(ROOT_DIR / "processed")
