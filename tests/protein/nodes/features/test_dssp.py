# basic
import functools
import pathlib

# test
import pytest

from graphein.protein.config import DSSPConfig, ProteinGraphConfig
from graphein.protein.edges import distance as D
from graphein.protein.features.nodes import rsa
from graphein.protein.graphs import construct_graph
from graphein.protein.subgraphs import extract_surface_subgraph
from graphein.protein.utils import ProteinGraphConfigurationError

# graphein
from graphein.utils.dependencies import is_tool

# ---------- input ----------
pdb_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "test_data"
    / "input_pdb_cryst1.pdb"
)
dssp_exe = "mkdssp"
RSA_THRESHOLD = 0.2

DSSP_AVAILABLE = is_tool("mkdssp")


@pytest.fixture
def dssp_graph():
    # ---------- graph config ----------
    params_to_change = {
        "granularity": "centroids",  # "atom", "CA", "centroids"
        "insertions": True,
        "edge_construction_functions": [
            # graphein.protein.edges.distance.add_peptide_bonds,
            D.add_distance_to_edges,
            D.add_hydrogen_bond_interactions,
            D.add_ionic_interactions,
            D.add_backbone_carbonyl_carbonyl_interactions,
            D.add_salt_bridges,
            # distance
            functools.partial(
                D.add_distance_threshold,
                long_interaction_threshold=4,
                threshold=4.5,
            ),
        ],
        "dssp_config": DSSPConfig(executable=dssp_exe),
        "graph_metadata_functions": [rsa],
    }
    config = ProteinGraphConfig(**params_to_change)
    # ---------- construct graph ----------
    return construct_graph(config=config, path=pdb_path, verbose=False)


# ---------- test: dssp DataFrame ----------
@pytest.mark.skipif(not DSSP_AVAILABLE, reason="DSSP not available")
def test_assert_nonempty_dssp_df(dssp_graph):
    """if not provided dssp version to dssp.add_dssp_df, will output an empty DataFrame"""
    if dssp_graph.graph["dssp_df"].empty:
        pytest.fail("DSSP dataframe is empty")


# ---------- test: surface subgraph nodes with insertion code ----------
@pytest.mark.skipif(not DSSP_AVAILABLE, reason="DSSP not available")
def test_extract_surface_subgraph_insertion_node(dssp_graph):
    """if not added insertion codes, will raise ProteinGraphConfigurationError"""
    try:
        # without the modification, the following line will raise
        # ProteinGraphConfigurationError RSA not defined for all nodes (H:TYR:52:A).
        s_g = extract_surface_subgraph(dssp_graph, RSA_THRESHOLD)
    except ProteinGraphConfigurationError as e:
        pytest.fail(
            "extract_surface_subgraph raised ProteinGraphConfigurationError:\n{e}"
        )
