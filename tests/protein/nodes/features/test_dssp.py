# basic
import functools
import pathlib

# test
import pytest

from graphein.protein.config import DSSPConfig, ProteinGraphConfig
from graphein.protein.edges import distance as D
from graphein.protein.features.nodes import rsa, secondary_structure
from graphein.protein.features.nodes.dssp import DSSP_COLS, compute_dssp_df
from graphein.protein.graphs import construct_graph, read_pdb_to_dataframe
from graphein.protein.subgraphs import extract_surface_subgraph
from graphein.protein.utils import ProteinGraphConfigurationError

# ---------- input ----------
pdb_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "test_data"
    / "input_pdb_cryst1.pdb"
)
RSA_THRESHOLD = 0.2


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
        "graph_metadata_functions": [rsa],
    }
    config = ProteinGraphConfig(**params_to_change)
    # ---------- construct graph ----------
    return construct_graph(config=config, path=pdb_path, verbose=False)


# ---------- test: dssp DataFrame ----------
def test_assert_nonempty_dssp_df(dssp_graph):
    """if not provided dssp version to dssp.add_dssp_df, will output an empty DataFrame"""
    if dssp_graph.graph["dssp_df"].empty:
        pytest.fail("DSSP dataframe is empty")


# ---------- test: surface subgraph nodes with insertion code ----------
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


# ---------- test: native DSSP matches mkdssp ----------
# Rows of mkdssp 4.6.1's classic output for 1lds, as parsed by Biopython
# (all columns but "aa"): a terminus, two strand residues, a turn, C-terminus.
MKDSSP_1LDS_ROWS = [
    ("A", 0, " ", "-", 203, 360.0, 160.8, 1, 0, 0.0, 2, -0.3, 0, 0.0, 0, 0.0),
    (
        "A",
        8,
        " ",
        "E",
        77,
        -129.2,
        131.2,
        9,
        18,
        -2.1,
        18,
        -3.0,
        -2,
        -0.5,
        2,
        -0.5,
    ),
    (
        "A",
        22,
        " ",
        "E",
        39,
        -122.3,
        134.7,
        23,
        -10,
        -2.4,
        -10,
        -1.9,
        -2,
        -0.4,
        2,
        -0.5,
    ),
    (
        "A",
        60,
        " ",
        "T",
        131,
        -111.6,
        26.6,
        61,
        1,
        -0.1,
        2,
        -0.4,
        -29,
        -0.1,
        -1,
        -0.3,
    ),
    (
        "A",
        96,
        " ",
        "-",
        182,
        -95.6,
        360.0,
        97,
        -87,
        -0.6,
        -85,
        -0.1,
        -2,
        -0.5,
        -1,
        -0.1,
    ),
]


def test_compute_dssp_df_matches_mkdssp():
    path = pdb_path.parent / "1lds.pdb"
    df = compute_dssp_df(read_pdb_to_dataframe(path=str(path)))
    assert list(df.columns) == DSSP_COLS
    assert len(df) == 97
    rows = df.drop(columns="aa").iloc[[0, 8, 22, 60, 96]]
    assert [tuple(r) for r in rows.itertuples(index=False)] == MKDSSP_1LDS_ROWS


def test_deprecated_dssp_config_still_works():
    config = ProteinGraphConfig(
        dssp_config=DSSPConfig(),
        graph_metadata_functions=[secondary_structure],
    )
    with pytest.warns(DeprecationWarning):
        g = construct_graph(
            config=config, path=str(pdb_path.parent / "1lds.pdb")
        )
    assert all("ss" in d for _, d in g.nodes(data=True))
