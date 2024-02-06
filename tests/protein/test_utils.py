import os

import networkx as nx
from pandas.testing import assert_frame_equal

from graphein.protein.graphs import (
    construct_graph,
    filter_dataframe,
    read_pdb_to_dataframe,
)
from graphein.protein.utils import (
    download_pdb,
    download_pdb_multiprocessing,
    save_graph_to_pdb,
    save_pdb_df_to_pdb,
    save_rgroup_df_to_pdb,
)


def test_save_graph_to_pdb():
    g = construct_graph(pdb_code="4hhb")

    save_graph_to_pdb(g, "/tmp/test_graph.pdb", hetatms=False)

    a = read_pdb_to_dataframe("/tmp/test_graph.pdb")
    # Check file exists
    assert os.path.isfile("/tmp/test_graph.pdb")

    # Check for equivalence between saved and existing DFs.
    # We drop the line_idx columns as these will be renumbered
    #assert_frame_equal(
    #    a.drop(["line_idx"], axis=1),
    #    g.graph["pdb_df"].drop(["line_idx", "node_id", "residue_id"], axis=1),
    #)
    assert_frame_equal(
        a, g.graph["pdb_df"].drop(["node_id", "residue_id", axis=1])
    )
    h = construct_graph(path="/tmp/test_graph.pdb")

    # We check for isomorphism rather than equality as array features are not
    # comparable
    assert nx.is_isomorphic(g, h)


def test_save_pdb_df_to_pdb():
    g = construct_graph(pdb_code="4hhb")

    save_pdb_df_to_pdb(g.graph["pdb_df"], "/tmp/test_pdb.pdb", hetatms=False)
    a = read_pdb_to_dataframe("/tmp/test_pdb.pdb")
    # Check file exists
    assert os.path.isfile("/tmp/test_graph.pdb")

    # We drop the line_idx columns as these will be renumbered
    #assert_frame_equal(
    #    a.drop(["line_idx"], axis=1),
    #    g.graph["pdb_df"].drop(["line_idx", "node_id", "residue_id"], axis=1),
    #)
    assert_frame_equal(
        a, g.graph["pdb_df"].drop(["node_id", "residue_id", axis=1])
    )

    # Now check for raw, unprocessed DF
    save_pdb_df_to_pdb(g.graph["raw_pdb_df"], "/tmp/test_pdb.pdb")
    h = construct_graph(path="/tmp/test_pdb.pdb")

    # We check for isomorphism rather than equality as array features are not
    # comparable
    assert nx.is_isomorphic(g, h)


def test_save_rgroup_df_to_pdb():
    g = construct_graph(pdb_code="4hhb")

    save_rgroup_df_to_pdb(g, "/tmp/test_rgroup.pdb", hetatms=False)
    a = read_pdb_to_dataframe("/tmp/test_rgroup.pdb")
    # Check file exists
    assert os.path.isfile("/tmp/test_rgroup.pdb")

    # We drop the line_idx columns as these will be renumbered
    assert_frame_equal(
        a.drop(["line_idx"], axis=1),
        filter_dataframe(
            g.graph["rgroup_df"], "record_name", ["HETATM"], False
        ).drop(["line_idx", "node_id", "residue_id"], axis=1),
    )


def test_download_obsolete_structure():
    fp = download_pdb(pdb_code="116L", check_obsolete=True)
    assert os.path.exists(fp)
    assert str(fp).endswith("216l.pdb")


def test_download_structure():
    fp = download_pdb(pdb_code="4hhb")
    assert os.path.exists(fp)
    assert str(fp).endswith("4hhb.pdb")


def test_download_structure_multi():
    fps = download_pdb_multiprocessing(pdb_codes=["4hhb", "4hhb"], out_dir=".")
    for path in fps:
        assert os.path.exists(path)
        assert str(path).endswith("4hhb.pdb")


if __name__ == "__main__":
    test_save_graph_to_pdb()
    test_save_pdb_df_to_pdb()
    test_save_rgroup_df_to_pdb()
