import os

import networkx as nx
from pandas.testing import assert_frame_equal

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph, read_pdb_to_dataframe
from graphein.protein.utils import (
    download_pdb,
    save_graph_to_pdb,
    save_pdb_df_to_pdb,
    save_rgroup_df_to_pdb,
)


def test_save_graph_to_pdb():
    g = construct_graph(pdb_code="4hhb")

    save_graph_to_pdb(g, "/tmp/test_graph.pdb")

    a = read_pdb_to_dataframe("/tmp/test_graph.pdb").df["ATOM"]
    # Check file exists
    assert os.path.isfile("/tmp/test_graph.pdb")

    # Check for equivalence between saved and existing DFs.
    # We drop the line_idx columns as these will be renumbered
    assert_frame_equal(
        a.drop(["line_idx"], axis=1),
        g.graph["pdb_df"].drop(["line_idx"], axis=1),
    )
    h = construct_graph(pdb_path="/tmp/test_graph.pdb")

    # We check for isomorphism rather than equality as array features are not comparable
    assert nx.is_isomorphic(g, h)


def test_save_pdb_df_to_pdb():
    g = construct_graph(pdb_code="4hhb")

    save_pdb_df_to_pdb(g.graph["pdb_df"], "/tmp/test_pdb.pdb")
    a = read_pdb_to_dataframe("/tmp/test_pdb.pdb").df["ATOM"]
    # Check file exists
    assert os.path.isfile("/tmp/test_graph.pdb")

    # We drop the line_idx columns as these will be renumbered
    assert_frame_equal(
        a.drop(["line_idx"], axis=1),
        g.graph["pdb_df"].drop(["line_idx"], axis=1),
    )

    # Now check for raw, unprocessed DF
    save_pdb_df_to_pdb(g.graph["raw_pdb_df"], "/tmp/test_pdb.pdb")
    h = construct_graph(pdb_path="/tmp/test_pdb.pdb")

    # We check for isomorphism rather than equality as array features are not comparable
    assert nx.is_isomorphic(g, h)


def test_save_rgroup_df_to_pdb():
    g = construct_graph(pdb_code="4hhb")

    save_rgroup_df_to_pdb(g, "/tmp/test_rgroup.pdb")
    a = read_pdb_to_dataframe("/tmp/test_rgroup.pdb").df["ATOM"]
    # Check file exists
    assert os.path.isfile("/tmp/test_rgroup.pdb")

    # We drop the line_idx columns as these will be renumbered
    assert_frame_equal(
        a.drop(["line_idx"], axis=1),
        g.graph["rgroup_df"].drop(["line_idx"], axis=1),
    )


def test_download_obsolete_structure():
    config = ProteinGraphConfig()
    fp = download_pdb(pdb_code="116L", config=config)
    assert str(fp).endswith("216l.pdb")


if __name__ == "__main__":
    test_save_graph_to_pdb()
    test_save_pdb_df_to_pdb()
    test_save_rgroup_df_to_pdb()
