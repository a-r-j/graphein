import urllib.error
from functools import partial

import pytest

from graphein.grn.config import GRNGraphConfig
from graphein.grn.edges import add_regnetwork_edges, add_trrust_edges
from graphein.grn.features.node_features import add_sequence_to_nodes
from graphein.grn.graphs import compute_grn_graph

GENE_LIST = ["AATF", "MYC", "USF1", "SP1", "TP53", "DUSP1"]


def test_construct_graph():
    config = GRNGraphConfig()

    # RegNetwork Web is intermittently down. It sits behind Cloudflare, which
    # answers pings even then, so skip when the download itself fails.
    try:
        g = compute_grn_graph(
            gene_list=GENE_LIST,
            edge_construction_funcs=[
                partial(
                    add_trrust_edges,
                    trrust_filtering_funcs=config.trrust_config.filtering_functions,
                ),
                partial(
                    add_regnetwork_edges,
                    regnetwork_filtering_funcs=config.regnetwork_config.filtering_functions,
                ),
            ],
            node_annotation_funcs=[add_sequence_to_nodes],
            edge_annotation_funcs=[],
        )
    except urllib.error.URLError as e:
        pytest.skip(f"GRN database download unavailable: {e}")
    print(g.edges(data=True))


def test_grn_config():
    config = GRNGraphConfig()
    assert config is not None


if __name__ == "__main__":
    test_construct_graph()
