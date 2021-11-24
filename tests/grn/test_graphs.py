from functools import partial

import pytest

from graphein.grn.config import GRNGraphConfig
from graphein.grn.edges import add_regnetwork_edges, add_trrust_edges
from graphein.grn.features.node_features import add_sequence_to_nodes
from graphein.grn.graphs import compute_grn_graph
from graphein.utils.utils import ping

GENE_LIST = ["AATF", "MYC", "USF1", "SP1", "TP53", "DUSP1"]


@pytest.mark.skipif(
    not ping("regnetworkweb.org"),
    reason="RegNetwork Web is intermittently down",
)
def test_construct_graph():
    config = GRNGraphConfig()

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
        node_annotation_funcs=[add_sequence_to_nodes],  # , molecular_weight],
        edge_annotation_funcs=[],
    )
    print(g.edges(data=True))


def test_grn_config():
    config = GRNGraphConfig()
    assert config is not None


if __name__ == "__main__":
    test_construct_graph()
