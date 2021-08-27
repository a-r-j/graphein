import networkx as nx

from graphein.ppi.config import PPIGraphConfig
from graphein.ppi.edges import add_biogrid_edges, add_string_edges
from graphein.ppi.features.node_features import add_sequence_to_nodes
from graphein.ppi.graphs import compute_ppi_graph

PROTEIN_LIST = [
    "CDC42",
    "CDK1",
    "KIF23",
    "PLK1",
    "RAC2",
    "RACGAP1",
    "RHOA",
    "RHOB",
]


# Test Graph Construction
def test_construct_graph():
    config = PPIGraphConfig()

    g = compute_ppi_graph(
        protein_list=PROTEIN_LIST,
        edge_construction_funcs=[add_biogrid_edges, add_string_edges],
        node_annotation_funcs=[add_sequence_to_nodes],
        config=config,
    )

    print(nx.info(g))

    # Check nodes and edges
    assert len(g.nodes()) == 8
    assert len(g.edges()) == 23

    # Check edge types are from string/biogrid
    # Check nodes are in our list
    for u, v, d in g.edges(data=True):
        assert d["kind"].issubset(set(["string", "biogrid"]))
        assert u in PROTEIN_LIST
        assert v in PROTEIN_LIST

    # Check sequence is defined if UniProt ID found
    for n, d in g.nodes(data=True):
        assert d["protein_id"] in PROTEIN_LIST
        if d["uniprot_ids"] is not None:
            for id in d["uniprot_ids"]:
                assert d[f"sequence_{id}"] is not None
