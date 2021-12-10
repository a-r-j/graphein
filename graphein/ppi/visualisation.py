"""Contains utilities for plotting PPI NetworkX graphs."""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import networkx as nx


def plot_ppi_graph(
    g: nx.Graph,
    colour_edges_by: str = "kind",
    with_labels: bool = True,
    **kwargs,
):
    """Plots a Protein-Protein Interaction Graph. Colours edges by kind.

    :param g: NetworkX graph of PPI network.
    :type g: nx.Graph
    :param colour_edges_by: Colour edges by this attribute. Currently, only supports 'kind', which colours edges by the source database, by default "kind"
    :param with_labels: Whether to show labels on nodes. Defaults to True.
    :type with_labels: bool, optional
    """
    if colour_edges_by == "kind":
        edge_colors = [
            "r"
            if g[u][v]["kind"] == {"string"}
            else "b"
            if g[u][v]["kind"] == {"biogrid"}
            else "y"
            for u, v in g.edges()
        ]
    else:
        raise ValueError(
            f"Edge colouring scheme: {colour_edges_by} not supported. Please use 'kind'"
        )
    nx.draw(g, with_labels=with_labels, edge_color=edge_colors, **kwargs)


if __name__ == "__main__":
    from functools import partial

    from graphein.ppi.config import PPIGraphConfig
    from graphein.ppi.edges import add_biogrid_edges, add_string_edges
    from graphein.ppi.graphs import compute_ppi_graph

    config = PPIGraphConfig()

    protein_list = [
        "CDC42",
        "CDK1",
        "KIF23",
        "PLK1",
        "RAC2",
        "RACGAP1",
        "RHOA",
        "RHOB",
    ]

    g = compute_ppi_graph(
        protein_list=protein_list,
        edge_construction_funcs=[
            partial(add_string_edges),
            partial(add_biogrid_edges),
        ],
    )

    plot_ppi_graph(g)
