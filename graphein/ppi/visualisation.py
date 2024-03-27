"""Contains utilities for plotting PPI NetworkX graphs."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import List, Optional

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import to_rgb


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
            (
                "r"
                if g[u][v]["kind"] == {"string"}
                else "b" if g[u][v]["kind"] == {"biogrid"} else "y"
            )
            for u, v in g.edges()
        ]
    else:
        raise ValueError(
            f"Edge colouring scheme: {colour_edges_by} not supported. Please use 'kind'"
        )
    nx.draw(g, with_labels=with_labels, edge_color=edge_colors, **kwargs)


def plotly_ppi_graph(
    g: nx.Graph,
    layout: nx.layout = nx.layout.circular_layout,
    title: Optional[str] = None,
    show_labels: bool = False,
    node_size_multiplier: float = 5.0,
    node_colourscale: str = "Viridis",
    edge_colours: Optional[List[str]] = None,
    edge_opacity: float = 0.5,
    height: int = 500,
    width: int = 500,
):
    """Plots a PPI graph.

    :param g: PPI graph
    :type g: nx.Graph
    :param layout: Layout algorithm to use. Default is circular_layout.
    :type layout: nx.layout
    :param title: Title of the graph. Default is None.
    :type title: str, optional
    :param show_labels: If True, shows labels on nodes. Default is False.
    :type show_labels: bool
    :param node_size_multiplier: Multiplier for node size. Default is 5.0.
    :type node_size_multiplier: float
    :param node_colourscale: Colour scale to use for node colours. Default is "Viridis". Options:
        'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
    :type node_colourscale: str
    :param edge_colours: List of colours (hexcode) to use for edges. Default is None (px.colours.qualitative.T10).
    :type edge_colours: List[str], optional
    :param edge_opacity: Opacity of edges. Default is 0.5.
    :type edge_opacity: float
    :param height: Height of the plot. Default is 500.
    :type height: int
    :param width: Width of the plot. Default is 500.
    :type width: int
    :return: Plotly figure of PPI Network
    :rtype: go.Figure
    """
    if edge_colours is None:
        edge_colours = px.colors.qualitative.T10
    edge_colours = [
        f"rgba{tuple(list(to_rgb(c)) + [edge_opacity])}" for c in edge_colours
    ]

    # Set positions
    nx.set_node_attributes(g, layout(g), "pos")

    # Get node and edge traces
    node_trace = get_node_trace(g, node_size_multiplier, node_colourscale)
    edge_trace = get_edge_trace(g, edge_colours)
    traces = [node_trace] + edge_trace

    # Get node labels if using them.
    if show_labels:
        text_trace = go.Scatter(
            x=node_trace["x"],
            y=node_trace["y"],
            mode="text",
            text=list(g.nodes()),
            textposition="bottom center",
            hoverinfo="text",
        )
        traces.append(text_trace)

    # Assemble plot from traces
    return go.Figure(
        data=traces,
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            width=width,
            height=height,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )


def get_node_trace(
    g: nx.Graph, node_size_multiplier: float, node_colourscale: str = "Viridis"
) -> go.Scatter:
    """Produces the node trace for the plotly plot.

    :param g: PPI graph with ['pos'] added to the nodes (eg via nx.layout function)
    :type g: nx.Graph
    :param node_size_multiplier: Multiplier for node size. Default is 5.0.
    :type node_size_multiplier: float
    :param node_colourscale: Colourscale to use for the nodes, defaults to "Viridis"
    :type node_colourscale: str, optional
    :return: Node trace for plotly plot
    :rtype: go.Scatter
    """
    node_x = []
    node_y = []
    node_size = []
    for n in g.nodes():
        x, y = g.nodes[n]["pos"]
        node_x.append(x)
        node_y.append(y)
        node_size.append(g.degree(n) * node_size_multiplier)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            colorscale=node_colourscale,
            reversescale=True,
            color=[],
            size=node_size,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )
    node_text = list(g.nodes())
    node_trace.marker.color = node_size
    node_trace.text = node_text
    return node_trace


def get_edge_trace(
    g: nx.Graph,
    edge_colours: Optional[List[str]] = None,
) -> List[go.Scatter]:
    """Gets edge traces from PPI graph. Returns a list of traces enabling edge colours to be set individually.

    :param g: _description_
    :type g: nx.Graph
    :return: _description_
    :rtype: List[go.Scatter]
    """

    if edge_colours is None:
        edge_colours = ["red", "blue", "yellow"]
    traces = []
    for u, v, d in g.edges(data=True):
        # Get positions
        x0, y0 = g.nodes[u]["pos"]
        x1, y1 = g.nodes[v]["pos"]
        # Assign colour
        if d["kind"] == {"string"}:
            colour = edge_colours[0]
        elif d["kind"] == {"biogrid"}:
            colour = edge_colours[1]
        else:
            colour = edge_colours[2]

        edge_trace = go.Scatter(
            line=dict(width=2, color=colour),
            hoverinfo="text",
            x=(x0, x1),
            y=(y0, y1),
            mode="lines",
            text=[
                " / ".join(list(edge_type)) for edge_type in g[u][v]["kind"]
            ],
        )
        traces.append(edge_trace)
    return traces


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
    plotly_ppi_graph(g)

# %%
