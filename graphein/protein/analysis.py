"""Contains utilities for computing analytics on and plotting summaries of Protein Structure Graphs."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import itertools
from collections import Counter
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly as ply
import plotly.express as px


def plot_degree_distribution(
    g: nx.Graph, title: Optional[str] = None
) -> plotly.graph_objects.Figure:
    """Plots the distribution of node degrees in the graph.

    :param g: networkx graph to plot the distribution of node degrees in.
    :type g: nx.Graph
    :param title: Title of plot. defaults to ``None``.
    :type title: Optional[str], optional
    :return: Plotly figure.
    :rtpe: plotly.graph_objects.Figure
    """
    if not title:
        title = g.graph["name"] + " - Degree Distribution"
    df = pd.DataFrame(g.degree())
    df.columns = ["node", "degree"]
    return ply.hist_frame(df, x="degree", title=title)


def plot_residue_composition(
    g: nx.Graph, sort_by: Optional[str] = None, plot_type: str = "bar"
) -> plotly.graph_objects.Figure:
    """Plots the residue composition of the graph.

    :param g: Protein graph to plot the residue composition of.
    :type g: nx.Graph
    :param sort_by: How to sort the values (``"alphabetical"``, ``"count"``), defaults to ``None`` (no sorting).
    :type sort_by: Optional[str], optional
    :param plot_type: How to plot the composition (``"bar"``, ``"pie"``), defaults to ``"bar"``.
    :type plot_type: str, optional
    :raises ValueError: Raises ValueError if ``sort_by`` is not one of ``"alphabetical"``, ``"count"``.
    :return: Plotly figure.
    :rtype: plotly.graph_objects.Figure
    """
    title = g.graph["name"] + " - Residue Composition"
    residues = [d["residue_name"] for _, d in g.nodes(data=True)]
    counts = pd.Series(Counter(residues))
    if not sort_by:
        counts = pd.DataFrame(counts)
    elif sort_by == "alphabetical":
        counts = pd.DataFrame(counts).sort_index()
    elif sort_by == "count":
        counts = pd.DataFrame(counts).sort_values(by=0, ascending=False)
    else:
        raise ValueError(
            f"sort_by: {sort_by} not supported. Please use on of 'count' or 'alphabetical'."
        )
    counts.columns = ["counts"]
    if plot_type == "bar":
        fig = px.bar(counts, title=title)
    elif plot_type == "pie":
        fig = px.pie(counts, values="counts", names=counts.index, title=title)
    return fig


def plot_degree_by_residue_type(
    g: nx.Graph, normalise_by_residue_occurrence: bool = True
) -> plotly.graph_objects.Figure:
    """Plots the distribution of node degrees in the graph.

    :param g: networkx graph to plot the distribution of node degrees by residue type of.
    :type g: nx.Graph
    :param normalise_by_residue_occurrence: Whether to normalise the degree by the number of residues of the same type.
    :type normalise_by_residue_occurrence: bool
    :return: Plotly figure.
    :rtpe: plotly.graph_objects.Figure
    """

    title = g.graph["name"] + " - Total Degree by Residue Type"

    residues = [d["residue_name"] for _, d in g.nodes(data=True)]
    counts = pd.Series(Counter(residues))
    residues = list(set(residues))

    degree_values = {r: 0 for r in residues}
    for r, deg in g.degree():
        degree_values[r[2:5]] += deg

    df = pd.Series(degree_values, index=residues)
    if normalise_by_residue_occurrence:
        df = df.divide(counts)
        title += " (Normalised by Residue Occurrence)"

    return ply.hist_frame(df, x=df.index, y=df.values, title=title)


def plot_edge_type_distribution(
    g: nx.Graph, plot_type: str = "bar", title: Optional[str] = None
) -> plotly.graph_objects.Figure:
    """Plots the distribution of edge types in the graph.

    :param g: NetworkX graph to plot the distribution of edge types in.
    :type g: nx.Graph
    :param plot_type: Type of plot to produce, defaults to ``"bar"``. One of ``"bar"``, ``"pie"``.
    :type plot_type: str, optional
    :param title: Title of plot. defaults to None
    :type title: Optional[str], optional
    :return: Plotly figure.
    :rtype: plotly.graph_objects.Figure
    """
    if not title:
        title = g.graph["name"]

    edges = [list(d["kind"]) for _, _, d in g.edges(data=True)]
    edges = list(itertools.chain.from_iterable(edges))
    counts = pd.Series(Counter(edges))
    counts = pd.DataFrame(counts)
    counts.columns = ["counts"]

    if plot_type == "bar":
        fig = px.bar(counts, title=title)
    elif plot_type == "pie":
        fig = px.pie(counts, values="counts", names=counts.index, title=title)

    return fig


def graph_summary(
    G: nx.Graph,
    summary_statistics: List[str] = [
        "degree",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
        "communicability_betweenness_centrality",
    ],
    custom_data: Optional[Union[pd.DataFrame, pd.Series]] = None,
    plot: bool = False,
) -> pd.DataFrame:
    """Returns a summary of the graph in a dataframe.

    :param G: NetworkX graph to get summary of.
    :type G: nx.Graph
    :param plot: Whether or not to plot the summary as a heatmap, defaults to ``False``.
    :type plot: bool
    :return: Dataframe of summary or plot.
    :rtype: pd.DataFrame
    """
    col_list = []
    col_names = []
    if "degree" in summary_statistics:
        degrees = pd.DataFrame(nx.degree(G))
        degrees.columns = ["node", "degree"]
        degrees.index = degrees["node"]
        degrees = degrees["degree"]
        col_list.append(degrees)
        col_names.append("degree")
    if "betweenness_centrality" in summary_statistics:
        betweenness = pd.Series(nx.betweenness_centrality(G))
        col_list.append(betweenness)
        col_names.append("betweenness_centrality")
    if "closeness_centrality" in summary_statistics:
        closeness = pd.Series(nx.closeness_centrality(G))
        col_list.append(closeness)
        col_names.append("closeness_centrality")
    if "eigenvector_centrality" in summary_statistics:
        eigenvector = pd.Series(nx.eigenvector_centrality_numpy(G))
        col_list.append(eigenvector)
        col_names.append("eigenvector_centrality")
    if "communicability_betweenness_centrality" in summary_statistics:
        communicability = pd.Series(
            nx.communicability_betweenness_centrality(G)
        )
        col_list.append(communicability)
        col_names.append("communicability_betweenness_centrality")

    df = pd.DataFrame(col_list).T
    df.columns = col_names
    df.index.name = "node"

    # Add custom data if provided
    if custom_data:
        df = pd.concat([df, custom_data], axis=1)

    if plot:
        return px.imshow(df.T)

    chain = [id.split(":")[0] for id in list(df.index)]
    residue_type = [id.split(":")[1] for id in list(df.index)]
    position = [id.split(":")[2] for id in list(df.index)]
    df["residue_type"] = residue_type
    df["position"] = position
    df["chain"] = chain

    return df


def plot_graph_metric_property_correlation(
    g: nx.Graph,
    summary_statistics: List[str] = [
        "degree",
        "betweenness_centrality",
        "closeness_centrality",
        "eigenvector_centrality",
        "communicability_betweenness_centrality",
    ],
    properties: List[str] = ["asa"],
    colour_by: Optional[str] = "residue_type",
    opacity: float = 0.2,
    diagonal_visible: bool = True,
    title: Optional[str] = None,
    height: int = 1000,
    width: int = 1000,
    font_size: int = 10,
) -> plotly.graph_objects.Figure:
    """Plots the correlation between graph metrics and properties.

    :param g: Protein graph to plot the correlation of.
    :type g: nx.Graph
    :param summary_statistics: List of graph metrics to employ in plot, defaults to
        ``["degree", "betweenness_centrality", "closeness_centrality", "eigenvector_centrality", "communicability_betweenness_centrality"]``.
    :type summary_statistics: List[str], optional
    :param properties: List of node properties to use in plot, defaults to ``["asa"]``.
    :type properties: List[str], optional
    :param colour_by: Controls colouring of points in plot. Options: ``"residue_type"``, ``"position"``, ``"chain"``, defaults to ``"residue_type"``.
    :type colour_by: Optional[str], optional
    :param opacity: Opacity of plot points, defaults to ``0.2``.
    :type opacity: float, optional
    :param diagonal_visible: Whether or not to show the diagonal plots, defaults to ``True``.
    :type diagonal_visible: bool, optional
    :param title: Title of plot, defaults to ``None``.
    :type title: Optional[str], optional
    :param height: Height of plot, defaults to ``1000``.
    :type height: int, optional
    :param width: Width of plot, defaults to ``1000``.
    :type width: int, optional
    :param font_size: Font size for plot text, defaults to ``10``.
    :type font_size: int, optional
    :return: Scatter plot matrix of graph metrics and protein properties.
    :rtype: plotly.graph_objects.Figure
    """
    if not title:
        title = (
            g.graph["name"] + " - Correlation of Graph Metrics with Properties"
        )
        if colour_by:
            title += f" (Coloured by {colour_by})"

    protein_properties = [
        pd.Series(nx.get_node_attributes(g, p), name=p) for p in properties
    ]
    protein_properties = pd.concat(protein_properties, axis=1)

    summary = graph_summary(g, summary_statistics=summary_statistics)
    summary = summary[
        summary_statistics + ["residue_type", "position", "chain"]
    ]
    dataf = pd.concat([summary, protein_properties], axis=1)

    fig = px.scatter_matrix(
        data_frame=dataf,
        # custom_data=dataf.index,
        dimensions=summary_statistics + properties,
        opacity=opacity,
        labels={
            col: col.replace("_", " ") for col in dataf.columns
        },  # remove underscore
        hover_name=dataf.index,
        color=colour_by,
        height=height,
        width=width,
        title=title,
    )
    fig.update_layout(font=dict(size=font_size))
    fig.update_traces(diagonal_visible=diagonal_visible)

    return fig
