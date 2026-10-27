"""Tests for graphein.ppi.visualisation"""

import matplotlib
import networkx as nx
import plotly.graph_objects as go

from graphein.ppi.visualisation import plot_ppi_graph, plotly_ppi_graph

matplotlib.use("Agg")


def _graph() -> nx.Graph:
    g = nx.Graph()
    g.add_edge("CDC42", "CDK1", kind={"string"})
    g.add_edge("CDK1", "PLK1", kind={"biogrid"})
    g.add_edge("PLK1", "RHOA", kind={"string", "biogrid"})
    return g


def test_plot_ppi_graph():
    plot_ppi_graph(_graph())


def test_plotly_ppi_graph():
    assert isinstance(plotly_ppi_graph(_graph()), go.Figure)
