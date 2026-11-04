"""Tests for graphein.rna.visualisation"""

import graphein.rna as gr
from graphein.protein.visualisation import plotly_protein_structure_graph


def test_plotly_rna_structure_graph_is_exported():
    # Public alias used in the RNA tutorial notebook.
    assert gr.plotly_rna_structure_graph is plotly_protein_structure_graph
