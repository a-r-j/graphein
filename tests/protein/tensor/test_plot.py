"""Tests for graphein.protein.tensor.plot."""

import pytest

from graphein.protein.tensor.data import Protein
from graphein.protein.tensor.plot import plot_distance_matrix, plot_structure

try:
    import torch

    TORCH_AVAIL = True
except ImportError:
    TORCH_AVAIL = False


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_plot_distance_matrix_atom_tensor():
    x = torch.full((4, 37, 3), 1e-5)
    x[:, :3] = torch.rand(4, 3, 3)
    fig = plot_distance_matrix(x)
    # One row/column per (non-filled) atom
    assert fig.data[0].z.shape == (12, 12)


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_plot_structure_trace_names():
    x = torch.rand(5, 37, 3)
    for atoms in [("CA",), ("CB", "N"), ("N", "CA", "C", "O", "CB")]:
        fig = plot_structure(x, atoms=atoms)
        assert [trace.name for trace in fig.data] == list(atoms)
    # CA is atom 1
    fig = plot_structure(x, atoms=("CA",))
    assert fig.data[0].x.tolist() == x[:, 1, 0].tolist()


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_plot_structure_coord_tensor():
    fig = plot_structure(torch.rand(5, 3))
    assert len(fig.data) == 1


@pytest.mark.skipif(not TORCH_AVAIL, reason="PyTorch not available")
def test_plot_dihedrals_uses_cached_dihedrals():
    protein = Protein(coords=torch.rand(6, 37, 3))
    protein.dihedrals(cache="dihedrals")
    fig = protein.plot_dihedrals()
    assert fig.data[0].z.shape == (6, 3)
