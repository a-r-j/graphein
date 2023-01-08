"""Plotting utilities for protein tensors."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import Any, Dict, List, Union

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch

from ..resi_atoms import ATOM_NUMBERING
from .angles import dihedrals_to_rad
from .representation import get_full_atom_coords
from .types import AtomTensor, CoordTensor, DihedralTensor


def plot_dihedrals(dihedrals: DihedralTensor, to_rad: bool = True):
    if dihedrals.shape[1] == 6 and to_rad:
        phi, psi, omg = dihedrals_to_rad(dihedrals)
        dihedrals = torch.stack([phi, psi, omg], dim=1)

    sns.heatmap(dihedrals, vmin=-np.pi, vmax=np.pi)


def plot_distance_matrix(
    x: Union[CoordTensor, AtomTensor], **kwargs: Dict[str, Any]
) -> go.Figure:
    if x.ndim == 3:
        x = get_full_atom_coords(x, **kwargs)

    dist_mat = torch.cdist(x, x)
    return px.imshow(dist_mat)


def plot_structure(
    x: Union[CoordTensor, AtomTensor],
    atoms: List[str] = ["N", "CA", "C", "O", "CB"],
    lines: bool = True,
) -> go.Figure:

    mode = "lines" if lines else "lines+markers"
    if x.ndim == 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], mode=mode))
    if x.ndim == 3:
        indices = [ATOM_NUMBERING[a] for a in atoms]
        fig = go.Figure()
        for idx in indices:
            fig.add_scatter3d(
                x=x[:, idx, 0],
                y=x[:, idx, 1],
                z=x[:, idx, 2],
                mode=mode,
                name=atoms[idx],
            )
    return fig
