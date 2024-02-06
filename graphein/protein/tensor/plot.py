"""Plotting utilities for protein tensors.

All plots are produced with Plotly and so are loggable to Weights and Biases.
"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import Any, Dict, List, Optional, Union

import plotly.express as px
import plotly.graph_objects as go
import torch

from ..resi_atoms import ATOM_NUMBERING
from .angles import dihedrals_to_rad
from .representation import get_full_atom_coords
from .types import AtomTensor, CoordTensor, DihedralTensor


def plot_dihedrals(
    dihedrals: DihedralTensor, to_rad: bool = True
) -> go.Figure:
    """Plots a heatmap of dihedral angles.

    .. code-block:: python
        import torch

        x = torch.rand((32, 6))
        plot_dihedrals(x)


    .. seealso::
        :meth:`graphein.protein.tensor.angles.dihedrals``
        :meth:`graphein.protein.tensor.angles.dihedrals_to_rad`
        :class:`graphein.protein.tensor.types.DihedralTensor`

    :param dihedrals: Tensor of Dihedral Angles
    :type dihedrals: graphein.protein.tensor.DihedralTensor
    :param to_rad: Whether or not to convert to radians, defaults to ``True``
    :type to_rad: bool, optional
    :returns: Plotly figure of dihedral angles (``px.imshow``)
    :rtype: go.Figure
    """
    if dihedrals.shape[1] == 6 and to_rad:
        dihedrals = dihedrals_to_rad(dihedrals, concat=True)

    return px.imshow(dihedrals, aspect="auto")  # , vmin=-np.pi, vmax=np.pi)


def plot_distance_matrix(
    x: Union[CoordTensor, AtomTensor], **kwargs: Dict[str, Any]
) -> go.Figure:
    """
    Computes and plots a distance matrix of a ``CoordTensor`` or ``AtomTensor``.

    :param x: Tensor of structure to plot
    :type x: Union[CoordTensor, AtomTensor]
    :return: Plotly figure of distance matrix (``px.imshow``)
    :rtype: go.Figure
    """
    if x.ndim == 3:
        x = get_full_atom_coords(x, **kwargs)

    dist_mat = torch.cdist(x, x)
    return px.imshow(dist_mat)


def plot_structure(
    x: Union[CoordTensor, AtomTensor],
    atoms: List[str] = ["N", "CA", "C", "O", "CB"],
    lines: bool = True,
    residue_ids: Optional[List[str]] = None,
) -> go.Figure:
    """Plots a protein structure in 3D.

    :param x: Coordinates of the protein. Either an AtomTensor
        ``(Length x Num Atom Types x 3)`` or a CoordTensor ``(Length x 3)``.
    :type x: Union[CoordTensor, AtomTensor]
    :param atoms: List of atoms to include in the plot,
        defaults to ``["N", "CA", "C", "O", "CB"]``.
    :type atoms: List[str], optional
    :param lines: Whether or not to join points with lines,
        defaults to ``True``.
    :type lines: bool, optional
    :return: Figure object
    :rtype: go.Figure
    """

    mode = "lines" if lines else "markers"
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
                text=residue_ids,
            )
    return fig
