"""Data and Batch Objects for working proteins in PyTorch Geometric"""

import itertools
import random
import traceback
from functools import partial

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Any, Callable, List, Optional, Tuple, Union

import looseversion
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import torch_geometric
from biopandas.pdb import PandasPdb
from loguru import logger as log
from torch_geometric.data import Batch, Data
from torch_geometric.data.separate import separate
from torch_geometric.utils import unbatch, unbatch_edge_index
from tqdm.contrib.concurrent import process_map

from ..resi_atoms import PROTEIN_ATOMS
from .angles import (
    alpha,
    dihedrals,
    get_backbone_bond_angles,
    get_backbone_bond_lengths,
    kappa,
    sidechain_torsion,
)
from .edges import compute_edges, edge_distances
from .geometry import apply_structural_noise, idealize_backbone, kabsch
from .io import (
    protein_df_to_chain_tensor,
    protein_df_to_tensor,
    protein_to_pyg,
    to_dataframe,
    to_pdb,
)
from .plot import plot_dihedrals, plot_distance_matrix, plot_structure
from .representation import (
    get_backbone,
    get_backbone_frames,
    get_c_alpha,
    get_full_atom_coords,
)
from .sequence import get_residue_id, get_sequence, residue_type_tensor
from .testing import (
    has_complete_backbone,
    is_complete_structure,
    is_tensor_equal,
)
from .types import (
    AtomTensor,
    BackboneFrameTensor,
    BackboneTensor,
    CoordTensor,
    DihedralTensor,
    EdgeTensor,
    PositionTensor,
    TorsionTensor,
)

PYG_VERSION = looseversion.LooseVersion(torch_geometric.__version__)


class Protein(Data):
    """ "A data object describing a homogeneous graph. ``Protein`` inherits from
    :ref:`torch_geometric.data.Data` and implements structural-biology specific
    methods in addition. ``Data`` and ``Protein`` objects can be directly
    converted between using ``Protein().from_data()`` and
    ``Protein().to_data()``. Think of ``Protein`` as an equivalent to ``Data``
    but with some protein-specific trimmings.

    The data object can hold node-level, link-level and graph-level attributes.
    In general, ``Data`` tries to mimic the behaviour of a regular Python
    dictionary. In addition, it provides useful functionality for analyzing
    graph structures, and provides basic PyTorch tensor functionalities. See
    here for the accompanying tutorial.

    There are several ways to instantiate a ``Protein``:

    .. code-block:: python

        import graphein.protein.tensor as gpt

        # From a PDB code
        protein = gpt.Protein()
        protein.from_pdb_code("4hhb", chain_selection="ABCD")

        # From a PDB file
        protein = gpt.Protein()
        protein.from_pdb_file("4hhb.pdb", chain_selection="ABCD")


        # From a BioPandas DataFrame
        p = PandasPdb().fetch_pdb("4hhb")
        df = p.df["ATOM"]
        protein = gpt.Protein().from_dataframe(df)

        # From a PyG Data object
        data = gpt.io.protein_to_pyg(
            pdb_code="4hhb", # Can alternatively pass a path or a uniprot ID (for AF2) with path=... and uniprot_id=...
            chain_selection="ABCD", # Select all 4 chains
            deprotonate=True, # Deprotonate the structure
            keep_insertions=False, # Remove insertions
            keep_hets=[], # Remove HETATMs
            model_index=1, # Select the first model
            # Can select a subset of atoms with atom_types=...
            )
        protein = gpt.Protein().from_data(data)


    The advantage is that it provides access to many protein-specific methods
    such as:

    .. code-block:: python

        import graphein.protein.tensor as gpt

        protein = gpt.Protein()
        protein.from_pdb_code("4hhb", chain_selection="ABCD")

        protein.dihedrals() # Backbone dihedral angles
        protein.sidechain_torsion() # Sidechain Torsion Angles
        protein.c_alpha() # Gets Ca coordinates
        protein.plot_structure() # Plots the 3D structure using plotly
        protein.idealize_backbone() # Idealizes the backbone geometry

    and many more!
    """

    def __init__(
        self,
        atom_list: List[str] = PROTEIN_ATOMS,
        fill_value: float = 1e-5,
        **kwargs,
    ) -> None:
        """
        .. seealso::

            :class:`graphein.protein.tensor.types.AtomTensor`
            :const:`graphein.protein.resi_atom.PROTEIN_ATOMS`

        :param atom_list: List of atoms to include in the AtomTensor.
        :type atom_list: List[str]
        :param fill_value: Value to fill in for missing values in the
            AtomTensor.
        :type fill_value: float
        :param kwargs: Additional keyword arguments to store as attributes.
        """
        super().__init__()
        self.fill_value = fill_value
        self.atom_list = atom_list
        for k, v in kwargs.items():
            setattr(self, k, v)

    # I/O
    def from_dataframe(
        self,
        df: pd.DataFrame,
        id: Optional[str] = None,
        atoms_to_keep: List[str] = PROTEIN_ATOMS,
        node_labels: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None,
    ) -> "Protein":
        """Instantiate a ``Protein`` object from a Pandas DataFrame.

        .. code-block:: python

            import graphein.protein.tensor as gpt
            from biopandas.pdb import PandasPdb

            p = PandasPdb().fetch_pdb("4hhb")
            df = p.df["ATOM"]

            protein = gpt.Protein()
            protein = protein.from_dataframe(
                df,
                atoms_to_keep=["N", "CA", "C", "O", "CB"]
                id="test_structure",
                node_labels=torch.ones(100, 1),
                graph_labels=torch.tensor(42)
                )

        .. seealso::
            :meth:`graphein.protein.tensor.io.protein_df_to_tensor`
            :meth:`graphein.protein.tensor.io.protein_df_to_chain_tensor`
            :meth:`graphein.protein.tensor.sequence.get_sequence`
            :meth:`graphein.protein.tensor.sequence.residue_type_tensor`
            :meth:`graphein.protein.tensor.sequence.get_residue_id`

        :param df: Protein structure DataFrame
        :type df: pd.DataFrame
        :param atoms_to_keep: List of atom names to preserve.
            See :ref:`graphein.protein.resi_atoms.PROTEIN_ATOMS`
        :type atoms_to_keep: List[str]
        :return: ``Protein` object populated with attributes.
        :rtype: Protein
        """
        self.id = id
        self.coords = protein_df_to_tensor(
            df, atoms_to_keep=atoms_to_keep, fill_value=self.fill_value
        )
        self.residues = get_sequence(
            df,
            chains="all",
            insertions=False,
            list_of_three=True,
        )
        self.chains = protein_df_to_chain_tensor(df)
        self.residue_id = get_residue_id(df)
        self.residue_type = residue_type_tensor(df)

        if node_labels is not None:
            self.node_labels = node_labels

        if graph_labels is not None:
            self.graph_labels = graph_labels
        return self

    def from_data(self, data: Data) -> "Protein":
        """Instantiate a ``Protein`` object from a PyTorch Geometric ``Data`` object.


        Example:

        .. code-block:: python

            >>> import graphein.protein.tensor as gpt

            >>> data = gpt.io.protein_to_pyg(
                ... pdb_code="4hhb", # Can alternatively pass a path or a uniprot ID (for AF2) with path=... and uniprot_id=...
                ... chain_selection="ABCD", # Select all 4 chains
                ... deprotonate=True, # Deprotonate the structure
                ... keep_insertions=False, # Remove insertions
                ... keep_hets=[], # Remove HETATMs
                ... model_index=1, # Select the first model
                ... # Can select a subset of atoms with atom_types=...
                ... )

            >>> protein = gpt.Protein()
            >>> protein = protein.from_data(data)
            >>> print(protein)

        :param data: PyTorch Geometric Data object
        :type data: Data
        :return: ``Protein`` object containing the same keys and values
        :rtype: Protein
        """
        keys = (
            data.keys()
            if PYG_VERSION >= looseversion.LooseVersion("2.4.0")
            else data.keys
        )

        for key in keys:
            setattr(self, key, getattr(data, key))
        return self

    def to_data(self) -> Data:
        """Convert a ``Protein`` object to PyTorch Geometric ``Data`` object.

        .. code-block:: python

            import graphein.protein.tensor as gpt
            protein = gpt.Protein().from_pdb_code(pdb_code="3eiy")

            type(protein) # Protein

            pyg_data = protein.to_data()
            type(pyg_data) # torch_geometric.data.Data

        :return: Data object containing the same keys and values
        :rtype: Data
        """
        data = Data()
        keys = (
            self.keys()
            if PYG_VERSION >= looseversion.LooseVersion("2.4.0")
            else self.keys
        )
        for i in keys:
            setattr(data, i, getattr(self, i))
        return data

    def to_dataframe(
        self,
        x: Optional[Union[AtomTensor, BackboneTensor]] = None,
        cache: Optional[str] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, PandasPdb]:
        """Compute a Pandas DataFrame from a ``Protein`` object.

        .. see:: :func:`graphein.protein.tensor.io.to_dataframe`

        :param x: AtomTensor, defaults to ``None`` (uses ``Protein.coords``)
        :type x: Optional[Union[AtomTensor, BackboneTensor]], optional
        :param cache: Whether to store the dataframe as a ``Protein`` attribute,
            defaults to ``None``.
        :type cache: Optional[str], optional
        :param kwargs: Keyword args for
            :func:`graphein.protein.tensor.io.to_dataframe`
        :return: DataFrame of protein structure.
        :rtype: Union[pd.DataFrame, PandasPdb]
        """

        if x is None:
            x = self.coords
        out = to_dataframe(
            x=x,
            fill_value=self.fill_value,
            residue_types=self.residue_type,
            chains=self.chains,
            **kwargs,
        )
        if cache is not None:
            setattr(self, cache, out)
        return out

    def to_pdb(
        self,
        out_path: Optional[str] = None,
        x: Optional[Union[AtomTensor, BackboneTensor]] = None,
        gz: bool = False,
        **kwargs,
    ):
        """Write a ``Protein`` object to a PDB file.

        .. code-block:: python

            import graphein.protein.tensor as gpt
            protein = gpt.Protein().from_pdb_code(pdb_code="3eiy")

            protein.to_pdb(out_path="3eiy.pdb")

            # Write only backbone
            protein.to_pdb(x=protein.backbone(), out_path="3eiy_backbone.pdb")


        .. see:: :func:`graphein.protein.tensor.io.to_pdb`

        .. seealso:: :meth:`graphein.protein.tensor.io.to_dataframe`


        :param x: ``AtomTensor``, defaults to ``None`` (uses ``Protein.coords``)
        :type x: Optional[Union[AtomTensor, BackboneTensor]], optional
        :param out_path: Path to output file, defaults to ``None``
            (``{Protein.id}.pdb``)
        :type out_path: Optional[str], optional
        :param gz: Whether or not to gzip the output, defaults to ``False``
        :type gz: bool, optional
        :param kwargs: Additional arguments to pass to
            :func:`graphein.protein.tensor.io.to_pdb`
        """
        if out_path is None:
            out_path = self.id + ".pdb"
        if x is None:
            x = self.coords
        to_pdb(x, out_path, gz, **kwargs)

    def from_pdb_code(
        self,
        pdb_code: str,
        chain_selection: Union[str, List[str]] = "all",
        deprotonate: bool = True,
        keep_insertions=False,
        keep_hets: List[str] = [],
        model_index: int = 1,
        atom_types: List[str] = PROTEIN_ATOMS,
        node_labels: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None,
    ) -> "Protein":  # sourcery skip: class-extract-method
        data = protein_to_pyg(
            pdb_code=pdb_code,
            chain_selection=chain_selection,
            deprotonate=deprotonate,
            keep_insertions=keep_insertions,
            keep_hets=keep_hets,
            model_index=model_index,
            atom_types=atom_types,
        )
        if node_labels is not None:
            data.node_labels = node_labels

        if graph_labels is not None:
            data.graph_labels = graph_labels

        self.from_data(data)
        return self

    def from_pdb_file(
        self,
        path: str,
        chain_selection: str = "all",
        deprotonate: bool = True,
        keep_insertions=False,
        keep_hets: List[str] = [],
        model_index: int = 1,
        atom_types: List[str] = PROTEIN_ATOMS,
        node_labels: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None,
    ) -> "Protein":
        data = protein_to_pyg(
            path=path,
            chain_selection=chain_selection,
            deprotonate=deprotonate,
            keep_insertions=keep_insertions,
            keep_hets=keep_hets,
            model_index=model_index,
            atom_types=atom_types,
        )
        if node_labels is not None:
            data.node_labels = node_labels
        if graph_labels is not None:
            data.graph_labels = graph_labels
        self.from_data(data)
        return self

    def save(self, out_path: str):
        """Save a ``Protein`` object to disk in tensor format.

        :param out_path: Path to save Protein to.
        :type out_path: str
        """
        torch.save(self, out_path)

    def edges(
        self,
        edge_type: str = "knn_10",
        x: Optional[torch.Tensor] = None,
        cache: Optional[str] = None,
        **kwargs,
    ) -> EdgeTensor:
        """
        Computes edges for a protein as specified by the ``edge_type`` argument.
        If ``x`` is not provided, the edges are calculated based on Alpha Carbon
        positions by default.

        .. code-block:: python
            import graphein.protein.tensor as gpt
            protein = gpt.Protein().from_pdb_code(pdb_code="3eiy")

            protein.compute_edges("knn_8") # Alpha carbon graph with KNN edges (k=8)
            protein.compute_edges("eps_6") # Alpha carbon graph with radius (r=6)

            fa = protein.full_atom_coords()
            protein.compute_edges("knn_10", x=fa, cache="atomic_edges") # Full atom graph with KNN edges
            protein.atomic_edges.shape # torch.Size([2, ...])

        .. seealso::

            :meth:`graphein.protein.tensor.edges.compute_edges`
            :meth:`graphein.protein.tensor.edges.radius_edges`
            :meth:`graphein.protein.tensor.edges.knn_edges`

        :param edge_type: Str denoting type of edges in form
            ``{edgetype}_{value}``. E.g. ``"knn_8"`` for KNN with ``k=8``,
            ``"eps_6"`` for radius graph with ``r=6``.
        :type edge_type: str
        :param x: Node feature/position matrix used to calculate edges. If
            ``None`` (default), the alpha carbon trace is used.
        :type x: Optional[torch.Tensor]
        :param cache: String to set edges as a ``Protein`` attribute. If
            ``None`` (default), edges are not saved.
        :type cache: Optional[str]
        """
        if x is None:
            x = self.alpha_carbon()
        out = compute_edges(x=x, edge_type=edge_type, batch=None, **kwargs)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def edge_distances(
        self,
        x: CoordTensor,
        edge_index: EdgeTensor,
        p: float = 2,
        cache: Optional[str] = None,
    ) -> torch.Tensor:
        """Computes the edges distances between nodes.

        :param x: Node positions
        :param edge_index: Edge indices
        :param p: The norm degree. Can be negative. Default: ``2``.``
        :returns: Edge distances
        :rtype: torch.Tensor
        """
        out = edge_distances(x=x, edge_index=edge_index, p=p)
        if cache is not None:
            setattr(self, cache, out)
        return out

    # Representation
    def alpha_carbon(self, cache: Optional[str] = None) -> CoordTensor:
        """Returns the alpha carbon coordinates of the protein as a tensor
        of shape ``Length x 3``.

        .. code-block:: python
            import graphein.protein.tensor as gpt
            protein = gpt.Protein().from_pdb_code(pdb_code="3eiy")

            protein.alpha_carbon().shape # torch.Size([374, 3])

            protein.alpha_carbon(cache="ca")
            protein.ca.shape # torch.Size([374, 3])

        .. seealso:: :func:`graphein.protein.tensor.representation.get_c_alpha`

        :param cache: If provided, the result will be cached in the ``Protein``
            object with the provided string as the attribute name. Default is
            ``None`` (not stored).
        :type cache: Optional[str]
        :return: Alpha carbon coordinates
        :rtype: graphein.protein.tensor.types.CoordTensor
        """

        out = get_c_alpha(self.coords)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def backbone(self, cache: Optional[str] = None) -> BackboneTensor:
        """
        Returns the backbone coordinates of the protein as a tensor
        of shape ``Length x 4 x 3`` (``[N, Ca, C, O]``).

        .. code-block:: python
            import graphein.protein.tensor as gpt
            protein = gpt.Protein().from_pdb_code(pdb_code="3eiy")

            protein.backbone().shape # torch.Size([374, 4, 3])

            protein.backbone(cache="bb")
            protein.bb.shape # torch.Size([374, 4, 3])

        .. see:: :meth:`graphein.protein.tensor.representation.get_backbone`

        :param cache: If provided, the result will be cached in the ``Protein``
            object with the provided string as the attribute name. Default is
            ``None`` (not stored).
        :type cache: Optional[str]
        :return: Backbone coordinates ``[Length x 4 x 3]``
        :rtype: graphein.protein.tensor.types.BackboneTensor
        """
        out = get_backbone(self.coords)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def backbone_frames(
        self, cache: Optional[str] = None
    ) -> Tuple[BackboneFrameTensor, CoordTensor]:
        """Computes backbone rotation frames from an idealised residue.

        .. see:: :func:`graphein.protein.tensor.representation.get_backbone_frames`

        .. seealso:: :func:`graphein.protein.tensor.reconstruction.get_ideal_backbone_coords`
        :param cache: If provided, the result will be cached in the ``Protein``
            object with the provided string as the attribute name. Default is
            ``None`` (not stored).
        :type cache: Optional[str]
        :return: _description_
        :rtype: Tuple[BackboneFrameTensor, CoordTensor]
        """
        out = get_backbone_frames(self.coords)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def idealize_backbone(
        self,
        x: Optional[Union[AtomTensor, BackboneTensor]] = None,
        lr: float = 1e-3,
        n_iter: int = 100,
        inplace: bool = False,
        cache: Optional[str] = None,
    ) -> AtomTensor:
        """Computes idealised backbone coordinates.

        .. seealso::
            :meth:`graphein.protein.tensor.geometry.idealize_backbone`

        :param lr: Learning rate to use for optimisation, defaults to ``1e-3``
        :type lr: float, optional
        :param n_iter: Number of optimisation steps, defaults to ``100``
        :type n_iter: int, optional
        :param inplace: Whether or not to optimise the backbone inplace,
            defaults to ``False``.
        :type inplace: bool, optional
        :return: Idealised backbone coordinates
        :rtype: AtomTensor
        """
        if x is None:
            x = self.coords
        return idealize_backbone(x, lr=lr, n_iter=n_iter, inplace=inplace)

    def full_atom_coords(self, cache: Optional[str] = None) -> CoordTensor:
        """Gets the full atom coordinates of the protein. Returns tensor of
        shape ``[Length x 3]``.

        .. see:: :func:`graphein.protein.tensor.representation.get_full_atom_coords`

        :param cache: Whether or not to store the coordinates as an attribute,
            defaults to ``None`` (note stored).
        :type cache: Optional[str], optional
        :return: Tensor of atom positions.
        :rtype: CoordTensor
        """
        out = get_full_atom_coords(self.coords, fill_value=self.fill_value)
        if cache is not None:
            setattr(self, cache, out)
        return out

    # Angles
    def dihedrals(
        self, rad: bool = True, embed: bool = True, cache: Optional[str] = None
    ) -> DihedralTensor:
        """
        .. see:: :func:`graphein.protein.tensor.angles.dihedrals`
        """
        out = dihedrals(self.coords, rad=rad, embed=embed)

        if cache is not None:
            setattr(self, cache, out)
        return out

    def sidechain_torsion(self, cache: Optional[str] = None) -> TorsionTensor:
        """
        Computes sidechain torsion angles.

        .. see:: :func:`graphein.protein.tensor.angles.sidechain_torsion`

        :param: cache: If provided, the result will be cached in the ``Protein``
            under the provided string as the attribute name. Default is ``None``
            , (not stored).
        :type cache: Optional[str]
        """
        return sidechain_torsion(self.coords, self.residues)

    def kappa(
        self,
        cache: Optional[str] = None,
        rad: bool = True,
        embed: bool = True,
    ) -> torch.Tensor:
        """
        Computes ``kappa`` virtual angle.

        .. see:: :func:`graphein.protein.tensor.angles.kappa`

        :param: cache: If provided, the result will be cached in the ``Protein``
            under the provided string as the attribute name. Default is ``None``,
            (not stored).
        :type cache: Optional[str]
        """
        out = kappa(self.coords, rad=rad, embed=embed)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def alpha(
        self, cache: Optional[str] = None, rad: bool = True, embed: bool = True
    ) -> torch.Tensor:
        """
        Computes ``alpha`` virtual angle.

        .. see:: :func:`graphein.protein.tensor.angles.alpha`

        :param cache: If provided, the result will be cached in the ``Protein``
            under the provided string as the attribute name. Default is ``None``,
            (not stored).
        :type cache: Optional[str]
        """
        out = alpha(self.coords, rad=rad, embed=embed)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def align_to(
        self,
        other: "Protein",
        ca_only: bool = True,
        return_transformed: bool = True,
        cache: Optional[str] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Aligns the ``Protein`` to another (``other``) tensor using the Kabsch
        algorithm.

        .. see:: :func:`graphein.protein.tensor.geometry.kabsch`

        """
        out = kabsch(
            self.coords,
            other.coords,
            ca_only=ca_only,
            fill_value=self.fill_value,
            return_transformed=return_transformed,
        )
        if cache and return_transformed:
            setattr(self, cache, out)
        elif cache:
            setattr(self, f"{cache}_R", out[0])
            setattr(self, f"{cache}_t", out[1])
        return out

    # Features
    def amino_acid_one_hot(
        self, num_types: int = 23, cache: Optional[str] = None
    ) -> torch.Tensor:
        out = F.one_hot(self.residue_type, num_classes=num_types)
        if cache is not None:
            setattr(self, cache, out)
        return out

    # Testing
    def is_complete(self) -> bool:
        """
        Checks if a ``Protein`` as all the requisite atoms present.

        .. see:: :func:`graphein.protein.tensor.testing.is_complete_structure`

        :return: Boolean indicating whether or not the ``Protein`` has a
            complete structure.
        :rtype: bool
        """
        return is_complete_structure(self.coords, self.residues)

    def has_complete_backbone(self) -> bool:
        """
        Checks if a ``Protein`` as all backbone atoms present.

        .. see:: :func:`graphein.protein.tensor.testing.has_complete_backbone`

        :return: Boolean indicating whether or not the ``Protein`` has a
            complete backbone.
        :rtype: bool
        """
        return has_complete_backbone(self.coords)

    def __eq__(self, __o: object) -> bool:
        # sourcery skip: merge-duplicate-blocks, merge-else-if-into-elif
        keys = (
            self.keys()
            if PYG_VERSION >= looseversion.LooseVersion("2.4.0")
            else self.keys
        )

        for i in keys:
            attr_self = getattr(self, i)
            attr_other = getattr(__o, i)

            if isinstance(attr_self, torch.Tensor):
                if not is_tensor_equal(attr_self, attr_other):
                    return False
            else:
                if attr_self != attr_other:
                    return False
        return True

    # Plotting
    def plot_distance_matrix(
        self, x: Optional[CoordTensor] = None
    ) -> go.Figure:
        """
        Plots a distance matrix. If ``x`` is not provided, it plots the
        alpha carbon contact map.

        :param x: Coordinates to plot, defaults to ``None``. If ``None``, the
            alpha carbon coordinates are used.
        """
        if x is None:
            x = self.alpha_carbon()
        return plot_distance_matrix(x)

    def plot_dihedrals(self) -> go.Figure:
        keys = (
            self.keys()
            if PYG_VERSION >= looseversion.LooseVersion("2.4.0")
            else self.keys
        )

        dh = (
            dihedrals(self.coords)
            if "dihedrals" not in keys
            else self.dihedrals
        )
        return plot_dihedrals(dh)

    def plot_structure(
        self, atoms: List[str] = ["N", "CA", "C", "O"], lines: bool = True
    ) -> go.Figure:
        """
        Plots a 3D structure of the protein in Plotly. This can be logged to
        WandB.

        .. code-block:: python
            import graphein.protein.tensor as gpt
            protein = gpt.Protein().from_pdb_code(pdb_code="3eiy")

            protein.plot_structure(atoms=["CA"], lines=True) # Plot CA trace only
            protein.plot_structure() # Plot backbone


        .. seealso:: :meth:`graphein.protein.tensor.plot.plot_structure`

        """
        residue_ids = self.residue_id if hasattr(self, "residue_id") else None
        return plot_structure(
            self.coords, atoms=atoms, lines=lines, residue_ids=residue_ids
        )

    def apply_structural_noise(
        self,
        x: Optional[Union[AtomTensor, CoordTensor]] = None,
        magnitude: float = 0.1,
        gaussian: bool = True,
        return_transformed: bool = True,
        cache: Optional[str] = None,
    ) -> Union[AtomTensor, CoordTensor]:
        """
        Applies noise to the structure of the protein.

        .. see:: :func:`graphein.protein.tensor.geometry.apply_structural_noise`

        :param x: Coordinates to apply noise to, defaults to ``None``. If
            ``None`` (default), ``self.coords`` is used.
        :param cache: If provided, the result will be cached in the ``Protein``
            under the provided string as the attribute name. Default is ``None``,
            (not stored).
        """
        if x is None:
            x = self.coords
        out = apply_structural_noise(
            x,
            magnitude=magnitude,
            gaussian=gaussian,
            return_transformed=return_transformed,
        )
        if cache is not None:
            setattr(self, cache, out)
        return out


class ProteinBatch(Batch):
    def __init__(
        self, fill_value: float = 1e-5, atom_list: List[str] = PROTEIN_ATOMS
    ) -> None:
        super().__init__()
        self.fill_value = fill_value
        self.atom_list = atom_list

    def from_batch(
        self, batch: Batch, fill_value: float = 1e-5
    ) -> "ProteinBatch":
        keys = (
            batch.keys()
            if PYG_VERSION >= looseversion.LooseVersion("2.4.0")
            else batch.keys
        )
        for key in keys:
            setattr(self, key, getattr(batch, key))

        if hasattr(batch, "_slice_dict"):
            self._slice_dict = batch._slice_dict

        if hasattr(batch, "_inc_dict"):
            self._inc_dict = batch._inc_dict

        if hasattr(batch, "_num_graphs"):
            self._num_graphs = batch._num_graphs

        # self.fill_value = fill_value
        return self

    @classmethod
    def from_protein_list(cls, proteins: List[Protein]):
        # sourcery skip: class-extract-method
        proteins = [Protein().from_data(p) for p in proteins]
        batch = Batch.from_data_list(proteins)
        return cls().from_batch(batch)

    def from_pdb_codes(
        self,
        pdb_codes: List[str],
        chain_selection: Optional[List[str]] = None,
        node_labels: Optional[List[torch.Tensor]] = None,
        graph_labels: Optional[List[torch.Tensor]] = None,
        model_index: Optional[List[int]] = None,
        atom_types: List[str] = PROTEIN_ATOMS,
        deprotonate: bool = True,
        keep_insertions: bool = False,
        keep_hets: List[str] = [],
    ):
        proteins = [
            Protein().from_pdb_code(
                pdb_code=pdb,
                chain_selection=(
                    chain_selection[i]
                    if chain_selection is not None
                    else "all"
                ),
                deprotonate=deprotonate,
                keep_insertions=keep_insertions,
                keep_hets=keep_hets,
                model_index=model_index[i] if model_index is not None else 1,
                atom_types=atom_types,
                node_labels=(
                    node_labels[i] if node_labels is not None else None
                ),
                graph_labels=(
                    graph_labels[i] if graph_labels is not None else None
                ),
            )
            for i, pdb in enumerate(pdb_codes)
        ]
        batch = Batch.from_data_list(proteins)
        self.from_batch(batch)
        return self

    def from_pdb_files(
        self,
        paths: List[str],
        chain_selection: Optional[List[str]] = None,
        node_labels: Optional[List[torch.Tensor]] = None,
        graph_labels: Optional[List[torch.Tensor]] = None,
        model_index: Optional[List[int]] = None,
        atom_types: List[str] = PROTEIN_ATOMS,
        deprotonate: bool = True,
        keep_insertions: bool = False,
        keep_hets: List[str] = [],
    ):
        proteins = [
            Protein().from_pdb_file(
                path=pdb,
                chain_selection=(
                    chain_selection[i]
                    if chain_selection is not None
                    else "all"
                ),
                deprotonate=deprotonate,
                keep_insertions=keep_insertions,
                keep_hets=keep_hets,
                model_index=model_index[i] if model_index is not None else 1,
                atom_types=atom_types,
                node_labels=(
                    node_labels[i] if node_labels is not None else None
                ),
                graph_labels=(
                    graph_labels[i] if graph_labels is not None else None
                ),
            )
            for i, pdb in enumerate(paths)
        ]
        batch = Batch.from_data_list(proteins)
        self.from_batch(batch)
        return self

    def to_batch(self) -> Batch:
        """Returns the ProteinBatch as a torch_geometric.data.Batch object."""
        batch = Batch()
        keys = (
            self.keys()
            if PYG_VERSION >= looseversion.LooseVersion("2.4.0")
            else self.keys
        )
        for key in keys:
            setattr(batch, key, getattr(self, key))
        return batch

    def save(self, out_path: str):
        """Save a ``ProteinBatch`` object to disk in tensor format.

        :param out_path: Path to save Protein to.
        :type out_path: str
        """
        torch.save(self, out_path)

    # Representation
    def alpha_carbon(self, cache: Optional[str] = None) -> CoordTensor:
        """Returns the alpha carbon coordinates of the protein as a tensor
        of shape Length x 3.

        .. code-block:: python
            import graphein.protein.tensor as gpt
            protein = gpt.ProteinBatch().from_pdb_codes(pdb_codes=["3eiy"])

            protein.alpha_carbon().shape # torch.Size([374, 3])

            protein.alpha_carbon(cache="ca")
            protein.ca.shape # torch.Size([374, 3])

        .. seealso:: :meth:`graphein.protein.tensor.representation.alpha_carbon`

        :param cache: If provided, the result will be cached in the Protein
            object with the provided string as the attribute name. Default is
            ``None`` (not stored).
        :type cache: Optional[str]
        :return: Alpha carbon coordinates
        :rtype: graphein.protein.tensor.types.CoordTensor
        """

        out = get_c_alpha(self.coords)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def backbone(self, cache: Optional[str] = None) -> BackboneTensor:
        """

        .. code-block:: python
            import graphein.protein.tensor as gpt
            protein = gpt.Protein().from_pdb_code(pdb_code="3eiy")

            protein.backbone().shape # torch.Size([374, 4, 3])

            protein.backbone(cache="bb")
            protein.bb.shape # torch.Size([374, 4, 3])

        .. seealso:: :meth:`graphein.protein.tensor.geometry.get_backbone`

        :param cache: If provided, the result will be cached in the ``Protein``
            object with the provided string as the attribute name. Default is
            ``None`` (not stored).
        :type cache: Optional[str]
        :return: Backbone coordinates ``[Length x 4 x 3]``
        :rtype: graphein.protein.tensor.types.BackboneTensor
        """
        out = get_backbone(self.coords)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def backbone_frames(
        self, cache: Optional[str] = None
    ) -> Tuple[BackboneFrameTensor, CoordTensor]:
        """Computes backbone rotation frames from an idealised residue.

        .. see:: :func:`graphein.protein.tensor.representation.get_backbone_frames`

        .. seealso:: :func:`graphein.protein.tensor.reconstruction.get_ideal_backbone_coords`
        :param cache: If provided, the result will be cached in the ``Protein``
            object with the provided string as the attribute name. Default is
            ``None`` (not stored).
        :type cache: Optional[str]
        :return: _description_
        :rtype: Tuple[BackboneFrameTensor, CoordTensor]
        """
        out = get_backbone_frames(self.coords)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def edges(
        self,
        edge_type: str = "knn_10",
        x: Optional[torch.Tensor] = None,
        cache: Optional[str] = None,
        **kwargs,
    ) -> EdgeTensor:
        """
        Computes edges for a batch of proteins as specified by the ``edge_type``
        argument. If ``x`` is not provided, the edges are calculated based on
        Alpha Carbon positions by default.

        .. code-block:: python
            import graphein.protein.tensor as gpt
            protein = gpt.Protein().from_pdb_code(pdb_code="3eiy")

            batch = gpt.ProteinBatch().from_data_list([protein, protein, protein])

            batch.compute_edges("knn_8") # Alpha carbon graph with KNN edges (k=8)
            batch.compute_edges("eps_6") # Alpha carbon graph with radius (r=6)

            fa = batch.full_atom_coords()
            batch.compute_edges("knn_10", x=fa, cache="atomic_edges") # Full atom graph with KNN edges
            batch.atomic_edges.shape # torch.Size([2, ...])

        .. seealso::

            :func:`graphein.protein.tensor.edges.compute_edges`
            :func:`graphein.protein.tensor.edges.radius_edges`
            :func:`graphein.protein.tensor.edges.knn_edges`

        :param edge_type: Str denoting type of edges in form
            ``{edgetype}_{value}``. E.g. ``"knn_8"`` for KNN with ``k=8``,
            ``"eps_6"`` for radius graph with ``r=6``.
        :type edge_type: str
        :param x: Node feature/position matrix used to calculate edges. If
            ``None`` (default), the alpha carbon trace is used.
        :type x: Optional[torch.Tensor]
        :param cache: String to set edges as a ``Protein`` attribute. If
            ``None`` (default), edges are not saved.
        :type cache: Optional[str]
        """
        if x is None:
            x = self.alpha_carbon()
        out = compute_edges(
            x=x, edge_type=edge_type, batch=self.batch, **kwargs
        )
        if cache is not None:
            setattr(self, cache, out)
        return out

    # Testing
    def is_complete(self) -> bool:
        return is_complete_structure(self.coords, self.residues)

    def has_complete_backbone(self) -> bool:
        """Returns ``True`` if the protein has a complete backbone, else
        ``False``

        .. see:: :func:`graphein.protein.tensor.testing.is_complete_structure`

        .. seealso:: :meth:`graphein.protein.tensor.data.ProteinBatch.is_complete`

            :meth:`graphein.protein.tensor.testing.has_complete_backbone`
        """
        return has_complete_backbone(self.coords, fill_value=self.fill_value)

    def protein_apply(
        self, func: Callable[["Protein"], Any], rebatch: bool = False
    ) -> Union["ProteinBatch", List[Any]]:
        """Applies a function ``func`` to each ``Protein`` in the batch and
        returns the result as a list of ``Proteins`` or a new ``ProteinBatch``.

        .. code-block:: python

            import graphein.protein.tensor as gpt
            from graphein.protein.tensor.plot import plot_structure

            batch = gpt.data.ProteinBatch().from_pdb_codes(
                pdb_codes=["3eiy", "4hhb", "1a0q"]
                )

            def single_plot(protein: gpt.Protein()):
                return plot_structure(protein.coords, lines=False)

            plots = batch.protein_apply(single_plot)
            plots[2]


        .. note::

            If the function requires multiple arguments, use ``functools.partial``

        :param func: Function to apply to each ``Protein`` in the batch.
        :type func: Callable[["Protein"], Any]
        :return: List of results from applying ``func`` to each ``Protein`` in
            the batch.
        :param rebatch: If ``True`` the ``Protein``s will be rebatched into a
            new ``ProteinBatch``. Else, a list of the output of ``func`` will be
            returned,
        :rtype: Union["ProteinBatch", List[Any]]

        .. seealso::

            :meth:`graphein.protein.tensor.data.ProteinBatch.apply_to`
        """
        proteins = self.to_protein_list()

        out = [func(p) for p in proteins]
        # return ProteinBatch().from_data_list(out) if rebatch else out
        return ProteinBatch.from_data_list(out) if rebatch else out

    def apply_to(self, func: Callable[["Protein"], Any], idx: int) -> Any:
        """Applies a function ``func`` to the ``Protein`` at index ``idx`` in
        the batch. Returns the result of ``func``.

        ..code-block::

            from graphein.protein.tensor.plot import plot_structure

            def single_plot(protein: gpt.Protein()):
            return plot_structure(protein.coords, lines=False)

            plot = batch.apply_to(single_plot, 2)

        .. seealso:: :meth:`graphein.protein.tensor.data.ProteinBatch.apply`

        :param func: Function to apply to each ``Protein`` in the batch.
        :type func: Callable[["Protein"], Any]
        :param idx: Idx of ``Protein`` in the batch to apple the ``func`` to.
        :type idx: int
        :return: Output of ``func``
        :rtype: Any
        """
        return func(self.get_protein(idx))

    def get_protein(self, idx: int) -> "Protein":
        """Returns the ``idx``th protein in the batch."""
        if not hasattr(self, "_slice_dict"):
            try:
                return self.to_protein_list()[idx]
            except Exception as e:
                raise e
        return separate(
            cls=Protein,
            batch=self,
            idx=idx,
            slice_dict=self._slice_dict,
            inc_dict=self._inc_dict,
            decrement=True,
        )

    def to_protein_list(self) -> List["Protein"]:
        """
        Unbatch to a list of Proteins.

        .. code-block:: python

            import graphein.protein.tensor as gpt

            batch = gpt.data.ProteinBatch().from_pdb_codes(pdb_codes=["3eiy", "4hhb", "1a0q"])
            proteins = batch.to_protein_list() # List[Protein]

        :returns: List of Proteins
        :rtype: List["Protein"]
        """
        if hasattr(self, "_slice_dict"):
            return [self.get_protein(i) for i in range(len(self))]

        proteins = [Protein() for _ in range(self.num_graphs)]

        # Iterate over attributes
        keys = (
            self.keys()
            if PYG_VERSION >= looseversion.LooseVersion("2.4.0")
            else self.keys
        )
        for k in keys:
            # Get attribute
            attr = getattr(self, k)
            # Skip ptr
            if k == "ptr":
                continue
            # Unbatch tensors
            if isinstance(attr, torch.Tensor) and k != "fill_value":
                if attr.shape[0] == len(proteins):
                    temp = [attr[i] for i in range(len(proteins))]
                try:
                    temp = unbatch(getattr(self, k), self.batch)
                # Try unbatch edge index if unbatch fails
                except:
                    temp = unbatch_edge_index(getattr(self, k), self.batch)
                # Set tensor attribute on proteins in list
                for i, p in enumerate(proteins):
                    setattr(p, k, temp[i])
            # Add batch list values to proteins in list
            elif isinstance(attr, list) or k == "fill_value":
                for i, p in enumerate(proteins):
                    setattr(p, k, attr[i])

        return proteins

    def __eq__(self, __o: object) -> bool:
        # sourcery skip: merge-duplicate-blocks, merge-else-if-into-elif
        keys = (
            self.keys()
            if PYG_VERSION >= looseversion.LooseVersion("2.4.0")
            else self.keys
        )
        for i in keys:
            attr_self = getattr(self, i)
            attr_other = getattr(__o, i)

            if isinstance(attr_self, torch.Tensor):
                if not is_tensor_equal(attr_self, attr_other):
                    return False
            else:
                if attr_self != attr_other:
                    return False
        return True

    def plot_structure(
        self,
        index: Optional[int] = None,
    ) -> go.Figure():
        plots = self.protein_apply(lambda x: x.plot_structure())
        if index is not None:
            plots = [plots[index]]
        plots = [p._data for p in plots]
        plot_data = list(itertools.chain.from_iterable(plots))
        return go.Figure(data=plot_data)

    # Features
    def amino_acid_one_hot(
        self, num_types: int = 23, cache: Optional[str] = None
    ) -> torch.Tensor:
        out = F.one_hot(self.residue_type, num_classes=num_types)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def edge_distances(
        self,
        x: CoordTensor,
        edge_index: EdgeTensor,
        p: float = 2,
        cache: Optional[str] = None,
    ) -> torch.Tensor:
        """Computes the edges distances between nodes.

        :param x: Node positions
        :param edge_index: Edge indices
        :param p: The norm degree. Can be negative. Default: ``2``.``
        :returns: Edge distances
        :rtype: torch.Tensor
        """
        out = edge_distances(x=x, edge_index=edge_index, p=p)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def apply_structural_noise(
        self,
        x: Optional[Union[AtomTensor, CoordTensor]] = None,
        magnitude: float = 0.1,
        gaussian: bool = True,
        return_transformed: bool = True,
        cache: Optional[str] = None,
    ) -> Union[AtomTensor, CoordTensor]:
        """
        Applies noise to the structure of the proteins in the batch.

        .. see:: :func:`graphein.protein.tensor.geometry.apply_structural_noise`

        :param x: Coordinates to apply noise to, defaults to ``None``. If
            ``None`` (default), ``self.coords`` is used.
        :param cache: If provided, the result will be cached in the
            ``ProteinBatch`` under the provided string as the attribute name.
            Default is ``None``, (not stored).
        """
        if x is None:
            x = self.coords
        out = apply_structural_noise(
            x,
            magnitude=magnitude,
            gaussian=gaussian,
            return_transformed=return_transformed,
        )
        if cache is not None:
            setattr(self, cache, out)
        return out


def to_protein(
    path: Optional[str] = None,
    pdb_code: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    chain_selection: str = "all",
    deprotonate: bool = True,
    keep_insertions: bool = False,
    keep_hets: List[str] = [],
    model_index: int = 1,
    atom_types: List[str] = PROTEIN_ATOMS,
) -> "Protein":
    """
    Parses a protein (from either: a PDB code, PDB file or a UniProt ID
    (via AF2 database) to a Graphein ``Protein`` object.


    .. code-block:: python

        import graphein.protein.tensor as gpt


        # From PDB code
        gpt.data.to_protein(pdb_code="3eiy", ...)

        # From PDB Path
        gpt.io.to_protein(path="3eiy.pdb", ...)

        # From UniProt ID
        gpt.io.to_protein(uniprot_id="Q5VSL9", ...)

    .. seealso::

        :func:`graphein.protein.tensor.io.protein_to_pyg`
        :func:`graphein.protein.tensor.data.to_protein_mp`


    :param path: Path to PDB or MMTF file. Default is ``None``.
    :param pdb_code: PDB accesion code. Default is ``None``.
    :param uniprot_id: UniProt ID. Default is ``None``.
    :param chain_selection: Selection of chains to include (e.g. ``"ABC"``) or
        ``"all"``. Default is ``"all"``.
    :param deprotonate: Whether or not to remove Hydrogens. Default is ``True``.
    :param keep_insertions: Whether or not to keep insertions.
    :param keep_hets: List of heteroatoms to include. E.g. ``["HOH"]``.
    :param model_index: Index of model in models containing multiple structures.
    :param atom_types: List of atom types to select. Default is:
        :const:`graphein.protein.resi_atoms.PROTEIN_ATOMS`
    :returns: ``Data`` object with attributes: ``x`` (AtomTensor), ``residues``
        (list of 3-letter residue codes), id (ID of protein), residue_id (E.g.
        ``"A:SER:1"``), residue_type (torch.Tensor), ``chains`` (torch.Tensor).
    :rtype: Protein
    """
    data = protein_to_pyg(
        path=path,
        pdb_code=pdb_code,
        uniprot_id=uniprot_id,
        chain_selection=chain_selection,
        keep_insertions=keep_insertions,
        deprotonate=deprotonate,
        keep_hets=keep_hets,
        model_index=model_index,
        atom_types=atom_types,
    )
    return Protein().from_data(data)


def _mp_constructor(
    args: Tuple[str, str],
    deprotonate,
    keep_insertions,
    keep_hets,
    model_index,
    atom_types,
    source: str,
):
    func = partial(
        to_protein,
        deprotonate=deprotonate,
        keep_insertions=keep_insertions,
        keep_hets=keep_hets,
        model_index=model_index,
        atom_types=atom_types,
    )
    try:
        if source == "pdb_code":
            return func(
                pdb_code=args[0],
                chain_selection=args[1],  # , model_index=args[2]
            )
        elif source == "path":
            return func(
                path=args[0],
                chain_selection=args[1],  # , model_index=args[2]
            )
        elif source == "uniprot_id":
            return func(
                uniprot_id=args[0],
                chain_selection=args[1],
                # model_index=args[2],
            )
    except Exception as ex:
        log.info(
            f"Graph construction error (PDB={args[0]})! {traceback.format_exc()}"
        )
        log.info(ex)
        return None


def to_protein_mp(
    paths: Optional[str] = None,
    pdb_codes: Optional[str] = None,
    uniprot_ids: Optional[str] = None,
    chain_selections: Optional[List[str]] = None,
    deprotonate: bool = True,
    keep_insertions: bool = False,
    keep_hets: List[str] = [],
    model_index: int = 1,
    atom_types: List[str] = PROTEIN_ATOMS,
    num_cores: int = 16,
) -> List["Protein"]:
    """
    Parallelised parsing of a list of proteins (from either: PDB codes, PDB
    files or UniProt IDs (via AF2 database) to a Graphein ``Protein`` object
    using multiprocessing.


    .. code-block:: python

        import graphein.protein.tensor as gpt

        # From PDB codes
        gpt.data.to_protein_mp(pdb_codes=["3eiy", "4hhb", ..., num_cores=8])

        # From PDB Paths
        gpt.io.to_protein_mp(paths=["3eiy.pdb", "4hhb.pdb", ...])

        # From UniProt IDs
        gpt.io.to_protein_mp(uniprot_ids=["Q5VSL9", ...])


    .. seealso::

        :func:`graphein.protein.tensor.io.protein_to_pyg`
        :func:`graphein.protein.tensor.data.to_protein`


    :param paths: Path to PDB or MMTF files. Default is ``None``.
    :param pdb_codes: PDB accesion code. Default is ``None``.
    :param uniprot_ids: UniProt ID. Default is ``None``.
    :param chain_selections: Selection of chains to include (e.g. ``"ABC"``) or
        ``"all"``. Default is ``"all"``.
    :param deprotonate: Whether or not to remove Hydrogens. Default is ``True``.
    :param keep_insertions: Whether or not to keep insertions.
    :param keep_hets: List of heteroatoms to include. E.g. ``["HOH"]``.
    :param model_index: Index of model in models containing multiple structures.
    :param atom_types: List of atom types to select. Default is:
        :const:`graphein.protein.resi_atoms.PROTEIN_ATOMS`
    :param num_cores: Number of cores to use for multiprocessing.
    :returns: ``Data`` object with attributes: ``x`` (AtomTensor), ``residues``
        (list of 3-letter residue codes), id (ID of protein), residue_id (E.g.
        ``"A:SER:1"``), residue_type (torch.Tensor), ``chains`` (torch.Tensor).
    :rtype: List[Protein]
    """
    assert (
        pdb_codes is not None or paths is not None or uniprot_ids is not None
    ), "Iterable of pdb codes, pdb paths or uniprot IDs required."

    if pdb_codes is not None:
        pdbs = pdb_codes
        source = "pdb_code"

    if paths is not None:
        pdbs = paths
        source = "path"

    if uniprot_ids is not None:
        pdbs = uniprot_ids
        source = "uniprot_id"

    if chain_selections is None:
        chain_selections = ["all"] * len(pdbs)

    # if model_indices is None:
    #    model_indices = [1] * len(pdbs)

    constructor = partial(
        _mp_constructor,
        source=source,
        deprotonate=deprotonate,
        keep_insertions=keep_insertions,
        keep_hets=keep_hets,
        model_index=model_index,
        atom_types=atom_types,
    )

    return list(
        process_map(
            constructor,
            [(pdb, chain_selections[i]) for i, pdb in enumerate(pdbs)],
            max_workers=num_cores,
        )
    )


def get_random_protein() -> "Protein":
    """Utility/testing function to get a random proteins."""
    pdbs = ["3eiy", "4hhb", "1a0q", "1hcn"]
    pdb = random.choice(pdbs)
    return Protein().from_pdb_code(pdb)


def get_random_batch(num_proteins: int = 8) -> "ProteinBatch":
    """Utility/testing function to get a random batch of proteins."""

    proteins = [get_random_protein() for _ in range(num_proteins)]
    # return ProteinBatch().from_protein_list(proteins)
    return ProteinBatch.from_protein_list(proteins)
