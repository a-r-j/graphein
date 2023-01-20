"""Data and Batch Objects for working proteins in PyTorch Geometric"""
import random

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Any, Callable, List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
import torch
from biopandas.pdb import PandasPdb
from torch_geometric.data import Batch, Data
from torch_geometric.utils import unbatch, unbatch_edge_index

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
from .geometry import idealize_backbone, kabsch
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
            pdb_code="4hhb", # Can alternatively pass a path or a uniprot ID (for AF2) with pdb_path=... and uniprot_id=...
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
        self, atom_list: List[str] = PROTEIN_ATOMS, fill_value: float = 1e-5
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
        """
        super().__init__()
        self.fill_value = fill_value
        self.atom_list = atom_list

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
        self.x = protein_df_to_tensor(
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
                ... pdb_code="4hhb", # Can alternatively pass a path or a uniprot ID (for AF2) with pdb_path=... and uniprot_id=...
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
        keys = data.keys
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
        for i in self.keys:
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

        :param x: AtomTensor, defaults to ``None`` (uses ``Protein.x``)
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
            x = self.x
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


        :param x: ``AtomTensor``, defaults to ``None`` (uses ``Protein.x``)
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
            x = self.x
        to_pdb(x, out_path, gz, **kwargs)

    def from_pdb_code(
        self,
        pdb_code: str,
        chain_selection: str = "all",
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
        pdb_path: str,
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
            pdb_path=pdb_path,
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

        out = get_c_alpha(self.x)
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
        out = get_backbone(self.x)
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
        out = get_backbone_frames(self.x)
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
            x = self.x
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
        out = get_full_atom_coords(self.x, fill_value=self.fill_value)
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
        out = dihedrals(self.x, rad=rad, embed=embed)

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
        return sidechain_torsion(self.x, self.residues)

    def kappa(self, cache: Optional[str] = None) -> torch.Tensor:
        """
        Computes ``kappa`` virtual angle.

        .. see:: :func:`graphein.protein.tensor.angles.kappa`

        :param: cache: If provided, the result will be cached in the ``Protein``
            under the provided string as the attribute name. Default is ``None``,
            (not stored).
        :type cache: Optional[str]
        """
        out = kappa(self.x)
        if cache is not None:
            setattr(self, cache, out)
        return out

    def alpha(self, cache: Optional[str] = None) -> torch.Tensor:
        """
        Computes ``alpha`` virtual angle.

        .. see:: :func:`graphein.protein.tensor.angles.alpha`

        :param cache: If provided, the result will be cached in the ``Protein``
            under the provided string as the attribute name. Default is ``None``,
            (not stored).
        :type cache: Optional[str]
        """
        out = alpha(self.x)
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
            self.x,
            other.x,
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

    # Testing
    def is_complete(self) -> bool:
        """
        Checks if a ``Protein`` as all the requisite atoms present.

        .. see:: :func:`graphein.protein.tensor.testing.is_complete_structure`

        :return: Boolean indicating whether or not the ``Protein`` has a
            complete structure.
        :rtype: bool
        """
        return is_complete_structure(self.x, self.residues)

    def has_complete_backbone(self) -> bool:
        """
        Checks if a ``Protein`` as all backbone atoms present.

        .. see:: :func:`graphein.protein.tensor.testing.has_complete_backbone`

        :return: Boolean indicating whether or not the ``Protein`` has a
            complete backbone.
        :rtype: bool
        """
        return has_complete_backbone(self.x)

    def __eq__(self, __o: object) -> bool:
        # sourcery skip: merge-duplicate-blocks, merge-else-if-into-elif
        for i in self.keys:
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
        dh = (
            dihedrals(self.x)
            if "dihedrals" not in self.keys
            else self.dihedrals
        )
        return plot_dihedrals(dh)

    def plot_structure(
        self, atoms: List[str] = ["N", "CA", "C", "O"], lines: bool = True
    ) -> go.Figure:
        """
        .. code-block:: python
            import graphein.protein.tensor as gpt
            protein = gpt.Protein().from_pdb_code(pdb_code="3eiy")

            protein.plot_structure(atoms=["CA"], lines=True) # Plot CA trace only
            protein.plot_structure() # Plot backbone


        .. seealso:: :meth:`graphein.protein.tensor.plot.plot_structure`

        """
        return plot_structure(self.x, atoms=atoms, lines=lines)


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
        for key in batch.keys:
            setattr(self, key, getattr(batch, key))
        self.fill_value = fill_value
        return self

    def from_protein_list(self, proteins: List[Protein]):
        proteins = [Protein().from_data(p) for p in proteins]
        batch = Batch.from_data_list(proteins)
        self.from_batch(batch)
        return self

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
                chain_selection=chain_selection[i]
                if chain_selection is not None
                else "all",
                deprotonate=deprotonate,
                keep_insertions=keep_insertions,
                keep_hets=keep_hets,
                model_index=model_index[i] if model_index is not None else 1,
                atom_types=atom_types,
                node_labels=node_labels[i]
                if node_labels is not None
                else None,
                graph_labels=graph_labels[i]
                if graph_labels is not None
                else None,
            )
            for i, pdb in enumerate(pdb_codes)
        ]
        batch = Batch.from_data_list(proteins)
        self.from_batch(batch)
        return self

    def from_pdb_files(
        self,
        pdb_paths: List[str],
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
                pdb_path=pdb,
                chain_selection=chain_selection[i]
                if chain_selection is not None
                else "all",
                deprotonate=deprotonate,
                keep_insertions=keep_insertions,
                keep_hets=keep_hets,
                model_index=model_index[i] if model_index is not None else 1,
                atom_types=atom_types,
                node_labels=node_labels[i]
                if node_labels is not None
                else None,
                graph_labels=graph_labels[i]
                if graph_labels is not None
                else None,
            )
            for i, pdb in enumerate(pdb_paths)
        ]
        batch = Batch.from_data_list(proteins)
        self.from_batch(batch)
        return self

    def to_batch(self) -> Batch:
        """Returns the ProteinBatch as a torch_geometric.data.Batch object."""
        batch = Batch()
        keys = self.keys
        for key in keys:
            setattr(batch, key, getattr(self, key))
        return batch

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

        out = get_c_alpha(self.x)
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
        out = get_backbone(self.x)
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
        out = get_backbone_frames(self.x)
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
        return is_complete_structure(self.x, self.residues)

    def has_complete_backbone(self) -> bool:
        """Returns ``True`` if the protein has a complete backbone, else
        ``False``

        .. see:: :func:`graphein.protein.tensor.testing.is_complete_structure`

        .. seealso:: :meth:`graphein.protein.tensor.data.ProteinBatch.is_complete`

            :meth:`graphein.protein.tensor.testing.has_complete_backbone`
        """
        return has_complete_backbone(self.x, fill_value=self.fill_value)

    def apply(
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
                return plot_structure(protein.x, lines=False)

            plots = batch.apply(single_plot)
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
        return ProteinBatch().from_data_list(out) if rebatch else out

    def apply_to(self, func: Callable[["Protein"], Any], idx: int) -> Any:
        """Applies a function ``func`` to the ``Protein`` at index ``idx`` in
        the batch. Returns the result of ``func``.

        ..code-block::

            from graphein.protein.tensor.plot import plot_structure

            def single_plot(protein: gpt.Protein()):
            return plot_structure(protein.x, lines=False)

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
        return self.to_protein_list()[idx]

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
        proteins = [Protein() for _ in range(self.num_graphs)]

        # Iterate over attributes
        for k in self.keys:
            # Get attribute
            attr = getattr(self, k)

            # Skip ptr
            if k == "ptr":
                continue
            # Unbatch tensors
            if isinstance(attr, torch.Tensor):
                try:
                    temp = unbatch(getattr(self, k), self.batch)
                # Try unbatch edge index if unbatch fails
                except:
                    temp = unbatch_edge_index(getattr(self, k), self.batch)
                # Set tensor attribute on proteins in list
                for i, p in enumerate(proteins):
                    setattr(p, k, temp[i])
            # Add batch list values to proteins in list
            elif isinstance(attr, list):
                for i, p in enumerate(proteins):
                    setattr(p, k, attr[i])

        return proteins

    def __eq__(self, __o: object) -> bool:
        # sourcery skip: merge-duplicate-blocks, merge-else-if-into-elif
        for i in self.keys:
            attr_self = getattr(self, i)
            attr_other = getattr(__o, i)

            if isinstance(attr_self, torch.Tensor):
                if not is_tensor_equal(attr_self, attr_other):
                    return False
            else:
                if attr_self != attr_other:
                    return False
        return True


def get_random_protein() -> "Protein":
    """Utility/testing function to get a random proteins."""
    pdbs = ["3eiy", "4hhb", "1a0q", "1hcn"]
    pdb = random.choice(pdbs)
    return Protein().from_pdb_code(pdb)


def get_random_batch(num_proteins: int = 8) -> "ProteinBatch":
    """Utility/testing function to get a random batch of proteins."""

    proteins = [get_random_protein() for _ in range(num_proteins)]
    return ProteinBatch().from_protein_list(proteins)
