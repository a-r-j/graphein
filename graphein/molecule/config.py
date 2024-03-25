"""Base Config object for use with Molecule Graph Construction."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from deepdiff import DeepDiff
from pydantic import BaseModel

from graphein.molecule.edges.atomic import add_atom_bonds
from graphein.molecule.edges.distance import (
    add_distance_threshold,
    add_fully_connected_edges,
    add_k_nn_edges,
)
from graphein.molecule.features.nodes.atom_type import atom_type_one_hot
from graphein.utils.config import PartialMatchOperator, PathMatchOperator

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


GraphAtoms = Literal[
    "C",
    "H",
    "O",
    "N",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
]
"""Allowable atom types for nodes in the graph."""


class MoleculeGraphConfig(BaseModel):
    """
    Config Object for Molecule Structure Graph Construction.

    :param verbose: Specifies verbosity of graph creation process.
    :type verbose: bool
    :param add_hs: Specifies whether hydrogens should be added to the graph.
    :type add_hs: bool
    :param generate_conformer: Specifies whether or not to generate a conformer for inputs that do not provide coordinates (e.g. SMILES).
        Uses :meth:`graphein.molecule.graphs.generate_3d` to generate conformers.
        Default: ``False``.
    :param edge_construction_functions: List of functions that take an ``nx.Graph`` and return an ``nx.Graph`` with desired
        edges added. Prepared edge constructions can be found in :ref:`graphein.protein.edges`
    :type edge_construction_functions: List[Callable]
    :param node_metadata_functions: List of functions that take an ``nx.Graph``
    :type node_metadata_functions: List[Callable], optional
    :param edge_metadata_functions: List of functions that take an
    :type edge_metadata_functions: List[Callable], optional
    :param graph_metadata_functions: List of functions that take an ``nx.Graph`` and return an ``nx.Graph`` with added
        graph-level features and metadata.
    :type graph_metadata_functions: List[Callable], optional
    """

    verbose: bool = False
    add_hs: bool = False
    generate_conformer: bool = False
    # Graph construction functions
    edge_construction_functions: List[Union[Callable, str]] = [
        add_fully_connected_edges,
        add_k_nn_edges,
        add_distance_threshold,
        add_atom_bonds,
    ]
    node_metadata_functions: Optional[List[Union[Callable, str]]] = [
        atom_type_one_hot
    ]
    edge_metadata_functions: Optional[List[Union[Callable, str]]] = None
    graph_metadata_functions: Optional[List[Callable]] = None

    def __eq__(self, other: Any) -> bool:
        """Overwrites the BaseModel __eq__ function in order to check more specific cases (like partial functions)."""
        if isinstance(other, MoleculeGraphConfig):
            return (
                DeepDiff(
                    self,
                    other,
                    custom_operators=[
                        PartialMatchOperator(types=[partial]),
                        PathMatchOperator(types=[Path]),
                    ],
                )
                == {}
            )
        return self.dict() == other
