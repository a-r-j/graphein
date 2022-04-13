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
from deepdiff.operator import BaseOperator
from pydantic import BaseModel
from typing_extensions import Literal

from graphein.molecule.edges.atomic import add_atom_bonds

from graphein.molecule.edges.distance import add_k_nn_edges, add_distance_threshold, add_fully_connected_edges

from graphein.molecule.features.nodes.atom_type import atom_type_one_hot

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

def partial_functions_equal(func1: partial, func2: partial) -> bool:
    """
    Determine whether two partial functions are equal.

    :param func1: Partial function to check
    :type func1: partial
    :param func2: Partial function to check
    :type func2: partial
    :return: Whether the two functions are equal
    :rtype: bool
    """
    if not (isinstance(func1, partial) and isinstance(func2, partial)):
        return False
    return all(
        getattr(func1, attr) == getattr(func2, attr)
        for attr in ["func", "args", "keywords"]
    )


class PartialMatchOperator(BaseOperator):
    """Custom operator for deepdiff comparison. This operator compares whether the two partials are equal."""

    def give_up_diffing(self, level, diff_instance):
        return partial_functions_equal(level.t1, level.t2)


class PathMatchOperator(BaseOperator):
    """Custom operator for deepdiff comparison. This operator compares whether the two pathlib Paths are equal."""

    def give_up_diffing(self, level, diff_instance):
        return level.t1 == level.t2

class MoleculeGraphConfig(BaseModel):
    """
    Config Object for Molecule Structure Graph Construction.

    :param verbose: Specifies verbosity of graph creation process.
    :type verbose: bool
    :param deprotonate: Specifies whether or not to remove ``H`` atoms from the graph.
    :type deprotonate: bool
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
    deprotonate: bool = False
    # Graph construction functions
    edge_construction_functions: List[Union[Callable, str]] = [
        add_fully_connected_edges, add_k_nn_edges, add_distance_threshold, add_atom_bonds
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