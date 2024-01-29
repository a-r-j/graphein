"""Base Config object for use with RNA Graph Construction."""

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

from graphein.rna.edges import add_atomic_edges
from graphein.utils.config import PartialMatchOperator, PathMatchOperator


class BpRNAConfig(BaseModel):
    """Config for managing BpRNA
    :param path: Path to bpRNA
    :type path: str
    """

    path: str = "./bpRNA"


class RNAGraphConfig(BaseModel):
    """
    Config Object for RNA Structure Graph Construction.

    :param granularity: Specifies the node types of the graph, defaults to
        ``"rna"`` for atoms as nodes. Other options are ``"rna-centroid"`` or
        any RNA atom name.
    :param verbose: Specifies verbosity of graph creation process.
    :type verbose: bool
    :param rna_df_processing_functions: List of functions that take a
        ``pd.DataFrame`` and return a ``pd.DataFrame``. This allows users to
        define their own series of processing functions for the RNA structure
        DataFrame and override the default sequencing of processing steps
        provided by Graphein. We refer users to our low-level API tutorial for
        more details.
    :type rna_df_processing_functions: Optional[List[Callable]]
    :param edge_construction_functions: List of functions that take an
        ``nx.Graph`` and return an ``nx.Graph`` with desired edges added.
        Prepared edge constructions can be found in :ref:`graphein.rna.edges`
    :type edge_construction_functions: List[Callable]
    :param node_metadata_functions: List of functions that take an ``nx.Graph``
    :type node_metadata_functions: List[Callable], optional
    :param edge_metadata_functions: List of functions that take an
    :type edge_metadata_functions: List[Callable], optional
    :param graph_metadata_functions: List of functions that take an ``nx.Graph``
        and return an ``nx.Graph`` with added graph-level features and metadata.
    :type graph_metadata_functions: List[Callable], optional
    """

    granularity: str = "rna_atom"
    verbose: bool = False
    insertions: bool = False
    keep_hets: List[str] = []

    # Graph construction functions
    edge_construction_functions: List[Union[Callable, str]] = [
        add_atomic_edges
    ]
    rna_df_processing_functions: Optional[List[Callable]] = None
    node_metadata_functions: Optional[List[Union[Callable, str]]] = None
    edge_metadata_functions: Optional[List[Union[Callable, str]]] = None
    graph_metadata_functions: Optional[List[Callable]] = None

    def __eq__(self, other: Any) -> bool:
        """
        Overwrites the BaseModel __eq__ function in order to check more
        specific cases (like partial functions).
        """
        if isinstance(other, RNAGraphConfig):
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
