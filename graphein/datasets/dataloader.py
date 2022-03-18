"""Dataloader class for working with protein structure graphs."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from Bio.PDB import PDBList

from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graphs_mp
from graphein.utils.utils import import_message

logging.basicConfig(level="DEBUG")
log = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    message = import_message(
        "graphein.datasets.dataloader", "torch", pip_install=True
    )
    log.warning(message)
    log.warning("PyTorch not installed. Skipping torch-related functionality.")

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    message = import_message(
        "graphein.datasets.dataloader", "torch_geometric", pip_install=True
    )
    log.warning(message)


class GrapheinDataLoader:
    def __init__(
        self,
        pdb_file_list: Optional[List[str]] = None,
        pdb_code_list: Optional[List[str]] = None,
        pdb_in_dir: Optional[str] = None,
        pdb_out_dir: str = "/tmp/",
        graph_labels: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
        node_labels: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
        chain_list: Optional[List[str]] = None,
        protein_graph_config: Optional[
            ProteinGraphConfig
        ] = ProteinGraphConfig(),
        num_cores: int = 16,
        graph_convertor: Optional[GraphFormatConvertor] = None,
        train_idxs: Optional[List[int]] = None,
        valid_idxs: Optional[List[int]] = None,
        test_idxs: Optional[List[int]] = None,
        batch_size: int = 32,
        shuffle_batches: bool = True,
        drop_last: bool = True,
    ):
        """Dataloader class for working with protein structure graphs.

        :param pdb_file_list: List of paths to PDB files to use, defaults to None
        :type pdb_file_list: Optional[List[str]], optional
        :param pdb_code_list: List of PDB codes to use, defaults to None
        :type pdb_code_list: Optional[List[str]], optional
        :param pdb_in_dir: Path to direcotry containing pdb files to use, defaults to None
        :type pdb_in_dir: Optional[str], optional
        :param pdb_out_dir: Download location for PDB files, defaults to "/tmp/"
        :type pdb_out_dir: str, optional
        :param graph_labels: Labels for protein graphs, defaults to None
        :type graph_labels: Optional[List[Union[np.ndarray, torch.Tensor]]], optional
        :param node_labels: Labels for nodes in protein graphs, defaults to None
        :type node_labels: Optional[List[Union[np.ndarray, torch.Tensor]]], optional
        :param chain_list: List of chain selections for each structures, defaults to None
        :type chain_list: Optional[List[str]], optional
        :param protein_graph_config: Configuration object for protein graph construction, defaults to ProteinGraphConfig()
        :type protein_graph_config: Optional[ ProteinGraphConfig ], optional
        :param num_cores: Number of cores to use for multiprocessing of graph construction, defaults to 16
        :type num_cores: int, optional
        :param graph_convertor: GraphFormatConversion object to handle nc.Graph -> PyG Data conversion. Also specifies which data are kept from the source nx.Graph, defaults to None
        :type graph_convertor: Optional[GraphFormatConvertor], optional
        :param train_idxs: List of indices corresponding to training examples, defaults to None
        :type train_idxs: Optional[List[int]], optional
        :param valid_idxs: List of indices corresponding to validation examples, defaults to None
        :type valid_idxs: Optional[List[int]], optional
        :param test_idxs: List of indices corresponding to test examples, defaults to None
        :type test_idxs: Optional[List[int]], optional
        :param batch_size: Batch size for dataloader, defaults to 32
        :type batch_size: int, optional
        :param shuffle_batches: Whether or not to shuffle batches, defaults to True
        :type shuffle_batches: bool, optional
        :param drop_last: Whether or not to drop last batch, defaults to True
        :type drop_last: bool, optional
        """
        # Graphein objects for constructing and converting graphs
        self.protein_graph_config = protein_graph_config
        self.graph_convertor = graph_convertor

        # PDB inputs
        self.pdb_file_list = pdb_file_list
        self.pdb_code_list = pdb_code_list
        self.pdb_in_dir = pdb_in_dir
        self.pdb_out_dir = pdb_out_dir
        self.chain_list = chain_list

        # Indexes for train/test/split
        self.train_idxs = train_idxs
        self.valid_idxs = valid_idxs
        self.test_idxs = test_idxs

        # Get PDBs from input
        if self.pdb_code_list:
            self._validate_pdb_codes()
            self.pdb_source = "code"
            self.pdb_list = self.pdb_code_list
        if self.pdb_file_list:
            self._validate_pdb_files()
            self.pdb_source = "file"
            self.pdb_list = self.pdb_file_list
        if self.pdb_in_dir:
            self.pdb_file_list = os.listdir(self.pdb_in_dir)
            self.pdb_file_list = [
                f for f in self.pdb_file_list if f.endswith(".pdb")
            ]
            self.pdb_source = "file"
            self.pdb_list = self.pdb_file_list
        if self.chain_list is None:
            self.chain_list = ["all"] * len(self.pdb_list)

        # Graph labels
        self.graph_labels = graph_labels
        self.node_labels = node_labels

        self._validate_input_data()

        self._num_cores = num_cores

        # Data loader params
        self.batch_size = batch_size
        self.shuffle_batches = shuffle_batches
        self.drop_last = drop_last

    def split_data(
        self,
    ) -> Tuple[GrapheinDataLoader, GrapheinDataLoader, GrapheinDataLoader]:
        """Splits data based on provided indices.

        :returns: Tuple of GrapheinDataLoaders for train, validation, and test
        :rtype: Tuple[GrapheinDataLoader, GrapheinDataLoader, GrapheinDataLoader]
        """
        loaders: List[GrapheinDataLoader] = []
        if self.train_idxs:
            train = GrapheinDataLoader(
                pdb_file_list=[
                    self.pdb_file_list[idx] for idx in self.train_idxs
                ]
                if self.pdb_file_list is not None
                else None,
                pdb_code_list=[
                    self.pdb_code_list[idx] for idx in self.train_idxs
                ]
                if self.pdb_code_list is not None
                else None,
                graph_labels=[
                    self.graph_labels[idx] for idx in self.train_idxs
                ]
                if self.graph_labels is not None
                else None,
                node_labels=[self.node_labels[idx] for idx in self.train_idxs]
                if self.node_labels is not None
                else None,
                chain_list=[self.chain_list[idx] for idx in self.train_idxs]
                if self.node_labels is not None
                else None,
                pdb_in_dir=self.pdb_in_dir,
                pdb_out_dir=self.pdb_out_dir,
                protein_graph_config=self.protein_graph_config,
                graph_convertor=self.graph_convertor,
                num_cores=self._num_cores,
                batch_size=self.batch_size,
                shuffle_batches=self.shuffle_batches,
                drop_last=self.drop_last,
            )
            self.train = train
            loaders.append(train)
        if self.valid_idxs:
            valid = GrapheinDataLoader(
                pdb_file_list=[
                    self.pdb_file_list[idx] for idx in self.valid_idxs
                ]
                if self.pdb_file_list is not None
                else None,
                pdb_code_list=[
                    self.pdb_code_list[idx] for idx in self.valid_idxs
                ]
                if self.pdb_code_list is not None
                else None,
                graph_labels=[
                    self.graph_labels[idx] for idx in self.valid_idxs
                ]
                if self.graph_labels is not None
                else None,
                node_labels=[self.node_labels[idx] for idx in self.valid_idxs]
                if self.node_labels is not None
                else None,
                chain_list=[self.chain_list[idx] for idx in self.valid_idxs]
                if self.node_labels is not None
                else None,
                pdb_in_dir=self.pdb_in_dir,
                pdb_out_dir=self.pdb_out_dir,
                protein_graph_config=self.protein_graph_config,
                graph_convertor=self.graph_convertor,
                num_cores=self._num_cores,
                batch_size=self.batch_size,
                shuffle_batches=self.shuffle_batches,
                drop_last=self.drop_last,
            )
            self.valid = valid
            loaders.append(valid)
        if self.test_idxs:
            test = GrapheinDataLoader(
                pdb_file_list=[
                    self.pdb_file_list[idx] for idx in self.test_idxs
                ]
                if self.pdb_file_list is not None
                else None,
                pdb_code_list=[
                    self.pdb_code_list[idx] for idx in self.test_idxs
                ]
                if self.pdb_code_list is not None
                else None,
                graph_labels=[self.graph_labels[idx] for idx in self.test_idxs]
                if self.graph_labels is not None
                else None,
                node_labels=[self.node_labels[idx] for idx in self.test_idxs]
                if self.node_labels is not None
                else None,
                chain_list=[self.chain_list[idx] for idx in self.test_idxs]
                if self.node_labels is not None
                else None,
                pdb_in_dir=self.pdb_in_dir,
                pdb_out_dir=self.pdb_out_dir,
                protein_graph_config=self.protein_graph_config,
                graph_convertor=self.graph_convertor,
                num_cores=self._num_cores,
                batch_size=self.batch_size,
                shuffle_batches=self.shuffle_batches,
                drop_last=self.drop_last,
            )
            self.test = test
            loaders.append(test)

        return tuple(loaders)

    def construct_graphs(self) -> List[nx.Graph]:
        """
        Constructs graphs for protein chains in the dataset.

        :param config: Config specifying protein graph construction parameters
        :type config: graphein.protein.ProteinGraphConfig
        :return: List of protein structure graphs
        :rtype: List[nx.Graph]
        """
        graph_list = construct_graphs_mp(
            pdb_code_it=self.pdb_code_list,
            pdb_path_it=self.pdb_file_list,
            chain_selections=self.chain_list,
            num_cores=self._num_cores,
            return_dict=False,
        )

        if self.graph_convertor:
            graph_list = self.convert_graphs(graph_list, self.graph_convertor)

        self.graphs = graph_list
        return graph_list

    def convert_graphs(
        self, graph_list: List[nx.Graph], convertor: GraphFormatConvertor
    ) -> List[Data]:
        """Converts graphs to desired format.

        :param graph_list: List of graphs to conert
        :type graph_list: List[nx.Graph]
        :param convertor: Graph format convertor
        :type convertor: GraphFormatConvertor
        :return: List of graphs converted to desired format.
        :rtype: List[Data]
        """
        graph_list = [convertor(g) for g in graph_list]
        if self.node_labels:
            self._validate_node_labels(graph_list)
            for i, g in enumerate(graph_list):
                g.node_y = self.node_labels[i]
        if self.graph_labels:
            for i, g in enumerate(graph_list):
                g.graph_y = self.graph_labels[i]
        return graph_list

    def dataloader(self) -> DataLoader:
        """Returns a PyTorch Geometric Dataloader for the dataset.

        :return: Pytorch Geometric Dataloader
        :rtype: DataLoader
        """
        loader = DataLoader(
            self.graphs,
            batch_size=self.batch_size,
            shuffle=self.shuffle_batches,
            drop_last=self.drop_last,
        )
        self.loader = loader
        return loader

    def download_pdbs(self):
        """
        Downloads dataset PDBs to a specified directories.

        :param path: Path to desired output location
        :type path: str
        """
        pdbl = PDBList()
        pdbl.download_pdb_files(
            pdb_codes=self.pdb_code_list, pdir=self.pdb_out_dir
        )

    def _validate_pdb_codes(self):
        """Validates provided PDB codes."""
        for code in self.pdb_code_list:
            assert (
                len(code) == 4
            ), f"PDB codes {code} must be 4 characters long."

    def _validate_pdb_files(self):
        """Validates provided PDB files by asserting that they exist."""
        for file in self.pdb_file_list:
            assert os.path.isfile(file), f"PDB file {file} does not exist."

    def _validate_node_labels(self, graph_list: List[nx.Graph]):
        """Checks length of node labels matches number of nodes in graph."""
        for i, g in enumerate(graph_list):
            assert (
                g.number_of_nodes() == self.node_labels[i].shape[0]
            ), f"Node label lengths ({self.node_labels[i].shape[0]}) do not match number of noes ({g.number_of_nodes()})"

    def _validate_input_data(self):
        """Checks length of provided inputs match with number of structures provided."""
        if self.node_labels:
            assert len(self.node_labels) == len(
                self.pdb_list
            ), f"Node labels length ({len(self.node_labels)}) does not match PDB list length ({len(self.pdb_list)})"
        if self.graph_labels:
            assert len(self.graph_labels) == len(
                self.pdb_list
            ), f"Graph labels length ({len(self.graph_labels)}) does not match PDB list length ({len(self.pdb_list)})"
        if self.chain_list:
            assert len(self.chain_list) == len(
                self.pdb_list
            ), f"Chain list length ({len(self.chain_list)}) does not match PDB list length ({len(self.pdb_list)})"

    def __repr__(self) -> str:
        """Print summary for dataloader."""
        summary = (
            f"Graphein Dataloader containing: {len(self.pdb_list)} PDBs. \n"
        )
        if self.graph_labels:
            summary += "Using graph labels.\n"
        if self.node_labels:
            summary += "Using node labels.\n"
        return summary

    def __len__(self) -> int:
        return len(self.pdb_list) if not self.graphs else len(self.graphs)


if __name__ == "__main__":
    import numpy as np

    pdb_list = [
        "3eiy",
        "4hhb",
        "1a0a",
        "1a0b",
        "1a0c",
        "1a0d",
        "1a0e",
        "1a0f",
        "1a0g",
        "1a0h",
        "1a0i",
        "1a0j",
        "1a0k",
        "1a0l",
        "1a0m",
        "1a0n",
        "1a0o",
        "1a0p",
        "1a0q",
        "1a0r",
        "1a0s",
        "1a0t",
        "1a0u",
        "1a0v",
        "1a0w",
        "1a0x",
        "1a0y",
    ]
    train_idxs = list(range(0, 10))
    valid_idx = list(range(10, 20))
    test_idx = list(range(20, 25))

    format_convertor = GraphFormatConvertor(
        "nx", "pyg", verbose="gnn", columns=None
    )

    dl = GrapheinDataLoader(
        pdb_code_list=pdb_list,
        train_idxs=train_idxs,
        valid_idxs=valid_idx,
        test_idxs=test_idx,
        graph_convertor=format_convertor,
        batch_size=2,
        graph_labels=[np.random.randint(1)] * 27,
    )

    dl.split_data()
    dl.train.construct_graphs()
    dl.valid.construct_graphs()
    dl.test.construct_graphs()
