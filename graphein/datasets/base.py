"""Base class for working with the PROTEINS_X datasets"""
import logging
import multiprocessing

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Code Repository: https://github.com/a-r-j/graphein
import traceback
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB import PDBList
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph

logging.basicConfig(level="DEBUG")
log = logging.getLogger(__name__)


class AbstractClassificationDataset(ABC):
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        protein_graph_config: Optional[
            ProteinGraphConfig
        ] = ProteinGraphConfig(),
        num_cores: int = 16,
    ):
        if df is not None:
            self.df = df
        else:
            self.df: pd.DataFrame = self._load_dataset()
        self._num_cores: int = num_cores
        self.config = protein_graph_config

        self.sequences: List[str] = self.get_sequences(self.df)
        self.graph_labels: List[str] = self.get_graph_labels(self.df)
        self.graph_classes: List[str] = list(set(self.graph_labels))
        self.residue_labels: List[str] = self.get_residue_labels(self.df)
        self.pdb_list: List[str] = self.get_pdb_list(self.df)
        self.chain_list: List[str] = self.get_chain_list(self.df)
        self.chain_length: List[str] = self.get_chain_lengths(self.df)
        self.node_labels: List[np.array] = [
            self.encode_interactions(ex) for ex in self.residue_labels
        ]
        if protein_graph_config is not None:
            self.graphs: List[nx.Graph] = self.construct_graphs()
        self.bad_pdbs: List[str] = []

    @abstractmethod
    def _load_dataset(self):
        pass

    @abstractmethod
    def split_data(self):
        pass

    def _split_data(
        self,
        strategy: str = "random",
        training_size: float = 0.7,
        validation_size: float = 0.2,
        test_size: float = 0.1,
        shuffle: bool = True,
    ):
        if strategy == "random":
            train_df, test_df = train_test_split(
                self.df,
                train_size=training_size,
                test_size=test_size + validation_size,
                shuffle=shuffle,
            )
            val_df, test_df = train_test_split(
                test_df,
                train_size=validation_size / (validation_size + test_size),
                test_size=test_size / (test_size + validation_size),
                shuffle=shuffle,
            )
        elif strategy == "stratified":
            y = self.df[["interactor"]]
            train_df, temp_df, _, _ = train_test_split(
                self.df,
                y,
                stratify=y,
                train_size=training_size,
                test_size=test_size + validation_size,
            )
            y = temp_df[["interactor"]]
            val_df, test_df, _, _ = train_test_split(
                temp_df,
                y,
                stratify=y,
                train_size=validation_size / (validation_size + test_size),
                test_size=test_size / (validation_size + test_size),
            )
        return train_df, val_df, test_df

    @staticmethod
    def get_sequences(df) -> List[str]:
        """
        Returns the list of protein sequences in the dataset

        :return: List of protein sequences
        :rtype: List[str]
        """
        return list(df["sequence"])

    @staticmethod
    def get_residue_labels(df) -> List[str]:
        """
        Returns residue labels for sequences in the dataset (in string form).

        '-' denotes a residue is non-interacting
        '+' denotes a residue does interact
        :return: List of residue labels for the dataset
        :rtype: List[str]
        """
        return list(df["interacting_residues"])

    @staticmethod
    def get_pdb_list(df) -> List[str]:
        """
        Returns list of PDBs in the dataset

        :return: List of PDB accession codes
        :rtype: List[str]
        """
        return list(df["PDB"])

    @staticmethod
    def get_graph_labels(df) -> List[str]:
        return list(df["interactor"])

    @staticmethod
    def get_chain_list(df) -> List[str]:
        """Returns list of proteins chains in dataset

        :return: List of protein chains
        :rtype: List[str]
        """
        return list(df["chain"])

    @staticmethod
    def get_chain_lengths(df) -> List[int]:
        """Returns list of chain lengths

        :return: List of chain lengths
        :rtype: List[int]
        """
        return list(df["length"])

    @staticmethod
    def encode_interactions(s: str) -> np.array:
        """
        Encodes interacting residue string as a binary numpy array

        :param s:  String of interaction status (e.g '----++----+-')
        :type s: str
        :return: binary numpy array representation of interacting residues (nodes)
        :rtype: np.array
        """
        arr = np.array([1 if c == "+" else 0 for c in s])
        return arr

    def construct_graphs(self) -> List[nx.Graph]:
        """
        Constructs graphs for protein chains in the dataset.

        :param config: Config specifying protein graph construction parameters
        :type config: graphein.protein.ProteinGraphConfig
        :return: List of protein structure graphs
        :rtype: List[nx.Graph]
        """
        pool = multiprocessing.Pool(self._num_cores)
        graph_list = list(
            pool.map(
                self._graph_constructor,
                [
                    (pdb, self.chain_list[i])
                    for i, pdb in enumerate(self.pdb_list)
                ],
            )
        )
        pool.close()
        pool.join()

        # for i, pdb in enumerate(pdbs):
        #    assert g.number_of_nodes() == len(
        #        self.residue_labels[i]
        #    ), f"Lengths do not match: Nodes: {g.number_of_nodes()} Labels: {len(self.residue_labels[i])}"

        return graph_list

    def _graph_constructor(self, args: Tuple[str, str]):
        """
        Partialed graph constructor for multiprocessing

        :param args: Tuple of pdb code and chain to build graph of
        :type args: Tuple[str, str]
        :return: Protein structure graph
        :rtype: nx.Graph
        """
        log.info(
            f"Constructing graph for: {args[0]}. Chain selection: {args[1]}"
        )
        func = partial(construct_graph, config=self.config)
        try:
            result = func(pdb_code=args[0], chain_selection=args[1])
            return result
        except:
            log.info(
                f"Graph construction error (PDB={args[0]})! {traceback.format_exc()}"
            )
            self.bad_pdbs.append(args[0])

    def download_pdbs(self, path: str):
        """
        Downloads dataset PDBs to a specified directories

        :param path: Path to desired output location
        :type path: str
        """
        pdbl = PDBList()
        pdbl.download_pdb_files(pdb_codes=self.pdb_list, pdir=path)

    def __len__(self) -> int:
        """Returns length of the dataset

        :returns: Dataset length
        :rtype: int
        """
        return len(self.df)
