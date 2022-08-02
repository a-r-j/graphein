"""Pytorch LightningDataModules for the ProteinsX datasets."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import abc
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from loguru import logger as log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.utils import get_obsolete_mapping
from graphein.utils.utils import import_message

try:
    from .torch_geometric_dataset import ProteinGraphDataset
except ImportError or NameError:
    pass

try:
    from torch_geometric.data import DataLoader
except ImportError:
    message = import_message(
        submodule="graphein.ml.datasets.proteins_x",
        package="torch_geometric",
        conda_channel="pyg",
        pip_install=True,
    )
    log.warning(message)

try:
    import pytorch_lightning as pl
except ImportError:
    message = import_message(
        submodule="graphein.ml.datasets.proteins_x",
        package="pytorch_lightning",
        pip_install=True,
    )
    log.warning(message)


class ProteinsXBase(pl.LightningDataModule, abc.ABC):
    """Abstract Base Class for ProteinsX datasets."""

    def __init__(
        self,
        root: str,
        graphein_config: ProteinGraphConfig,
        graph_format_convertor: GraphFormatConvertor,
        split_sizes: List[float] = [0.7, 0.1, 0.2],  # Train, val, test
        split_strategy: str = "random",
        dset_fraction: float = 1.0,
        drop_obsolete: bool = True,
        num_cores: int = 16,
        graph_labels: bool = True,
        node_labels: bool = True,
        batch_size: int = 16,
        graph_transformation_funcs: Optional[List[Callable]] = None,
        pdb_transform: Optional[List[Callable]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """
        Initialises DataModule.

        :param root: Path to the root directory of the dataset. Downloaded
            structures will be placed in ``root/raw`` and processed files in
            ``root/processed``.
        :type root: str
        :param graphein_config: ProteinGraphConfig object.
        :type graphein_config: ProteinGraphConfig
        :param graph_format_convertor: GraphFormatConvertor object.
        :type graph_format_convertor: GraphFormatConvertor
        :param split_sizes: List of floats between 0 and 1. E.g.
            ``[0.7, 0.1, 0.2]`` uses 70% of the dataset for training, 10% for
            validation and 20% for testing.
        :type split_sizes: List[float]
        :param split_strategy: Strategy to use for splitting the dataset.
            Default is ``random``.
        :type split_strategy: str
        :param dset_fraction: Fraction of the dataset to use. Default is
            ``1.0`` (uses all the data).
        :type dset_fraction: float
        :param drop_obsolete: Whether to drop obsolete PDBs. Default is
            ``True``.
        :type drop_obsolete: bool
        :param num_cores: Number of cores to use for multiprocessing. Default
            is ``16``.
        :type num_cores: int
        :param graph_labels: Whether to add graph-level labels. Default is
            ``True``.
        :type graph_labels: bool
        :param node_labels: Whether to add node-level labels. Defaults to
            ``True``.
        :type node_labels: bool
        :param batch_size: Batch size for the dataloader. Default is ``16``.
        :type batch_size: int
        :param graph_transformation_funcs: List of functions that consume a
            ``nx.Graph`` and return a ``nx.Graph``. Applied to graphs after
            construction but before conversion to pyg. Defaults to ``None``.
        :type graph_transformation_funcs: Optional[List[Callable]]
        :param pdb_transform: List of functions that consume a list of paths to
            the downloaded structures. This provides an entry point to apply
            pre-processing from bioinformatics tools of your choosing. Defaults
            to ``None``.
        :type pdb_transform: Optional[List[Callable]]
        :param transform: A function/transform that takes in a
            ``torch_geometric.data.Data`` object and returns a transformed
            version. The data object will be transformed before every access.
            Defaults to ``None``.
        :type transform: Optional[Callable]
        :param pre_transform:  A function/transform that takes in an
            ``torch_geometric.data.Data`` object and returns a transformed
            version. The data object will be transformed before being saved to
            disk. Defaults to ``None``.
        :type pre_transform: Optional[Callable], optional
        :param pre_filter:  A function that takes in a
            ``torch_geometric.data.Data`` object and returns a boolean value,
            indicating whether the data object should be included in the final
            dataset. Optional, defaults to ``None``.
        :type pre_filter: Optional[Callable], optional
        """
        # Graphein configs
        self.graphein_config = graphein_config
        self.graph_format_convertor = graph_format_convertor

        # Labels
        self.graph_labels = graph_labels
        self.node_labels = node_labels

        # Dataloader Config
        self.root = root
        self.batch_size = batch_size
        self.graph_transformation_funcs = graph_transformation_funcs
        self.pdb_transform = pdb_transform
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        # Data split
        self.split_sizes = split_sizes
        self.drop_obsolete = drop_obsolete
        self.split_strategy = split_strategy
        self.dset_fraction = dset_fraction

        # Compute config
        self.num_cores = num_cores

    @abc.abstractmethod
    def load_data(self):
        """Loads dataset from disk."""
        ...

    def setup(self):
        """Sequences dataset initialisation steps."""
        self.load_data()
        self.check_for_obsolete_pdbs()
        self.get_graph_labels()
        self.get_node_labels()
        self.split_data()
        self.setup_val_dataset()
        self.setup_test_dataset()

    def setup_train_dataset(self):
        self.train_ds = ProteinGraphDataset(
            root=self.root,
            pdb_codes=list(self.train_split["PDB"]),
            graph_labels=list(self.train_split["interactor_label"])
            if self.graph_labels
            else None,
            node_labels=list(self.train_split["node_labels"])
            if self.node_labels
            else None,
            chain_selections=list(self.train_split["chain"]),
            graphein_config=self.graphein_config,
            graph_format_convertor=self.graph_format_convertor,
            num_cores=self.num_cores,
            graph_transformation_funcs=self.graph_transformation_funcs,
            transform=self.transform,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
        )

    def setup_val_dataset(self):
        self.valid_ds = ProteinGraphDataset(
            root=self.root,
            pdb_codes=list(self.val_split["PDB"]),
            graph_labels=list(self.val_split["interactor_label"])
            if self.graph_labels
            else None,
            node_labels=list(self.val_split["node_labels"])
            if self.node_labels
            else None,
            chain_selections=list(self.val_split["chain"]),
            graphein_config=self.graphein_config,
            graph_format_convertor=self.graph_format_convertor,
            num_cores=self.num_cores,
            graph_transformation_funcs=self.graph_transformation_funcs,
            transform=self.transform,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
        )

    def setup_test_dataset(self):
        self.test_ds = ProteinGraphDataset(
            root=self.root,
            pdb_codes=list(self.test_split["PDB"]),
            graph_labels=list(self.test_split["interactor_label"])
            if self.graph_labels
            else None,
            node_labels=list(self.test_split["node_labels"])
            if self.node_labels
            else None,
            chain_selections=list(self.test_split["chain"]),
            graphein_config=self.graphein_config,
            graph_format_convertor=self.graph_format_convertor,
            num_cores=self.num_cores,
            graph_transformation_funcs=self.graph_transformation_funcs,
            transform=self.transform,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_cores,
        )

    def valid_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_ds, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False
        )

    def split_data(self):
        """Splits dataset according to split_sizes and split_strategy."""
        if self.split_strategy == "random":
            train_ds, val_ds = train_test_split(
                self.ds, train_size=self.split_sizes[0]
            )
            val_ds, test_ds = train_test_split(
                val_ds,
                test_size=self.split_sizes[-1]
                / (self.split_sizes[1] + self.split_sizes[2]),
            )
        else:
            # Todo: BLAST-based splitting
            raise NotImplementedError("Only random split is implemented.")

        self.train_split = train_ds
        self.val_split = val_ds
        self.test_split = test_ds

    def get_graph_labels(self):
        """Converts interactor types to numeric graph-level labels"""
        le = LabelEncoder()
        self.ds["interactor_label"] = le.fit_transform(self.ds["interactor"])

    def get_node_labels(self, one_hot: bool = False):
        """Converts node labels to numeric node-level labels"""
        node_labels = []
        for i in self.ds["interacting_residues"]:
            indices = self.find_occurrences(i, "+")
            if one_hot:
                arr = np.zeros(len(i))
                arr[indices] = 1
                node_labels.append(arr)
            else:
                node_labels.append(indices)
        self.ds["node_labels"] = node_labels

    def check_for_obsolete_pdbs(self):
        """Checks for obsolete PDBs and removes them from the dataset."""
        obs_pdbs = [
            pdb
            for pdb in self.ds["PDB"]
            if pdb in get_obsolete_mapping().keys()
        ]
        print(f"Found {len(obs_pdbs)} obsolete pdbs: {obs_pdbs}")
        self.obs_pdbs = obs_pdbs
        if self.drop_obsolete:
            self.ds = self.ds.loc[~self.ds["PDB"].isin(obs_pdbs)]

    @staticmethod
    def find_occurrences(s: str, ch: str) -> np.ndarray:
        return np.array([i for i, letter in enumerate(s) if letter == ch])


class ProteinsMetal(ProteinsXBase):
    """LightningDataModule for ProteinsMetal dataset."""

    def load_data(self):
        """Loads dataset from disk."""
        self.ds = pd.read_csv(
            Path(__file__).parent.parent.parent.parent
            / "datasets"
            / "proteins_metal"
            / "proteins_metal.csv"
        ).sample(frac=self.dset_fraction)


class ProteinsNucleic(ProteinsXBase):
    """LightningDataModule for ProteinsNucleic dataset."""

    def load_data(self):
        """Loads dataset from disk."""
        self.ds = pd.read_csv(
            Path(__file__).parent.parent.parent.parent
            / "datasets"
            / "proteins_nucleic"
            / "proteins_nucleic.csv"
        ).sample(frac=self.dset_fraction)


class ProteinsNucleotides(ProteinsXBase):
    """LightningDataModule for ProteinsNucleotides dataset."""

    def load_data(self):
        """Loads dataset from disk."""
        self.ds = pd.read_csv(
            Path(__file__).parent.parent.parent.parent
            / "datasets"
            / "proteins_nucleotides"
            / "proteins_nucleotides.csv"
        ).sample(frac=self.dset_fraction)


class ProteinsLigands(ProteinsXBase):
    """LightningDataModule for ProteinsLigands dataset."""

    def load_data(self):
        """Loads dataset from disk."""
        self.ds = pd.read_csv(
            Path(__file__).parent.parent.parent.parent
            / "datasets"
            / "proteins_ligands"
            / "proteins_ligands.csv"
        ).sample(frac=self.dset_fraction)
