"""Utilities for loading FoldComp databases for deep learning."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import asyncio
import contextlib
import os
import random
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
from biopandas.pdb import PandasPdb
from loguru import logger as log
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graphein.protein.tensor import Protein
from graphein.utils.dependencies import import_message

try:
    import foldcomp
except ImportError:
    message = import_message(
        "graphein.ml.datasets.foldcomp", "foldcomp", None, True, extras=True
    )
    log.warning(message)

try:
    import lightning as L
except ImportError:
    message = import_message(
        "graphein.ml.datasets.foldcomp", "lightning", "conda-forge", True
    )
    log.warning(message)

try:
    import nest_asyncio

    nest_asyncio.apply()
except ImportError:
    message = import_message(
        "graphein.ml.datasets.foldcomp", "nest_asyncio", None, True
    )
    message += "You can safely ignore this message if you are not working in a Jupyter Notebook."
    log.warning(message)

FOLDCOMP_DATABASE_TYPES: List[str] = [
    "afdb_swissprot_v4",
    "afdb_uniprot_v4",
    "afdb_rep_dark_v4",
    "highquality_clust30",
    "afdb_rep_v4",
]
"""
Currently supported FoldComp databases. See:
https://github.com/steineggerlab/foldcomp
"""


GraphTransform = Callable[[Union[Data, Protein]], Union[Data, Protein]]


class FoldCompDataset(Dataset):
    def __init__(
        self,
        root: str,
        database: str,
        ids: Optional[List[str]] = None,
        exclude_ids: Optional[List[str]] = None,
        fraction: float = 1.0,
        use_graphein: bool = True,
        transform: Optional[List[GraphTransform]] = None,
    ):
        """Dataset class for FoldComp databases.

        :param root: Directory where the dataset should be saved.
        :type root: str
        :param database: Name of the database. See:
            :const:`FOLDCOMP_DATABASE_TYPES`.
        :type database: str
        :param ids: List of protein IDs to include in the dataset. If ``None``,
            all proteins are included. Default is ``None``.
        :type ids: Optional[List[str]]
        :param exclude_ids: List of protein IDs to exclude from the dataset.
            If ``None``, all proteins are included. Default is ``None``.
        :type exclude_ids: Optional[List[str]]
        :param fraction: Fraction of database to use, defaults to ``1.0``.
        :type fraction: float, optional
        :param use_graphein: Whether or not to use Graphein's ``Protein``
            objects or to use standard PyG ``Data``, defaults to ``True``.
        :type use_graphein: bool, optional
        :param transform: List of functions/transform that take in a
            ``Data``/``Protein`` object and return a transformed version.
            The data object will be transformed before every access.
            (default: ``None``).
        :type transform: Optional[List[GraphTransform]]
        """
        self.root = Path(root).resolve()
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.database = database
        self.ids = ids
        self.exclude_ids = exclude_ids
        self.fraction = fraction
        self.use_graphein = use_graphein
        self.transform = transform

        _database_files = [
            "$db",
            "$db.dbtype",
            "$db.index",
            "$db.lookup",
            "$db.source",
        ]
        self.database_files = [
            f.replace("$db", self.database) for f in _database_files
        ]
        self._get_indices()
        super().__init__(
            root=self.root, transform=None, pre_transform=None  # type: ignore
        )

    @property
    def raw_file_names(self) -> List[str]:
        """Return a list of raw file names expected to be found in the database.

        Returns empty string as we don't expect any raw files.

        :return: List of raw file names
        :rtype: List[str]
        """
        return [""]

    @property
    def processed_file_names(self):
        """Returns a token to force self.process() to be called."""
        return ["_"]

    def download(self):
        """Downloads foldcomp database if not already downloaded."""

        if not all(os.path.exists(self.root / f) for f in self.database_files):
            log.info(f"Downloading FoldComp dataset {self.database}...")
            try:
                foldcomp.setup(self.database)
            except RuntimeError:
                _ = self.async_setup()
                asyncio.run(_)
            log.info("Download complete.")
            log.info("Moving files to raw directory...")

            for f in self.database_files:
                shutil.move(f, self.root)
        else:
            log.info(f"FoldComp database already downloaded: {self.root}.")

    async def async_setup(self):
        """Asynchronous setup of foldcomp database (for jupyter support)."""
        await foldcomp.setup_async(self.database)

    def _get_indices(self):
        """Get indices for the dataset."""
        # Read in look up file
        LOOKUP_FILE = Path(self.root) / f"{self.database}.lookup"
        if not os.path.exists(LOOKUP_FILE):
            self.download()
        with open(LOOKUP_FILE, "r") as f:
            accessions = f.readlines()
        # Extract accessions
        accessions = [x.strip().split("\t")[1] for x in tqdm(accessions)]
        # Get indices
        if self.ids is not None:
            accessions = (
                self.ids
            )  # [acc for acc in accessions if acc in tqdm(self.ids)]
        # Exclude indices
        if self.exclude_ids is not None:
            log.info(f"Excluding {len(self.exclude_ids)} chains...")
            accessions = [
                acc for acc in tqdm(accessions) if acc not in self.exclude_ids
            ]
        # Sub sample
        log.info(f"Sampling fraction: {self.fraction}...")
        accessions = random.sample(
            accessions, int(len(accessions) * self.fraction)
        )
        self.ids = accessions
        log.info("Creating index...")
        indices = dict(enumerate(accessions))
        self.idx_to_protein = indices
        self.protein_to_idx = {v: k for k, v in indices.items()}
        log.info(f"Dataset contains {len(self.protein_to_idx)} chains.")

    def process(self):
        """Initialises the database."""
        # Open the database
        log.info("Opening database...")
        if self.ids is not None:
            self.db = foldcomp.open(self.root / self.database, ids=self.ids)  # type: ignore
        else:
            self.db = foldcomp.open(self.root / self.database)  # type: ignore

    @staticmethod
    def _parse_dataframe(pdb_string: str) -> pd.DataFrame:
        """Reads a PDB string into a Pandas dataframe."""
        pdb: List[str] = pdb_string.split("\n")
        return PandasPdb().read_pdb_from_list(pdb).df["ATOM"]

    def process_pdb(self, pdb_string: str, name: str) -> Union[Protein, Data]:
        """Process a PDB string into a Graphein Protein object."""
        df = self._parse_dataframe(pdb_string)
        data = Protein().from_dataframe(df, id=name)
        if not self.use_graphein:
            data = data.to_data()
        return data

    def len(self) -> int:
        """Returns length of the dataset"""
        return len(self.protein_to_idx)

    def get(self, idx) -> Union[Data, Protein]:
        """Retrieves a protein from the dataset. Can idx on either the protein
        ID or its index."""
        if isinstance(idx, str):
            idx = self.protein_to_idx[idx]
        name, pdb = self.db[idx]

        out = self.process_pdb(pdb, name)

        # Apply transforms, if any
        if self.transform is not None:
            for transform in self.transform:
                out = transform(out)

        return out


class FoldCompLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        database: str,
        batch_size: int,
        use_graphein: bool = True,
        train_split: Optional[Union[List[str], float]] = None,
        val_split: Optional[Union[List[str], float]] = None,
        test_split: Optional[Union[List[str], float]] = None,
        transform: Optional[List[GraphTransform]] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        """Creates a PyTorch Lightning DataModule for FoldComp datasets.

        :param data_dir: Path to dataset directory.
        :type data_dir: str
        :param database: Name of the FoldComp database to use. See:
            :const:`FOLDCOMP_DATABASE_TYPES`.
        :type database: str
        :param batch_size: Batch size to use.
        :type batch_size: int
        :param use_graphein: Whether to use Graphein ``Protein`` objects or
            PyTorch Geometric ``Data`` objects, defaults to ``True``.
        :type use_graphein: bool, optional
        :param train_split: List of IDs or a float specifying fraction of  the
            dataset to use for training, defaults to ``None`` (whole dataset).
        :type train_split: Optional[Union[List[str], float]], optional
        :param val_split: List of IDs or a float specifying fraction of  the
            dataset to use for training, defaults to ``None`` (whole dataset).
        :type val_split: Optional[Union[List[str], float]], optional
        :param test_split: List of IDs or a float specifying fraction of  the
            dataset to use for training, defaults to ``None`` (whole dataset).
        :type test_split: Optional[Union[List[str], float]], optional
        :param transform: List of functions/transform that take in a
            ``Data``/``Protein`` object and return a transformed version.
            The data object will be transformed before every access.
            (default: ``None``).
        :type transform: Optional[List[GraphTransform]]
        :param num_workers: Number of workers to use for data loading, defaults
            to ``4``.
        :type num_workers: int, optional
        :param pin_memory: Whether to pin memory for data loading, defaults to
            ``True``
        :type pin_memory: bool, optional
        """
        super().__init__()
        self.data_dir = data_dir
        self.database = database
        self.batch_size = batch_size
        self.use_graphein = use_graphein
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.transform = transform
        if (
            isinstance(train_split, float)
            and isinstance(val_split, float)
            and isinstance(test_split, float)
        ):
            self._split_data(train_split, val_split, test_split)
            log.info(
                f"Split data into train ({train_split}, {len(self.train_split)}), val ({val_split}, {len(self.val_split)}) and test ({test_split}, {len(self.test_split)}) sets."
            )

        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None):
        self.train_dataset()
        self.val_dataset()
        self.test_dataset()

    def _get_indices(self):
        """Loads the whole database to extract the indices."""
        log.info("Getting indices (loading complete database)...")
        ds = FoldCompDataset(
            self.data_dir,
            self.database,
            ids=None,
            use_graphein=self.use_graphein,
        )
        self.ids = ds.ids
        ds.db.close()

    def _split_data(
        self, train_split: float, val_split: float, test_split: float
    ):
        """Split the database into non-overlapping train, validation and test"""
        if not hasattr(self, "ids"):
            self._get_indices()
            train, test = train_test_split(self.ids, test_size=1 - train_split)

            size = test_split / (test_split + val_split)
            val, test = train_test_split(test, test_size=size)
            self.train_split = train
            self.val_split = val
            self.test_split = test

    def _create_dataset(self, split: str) -> FoldCompDataset:
        """Initialises a FoldCompDataset for a given split.

        :param split: Split to initialise. Must be one of ``train``, ``val``, or
            ``test``.
        :type split: str
        :raises ValueError: If the split is invalid.
        :return: FoldCompDataset for the given split.
        :rtype: FoldCompDataset
        """
        data_split = getattr(self, f"{split}_split")
        return FoldCompDataset(
            self.data_dir,
            self.database,
            ids=data_split,
            use_graphein=self.use_graphein,
            transform=self.transform,
        )

    def train_dataset(self):
        """Initialises the training dataset."""
        log.info("Creating training dataset...")
        self.train_ds = self._create_dataset("train")

    def val_dataset(self):
        """Initialises the validation dataset."""
        log.info("Creating validation dataset...")
        self.val_ds = self._create_dataset("val")

    def test_dataset(self):
        """Initialises the test dataset."""
        log.info("Creating test dataset...")
        self.test_ds = self._create_dataset("test")

    def train_dataloader(self) -> DataLoader:
        """Returns a training dataloader.

        :return: Training dataloader.
        :rtype: DataLoader
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns a validation dataloader.

        :return: Validation dataloader.
        :rtype: DataLoader
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Returns a test dataloader.

        :return: Test dataloader.
        :rtype: DataLoader
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
