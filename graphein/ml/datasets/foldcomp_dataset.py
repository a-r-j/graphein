"""Utilities for loading FoldComp databases for deep learning."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import asyncio
import os
import random
import shutil
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
)

import numpy as np
import torch
from loguru import logger as log
from sklearn.model_selection import train_test_split
from torch_geometric import transforms as T
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graphein.protein.resi_atoms import (
    ATOM_NUMBERING,
    STANDARD_AMINO_ACID_MAPPING_1_TO_3,
    STANDARD_AMINO_ACIDS,
)
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
    "afdb_swissprot_v4",  # AlphaFoldDB Swiss-Prot
    "afdb_uniprot_v4",  # AlphaFoldDB Uniprot
    "afdb_rep_v4",  # AlphaFoldDB Cluster Representatives
    "afdb_rep_dark_v4",  # AlphaFoldDB Cluster Representatives (Dark Clusters)
    "esmatlas",  # ESMAtlas full (v0 + v2023_02)
    "esmatlas_v2023_02",  # ESMAtlas v2023_02
    "highquality_clust30",  # ESMAtlas high-quality
]
"""
Currently supported FoldComp databases. See:
https://github.com/steineggerlab/foldcomp
"""


GraphTransform = Callable[[Union[Data, Protein]], Union[Data, Protein]]


ATOM_MAP = {
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
    ],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "GLY": ["N", "CA", "C", "O"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
}


class FoldCompDataset(Dataset):
    def __init__(
        self,
        root: str,
        database: Literal[
            "afdb_swissprot_v4",
            "afdb_uniprot_v4",
            "afdb_rep_v4",
            "afdb_rep_dark_v4",
            "esmatlas",
            "esmatlas_v2023_02",
            "highquality_clust30",
        ],
        ids: Optional[List[str]] = None,
        exclude_ids: Optional[List[str]] = None,
        fraction: float = 1.0,
        use_graphein: bool = True,
        transform: Optional[T.BaseTransform] = None,
    ):
        """Dataset class for FoldComp databases.

        :param root: Directory where the dataset should be saved.
        :type root: str
        :param database: Name of the database. See:
            :const:`FOLDCOMP_DATABASE_TYPES`.
        :type database: Literal
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

        self._database_files = [
            f"{self.database}",
            f"{self.database}.dbtype",
            f"{self.database}.index",
            f"{self.database}.lookup",
            f"{self.database}.source",
        ]
        self._get_indices()

        super().__init__(
            root=self.root,
            transform=self.transform,
            pre_transform=None,  # type: ignore
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

        if not all(
            os.path.exists(self.root / f) for f in self._database_files
        ):
            log.info(f"Downloading FoldComp dataset {self.database}...")
            curr_dir = os.getcwd()
            os.chdir(self.root)
            try:
                foldcomp.setup(self.database)
            except RuntimeError:
                _ = self.async_setup()
                asyncio.run(_)
            os.chdir(curr_dir)
            log.info("Download complete.")
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
            self.db = foldcomp.open(
                self.root / self.database, ids=self.ids, decompress=False
            )  # type: ignore
        else:
            self.db = foldcomp.open(self.root / self.database, decompress=False)  # type: ignore

    @staticmethod
    def fc_to_pyg(data: Dict[str, Any], name: Optional[str] = None) -> Protein:
        # Map sequence to 3-letter codes
        res = [STANDARD_AMINO_ACID_MAPPING_1_TO_3[r] for r in data["residues"]]
        residue_type = torch.tensor(
            [STANDARD_AMINO_ACIDS.index(res) for res in data["residues"]],
        )
        n_res = len(res)

        # Get residue numbers
        res_num = np.arange(n_res)

        # Get list of atom types
        atom_types = []
        atom_counts = []
        for r in res:
            atom_types += ATOM_MAP[r]
            atom_counts.append(len(ATOM_MAP[r]))
        atom_types += ["OXT"]
        atom_counts[-1] += 1

        # Get atom indices
        atom_idx = np.array([ATOM_NUMBERING[atm] for atm in atom_types])

        # Initialize coordinates
        coords = np.ones((n_res, 37, 3)) * 1e-5

        res_idx = np.repeat(res_num, atom_counts)
        coords[res_idx, atom_idx, :] = np.array(data["coordinates"])
        b_factor = np.array(data["b_factors"]) / 100

        return Protein(
            coords=torch.from_numpy(coords).float(),
            residues=res,
            residue_id=[f"A:{m}:{str(n)}" for m, n in zip(res, res_num)],
            chains=torch.zeros(n_res),
            residue_type=residue_type.long(),
            b_factor=torch.from_numpy(b_factor).float(),
            id=name,
            x=torch.zeros(n_res),
            seq_pos=torch.from_numpy(res_num).unsqueeze(-1),
        )

    def len(self) -> int:
        """Returns length of the dataset"""
        return len(self.protein_to_idx)

    def get(self, idx) -> Union[Data, Protein]:
        """Retrieves a protein from the dataset. Can idx on either the protein
        ID or its index."""
        if isinstance(idx, str):
            idx = self.protein_to_idx[idx]

        name = self.idx_to_protein[idx]
        data = foldcomp.get_data(self.db[idx])  # type: ignore
        return self.fc_to_pyg(data, name)


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
        transform: Optional[Iterable[Callable]] = None,
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
        :type transform: Optional[Iterable[Callable]]
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
        self.transform = (
            self._compose_transforms(transform)
            if transform is not None
            else None
        )

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

    def _compose_transforms(self, transforms: Iterable[Callable]) -> T.Compose:
        try:
            return T.Compose(list(transforms.values()))
        except Exception:
            return T.Compose(transforms)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset()
        elif stage == "validate":
            self.val_dataset()
        elif stage == "test":
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
