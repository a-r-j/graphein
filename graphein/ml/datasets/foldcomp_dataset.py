import asyncio
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from biopandas.pdb import PandasPdb
from loguru import logger as log
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from graphein.protein.tensor import Protein
from graphein.utils.utils import import_message

try:
    import foldcomp
except ImportError:
    message = import_message(
        "graphein.ml.datasets.foldcomp", "foldcomp", None, True
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

DATABASE_TYPES: List[str] = [
    "afdb_swissprot_v4",
    "afdb_uniprot_v4",
    "afdb_rep_dark_v4",
    "highquality_clust30",
    "afdb_rep_v4",
]
"""Currently supported FoldComp databases. See: https://github.com/steineggerlab/foldcomp"""


class FoldCompDataset(Dataset):
    def __init__(
        self,
        root: str,
        database: str,
        ids: Optional[List[str]],
        fraction: float = 1.0,
        use_graphein: bool = True,
    ):
        """Dataset class for FoldComp databases.

        :param root: Directory where the dataset should be saved.
        :type root: str
        :param database: Name of the database. See: :const:`DATABASE_TYPES`.
        :type database: str
        :param ids: List of protein IDs to include in the dataset. If ``None``,
            all proteins are included. Default is ``None``.
        :type ids: Optional[List[str]]
        :param fraction: Fraction of database to use, defaults to ``1.0``.
        :type fraction: float, optional
        :param use_graphein: Whether or not to use Graphein's ``Protein`` objects or
            to use standard PyG ``Data``, defaults to ``True``.
        :type use_graphein: bool, optional
        """
        self.root = Path(root).resolve()
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.database = database
        self.ids = ids
        self.fraction = fraction
        self.use_graphein = use_graphein

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
        super().__init__(root=self.root, transform=None, pre_transform=None)

    @property
    def raw_file_names(self) -> List[str]:
        """Return a list of raw file names expected to be found in the database.

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

            for f in self.raw_file_names:
                shutil.move(f, self.root / f)
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
            accessions = [acc for acc in accessions if acc in tqdm(self.ids)]
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
            self.db = foldcomp.open(self.root / self.database, ids=self.ids)
        else:
            self.db = foldcomp.open(self.root / self.database)

    @staticmethod
    def _parse_dataframe(pdb_string: str) -> pd.DataFrame:
        """Reads a PDB string into a Pandas dataframe."""
        pdb: List[str] = pdb_string.split("\n")
        return PandasPdb().read_pdb_from_list(pdb).df["ATOM"]

    def process_pdb(self, pdb_string: str, name: str) -> Union[Protein, Data]:
        """Process a PDB string into a Graphein Protein object."""
        df = self.parse_dataframe(pdb_string)
        data = Protein().from_dataframe(df, id=name)
        if not self.use_graphein:
            data = data.to_data()
        return data

    def len(self) -> int:
        """Returns length of the dataset"""
        return len(self.protein_to_idx)

    def get(self, idx):
        """Retrieves a protein from the dataset. Can idx on either the protein
        ID or its index."""
        if isinstance(idx, str):
            idx = self.protein_to_idx[idx]
        name, pdb = self.db[idx]

        return self.process_pdb(pdb, name)
