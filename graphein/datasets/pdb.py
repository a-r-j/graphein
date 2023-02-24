import gzip
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import wget
from loguru import logger as log

from graphein.protein.utils import download_pdb_multiprocessing, read_fasta


class PDBManager:
    def __init__(self, root_dir="."):
        self.root_dir = Path(root_dir)
        self.download_metadata()
        self.df = self.parse()

    def download_metadata(self):
        self._download_ligand_map()
        self._download_source_map()
        self._download_exp_type()
        self._download_resolution()
        self._download_pdb_sequences()

    @property
    def num_unique_pdbs(self) -> int:
        """Return the number of unique PDB IDs in the dataset.

        :return: Number of unique PDB IDs.
        :rtype: int
        """
        return len(self.df.pdb.unique())

    @property
    def unique_pdbs(self) -> List[str]:
        """Return a list of unique PDB IDs in the dataset.

        :return: List of unique PDB IDs.
        :rtype: List[str]
        """
        return self.df.pdb.unique().tolist()

    @property
    def longest_chain(self) -> int:
        """Return the length of the longest chain in the dataset.

        :return: Length of the longest chain.
        :rtype: int
        """
        return self.df.length.max()

    @property
    def shortest_chain(self) -> int:
        """Return the length of the shortest chain in the dataset.

        :return: Length of the shortest chain.
        :rtype: int
        """
        return self.df.length.min()

    @property
    def highest_resolution(self) -> float:
        """Return the highest resolution in the dataset.

        :return: Highest resolution.
        :rtype: float
        """
        return self.df.resolution.min()

    @property
    def lowest_resolution(self) -> float:
        """Return the lowest resolution in the dataset.

        :return: Lowest resolution.
        :rtype: float
        """
        return self.df.resolution.max()

    def _download_pdb_sequences(self):
        # Download
        if not os.path.exists(self.root_dir / "pdb_seqres.txt.gz"):
            log.info("Downloading PDB sequences")
            wget.download(
                "https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
            )
            log.info("Downloaded sequences")

        # Unzip
        if not os.path.exists(self.root_dir / "pdb_seqres.txt"):
            log.info("Unzipping PDB sequences")
            with gzip.open(self.root_dir / "pdb_seqres.txt.gz", "rb") as f_in:
                with open(self.root_dir / "pdb_seqres.txt", "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            log.info("Unzipped sequences")

    def _download_ligand_map(self):
        if not os.path.exists(self.root_dir / "cc-to-pdb.tdd"):
            log.info("Downloading ligand map")
            wget.download(
                "http://ligand-expo.rcsb.org/dictionaries/cc-to-pdb.tdd"
            )
            log.info("Downloaded ligand map")

    def _download_source_map(self):
        if not os.path.exists(self.root_dir / "source.idx"):
            log.info("Downloading source map")
            wget.download(
                "https://files.wwpdb.org/pub/pdb/derived_data/index/source.idx"
            )
            log.info("Downloaded source map")

    def _download_exp_type(self):
        # https://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt
        if not os.path.exists(self.root_dir / "pdb_entry_type.txt"):
            log.info("Downloading experiment type map")
            wget.download(
                "https://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt"
            )
            log.info("Downloaded experiment type map")

    def _download_resolution(self):
        # https://files.wwpdb.org/pub/pdb/derived_data/index/resolu.idx
        if not os.path.exists(self.root_dir / "resolu.idx"):
            log.info("Downloading resolution map")
            wget.download(
                "https://files.wwpdb.org/pub/pdb/derived_data/index/resolu.idx"
            )
            log.info("Downloaded resolution map")

    def _parse_resolution(self) -> Dict[str, float]:
        res = {}
        with open(self.root_dir / "resolu.idx") as f:
            for line in f:
                line = line.strip()
                params = line.split()
                if not params or len(params) != 3:
                    continue
                pdb = params[0]
                resolution = params[2]
                try:
                    res[pdb.lower()] = float(resolution)
                except ValueError:
                    continue
        return res

    def _parse_experiment_type(self) -> Dict[str, str]:
        df = pd.read_csv(
            self.root_dir / "pdb_entry_type.txt", sep="\t", header=None
        )
        df.dropna(inplace=True)
        return pd.Series(df[2].values, index=df[0]).to_dict()

    def _parse_source_map(self) -> Dict[str, str]:
        map = {}
        with open("source.idx") as f:
            for line in f:
                line = line.strip()
                params = line.split()
                if params[0] in {
                    "Mon",
                    "Tue",
                    "Wed",
                    "Thu",
                    "Fri",
                    "Sat",
                    "Sun",
                }:
                    continue
                map[params[0].lower()] = " ".join(params[1:])

        del map["protein"]
        del map["idcode"]
        del map["------"]
        return map

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        update: bool = False,
    ) -> pd.DataFrame:
        """Sample a subset of the dataset.

        :param n: Number of proteins to select, defaults to ``None``.
        :type n: Optional[int], optional
        :param frac: Fraction of proteins to select, defaults to ``None``.
        :type frac: Optional[float], optional
        :param replace: Whether or not to sample with replacement, defaults to
            ``False``
        :type replace: bool, optional
        :param update: Whether or not to update the DataFrame in place,
            defaults to ``False``.
        :type update: bool, optional
        :return: DataFrame of sampled proteins.
        :rtype: pd.DataFrame
        """
        df = self.df.sample(n=n, frac=frac, replace=replace)
        if update:
            self.df = df
        return df

    def _parse_ligand_map(self) -> Dict[str, List[str]]:
        map = {}
        with open(self.root_dir / "cc-to-pdb.tdd") as f:
            for line in f:
                line = line.strip()
                params = line.split()
                map[params[0]] = params[1:]
        inv = {}
        for k, v in map.items():
            for x in v:
                inv.setdefault(x, []).append(str(k))
        return inv

    def parse(self) -> pd.DataFrame:
        fasta = read_fasta("pdb_seqres.txt")

        # Iterate over fasta and parse metadata
        records = []
        for k, v in fasta.items():
            seq = v
            params = k.split()
            id = params[0]
            pdb = params[0].split("_")[0]
            chain = params[0].split("_")[1]
            length = int(params[2].split(":")[1])
            molecule_type = params[1].split(":")[1]
            name = " ".join(params[3:])
            record = {
                "id": id,
                "pdb": pdb,
                "chain": chain,
                "length": length,
                "molecule_type": molecule_type,
                "name": name,
                "sequence": seq,
            }
            records.append(record)

        df = pd.DataFrame.from_records(records)
        df["ligands"] = df.pdb.map(self._parse_ligand_map())
        df["ligands"] = df["ligands"].fillna("").apply(list)
        df["source"] = df.pdb.map(self._parse_source_map())
        df["resolution"] = df.pdb.map(self._parse_resolution())
        df["experiment_type"] = df.pdb.map(self._parse_experiment_type())
        return df

    def molecule_type(
        self, type: str = "protein", update: bool = False
    ) -> pd.DataFrame:
        """Select proteins by molecule type. [`protein`, `dna`, `rna`]

        :param type: Typle of molecule, defaults to "protein"
        :type type: str, optional
        :param update: whether to select in place, defaults to False
        :type update: bool, optional
        :return: DataFrame of selected proteins
        :rtype: pd.DataFrame
        """
        df = self.df.loc[self.df.molecule_type == type]

        if update:
            self.df = df
        return df

    def longer_than(self, length: int, update: bool = False) -> pd.DataFrame:
        """Select proteins longer than a given length.

        :param length: Minimum length of protein.
        :type length: int
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``
        :type update: bool, optional
        :return: DataFrame of selected proteins.
        :rtype: pd.DataFrame
        """
        df = self.df.loc[self.df.length > length]

        if update:
            self.df = df
        return df

    def shorter_than(self, length: int, update: bool = False) -> pd.DataFrame:
        """
        Select proteins shorter than a given length.

        :param length: Maximum length of protein.
        :type length: int
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``
        :type update: bool, optional
        :return: DataFrame of selected proteins.
        :rtype: pd.DataFrame
        """
        df = self.df.loc[self.df.length < length]

        if update:
            self.df = df
        return df

    def oligomeric(self, oligomer: int = 1, update: bool = False):
        df = self.df.loc[self.df.length == oligomer]

        if update:
            self.df = df
        return df

    def has_ligand(self, ligand: str, update: bool = False) -> pd.DataFrame:
        """
        Select proteins that have a given ligand.

        :param ligand: Ligand to select. (PDB ligand code)
        :type ligand: str
        :param update: Whether to update the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional
        :return: DataFrame of selected proteins.
        :rtype: pd.DataFrame
        """
        df = self.df.loc[self.df.ligands.map(lambda x: ligand in x)]

        if update:
            self.df = df
        return df

    def has_ligands(
        self, ligands: List[str], inverse: bool = False, update: bool = False
    ):
        """Selects proteins that have all the ligands in the list.

        If inverse is ``True``, selects proteins that do not have all the
        ligands in the list.

        :param ligand: List of ligands. (PDB ligand codes)
        :type ligand: List[str]
        :param inverse: Whether to inverse the selection, defaults to ``False``.
        :type inverse: bool, optional
        :param update: Whether to update the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional
        :return: DataFrame of selected proteins.
        :rtype: pd.DataFrame
        """
        if inverse:
            df = self.df.loc[
                self.df.ligands.map(lambda x: not set(ligands).issubset(x))
            ]
        else:
            df = self.df.loc[
                self.df.ligands.map(lambda x: set(ligands).issubset(x))
            ]

        if update:
            self.df = df
        return df

    def to_dict(self) -> Dict[str, str]:
        """Returns a dictionary of sequences indexed by chain.

        :return: Dictionary of sequences.
        :rtype: Dict[str, str]
        """
        return (
            self.df[["id", "sequence"]].set_index("id").to_dict()["sequence"]
        )

    def to_fasta(self, filename: str):
        """Write the dataset to a FASTA file (indexed by chain id).

        :param filename: Name of the output FASTA file.
        :type filename: str
        """
        with open(filename, "w") as f:
            for k, v in self.to_dict().items():
                f.write(f">{k}\n")
                f.write(f"{v}\n")

    def standard_alphabet(self, update: bool = False):
        """
        Removes sequences with non-standard amino acids.

        :param update: Update the DataFrame in place
        :type update: bool, optional
        :returns: DataFrame
        :rtype: pd.DataFrame
        """
        df = self.df.loc[
            self.df.sequence.map(
                lambda x: set(x).issubset(set("ACDEFGHIKLMNPQRSTVWY"))
            )
        ]
        if update:
            self.df = df
        return df

    def download(
        self,
        out_dir=".",
        overwrite: bool = False,
        max_workers: int = 8,
        chunksize: int = 32,
    ):
        """Downloads PDB files in current selection.

        :param out_dir: Output directory, defaults to ""
        :type out_dir: str, optional
        :param overwrite: Overwrite existing files, defaults to False
        :type overwrite: bool, optional
        :param max_workers: Number of processes to use, defaults to 8
        :type max_workers: int, optional
        :param chunksize: Chunk size for each worker, defaults to 32
        :type chunksize: int, optional
        """
        log.info(f"Downloading {len(self.unique_pdbs)} PDB files...")
        download_pdb_multiprocessing(
            self.unique_pdbs,
            out_dir,
            overwrite=overwrite,
            max_workers=max_workers,
            chunksize=chunksize,
        )
