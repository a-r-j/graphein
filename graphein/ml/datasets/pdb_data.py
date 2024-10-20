import gzip
import math
import os
import shutil
import subprocess
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import wget
from biopandas.pdb import PandasPdb
from loguru import logger as log
from pandas.core.groupby.generic import DataFrameGroupBy
from tqdm import tqdm

from graphein.protein.utils import (
    cast_pdb_column_to_type,
    download_pdb_multiprocessing,
    extract_chains_to_file,
    read_fasta,
)
from graphein.utils.dependencies import is_tool


class PDBManager:
    """A utility for creating selections of experimental PDB structures."""

    def __init__(
        self,
        root_dir: str = ".",
        structure_format: str = "pdb",
        splits: Optional[List[str]] = None,
        split_ratios: Optional[List[float]] = None,
        split_time_frames: Optional[List[np.datetime64]] = None,
        assign_leftover_rows_to_split_n: int = 0,
        labels: Optional[
            List[Literal["uniprot_id", "cath_code", "ec_number"]]
        ] = None,
    ):
        """Instantiate a selection of experimental PDB structures.

        :param root_dir: The directory in which to store all PDB entries,
            defaults to ``"."``.
        :type root_dir: str, optional
        :param structure_format: Whether to use ``.pdb``, ``.mmtf`` or ``mmcif`` file.
            Defaults to ``"pdb"``.
        :type structure_format: str, optional
        :param splits: A list of names corresponding to each dataset split,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param split_ratios: Proportions into which to split the current
            selection of PDB entries, defaults to ``None``.
        :type split_ratios: Optional[List[float]], optional
        :param split_time_frames: Time periods into which to segment the current
            selection of PDB entries, defaults to ``None``.
        :type split_time_frames: Optional[List[np.datetime64]], optional
        :param assign_leftover_rows_to_split_n: Index of the split to which
            to assign any rows remaining after creation of new dataset splits,
            defaults to ``0``.
        :type assign_leftover_rows_to_split_n: int, optional
        :param labels: A list of names corresponding to metadata labels that should be included in PDB manager dataframe,
            defaults to ``None``.
        :type labels: Optional[List[Literal["uniprot_id", "cath_code", "ec_number"]]], optional
        """
        # Arguments
        self.root_dir = Path(root_dir)

        # Constants
        self.pdb_sequences_url = (
            "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
        )
        self.ligand_map_url = (
            "http://ligand-expo.rcsb.org/dictionaries/cc-to-pdb.tdd"
        )
        self.source_map_url = (
            "https://files.wwpdb.org/pub/pdb/derived_data/index/source.idx"
        )
        self.resolution_url = (
            "https://files.wwpdb.org/pub/pdb/derived_data/index/resolu.idx"
        )
        self.pdb_entry_type_url = (
            "https://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt"
        )
        self.pdb_deposition_date_url = (
            "https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx"
        )
        self.pdb_availability_url = "https://files.wwpdb.org/pub/pdb/compatible/pdb_bundle/pdb_bundle_index.txt"

        self.pdb_chain_cath_uniprot_url = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_cath_uniprot.tsv.gz"

        self.cath_id_cath_code_url = "http://download.cathdb.info/cath/releases/daily-release/newest/cath-b-newest-all.gz"

        self.pdb_chain_ec_number_url = "https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_enzyme.tsv.gz"

        self.pdb_dir = self.root_dir / "pdb"
        if not os.path.exists(self.pdb_dir):
            os.makedirs(self.pdb_dir)

        self.structure_format = structure_format

        self.pdb_seqres_archive_filename = Path(self.pdb_sequences_url).name
        self.pdb_seqres_filename = Path(self.pdb_seqres_archive_filename).stem
        self.ligand_map_filename = Path(self.ligand_map_url).name
        self.source_map_filename = Path(self.source_map_url).name
        self.resolution_filename = Path(self.resolution_url).name
        self.pdb_entry_type_filename = Path(self.pdb_entry_type_url).name
        self.pdb_deposition_date_filename = Path(
            self.pdb_deposition_date_url
        ).name
        self.pdb_availability_filename = Path(self.pdb_availability_url).name
        self.pdb_chain_cath_uniprot_filename = Path(
            self.pdb_chain_cath_uniprot_url
        ).name
        self.cath_id_cath_code_filename = Path(self.cath_id_cath_code_url).name
        self.pdb_chain_ec_number_filename = Path(
            self.pdb_chain_ec_number_url
        ).name

        self.list_columns = ["ligands"]

        # Data
        self.download_metadata()
        self.df = self.parse(labels)
        self.source = self.df.copy()

        # Splits
        self.splits_provided = splits is not None
        if self.splits_provided:
            assert len(set(splits)) == len(
                splits
            ), f"Split names must be unique: {splits}."
            self.splits = splits
            self.df_splits = {split: None for split in splits}
            self.assign_leftover_rows_to_split_n = (
                assign_leftover_rows_to_split_n
            )
            # Sequence-based ratio splits
            if split_ratios is not None:
                assert len(splits) == len(
                    split_ratios
                ), f"Number of splits ({splits}) must match number of split ratios ({split_ratios})."
                assert math.isclose(
                    sum(split_ratios), 1.0
                ), f"Split ratios must sum to 1.0: {split_ratios}."
                self.split_ratios = split_ratios
            # Time-based splits
            if split_time_frames is not None:
                assert len(splits) == len(
                    split_time_frames
                ), f"Number of splits ({splits}) must match number of split time frames ({split_time_frames})."
                assert self._frames_are_sequential(
                    split_time_frames
                ), f"Split time frames must be sequential: {split_time_frames}."
                self.split_time_frames = split_time_frames

    def download_metadata(self):
        """Download all PDB metadata."""
        self._download_pdb_sequences()
        self._download_ligand_map()
        self._download_source_map()
        self._download_resolution()
        self._download_entry_metadata()
        self._download_exp_type()
        self._download_pdb_availability()
        self._download_pdb_chain_cath_uniprot_map()
        self._download_cath_id_cath_code_map()
        self._download_pdb_chain_ec_number_map()

    def get_unavailable_pdb_files(
        self, splits: Optional[List[str]] = None
    ) -> List[str]:
        """Returns a list of PDB files unavailable for download.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :return: List of PDB IDs.
        :rtype: List[str]
        """
        splits_df = self.get_splits(splits)
        return splits_df.loc[
            splits_df.pdb_file_available == False, "pdb"
        ].tolist()

    def get_num_unique_pdbs(self, splits: Optional[List[str]] = None) -> int:
        """Return the number of unique PDB IDs in the dataset.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: Number of unique PDB IDs.
        :rtype: int
        """
        splits_df = self.get_splits(splits)
        return len(splits_df.pdb.unique())

    def get_unique_pdbs(self, splits: Optional[List[str]] = None) -> List[str]:
        """Return a list of unique PDB IDs in the dataset.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: List of unique PDB IDs.
        :rtype: List[str]
        """
        splits_df = self.get_splits(splits)
        return splits_df.pdb.unique().tolist()

    def get_num_chains(self, splits: Optional[List[str]] = None) -> int:
        """Return the number of chains in the dataset.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: Number of chains.
        :rtype: int
        """
        splits_df = self.get_splits(splits)
        return len(splits_df)

    def get_longest_chain(self, splits: Optional[List[str]] = None) -> int:
        """Return the length of the longest chain in the dataset.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: Length of the longest chain.
        :rtype: int
        """
        splits_df = self.get_splits(splits)
        return splits_df.length.max()

    def get_shortest_chain(self, splits: Optional[List[str]] = None) -> int:
        """Return the length of the shortest chain in the dataset.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: Length of the shortest chain.
        :rtype: int
        """
        splits_df = self.get_splits(splits)
        return splits_df.length.min()

    def get_best_resolution(self, splits: Optional[List[str]] = None) -> float:
        """Return the best resolution in the dataset.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: Best resolution.
        :rtype: float
        """
        splits_df = self.get_splits(splits)
        return splits_df.resolution.min()

    def get_worst_resolution(
        self, splits: Optional[List[str]] = None
    ) -> float:
        """Return the worst resolution in the dataset.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: Worst resolution.
        :rtype: float
        """
        splits_df = self.get_splits(splits)
        return splits_df.resolution.max()

    def get_experiment_types(
        self, splits: Optional[List[str]] = None
    ) -> List[str]:
        """Return list of different experiment types in the dataset.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: List of experiment types.
        :rtype: List[str]
        """
        splits_df = self.get_splits(splits)
        return splits_df.experiment_type.unique()

    def get_molecule_types(
        self, splits: Optional[List[str]] = None
    ) -> List[str]:
        """Return list of different molecule types in the dataset.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: List of molecule types.
        :rtype: List[str]
        """
        splits_df = self.get_splits(splits)
        return splits_df.molecule_type.unique()

    def get_molecule_names(
        self, splits: Optional[List[str]] = None
    ) -> List[str]:
        """Return list of molecule names in the dataset.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: List of molecule names.
        :rtype: List[str]
        """
        splits_df = self.get_splits(splits)
        return splits_df.name.unique()

    def _frames_are_sequential(
        self, split_time_frames: List[np.datetime64]
    ) -> bool:
        """Check if all provided frames are sequentially ordered.

        :param split_time_frames: Time frames into which to split
            selected PDB entries.
        :type split_time_frames: List[np.datetime64]

        :return: Whether all provided frames are sequentially ordered.
        :rtype: bool
        """
        frames_are_sequential = True
        last_frame_index = len(split_time_frames) - 1
        for frame_index in range(len(split_time_frames)):
            frame = split_time_frames[frame_index]
            frames_are_backwards_sequential = frame_index == 0 or (
                frame_index > 0 and frame > split_time_frames[frame_index - 1]
            )
            frames_are_forwards_sequential = (
                frame_index < last_frame_index
                and frame < split_time_frames[frame_index + 1]
            ) or frame_index == last_frame_index
            frames_are_sequential = all(
                [
                    frames_are_backwards_sequential,
                    frames_are_forwards_sequential,
                ]
            )
        return frames_are_sequential

    def _download_pdb_sequences(self):
        """Download PDB sequences from
        https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz.
        """
        if not os.path.exists(
            self.root_dir / self.pdb_seqres_archive_filename
        ):
            log.info("Downloading PDB sequences...")
            wget.download(self.pdb_sequences_url, out=str(self.root_dir))
            log.debug("Downloaded sequences")

        # Unzip all collected sequences
        if not os.path.exists(self.root_dir / self.pdb_seqres_filename):
            log.info("Unzipping PDB sequences...")
            with gzip.open(
                self.root_dir / self.pdb_seqres_archive_filename, "rb"
            ) as f_in:
                with open(
                    self.root_dir / self.pdb_seqres_filename, "wb"
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            log.debug("Unzipped sequences")

    def _download_ligand_map(self):
        """Download ligand map from
        http://ligand-expo.rcsb.org/dictionaries/cc-to-pdb.tdd.
        """
        if not os.path.exists(self.root_dir / self.ligand_map_filename):
            log.info("Downloading ligand map...")
            wget.download(self.ligand_map_url, out=str(self.root_dir))
            log.debug("Downloaded ligand map")

    def _download_source_map(self):
        """Download source map from
        https://files.wwpdb.org/pub/pdb/derived_data/index/source.idx.
        """
        if not os.path.exists(self.root_dir / self.source_map_filename):
            log.info("Downloading source map...")
            wget.download(self.source_map_url, out=str(self.root_dir))
            log.debug("Downloaded source map")

    def _download_resolution(self):
        """Download source map from
        https://files.wwpdb.org/pub/pdb/derived_data/index/resolu.idx.
        """
        if not os.path.exists(self.root_dir / self.resolution_filename):
            log.info("Downloading resolution map...")
            wget.download(self.resolution_url, out=str(self.root_dir))
            log.debug("Downloaded resolution map")

    def _download_entry_metadata(self):
        """Download PDB entry metadata from
        https://files.wwpdb.org/pub/pdb/derived_data/index/entries.idx.
        """
        if not os.path.exists(
            self.root_dir / self.pdb_deposition_date_filename
        ):
            log.info("Downloading entry metadata...")
            wget.download(self.pdb_deposition_date_url, out=str(self.root_dir))
            log.debug("Downloaded entry metadata")

    def _download_exp_type(self):
        """Download PDB experiment metadata from
        https://files.wwpdb.org/pub/pdb/derived_data/pdb_entry_type.txt.
        """
        if not os.path.exists(self.root_dir / self.pdb_entry_type_filename):
            log.info("Downloading experiment type map...")
            wget.download(self.pdb_entry_type_url, out=str(self.root_dir))
            log.debug("Downloaded experiment type map")

    def _download_pdb_availability(self):
        """Download PDB availability metadata from
        https://files.wwpdb.org/pub/pdb/compatible/pdb_bundle/pdb_bundle_index.txt
        """
        if not os.path.exists(self.root_dir / self.pdb_availability_filename):
            log.info("Downloading PDB availability map...")
            wget.download(self.pdb_availability_url, out=str(self.root_dir))
            log.debug("Downloaded PDB availability map")

    def _download_pdb_chain_cath_uniprot_map(self):
        """Download mapping from PDB chain to uniprot accession and CATH ID from
        https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_cath_uniprot.tsv.gz
        """
        if not os.path.exists(
            self.root_dir / self.pdb_chain_cath_uniprot_filename
        ):
            log.info("Downloading Uniprot CATH map...")
            wget.download(
                self.pdb_chain_cath_uniprot_url, out=str(self.root_dir)
            )
            log.debug("Downloaded Uniprot CATH map")

    def _download_cath_id_cath_code_map(self):
        """Download mapping from CATH IDs to CATH code from
        http://download.cathdb.info/cath/releases/daily-release/newest/cath-b-newest-all.gz
        """
        if not os.path.exists(self.root_dir / self.cath_id_cath_code_filename):
            log.info("Downloading CATH ID to CATH code map...")
            wget.download(self.cath_id_cath_code_url, out=str(self.root_dir))
            log.debug("Downloaded CATH ID to CATH code map")

    def _download_pdb_chain_ec_number_map(self):
        """Download mapping from PDB chains to EC number from
        https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/tsv/pdb_chain_enzyme.tsv.gz
        """
        if not os.path.exists(
            self.root_dir / self.pdb_chain_ec_number_filename
        ):
            log.info("Downloading EC number map...")
            wget.download(self.pdb_chain_ec_number_url, out=str(self.root_dir))
            log.debug("Downloaded EC number map")

    def _parse_ligand_map(self) -> Dict[str, List[str]]:
        """Parse the ligand maps for all PDB records.

        :return: Dictionary of PDB entries with their
            corresponding ligand map values.
        :rtype: Dict[str, List[str]]
        """
        ligand_map = {}
        with open(self.root_dir / self.ligand_map_filename) as f:
            for line in f:
                line = line.strip()
                params = line.split()
                ligand_map[params[0]] = params[1:]
        inv = {}
        for k, v in ligand_map.items():
            for x in v:
                inv.setdefault(x, []).append(str(k))
        return inv

    def _parse_source_map(self) -> Dict[str, str]:
        """Parse the source maps for all PDB records.

        :return: Dictionary of PDB entries with their
            corresponding source map values.
        :rtype: Dict[str, str]
        """
        source_map = {}
        with open(self.root_dir / self.source_map_filename) as f:
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
                source_map[params[0].lower()] = " ".join(params[1:])

        del source_map["protein"]
        del source_map["idcode"]
        del source_map["------"]
        return source_map

    def _parse_resolution(self) -> Dict[str, float]:
        """Parse the PDB resolutions for all PDB records.

        :return: Dictionary of PDB resolutions with their
            corresponding values.
        :rtype: Dict[str, float]
        """
        res = {}
        with open(self.root_dir / self.resolution_filename) as f:
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

    def _parse_entries(self) -> Dict[str, datetime]:
        with open(self.root_dir / self.pdb_deposition_date_filename, "r") as f:
            lines = f.readlines()
        lines = lines[2:]  # Skip header
        # Note: There's a badly formatted line we need to deal with instead of
        # using Pandas' builtin CSV parser.
        lines = [l.replace('"', "") for l in lines]

        df = pd.read_csv(
            StringIO("".join(lines)),
            sep="\t",
            header=None,
            skipinitialspace=True,
        )
        df.columns = [
            "id",
            "name",
            "date",
            "title",
            "source",
            "authors",
            "resolution",
            "experiment_type",
        ]
        df.dropna(subset=["id"], inplace=True)

        df.id = df.id.str.lower()
        df.date = pd.to_datetime(df.date, format="%m/%d/%y")
        return pd.Series(df["date"].values, index=df["id"]).to_dict()

    def _parse_experiment_type(self) -> Dict[str, str]:
        """Parse the experiment types for all PDB records.

        :return: Dictionary of PDB entries with their
            corresponding experiment types.
        :rtype: Dict[str, str]
        """
        df = pd.read_csv(
            self.root_dir / self.pdb_entry_type_filename, sep="\t", header=None
        )
        df.dropna(inplace=True)
        return pd.Series(df[2].values, index=df[0]).to_dict()

    def _parse_pdb_availability(self) -> Dict[str, bool]:
        """Parse the PDB file availability for all PDB records.

        :return: List of available PDB files for download from the RCSB PDB.
        :rtype: List[str]
        """
        with open(self.root_dir / self.pdb_availability_filename, "r") as f:
            ids = f.read().splitlines()
        log.info(f"Found {len(ids)} PDB files unavailable for download.")
        ids = {id: False for id in ids}
        return ids

    def _parse_uniprot_id(self) -> Dict[str, str]:
        """Parse the uniprot ID for all PDB chains.

        :return: Dictionary of PDB chain ID with their
            corresponding uniprot ID.
        :rtype: Dict[str, str]
        """
        uniprot_mapping = {}
        with gzip.open(
            self.root_dir / self.pdb_chain_cath_uniprot_filename, "rt"
        ) as f:
            for line in f:
                try:
                    pdb, chain, uniprot_id, cath_id = line.strip().split("\t")
                    key = f"{pdb}_{chain}"
                    uniprot_mapping[key] = uniprot_id
                except ValueError:
                    continue
        return uniprot_mapping

    def _parse_cath_id(self) -> Dict[str, str]:
        """Parse the CATH ID for all PDB chains.

        :return: Dictionary of PDB chain ID with their
            corresponding CATH ID.
        :rtype: Dict[str, str]
        """
        cath_mapping = {}
        with gzip.open(
            self.root_dir / self.pdb_chain_cath_uniprot_filename, "rt"
        ) as f:
            next(f)  # Skip header line
            for line in f:
                try:
                    pdb, chain, uniprot_id, cath_id = line.strip().split("\t")
                    key = f"{pdb}_{chain}"
                    cath_mapping[key] = cath_id
                except ValueError:
                    continue
        return cath_mapping

    def _parse_cath_code(self) -> Dict[str, str]:
        """Parse the CATH code for all CATH IDs.

        :return: Dictionary of CATH ID with their
            corresponding CATH code.
        :rtype: Dict[str, str]
        """
        cath_mapping = {}
        with gzip.open(
            self.root_dir / self.cath_id_cath_code_filename, "rt"
        ) as f:
            print(f)
            for line in f:
                print(line)
                try:
                    cath_id, cath_version, cath_code, cath_segment = (
                        line.strip().split()
                    )
                    cath_mapping[cath_id] = cath_code
                    print(cath_id, cath_code)
                except ValueError:
                    continue
        return cath_mapping

    def _parse_ec_number(self) -> Dict[str, str]:
        """Parse the CATH ID for all PDB chains and adds None when no EC number is present.

        :return: Dictionary of PDB chain ID with their
            corresponding EC number.
        :rtype: Dict[str, str]
        """
        ec_mapping = {}
        with gzip.open(
            self.root_dir / self.pdb_chain_ec_number_filename, "rt"
        ) as f:
            next(f)  # Skip header line
            for line in f:
                try:
                    pdb, chain, uniprot_id, ec_number = line.strip().split(
                        "\t"
                    )
                    key = f"{pdb}_{chain}"
                    ec_number = None if ec_number == "?" else ec_number
                    ec_mapping[key] = ec_number
                except ValueError:
                    continue
        return ec_mapping

    def parse(
        self,
        labels: Optional[
            List[Literal["uniprot_id", "cath_code", "ec_number"]]
        ] = None,
    ) -> pd.DataFrame:
        """Parse all PDB sequence records.

        :param labels: A list of names corresponding to metadata labels that should be included in PDB manager dataframe,
            defaults to ``None``.
        :type labels: Optional[List[str]], optional

        :return: DataFrame containing PDB sequence entries
            with their corresponding metadata.
        :rtype: pd.DataFrame
        """
        fasta = read_fasta(self.root_dir / self.pdb_seqres_filename)

        # Iterate over FASTA and parse metadata
        records = []
        for k, v in fasta.items():
            seq = v
            params = k.split()
            pdb_id = params[0]
            pdb = params[0].split("_")[0]
            chain = params[0].split("_")[1]
            length = int(params[2].split(":")[1])
            molecule_type = params[1].split(":")[1]
            name = " ".join(params[3:])
            split = "N/A"  # Assign rows to the null split
            record = {
                "id": pdb_id,
                "pdb": pdb,
                "chain": chain,
                "length": length,
                "molecule_type": molecule_type,
                "name": name,
                "sequence": seq,
                "split": split,
            }
            records.append(record)

        df = pd.DataFrame.from_records(records)
        df["n_chains"] = df.groupby("pdb")["pdb"].transform("count")
        df["ligands"] = df.pdb.map(self._parse_ligand_map())
        df["ligands"] = df["ligands"].fillna("").apply(list)
        df["source"] = df.pdb.map(self._parse_source_map())
        df["resolution"] = df.pdb.map(self._parse_resolution())
        df["deposition_date"] = df.pdb.map(self._parse_entries())
        df["experiment_type"] = df.pdb.map(self._parse_experiment_type())
        df["pdb_file_available"] = df.pdb.map(self._parse_pdb_availability())
        df["pdb_file_available"] = df["pdb_file_available"].fillna(True)
        if labels:
            if "uniprot_id" in labels:
                df["uniprot_id"] = df.id.map(self._parse_uniprot_id())
            if "cath_code" in labels:
                df["cath_id"] = df.id.map(self._parse_cath_id())
                df["cath_code"] = df.cath_id.map(self._parse_cath_code())
            if "ec_number" in labels:
                df["ec_number"] = df.id.map(self._parse_ec_number())

        return df

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        splits: Optional[List[str]] = None,
        replace: bool = False,
        update: bool = False,
    ) -> pd.DataFrame:
        """Sample a subset of the dataset.

        :param n: Number of molecules to select, defaults to ``None``.
        :type n: Optional[int], optional
        :param frac: Fraction of molecules to select, defaults to ``None``.
        :type frac: Optional[float], optional
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param replace: Whether or not to sample with replacement, defaults to
            ``False``.
        :type replace: bool, optional
        :param update: Whether or not to update the DataFrame in place,
            defaults to ``False``.
        :type update: bool, optional

        :return: DataFrame of sampled molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.sample(n=n, frac=frac, replace=replace)

        if update:
            self.df = df
        return df

    def get_splits(
        self,
        splits: Optional[List[str]] = None,
        df_splits: Optional[Dict[str, pd.DataFrame]] = None,
        source: bool = False,
    ) -> pd.DataFrame:
        """Return DataFrame entries belonging to the splits given.

        :param split: Names of splits from which to select entries,
            defaults to ``None``.
        :type split: Optional[List[str]], optional
        :param split: Dictionary of split names mapping to split DataFrames,
            defaults to ``None``.
        :type split: Optional[List[str]], optional
        :param source: Whether to filter based on the source DataFrame,
            defaults to ``False``.
        :type source: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        df = self.source if source else self.df
        if splits is not None and len(splits) > 0:
            df_splits = df_splits if df_splits is not None else self.df_splits
            all_splits_df = pd.concat([df_splits[split] for split in splits])
            assert len(all_splits_df) > 0, "Combined splits must be non-empty."
        splits_df = (
            all_splits_df.loc[all_splits_df.split.isin(splits)]
            if splits is not None
            else df
        )
        assert len(splits_df) > 0, "Requested splits must be non-empty."
        return splits_df

    def remove_unavailable_pdbs(
        self, splits: Optional[List[str]] = None, update: bool = False
    ) -> pd.DataFrame:
        """
        Removes PDB files that are not available for download from the
        selection.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional
        :return: DataFrame of selected molecules available for download in PDB
            format.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.loc[splits_df.pdb_file_available == True]
        if update:
            self.df = df
        return df

    def molecule_type(
        self,
        type: str = "protein",
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """Select molecules by molecule type. [`protein`, `dna`, `rna`]

        :param type: Type of molecule, defaults to "protein".
        :type type: str, optional
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.loc[splits_df.molecule_type == type]

        if update:
            self.df = df
        return df

    def experiment_type(
        self,
        type: str = "diffraction",
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """
        Select molecules by experiment type:
        [``diffraction``, ``NMR``, ``EM``, ``other``]

        :param type: Experiment type of molecule, defaults to "diffraction".
        :type type: str, optional
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.loc[splits_df.experiment_type == type]

        if update:
            self.df = df
        return df

    def experiment_types(
        self,
        types: List[str] = ["diffraction"],
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """
        Select molecules by experiment types:
        [``diffraction``, ``NMR``, ``EM``, ``other``]

        :param types: Experiment types of molecules, defaults to "diffraction".
        :type types: List[str], optional
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.loc[splits_df.experiment_type.isin(types)]

        if update:
            self.df = df
        return df

    def name(
        self,
        substrings: List[str],
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """
        Select molecules by substrings present in their names:
        e.g., [``DNA``, ``RNA``]

        :param substrings: Substrings to be found within the name field of each molecule.
        :type type: str, optional
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.loc[splits_df.name.str.contains("|".join(substrings))]

        if update:
            self.df = df
        return df

    def compare_length(
        self,
        length: int,
        comparison: str = "equal",
        compare_pdb_groups: bool = False,
        splits: Optional[List[str]] = None,
        update: bool = False,
    ):
        """Select molecules with a given length.

        :param length: Length of molecule.
        :type length: int
        :param comparison: Comparison operator. One of ``"equal"``,
            ``"less"``, or ``"greater"``, defaults to ``"equal"``.
        :type comparison: str, optional
        :param compare_pdb_groups: Whether to group chains by PDB codes
            to track a PDB code's number of associated chains.
        :type compare_pdb_groups: bool, optional
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)

        if comparison == "equal":
            df = (
                splits_df[
                    splits_df.groupby("pdb")["pdb"].transform("size") == length
                ]
                if compare_pdb_groups
                else splits_df[splits_df.length == length]
            )
        elif comparison == "less":
            df = (
                splits_df[
                    splits_df.groupby("pdb")["pdb"].transform("size") < length
                ]
                if compare_pdb_groups
                else splits_df[splits_df.length < length]
            )
        elif comparison == "greater":
            df = (
                splits_df[
                    splits_df.groupby("pdb")["pdb"].transform("size") > length
                ]
                if compare_pdb_groups
                else splits_df[splits_df.length > length]
            )
        else:
            raise ValueError(
                "Comparison must be one of 'equal', 'less', or 'greater'."
            )

        if update:
            self.df = df
        return df

    def length_longer_than(
        self,
        length: int,
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """Select molecules longer than a given length.

        :param length: Minimum length of molecule.
        :type length: int
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        return self.compare_length(length, "greater", False, splits, update)

    def length_shorter_than(
        self,
        length: int,
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """
        Select molecules shorter than a given length.

        :param length: Maximum length of molecule.
        :type length: int
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        return self.compare_length(length, "less", False, splits, update)

    def length_equal_to(
        self,
        length: int,
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """Select molecules equal to a given length.

        :param length: Exact length of molecule.
        :type length: int
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        return self.compare_length(length, "equal", False, splits, update)

    def oligomeric(
        self,
        oligomer: int = 1,
        comparison: str = "equal",
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """Select molecules with a given oligmeric length.
        I.e. ``df.n_chains ==/ =< / >=  oligomer``

        N.b. the `comparison` arguments for `"greater"` and `"less"` are
        `>=` and `=<` respectively.

        :param length: Oligomeric length of molecule, defaults to ``1``.
        :type length: int
        :param comparison: Comparison operator. One of ``"equal"``,
            ``"less"``, or ``"greater"``, defaults to ``"equal"``.
        :type comparison: str, optional
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected oligmers.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        if comparison == "equal":
            df = splits_df.loc[splits_df.n_chains == oligomer]
        elif comparison == "less":
            df = splits_df.loc[splits_df.n_chains <= oligomer]
        elif comparison == "greater":
            df = splits_df.loc[splits_df.n_chains >= oligomer]
        else:
            raise ValueError(
                "Comparison must be one of 'equal', 'less', or 'greater'."
            )
        if update:
            self.df = df
        return df

    def resolution_better_than_or_equal_to(
        self,
        resolution: float,
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """
        Select molecules with a resolution better than or equal to the given
        value.

        Conventions for PDB resolution values are used, where a lower
        resolution value indicates a better resolution for a molecule overall.

        :param resolution: Worst molecule resolution allowed.
        :type resolution: int
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.loc[splits_df.resolution <= resolution]

        if update:
            self.df = df
        return df

    def resolution_worse_than_or_equal_to(
        self,
        resolution: int,
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """
        Select molecules with a resolution worse than or equal to the
        given value.

        Conventions for PDB resolution values are used, where a higher
        resolution value indicates a worse resolution for a molecule overall.

        :param resolution: Best molecule resolution allowed.
        :type resolution: int
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.loc[splits_df.resolution >= resolution]

        if update:
            self.df = df
        return df

    def has_ligand(
        self,
        ligand: str,
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """
        Select molecules that contain a given ligand.

        :param ligand: Ligand to select.
            (PDB ligand code - http://ligand-expo.rcsb.org/)
        :type ligand: str
        :param update: Whether to update the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.loc[splits_df.ligands.map(lambda x: ligand in x)]

        if update:
            self.df = df
        return df

    def has_ligands(
        self,
        ligands: List[str],
        splits: Optional[List[str]] = None,
        inverse: bool = False,
        update: bool = False,
    ):
        """Select molecules that contain all ligands in the provided list.

        If inverse is ``True``, selects molecules that do not have all the
        ligands in the list.

        :param ligand: List of ligands.
            (PDB ligand codes - http://ligand-expo.rcsb.org/)
        :type ligand: List[str]
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param inverse: Whether to inverse the selection,
            defaults to ``False``.
        :type inverse: bool, optional
        :param update: Whether to update the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        if inverse:
            df = splits_df.loc[
                splits_df.ligands.map(lambda x: not set(ligands).issubset(x))
            ]
        else:
            df = splits_df.loc[
                splits_df.ligands.map(lambda x: set(ligands).issubset(x))
            ]

        if update:
            self.df = df
        return df

    def remove_non_standard_alphabet_sequences(
        self, splits: Optional[List[str]] = None, update: bool = False
    ):
        """
        Remove sequences with non-standard characters.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to update the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame containing only sequences with standard characters.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.loc[
            splits_df.sequence.map(
                lambda x: set(x).issubset(set("ACDEFGHIKLMNPQRSTVWY"))
            )
        ]
        if update:
            self.df = df
        return df

    def select_complexes_with_grouped_molecule_types(
        self,
        molecule_types_to_group: List[str],
        splits: Optional[List[str]] = None,
        update: bool = False,
    ):
        """
        Select complexes containing at least one instance of each
        provided molecule type.

        :param molecule_types_to_group: Names of molecule types by which to assemble complexes.
        :type molecule_types_to_group: List[str]
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to update the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame containing only complexes with at least one instance
          of each provided molecule type.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.groupby("pdb").filter(
            lambda group: all(
                [
                    molecule_type_to_group in group["molecule_type"].values
                    for molecule_type_to_group in molecule_types_to_group
                ]
            )
        )
        if update:
            self.df = df

    def has_uniprot_id(
        self,
        select_ids: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """
        Select entries that have a uniprot ID.

        :param select_ids: If present, filter for only these IDs. If not present, filter for entries
            that have any uniprot ID.
            defaults to ``None``.
        :type select_ids: Optional[List[str]], optional
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.dropna(subset=["uniprot_id"])

        if select_ids:
            df = df[df["uniprot_id"].isin(select_ids)]

        if update:
            self.df = df
        return df

    def has_cath_code(
        self,
        select_ids: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """
        Select entries that have a cath code.

        :param select_ids: If present, filter for only these CATH codes. If not present, filter for entries
            that have any cath code.
            defaults to ``None``.
        :type select_ids: Optional[List[str]], optional
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.dropna(subset=["cath_code"])

        if select_ids:
            df = df[df["cath_code"].isin(select_ids)]

        if update:
            self.df = df
        return df

    def has_ec_number(
        self,
        select_ids: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
        update: bool = False,
    ) -> pd.DataFrame:
        """
        Select entries that have an EC number.

        :param select_ids: If present, filter for only these ec_numbers. If not present, filter for entries
            that have any EC number
            defaults to ``None``.
        :type select_ids: Optional[List[str]], optional
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        splits_df = self.get_splits(splits)
        df = splits_df.dropna(subset=["ec_number"])

        if select_ids:
            df = df[df["ec_number"].isin(select_ids)]

        if update:
            self.df = df
        return df

    def split_df_proportionally(
        self,
        df: pd.DataFrame,
        splits: List[str],
        split_ratios: List[float],
        assign_leftover_rows_to_split_n: int = 0,
        random_state: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split the provided DataFrame iteratively according to given proportions.

        :param df: DataFrame to split.
        :type df: pd.DataFrame
        :param splits: Names of splits into which to divide the
            provided DataFrame.
        :type splits: List[str]
        :param split_ratios: Ratios by which to split the provided DataFrame.
        :type split_ratios: List[float]
        :param assign_leftover_rows_to_split_n: To which split to assign
            leftover rows, defaults to ``0``.
        :type assign_leftover_rows_to_split_n: int, optional
        :param random_state: Random seed to use for DataFrame splitting,
            defaults to ``42``.
        :type random_state: int, optional
        :return: Dictionary of DataFrame splits.
        :rtype: Dict[str, pd.DataFrame]
        """
        assert len(splits) == len(
            split_ratios
        ), f"Number of splits ({len(splits)}) must match number of split ratios ({len(split_ratios)})"
        assert (
            sum(split_ratios) == 1
        ), f"Split ratios must sum to 1, got {sum(split_ratios)} ({split_ratios})"

        # Calculate the size of each split
        split_sizes = [int(len(df) * ratio) for ratio in split_ratios]

        # Assign leftover rows to a specified split
        num_remaining_rows = len(df) - sum(split_sizes)
        if num_remaining_rows > 0:
            split_sizes[assign_leftover_rows_to_split_n] += num_remaining_rows

        # Without replacement, randomly shuffle rows within the input DataFrame
        df_sampled = df.sample(
            frac=1.0, replace=False, random_state=random_state
        )

        # Split DataFrames
        start_idx = 0
        df_splits = {}
        for split_index, split_size in enumerate(split_sizes):
            split = splits[split_index]
            end_idx = start_idx + split_size
            df_split = df_sampled.iloc[start_idx:end_idx]
            df_splits[split] = df_split
            start_idx = end_idx

        # Ensure there are no duplicated rows between splits
        all_splits_df = pd.concat([df_splits[split] for split in splits])
        assert len(all_splits_df) == len(
            df
        ), "Number of rows changed during split operations."
        assert len(
            all_splits_df.drop(self.list_columns, axis=1).drop_duplicates()
        ) == len(df), "Duplicate rows found in splits."

        df_split_sizes = " ".join(
            [str(df_splits[split].shape[0]) for split in df_splits]
        )
        log.info(
            f"Proportionally-derived dataset splits of sizes: {df_split_sizes}"
        )

        return df_splits

    def merge_df_splits(
        self,
        first_df_split: pd.DataFrame,
        second_df_split: pd.DataFrame,
        split: str,
    ) -> pd.DataFrame:
        """Reconcile an existing DataFrame split with a new split.

        :param first_df_split: Existing DataFrame split.
        :type first_df_split: pd.DataFrame
        :param second_df_split: New DataFrame split.
        :type second_df_split: pd.DataFrame
        :param split: Name of DataFrame split.
        :type split: str

        :return: Merged DataFrame split.
        :rtype: pd.DataFrame
        """
        # Coerce list columns into tuple columns
        # Ref: https://stackoverflow.com/questions/45991496/merging-dataframes-with-unhashable-columns
        for df_split in [first_df_split, second_df_split]:
            for list_column in self.list_columns:
                if list_column in df_split.columns:
                    df_split[list_column] = df_split[list_column].apply(tuple)

        # Merge DataFrame splits
        merge_columns = [
            c for c in first_df_split.columns.to_list() if "split" not in c
        ]
        merged_df_split = pd.merge(
            first_df_split.drop("split", axis=1),
            second_df_split.drop("split", axis=1),
            how="inner",
            on=merge_columns,
        )

        # Coerce tuple columns back into list columns
        for df_split in [first_df_split, second_df_split]:
            for list_column in self.list_columns:
                if list_column in df_split.columns:
                    df_split[list_column] = df_split[list_column].apply(list)

        # Track split names
        merged_df_split["split"] = split
        return merged_df_split

    def split_clusters(
        self,
        df: pd.DataFrame,
        update: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Split clusters derived by MMseqs2.

        :param df: DataFrame containing the clusters derived by MMseqs2.
        :type df: pd.DataFrame
        :param update: Whether to update the selection to the representative
            sequences, defaults to ``False``.
        :type update: bool, optional

        :return: A Dictionary of split names mapping to DataFrames of
            randomly-split representative sequences.
        :rtype: Dict[str, pd.DataFrame]
        """
        split_ratios_provided = self.split_ratios is not None
        assert split_ratios_provided, "Split ratios must be provided."

        # Split clusters
        ratio_str = " ".join([str(r) for r in self.split_ratios])
        log.info(f"Randomly splitting clusters into ratios: {ratio_str}...")
        df_splits = self.split_df_proportionally(
            df,
            self.splits,
            self.split_ratios,
            self.assign_leftover_rows_to_split_n,
        )
        log.debug("Done splitting clusters")

        # Update splits
        for split in self.splits:
            if update:
                df_split = df_splits[split]
                if self.df_splits[split] is not None:
                    self.df_splits[split] = self.merge_df_splits(
                        self.df_splits[split], df_split, split
                    )
                else:
                    df_split.split = split
                    self.df_splits[split] = df_split
                df_splits[split] = self.df_splits[split]

        return df_splits

    def cluster(
        self,
        min_seq_id: float = 0.3,
        coverage: float = 0.8,
        update: bool = False,
        fasta_fname: Optional[str] = None,
        cluster_fname: Optional[str] = None,
        overwrite: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Cluster sequences in selection using MMseqs2.

        By default, the clusters are stored in a file named:
        ``f"pdb_cluster_rep_seq_id_{min_seq_id}_c_{coverage}.fasta"``.

        :param min_seq_id: Sequence identity, defaults to ``0.3``.
        :type min_seq_id: float, optional
        :param coverage: Clustering coverage, defaults to ``0.8``.
        :type coverage: float, optional
        :param update: Whether to update the selection to the representative
            sequences, defaults to ``False``.
        :type update: bool, optional
        :param fasta_fname: Name of FASTA file to which to write,
            defaults to ``None``.
        :type fasta_fname: Optional[str], optional
        :param cluster_fname: Custom name for cluster file,
            defaults to ``None``.
        :type cluster_fname: Optional[str], optional
        :param overwrite: Whether to overwrite cached clusters,
            defaults to ``False``.
        :type overwrite: bool, optional

        :return: Either a single DataFrame of representative sequences or a
            Dictionary of split names mapping to DataFrames of randomly-split
            representative sequences.
        :rtype: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        """
        # Build name of FASTA and cluster files
        if cluster_fname is None:
            fasta_fname = "pdb.fasta"
            cluster_fname = (
                f"pdb_cluster_rep_seq_id_{min_seq_id}_c_{coverage}.fasta"
            )

        # Do clustering if overwriting or no clusters were found
        if not os.path.exists(self.root_dir / cluster_fname) or overwrite:
            # Remove existing file if we are overwriting
            if os.path.exists(self.root_dir / cluster_fname) and overwrite:
                log.info(
                    f"Overwriting. Removing old cluster file:  {self.root_dir / cluster_fname}"
                )
                os.remove(self.root_dir / cluster_fname)

            # Create clusters
            log.info("Creating clusters...")
            # Write selection to FASTA
            log.info(
                f"Writing current selection ({len(self.df)} chains) to FASTA..."
            )
            self.to_fasta(str(self.root_dir / fasta_fname))
            if not is_tool("mmseqs"):
                log.error(
                    "MMseqs2 not found. Please install it: conda install -c conda-forge -c bioconda mmseqs2"
                )
            else:
                # Run MMSeqs
                cmd = f"mmseqs easy-cluster {str(self.root_dir / fasta_fname)} pdb_cluster tmp --min-seq-id {min_seq_id} -c {coverage} --cov-mode 1"
                log.info(f"Clustering with: {cmd}")
                subprocess.run(cmd.split())
                os.rename(
                    "pdb_cluster_rep_seq.fasta", self.root_dir / cluster_fname
                )
                log.info("Done clustering!")
        # Otherwise, read from disk
        elif os.path.exists(self.root_dir / cluster_fname):
            log.info(
                f"Found existing clusters. Loading clusters from disk: {self.root_dir / cluster_fname}"
            )

        # Read FASTA
        df = self.from_fasta(ids="chain", filename=str(cluster_fname))
        if update:
            self.df = df

        # Split FASTA
        return self.split_clusters(df, update) if self.splits_provided else df

    def split_df_into_time_frames(
        self,
        df: pd.DataFrame,
        splits: List[str],
        split_time_frames: List[np.datetime64],
    ) -> Dict[str, pd.DataFrame]:
        """
        Split the provided DataFrame sequentially according to given
        time frames.

        :param df: DataFrame to split.
        :type df: pd.DataFrame
        :param splits: Names of splits into which to divide the
            provided DataFrame.
        :type splits: List[str]
        :param split_time_frames: Time frames into which to split the
            provided DataFrame.
        :type split_time_frames: List[np.datetime64]
        :return: Dictionary of DataFrame splits.
        :rtype: Dict[str, pd.DataFrame]
        """
        assert len(splits) == len(
            split_time_frames
        ), f"Number of splits ({len(splits)}) must match number of time frames ({len(split_time_frames)})."
        assert self._frames_are_sequential(
            split_time_frames
        ), "Time frames must be sequential."

        # Split DataFrames
        start_datetime = df.deposition_date.min()
        df_splits = {}
        for split_index in range(len(splits)):
            split = splits[split_index]
            end_datetime = split_time_frames[split_index]
            df_split = df.loc[
                (df.deposition_date >= start_datetime)
                & (df.deposition_date < end_datetime)
            ]
            df_split.split = split
            df_splits[split] = df_split
            start_datetime = end_datetime

        # Identify any remaining rows
        start_datetime = end_datetime
        end_datetime = df.deposition_date.max()
        num_remaining_rows = df.loc[
            (df.deposition_date >= start_datetime)
            & (df.deposition_date <= end_datetime)
        ].shape[0]

        # Ensure there are no duplicated rows between splits
        all_splits_df = pd.concat([df_splits[split] for split in splits])
        assert (
            len(all_splits_df) == len(df) - num_remaining_rows
        ), "Number of rows changed during split operations."
        assert (
            len(
                all_splits_df.drop(self.list_columns, axis=1).drop_duplicates()
            )
            == len(df) - num_remaining_rows
        ), "Duplicate rows found in splits."

        df_split_sizes = " ".join(
            [str(df_splits[split].shape[0]) for split in df_splits]
        )
        log.info(
            f"Deposition date-derived dataset splits of sizes: {df_split_sizes}"
        )

        return df_splits

    def split_by_deposition_date(
        self,
        df: pd.DataFrame,
        update: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Split molecules based on their deposition date.

        :param df: DataFrame containing the molecule sequences to split.
        :type df: pd.DataFrame
        :param update: Whether to update the selection to the PDB entries
            defaults to ``False``.
        :type update: bool, optional

        :return: A Dictionary of split names mapping to DataFrames of
            sequence splits based on the sequential time frames given.
        :rtype: Dict[str, pd.DataFrame]
        """
        split_time_frames_provided = self.split_time_frames is not None
        assert (
            split_time_frames_provided
        ), "Split time frames must be provided."

        # Split sequences
        time_frames = " ".join([str(f) for f in self.split_time_frames])
        log.info(f"Splitting sequences into time frames: {time_frames}")
        df_splits = self.split_df_into_time_frames(
            df, self.splits, self.split_time_frames
        )
        log.info("Done splitting sequences")

        # Update splits
        for split in self.splits:
            if update:
                df_split = df_splits[split]
                if self.df_splits[split] is not None:
                    self.df_splits[split] = self.merge_df_splits(
                        self.df_splits[split], df_split, split
                    )
                else:
                    df_split.split = split
                    self.df_splits[split] = df_split
                df_splits[split] = self.df_splits[split]

        return df_splits

    def filter_by_deposition_date(
        self, max_deposition_date: np.datetime64, update: bool = False
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Select molecules deposited on or before a given date.

        :param max_deposition_date: Maximum deposition date of molecule.
        :type max_deposition_date: np.datetime64
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional

        :return: Either a single DataFrame of sequences or a
            Dictionary of split names mapping to DataFrames of
            sequences split successively by their deposition date.
        :rtype: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        """
        # Drop missing deposition dates
        df = self.df.dropna().loc[
            self.df.deposition_date < max_deposition_date
        ]
        if update:
            self.df = df

        # Split sequences
        return (
            self.split_by_deposition_date(df, update)
            if self.splits_provided
            else df
        )

    def reset(self) -> pd.DataFrame:
        """Reset the dataset to the original DataFrame source.

        :return: The source dataset DataFrame.
        :rtype: pd.DataFrame
        """
        self.df = self.source.copy()
        return self.df

    def download_pdbs(
        self,
        out_dir: str = ".",
        format: str = "pdb",
        splits: Optional[List[str]] = None,
        overwrite: bool = False,
        max_workers: int = 8,
        chunksize: int = 32,
    ):
        """Download PDB files in the current selection.

        :param out_dir: Output directory, defaults to ``"."``
        :type out_dir: str, optional
        :param format: Filetype to download. ``pdb``, ``mmtf``, ``mmcif`` or ``bcif``.
        :type format: str
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param overwrite: Overwrite existing files, defaults to ``False``.
        :type overwrite: bool, optional
        :param max_workers: Number of processes to use, defaults to ``8``.
        :type max_workers: int, optional
        :param chunksize: Chunk size for each worker, defaults to ``32``.
        :type chunksize: int, optional
        """
        log.info(
            f"Downloading {len(self.get_unique_pdbs(splits))} PDB files..."
        )
        download_pdb_multiprocessing(
            self.get_unique_pdbs(splits),
            out_dir,
            format=format,
            overwrite=overwrite,
            max_workers=max_workers,
            chunksize=chunksize,
        )

    def write_chains(
        self,
        splits: Optional[List[str]] = None,
        models: List[int] = [1],
        force: bool = False,
    ) -> List[Path]:
        # sourcery skip: simplify-len-comparison, use-named-expression
        """Write chains in current selection to disk. e.g., we create a file
        of the form ``4hbb_A.pdb`` for chain ``A`` of PDB file ``4hhb.pdb``.

        If the PDB files are not contained in ``self.pdb_dir``, they are
        downloaded.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param models: List of indices of models from which to extract chains,
            defaults to ``[1]``.
        :type models: List[int], optional
        :param force: Whether to force downloads of selections containing
            unavailable PDBs, defaults to ``False``.
        :type force: bool, optional
        :return: List of paths to written files.
        :rtype: List[Path]
        """

        splits_df = self.get_splits(splits)
        if not force:
            self._check_download_availability(splits_df, raise_error=True)
        # Get dictionary of PDB code : List[Chains]
        df = splits_df.groupby("pdb")["chain"].agg(list).to_dict()

        # Check we have all source PDB files
        downloaded = os.listdir(self.pdb_dir)
        downloaded = [
            f for f in downloaded if f.endswith(f".{self.structure_format}")
        ]

        to_download = [
            k
            for k in df.keys()
            if f"{k}.{self.structure_format}" not in downloaded
        ]
        if len(to_download) > 0:
            log.info(f"Downloading {len(to_download)} PDB files...")
            download_pdb_multiprocessing(
                to_download, self.pdb_dir, overwrite=True
            )
            log.info("Done downloading PDB files")

        # Iterate over dictionary and write chains to separate files
        log.info("Extracting chains...")
        paths = []
        for k, v in tqdm(df.items()):
            in_file = os.path.join(
                self.pdb_dir, f"{k}.{self.structure_format}"
            )
            paths.append(
                extract_chains_to_file(
                    in_file, v, out_dir=self.pdb_dir, models=models
                )
            )
        log.info("Done extracting chains")

        # Flatten list of paths
        return [Path(num) for sublist in paths for num in sublist]

    def from_fasta(
        self, ids: str, filename: str, splits: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Create a selection from a FASTA file.

        :param ids: One of ``"chain"`` or ``"pdb"``. i.e., Whether the
            FASTA is indexed by chains (i.e., ``3eiy_A``) or PDB ids (``3eiy``).
        :type ids: str
        :param filename: Name of FASTA file.
        :type filename: str
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        fasta = read_fasta(self.root_dir / filename)
        seq_ids = list(fasta.keys())
        splits_df = self.get_splits(splits, source=True)
        if ids == "chain":
            return splits_df.loc[splits_df.id.isin(seq_ids)]
        elif ids == "pdb":
            return splits_df.loc[splits_df.pdb.isin(seq_ids)]
        else:
            raise ValueError(
                "Invalid parameter ids. Must be 'chain' or 'pdb'."
            )

    def to_chain_sequence_mapping_dict(
        self, splits: Optional[List[str]]
    ) -> Dict[str, str]:
        """Return a dictionary of sequences indexed by chains.

        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional

        :return: Dictionary of chain-sequence mappings.
        :rtype: Dict[str, str]
        """
        splits_df = self.get_splits(splits)
        self._check_download_availability(splits_df, raise_error=False)
        return (
            splits_df[["id", "sequence"]].set_index("id").to_dict()["sequence"]
        )

    def to_fasta(self, filename: str, splits: Optional[List[str]] = None):
        """Write the dataset to a FASTA file (indexed by chain id).

        :param filename: Name of the output FASTA file.
        :type filename: str
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        """
        self._check_download_availability(
            self.get_splits(splits), raise_error=False
        )
        with open(filename, "w") as f:
            for k, v in self.to_chain_sequence_mapping_dict(splits).items():
                f.write(f">{k}\n")
                f.write(f"{v}\n")

    @staticmethod
    def _check_download_availability(df: pd.DataFrame, raise_error: bool):
        """Check whether the selection contains PDBs that are not available
        for download in PDB format.

        :param df: DataFrame of selected molecules to check.
        :type df: pd.DataFrame
        :param raise_error: Whether to raise an error if unavailable PDBs are
            found. Otherwise a warning is logged.
        :type raise_error: bool
        :raises ValueError: Raised if ``raise_error`` is ``True`` and
            unavailable PDB files are found in ``df``.
        """
        if not all(df.pdb_file_available):
            unavailable = df.loc[
                df.pdb_file_available == False, "pdb"
            ].to_list()
            if raise_error:
                raise ValueError(
                    f"You are exporting a selection that contains {len(unavailable)} PDB(s) unavailable for download in PDB format: {unavailable}"
                )
            else:
                log.warning(
                    f"You are exporting a selection that contains {len(unavailable)} PDB(s) unavailable for download in PDB format: {unavailable}"
                )

    def to_csv(self, fname: str, splits: Optional[List[str]] = None):
        """Write the selection to a CSV file.

        :param fname: Path to CSV file.
        :type fname: str
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        """
        splits_df = self.get_splits(splits)
        self._check_download_availability(splits_df, raise_error=False)
        log.info(
            f"Writing selection ({len(splits_df)} chains) to CSV file: {fname}"
        )

        splits_df.to_csv(fname, index=False)

    def merge_pdb_chain_groups(self, group: DataFrameGroupBy) -> pd.DataFrame:
        """Combine groups of chains associated with the same PDB code.

        :param group: A DataFrame group representing collections of
            PDB codes with their associated chains.
        :type group: DataFrameGroupBy
        :return: Group of PDB codes and their associated chains as a DataFrame.
        :rtype: pd.DataFrame
        """
        return pd.DataFrame(
            {"pdb": [group["pdb"].iloc[0]], "chain": [group["chain"].tolist()]}
        )

    def select_pdb_by_criterion(
        self,
        pdb: PandasPdb,
        field: str,
        field_values: List[Any],
        pdb_code: str,
    ) -> PandasPdb:
        """Filter a PDB using a field selection.

        :param pdb: The PDB object to filter by a field.
        :type pdb: PandasPdb
        :param field: The field by which to filter the PDB.
        :type field: str
        :param field_values: The field values by which to filter
            the PDB.
        :type field_values: List[Any]
        :param pdb_code: The PDB code associated with a given PDB object.
        :type pdb_code: str

        :return: The filtered PDB object.
        :rtype: PandasPdb
        """
        for key in pdb.df:
            if field in pdb.df[key]:
                filtered_pdb = pdb.df[key][
                    pdb.df[key][field].isin(field_values)
                ]
                if "ATOM" in key and len(filtered_pdb) == 0:
                    log.warning(
                        f"DataFrame for PDB {pdb_code} does not contain any standard atoms after filtering"
                    )
                pdb.df[key] = filtered_pdb
        return pdb

    def write_out_pdb_chain_groups(
        self,
        df: pd.DataFrame,
        pdb_dir: str,
        out_dir: str,
        split: str,
        merge_fn: Callable,
        atom_df_name: str = "ATOM",
        max_num_chains_per_pdb_code: int = -1,
        models: List[int] = [1],
    ):
        """Record groups of PDB codes and associated chains
        as collated PDB files.

        :param df: DataFrame containing the PDB codes and chains
            to group and record.
        :type df: pd.DataFrame
        :param pdb_dir: Path to PDB input directory.
        :type pdb_dir: str
        :param out_dir: The root directory in which to store records.
        :type out_dir: str
        :param split: The split with which to store records.
        :type split: str
        :param merge_fn: The PDB code-chain grouping function to use.
        :type merge_fn: Callable
        :param atom_df_name: Name of the DataFrame by which to access
            ATOM entries within a PandasPdb object.
        :type atom_df_name: str, defaults to ``ATOM``
        :param max_num_chains_per_pdb_code: Maximum number of chains
            to collate into a matching PDB file, defaults to ``-1``.
        :type max_num_chains_per_pdb_code: int, optional
        :param models: List of indices of models from which to extract chains,
            defaults to ``[1]``.
        :type models: List[int], optional
        """
        if len(df) > 0:
            split_dir = Path(out_dir) / split
            os.makedirs(str(split_dir), exist_ok=True)

            df_sorted = df.sort_values("pdb")
            df_merged = df_sorted.groupby("pdb").apply(merge_fn)
            df_merged = df_merged.reset_index(drop=True)

            for _, entry in tqdm(df_merged.iterrows()):
                entry_pdb_code, entry_chains = entry["pdb"], entry["chain"]

                input_pdb_filepath = (
                    Path(pdb_dir) / f"{entry_pdb_code}.{self.structure_format}"
                )
                output_pdb_filepath = (
                    split_dir / f"{entry_pdb_code}.{self.structure_format}"
                )

                if not os.path.exists(str(output_pdb_filepath)):
                    try:
                        pdb = (
                            PandasPdb()
                            .read_pdb(str(input_pdb_filepath))
                            .get_models(models)
                        )
                    except FileNotFoundError:
                        log.info(
                            f"Failed to load {str(input_pdb_filepath)}. Perhaps it is not longer available to download from the PDB?"
                        )
                        continue
                    # work around int-typing bug for `model_id` within version `0.5.0.dev0` of BioPandas -> appears when calling `to_pdb()`
                    cast_pdb_column_to_type(
                        pdb, column_name="model_id", type=str
                    )
                    # select only from chains available in the PDB file
                    pdb_atom_chains = (
                        pdb.df[atom_df_name].chain_id.unique().tolist()
                    )
                    chains = [
                        chain
                        for chain in entry_chains
                        if chain in pdb_atom_chains
                    ]
                    chains = (
                        chains
                        if max_num_chains_per_pdb_code == -1
                        else chains[:max_num_chains_per_pdb_code]
                    )
                    pdb_chains = self.select_pdb_by_criterion(
                        pdb, "chain_id", chains, entry_pdb_code
                    )
                    # export selected chains within the same PDB file
                    pdb_chains.to_pdb(str(output_pdb_filepath))

    def write_df_pdbs(
        self,
        pdb_dir: str,
        df: pd.DataFrame,
        out_dir: str = "collated_pdb",
        splits: Optional[List[str]] = None,
        max_num_chains_per_pdb_code: int = -1,
        models: List[int] = [1],
    ):
        """Write the given selection as a collection of PDB files.

        :param pdb_dir: Path to PDB input directory.
        :type pdb_dir: str
        :param df: DataFrame on which to perform the operation,
            defaults to ``None``.
        :type df: pd.DataFrame
        :param out_dir: Name of directory in which to store
            collated PDB files.
        :type out_dir: str, optional
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param max_num_chains_per_pdb_code: Maximum number of chains
            to collate into a matching PDB file, defaults to ``-1``.
        :type max_num_chains_per_pdb_code: int, optional
        :param models: List of indices of models from which to extract chains,
            defaults to ``[1]``.
        :type models: List[int], optional
        """
        out_dir = Path(pdb_dir) / out_dir
        os.makedirs(out_dir, exist_ok=True)

        if splits is not None:
            for split in splits:
                split_df = df.loc[df.split == split]
                self.write_out_pdb_chain_groups(
                    df=split_df,
                    pdb_dir=pdb_dir,
                    out_dir=out_dir,
                    split=split,
                    merge_fn=self.merge_pdb_chain_groups,
                    max_num_chains_per_pdb_code=max_num_chains_per_pdb_code,
                    models=models,
                )
        else:
            self.write_out_pdb_chain_groups(
                df=df,
                pdb_dir=pdb_dir,
                out_dir=out_dir,
                split="full",
                merge_fn=self.merge_pdb_chain_groups,
                max_num_chains_per_pdb_code=max_num_chains_per_pdb_code,
                models=models,
            )

    def export_pdbs(
        self,
        pdb_dir: str,
        splits: Optional[List[str]] = None,
        max_num_chains_per_pdb_code: int = -1,
        models: List[int] = [1],
        force: bool = False,
    ):
        """Write the selection as a collection of PDB files.

        :param pdb_dir: Path to PDB output directory.
        :type pdb_dir: str
        :param splits: Names of splits for which to perform the operation,
            defaults to ``None``.
        :type splits: Optional[List[str]], optional
        :param max_num_chains_per_pdb_code: Maximum number of chains
            to collate into a matching PDB file, defaults to ``-1``.
        :type max_num_chains_per_pdb_code: int, optional
        :param models: List of indices of models from which to extract chains,
            defaults to ``[1]``.
        :type models: List[int], optional
        :param force: Whether to raise an error if the download selection
            contains PDBs which are not available in PDB format.
        """
        split_dfs = self.get_splits(splits)
        if not force:
            self._check_download_availability(split_dfs, raise_error=True)
        log.info(
            f"Writing selection ({len(split_dfs)} PDB chains) to directory: {pdb_dir}"
        )

        self.write_df_pdbs(
            pdb_dir,
            split_dfs,
            splits=splits,
            max_num_chains_per_pdb_code=max_num_chains_per_pdb_code,
            models=models,
        )
        log.info("Done writing selection of PDB chains")
