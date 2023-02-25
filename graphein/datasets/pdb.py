import gzip
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import wget
from loguru import logger as log

from graphein.protein.utils import (
    download_pdb_multiprocessing,
    is_tool,
    read_fasta,
)


class PDBManager:
    def __init__(
        self,
        root_dir: str = ".",
        splits: Optional[List[str]] = None,
        split_ratios: Optional[List[float]] = None,
        assign_leftover_rows_to_split_n: int = 0,
    ):
        self.root_dir = Path(root_dir)
        self.download_metadata()
        self.df = self.parse()
        self.source = self.df.copy()

        self.list_columns = ["ligands"]

        self.splits_provided = splits is not None and split_ratios is not None
        if self.splits_provided:
            assert len(set(splits)) == len(splits)
            assert len(splits) == len(split_ratios)
            assert sum(split_ratios) == 1.0

            self.splits = splits
            self.split_ratios = split_ratios
            self.df_splits = {split: None for split in splits}

            self.assign_leftover_rows_to_split_n = (
                assign_leftover_rows_to_split_n
            )

    def download_metadata(self):
        """Download all PDB metadata."""
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
        """Parse the PDB resolutions for all PDB records.

        :return: Dictionary of PDB resolutions with their
            corresponding values.
        :rtype: Dict[str, float]
        """
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
        """Parse the experiment types for all PDB records.

        :return: Dictionary of PDB entries with their
            corresponding experiment types.
        :rtype: Dict[str, str]
        """
        df = pd.read_csv(
            self.root_dir / "pdb_entry_type.txt", sep="\t", header=None
        )
        df.dropna(inplace=True)
        return pd.Series(df[2].values, index=df[0]).to_dict()

    def _parse_source_map(self) -> Dict[str, str]:
        """Parse the source maps for all PDB records.

        :return: Dictionary of PDB entries with their
            corresponding source map values.
        :rtype: Dict[str, str]
        """
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

        :param n: Number of molecules to select, defaults to ``None``.
        :type n: Optional[int], optional
        :param frac: Fraction of molecules to select, defaults to ``None``.
        :type frac: Optional[float], optional
        :param replace: Whether or not to sample with replacement, defaults to
            ``False``.
        :type replace: bool, optional
        :param update: Whether or not to update the DataFrame in place,
            defaults to ``False``.
        :type update: bool, optional
        :return: DataFrame of sampled molecules.
        :rtype: pd.DataFrame
        """
        df = self.df.sample(n=n, frac=frac, replace=replace)
        if update:
            self.df = df
        return df

    def _parse_ligand_map(self) -> Dict[str, List[str]]:
        """Parse the ligand maps for all PDB records.

        :return: Dictionary of PDB entries with their
            corresponding ligand map values.
        :rtype: Dict[str, List[str]]
        """
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
        """Parse all PDB sequence records.

        :return: DataFrame containing PDB sequence entries
            with their corresponding metadata.
        :rtype: pd.DataFrame
        """
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
        """Select molecules by molecule type. [`protein`, `dna`, `rna`]

        :param type: Type of molecule, defaults to "protein".
        :type type: str, optional
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional
        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        df = self.df.loc[self.df.molecule_type == type]

        if update:
            self.df = df
        return df

    def longer_than(self, length: int, update: bool = False) -> pd.DataFrame:
        """Select molecules longer than a given length.

        :param length: Minimum length of molecule.
        :type length: int
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional
        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        df = self.df.loc[self.df.length > length]

        if update:
            self.df = df
        return df

    def shorter_than(self, length: int, update: bool = False) -> pd.DataFrame:
        """
        Select molecules shorter than a given length.

        :param length: Maximum length of molecule.
        :type length: int
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional
        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        df = self.df.loc[self.df.length < length]

        if update:
            self.df = df
        return df

    def oligomeric(self, oligomer: int = 1, update: bool = False):
        """
        Select molecules with a given oligmeric length.

        :param length: Oligomeric length of molecule, defaults to ``1``.
        :type length: int
        :param update: Whether to modify the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional
        :return: DataFrame of selected oligmers.
        :rtype: pd.DataFrame
        """
        df = self.df.loc[self.df.length == oligomer]

        if update:
            self.df = df
        return df

    def has_ligand(self, ligand: str, update: bool = False) -> pd.DataFrame:
        """
        Select molecules that contain a given ligand.

        :param ligand: Ligand to select. (PDB ligand code)
        :type ligand: str
        :param update: Whether to update the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional
        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        df = self.df.loc[self.df.ligands.map(lambda x: ligand in x)]

        if update:
            self.df = df
        return df

    def has_ligands(
        self, ligands: List[str], inverse: bool = False, update: bool = False
    ):
        """Select molecules that contain all ligands in the provided list.

        If inverse is ``True``, selects molecules that do not have all the
        ligands in the list.

        :param ligand: List of ligands. (PDB ligand codes)
        :type ligand: List[str]
        :param inverse: Whether to inverse the selection, defaults to ``False``.
        :type inverse: bool, optional
        :param update: Whether to update the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional
        :return: DataFrame of selected molecules.
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
        """Return a dictionary of sequences indexed by chains.

        :return: Dictionary of chain-sequence mappings.
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
        Remove sequences with non-standard characters.

        :param update: Whether to update the DataFrame in place, defaults to
            ``False``.
        :type update: bool, optional
        :returns: DataFrame containing only sequences with standard characters.
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
        """Download PDB files in the current selection.

        :param out_dir: Output directory, defaults to ``"."``
        :type out_dir: str, optional
        :param overwrite: Overwrite existing files, defaults to ``False``.
        :type overwrite: bool, optional
        :param max_workers: Number of processes to use, defaults to ``8``.
        :type max_workers: int, optional
        :param chunksize: Chunk size for each worker, defaults to ``32``.
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

    def reset(self) -> pd.DataFrame:
        """Reset the dataset to the original DataFrame source.

        :return: The source dataset DataFrame.
        :rtype: pd.DataFrame
        """
        self.df = self.source.copy()
        return self.df

    def split_df_proportionally(
        self,
        df: pd.DataFrame,
        splits: List[str],
        split_ratios: List[float],
        assign_leftover_rows_to_split_n: int = 0,
        random_state: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """Split the provided DataFrame iteratively according to given proportions.

        :param df: DataFrame to split.
        :type df: pd.DataFrame
        :param splits: Names of splits into which to divide the provided DataFrame.
        :type splits: List[str]
        :param split_ratios: Ratios by which to split the provided DataFrame.
        :type split_ratios: List[float]
        :param assign_leftover_rows_to_split_n: To which split to assign leftover rows,
            defaults to ``0``.
        :type assign_leftover_rows_to_split_n: int, optional
        :param random_state: Random seed to use for DataFrame splitting, defaults to
            ``42``.
        :type random_state: int, optional
        :return: Dictionary of DataFrame splits.
        :rtype: Dict[str, pd.DataFrame]
        """
        # Calculate the size of each split
        split_sizes = [int(len(df) * ratio) for ratio in split_ratios]

        # Assign leftover rows to a specified split
        num_remaining_rows = len(df) - sum(split_sizes)
        if num_remaining_rows > 0:
            split_sizes[assign_leftover_rows_to_split_n] += num_remaining_rows

        # Without replacement, randomly shuffle rows within the input DataFrame
        df_sampled = df.sample(frac=1.0, random_state=random_state)

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
        all_rows = pd.concat([df_splits[split] for split in splits])
        assert len(all_rows) == len(
            df
        ), "Number of rows changed during split operations."
        assert len(
            all_rows.drop(self.list_columns, axis=1).drop_duplicates()
        ) == len(df), "Duplicate rows found in splits."

        return df_splits

    def merge_df_splits(
        self,
        first_df_split: pd.DataFrame,
        second_df_split: pd.DataFrame,
        df_primary_key: str = "id",
    ) -> pd.DataFrame:
        """Reconcile an existing DataFrame split with a new split.

        :param first_df_split: Existing DataFrame split.
        :type first_df_split: pd.DataFrame
        :param second_df_split: New DataFrame split.
        :type second_df_split: pd.DataFrame
        :param df_primary_key: Primary column name, defaults to ``"id"``.
        :type df_primary_key: str, optional
        :return: Merged DataFrame split.
        :rtype: pd.DataFrame
        """
        merged_df_split = pd.merge(
            first_df_split, second_df_split, how="inner", on=[df_primary_key]
        )
        return merged_df_split

    def cluster(
        self,
        min_seq_id: float = 0.3,
        coverage: float = 0.8,
        force_process_splits: bool = False,
        update: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Cluster sequences in selection using MMseqs2.

        :param min_seq_id: Sequence identity, defaults to ``0.3``.
        :type min_seq_id: float, optional
        :param coverage: Clustering coverage, defaults to ``0.8``.
        :type coverage: float, optional
        :param force_process_splits: Whether to forcibly (re)process splits,
            defaults to ``False``.
        :type force_process_splits: bool, optional
        :param update: Whether to update the selection to the representative
            sequences, defaults to ``False``.
        :type update: bool, optional

        :return: Either a single DataFrame of representative sequences or a
            Dictionary of split names mapping to DataFrames of randomly-split
            representative sequences.
        :rtype: Union[pd.DataFrame, List[pd.DataFrame]]
        """
        # Write fasta
        self.to_fasta("pdb.fasta")
        if not is_tool("mmseqs"):
            log.error(
                "MMseqs2 not found. Please install it: conda install -c conda-forge -c bioconda mmseqs2"
            )

        # Create clusters
        if not os.path.exists("pdb_cluster_rep_seq.fasta"):
            cmd = f"mmseqs easy-cluster pdb.fasta pdb_cluster tmp --min-seq-id {min_seq_id} -c {coverage} --cov-mode 1"
            log.info(f"Clustering with: {cmd}")
            subprocess.run(cmd.split())
            log.info("Clustering done!")

        # Read fasta
        df = self.from_fasta(ids="chain", filename="pdb_cluster_rep_seq.fasta")
        if update:
            self.df = df

        # Split fasta
        if self.splits_provided:
            df_splits = self.df_splits
            all_splits_exist = all(
                [self.df_splits[split] is not None for split in self.df_splits]
            )
            if not all_splits_exist or force_process_splits:
                log.info(
                    f"Randomly splitting clusters into ratios: {' '.join([str(r) for r in self.split_ratios])}"
                )
                df_splits = self.split_df_proportionally(
                    df,
                    self.splits,
                    self.split_ratios,
                    self.assign_leftover_rows_to_split_n,
                )
                log.info("Done splitting clusters!")

            # Update splits
            for split in self.splits:
                if update:
                    df_split = df_splits[split]
                    if self.df_splits[split] is not None:
                        self.df_splits[split] = self.merge_df_splits(
                            self.df_splits[split], df_split
                        )
                    else:
                        self.df_splits[split] = df_split

            return df_splits

        return df

    def from_fasta(self, ids: str, filename: str) -> pd.DataFrame:
        """Create a selection from a FASTA file.

        :param ids: Whether the FASTA is indexed by chains (i.e., ``3eiy_A``)
            or PDB ids (``3eiy``).
        :type ids: str
        :param filename: Name of FASTA file.
        :type filename: str
        :return: DataFrame of selected molecules.
        :rtype: pd.DataFrame
        """
        fasta = read_fasta(filename)
        seq_ids = list(fasta.keys())
        if ids == "chain":
            return self.source.loc[self.source.id.isin(seq_ids)]
        elif ids == "pdb":
            return self.source.loc[self.source.pdb.isin(seq_ids)]


if __name__ == "__main__":
    pdb_manager = PDBManager(
        root_dir=".",
        splits=["train", "val", "test"],
        split_ratios=[0.8, 0.1, 0.1],
    )
    cluster_dfs = pdb_manager.cluster()
    print(cluster_dfs)
