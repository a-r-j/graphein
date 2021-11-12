"""Class for working with the PSCDB dataset"""
import logging
import multiprocessing
import time

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Code Repository: https://github.com/a-r-j/graphein
import traceback
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from Bio.PDB import PDBList
from tqdm import tqdm

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph

logging.basicConfig(level="DEBUG")
log = logging.getLogger(__name__)


class PSCDB:
    def __init__(
        self,
        compute_bound_graphs: bool = False,
        compute_free_graphs: bool = False,
        protein_graph_config: Optional[
            ProteinGraphConfig
        ] = ProteinGraphConfig(),
        num_cores: int = 8,
    ):
        """
        Initialises the PSCDB Dataset class

        :param compute_bound_graphs: Whether or not to compute (ligand) bound protein graphs
        :type compute_bound_graphs:  bool
        :param compute_free_graphs: Whether or not to compute unbound protein graphs
        :type compute_free_graphs: bool
        :param protein_graph_config: Config object to be used for protein graph construction
        :type protein_graph_config: graphein.protein.config.ProteinGraphConfig
        :param num_cores: Number of cores to use for graph construction. The more the merrier :)
        :type num_cores: int, defaults to 8.
        """
        self.df = self._load_dataset()
        self.bound_pdbs: List[str] = self.get_bound_pdb()
        self.bound_chains: List[str] = self.get_bound_chains()
        self.free_pdbs: List[str] = self.get_free_pdb()
        self.free_chains: List[str] = self.get_free_chains()
        self.protein_names: List[str] = self.get_protein_names()
        self.ligands: List[str] = self.get_ligands()
        self.classification: List[str] = self.get_classification()
        self.motion_type: List[str] = self.get_motion_type()
        self.PSCID: List[str] = self.get_pscid()
        self.protein_names: List[str] = self.get_protein_names()

        self._num_cores: int = num_cores
        self.bad_pdbs: List[str] = []

        if compute_bound_graphs:
            self.config = protein_graph_config
            self.bound_graphs: List[nx.Graph] = self.construct_graphs(
                pdbs=self.bound_pdbs,
                chains=self.bound_chains,
            )
        if compute_free_graphs:
            self.config = protein_graph_config
            self.free_graphs: List[nx.Graph] = self.construct_graphs(
                pdbs=self.free_pdbs,
                chains=self.free_chains,
            )

    @staticmethod
    def _load_dataset() -> pd.DataFrame:
        file_path = Path(
            "../../datasets/pscdb/structural_rearrangement_data.csv"
        )
        df = pd.read_csv(file_path)
        return df

    def get_pscid(self) -> List[str]:
        """
        Returns list of PSCID (PSCDB identifiers)

        :return: List of PSCDB Identifiers
        :rtype: List[str]
        """
        return list(self.df["PSCID"])

    def get_protein_names(self) -> List[str]:
        return list(self.df["Protein Name"])

    def get_ligands(self) -> List[str]:
        return list(self.df["Ligands"])

    def get_classification(self) -> List[str]:
        return list(self.df["Classification(?)"])

    def get_motion_type(self) -> List[str]:
        return list(self.df["motion_type"])

    def get_bound_pdb(self):
        return list(self.df["Bound PDB"])

    def get_bound_chains(self):
        return list(self.df["Bound Chains"])

    def get_free_pdb(self):
        return list(self.df["Free PDB"])

    def get_free_chains(self):
        return list(self.df["Free Chains"])

    def construct_graphs(
        self, pdbs: List[str], chains: List[str]
    ) -> List[nx.Graph]:
        pool = multiprocessing.Pool(self._num_cores)
        graph_list = list(
            pool.map(
                self._graph_constructor,
                [(pdb, chains[i]) for i, pdb in enumerate(pdbs)],
            )
        )
        pool.close()
        pool.join()
        return graph_list

    def _graph_constructor(self, args: Tuple[str, str]):
        log.info(
            f"Constructing graph for: {args[0]}. Chain selection: {args[1]}"
        )
        func = partial(construct_graph, config=self.config)
        try:
            result = func(pdb_code=args[0], chain_selection=args[1])
            return result
        except Exception:
            log.info(
                f"Graph construction error (PDB={args[0]})! {traceback.format_exc()}"
            )
            self.bad_pdbs.append(args[0])

    def split_dataset(
        self,
        strategy: str,
        train_size: float = 0.7,
        val_size: float = 0.2,
        test_size: float = 0.1,
    ):
        assert (
            train_size + val_size + test_size == 1.0
        ), "Train, test & val sizes must sum to 1"

        if strategy == "random":
            pass

    def download_pdbs(self, path: str):
        """
        Downloads dataset PDBs to a specified directories

        :param path: Path to desired output location
        :type path: str
        """
        pdbl = PDBList()
        pdbl.download_pdb_files(pdb_codes=self.pdb_list, pdir=path)

    def __len__(self):
        """Returns length of the dataset

        :returns: Dataset length
        :rtype: int
        """
        return len(self.df)


if __name__ == "__main__":
    ts = time.time()
    c = PSCDB(compute_free_graphs=True, num_cores=16)
    print(time.time() - ts)
