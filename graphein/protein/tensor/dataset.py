"""Pytorch Geometric Dataset classes for Protein Graphs."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import os
from collections.abc import Sequence
from difflib import get_close_matches
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Union

import networkx as nx
import numpy as np
from loguru import logger as log
from tqdm import tqdm

from graphein.protein.utils import (  # download_pdb,
    download_alphafold_structure,
    download_pdb_multiprocessing,
    read_fasta,
)
from graphein.utils.dependencies import import_message

from ..folding_utils import esmfold
from .data import Protein, ProteinBatch, to_protein_mp

try:
    import torch
    from torch_geometric.data import Data, Dataset, InMemoryDataset
except ImportError:
    import_message(
        "graphein.ml.datasets.torch_geometric_dataset",
        "torch_geometric",
        conda_channel="pyg",
        pip_install=True,
    )

IndexType = Union[slice, torch.Tensor, np.ndarray, Sequence]


class ProteinDataset(Dataset):
    def __init__(
        self,
        root: str,
        pdb_dir: str,
        out_dir: str,
        overwrite: bool = False,
        paths: Optional[List[str]] = None,
        pdb_codes: Optional[List[str]] = None,
        uniprot_ids: Optional[List[str]] = None,
        sequences: Optional[Union[List[str], str]] = None,
        graph_labels: Optional[List[torch.Tensor]] = None,
        node_labels: Optional[List[torch.Tensor]] = None,
        chain_selections: Optional[List[str]] = None,
        graph_transformation_funcs: Optional[List[Callable]] = None,
        pdb_transform: Optional[List[Callable]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        num_cores: int = 16,
        af_version: int = 2,
    ):
        """Dataset class for protein graphs.

        Dataset base class for creating graph datasets.
        See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
        create_dataset.html>`__ for the accompanying tutorial.

        :param root: Root directory where the dataset should be saved.
        :type root: str
        :param pdb_dir: Directory to save PDB files to. If not specified
            (`pdb_dir=`None``), will default to ``root/pdb``. Defaults to
            ``None``.
        :type pdb_dir: str, optional
        :param out_dir: Directory to save output files to. If not specified
            (``out_dir=None``), will default to ``root/processed``. Defaults to
            ``None``.
        :param overwrite: Whether to overwrite existing files. Defaults to
            ``False``.
        :type overwrite: bool, optional
        :param paths: List of full path of PDB or MMTF files to load. Defaults to
            ``None``.
        :type paths: Optional[List[str]], optional
        :param pdb_codes: List of PDB codes to download and parse from the PDB.
            Defaults to ``None``.
        :type pdb_codes: Optional[List[str]], optional
        :param uniprot_ids: List of Uniprot IDs to download and parse from
            Alphafold Database. Defaults to ``None``.
        :type uniprot_ids: Optional[List[str]], optional
        :param sequences: List of protein sequences to predict structures of with
            ESMFold. Also accepts a path to a fasta file. Defaults to ``None``.
        :type sequences: Optional[Union[List[str], str]], optional
        :param graph_label_map: Dictionary mapping PDB/Uniprot IDs to
            graph-level labels. Defaults to ``None``.
        :type graph_label_map: Optional[Dict[str, Tensor]], optional
        :param node_label_map: Dictionary mapping PDB/Uniprot IDs to node-level
            labels. Defaults to ``None``.
        :type node_label_map: Optional[Dict[str, torch.Tensor]], optional
        :param chain_selection_map: Dictionary mapping, defaults to ``None``.
        :type chain_selection_map: Optional[Dict[str, List[str]]], optional
        :param graph_transformation_funcs: List of functions that consume a
            ``nx.Graph`` and return a ``nx.Graph``. Applied to graphs after
            construction but before conversion to pyg. Defaults to ``None``.
        :type graph_transformation_funcs: Optional[List[Callable]], optional
        :param pdb_transform: List of functions that consume a list of paths to
            the downloaded structures. This provides an entry point to apply
            pre-processing from bioinformatics tools of your choosing. Defaults
            to ``None``.
        :type pdb_transform: Optional[List[Callable]], optional
        :param transform: A function/transform that takes in a
            ``torch_geometric.data.Data`` object and returns a transformed
            version. The data object will be transformed before every access.
            Defaults to ``None``.
        :type transform: Optional[Callable], optional
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
        :param num_cores: Number of cores to use for multiprocessing of graph
            construction, defaults to ``16``.
        :type num_cores: int, optional
        :param af_version: Version of AlphaFoldDB structures to use,
            defaults to ``2``.
        :type af_version: int, optional
        """
        # Setup paths
        self.root = root
        self.pdb_dir = pdb_dir
        self.out_dir = out_dir
        self.overwrite = overwrite

        # Setup inputs
        self.pdb_codes = (
            [pdb.lower() for pdb in pdb_codes]
            if pdb_codes is not None
            else None
        )
        self.uniprot_ids = (
            [up.upper() for up in uniprot_ids]
            if uniprot_ids is not None
            else None
        )
        if sequences is not None:
            self.sequences = sequences
            self.fold_sequences()

        self.paths = paths
        if self.paths is None:
            if self.pdb_codes and self.uniprot_ids:
                self.structures = self.pdb_codes + self.uniprot_ids
            elif self.pdb_codes:
                self.structures = pdb_codes
            elif self.uniprot_ids:
                self.structures = uniprot_ids
        # Use local saved pdb_files instead of download or move them to
        # self.root/raw dir
        else:
            if isinstance(self.paths, list):
                self.structures = [
                    os.path.splitext(os.path.split(path)[-1])[0]
                    for path in self.paths
                ]
                self.path, _ = os.path.split(self.paths[0])

        # Labels & chains
        self.chain_selections = chain_selections
        if self.chain_selections is not None:
            self.chain_selections = [
                chain if chain is not None else "all"
                for chain in self.chain_selections
            ]
        self.graph_labels = graph_labels
        self.node_labels = node_labels
        self.validate_input()
        self.bad_pdbs: List[str] = []

        # Configs
        self.num_cores = num_cores
        self.pdb_transform = pdb_transform
        self.graph_transformation_funcs = graph_transformation_funcs
        self.af_version = af_version

        log.info(f"Creating a dataset from {len(self.structures)} structures.")
        log.info(f"PDB Directory: {self.raw_dir}")
        log.info(f"Output Directory: {self.processed_dir}")

        # Do overwriting
        if self.overwrite:
            log.info(
                "Overwrite flag is set. Existing files will be overwritten."
            )
        else:
            log.info(
                "Overwrite flag is not set. Existing files will not be regenerated."
            )
        if self.overwrite:
            log.info("Removing existing (processed) files...")
            for f in tqdm(self.processed_file_names):
                if os.path.exists(os.path.join(self.processed_dir, f)):
                    os.remove(os.path.join(self.processed_dir, f))
        super().__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files in the dataset."""
        return [f"{pdb}.pdb" for pdb in self.structures]

    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files to look for"""
        if self.chain_selections is not None:
            return [
                f"{pdb}_{chain}.pt"
                for pdb, chain in zip(self.structures, self.chain_selections)
            ]
        else:
            return [f"{pdb}.pt" for pdb in self.structures]

    @property
    def raw_dir(self) -> str:
        if hasattr(self, "pdb_dir"):
            return self.pdb_dir
        else:
            return os.path.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        if hasattr(self, "out_dir"):
            return self.out_dir
        else:
            return os.path.join(self.root, "processed")

    def validate_input(self):
        if self.graph_labels is not None:
            assert len(self.structures) == len(
                self.graph_labels
            ), "Number of proteins and graph labels must match"
        if self.node_labels is not None:
            assert len(self.structures) == len(
                self.node_labels
            ), "Number of proteins and node labels must match"
        if self.chain_selections is not None:
            assert len(self.structures) == len(
                self.chain_selections
            ), "Number of proteins and chain selections must match"
            assert len(
                {
                    f"{pdb}_{chain}"
                    for pdb, chain in zip(
                        self.structures, self.chain_selections
                    )
                }
            ) == len(self.structures), "Duplicate protein/chain combinations"

    def download(self):
        """Download the PDB files from RCSB or AlphaFold."""
        if self.pdb_codes:
            # Only download undownloaded PDBs
            to_download = [
                pdb
                for pdb in set(self.pdb_codes)
                if (
                    not os.path.exists(Path(self.raw_dir) / f"{pdb}.pdb")
                    or self.overwrite == True
                )
            ]
            log.info(f"Downloading {len(to_download)} PDBs")
            download_pdb_multiprocessing(
                to_download,
                self.raw_dir,
                max_workers=self.num_cores,
                strict=False,
            )
            self.bad_pdbs = self.bad_pdbs + [
                pdb
                for pdb in set(self.pdb_codes)
                if not os.path.exists(Path(self.raw_dir) / f"{pdb}.pdb")
            ]
            if len(self.bad_pdbs) > 0:
                log.warning(
                    f"Failed to download {len(self.bad_pdbs)} PDBs: {self.bad_pdbs}"
                )
            log.info("PDB download complete")
        if self.uniprot_ids:
            log.info(
                f"Downloading {len(self.uniprot_ids)}AlphaFold structures"
            )
            to_download = [
                pdb
                for pdb in set(self.uniprot_ids)
                if not os.path.exists(Path(self.raw_dir) / f"{pdb}.pdb")
                or self.overwrite == True
            ]
            [
                download_alphafold_structure(
                    uniprot,
                    out_dir=self.raw_dir,
                    version=self.af_version,
                    aligned_score=False,
                )
                for uniprot in tqdm(to_download)
            ]
            log.info("AlphaFold download complete")

    def fold_sequences(self):
        if isinstance(self.sequences, list):
            self.sequences = dict(enumerate(self.sequences))
        elif isinstance(self.sequences, str):
            assert os.path.exists(
                self.sequences
            ), f"Sequence file {self.sequences} does not exist"
            self.sequences = read_fasta(self.sequences)
        else:
            raise ValueError(
                "Sequences must be a list of sequences or a path to a file containing sequences"
            )

        log.info(f"Folding {len(self.sequences)} sequences...")
        if not hasattr(self, "structures"):
            self.structures = []
        for id, seq in tqdm(self.sequences.items()):
            # Add ID to list of structures
            self.structures += [id]
            # Check for existence of file
            if (
                os.path.exists(os.path.join(self.raw_dir, f"{id}.pdb"))
                and not self.overwrite
            ):
                continue
            else:
                esmfold(seq, out_path=os.path.join(self.raw_dir, f"{id}.pdb"))

    def len(self) -> int:
        """Returns length of data set (number of structures)."""
        return len(self.structures)

    def transform_pdbs(self):
        """
        Performs pre-processing of PDB structures before constructing graphs.
        """
        structure_files = [
            f"{self.raw_dir}/{pdb}.pdb" for pdb in self.structures
        ]
        for func in self.pdb_transform:
            func(structure_files)

    def transform_protein_graphs(self, graph: Protein) -> Protein:
        """
        Applies a series of transformations to a protein graph.

        :param graph: Protein graph
        :type graph: Protein
        :return: Transformed protein graph
        :rtype: Protein
        """
        for func in self.graph_transformation_funcs:
            graph = func(graph)
        return graph

    def process(self):
        """Processes structures from files into PyTorch Geometric Data."""
        log.info("Processing PDB files into graphs...")
        # Preprocess PDB files
        if self.pdb_transform:
            log.info(f"Applying PDB transformations: {self.pdb_transform}")
            self.transform_pdbs()
        else:
            log.info("No PDB transformations specified.")

        # Chunk dataset for parallel processing
        chunk_size = 128

        def divide_chunks(l: List[str], n: int = 2) -> Generator:
            for i in range(0, len(l), n):
                yield l[i : i + n]

        examples: List[List[str]] = list(
            divide_chunks(self.structures, chunk_size)
        )
        idxs = list(divide_chunks(range(len(self.structures)), chunk_size))

        if self.chain_selections is None:
            chain_selections = ["all"] * len(self.structures)
        else:
            chain_selections = self.chain_selections

        chain_selections: List[List[str]] = list(
            divide_chunks(chain_selections, chunk_size)
        )

        for idx, pdbs, chains in tqdm(zip(idxs, examples, chain_selections)):
            # Create graph objects
            file_names: List[str] = [f"{self.raw_dir}/{pdb}" for pdb in pdbs]
            file_names: List[str] = [
                f if f.endswith(".pdb") else f"{f}.pdb" for f in file_names
            ]
            print(file_names)
            graphs = to_protein_mp(
                paths=file_names,
                # config=self.config,
                # TODO args
                chain_selections=chains,
                num_cores=self.num_cores,
            )

            if self.graph_transformation_funcs is not None:
                graphs = [self.transform_protein_graphs(g) for g in graphs]

            # Assign labels
            if self.graph_labels is not None:
                for graph_idx, label_idx in enumerate(idx):
                    graphs[graph_idx].graph_y = self.graph_labels[label_idx]
            if self.node_labels is not None:
                for graph_idx, label_idx in enumerate(idx):
                    graphs[graph_idx].node_y = self.node_labels[label_idx]

            # Apply filters and transforms
            if self.pre_filter is not None:
                graphs = [g for g in graphs if self.pre_filter(g)]

            if self.pre_transform is not None:
                graphs = [self.pre_transform(data) for data in graphs]

            # Save graphs as pickles
            for i, (pdb, chain) in enumerate(zip(pdbs, chains)):
                if self.chain_selections is None:
                    torch.save(
                        graphs[i],
                        os.path.join(self.processed_dir, f"{pdb}.pt"),
                    )
                else:
                    torch.save(
                        graphs[i],
                        os.path.join(self.processed_dir, f"{pdb}_{chain}.pt"),
                    )

    def get(self, idx: int) -> Protein:
        """
        Returns Protein Data object for a given index.

        :param idx: Index to retrieve.
        :type idx: int
        :return: Protein data object.
        """
        if isinstance(idx, int):
            if self.chain_selections is None:
                return torch.load(
                    os.path.join(
                        self.processed_dir, f"{self.structures[idx]}.pt"
                    )
                )
            return torch.load(
                os.path.join(
                    self.processed_dir,
                    f"{self.structures[idx]}_{self.chain_selections[idx]}.pt",
                )
            )
        elif isinstance(idx, str):
            try:
                return torch.load(
                    os.path.join(self.processed_dir, f"{idx}.pt")
                )
            except FileNotFoundError:
                files = os.listdir(self.processed_dir)
                similar_files = get_close_matches(
                    f"{idx}.pt", files, n=5, cutoff=0.7
                )
                log.error(
                    f"File {idx}.pt not found. Did you mean: {similar_files}"
                )

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType, str],
    ) -> Union["Dataset", Data]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices."""
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data
        elif isinstance(idx, str):
            data = self.get(idx)
            data = data if self.transform is None else self.transform(data)
            return data
        else:
            return self.index_select(idx)


class ProteinGraphListDataset(InMemoryDataset):
    def __init__(
        self, root: str, data_list: List[Data], name: str, transform=None
    ):
        """Creates a dataset from a list of PyTorch Geometric Data objects.

        :param root: Root directory where the dataset is stored.
        :type root: str
        :param data_list: List of protein graphs as PyTorch Geometric Data
            objects.
        :type data_list: List[Data]
        :param name: Name of dataset. Data will be saved as ``data_{name}.pt``.
        :type name: str
        :param transform: A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        :type transform: Optional[Callable], optional
        """
        self.data_list = data_list
        self.name = name
        super().__init__(root, transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        """The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        return f"{self.name}.pt"

    def process(self):
        """Saves data files to disk."""
        torch.save(self.collate(self.data_list), self.processed_paths[0])
        log.info(f"Saved dataset to disk: {self.name}")
