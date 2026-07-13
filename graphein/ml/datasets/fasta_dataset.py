"""Dataset class for working with sequence-based data."""
# %%
# Graphein
# Author: Arian Jamasb
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import os
import pathlib
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from loguru import logger as log
from tqdm import tqdm

from graphein.protein.folding_utils import (
    esm_embed_fasta,
    esmfold,
    esmfold_fasta,
)
from graphein.protein.tensor.data import Protein
from graphein.protein.utils import read_fasta
from graphein.utils.dependencies import import_message

try:
    from torch_geometric.data import Data, Dataset, InMemoryDataset
except ImportError:
    message = import_message(
        "graphein.ml.datasets.fasta_dataset",
        package="torch_geometric",
        pip_install=True,
        conda_channel="pyg",
    )
    log.warning(message)

esm_embed_params: Dict[str, Any] = {
    "model": "esm2_t33_650M_UR50D",
    "max_tokens": None,
    "repr_layers": [33],
    "include": ["mean", "per_tok"],
    "truncation_seq_length": None,
}


class InMemoryFASTADataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        fasta_file: str,
        representative_sequence_or_structure: Optional[
            Union[str, os.PathLike]
        ] = None,
        esmfold_params: Optional[Dict[str, Any]] = None,
        esm_embed_params: Optional[Dict[str, Any]] = None,
        node_labels: Optional[List] = None,
        graph_labels: Optional[List] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        """Dataset class for working with Sequence Datasets. Provides utilities
        for batch folding and embedding with ESM(Fold).

        # TODO 1. Set representative structure. For protein engineering tasks
        we can have a setup where we predict a single WT structure, which we
        use as the structure for the mutants & simply appropriately modify
        the residue types.


        # TODO 2. FoldComp compression of the predicted structures. Ideally this
        would run in the ESMFold step, but we can also do it post-hoc.

        """
        self.name = name
        self.fasta_file = fasta_file
        self.fasta = read_fasta(fasta_file)
        self.root = root

        self.representative_sequence_or_structure = (
            representative_sequence_or_structure
        )

        self.node_labels = node_labels
        self.graph_labels = graph_labels

        self.esm_embed_params = esm_embed_params
        self.esmfold_params = esmfold_params

        self.embedding_dir = pathlib.Path(self.raw_dir) / "embeddings"
        self.structure_dir = pathlib.Path(self.raw_dir) / "structures"

        self._set_representative()
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        files = [f"{self.name}.fasta"]
        if self.esmfold_params is not None:
            files.extend(f"structures/{k}.pdb" for k in self.fasta.keys())
        if self.esmfold_params is not None:
            files.extend(f"embeddings/{k}.pt" for k in self.fasta.keys())
        return files

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{self.name}.pt"]

    def _set_representative(self):
        if self.representative_sequence_or_structure is None:
            self.representative = None
        elif os.path.exists(self.representative_sequence_or_structure):
            self.representative = Protein().from_pdb_file(
                self.representative_sequence_or_structure
            )
        else:
            esmfold(
                self.representative_sequence_or_structure,
                self.structure_dir / "repseq.pdb",
            )
            self.representative = Protein().from_pdb_file(
                self.structure_dir / "repseq.pdb"
            )

    def embed(self):
        log.info(
            f"Creating ESM embeddings for {len(self.fasta)} sequences in {self.embedding_dir}"
        )
        esm_embed_fasta(
            fasta=self.fasta_file,
            out_dir=self.embedding_dir,
            **self.esm_embed_params,
        )
        log.info(
            f"Created ESM embeddings for {len(self.fasta)} sequences in {self.embedding_dir}"
        )

    def fold(self):
        log.info(
            f"Folding {len(self.fasta)} sequences in {self.structure_dir}"
        )
        esmfold_fasta(
            self.fasta_file, self.structure_dir, **self.esmfold_params
        )
        log.info(f"Folded {len(self.fasta)} sequences in {self.structure_dir}")

    def download(self):
        if self.esmfold_params is not None:
            self.fold()
        if self.esm_embed_params is not None:
            self.embed()

    def process(self):
        # Load structures
        if self.esm_embed_params is not None:
            structures = {
                id: Protein().from_pdb_file(
                    pathlib.Path(self.structure_dir) / f"{id}.pdb"
                )
                for id, seq in tqdm(self.fasta.items())
            }
        else:
            structures = None

        # Load embeddings
        if self.esm_embed_params is not None:
            embeddings = {
                id: torch.load(self.embedding_dir / f"{id}.pt")
                for id, seq in tqdm(self.fasta.items())
            }
        else:
            embeddings = None

        # If we have structures, use it as the base Data object.
        if structures is not None:
            data = structures
        else:
            data = {k: Data(id=k) for k in self.fasta.keys()}

        # Set sequence
        for k, v in data.items():
            data[k].sequence = self.fasta[k]

        # Set embeddings
        # TODO unpack
        if embeddings is not None:
            for k, v in data.items():
                data[k].embedding = embeddings[k]

        # Set node labels
        if self.node_labels is not None:
            for k in data.keys():
                data[k].node_labels = self.node_labels[k]

        # Set graph labels
        if self.graph_labels is not None:
            for k in data.keys():
                data[k].graph_labels = self.graph_labels[k]

        data = list(data.values())

        # Apply pre-filter
        if self.pre_filter is not None:
            data = [d for d in data if self.pre_filter(d)]

        # Apply Pre-transform
        if self.pre_transform is not None:
            data = [self.pre_transform(d) for d in data]

        data, slices = self.collate(data)
        torch.save((data, slices), self.processed_paths[0])
