from typing import List, Optional, Tuple, Union

import pandas as pd
import plotly.graph_objects as go
import torch
from torch_geometric.data import Batch, Data

from ..resi_atoms import PROTEIN_ATOMS
from .angles import (
    alpha,
    dihedrals,
    get_backbone_bond_angles,
    get_backbone_bond_lengths,
    kappa,
    sidechain_torsion,
)
from .geometry import whole_protein_kabsch
from .io import protein_df_to_tensor, protein_to_pyg
from .plot import plot_dihedrals, plot_structure
from .representation import get_backbone, get_c_alpha, get_full_atom_coords
from .testing import (
    has_complete_backbone,
    is_complete_structure,
    is_tensor_equal,
)
from .types import BackboneTensor, CoordTensor, DihedralTensor, TorsionTensor


class Protein(Data):
    def __init__(self, fill_value: float = 1e-5) -> None:
        super().__init__()
        self.fill_value = fill_value

    # I/O
    def from_dataframe(
        self, df: pd.DataFrame, atoms_to_keep: List[str]
    ) -> "Protein":
        self.x = protein_df_to_tensor(df)
        # TODO chains, residue types
        return self

    def from_data(self, data: Data) -> "Protein":
        keys = data.keys
        for key in keys:
            setattr(self, key, getattr(data, key))
        return self

    def to_data(self) -> Data:
        data = Data()
        for i in self.keys:
            setattr(data, i, getattr(self, i))
        return data

    def from_pdb_code(
        self,
        pdb_code: str,
        chain_selection: str = "all",
        deprotonate: bool = True,
        keep_insertions=False,
        keep_hets: List[str] = [],
        model_index: int = 1,
        atom_types: List[str] = PROTEIN_ATOMS,
        node_labels: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None,
    ) -> "Protein":  # sourcery skip: class-extract-method
        data = protein_to_pyg(
            pdb_code=pdb_code,
            chain_selection=chain_selection,
            deprotonate=deprotonate,
            keep_insertions=keep_insertions,
            keep_hets=keep_hets,
            model_index=model_index,
            atom_types=atom_types,
        )
        if node_labels is not None:
            data.node_labels = node_labels

        if graph_labels is not None:
            data.graph_labels = graph_labels

        self.from_data(data)
        return self

    def from_pdb_file(
        self,
        pdb_path: str,
        chain_selection: str = "all",
        deprotonate: bool = True,
        keep_insertions=False,
        keep_hets: List[str] = [],
        model_index: int = 1,
        atom_types: List[str] = PROTEIN_ATOMS,
        node_labels: Optional[torch.Tensor] = None,
        graph_labels: Optional[torch.Tensor] = None,
    ) -> "Protein":
        data = protein_to_pyg(
            pdb_path=pdb_path,
            chain_selection=chain_selection,
            deprotonate=deprotonate,
            keep_insertions=keep_insertions,
            keep_hets=keep_hets,
            model_index=model_index,
            atom_types=atom_types,
        )
        if node_labels is not None:
            data.node_labels = node_labels
        if graph_labels is not None:
            data.graph_labels = graph_labels
        self.from_data(data)
        return self

    # Representation
    def alpha_carbon(self) -> CoordTensor:
        """Returns the alpha carbon coordinates of the protein as a tensor
        of shape Length x 3.

        :return: Alpha carbon coordinates
        :rtype: graphein.protein.tensor.types.CoordTensor
        """
        return get_c_alpha(self.x)

    def backbone(self) -> BackboneTensor:
        return get_backbone(self.x)

    def full_atom_coords(self) -> CoordTensor:
        return get_full_atom_coords(self.x, fill_value=self.fill_value)

    # Angles
    def dihedrals(self) -> DihedralTensor:
        return dihedrals(self.x)

    def sidechain_torsion(self) -> TorsionTensor:
        return sidechain_torsion(self.x)

    def kappa(self) -> torch.Tensor:
        return kappa(self.x)

    def alpha(self) -> torch.Tensor:
        return alpha(self.x)

    def align_to(
        self, other: "Protein", ca_only: bool = True, return_rot: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(other, Protein):
            return whole_protein_kabsch(
                self.x, other.x, ca_only=ca_only, fill_value=self.fill_value
            )
        elif isinstance(other, torch.Tensor):
            return whole_protein_kabsch(
                self.x, other, ca_only=ca_only, fill_value=self.fill_value
            )

    # Testing
    def is_complete() -> bool:
        raise NotImplementedError

    def has_complete_backbone(self) -> bool:
        return has_complete_backbone(self.x)

    def __eq__(self, __o: object) -> bool:
        # sourcery skip: merge-duplicate-blocks, merge-else-if-into-elif
        for i in self.keys:
            attr_self = getattr(self, i)
            attr_other = getattr(__o, i)

            if isinstance(attr_self, torch.Tensor):
                if not is_tensor_equal(attr_self, attr_other):
                    return False
            else:
                if attr_self != attr_other:
                    return False
        return True

    # Plotting
    def plot_dihedrals(self) -> go.Figure:
        if "dihedrals" not in self.keys:
            dh = dihedrals(self.x)
        else:
            dh = self.dihedrals

        return plot_dihedrals(dh)

    def plot_structure(
        self, atoms: List[str] = ["N", "CA", "C", "O"], lines: bool = True
    ) -> go.Figure:
        return plot_structure(self.x, atoms=atoms, lines=lines)


class ProteinBatch(Batch):
    def __init__(self, fill_value: float = 1e-5) -> None:
        super().__init__()
        self.fill_value = fill_value

    def from_batch(
        self, batch: Batch, fill_value: float = 1e-5
    ) -> "ProteinBatch":
        for key in batch.keys:
            setattr(self, key, getattr(batch, key))
        self.fill_value = fill_value
        return self

    def from_protein_list(self, proteins: List[Protein]):
        proteins = [Protein().from_data(p) for p in proteins]
        batch = Batch.from_data_list(proteins)
        self.from_batch(batch)
        return self

    def from_pdb_codes(self, pdb_codes: List[str]):
        raise NotImplementedError

    def from_files(self, pdb_paths: List[str]):
        raise NotImplementedError

    def to_batch(self) -> Batch:
        batch = Batch()
        keys = self.keys
        for key in keys:
            setattr(batch, key, getattr(self, key))
        return batch
