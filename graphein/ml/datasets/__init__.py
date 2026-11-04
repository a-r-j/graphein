from .pdb_data import PDBManager

InMemoryProteinGraphDataset = None
ProteinGraphDataset = None
ProteinGraphListDataset = None

try:
    from .torch_geometric_dataset import (
        InMemoryProteinGraphDataset,
        ProteinGraphDataset,
        ProteinGraphListDataset,
    )
except (NameError, ImportError):
    pass
