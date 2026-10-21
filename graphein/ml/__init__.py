# graphein/ml/__init__.py
from .clustering import *
from .conversion import GraphFormatConvertor
from .utils import add_labels_to_graph

InMemoryProteinGraphDataset = None
ProteinGraphDataset = None
ProteinGraphListDataset = None

try:
    from .datasets import (
        InMemoryProteinGraphDataset,
        ProteinGraphDataset,
        ProteinGraphListDataset,
    )
except (ImportError, NameError):
    pass

try:
    from .visualisation import *
except (ImportError, NameError):
    pass
