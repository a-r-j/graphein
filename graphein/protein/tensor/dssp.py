"""DSSP secondary structure, H-bonds and solvent accessibility for tensors.

The implementation lives in :mod:`graphein.protein.dssp_core` (which does
not require PyTorch, so it also backs the ``networkx`` DSSP features); this
module re-exports it for the tensor API.
"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from graphein.protein.dssp_core import (  # noqa: F401
    DSSP8_STATES,
    accessibility,
    accessibility_np,
    assign_dssp,
    assign_dssp8,
    assign_dssp8_np,
    assign_dssp_np,
    get_hbond_map,
    get_hbond_map_np,
)
