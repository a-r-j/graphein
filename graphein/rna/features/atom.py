"""Functions for featurising RNA Structure Graphs."""

# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from typing import Any, Dict

from graphein.rna.constants import RNA_ATOM_BOND_STATE, RNA_ATOMIC_RADII


def add_atomic_radii(n: str, d: Dict[str, Any]) -> float:
    """Add atomic radii to nodes based on values provided in:

    Structures of the Molecular Components in DNA and RNA with Bond
    Lengths Interpreted as Sums of Atomic Covalent Radii
    Raji Heyrovska

    Atoms in the RNA structure are mapped to their bond states
    (:const:`~graphein.rna.constants.RNA_ATOM_BOND_STATE`),
    which are in turn mapped to the corresponding atomic radii
    (:const:`~graphein.rna.constants.RNA_ATOMIC_RADII`).

    :param n: The node to add the atomic radius to.
        Unused, the argument is provided to retain a consistent function
        signature.
    :type n: str
    :param d: The node data.
    :type d: Dict[str, Any]
    :return: The atomic radius of the node.
    :rtype: float
    """

    base = d["residue_name"]
    atom = d["atom_type"]
    radius = RNA_ATOMIC_RADII[RNA_ATOM_BOND_STATE[base][atom]]
    d["atomic_radius"] = radius
    return radius
