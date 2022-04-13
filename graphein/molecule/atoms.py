"""
Author: Eric J. Ma, Arian Jamasb
Purpose: This is a set of utility variables and functions that can be used
across the Graphein project.

These include various collections of standard & non-standard/modified amino acids and their names, identifiers and properties.

We also include mappings of covalent radii and bond lengths for the amino acids used in assembling atomic protein graphs.
"""
# Graphein
# Author: Eric J. Ma, Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein


from typing import Dict, List

import numpy as np

BASE_ATOMS: List[str] = [
    "C",
    "H",
    "O",
    "N",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
]
"""Vocabulary of 11 standard atom types."""