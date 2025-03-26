"""Provides utility functions for use across Graphein."""
import logging

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import os
from functools import lru_cache
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Tuple, Union
from urllib.request import urlopen

import networkx as nx
import numpy as np
import pandas as pd
import wget
from Bio.PDB import PDBList
from biopandas.pdb import PandasPdb

log = logging.getLogger(__name__)


class ProteinLigandGraphConfigurationError(Exception):
    """Exception when an invalid Graph configuration if provided to a downstream function or method."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message
