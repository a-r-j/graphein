from graphein.utils.utils import *

from ._version import get_versions

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: BSD 3 clause
# Code Repository: https://github.com/a-r-j/graphein
from .protein import *
from .rna import *

__author__ = "Arian Jamasb <arian@jamasb.io>"


__version__ = "1.0.0"  # get_versions()["version"]
del get_versions
