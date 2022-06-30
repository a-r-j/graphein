# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: BSD 3 clause
# Code Repository: https://github.com/a-r-j/graphein
from loguru import logger
from rich.logging import RichHandler

from graphein.utils.utils import *

# from .protein import *
# from .rna import *
from .testing import *

__author__ = "Arian Jamasb <arian@jamasb.io>"
__version__ = "1.5.0rc1"


logger.configure(
    handlers=[
        {"sink": RichHandler(rich_tracebacks=True), "format": "{message}"}
    ]
)
