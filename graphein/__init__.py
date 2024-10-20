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
__version__ = "1.7.7"


logger.configure(
    handlers=[
        {"sink": RichHandler(rich_tracebacks=True), "format": "{message}"}
    ]
)

logger.disable("graphein")


def verbose(enabled: bool = False):
    """Enable/Disable logging.

    :param enabled: Whether or not to enable logging, defaults to ``False``.
    :type enabled: bool, optional
    """
    if not enabled:
        logger.disable("graphein")
    else:
        logger.enable("graphein")
