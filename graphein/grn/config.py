# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Ramon Vinas
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from pydantic import BaseModel


class TRRUSTConfig(BaseModel):
    filtering_functions: Optional[List[Callable]] = None
    root_dir: Optional[Path] = None
    kwargs: Optional[Dict[str, Union[str, int, float]]] = None


class RegNetworkConfig(BaseModel):
    filtering_functions: Optional[List[Callable]] = None
    root_dir: Optional[Path] = None
    kwargs: Optional[Dict[str, Union[str, int, float]]] = None


class GRNGraphConfig(BaseModel):
    kwargs: Optional[Dict[str, Union[str, int, float]]] = {}
    trrust_config: Optional[TRRUSTConfig] = TRRUSTConfig()
    regnetwork_config: Optional[RegNetworkConfig] = RegNetworkConfig()

    class Config:
        arbitrary_types_allowed: bool = True
