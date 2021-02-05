# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Ramon Vinas
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class TRRUSTConfig(BaseModel):
    pass


class RegNetworkConfig(BaseModel):
    pass


class GRNGraphConfig(BaseModel):
    kwargs: Optional[Dict[str, Union[str, int, float]]] = {}
    trrust_config: Optional[TRRUSTConfig] = None
    regnetwork_config: Optional[RegNetworkConfig] = None

    class Config:
        arbitrary_types_allowed: bool = True
