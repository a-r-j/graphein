# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Ramon Vinas
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel


class TRRUSTConfig(BaseModel):
    """
    Config object for parsing gene regulatory networks from TRRUST: https://www.grnpedia.org/trrust/

    :param filtering_functions: List of functions to apply to the the TRRUST dataframe prior to graph construction. Defaults to None
    :type filtering_functions: List[Callable], optional
    :param root_dir: Specifies location of TRRUST dataset (will download to this path if not available). Defaults to None.
    :type root_dir: pathlib.Path, optional
    :param kwargs:
    :type kwargs: Dict[str, Union[str, int, float]], optional
    """

    filtering_functions: list[Callable] | None = None
    root_dir: Path | None = None
    kwargs: dict[str, str | int | float] | None = None


class RegNetworkConfig(BaseModel):
    """Config object containing parameters for parsing gene regulatory networks from RegNetwork: http://regnetworkweb.org/.

    :param filtering_functions: List of functions to apply to the the RegNetwork dataframe prior to graph construction. Defaults to None
    :type filtering_functions: List[Callable], optional


    """

    filtering_functions: list[Callable] | None = None
    root_dir: Path | None = None
    kwargs: dict[str, str | int | float] | None = None


class GRNGraphConfig(BaseModel):
    """Config object for gene regulatory network graph construction.

    :param kwargs: Keyword args for GRN graph construction
    :type kwargs: Dict[str, Union[str, int, float]], optional
    :param trrust_config: Config object specifying parameters for parsing TRRUST. Defaults to default config object.
    :type trrust_config: graphein.grn.config.TRRUSTConfig, optional
    :param regnetwork_config: Config object specifying parameters for parsing RegNetwork. Defaults to default config object.
    :type regnetwork_config: graphein.grn.config.RegNetworkConfig, optional
    """

    kwargs: dict[str, str | int | float] | None = {}
    trrust_config: TRRUSTConfig | None = TRRUSTConfig()
    regnetwork_config: RegNetworkConfig | None = RegNetworkConfig()

    class Config:
        arbitrary_types_allowed: bool = True
