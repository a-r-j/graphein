"""Utilities for retrieving molecular data from ZINC.

Adapted from smilite (https://github.com/rasbt/smilite) by Sebastian Raschka.
"""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT

# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import ssl
from typing import Dict, List
from urllib.error import URLError

from tqdm.rich import tqdm

from graphein.utils.dependencies import import_message

try:
    from smilite import *
except ImportError:
    import_message(
        "graphein.molecule.zinc", "smilite", "smilite", True, extras=True
    )


def get_smiles_from_zinc(zinc_id: str, backend="zinc15") -> str:
    """
    Gets the corresponding SMILE string for a ZINC ID query from
    the ZINC online database (https://zinc.docking.org/). Requires an internet connection.

    :param zinc_id: A valid ZINC ID, e.g. ``'ZINC00029323'``
    :type zinc_id: str
    :param backend: ``zinc12`` or ``zinc15``
    :type backend: str
    :returns: the SMILE string for the corresponding ZINC ID.
        E.g., ``'COc1cccc(c1)NC(=O)c2cccnc2'``
    :rtype: str
    """
    try:
        return smilite.get_zinc_smile(zinc_id, backend=backend)
    except URLError:
        ssl._create_default_https_context = ssl._create_unverified_context
        return smilite.get_zinc_smile(zinc_id, backend=backend)


def get_zinc_id_from_smile(smile: str, backend: str = "zinc15") -> List[str]:
    """
    Gets the corresponding ZINC ID(s) for a SMILE string query from
    the ZINC online database. Requires an internet connection.

    :param smile_str: A valid SMILE string, e.g.,
            ``C[C@H]1CCCC[NH+]1CC#CC(c2ccccc2)(c3ccccc3)O'``
    :type smile_str: str
    :param backend:  Specifies the database backend, ``"zinc12"`` or ``"zinc15"``.
    :type backend: str
    :returns: the SMILE string for the corresponding ZINC ID(s) in a list.
        E.g., ``['ZINC01234567', 'ZINC01234568', 'ZINC01242053', 'ZINC01242055']``
    :rtype: List[str]
    """
    try:
        return smilite.get_zincid_from_smile(smile, backend=backend)
    except URLError:
        ssl._create_default_https_context = ssl._create_unverified_context
        return smilite.get_zincid_from_smile(smile, backend=backend)


def batch_get_smiles_from_zinc(
    zinc_ids: List[str], backend: str = "zinc15"
) -> Dict[str, str]:
    """Gets the corresponding SMILE string(s) for a list of ZINC IDs.

    :param zinc_ids: List of ZINC IDs, e.g., ``['ZINC00029323', 'ZINC00029324']``
    :type zinc_ids: List[str]
    :return: _description_
    :rtype: Dict[str, str]
    """
    return {
        zinc_id: get_smiles_from_zinc(zinc_id, backend=backend)
        for zinc_id in tqdm(zinc_ids)
    }


def batch_get_zinc_id_from_smiles(
    smiles: List[str], backend: str = "zinc15"
) -> Dict[str, List[str]]:
    """
    Gets the corresponding ZINC ID for a list of smile string queries from
    the ZINC online database. Requires an internet connection.

    :param smile_str: A list of valid SMILE string, e.g.,
        ``["C[C@H]1CCCC[NH+]1CC#CC(c2ccccc2)(c3ccccc3)O", "CCC"]``
    :type smile_str: str
    :param backend:  Specifies the database backend, ``"zinc12"`` or ``"zinc15"``.
    :type backend: str
    :returns: the SMILE string for the corresponding ZINC ID(s) in a list.
    :rtype: Dict[str, List[str]]
    """
    return {
        s: get_zinc_id_from_smile(s, backend=backend) for s in tqdm(smiles)
    }
