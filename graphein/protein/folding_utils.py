"""Utilities for using structure prediction models.

N.B. ESMFold functionality require torch & ESMFold to be installed.

.. code-block:: bash
    pip install "fair-esm[esmfold]"
    # OpenFold and its remaining dependency
    pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
    pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

"""
from functools import lru_cache
from typing import Dict, Optional

import requests
from loguru import logger as log

from graphein.utils.utils import import_message

try:
    import torch
except ImportError:
    message = import_message(
        "graphein.protein.folding_utils",
        package="torch",
        conda_channel="pytorch",
        pip_install=True,
    )
    log.warning(message)


try:
    import esm
except ImportError:
    message = import_message(
        "graphein.protein.folding_utils",
        package="fair-esm[esmfold]",
        pip_install=True,
    )
    log.warning(message)


@lru_cache
def _get_model() -> torch.nn.Module:
    """Loads the ESMFold model."""
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    return model


def esmfold(sequence: str, out_path: str):
    """Fold a protein sequence using the ESMFold model.

    Multimer prediction can be done with chains separated by ``:``.

    :param sequence: Amino acid sequence in one-letter code.
    :type sequence: str
    :param out_path: Path to save the PDB file to.
    :type out_path: str
    """

    model = _get_model()

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(out_path, "w") as f:
        f.write(output)


def esmfold_web(
    sequence: str, out_path: Optional[str] = None, version: int = 1
):
    """Fold a protein sequence using the ESMFold model from the ESMFold server at
    https://api.esmatlas.com/foldSequence/v1/pdb/.

    Parameters
    ----------
    sequence : str
        A protein sequence in one-letter code.
    out_path : str, optional
        Path to save the PDB file to. If `None`, the file is not saved.
        Defaults to `None`.
    version : int, optional
        The version of the ESMFold model to use. Defaults to `1`.
    """
    URL = f"https://api.esmatlas.com/foldSequence/v{version}/pdb/"

    headers: Dict[str, str] = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    cif = requests.post(URL, data=sequence, headers=headers).text
    # append header
    header = "\n".join(
        [f"data_{sequence}", "#", f"_entry.id\t{sequence}", "#\n"]
    )
    cif = header + cif
    if out_path is not None:
        with open(out_path, "w") as f:
            f.write(cif)
