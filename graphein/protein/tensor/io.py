"""Utilities for parsing proteins into and out of tensors."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import collections
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from loguru import logger as log

from graphein.utils.dependencies import import_message

from ..graphs import (
    deprotonate_structure,
    filter_hetatms,
    read_pdb_to_dataframe,
    remove_insertions,
    select_chains,
    sort_dataframe,
)
from ..resi_atoms import (
    ATOM_NUMBERING,
    ELEMENT_SYMBOL_MAP,
    PROTEIN_ATOMS,
    STANDARD_AMINO_ACID_MAPPING_1_TO_3,
    STANDARD_AMINO_ACIDS,
)
from .representation import get_full_atom_coords
from .sequence import (
    get_residue_id,
    get_sequence,
    infer_residue_types,
    residue_type_tensor,
)
from .types import AtomTensor

try:
    from torch_geometric.data import Data
except ImportError:
    message = import_message(
        submodule="graphein.protein.tensor.io",
        package="torch_geometric",
        conda_channel="pyg",
        pip_install=True,
    )
    log.warning(message)

try:
    import torch
except ImportError:
    message = import_message(
        submodule="graphein.protein.tensor.io",
        package="torch",
        conda_channel="pytorch",
        pip_install=True,
    )
    log.warning(message)


def get_protein_length(df: pd.DataFrame, insertions: bool = True) -> int:
    """Return the number of unique amino acids in the protein.

    Note for future development: this function may return incorrect results
    on raw PDB files containing insertions/altlocs. This should not be the case
    for proteins processed by the Graphein pre-processing workflow. Caveat
    emptor.

    :param df: Protein DataFrame to get length of.
    :type df: pd.DataFrame
    :param insertions: Whether or not to include insertions in the length.
        Defaults to ``True``.
    :type insertions: bool
    :return: Number of unique residues.
    :rtype: int
    """
    # Get unique residues:
    if "residue_id" not in df.columns:
        df["residue_id"] = (
            df["chain_id"]
            + ":"
            + df["residue_name"]
            + ":"
            + df["residue_number"].astype(str)
        )
        if insertions:
            df["residue_id"] = df.residue_id + ":" + df.insertion
    ids = df["residue_id"].values

    return len(set(ids))


def protein_to_pyg(
    path: Optional[Union[str, os.PathLike]] = None,
    pdb_code: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    chain_selection: Union[str, List[str]] = "all",
    deprotonate: bool = True,
    keep_insertions: bool = True,
    keep_hets: List[str] = [],
    model_index: int = 1,
    atom_types: List[str] = PROTEIN_ATOMS,
    remove_nonstandard: bool = True,
    store_het: bool = False,
    store_bfactor: bool = False,
    fill_value_coords: float = 1e-5,
) -> Data:
    """
    Parses a protein (from either: a PDB code, PDB file or a UniProt ID
    (via AF2 database) to a PyTorch Geometric ``Data`` object.


    .. code-block:: python

        import graphein.protein.tensor as gpt

        # From PDB code
        gpt.io.protein_to_pyg(pdb_code="3eiy")

        # From PDB Path
        gpt.io.protein_to_pyg(path="3eiy.pdb")

        # From MMTF Path
        gpt.io.protein_to_pyg(path="3eiy.mmtf")

        # From UniProt ID
        gpt.io.protein_to_pyg(uniprot_id="Q5VSL9")


    :param path: Path to PDB or MMTF file. Default is ``None``.
    :type path: Union[str, os.PathLike]
    :param pdb_code: PDB accesion code. Default is ``None``.
    :type pdb_code: str
    :param uniprot_id: UniProt ID. Default is ``None``.
    :type uniprot_id: str
    :param chain_selection: Selection of chains to include (e.g.
        ``["A", "C", "AB"]``) or ``"all"``. Default is ``"all"``.
    :type chain_selection: Union[str, List[str]]
    :param deprotonate: Whether or not to remove Hydrogens. Default is ``True``.
    :type deprotonate: bool
    :param keep_insertions: Whether or not to keep insertions. Default is
        ``True``.
    :type keep_insertions: bool
    :param keep_hets: List of heteroatoms to include. E.g. ``["HOH"]``.
    :type keep_hets: List[str]
    :param model_index: Index of model in models containing multiple structures.
    :type model_index: int
    :param atom_types: List of atom types to select. Default is:
        :const:`graphein.protein.resi_atoms.PROTEIN_ATOMS`
    :type atom_types: List[str]
    :param remove_nonstandard: Whether or not to remove non-standard residues.
        Default is ``True``.
    :type remove_nonstandard: bool
    :param store_het: Whether or not to store heteroatoms in the ``Data``
        object. Default is ``False``.
    :type store_het: bool
    :param store_bfactor: Whether or not to store bfactors in the ``Data``
        object. Default is ``False.
    :type store_bfactor: bool
    :param fill_value_coords: Fill value to use for positions in atom37
        representation that are not filled. Defaults to 1e-5
    :type fill_value_coords: float
    :returns: ``Data`` object with attributes: ``x`` (AtomTensor), ``residues``
        (list of 3-letter residue codes), id (ID of protein), residue_id (E.g.
        ``"A:SER:1"``), residue_type (torch.Tensor), ``chains`` (torch.Tensor).
    :rtype: torch_geometric.data.Data
    """

    # Get ID
    if path is not None:
        id = (
            os.path.splitext(path)[0].split("/")[-1]
            + "_"
            + "".join(chain_selection)
            if chain_selection != "all"
            else os.path.splitext(path)[0].split("/")[-1]
        )
    elif pdb_code is not None:
        id = (
            pdb_code + "_" + "".join(chain_selection)
            if chain_selection != "all"
            else pdb_code
        )
    elif uniprot_id is not None:
        id = (
            uniprot_id + "_" + "".join(chain_selection)
            if chain_selection != "all"
            else uniprot_id
        )
    else:
        id = None

    if df is None:
        df = read_pdb_to_dataframe(
            path=path,
            pdb_code=pdb_code,
            uniprot_id=uniprot_id,
            model_index=model_index,
        )
    if chain_selection != "all":
        if isinstance(chain_selection, str):
            chain_selection = [chain_selection]
        df = select_chains(df, chain_selection)

    if deprotonate:
        df = deprotonate_structure(df)
    if not keep_insertions:
        df = remove_insertions(df)
    # Remove hetatms
    hets = filter_hetatms(df, keep_hets=keep_hets)

    if store_het:
        hetatms = df.loc[df.record_name == "HETATM"]
        all_hets = list(set(hetatms.residue_name))
        het_data = collections.defaultdict(dict)
        for het in all_hets:
            het_data[het]["coords"] = torch.tensor(
                hetatms.loc[hetatms.residue_name == het][
                    ["x_coord", "y_coord", "z_coord"]
                ].values
            )
            het_data[het]["atoms"] = hetatms.loc[hetatms.residue_name == het][
                "atom_name"
            ].values
            het_data[het]["residue_number"] = torch.tensor(
                hetatms.loc[hetatms.residue_name == het][
                    "residue_number"
                ].values
            )
            het_data[het]["element_symbol"] = hetatms.loc[
                hetatms.residue_name == het
            ]["element_symbol"].values

    df = df.loc[df.record_name == "ATOM"]
    if remove_nonstandard:
        df = df.loc[
            df.residue_name.isin(STANDARD_AMINO_ACID_MAPPING_1_TO_3.values())
        ]
    df = pd.concat([df] + hets)
    df = sort_dataframe(df)

    df["residue_id"] = (
        df["chain_id"]
        + ":"
        + df["residue_name"]
        + ":"
        + df["residue_number"].astype(str)
    )
    if keep_insertions:
        df["residue_id"] = df.residue_id + ":" + df.insertion

    out = Data(
        coords=protein_df_to_tensor(
            df,
            atoms_to_keep=atom_types,
            fill_value=fill_value_coords,
        ),
        residues=get_sequence(
            df,
            chains=chain_selection,
            insertions=keep_insertions,
            list_of_three=True,
        ),
        id=id,
        residue_id=get_residue_id(df),
        residue_type=residue_type_tensor(df),
        chains=protein_df_to_chain_tensor(df),
    )

    if store_het:
        out.hetatms = [het_data]

    if store_bfactor:
        # group by residue_id and average b_factor per residue
        residue_bfactors = df.groupby("residue_id")["b_factor"].mean(
            numeric_only=True
        )
        out.bfactor = torch.from_numpy(residue_bfactors.values)

    return out


def protein_df_to_chain_tensor(
    df: pd.DataFrame,
    chains_to_keep: Optional[List[str]] = None,
    insertions: bool = True,
    one_hot: bool = False,
    dtype: torch.dtype = torch.int64,
    device: torch.device = torch.device("cpu"),
    per_atom: bool = False,
) -> torch.Tensor:
    """Returns a tensor of chain IDs for a protein structure.

    :param df: DataFrame of protein structure. Must have a column called
        ``"chain_id"`` (and ``insertion`` if the ``insertions=True``).
    :type df: pd.DataFrame
    :param chains_to_keep: List of chains to retain, defaults to ``None``
        (all chains).
    :type chains_to_keep: Optional[List[str]], optional
    :param insertions: Whether or not to keep insertions, defaults to ``True``
    :type insertions: bool, optional
    :param one_hot: Whether or not to return a one-hot encoded tensor
        (``L x num_chains``). If ``False`` an integer tensor is returned.
        Defaults to ``False``.
    :type one_hot: bool, optional
    :return: One hot encoded or integer tensor indicating chain membership for
        each residue.
    :rtype: torch.Tensor
    """
    if "residue_id" not in df.columns:
        get_residue_id(df, insertions=insertions)

    # Select chains to keep from user input
    if chains_to_keep is not None:
        df = df.loc[df.chain_id.isin(chains_to_keep)]

    # Keep or remove insertions
    if not insertions:
        df = df.loc[df.insertion.isin(["", " "])]

    if not per_atom:
        chains = pd.Series(df.residue_id.unique())
        chains = chains.str.split(":").str[0]
    else:
        chains = df.chain_id

    # One-hot encode chain IDs
    chains = pd.get_dummies(chains)
    chains = torch.tensor(chains.values, dtype=dtype, device=device)

    # Integers instead of one-hot
    if not one_hot:
        chains = torch.argmax(chains, dim=1)

    return chains


def protein_df_to_tensor(
    df: pd.DataFrame,
    atoms_to_keep: List[str] = PROTEIN_ATOMS,
    insertions: bool = True,
    fill_value: float = 1e-5,
) -> AtomTensor:
    """
    Transforms a DataFrame of a protein structure into a
    ``Length x Num_Atoms (default 37) x 3`` tensor.

    :param df: DataFrame of protein structure.
    :type df: pd.DataFrame
    :param atoms_to_keep: List of atom types to retain in the tensor.
        Defaults to :const:`~graphein.protein.resi_atoms.PROTEIN_ATOMS`
    :type atoms_to_keep: List[str]
    :param insertions: Whether or not to keep insertions. Defaults to ``True``.
    :type insertions: bool
    :param fill_value: Value to fill missing entries with. Defaults to ``1e-5``.
    :type fill_value: float
    :returns: ``Length x Num_Atoms (default 37) x 3`` tensor.
    :rtype: graphein.protein.tensor.types.AtomTensor
    """
    num_residues = get_protein_length(df, insertions=insertions)
    df = df.loc[df["atom_name"].isin(atoms_to_keep)]
    residue_indices = pd.factorize(
        pd.Series(get_residue_id(df, unique=False))
    )[0]
    atom_indices = df["atom_name"].map(lambda x: atoms_to_keep.index(x)).values

    positions: AtomTensor = (
        torch.zeros((num_residues, len(atoms_to_keep), 3)) + fill_value
    )
    positions[residue_indices, atom_indices] = torch.tensor(
        df[["x_coord", "y_coord", "z_coord"]].values
    ).float()

    return positions


def to_dataframe(
    x: AtomTensor,
    fill_value: float = 1e-5,
    residue_types: Optional[List[str]] = None,
    chains: Optional[Union[List[Union[str, int]], torch.Tensor]] = None,
    insertions: Optional[List[Union[str, float]]] = None,
    b_factors: Optional[Union[List[float], torch.Tensor]] = None,
    occupancy: Optional[Union[List[float], torch.Tensor]] = None,
    charge: Optional[List[int]] = None,
    alt_loc: Optional[List[Union[str, int]]] = None,
    segment_id: Optional[List[Union[str, int]]] = None,
    biopandas: bool = False,
) -> Union[pd.DataFrame, PandasPdb]:
    """Converts an ``AtomTensor`` to a DataFrame.

    ``AtomTensors`` are not a full specification of a structure so missing
    values can be manually provided as arguments - otherwise default values are
    used.

    .. code-block:: python

        import graphein.protein.tensor as gpt

        protein = gpt.Protein().from_pdb_code("3eiy")
        to_dataframe(protein.coords)

    .. seealso::

        :class:`graphein.protein.tensor.types.AtomTensor`
        :meth:`graphein.protein.tensor.io.to_pdb`
        :meth:`graphein.protein.tensor.sequence.infer_residue_types`


    :param x: AtomTensor to convert (Shape: ``num residues x 37 x 3``).
    :type x: AtomTensor
    :param fill_value: Fill value used to denote missing atoms in the
    ``AtomTensor``, defaults to ``1e-5``.
    :type fill_value: float, optional
    :param residue_types: List of three-letter residue IDs (length: num residues
        ), defaults to ``None`` (inferred from ``AtomTensor``; this may break
        for incomplete structures with missing atoms.)
    :type residue_types: Optional[List[str]], optional
    :param chains: List or tensor of chain IDs, defaults to ``None`` (``"A"``
        for all residues).
    :type chains: Optional[Union[List[Union[str, int]], torch.Tensor]], optional
    :param insertions: List of insertion codes, defaults to ``None`` (``""``).
    :type insertions: Optional[List[Union[str, float]]], optional
    :param b_factors: List or tensor of b factors (length: num residues),
        defaults to ``None`` (``""``). If ``b_factors`` is of length/shape
        number of residues (as opposed to number of atoms) it is automatically
        unravelled to the correct length.
    :type b_factors: Optional[List[Union[str, float]]], optional
    :param occupancy: List or tensor of occupancy values (length: num residues),
        defaults to ``None`` (``1.0``).
    :type occupancy: Optional[List[Union[str, float]]], optional
    :param charge: List or or tensor of atom charges, defaults to ``None``
        (``"0"``).
    :type charge: Optional[List[int]], optional
    :param alt_loc: List or tensor of alt_loc codes, defaults to ``None``
        (``""``)
    :type alt_loc: Optional[List[Union[str, int]]], optional
    :param segment_id: List or tensor of segment IDs, defaults to ``None``
        (``""``).
    :type segment_id: Optional[List[Union[str, int]]], optional
    :param biopandas: Whether to return a ``pd.DataFrame`` or ``BioPandas``
        ``PandasPdb`` object, defaults to ``False`` (``pd.DataFrame``).
    :type biopandas: bool, optional
    :return: ``DataFrame`` or ``PandasPdb object``.
    :rtype: Union[pd.DataFrame, PandasPdb]
    """
    nz = (x - fill_value).nonzero()
    nz = nz[torch.where(nz[:, 2] == 0)]
    res_nums = nz[:, 0] + 1

    atom_number = np.arange(1, len(res_nums) + 1)
    numbering_map = {v: k for k, v in ATOM_NUMBERING.items()}
    atom_type = nz[:, 1]
    atom_type = [numbering_map[a.item()] for a in atom_type]

    if residue_types is None:
        residue_types = infer_residue_types(
            x, fill_value=fill_value, return_sequence=False
        )
    if isinstance(residue_types, torch.Tensor):
        residue_types = [
            STANDARD_AMINO_ACID_MAPPING_1_TO_3[STANDARD_AMINO_ACIDS[a]]
            for a in residue_types
        ]
    residue_types = [residue_types[a - 1] for a in res_nums]
    element_symbols = [ELEMENT_SYMBOL_MAP[a] for a in atom_type]

    chains = ["A"] * len(res_nums) if chains is None else chains[res_nums - 1]
    if b_factors is not None:
        num_b_factors = (
            len(b_factors)
            if isinstance(b_factors, list)
            else b_factors.shape[0]
        )
        b_factors = (
            b_factors[res_nums - 1]
            if num_b_factors == x.shape[0]
            else b_factors
        )
        if isinstance(b_factors, torch.Tensor):
            b_factors = b_factors.tolist()
    else:
        b_factors = [0.0] * len(res_nums)
    if segment_id is None:
        segment_id = [""] * len(res_nums)
    if insertions is None:
        insertions = [""] * len(res_nums)
    if occupancy is None:
        occupancy = [1.0] * len(res_nums)
    if charge is None:
        charge = [0] * len(res_nums)
    if alt_loc is None:
        alt_loc = [""] * len(res_nums)

    blank_1 = [""] * len(res_nums)
    blank_2 = [""] * len(res_nums)
    blank_3 = [""] * len(res_nums)
    blank_4 = [""] * len(res_nums)

    # NB brittle, bad assumption; may break
    record_names = ["ATOM" if i < 37 else "HETATM" for i in nz[:, 1]]
    coords = get_full_atom_coords(x)[0]

    out = {
        "record_name": record_names,
        "atom_number": atom_number,
        "blank_1": blank_1,
        "atom_name": atom_type,
        "alt_loc": alt_loc,
        "residue_name": residue_types,
        "blank_2": blank_2,
        "chain_id": chains,
        "residue_number": res_nums,
        "insertion": insertions,
        "blank_3": blank_3,
        "x_coord": coords[:, 0],
        "y_coord": coords[:, 1],
        "z_coord": coords[:, 2],
        "occupancy": occupancy,
        "b_factor": b_factors,
        "blank_4": blank_4,
        "segment_id": segment_id,
        "element_symbol": element_symbols,
        "charge": charge,
        "line_idx": atom_number,
    }
    df = pd.DataFrame().from_dict(out)
    if biopandas:
        ppdb = PandasPdb()
        ppdb.df["ATOM"] = df
        return ppdb
    return df


def to_pdb(x: AtomTensor, out_path: str, gz: bool = False, **kwargs):
    """
    Writes an ``AtomTensor`` to a PDB file.

    .. seealso::

        :class:`graphein.protein.tensor.types.AtomTensor`
        :func:`graphein.protein.tensor.to_dataframe`

    :param x: ``AtomTensor`` describing protein structure to write.
    :type x: AtomTensor
    :param out_path: Path to output pdb file.
    :type out_path: str
    :param gz: Whether to gzip out the output, defaults to ``False``.
    :type gz: bool, optional
    :param kwargs: Keyword args for :func:`graphein.protein.tensor.to_dataframe`
    """
    df = to_dataframe(x, **kwargs)
    ppdb = PandasPdb()
    ppdb.df["ATOM"] = df
    ppdb.to_pdb(path=out_path, gz=gz, append_newline=True)
