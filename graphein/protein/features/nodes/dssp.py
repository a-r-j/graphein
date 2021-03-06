from typing import Any, Dict, Optional

import networkx as nx
import pandas as pd
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, residue_max_acc

DSSP_COLS = [
    "chain",
    "resnum",
    "icode",
    "aa",
    "ss",
    "exposure_rsa",
    "phi",
    "psi",
    "dssp_index",
    "NH_O_1_relidx",
    "NH_O_1_energy",
    "O_NH_1_relidx",
    "O_NH_1_energy",
    "NH_O_2_relidx",
    "NH_O_2_energy",
    "O_NH_2_relidx",
    "O_NH_2_energy",
]


def parse_dssp_df(dssp: Dict[str, Any]) -> pd.DataFrame:
    # Parse DSSP output to DataFrame
    appender = []
    for k in dssp[1]:
        to_append = []
        y = dssp[0][k]
        chain = k[0]
        residue = k[1]
        het = residue[0]
        resnum = residue[1]
        icode = residue[2]
        to_append.extend([chain, resnum, icode])
        to_append.extend(y)
        appender.append(to_append)

    return pd.DataFrame.from_records(appender, columns=DSSP_COLS)


def add_dssp_df(G: nx.Graph) -> nx.Graph:
    raise NotImplementedError


def process_dssp_df(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def add_dssp_features(G: nx.Graph) -> nx.Graph:
    if G.graph["pdb_code"] is not None:
        raise NotImplementedError
        # d = dssp_dict_from_pdb_file(pdb_code + ".pdb") # Todo fix paths
    elif G.graph["file_path"] is not None:
        d = dssp_dict_from_pdb_file(G.graph["file_path"])

    dssp_df = parse_dssp_df(d)
    dssp_df = process_dssp_df(d)

    # Assign features
    G.graph["dssp_secondary_structure"] = dssp_df["ss"]
    G.graph["dssp_exposure_rsa"] = dssp_df["exposure_rsa"]
    G.graph["dssp_exposure_asa"] = dssp_df["exposure_asa"]
    return G
