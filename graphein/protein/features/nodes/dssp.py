from typing import Any, Dict, Optional

import os
import networkx as nx
import pandas as pd
from Bio.PDB.DSSP import dssp_dict_from_pdb_file, residue_max_acc
from Bio.Data.IUPACData import protein_letters_1to3

from graphein.protein.utils import download_pdb


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


def process_dssp_df(df: pd.DataFrame) -> pd.DataFrame:

    # Convert 1 letter aa code to 3 letter
    amino_acids = df["aa"].tolist()

    for i, amino_acid in enumerate(amino_acids):
        amino_acids[i] = protein_letters_1to3[amino_acid].upper()
    df["aa"] = amino_acids

    # Construct node IDs

    node_ids = []

    for i, row in df.iterrows():
        node_id = row["chain"] + ":" + row["aa"] + ":" + str(row["resnum"])
        node_ids.append(node_id)
    df["node_id"] = node_ids

    df.set_index("node_id", inplace=True)

    return df



def add_dssp_feature(G: nx.Graph, feature: str) -> nx.Graph:
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

def add_dssp_df(G: nx.Graph) -> nx.Graph:

    config = G.graph["config"]
    pdb_id = G.graph["pdb_id"]

    # To add - Check for DSSP installation

    # Check for existence of pdb file. If not, download it.
    if not os.path.isfile(config.pdb_dir / pdb_id):
        pdb_file = download_pdb(config, pdb_id)
    else:
        pdb_file = config.pdb_dir + pdb_id + ".pdb"

    # Todo - add executable from config
    dssp_dict = dssp_dict_from_pdb_file(pdb_file, DSSP="mkdssp")
    dssp_dict = parse_dssp_df(dssp_dict)
    dssp_dict = process_dssp_df(dssp_dict)

    G.graph["dssp_df"] = dssp_dict

    print(G.graph["dssp_df"])
    return G
