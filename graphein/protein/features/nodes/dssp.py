from typing import Dict, Any
import pandas as pd
import networkx as nx
from Bio.PDB.DSSP import residue_max_acc, dssp_dict_from_pdb_file

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
    G


def process_dssp_df(df: pd.DataFrame) -> pd.DataFrame:


def add_dssp_features(G: nx.Graph) -> nx.Graph:
    if G.graph["pdb_code"] is not None:
        d = dssp_dict_from_pdb_file(pdb_code + ".pdb") # Todo fix paths
    elif G.graph["file_path"] is not None:
        d = dssp_dict_from_pdb_file(G.graph["file_path"])
    
    dssp_df = parse_dssp_df(d)
    dssp_df = process_dssp_df(d)

    # Assign features
    G.graph["dssp_secondary_structure"] = dssp_df["ss"]
    G.graph["dssp_exposure_rsa"] = dssp_df["exposure_rsa"]
    G.graph["dssp_exposure_asa"] = dssp_df["exposure_asa"]
    return G


def _get_protein_features(
        self, pdb_code: Optional[str], file_path: Optional[str], chain_selection: str
) -> pd.DataFrame:
    """
    :param file_path: (str) file path to PDB file
    :param pdb_code: (str) String containing four letter PDB accession
    :return df (pd.DataFrame): Dataframe containing output of DSSP (Solvent accessibility, secondary structure for each residue)
    """

    # Run DSSP on relevant PDB file
    if pdb_code:
        d = dssp_dict_from_pdb_file(self.pdb_dir + pdb_code + ".pdb")
    if file_path:
        d = dssp_dict_from_pdb_file(file_path)

    # Subset dataframe to those in chain_selection
    if chain_selection != "all":
        df = df.loc[df["chain"].isin(chain_selection)]
    # Rename cysteines to 'C'
    df["aa"] = df["aa"].str.replace("[a-z]", "C")
    df = df[df["aa"].isin(list(aa1))]

    # Drop alt_loc residues
    df = df.loc[df["icode"] == " "]

    # Add additional Columns
    df["aa_three"] = df["aa"].apply(one_to_three)
    df["max_acc"] = df["aa_three"].map(residue_max_acc["Sander"].get)
    df[["exposure_rsa", "max_acc"]] = df[["exposure_rsa", "max_acc"]].astype(float)
    df["exposure_asa"] = df["exposure_rsa"] * df["max_acc"]
    df["index"] = df["chain"] + ":" + df["aa_three"] + ":" + df["resnum"].apply(str)
    return df
