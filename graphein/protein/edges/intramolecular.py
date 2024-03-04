"""Featurization functions for graph edges."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd
from loguru import logger as log

from graphein.protein.utils import download_pdb


def peptide_bonds(G: nx.Graph) -> nx.Graph:
    """
    Adds peptide backbone to residues in each chain.

    :param G: nx.Graph protein graph.
    :type G: nx.Graph
    :returns: nx.Graph protein graph with added peptide bonds.
    :rtype: nx.Graph
    """
    log.debug("Adding peptide bonds to graph")
    # Iterate over every chain
    for chain_id in G.graph["chain_ids"]:
        # Find chain residues
        chain_residues = [
            (n, v) for n, v in G.nodes(data=True) if v["chain_id"] == chain_id
        ]

        # Iterate over every residue in chain
        for i, residue in enumerate(chain_residues):
            # Checks not at chain terminus - is this versatile enough?
            if i == len(chain_residues) - 1:
                continue
            # Asserts residues are on the same chain
            cond_1 = (
                residue[1]["chain_id"] == chain_residues[i + 1][1]["chain_id"]
            )
            # Asserts residue numbers are adjacent
            cond_2 = (
                abs(
                    residue[1]["residue_number"]
                    - chain_residues[i + 1][1]["residue_number"]
                )
                == 1
            )

            # If this checks out, we add a peptide bond
            if cond_1 and cond_2:
                # Adds "peptide_bond" between current residue and the next
                if G.has_edge(i, i + 1):
                    G.edges[i, i + 1]["kind"].add("peptide_bond")
                else:
                    G.add_edge(
                        residue[0],
                        chain_residues[i + 1][0],
                        kind={"peptide_bond"},
                    )
            else:
                continue
    return G


####################################
#                                  #
#     GetContacts Interactions     #
#                                  #
####################################


def get_contacts_df(config: GetContactsConfig, pdb_name: str) -> pd.DataFrame:
    """
    Reads GetContact File and returns it as a ``pd.DataFrame``.

    :param config: GetContactsConfig object
    :type config: GetContactsConfig
    :param pdb_name: Name of PDB file. Output contacts files are named:
        ``{pdb_name}_contacts.tsv``.
    :type pdb_name: str
    :return: DataFrame of parsed GetContacts output.
    :rtype: pd.DataFrame
    """
    if not config.contacts_dir:
        config.contacts_dir = Path("/tmp/")

    contacts_file = config.contacts_dir / f"{pdb_name}_contacts.tsv"

    # Check for existence of GetContacts file
    if not os.path.isfile(contacts_file):
        log.info("GetContacts file not found. Running GetContacts...")
        run_get_contacts(config, pdb_name)

    contacts_df = read_contacts_file(config, contacts_file)

    # remove temp GetContacts file
    if config.contacts_dir == "/tmp/":
        os.remove(contacts_file)

    return contacts_df


def run_get_contacts(
    config: GetContactsConfig,
    pdb_id: Optional[str] = None,
    file_name: Optional[str] = None,
):
    """
    Runs GetContacts on a protein structure. If no file_name is provided, a
    PDB file is downloaded for the pdb_id

    :param config: GetContactsConfig object containing GetContacts parameters
    :type config: graphein.protein.config.GetContactsConfig
    :param pdb_id: 4-character PDB accession code
    :type pdb_id: str, optional
    :param file_name: PDB_name file to use, if annotations to be retrieved from
        the PDB
    :type file_name: str, optional
    """
    # Check for GetContacts Installation
    assert os.path.isfile(
        f"{config.get_contacts_path}/get_static_contacts.py"
    ), "No GetContacts Installation Detected. Please install from: \
    https://getcontacts.github.io"

    # Check for existence of pdb file. If not, download it.
    if not os.path.isfile(config.pdb_dir / file_name):
        log.debug(
            f"No pdb file found for {config.pdb_dir / file_name}. \
            Checking pdb_id..."
        )
        if not os.path.isfile(config.pdb_dir / pdb_id):
            log.debug(
                f"No pdb file found for {config.pdb_dir / pdb_id}. \
                Downloading..."
            )
            pdb_file = download_pdb(pdb_code=pdb_id, out_dir=config.pdb_dir)
        else:
            pdb_file = config.pdb_dir + pdb_id + ".pdb"

    # Run GetContacts
    command = f"{config.get_contacts_path}/get_static_contacts.py "
    command += f"--structure {pdb_file} "
    command += f'--output {(config.contacts_dir / (pdb_id + "_contacts.tsv")).as_posix()} '
    command += "--itypes all"  # --sele "protein"'

    log.info(f"Running GetContacts with command: {command}")

    subprocess.run(command, shell=True)

    # Check it all checks out
    assert os.path.isfile(config.contacts_dir / (pdb_id + "_contacts.tsv"))
    log.info(f"Computed Contacts for: {pdb_id}")


def read_contacts_file(
    config: GetContactsConfig, contacts_file: str
) -> pd.DataFrame:
    """
    Parses GetContacts file to an edge list (pd.DataFrame).

    :param config: GetContactsConfig object
        (:ref:`~graphein.protein.config.GetContactsConfig`)
    :type config: GetContactsConfig
    :param contacts_file: file name of contacts file
    :type contacts_file: str
    :return: Pandas Dataframe of edge list
    :rtype: pd.DataFrame
    """
    log.debug(f"Parsing GetContacts output file at: {contacts_file}")

    contacts_file = open(contacts_file, "r").readlines()
    contacts = []

    # Extract every contact and residue types
    for contact in contacts_file[2:]:
        contact = contact.strip().split("\t")

        interaction_type = contact[1]
        res1 = contact[2]
        res2 = contact[3]

        # Remove atom names if not using atom granularity
        if config.granularity != "atom":
            res1 = res1.split(":")
            res2 = res2.split(":")

            res1 = res1[0] + ":" + res1[1] + ":" + res1[2]
            res2 = res2[0] + ":" + res2[1] + ":" + res2[2]

        contacts.append([res1, res2, interaction_type])

    edges = pd.DataFrame(
        contacts, columns=["res1", "res2", "interaction_type"]
    )

    return edges.drop_duplicates()


def add_contacts_edge(G: nx.Graph, interaction_type: str) -> nx.Graph:
    """
    Adds specific interaction types to the protein graph.

    :param G: Networkx protein graph.
    :type G: nx.Graph
    :param interaction_type: Interaction type to be added.
    :type interaction_type: str
    :return G: nx.Graph with specified interaction-based edges added.
    :rtype: nx.Graph
    """
    log.debug(f"Adding {interaction_type} edges to graph")

    if "contacts_df" not in G.graph:
        log.info("No 'contacts_df' found in G.graph. Running GetContacts.")

        G.graph["contacts_df"] = get_contacts_df(
            G.graph["config"].get_contacts_config, G.graph["pdb_id"]
        )

    contacts = G.graph["contacts_df"]

    # Select specific interaction type
    interactions = contacts.loc[
        contacts["interaction_type"] == interaction_type
    ]

    for label, [res1, res2, interaction_type] in interactions.iterrows():
        # Check residues are actually in graph
        if not (G.has_node(res1) and G.has_node(res2)):
            continue

        if G.has_edge(res1, res2):
            G.edges[res1, res2]["kind"].add(interaction_type)
        else:
            G.add_edge(res1, res2, kind={interaction_type})

    return G


def hydrogen_bond(G: nx.Graph) -> nx.Graph:
    """
    Adds hydrogen bonds to protein structure graph.

    :param G: nx.Graph to add hydrogen bonds to.
    :type G: nx.Graph
    :return: nx.Graph with hydrogen bonds added.
    :rtype: nx.Graph
    """
    return add_contacts_edge(G, "hb")


def salt_bridge(G: nx.Graph) -> nx.Graph:
    """
    Adds salt bridges to protein structure graph.

    :param G: nx.Graph to add salt bridges to.
    :type G: nx.Graph
    :return: nx.Graph with salt bridges added.
    :rtype: nx.Graph
    """
    return add_contacts_edge(G, "sb")


def pi_cation(G: nx.Graph) -> nx.Graph:
    """
    Adds pi-cation interactions to protein structure graph.

    :param G: nx.Graph to add pi-cation interactions to.
    :type G: nx.Graph
    :return: nx.Graph with pi-pi_cation interactions added.
    :rtype: nx.Graph
    """

    return add_contacts_edge(G, "pc")


def pi_stacking(G: nx.Graph) -> nx.Graph:
    """
    Adds pi-stacking interactions to protein structure graph

    :param G: nx.Graph to add pi-stacking interactions to
    :type G: nx.Graph
    :return: nx.Graph with pi-stacking interactions added
    :rtype: nx.Graph
    """
    return add_contacts_edge(G, "ps")


def t_stacking(G: nx.Graph) -> nx.Graph:
    """
    Adds t-stacking interactions to protein structure graph.

    :param G: nx.Graph to add t-stacking interactions to.
    :type G: nx.Graph
    :return: nx.Graph with t-stacking interactions added.
    :rtype: nx.Graph
    """
    return add_contacts_edge(G, "ts")


def hydrophobic(G: nx.Graph) -> nx.Graph:
    """
    Adds hydrophobic interactions to protein structure graph.

    :param G: nx.Graph to add hydrophobic interaction edges to.
    :type G: nx.Graph
    :return: nx.Graph with hydrophobic interactions added.
    :rtype: nx.Graph
    """
    return add_contacts_edge(G, "hp")


def van_der_waals(G: nx.Graph) -> nx.Graph:
    """
    Adds van der Waals interactions to protein structure graph.

    :param G: nx.Graph to add van der Waals interactions to.
    :type G: nx.Graph
    :return: nx.Graph with van der Waals interactions added.
    :rtype: nx.Graph
    """
    return add_contacts_edge(G, "vdw")
