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

from graphein.protein.utils import download_pdb


####################################
#                                  #
#     GetContacts Interactions     #
#                                  #
####################################


def get_contacts_df(config: GetContactsConfig, pdb_name: str):
    if not config.contacts_dir:
        config.contacts_dir = Path("/tmp/")

    contacts_file = config.contacts_dir / (pdb_name + "_contacts.tsv")

    # Check for existence of GetContacts file
    if not os.path.isfile(contacts_file):
        run_get_contacts(config, pdb_name)

    contacts_df = read_contacts_file(config, contacts_file)

    # remove temp GetContacts file
    if config.contacts_dir == "/tmp/":
        os.remove(contacts_file)

    return contacts_df


def run_get_contacts(
    config: GetContactsConfig,
    pdb_id: Optional[str],
    file_name: Optional[str] = None,
):
    # Check for GetContacts Installation
    assert os.path.isfile(
        f"{config.get_contacts_path}/get_static_contacts.py"
    ), "No GetContacts Installation Detected"

    # Check for existence of pdb file. If not, download it.
    if not os.path.isfile(config.pdb_dir / pdb_id):
        pdb_file = download_pdb(config, pdb_id)
    else:
        pdb_file = config.pdb_dir + pdb_id + ".pdb"

    # Run GetContacts
    command = f"{config.get_contacts_path}/get_static_contacts.py "
    command += f"--structure {pdb_file} "
    command += f'--output {(config.contacts_dir / (pdb_id + "_contacts.tsv")).as_posix()} '
    command += "--itypes all"  # --sele "protein"'
    print(command)
    subprocess.run(command, shell=True)

    # Check it all checks out
    assert os.path.isfile(config.contacts_dir / (pdb_id + "_contacts.tsv"))
    print(f"Computed Contacts for: {pdb_id}")


def read_contacts_file(
    config: GetContactsConfig, contacts_file
) -> pd.DataFrame:
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
    Adds specific interaction types to the protein graph
    :param G: networkx protein graph
    :type G: nx.Graph
    :param interaction_type: interaction type to be added
    :type interaction_type: str
    :return G: nx.Graph
    """
    # Load contacts df
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
    return add_contacts_edge(G, "hb")


def salt_bridge(G: nx.Graph) -> nx.Graph:
    return add_contacts_edge(G, "sb")


def pi_cation(G: nx.Graph) -> nx.Graph:
    return add_contacts_edge(G, "pc")


def pi_stacking(G: nx.Graph) -> nx.Graph:
    return add_contacts_edge(G, "ps")


def t_stacking(G: nx.Graph) -> nx.Graph:
    return add_contacts_edge(G, "ts")


def hydrophobic(G: nx.Graph) -> nx.Graph:
    return add_contacts_edge(G, "hp")


def van_der_waals(G: nx.Graph) -> nx.Graph:
    return add_contacts_edge(G, "vdw")
