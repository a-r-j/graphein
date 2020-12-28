"""Featurization functions for graph edges."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import os
import networkx as nx
import glob
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

from typing import Optional
from pydantic import BaseModel
from graphein.protein.utils import download_pdb


def peptide_bonds(G: nx.Graph) -> nx.Graph:
    """
    Adds peptide backbone to residues in each chain
    :param G: networkx protein graph
    :return G; networkx protein graph with added peptide bonds
    """

    # Iterate over every chain
    for chain_id in G.graph["chain_ids"]:

        # Find chain residues
        chain_residues = [
            n for n, v in G.nodes(data=True) if v["chain_id"] == chain_id
        ]

        # Iterate over every residue in chain
        for i, residue in enumerate(chain_residues):

            # Checks not at chain terminus - is this versatile enough?
            if i == len(chain_residues) - 1:
                pass
            else:
                # PLACE HOLDER EDGE FEATURE
                # Adds "peptide bond" between current residue and the next
                G.add_edge(
                    residue,
                    chain_residues[i + 1],
                    attr="peptide_bond",
                    color="b",
                )

    return G

####################################
#                                  #
#     Distance-based Edges         #
#                                  #
####################################
# Todo distance-based edge funcs

"""
def _distance_based_edges(G: nx.Graph, cutoff: float) -> pd.DataFrame:
        
        Calculate distance-based edges from coordinates in 3D structure.

        Produce Edge list dataframe based on pairwise distance matrix calculation
        :param protein_df: PandasPDB Dataframe
        :param cutoff: Distance threshold to create an edge (Angstroms)
        :return: dists : pandas dataframe of edge list and distance
        
        # Create distance matrix
        coords = protein_df[["x_coord", "y_coord", "z_coord"]]
        dists = pairwise_distances(np.asarray(coords))
        # Filter distance matrix and select lower triangle
        dists = pd.DataFrame(np.tril(np.where(dists < cutoff, dists, 0)))
        # Reshape to produce edge list
        dists.values[[np.arange(len(dists))] * 2] = np.nan
        dists = dists.stack().reset_index()
        # Filter to remove edges that exceed cutoff
        dists = dists.loc[dists[0] != 0]

        if self.long_interaction_threshold:
            dists = dists.loc[
                abs(abs(dists["level_0"]) - abs(dists["level_1"]))
                > self.long_interaction_threshold
            ]

        if self.verbose:
            print(f"Calcuclated {len(dists)} distance-based edges")
        return dists
"""


####################################
#                                  #
#     GetContacts Interactions     #
#                                  #
####################################


def get_contacts_df(config: BaseModel, pdb_id: str):

    if not config.contacts_dir:
        config.contacts_dir = "/tmp/"

    contacts_file = config.contacts_dir + pdb_id + "_contacts.tsv"

    # Check for existence of GetContacts file
    if not os.path.isfile(contacts_file):
        run_get_contacts(config, pdb_id)

    contacts_df = read_contacts_file(config, contacts_file)

    # remove temp GetContacts file
    if config.contacts_dir == "/tmp/":
        os.remove(contacts_file)

    return contacts_df


def run_get_contacts(config: BaseModel, pdb_id: Optional[str], file_name: Optional[str] = None):
    # Check for GetContacts Installation
    assert os.path.isfile(f"{config.get_contacts_path}/get_static_contacts.py"), "No GetContacts Installation Detected"

    # Check for existence of pdb file. If not, download it.
    if not os.path.isfile(config.pdb_dir + pdb_id):
        pdb_file = download_pdb(config, pdb_id)
    else:
        pdb_file = config.pdb_dir + pdb_id + ".pdb"

    # Run GetContacts
    command = f"{config.get_contacts_path}/get_static_contacts.py "
    command += f"--structure {pdb_file} "
    command += (
        f'--output {config.contacts_dir + pdb_id + "_contacts.tsv"} '
    )
    command += "--itypes all"  # --sele "protein"'
    subprocess.run(command, shell=True)

    # Check it all checks out
    assert os.path.isfile(config.contacts_dir + pdb_id + "_contacts.tsv")
    print(f"Computed Contacts for: {pdb_id}")


def read_contacts_file(config: BaseModel, contacts_file) -> pd.DataFrame:
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


def add_contacts_edge(G: nx.Graph, interaction_type: str, config=BaseModel) -> nx.Graph:
    """
    Adds specific interaction types to the protein graph
    :param G: networkx protein graph
    :type G: nx.Graph
    :param interaction_type: interaction type to be added
    :type interaction_type: str
    :return G: nx.Graph
    """
    G.graph["contacts_df"] = get_contacts_df(config, G.graph["pdb_id"])
    # Load contacts df
    contacts = G.graph["contacts_df"]
    # Select specific interaction type
    interactions = contacts.loc[
        contacts["interaction_type"] == interaction_type
    ]

    for label, [res1, res2, interaction_type] in interactions.iterrows():
        # Place holder
        G.add_edge(res1, res2, attr=interaction_type, color="r")

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
