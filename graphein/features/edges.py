"""
Featurization functions for graph edges.
"""

import os
import pandas as pd

def peptide_bonds(G):
    """
    Adds peptide backbone to residues in each chain
    :param G: networkx protein graph
    :return
    """

    # Iterate over every chain
    for chain_id in G.graph["chain_ids"]:

        # Find chain residues
        chain_residues = [n for n,v in G.nodes(data=True) if v['chain_id'] == chain_id]

        # Iterate over every residue in chain
        for i, residue in enumerate(chain_residues):

            # Checks not at chain terminus - is this versatile enough?
            if i == len(chain_residues)-1:
                pass
            else:
                # PLACE HOLDER EDGE FEATURE
                # Adds "peptide bond" between current residue and the next
                G.add_edge(residue, chain_residues[i+1], attr="peptide_bond", color="b")

    return G

####################################
#                                  #
#     GetContacts Interactions     #
#                                  #
####################################


def get_contacts_df(config, pdb_id):

    if not config.contacts_dir:
        config.contacts_dir = "/tmp/"

    contacts_file = config.contacts_dir + pdb_id + "_contacts.tsv"

    if os.path.isfile(contacts_file) == False:
        # NEED TO ADD
        run_get_contacts(config)

    contacts_df = read_contacts_file(config, contacts_file)

    # remove temp GetContacts file
    if config.contacts_dir == "/tmp/":
        os.remove(contacts_file)

    return contacts_df

def read_contacts_file(config, contacts_file):

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

    edges = pd.DataFrame(contacts,
                columns=["res1", "res2", "interaction_type"])

    return edges.drop_duplicates()



def add_contacts_edge(G, interaction_type):
    """
    Adds specific interaction types to the protein graph
    :param G: networkx protein graph
    :param interaction_type: interaction type to be added
    :return
    """

    # Load contacts df
    contacts = G.graph["contacts_df"]
    # Select specific interaction type
    interactions = contacts.loc[contacts["interaction_type"] == interaction_type]

    for label, [res1, res2, interaction_type] in interactions.iterrows():
        # Place holder
        G.add_edge(res1, res2, attr=interaction_type, color="r")

    return G


def hydrogen_bond(G):
    return add_contacts_edge(G, "hb")

def salt_bridge(G):
    return add_contacts_edge(G, "sb")

def pi_cation(G):
    return add_contacts_edge(G, "pc")

def pi_stacking(G):
    return add_contacts_edge(G, "ps")

def t_stacking(G):
    return add_contacts_edge(G, "ts")

def hydrophobic(G):
    return add_contacts_edge(G, "hp")

def van_der_waals(G):
    return add_contacts_edge(G, "vdw")
