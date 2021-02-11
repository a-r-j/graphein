import argparse

import pandas as pd
import torch as torch
from tqdm import tqdm

from graphein.protein.config import (
    DSSPConfig,
    GetContactsConfig,
    ProteinGraphConfig,
)
from graphein.protein.edges.intramolecular import (
    hydrogen_bond,
    hydrophobic,
    pi_cation,
    pi_stacking,
    salt_bridge,
    t_stacking,
    van_der_waals,
)
from graphein.protein.graphs import construct_graph

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(
        description="Pass Graph Construction Params"
    )
    parser.add_argument("-o", "--out_dir", required=False, type=str)
    parser.add_argument("-n", "--node_featuriser", required=True, type=str)
    parser.add_argument("-s", "--include_ss", required=True, type=bool)
    parser.add_argument("-c", "--get_contacts_path", required=True, type=str)
    parser.add_argument(
        "-e", "--edge_distance_cutoff", required=False, type=float
    )
    args = parser.parse_args()
    # attention = args.attention
    """

    # Load Dataframe
    # df = pd.read_feather('structural_rearrangement_data.feather')
    df = pd.read_csv("structural_rearrangement_data.csv")
    print(df)

    # Initialise Graph Constructor
    configs = {
        "granularity": "CA",
        "keep_hets": False,
        "insertions": False,
        "verbose": False,
        "pdb_dir": "../examples/pdbs/",
        "get_contacts_config": GetContactsConfig(
            contacts_dir="../examples/contacts/",
            pdb_dir="../examples/contacts/",
        ),
        "dssp_config": DSSPConfig(),
    }

    config = ProteinGraphConfig(**configs)

    config.edge_construction_functions = [
        salt_bridge,
        hydrogen_bond,
        van_der_waals,
        pi_cation,
        pi_stacking,
        hydrophobic,
        t_stacking,
    ]
    # Test High-level API

    # Iterate over rows to produce Graph, pickle graph and label
    for row in tqdm(range(len(df))):
        example = df.iloc[row]
        file_path = f'pdbs/{example["Free PDB"]}.pdb'
        contact_file = f'contacts/{example["Free PDB"]}_contacts.tsv'

        g = construct_graph(config=config, pdb_code=example["Free PDB"])

        print(g)

    print("Successfully computed all graphs")

# Example Run:
# python make_rearrangement_data.py -o 'none' -n 'meiler' -s True -c '/home/arj39/Documents/github/getcontacts'
# python make_rearrangement_data.py -o 'none' -n 'meiler' -s True -c '/Users/arianjamasb/github/getcontacts'
