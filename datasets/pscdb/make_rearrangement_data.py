import argparse
import torch as torch
import pandas as pd
from tqdm import tqdm
from graphein.construct_graphs import ProteinGraph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pass Graph Construction Params')
    parser.add_argument('-o', '--out_dir', required=False, type=str)
    parser.add_argument('-n', '--node_featuriser', required=True, type=str)
    parser.add_argument('-s', '--include_ss', required=True, type=bool)
    parser.add_argument('-c', '--get_contacts_path', required=True, type=str)
    parser.add_argument('-e', '--edge_distance_cutoff', required=False, type=float)
    args = parser.parse_args()
    # attention = args.attention

    # Load Dataframe
    #df = pd.read_feather('structural_rearrangement_data.feather')
    df = pd.read_csv('structural_rearrangement_data.csv')
    print(df)

    # Initialise Graph Constructor
    pg = ProteinGraph(granularity='CA',
                      insertions=False,
                      keep_hets=False,
                      node_featuriser='meiler',
                      get_contacts_path=args.get_contacts_path,
                      pdb_dir='pdbs/',
                      contacts_dir='contacts/',
                      exclude_waters=True,
                      covalent_bonds=False,
                      include_ss=args.include_ss,
                      include_ligand=False,
                      edge_distance_cutoff=args.edge_distance_cutoff
                      )

    # Iterate over rows to produce Graph, pickle graph and label
    for row in tqdm(range(len(df))):
        example = df.iloc[row]
        file_path = f'pdbs/{example["Free PDB"]}.pdb'
        contact_file = f'contacts/{example["Free PDB"]}_contacts.tsv'

        g = pg.dgl_graph_from_pdb_code(pdb_code=example['Free PDB'],
                                       chain_selection=list(example['Free Chains']))

        print(g)

    print('Successfully computed all graphs')

# Example Run:
#python make_rearrangement_data.py -o 'none' -n 'meiler' -s True -c '/home/arj39/Documents/github/getcontacts'
#python make_rearrangement_data.py -o 'none' -n 'meiler' -s True -c '/Users/arianjamasb/github/getcontacts'