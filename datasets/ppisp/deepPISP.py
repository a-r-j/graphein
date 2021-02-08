import pickle

import pandas as pd
import torch
from Bio.PDB import *
from torch_geometric.data import Data
from tqdm import tqdm

from graphein.construct_graphs import ProteinGraph

# Get DeepPPISP Data
with open("all_dset_list.pkl", "rb") as f:
    index = pickle.load(f)

with open("dset186_label.pkl", "rb") as f:
    dset186_labels = pickle.load(f)

with open("dset164_label.pkl", "rb") as f:
    dset164_labels = pickle.load(f)

with open("dset72_label.pkl", "rb") as f:
    dset72_labels = pickle.load(f)

labels = dset186_labels + dset164_labels + dset72_labels


# Get PSSMs
with open("dset186_pssm_data.pkl", "rb") as f:
    dset_186_pssms = pickle.load(f)

with open("dset164_pssm_data.pkl", "rb") as f:
    dset_164_pssms = pickle.load(f)

with open("dset72_pssm_data.pkl", "rb") as f:
    dset_72_pssms = pickle.load(f)

pssms = dset_186_pssms + dset_164_pssms + dset_72_pssms

# write labels
pickle.dump(labels, open("ppisp_node_labels.p", "wb"))

df = pd.DataFrame(index)
df.columns = [
    "pos_index",
    "example_index",
    "res_position",
    "dataset",
    "pdb",
    "length",
]

df = df.loc[df["res_position"] == 0]
# Get PDB accession and chains
df[["pdb_code", "chains"]] = df.pdb.str.split("_", expand=True)
# These columns don't follow the format
df.loc[df["dataset"] == "dset164", "pdb_code"] = (
    df.copy().loc[df["dataset"] == "dset164"]["pdb"].str.slice(stop=4)
)
df.loc[df["dataset"] == "dset164", "chains"] = (
    df.copy().loc[df["dataset"] == "dset164"]["pdb"].str.slice(-1)
)
df["chains"] = df["chains"].fillna("all")
# Remove Obsolete structures
obsolete = ["3NW0", "3VDO"]
replacements = ["", "4NQW"]
df = df.loc[~df["pdb_code"].isin(obsolete)]

# Assign training/test status
with open("training_list.pkl", "rb") as f:
    train = pickle.load(f)

with open("testing_list.pkl", "rb") as f:
    test = pickle.load(f)

df.loc[df["pos_index"].isin(train), "train"] = 1
df.loc[df["pos_index"].isin(test), "train"] = 0
df.reset_index(inplace=True)
# Write Dataframe
df.to_csv("deepppisp_clean.csv")

# Initialise Protein Graph Class
pg = ProteinGraph(
    granularity="CA",
    insertions=False,
    keep_hets=False,
    node_featuriser="meiler",
    get_contacts_path="/Users/arianjamasb/github/getcontacts/",
    pdb_dir="ppisp_pdbs/",
    contacts_dir="ppisp_contacts/",
    exclude_waters=True,
    covalent_bonds=False,
    include_ss=True,
    include_ligand=False,
    edge_distance_cutoff=None,
)

graph_list = []
test_indices = []
train_indices = []
idx_counter = 0
for example in tqdm(range(len(labels))):
    # Create Protein Graph
    try:
        g = pg.dgl_graph_from_pdb_code(
            pdb_code=df["pdb_code"][example],
            chain_selection=list(df["chains"][example]),
            edge_construction=["contacts"],
        )

        # Create PSSM Feats and label
        df_index = df.iloc[example]["example_index"]
        label = labels[df_index]
        pssm = pssms[df_index]

    except:
        break

    # Ensure node labels match number of nodes. There are a few cases (~5) where this doesn't hold. We skip these.
    if g.number_of_nodes() != len(label):
        print("label length does not match ", example)
        print(g.number_of_nodes())
        print(len(label))
        continue
    if g.number_of_nodes() != len(pssm):
        print(g.number_of_nodes())
        print(len(pssm))
        print("pssm length does not match", example)
        continue

    if df["train"][example] == 0:
        test_indices.append(idx_counter)
    if df["train"][example] == 1:
        train_indices.append(idx_counter)
    idx_counter += 1

    node_features = torch.cat(
        (
            g.ndata["h"],
            g.ndata["ss"],
            g.ndata["asa"],
            g.ndata["rsa"],
            torch.Tensor(pssm),
        ),
        dim=1,
    )
    label = torch.Tensor(label).unsqueeze(dim=1)
    geom_graph = Data(
        x=torch.cat((node_features, label), dim=1),
        edge_index=torch.stack(g.edges(), dim=1),
    )
    graph_list.append(geom_graph)

# Normalize features
feats = torch.cat([g.x[:, :-1] for g in graph_list])
max_feats = torch.max(feats, dim=0)[0]
min_feats = torch.min(feats, dim=0)[0]

max_feats[max_feats == 0] = 1

for g in graph_list:
    g.x[:, :-1] -= min_feats
    g.x[:, :-1] /= max_feats

test_graphs = [graph_list[i] for i in test_indices]
train_graphs = [graph_list[i] for i in train_indices]


pickle.dump(train_graphs, open("ppisp_train_data_pssm_contacts.p", "wb"))
pickle.dump(test_graphs, open("ppisp_test_data_pssm_contacts.p", "wb"))
