from pyaaisc import Aaindex
import networkx as nx
from graphein.utils import protein_letters_3to1_all_caps



def fetch_AAIndex(accession):

    aaindex = Aaindex()

    record = aaindex.get(accession)
    title = record.title
    index_data = record.index_data

    return title, index_data


def aaindex1(G: nx.Graph, accession: str) -> nx.Graph:

    title, index_data = fetch_AAIndex(accession)
    G.graph["aaindex1"] = accession + ": " + title

    for n in G.nodes:

        residue = n.split(":")[1]
        residue = protein_letters_3to1_all_caps(residue)

        aaindex = index_data[residue]

        # Change to accession number?
        G.nodes[n]["aaindex1"] = aaindex

    return G


