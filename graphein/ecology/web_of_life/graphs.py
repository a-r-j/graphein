import networkx as nx
import pandas as pd
from graphein.ecology.web_of_life.parse_web_of_life import parse_dataset, load_references


def initialize_graph_with_metadata(dataset_name: str) -> nx.Graph:
    G = nx.Graph()
        
    return G


def add_nodes_to_graph(df: pd.DataFrame):

    if "Num. of hosts sampled" in df.columns:
        num_hosts = df["Num. of hosts sampled"]
        df.drop(columns=["Num. of hosts sampled"], inplace=True)
    print(df)
    
    cols = list(df.columns.values)
    cols.remove("Unnamed: 0")

    node_list = cols + list(df["Unnamed: 0"].values)
    print(node_list)


def construct_graph(dataset_name: str):

    # Parse dataset from web of life datasets
    df = parse_dataset(dataset_name)

    # Annotate graph with metadata
    g = initialize_graph_with_metadata()

    # Add nodes to graph
    g = add_nodes_to_graph(g)


if __name__ == "__main__":
    df = construct_graph("A_HP_001")



    print(df)

