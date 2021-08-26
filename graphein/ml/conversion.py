from typing import List, Optional

import networkx as nx
import numpy as np
import torch

from graphein.utils.utils import import_message

try:
    from torch_geometric.data import Data
except ImportError:
    import_message(
        submodule="graphein.ml.conversion",
        package="torch_geometric",
        pip_install=True,
    )


class graph_format_convert:
    def __init__(
        self,
        src_format: str,
        dst_format: str,
        verbose: str = "gnn",
        columns: Optional[List[str]] = None,
    ):
        supported_format = ["nx", "pyg", "dgl"]
        if (src_format not in supported_format) or (
            dst_format not in supported_format
        ):
            raise ValueError(
                "Please specify from supported format, "
                + "/".join(supported_format)
            )
        self.src_format = src_format
        self.dst_format = dst_format

        supported_verbose_format = ["gnn", "default", "all_info"]
        if (columns is None) and (verbose not in supported_verbose_format):
            raise ValueError(
                "Please specify the supported verbose mode ("
                + "/".join(supported_verbose_format)
                + ") or specify column names!"
            )

        if columns is None:
            if verbose == "gnn":
                columns = [
                    "edge_index",
                    "coords",
                    "dist_mat",
                    "name",
                    "node_id",
                ]
            elif verbose == "default":
                columns = [
                    "b_factor",
                    "chain_id",
                    "coords",
                    "dist_mat",
                    "edge_index",
                    "kind",
                    "name",
                    "node_id",
                    "residue_name",
                ]
            elif verbose == "all_info":
                columns = [
                    "atom_type",
                    "b_factor",
                    "chain_id",
                    "chain_ids",
                    "config",
                    "coords",
                    "dist_mat",
                    "edge_index",
                    "element_symbol",
                    "kind",
                    "name",
                    "node_id",
                    "node_type",
                    "pdb_df",
                    "raw_pdb_df",
                    "residue_name",
                    "residue_number",
                    "rgroup_df",
                    "sequence_A",
                    "sequence_B",
                ]
        self.columns = columns

        self.type2form = {
            "atom_type": "str",
            "b_factor": "float",
            "chain_id": "str",
            "coords": "np.array",
            "dist_mat": "np.array",
            "element_symbol": "str",
            "node_id": "str",
            "residue_name": "str",
            "residue_number": "int",
            "edge_index": "torch.tensor",
            "kind": "str",
        }

    def convert_nx_to_dgl(self, G: nx.Graph) -> dgl.DGLGraph:
        import dgl

        g = dgl.DGLGraph()
        node_id = [n for n in G.nodes()]
        G = nx.convert_node_labels_to_integers(G)

        ## add node level feat

        node_dict = {}
        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            for key, value in feat_dict.items():
                if str(key) in self.columns:
                    node_dict[str(key)] = (
                        [value] if i == 0 else node_dict[str(key)] + [value]
                    )

        string_dict = {}
        node_dict_transformed = {}
        for i, j in node_dict.items():
            if i == "coords":
                node_dict_transformed[i] = torch.Tensor(np.asarray(j)).type(
                    "torch.FloatTensor"
                )
            elif i == "dist_mat":
                node_dict_transformed[i] = torch.Tensor(
                    np.asarray(j[0].values)
                ).type("torch.FloatTensor")
            elif self.type2form[i] == "str":
                string_dict[i] = j
            elif self.type2form[i] == "float":
                node_dict_transformed[i] = torch.Tensor(np.array(j))
            elif self.type2form[i] == "int":
                node_dict_transformed[i] = torch.Tensor(np.array(j))

        g.add_nodes(
            len(node_id),
            node_dict_transformed,
        )

        edge_dict = {}
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        # add edge level features
        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            for key, value in feat_dict.items():
                if str(key) in self.columns:
                    edge_dict[str(key)] = (
                        list(value)
                        if i == 0
                        else edge_dict[str(key)] + list(value)
                    )

        edge_transform_dict = {}
        for i, j in node_dict.items():
            if self.type2form[i] == "str":
                string_dict[i] = j
            elif self.type2form[i] == "float":
                edge_transform_dict[i] = torch.Tensor(np.array(j))
            elif self.type2form[i] == "int":
                edge_transform_dict[i] = torch.Tensor(np.array(j))

        g.add_edges(edge_index[0], edge_index[1], edge_transform_dict)

        # add graph level features
        graph_dict = {}
        for i, feat_name in enumerate(G.graph):
            if str(feat_name) in self.columns:
                graph_dict[str(feat_name)] = [G.graph[feat_name]]

        return g

    def convert_nx_to_pyg(self, G: nx.Graph) -> Data:
        import torch_geometric
        from torch_geometric.data import Data

        # Initialise dict used to construct Data object
        data = {}

        # Assign node ids as a feature
        data["node_id"] = [n for n in G.nodes()]
        G = nx.convert_node_labels_to_integers(G)

        # Construct Edge Index
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        # Add node features
        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            for key, value in feat_dict.items():
                if str(key) in self.columns:
                    data[str(key)] = (
                        [value] if i == 0 else data[str(key)] + [value]
                    )

        # Add edge features
        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            for key, value in feat_dict.items():
                if str(key) in self.columns:
                    data[str(key)] = (
                        list(value) if i == 0 else data[str(key)] + list(value)
                    )

        # Add graph-level features
        for i, feat_name in enumerate(G.graph):
            if str(feat_name) in self.columns:
                data[str(feat_name)] = [G.graph[feat_name]]

        if "edge_index" in self.columns:
            data["edge_index"] = edge_index.view(2, -1)

        data = torch_geometric.data.Data.from_dict(data)
        data.num_nodes = G.number_of_nodes()
        return data

    def convert_nx_to_nx(self, G: nx.Graph) -> nx.Graph:
        return G

    def convert_dgl_to_nx(self, G: dgl.DGLGraph) -> nx.Graph:
        import dgl

        node_attrs = G.node_attr_schemes().keys()
        edge_attrs = G.edge_attr_schemes().keys()
        nx_g = dgl.to_networkx(G, node_attrs, edge_attrs)
        return nx_g

    def convert_pyg_to_nx(self, G: Data) -> nx.Graph:
        import torch_geometric

        return torch_geometric.utils.to_networkx(G)

    def __call__(self, G: nx.Graph):
        nx_g = eval("self.convert_" + self.src_format + "_to_nx(G)")
        dst_g = eval("self.convert_nx_to_" + self.dst_format + "(nx_g)")
        return dst_g


def convert_nx_to_pyg_data(G: nx.Graph) -> Data:
    # Initialise dict used to construct Data object
    data = {}

    # Assign node ids as a feature
    data["node_id"] = [n for n in G.nodes()]
    G = nx.convert_node_labels_to_integers(G)

    # Construct Edge Index
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    # Add node features
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    # Add edge features
    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = (
                list(value) if i == 0 else data[str(key)] + list(value)
            )

    # Add graph-level features
    for i, feat_name in enumerate(G.graph):
        data[str(feat_name)] = [G.graph[feat_name]]

    data["edge_index"] = edge_index.view(2, -1)
    data = Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data
