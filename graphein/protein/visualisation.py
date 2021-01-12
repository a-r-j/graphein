"""Functions for plotting protein graphs and meshes"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

from itertools import count
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.ops import sample_points_from_meshes


def plot_pointcloud(mesh: Meshes, title: str = "") -> None:
    """
    Plots pytorch3d Meshes object as pointcloud
    :param mesh: Meshes object to plot
    :param title: Title of plot
    :return:
    """
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()


def colour_nodes(
    G: nx.Graph, colour_map: matplotlib.colors.ListedColormap, colour_by: str
) -> List[Tuple[float, float, float, float]]:
    """
    Computes node colours based on "degree", "seq_position" or node attributes
    :param G: Graph to compute node colours for
    :param colour_map:  Colourmap to use.
    :param colour_by: Manner in which to colour nodes. If node_types "degree" or "seq_position", this must correspond to a node feature
    :return: List of node colours
    """
    # get number of nodes
    n = G.number_of_nodes()

    # Get max number of edges connected to a single node
    edge_max = max([G.degree[i] for i in G.nodes()])

    # Define color range proportional to number of edges adjacent to a single node
    if colour_by == "degree":
        colors = [colour_map(G.degree[i] / edge_max) for i in G.nodes()]
    elif colour_by == "seq_position":
        colors = [colour_map(i / n) for i in range(n)]
    else:
        node_types = set(nx.get_node_attributes(G, colour_by).values())
        mapping = dict(zip(sorted(node_types), count()))
        colors = [
            colour_map(mapping[d[colour_by]] / len(node_types))
            for n, d in G.nodes(data=True)
        ]

    return colors


def colour_edges(
    G: nx.Graph,
    colour_map: matplotlib.colors.ListedColormap,
    colour_by: str = "kind",
) -> List[Tuple[float, float, float, float]]:
    """
    Computes edge colours based on the kind of bond/interaction.
    :param G: nx.Graph protein structure graph to compute edge colours from
    :param colour_map: Colourmap to use
    :param colour_by: Edge attribute to colour by. Currently only "kind" is supported
    :return: List of edge colours
    """
    if colour_by == "kind":
        edge_types = set(
            frozenset(a) for a in nx.get_edge_attributes(G, "kind").values()
        )
        mapping = dict(zip(sorted(edge_types), count()))
        colors = [
            colour_map(
                mapping[frozenset(G.edges[i]["kind"])] / len(edge_types)
            )
            for i in G.edges()
        ]
    return colors


def plot_protein_structure_graph(
    G: nx.Graph,
    angle: int,
    figsize: Tuple[int, int] = (10, 7),
    node_alpha: float = 0.7,
    node_size_min: float = 20.0,
    node_size_multiplier: float = 20.0,
    label_node_ids: bool = True,
    node_colour_map=plt.cm.plasma,
    edge_color_map=plt.cm.plasma,
    colour_nodes_by: str = "degree",
    colour_edges_by: str = "type",
    edge_alpha: float = 0.5,
    plot_style: str = "ggplot",
    out_path: Optional[str] = None,
    out_format: str = ".png",
):
    """
    Plots protein structure graph in Axes3D.
    :param G:  nx.Graph Protein Structure graph to plot
    :param angle:  View angle
    :param figsize: Size of figure
    :param node_alpha: Controls node transparency
    :param node_size_min: Specifies node minimum size
    :param node_size_multiplier: Scales node size by a constant. Node sizes reflect degree.
    :param label_node_ids: bool indicating whether or not to plot node_id labels
    :param node_colour_map: colour map to use for nodes
    :param edge_color_map: colour map to use for edges
    :param colour_nodes_by: Specifies how to colour nodes. "degree"m "seq_position" or a node eature
    :param colour_edges_by: Specifies how to colour edges. Currently only "kind" is supported
    :param edge_alpha: Controls edge transparency
    :param plot_style: matplotlib style sheet to use
    :param out_path: If not none, writes plot to this location
    :param out_format: Fileformat to use for plot
    :return:
    """
    # Get Node Attributes
    pos = nx.get_node_attributes(G, "coords")

    # Get node colours
    node_colors = colour_nodes(
        G, colour_map=node_colour_map, colour_by=colour_nodes_by
    )
    edge_colors = colour_edges(
        G, colour_map=edge_color_map, colour_by=colour_edges_by
    )

    # 3D network plot
    with plt.style.context((plot_style)):

        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for i, (key, value) in enumerate(pos.items()):
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(
                xi,
                yi,
                zi,
                color=node_colors[i],
                s=node_size_min + node_size_multiplier * g.degree[key],
                edgecolors="k",
                alpha=node_alpha,
            )
            if label_node_ids:
                label = list(G.nodes())[i]
                ax.text(xi, yi, zi, label)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i, j in enumerate(g.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))

            # Plot the connecting lines
            ax.plot(x, y, z, c=edge_colors[i], alpha=edge_alpha)

    # Set the initial view
    ax.view_init(30, angle)
    # Hide the axes
    ax.set_axis_off()
    if out_path is not None:
        plt.savefig(out_path + str(angle).zfill(3) + out_format)
        plt.close("all")

    return plt


if __name__ == "__main__":
    from graphein.protein.config import ProteinGraphConfig
    from graphein.protein.edges.atomic import (
        add_atomic_edges,
        add_bond_order,
        add_ring_status,
    )
    from graphein.protein.edges.distance import (
        add_aromatic_sulphur_interactions,
        add_delaunay_triangulation,
        add_disulfide_interactions,
        add_hydrophobic_interactions,
        add_ionic_interactions,
    )
    from graphein.protein.edges.intramolecular import (
        hydrogen_bond,
        peptide_bonds,
        salt_bridge,
    )
    from graphein.protein.features.nodes.amino_acid import (
        expasy_protein_scale,
        meiler_embedding,
    )
    from graphein.protein.graphs import construct_graph
    from graphein.protein.meshes import (
        convert_verts_and_face_to_mesh,
        create_mesh,
    )

    # Test Point cloud plotting
    # v, f, a = create_mesh(pdb_code="3eiy")
    # m = convert_verts_and_face_to_mesh(v, f)
    # plot_pointcloud(m, "Test")
    # TEST PROTEIN STRUCTURE GRAPH PLOTTING
    configs = {
        "granularity": "atom",
        "keep_hets": False,
        "deprotonate": True,
        "insertions": False,
        "verbose": False,
    }

    config = ProteinGraphConfig(**configs)
    config.edge_construction_functions = [
        add_atomic_edges,
        add_ring_status,
        add_bond_order,
    ]

    config.node_metadata_functions = [meiler_embedding, expasy_protein_scale]
    # g = construct_graph(config=config, pdb_path="../../examples/pdbs/1a1e.pdb", pdb_code="1a1e")

    g = construct_graph(
        config=config, pdb_path="../../examples/pdbs/1a1e.pdb", pdb_code="1a1e"
    )
    print(nx.info(g))

    p = plot_protein_structure_graph(
        g,
        30,
        (10, 7),
        colour_nodes_by="element_symbol",
        colour_edges_by="kind",
        label_node_ids=False,
    )

    p.show()
