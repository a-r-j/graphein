from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import (add_peptide_bonds,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,
                                             add_ionic_interactions,
                                             add_aromatic_interactions,
                                             add_aromatic_sulphur_interactions,
                                             add_cation_pi_interactions
                                            )

from graphein.protein.visualisation import plot_protein_structure_graph


config = ProteinGraphConfig()


new_edge_funcs = {"edge_construction_functions": [add_peptide_bonds,
                                                  add_hydrogen_bond_interactions,
                                                  add_disulfide_interactions,
                                                  add_ionic_interactions,
                                                  add_aromatic_sulphur_interactions,
                                                  add_cation_pi_interactions]
                 }

config = ProteinGraphConfig(**new_edge_funcs)
print(config.dict())

g = construct_graph(config=config, pdb_code="3eiy")
print(g)
p = plot_protein_structure_graph(G=g, angle=0, colour_edges_by="kind", colour_nodes_by="degree", label_node_ids=False)
p.show()