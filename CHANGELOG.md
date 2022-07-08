### 1.5.1

#### Changes

* [Feature] - [#186](https://github.com/a-r-j/graphein/pull/186) adds support for scaling node sizes in plots by a computed feature. Contribution by @cimranm
* [Patch] - [#187](https://github.com/a-r-j/graphein/pull/187) updates sequence retrieval due to UniProt API changes.

### 1.5.0

#### Protein

* [Feature] - #165 adds support for direct AF2 graph construction.
* [Feature] - #165 adds support for selecting model indices from PDB files.
* [Feature] - #165 adds support for extracting interface subgraphs from complexes.
* [Feature] - #165 adds support for computing the radius of gyration of a structure.
* [Feature] - #165 adds support for adding distances to protein edges.
* [Feature] - #165 adds support for fully connected edges in protein graphs.
* [Feature] - #165 adds support for distance window-based edges for protein graphs.
* [Feature] - #165 adds support for transformer-like positional encoding of protein sequences.
* [Feature] - #165 adds support for plddt-like colouring of AF2 graphs
* [Feature] - #165 adds support for plotting PyG Data object (e.g. for logging to WandB).
* [Feature] - [#170](https://github.com/a-r-j/graphein/pull/170) Adds support for viewing edges in `graphein.protein.visualisation.asteroid_plot`. Contribution by @avivko.
* [Patch] - [#178](https://github.com/a-r-j/graphein/pull/178) Fixes [#171](https://github.com/a-r-j/graphein/pull/171) and optimizes `graphein.protein.features.nodes.dssp`. Contribution by @avivko.
* [Patch] - [#174](https://github.com/a-r-j/graphein/pull/174) prevents insertions always being removed. Resolves [#173](https://github.com/a-r-j/graphein/issues/173). Contribution by @OliverT1.
* [Patch] - #165 Refactors HETATM selections.

#### Molecules

* [Feature] - #165 adds additional graph-level molecule features.
* [Feature] - #165 adds support for generating conformers (and 3D graphs) from SMILES inputs
* [Feature] - #163 Adds support for molecule graph generation from an RDKit.Chem.Mol input.
* [Feature] - #163 Adds support for multiprocess molecule graph construction.

#### RNA

* [Feature] - #165 adds support for 3D RNA graph construction.
* [Feature] - #165 adds support for generating RNA SS from sequence using the Nussinov Algorithm.

#### Changes

* [Patch] - #163 uses tqdm.contrib.process_map insteap of multiprocessing.Pool.map to provide progress bars in multiprocessing.
* [Fix] - #165 makes returned subgraphs editable objects rather than views
* [Fix] - #165 fixes global logging set to "debug".
* [Fix] - #165 uses rich progress for protein graph construction.
* [Fix] - #165 sets saner default for node size in 3d plotly plots
* [Dependency] - #165 Changes CLI to use rich-click instead of click for prettier formatting.
* [Package] - #165 Adds support for logging with loguru and rich
* [Package] - Pin BioPandas version to 0.4.1 to support additional parsing features.

#### Breaking Changes

* #165 adds RNA SS edges into graphein.protein.edges.base_pairing
* #163 changes separate filetype input paths to `graphein.molecule.graphs.construct_graph`. Interface is simplified to simply `path="some/path.extension"` instead of separate inputs like `mol2_path=...` and `sdf_path=...`.

### 1.4.0 - UNRELEASED

* [Patch] - #158 changes the eigenvector computation method from `nx.eigenvector_centrality` to `nx.eigenvector_centrality_numpy`.
* [Feature] - #154 adds a way of checking that DSSP is executable before trying to use it. #154
* [Feature] - #157 adds support for small molecule graphs using RDKit. Resolves #155.
* [Feature] - #159 adds support for conversion to Jraph graphs for JAX users.

#### Breaking Changes

* #157 refactors config matching operators from `graphein.protein.config` to `graphein.utils.config`
* #157 refactors config parsing operators from `graphein.utils.config` to `graphein.utils.config_parser`

### 1.3.0 - 5/4/22

* [Feature] - #141 adds edge construction based on sequence distance.
* [Feature] - #143 adds equality and isomorphism testing functions between graphs, nodes and edges ([#142](https://github.com/a-r-j/graphein/issues/142))
* [Feature] - #144 adds support for chain-level and secondary structure-level graphs with associated visualisation tools and tutorial. Resolves [#128](https://github.com/a-r-j/graphein/issues/128)
* [Feature] - #144 adds support for chord diagram visualisations.
* [Feature] - #144 adds support for automagically downloading new PDB files for obsolete structures.
* [Feature] - #150 adds support for hydrogen bond donor and acceptor counts node features. #145
* [Misc] - #144 makes visualisation functions accessible in the `graphein.protein` namespace. #138
* [Bugfix] - #147 fixes error in `add_distance_threshold` introduced in v1.2.1 that would prevent the edges being added to the graph. [#146](https://github.com/a-r-j/graphein/issues/146)
* [Bugfix] - #149 fixes a bug in `add_beta_carbon_vector` that would cause coordinates to be extracted for multiple positions if the residue has an altloc. Resolves [#148](https://github.com/a-r-j/graphein/issues/148)

### 1.2.1 - 16/3/22

* [Feature] - #124 adds support for vector features associated protein protein geometry. #120 #122
* [Feature] - #124 adds visualisation of vector features in 3D graph plots.
* [Feature] - #121 adds functions for saving graph data to PDB files.
* [Bugfix] - #136 changes generator comprehension when updating coordinates in subgraphs to list comprehension to allow pickling
* [Bugfix] - #136 fixes bug in edge construction functions using chain selections where nodes from unselected chains would be added to the graph.

#### Breaking Changes

* #124 refactors `graphein.protein.graphs.compute_rgroup_dataframe` and moves it to `graphein.protein.utils`. All internal references have been moved accordingly.

### 1.2.0 - 4/3/2022

* [Feature] - #104 adds support for asteroid plots and distance matrix visualisation.
* [Feature] - #104 adds support for protein graph analytics (`graphein.protein.analysis`)
* [Feature] - #110 adds support for secondary structure & surface-based subgraphs
* [Feature] - #113 adds CLI support(!)
* [Feature] - #116 adds support for onehot-encoded amino acid features as node attributes.
* [Feature] - #119 Adds plotly-based visualisation for PPI Graphs
* [Bugfix] - #110 fixes minor bug in `asa` where it would fail if added as a first/only dssp feature.
* [Bugfix] - #110 Adds install for DSSP in Dockerfile
* [Bugfix] - #110 Adds conda install & DSSP to tests
* [Bugfix] - #119 Delaunay Triangulation computed over all atoms by default. Adds an option to restrict it to certain atom types.
* [Bugfix] - #119 Minor fixes to stability of RNA Graph Plotting
* [Bugfix] - #119 add tolerance parameter to add_atomic_edges
* [Documentation] - #104 Adds notebooks for visualisation, RNA SS Graphs, protein graph analytics
* [Documentation] - #119 Overhaul of docs & tutorial notebooks. Adds interactive plots to docs, improves docstrings, doc formatting, doc requirements.

#### Breaking Changes

* #119 - Refactor RNA Graph constants from graphein.rna.graphs to graphein.rna.constants. Only problematic if constants were accessed directly. All internal references have been moved accordingly.

### 1.1.1 - 19/02/2022

* [Bugfix] - #107 improves robustness of removing insertions and hetatms, resolves #98
* [Packaging] - #108 fixes version mismatches in pytorch_geometric in docker install

### 1.1.0 - 19/02/2022

* [Packaging] - #100 adds docker support.
* [Feature] - #96 Adds support for extracting subgraphs
* [Packaging] - #101 adds support for devcontainers for remote development.
* [Bugfixes] - #95 adds improved robustness for edge construction functions in certain edge cases. Insertions in the PDB were occasionally not picked up due to a brittle implementations. Resolves #74 and #98

### 1.0.11 - 01/02/2022

* [Improvement] - #79 Replaces `Literal` references with `typing_extensions.Literal` for Python 3.7 support.

#### 1.0.10 - 23/12/2021

* [Bug] Adds a fix for #74. Adding a disulfide bond to a protein with no disulphide bonds would fail. This was fixed by adding a check for the presence of a minimum of two CYS residues.
