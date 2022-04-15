### 1.4.0 - UNRELEASED

* [Patch] - #158 changes the eigenvector computation method from `nx.eigenvector_centrality` to `nx.eigenvector_centrality_numpy`.

### 1.3.1 - UNRELEASED

* [Feature] - #154 adds a way of checking that DSSP is executable before trying to use it. #154

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
