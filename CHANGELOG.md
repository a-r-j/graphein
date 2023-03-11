### 1.6.0dev - UNRELEASED

#### New Features
* [Metrics] - [#245](https://github.com/a-r-j/graphein/pull/221) Adds a selection of structural metrics relevant to protein structures.
* [Tensor Operations] - [#244](https://github.com/a-r-j/graphein/pull/244) Adds suite of utilities for working directly with tensor-based representations of proteins (graphein.protein.tensor).
* [Tensor Operations] - [#244](https://github.com/a-r-j/graphein/pull/244) Adds suite of utilities for working with ESMfold (graphein.protein.folding_utils).



#### Improvements

* [Logging] - [#221](https://github.com/a-r-j/graphein/pull/221) Adds global control of logging with `graphein.verbose(enabled=False)`.
* [Logging] - [#242](https://github.com/a-r-j/graphein/pull/242) Adds control of protein graph construction logging. Resolves [#238](https://github.com/a-r-j/graphein/issues/238)

#### Protein

* [Bugfix] - [#222]https://github.com/a-r-j/graphein/pull/222) Fixes entrypoint for user-defined `df_processing_funcs` ([#216](https://github.com/a-r-j/graphein/issues/216))
* [Feature] - [#264](https://github.com/a-r-j/graphein/pull/264) Adds entrypoint to `graphein.protein.graphs.construct_graph` for passing in a BioPandas dataframe directly.
* [Feature] - [#229](https://github.com/a-r-j/graphein/pull/220) Adds support for filtering KNN edges based on self-loops and chain membership. Contribution by @anton-bushuiev.
* [Feature] - [#234](https://github.com/a-r-j/graphein/pull/234) Adds support for aggregating node features over residues (`graphein.protein.features.sequence.utils.aggregate_feature_over_residues`).
* [Bugfix] - [#234](https://github.com/a-r-j/graphein/pull/234) fixes use of nullcontext in silent graph construction.
* [Bugfix] - [#234](https://github.com/a-r-j/graphein/pull/234) Fixes division by zero errors for edge colouring in visualisation.
* [Bugfix] - [#254](https://github.com/a-r-j/graphein/pull/254) Fix peptide bond addition for all atom graphs.
* [Bugfix] - [#223](https://github.com/a-r-j/graphein/pull/220) Fix handling of insertions in protein graphs. Insertions are now given IDs like: `A:SER:12:A`. Contribution by @manonreau.
* [Bugfix] - [#226](https://github.com/a-r-j/graphein/pull/226) Catches failed AF2 structure downloads [#225](https://github.com/a-r-j/graphein/issues/225)

* [Bugfix] - [#229](https://github.com/a-r-j/graphein/pull/220) Fixes bug in KNN edge computation. Contribution by @anton-bushuiev.
* [Bugfix] - [#220](https://github.com/a-r-j/graphein/pull/220) Fixes edge metadata conversion to PyG. Contribution by @manonreau.
* [Bugfix] - [#220](https://github.com/a-r-j/graphein/pull/220) Fixes centroid atom grouping & avoids unnecessary edge computation where none are found. Contribution by @manonreau.

* [Bugfix] - [#268](https://github.com/a-r-j/graphein/pull/268) Fixes 'sequence' metadata feature for atomistic graphs, removing duplicate residues.  Contribution by @kamurani.


#### ML
* [Bugfix] - [#234](https://github.com/a-r-j/graphein/pull/234) - Fixes bugs and improves `conversion.convert_nx_to_pyg` and `visualisation.plot_pyg_data`. Removes distance matrix (`dist_mat`) from defualt set of features converted to tensor.

#### Utils
* [Improvement] - [#234](https://github.com/a-r-j/graphein/pull/234) - Adds `parse_aggregation_type` to retrieve aggregation functions.

#### Constants
* [Improvement] - [#234](https://github.com/a-r-j/graphein/pull/234) - Adds 1 to 3 mappings to `graphein.protein.resi_atoms`.


#### Documentation
* [Tensor Module] - [#244](https://github.com/a-r-j/graphein/pull/244) Documents new graphein.protein.tensor module.
* [CI] - [#244](https://github.com/a-r-j/graphein/pull/244) Updates to intersphinx maps


#### Package
* [CI] - [#244](https://github.com/a-r-j/graphein/pull/244) CI now runs for python 3.8, 3.9 and torch 1.12.0 and 1.13.0
* [CI] - [#244](https://github.com/a-r-j/graphein/pull/244) Separate builds for core library and library with DL dependencies.
* [Licence] - [#244](https://github.com/a-r-j/graphein/pull/244) Bump to 2023


### 1.5.2 - 19/9/2022

#### Protein

* [Bugfix] - [#206](https://github.com/a-r-j/graphein/pull/206) Fixes `KeyError` when using `graphein.protein.edges.distance.node_coords`
* [Bugfix] - Includes missing data files in `MANIFEST.in` #205

#### GRN

* [Bugfix] - [#208](https://github.com/a-r-j/graphein/pull/208) - Resolves SSL issues with RegNetwork.

#### ML

* [Feature] - [#208](https://github.com/a-r-j/graphein/pull/208) support for loading local pdb files by ``ProteinGraphDataset`` and ``InMemoryProteinGraphDataset``.

>by adding a params:`pdb_paths` and set the `self.raw_dir` to the root path(`self.pdb_path`) of pdb_paths list (the root path should be only one, pdb files should be under the same folder).
>
> it allows loading pdb files from the `self.pdb_path` instead of loading from `self.raw`.
> If you wish to download from af2 or pdb, just set `pdb_paths` to `None` and it goes back to the former version.

#### CI

* [Bugfix] - [#208](https://github.com/a-r-j/graphein/pull/208) explicitly installs `jupyter_contrib_nbextensions` in Docker.

### 1.5.1

#### Protein

* [Feature] - [#186](https://github.com/a-r-j/graphein/pull/186) adds support for scaling node sizes in plots by a computed feature. Contribution by @cimranm
* [Feature] - [#189](https://github.com/a-r-j/graphein/pull/189/) adds support for parallelised download from the PDB.
* [Feature] - [#189](https://github.com/a-r-j/graphein/pull/189/) adds support for: van der waals interactions, vdw clashes, pi-stacking interactions, t_stacking interactions, backbone carbonyl-carbonyl interactions, salt bridges
* [Feature] - [#189](https://github.com/a-r-j/graphein/pull/189/) adds a `residue_id` column to PDB dfs to enable easier accounting in atom graphs.
* [Feature] - [#189](https://github.com/a-r-j/graphein/pull/189/) refactors torch geometric datasets to use parallelised download for faster dataset preparation.

#### Bugfixes

* [Patch] - [#187](https://github.com/a-r-j/graphein/pull/187) updates sequence retrieval due to UniProt API changes.
* [Patch] - [#189](https://github.com/a-r-j/graphein/pull/189) fixes bug where chains and PDB identifiers were not properly aligned in `ml.ProteinGraphDataset`.
* [Patch] - [#201](https://github.com/a-r-j/graphein/pull/201) Adds missing `MSE` to `graphein.protein.resi_atoms.RESI_NAMES`, `graphein.protein.resi_atoms.RESI_THREE_TO_1`. [#200](https://github.com/a-r-j/graphein/issues/200)
* [Patch] - [#201](https://github.com/a-r-j/graphein/pull/201) Fixes bug where check for same-chain always evaluates as False. [#199](https://github.com/a-r-j/graphein/issues/199)
* [Patch] - [#201](https://github.com/a-r-j/graphein/pull/201) Fixes bug where deprotonation would only remove hydrogens based on `atom_name` rather than `element_symbol`. [#198](https://github.com/a-r-j/graphein/issues/198)
* [Patch] - [#201](https://github.com/a-r-j/graphein/pull/201) Fixes bug in ProteinGraphDataset input validation.

#### Breaking Changes

* [#189](https://github.com/a-r-j/graphein/pull/189/) refactors PDB download util. Now returns path to download file, does not accept a config object but instead receives the output directory path directly.

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
