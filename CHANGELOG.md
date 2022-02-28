### 1.2.0 - Unreleased

* [Feature] - #104 adds support for asteroid plots and distance matrix visualisation.
* [Feature] - #104 adds support for protein graph analytics (`graphein.protein.analysis`)
* [Feature] - #110 adds support for secondary structure & surface-based subgraphs
* [Feature] - #113 adds CLI support(!)
* [Bugfix] - #110 fixes minor bug in `asa` where it would fail if added as a first/only dssp feature.
* [Bugfix] - #110 Adds install for DSSP in Dockerfile
* [Bugfix] - #110 Adds conda install & DSSP to tests
* [Documentation] - #104 Adds notebooks for visualisation, RNA SS Graphs, protein graph analytics

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
