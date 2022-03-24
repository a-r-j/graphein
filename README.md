[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/a-r-j/graphein-binder/master?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fa-r-j%252Fgraphein%26urlpath%3Dlab%252Ftree%252Fgraphein%252Fnotebooks%26branch%3Dmaster)
[![PyPI version](https://badge.fury.io/py/graphein.svg)](https://badge.fury.io/py/graphein)
![supported python versions](https://img.shields.io/pypi/pyversions/graphein)
[![Docs](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](http://www.graphein.ai)
[![DOI:10.1101/2020.07.15.204701](https://zenodo.org/badge/DOI/10.1101/2020.07.15.204701.svg)](https://doi.org/10.1101/2020.07.15.204701)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![CodeFactor](https://www.codefactor.io/repository/github/a-r-j/graphein/badge)](https://www.codefactor.io/repository/github/a-r-j/graphein)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=a-r-j_graphein&metric=alert_status)](https://sonarcloud.io/dashboard?id=a-r-j_graphein)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=a-r-j_graphein&metric=bugs)](https://sonarcloud.io/dashboard?id=a-r-j_graphein)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=a-r-j_graphein&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=a-r-j_graphein)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=a-r-j_graphein&metric=reliability_rating)](https://sonarcloud.io/dashboard?id=a-r-j_graphein)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/graphein)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![banner](https://github.com/a-r-j/graphein/blob/master/imgs/graphein.png?raw=true)](http://www.graphein.ai)

<br></br>

[Documentation](http://www.graphein.ai) | [Paper](https://www.biorxiv.org/content/10.1101/2020.07.15.204701v1) | [Tutorials](http://graphein.ai/notebooks_index.html) | [Installation](#installation)

Protein & Interactomic Graph Library

This package provides functionality for producing geometric representations of protein and RNA structures, and biological interaction networks. We provide compatibility with standard PyData formats, as well as graph objects designed for ease of use with popular deep learning libraries.

## What's New?

|   |   |
|---|---|
| [Extracting subgraphs from protein graphs](http://graphein.ai/notebooks/subgraphing_tutorial.html)  |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/subgraphing_tutorial.ipynb)   |
| [Protein Graph Analytics](http://graphein.ai/notebooks/protein_graph_analytics.html)  |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/protein_graph_analytics.ipynb) |
| [Graphein CLI](http://graphein.ai/getting_started/usage.html)  |   |
| [Protein Graph Creation from AlphaFold2!](http://graphein.ai/notebooks/alphafold_protein_graph_tutorial.html)  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/residue_graphs.ipynb) |
| [Protein Graph Visualisation!](http://graphein.ai/notebooks/interactive_plotly_example.html) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/interactive_plotly_example.ipynb)
| [RNA Graph Construction from Dotbracket notation](http://graphein.ai/modules/graphein.rna.html) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/rna_graph_tutorial.ipynb) |
| [Protein - Protein Interaction Network Support & Structural Interactomics (Using AlphaFold2!)](http://graphein.ai/notebooks/ppi_tutorial.html) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/ppi_graph.ipynb) |
| [High and Low-level API for massive flexibility - create your own bespoke workflows!](http://graphein.ai/notebooks/residue_graphs.html) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/residue_graphs.ipynb) |

## Example usage

Graphein provides both a programmatic API and a command-line interface for constructing graphs.

### CLI

Graphein configs can be specified as `.yaml` files to batch process graphs from the commandline.

[Docs](http://graphein.ai/getting_started/usage.html)

```bash
graphein -c config.yaml -p path/to/pdbs -o path/to/output
```

### Creating a Protein Graph

|   |   |   |
|---|---|---|
[Tutorial (Residue-level)](http://graphein.ai/notebooks/residue_graphs.html) | [Tutorial (Atomic)](http://graphein.ai/notebooks/atom_graph_tutorial.html) | [Docs](http://graphein.ai/modules/graphein.protein.html#module-graphein.protein.graphs)
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/residue_graphs.ipynb) | [![Open In Colab(https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/atom_graph_tutorial.ipynb) | |

```python
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph

config = ProteinGraphConfig()
g = construct_graph(config=config, pdb_code="3eiy")
```

### Creating a Protein Graph from the AlphaFold Protein Structure Database

|   |   |
|---|---|
| [Tutorial](http://graphein.ai/notebooks/alphafold_protein_graph_tutorial.html) | [Docs](http://graphein.ai/modules/graphein.protein.html#module-graphein.protein.graphs) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/residue_graphs.ipynb)|

```python
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.utils import download_alphafold_structure

config = ProteinGraphConfig()
fp = download_alphafold_structure("Q5VSL9", aligned_score=False)
g = construct_graph(config=config, pdb_path=fp)
```

### Creating a Protein Mesh

|   |   |
|---|---|
| [Tutorial](http://graphein.ai/notebooks/protein_mesh_tutorial.html) | [Docs](http://graphein.ai/modules/graphein.protein.html#module-graphein.protein.meshes) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/protein_mesh_tutorial.ipynb) | |

```python
from graphein.protein.config import ProteinMeshConfig
from graphein.protein.meshes import create_mesh

verts, faces, aux = create_mesh(pdb_code="3eiy", config=config)
```

### Creating an RNA Graph

|   |   |
|---|---|
|[Tutorial](http://graphein.ai/notebooks/rna_notebooks.html) | [Docs](http://graphein.ai/modules/graphein.rna.html) |
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/rna_graph_tutorial.ipynb) | |

```python
from graphein.rna.graphs import construct_rna_graph
# Build the graph from a dotbracket & optional sequence
rna = construct_rna_graph(dotbracket='..(((((..(((...)))..)))))...',
                          sequence='UUGGAGUACACAACCUGUACACUCUUUC')
```

### Creating a Protein-Protein Interaction Graph

|   |   |
|---|---|
| [Tutorial](http://graphein.ai/notebooks/ppi_tutorial.html) | [Docs](http://graphein.ai/modules/graphein.ppi.html) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/ppi_graph.ipynb)|

```python
from graphein.ppi.config import PPIGraphConfig
from graphein.ppi.graphs import compute_ppi_graph
from graphein.ppi.edges import add_string_edges, add_biogrid_edges

config = PPIGraphConfig()
protein_list = ["CDC42", "CDK1", "KIF23", "PLK1", "RAC2", "RACGAP1", "RHOA", "RHOB"]

g = compute_ppi_graph(config=config,
                      protein_list=protein_list,
                      edge_construction_funcs=[add_string_edges, add_biogrid_edges]
                     )
```

### Creating a Gene Regulatory Network Graph

|   |   |
|---|---|
|[Tutorial](http://graphein.ai/notebooks/grn_tutorial.html) | [Docs](http://graphein.ai/modules/graphein.grn.html) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/grn_tutorial.ipynb) |

```python
from graphein.grn.config import GRNGraphConfig
from graphein.grn.graphs import compute_grn_graph
from graphein.grn.edges import add_regnetwork_edges, add_trrust_edges

config = GRNGraphConfig()
gene_list = ["AATF", "MYC", "USF1", "SP1", "TP53", "DUSP1"]

g = compute_grn_graph(
    gene_list=gene_list,
    edge_construction_funcs=[
        partial(add_trrust_edges, trrust_filtering_funcs=config.trrust_config.filtering_functions),
        partial(add_regnetwork_edges, regnetwork_filtering_funcs=config.regnetwork_config.filtering_functions),
    ],
)
```

## Installation

### Pip

The simplest install is via pip. *N.B this does not install ML/DL libraries which are required for conversion to their data formats and for generating protein structure meshes with PyTorch 3D.* [Further details](http://graphein.ai//getting_started/installation.html)

```bash
pip install graphein # For base install
pip install graphein[extras] # For additional featurisation dependencies
pip install graphein[dev] # For dev dependencies
pip install graphein[all] # To get the lot
```

However, there are a number of (optional) utilities ([DSSP](https://anaconda.org/salilab/dssp), [PyMol](https://pymol.org/2/), [GetContacts](https://getcontacts.github.io/)) that are not available via PyPI:

```
conda install -c salilab dssp # Required for computing secondary structural features
conda install -c schrodinger pymol # Required for PyMol visualisations & mesh generation

# GetContacts - used as an alternative way to compute intramolecular interactions
conda install -c conda-forge vmd-python
git clone https://github.com/getcontacts/getcontacts

# Add folder to PATH
echo "export PATH=\$PATH:`pwd`/getcontacts" >> ~/.bashrc
source ~/.bashrc
To test the installation, run:

cd getcontacts/example/5xnd
get_dynamic_contacts.py --topology 5xnd_topology.pdb \
                        --trajectory 5xnd_trajectory.dcd \
                        --itypes hb \
                        --output 5xnd_hbonds.tsv
```

### Conda environment

The dev environment includes GPU Builds (CUDA 11.1) for each of the deep learning libraries integrated into graphein.

```bash
git clone https://www.github.com/a-r-j/graphein
cd graphein
conda env create -f environment-dev.yml
pip install -e .
```

A lighter install can be performed with:

```bash
git clone https://www.github.com/a-r-j/graphein
cd graphein
conda env create -f environment.yml
pip install -e .
```

### Dockerfile

We provide two `docker-compose` files for CPU (`docker-compose.cpu.yml`) and GPU usage (`docker-compose.yml`) locally. For GPU usage please ensure that you have [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed. Ensure that you install the locally mounted volume after entering the container (`pip install -e .`). This will also setup the dev environment locally.

To build (GPU) run:

```
docker-compose up -d --build # start the container
docker-compose down # stop the container
```

## Citing Graphein

Please consider citing graphein if it proves useful in your work.

```bibtex
@article {Jamasb2020.07.15.204701,
 author = {Jamasb, Arian R. and Vi{\~n}as, Ramon and Ma, Eric J. and Harris, Charlie and Huang, Kexin and Hall, Dominic and Li{\'o}, Pietro and Blundell, Tom L.},
 title = {Graphein - a Python Library for Geometric Deep Learning and Network Analysis on Protein Structures and Interaction Networks},
 elocation-id = {2020.07.15.204701},
 year = {2021},
 doi = {10.1101/2020.07.15.204701},
 publisher = {Cold Spring Harbor Laboratory},
 abstract = {Geometric deep learning has well-motivated applications in the context of biology, a domain where relational structure in datasets can be meaningfully leveraged. Currently, efforts in both geometric deep learning and, more broadly, deep learning applied to biomolecular tasks have been hampered by a scarcity of appropriate datasets accessible to domain specialists and machine learning researchers alike. However, there has been little exploration of how to best to integrate and construct geometric representations of these datatypes. To address this, we introduce Graphein as a turn-key tool for transforming raw data from widely-used bioinformatics databases into machine learning-ready datasets in a high-throughput and flexible manner. Graphein is a Python library for constructing graph and surface-mesh representations of protein structures and biological interaction networks for computational analysis. Graphein provides utilities for data retrieval from widely-used bioinformatics databases for structural data, including the Protein Data Bank, the recently-released AlphaFold Structure Database, and for biomolecular interaction networks from STRINGdb, BioGrid, TRRUST and RegNetwork. The library interfaces with popular geometric deep learning libraries: DGL, PyTorch Geometric and PyTorch3D though remains framework agnostic as it is built on top of the PyData ecosystem to enable inter-operability with scientific computing tools and libraries. Graphein is designed to be highly flexible, allowing the user to specify each step of the data preparation, scalable to facilitate working with large protein complexes and interaction graphs, and contains useful pre-processing tools for preparing experimental files. Graphein facilitates network-based, graph-theoretic and topological analyses of structural and interaction datasets in a high-throughput manner. As example workflows, we make available two new protein structure-related datasets, previously unused by the geometric deep learning community. We envision that Graphein will facilitate developments in computational biology, graph representation learning and drug discovery.Availability and implementation Graphein is written in Python. Source code, example usage and tutorials, datasets, and documentation are made freely available under the MIT License at the following URL: graphein.aiCompeting Interest StatementThe authors have declared no competing interest.},
 URL = {https://www.biorxiv.org/content/early/2021/10/12/2020.07.15.204701},
 eprint = {https://www.biorxiv.org/content/early/2021/10/12/2020.07.15.204701.full.pdf},
 journal = {bioRxiv}
}

```
