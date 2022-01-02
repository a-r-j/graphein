[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/a-r-j/graphein-binder/master?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252Fa-r-j%252Fgraphein%26urlpath%3Dlab%252Ftree%252Fgraphein%252Fnotebooks%26branch%3Dmaster)
[![PyPI version](https://badge.fury.io/py/graphein.svg)](https://badge.fury.io/py/graphein)
![supported python versions](https://img.shields.io/pypi/pyversions/graphein)
[![Docs](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](http://www.graphein.ai)
[![DOI:10.1101/2020.07.15.204701](https://zenodo.org/badge/DOI/10.1101/2020.07.15.204701.svg)](https://doi.org/10.1101/2020.07.15.204701)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
<a href="https://github.com/badges/shields/pulse" alt="Activity">
[![CodeFactor](https://www.codefactor.io/repository/github/a-r-j/graphein/badge)](https://www.codefactor.io/repository/github/a-r-j/graphein)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=a-r-j_graphein&metric=alert_status)](https://sonarcloud.io/dashboard?id=a-r-j_graphein)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=a-r-j_graphein&metric=bugs)](https://sonarcloud.io/dashboard?id=a-r-j_graphein)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=a-r-j_graphein&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=a-r-j_graphein)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=a-r-j_graphein&metric=reliability_rating)](https://sonarcloud.io/dashboard?id=a-r-j_graphein)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/graphein)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![banner](docs/source/_static/graphein.png)](http://www.graphein.ai)

<br></br>

[Documentation](http://www.graphein.ai) | [Paper](https://www.biorxiv.org/content/10.1101/2020.07.15.204701v1) | [Tutorials](http://graphein.ai/notebooks_index.html) | [Installation](#installation)

Protein & Interactomic Graph Library

This package provides functionality for producing geometric representations of protein and RNA structures, and biological interaction networks. We provide compatibility with standard PyData formats, as well as graph objects designed for ease of use with popular deep learning libraries.

## What's New?
* [Protein Graph Creation from AlphaFold2!](http://graphein.ai/notebooks/alphafold_protein_graph_tutorial.html)
* [Protein Graph Visualisation!](http://graphein.ai/notebooks/protein_mesh_tutorial.html)
* [RNA Graph Construction from Dotbracket notation](http://graphein.ai/modules/graphein.rna.html)
* [Protein - Protein Interaction Network Support & Structural Interactomics (Using AlphaFold2!)](http://graphein.ai/notebooks/ppi_tutorial.html)
* [High and Low-level API for massive flexibility - create your own bespoke workflows!](http://graphein.ai/notebooks/residue_graphs.html)

## Example usage
### Creating a Protein Graph
[Tutorial (Residue-level)](http://graphein.ai/notebooks/residue_graphs.html) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/residue_graphs.ipynb) | [Tutorial - Atomic](http://graphein.ai/notebooks/atom_graph_tutorial.html) [![Open In Colab(https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/atom_graph_tutorial.ipynb) | [Docs](http://graphein.ai/modules/graphein.protein.html#module-graphein.protein.graphs)

```python
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph

config = ProteinGraphConfig()
g = construct_graph(config=config, pdb_code="3eiy")
```

### Creating a Protein Graph from the AlphaFold Protein Structure Database
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-r-j/graphein/blob/master/notebooks/residue_graphs.ipynb) [Tutorial](http://graphein.ai/notebooks/alphafold_protein_graph_tutorial.html) | [Docs](http://graphein.ai/modules/graphein.protein.html#module-graphein.protein.graphs)
```python
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.utils import download_alphafold_structure

config = ProteinGraphConfig()
fp = download_alphafold_structure("Q5VSL9", aligned_score=False)
g = construct_graph(config=config, pdb_path=fp)
```

### Creating a Protein Mesh
[Tutorial](http://graphein.ai/notebooks/protein_mesh_tutorial.html) | [Docs](http://graphein.ai/modules/graphein.protein.html#module-graphein.protein.meshes)
```python
from graphein.protein.config import ProteinMeshConfig
from graphein.protein.meshes import create_mesh

verts, faces, aux = create_mesh(pdb_code="3eiy", config=config)
```
### Creating an RNA Graph
Tutorial | [Docs](http://graphein.ai/modules/graphein.rna.html)
```python
from graphein.rna.graphs import construct_rna_graph
# Build the graph from a dotbracket & optional sequence
rna = construct_rna_graph(dotbracket='..(((((..(((...)))..)))))...',
                          sequence='UUGGAGUACACAACCUGUACACUCUUUC')
```

### Creating a Protein-Protein Interaction Graph
[Tutorial](http://graphein.ai/notebooks/ppi_tutorial.html) | [Docs](http://graphein.ai/modules/graphein.ppi.html)
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
[Tutorial](http://graphein.ai/notebooks/grn_tutorial.html) | [Docs](http://graphein.ai/modules/graphein.grn.html)
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
The simplest install is via pip. *N.B this does not install ML/DL libraries which are required for conversion to their data formats and for generating protein structure meshes with PyTorch 3D.* [Further details]
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
We also provide a [Dockerfile](https://github.com/a-r-j/graphein/pull/69)

## Citing Graphein

Please consider citing graphein if it proves useful in your work.

```biblatex
@article{Jamasb2020,
  doi = {10.1101/2020.07.15.204701},
  url = {https://doi.org/10.1101/2020.07.15.204701},
  year = {2020},
  month = jul,
  publisher = {Cold Spring Harbor Laboratory},
  author = {Arian Rokkum Jamasb and Pietro Lio and Tom Blundell},
  title = {Graphein - a Python Library for Geometric Deep Learning and Network Analysis on Protein Structures}
}
```
