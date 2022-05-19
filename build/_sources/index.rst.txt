.. Graphein documentation master file, created by
   sphinx-quickstart on Mon Jun  8 18:43:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Graphein's documentation!
====================================
This package provides functionality for producing a number of types of graph-based representations of proteins. We provide compatibility with standard geometric deep learning library formats (currently: NetworkX nx.Graph, pytorch_geometric.data.Data and dgl.DGLGraph), as well as graph objects designed for ease of use with popular deep learning libraries.

The repository can be found at `a-r-j/graphein <https://www.github.com/a-r-j/graphein>`_

.. note::
   This is an early-stage project and a lot more documentation and functionality is planned to be included. If you are a structural biologist or machine learning researcher in computational biology, my inbox is always open for suggestions and assistance!

.. include:: readme.rst


.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   getting_started/installation
   getting_started/introduction
   getting_started/usage
   license


.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   protein_notebooks
   rna_notebooks
   ppi_notebooks
   grn_notebooks

.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:
   :caption: Machine Learning

   datasets

.. toctree::
   :glob:
   :maxdepth: 4
   :hidden:
   :caption: API Reference

   modules/graphein.protein
   modules/graphein.rna
   modules/graphein.ppi
   modules/graphein.grn
   modules/graphein.ml
   modules/graphein.utils

.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:
   :caption: Contributing

   contributing/contributing
   contributing/code_of_conduct
   contributing/contributors


If Graphein proves useful to your work, please consider citing:

.. code-block:: latex

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



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
