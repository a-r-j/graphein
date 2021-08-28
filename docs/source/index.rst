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

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation
   getting_started/introduction
   license


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Tutorials

   notebooks_index

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Datasets

   datasets

.. toctree::
   :glob:
   :maxdepth: 4
   :caption: API Reference

   modules/graphein.protein
   modules/graphein.rna
   modules/graphein.ppi
   modules/graphein.grn
   modules/graphein.ml



If Graphein proves useful to your work, please consider citing:

.. code-block:: latex

   @article{Jamasb2020,
     doi = {10.1101/2020.07.15.204701},
     url = {https://doi.org/10.1101/2020.07.15.204701},
     year = {2020},
     month = jul,
     publisher = {Cold Spring Harbor Laboratory},
     author = {Arian Rokkum Jamasb and Pietro Lio and Tom Blundell},
     title = {Graphein - a Python Library for Geometric Deep Learning and Network Analysis on Protein Structures}
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
