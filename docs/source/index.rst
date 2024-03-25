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
   molecule_notebooks
   rna_notebooks
   ppi_notebooks
   grn_notebooks

.. toctree::
   :glob:
   :maxdepth: 2
   :hidden:
   :caption: Machine Learning

   ml_protein_tensors
   datasets
   ml_examples

.. toctree::
   :glob:
   :maxdepth: 4
   :hidden:
   :caption: API Reference

   modules/graphein.protein
   modules/graphein.protein.tensor
   modules/graphein.molecule
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

   @inproceedings{jamasb2022graphein,
      title={Graphein - a Python Library for Geometric Deep Learning and Network Analysis on Biomolecular Structures and Interaction Networks},
      author={Arian Rokkum Jamasb and Ramon Vi{\~n}as Torn{\'e} and Eric J Ma and Yuanqi Du and Charles Harris and Kexin Huang and Dominic Hall and Pietro Lio and Tom Leon Blundell},
      booktitle={Advances in Neural Information Processing Systems},
      editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
      year={2022},
      url={https://openreview.net/forum?id=9xRZlV6GfOX}
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
