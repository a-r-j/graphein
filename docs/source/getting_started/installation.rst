Installation
===========
Graphein depends on a number of other libraries for constructing protein graphs and meshes. These should be installed in advance.

.. note::
    We recommend installing Graphein in a virtual environment.
    ..

.. note::
    Some of these packages have more involved setup depending on your requirements (i.e. CUDA). Please refer to the original packages for more detailed information

.. code-block::

    conda create -n graphein python=3.7

Installing PyTorch Libraries
-----------------------------

.. code-block:: bash

    pip install torch
    pip install dgl
    pip install pytorch3d

Installing Pytorch Geometric
------------------------------
.. code-block:: bash

    $ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    $ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    $ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    $ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
    $ python setup.py install or pip install torch-geometric

Install all needed packages with ${CUDA} replaced by either cpu, cu92, cu100 or cu101 depending on your PyTorch installation:

GetContacts Requirements
------------------------

.. code-block:: bash

     # Install get_contact_ticc.py dependencies
     $ conda install scipy numpy scikit-learn matplotlib pandas cython seaborn
     $ pip install ticc==0.1.4

     # Install vmd-python dependencies
     $ conda install netcdf4 numpy pandas seaborn  expat tk=8.5  # Alternatively use pip
     $ brew install netcdf pyqt # Assumes https://brew.sh/ is installed

     # Set up vmd-python library
     $ git clone https://github.com/Eigenstate/vmd-python.git
     $ cd vmd-python
     $ python setup.py build
     $ python setup.py install
     $ cd ..

     # Set up getcontacts library
     $ git clone https://github.com/getcontacts/getcontacts.git
     $ echo "export PATH=`pwd`/getcontacts:\$PATH" >> ~/.bash_profile
     $ source ~/.bash_profile

     # Test installation
     $ cd getcontacts/example/5xnd
     $ get_dynamic_contacts.py --topology 5xnd_topology.pdb \
                               --trajectory 5xnd_trajectory.dcd \
                               --itypes hb \
                               --output 5xnd_hbonds.tsv

Install DSSP
------------

We use DSSP to compute secondary structure features of proteins.

.. code-block:: bash

    conda install -c salilab dssp

IPyMol
------

Install IPyMol from GitHub. The release on PyPI appears to behind the repository and some required functionality is unavailable.
https://github.com/cxhernandez/ipymol

.. code-block:: bash

    git clone https://github.com/cxhernandez/ipymol
    cd ipymol
    pip install .

Install Graphein
----------------

.. code-block:: bash

    git clone https://github.com/a-r-j/grahein
    cd graphein
    pip install -e .





