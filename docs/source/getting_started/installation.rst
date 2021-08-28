Installation
============
Graphein depends on a number of other libraries for constructing graphs, meshes and adding features.

The source code for Graphein can be viewed at `a-r-j/graphein <https://www.github.com/a-r-j/graphein>`_

We are actively working on a simpler install path (via conda). However, we have a few core dependencies that are proving difficult.

Core Install
---------------------

At present, the simplest installation is as follows:

.. code-block:: bash

    git clone https://www.github.com/a-r-j/graphein
    cd graphein
    conda env create -f environment.yml
    pip install .


Dev Install
---------------------

Alternatively, if you wish to install Graphein in the dev environment (includes GPU build of all relevant geometric deep learning libraries) you can run:

.. code-block:: bash

    git clone https://www.github.com/a-r-j/graphein
    cd graphein
    conda env create -f environment-dev.yml
    pip install -e .

Optional Dependencies
---------------------
.. note::
    Some of these packages have more involved setup depending on your requirements (i.e. CUDA). Please refer to the original packages for more detailed information


Installing Deep Learning Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda install -c pytorch pytorch
    conda install -c pytorch3d pytorch3d #  NB requires fvcore and iopath
    conda install -c dglteam dgl
    conda install pytorch-geometric -c rusty1s -c conda-forge


GetContacts
^^^^^^^^^^^^^^

GetContacts is an optional dependency for computing intramolecular contacts in `.pdb` files. We provide distance-based heuristics for this in `graphein.protein.edges.distance` so this is not a hard requirement.

Please see the `GetContacts documentation <https://getcontacts.github.io/getting_started.html>_` for up-to-date installation instructions.

MacOS:

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

Linux:

.. code-block:: bash

  # Install get_contact_ticc.py dependencies
  conda install scipy numpy scikit-learn matplotlib pandas cython
  pip install ticc==0.1.4

  # Set up vmd-python library
  conda install -c https://conda.anaconda.org/rbetz vmd-python

  # Set up getcontacts library
  git clone https://github.com/getcontacts/getcontacts.git
  echo "export PATH=`pwd`/getcontacts:\$PATH" >> ~/.bashrc
  source ~/.bashrc

