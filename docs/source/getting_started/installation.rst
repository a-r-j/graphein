Installation
============
Graphein depends on a number of other libraries for constructing graphs, meshes and adding features.

The source code for Graphein can be viewed at `a-r-j/graphein <https://www.github.com/a-r-j/graphein>`_

.. note::
    For full functionality, there are a number of additional (optional) installs for libraries not available via pip/conda. Please see below

Core Install
---------------------



At present, the simplest installation is via `PyPI <https://pypi.org/project/graphein/>`_ . The base install can also be performed using the provided `conda environment <https://github.com/a-r-j/graphein/blob/master/environment.yml>`_.

.. tab:: Pip

    .. code-block:: bash

        pip install graphein         # For base install
        pip install graphein[extras] # For additional featurisation dependencies


.. tab:: Conda

    .. code-block:: bash

        git clone https://www.github.com/a-r-j/graphein
        cd graphein
        conda env create -f environment.yml
        pip install .


Docker Install
---------------------

We provide two ``docker-compose`` files for `CPU <https://github.com/a-r-j/graphein/blob/master/docker-compose.cpu.yml>`_ (``docker-compose.cpu.yml``) and `GPU <https://github.com/a-r-j/graphein/blob/master/docker-compose.cpu.yml>`_ usage (``docker-compose.yml``) locally. For GPU usage please ensure that you have `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ installed. Ensure that you install the locally mounted volume after entering the container (``pip install -e .``). **This will also setup the dev environment locally**.

The Dockerfile is viewable `here <https://github.com/a-r-j/graphein/blob/master/Dockerfile>`_


.. tab:: CPU

    .. code-block:: bash

        docker-compose.cpu up -d --build    # start the container
        docker-compose.cpu down             # stop the container


.. tab:: GPU

    .. code-block:: bash

        docker-compose up -d --build        # start the container
        docker-compose down                 # stop the container

Dev Install
---------------------
The Dev install of Graphein contains additional dependendencies for development, such as testing frameworks and documentation tools. If you wish to contribute to Graphein, this is the installation method you should use.

Alternatively, if you wish to install Graphein in the dev environment (includes **GPU** builds (``CUDA`` 11.1) of all relevant geometric deep learning libraries) you can use the provided `conda environment <https://github.com/a-r-j/graphein/blob/master/environment-dev.yml>`_:

.. tab:: Pip

    .. code-block:: bash

        pip install graphein[dev] # For dev dependencies
        pip install graphein[all] # To get the lot

.. tab:: Conda

    .. code-block:: bash

        git clone https://www.github.com/a-r-j/graphein
        cd graphein
        conda env create -f environment-dev.yml
        pip install -e .  # Install in editable mode



Devcontainer
^^^^^^^^^^^^^
We `provide a devcontainer <https://github.com/a-r-j/graphein/tree/master/.devcontainer>`_ for the dev environment. This is a lightweight container that can be used to run the dev environment locally.

`More information about devcontainers <https://code.visualstudio.com/docs/remote/containers>`_

Optional Dependencies
---------------------
However, there are a number of (optional) utilities `DSSP <https://anaconda.org/salilab/dssp>`_, `PyMol <https://pymol.org/2/>`_, `GetContacts <https://getcontacts.github.io/>`_ that are not available via PyPI:

.. code-block:: bash

    conda install -c salilab dssp # Required for computing secondary structural features
    conda install -c schrodinger pymol # Required for PyMol visualisations & mesh generation

.. note::
    Some of these packages have more involved setup depending on your requirements (i.e. ``CUDA``). Please refer to the original packages for more detailed information


Installing Deep Learning Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Due to the many possible configurations of deep learning libraries, we deliberately do not provide a single install via PyPI. However, the conda dev environment described above contains GPU builds for CUDA 11.1 and PyTorch. The ``Dockerfile`` for the GPU build is provided in the ``docker-compose.yml`` file.

.. code-block:: bash

    conda install -c pytorch pytorch
    conda install -c pytorch3d pytorch3d #  NB requires fvcore and iopath
    conda install -c dglteam dgl
    conda install pytorch-geometric -c rusty1s -c conda-forge


GetContacts
^^^^^^^^^^^^^^

``GetContacts`` is an optional dependency for computing intramolecular contacts in ``.pdb`` files. We provide distance-based heuristics for this in ``graphein.protein.edges.distance`` so this is not a hard requirement.

Please see the `GetContacts documentation <https://getcontacts.github.io/getting_started.html>`_ for up-to-date installation instructions.


.. tab:: MacOS

    .. code-block:: bash

        # Install get_contact_ticc.py dependencies
        conda install scipy numpy scikit-learn matplotlib pandas cython seaborn
        pip install ticc==0.1.4

        # Install vmd-python dependencies
        conda install netcdf4 numpy pandas seaborn  expat tk=8.5  # Alternatively use pip
        brew install netcdf pyqt # Assumes https://brew.sh/ is installed

        # Set up vmd-python library
        git clone https://github.com/Eigenstate/vmd-python.git
        cd vmd-python
        python setup.py build
        python setup.py install
        cd ..

        # Set up getcontacts library
        git clone https://github.com/getcontacts/getcontacts.git
        echo "export PATH=`pwd`/getcontacts:\$PATH" >> ~/.bash_profile
        source ~/.bash_profile

        # Test installation
        cd getcontacts/example/5xnd
        get_dynamic_contacts.py --topology 5xnd_topology.pdb \
                                --trajectory 5xnd_trajectory.dcd \
                                --itypes hb \
                                --output 5xnd_hbonds.tsv

.. tab:: Linux

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
