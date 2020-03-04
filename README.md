# Graphein
Protein Graph Library

This pckage provides functionality for producing a number of types of graph-based representations of proteins. We provide compatibility with standard formats, as well as graph objects designed for ease of use in deep learning.

## Installation
### MacOS
Create env
```
conda create --name graphein
conda activate graphein
```
1. Install `vmd-python`

`conda install -c conda-forge vmd-python`

**N.B.** if you are not on linux you will have to compile from source.

https://github.com/Eigenstate/vmd-python

**N.B.** if you do not compile from source (e.g. use `brew install netcdf` you will need to export the path i.e  `export PATH="path_to_your_netcdf:$PATH"`. `brew` will typically use `/user/local/opt/netcdf/`)

2. Install Get Contacts
```
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
```

Install residual requirements

```
pip install -r requirements.txt
```

### Linux
Install GetContacts
```
  # Install get_contact_ticc.py dependencies
  conda install scipy numpy scikit-learn matplotlib pandas cython
  pip install ticc==0.1.4
  
  # Set up vmd-python library
  conda install -c https://conda.anaconda.org/rbetz vmd-python
  
  # Set up getcontacts library
  git clone https://github.com/getcontacts/getcontacts.git
  echo "export PATH=`pwd`/getcontacts:\$PATH" >> ~/.bashrc
  source ~/.bashrc
```