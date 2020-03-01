# Graphein
Protein Graph Library

## Installation
### MacOS
Create env
```
conda create --name graphein
conda activate graphein
```

Install Get Contacts
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