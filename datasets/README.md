# Datasets

Here we provide a few worked examples of creating graph datasets from protein structures

## PPISP
The data contained within PPISP is drawn from DeepPPISP [1]. They collate a number of protein-protein interaction structures from three existing datasets. This is a node-classification task, where the task to is to predict whether or not a residue in the graph participates in a protein-protein interaction. The authors make available additional evolutionary information in the form of a PSSM for each protein.

The authors describe the dataset constuction as follows: The three benchmark datasets are given, i.e., Dset_186, Dset_72 and PDBset_164. Dset_186 consists of 186 protein sequences with the resolution less than 3.0 Å with sequence homology less than 25%. Dset_72 and PDBset_164 were constructed as the same as Dset_186. Dset_72 has 72 protein sequences and PDBset_164 consists of 164 protein sequences. These protein sequences in the three benchmark datasets have been annotated. Thus, we have 422 different annotated protein sequences. We remove two protein sequences for they do not have PSSM file.

## PSCDB
The data contained with PSCDB is drawn from the Protein Structural Change Database [2] . The dataset consists of paired protein structures in their bound and unbound forms across 7 classes of structural rearrangement motion. Several tasks can be formulated with this dataset. E.g. predicting the bound conformation of a protein as and edge-prediction task or graph-classification task predicting which class of structural rearrangement a protein undergoes upon ligand binding.


## References
[1] Min Zeng, Fuhao Zhang, Fang-Xiang Wu, Yaohang Li, Jianxin Wang, Min Li. Protein-protein interaction site prediction through combining local and global features with deep neural networks. Bioinformatics. DOI:10.1093/bioinformatics/btz699

[2] Amemiya, T., Koike, R., Kidera, A., & Ota, M. (2011). PSCDB: a database for protein structural change upon ligand binding. Nucleic Acids Research, 40(D1), D554–D558. https://doi.org/10.1093/nar/gkr966