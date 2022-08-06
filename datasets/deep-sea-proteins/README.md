# Deep-Sea Protein Structure Dataset

The prediction of molecular protein adaptations is a key challenge in protein engineering. In particular, proteins of extremophiles often exhibit desirable properties, like a tolerance to extremely high temperature and/or pressure. A promising resource for such proteins is the deep-sea, which is the largest extreme environment on earth. In the last years, through large-scale metagenomic projects, increasing protein data from these environments has been provided. Not surprisingly, there is a great interest in systematically analyzing the data currently available.

## Dataset Composition

We compiled a data set of 1281 experimental protein structures from 25 deep-sea organisms from the Protein Databank (PDB) and paired them with orthologous proteins. This data set is one of the first to provide protein structure pairs for building data-driven methods and analyzing structural protein adaptations to the extreme environmental conditions in the deep-sea. We thoroughly removed redundancy and processed the data set into cross-validation folds for easy use in machine learning. We also annotated the protein pairs by the environmental preferences of the deep-sea and decoy source organisms. In this way, thermopiles, mesophiles and piezophiles can be compared directly. The final data set includes 501 deep-sea protein chains and 8200 decoy protein chains that come from 20 different deep-sea and 1379 decoy organisms and form 17 148 pairs. For further details and a machine learning-based analysis of the data set, see [1].

[1] <https://onlinelibrary.wiley.com/doi/10.1002/prot.26337>

[Description source](https://www.zbh.uni-hamburg.de/en/forschung/amd/datasets/deep-sea-protein-structure.html)
