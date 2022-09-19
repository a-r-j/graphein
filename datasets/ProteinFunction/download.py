"""Download utility for ProteinFunction dataset from Hermosilla et al.

https://github.com/phermosilla/IEConv_proteins/tree/master/Datasets/data/ProtFunct
"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

import wget

BASE_URL: str = "https://raw.githubusercontent.com/phermosilla/IEConv_proteins/master/Datasets/data/ProtFunct/"


def download_data():
    # Download chain_functions
    wget.download(f"{BASE_URL}chain_functions.txt")
    wget.download(f"{BASE_URL}unique_functions.txt")
    wget.download(f"{BASE_URL}training.txt")
    wget.download(f"{BASE_URL}validation.txt")
    wget.download(f"{BASE_URL}testing.txt")


if __name__ == "__main__":
    download_data()
