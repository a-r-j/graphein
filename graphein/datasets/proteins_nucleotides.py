"""Class for working with the PROTEINS_NUCLEOTIDES dataset"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Code Repository: https://github.com/a-r-j/graphein

from functools import lru_cache
from pathlib import Path

import pandas as pd

from graphein.datasets.base import AbstractClassificationDataset


class PROTEINS_NUCLEOTIDES(AbstractClassificationDataset):
    @lru_cache
    def _load_dataset(self=None) -> pd.DataFrame:
        file_path = (
            Path(__file__).parent.parent.parent.resolve()
            / "datasets"
            / "proteins_nucleotides"
            / "PROTEINS_NUCLEOTIDES.csv"
        )
        df = pd.read_csv(file_path)
        return df

    def split_data(
        self,
        strategy: str = "random",
        training_size: float = 0.7,
        validation_size: float = 0.2,
        test_size: float = 0.1,
        shuffle: bool = True,
    ):
        train_df, val_df, test_df = self._split_data(
            strategy, training_size, validation_size, test_size, shuffle
        )

        self.train = PROTEINS_NUCLEOTIDES(train_df)
        self.validation = PROTEINS_NUCLEOTIDES(val_df)
        self.test = PROTEINS_NUCLEOTIDES(test_df)


if __name__ == "__main__":
    c = PROTEINS_NUCLEOTIDES()
