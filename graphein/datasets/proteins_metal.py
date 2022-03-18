"""Class for working with the PROTEINS_METAL dataset"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Code Repository: https://github.com/a-r-j/graphein

from functools import lru_cache
from pathlib import Path

import pandas as pd

from graphein.datasets.base import AbstractClassificationDataset
from graphein.protein.config import ProteinGraphConfig


class PROTEINS_METAL(AbstractClassificationDataset):
    @lru_cache
    def _load_dataset(self=None) -> pd.DataFrame:
        file_path = (
            Path(__file__).parent.parent.parent.resolve()
            / "datasets"
            / "proteins_metal"
            / "PROTEINS_METAL.csv"
        )
        return pd.read_csv(file_path)

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

        self.train: PROTEINS_METAL = PROTEINS_METAL(train_df)
        self.validation: PROTEINS_METAL = PROTEINS_METAL(val_df)
        self.test: PROTEINS_METAL = PROTEINS_METAL(test_df)


if __name__ == "__main__":
    config = ProteinGraphConfig(keep_hets=True)
    c = PROTEINS_METAL()
    c.split_data()
    c.train.construct_graphs(config)

    print(c.df)
