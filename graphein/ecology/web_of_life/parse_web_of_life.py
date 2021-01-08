import pandas as pd
from pathlib import Path
from functools import lru_cache

@lru_cache()
def load_references():
    return pd.read_csv((Path(__file__).parent / "data" / ("references.csv")).absolute())
    


def parse_dataset(dataset_name: str) -> pd.DataFrame:
    ref = load_references()
    print(ref)
    return pd.read_csv((Path(__file__).parent / "data" / (dataset_name + ".csv")).absolute())


if __name__ == "__main__":
    df = parse_dataset("A_HP_001")

    print(df)
