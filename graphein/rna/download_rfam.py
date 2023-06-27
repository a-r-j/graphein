from typing import List, Optional

import requests
import pandas as pd
from tqdm import tqdm

# RFAM API endpoint to retrieve family information
RFAM_API_URL = 'https://rfam.org/family'


class FamilyNotFound(ValueError):
    pass


def _get_RFAM_family_df(family_id: str):
    """
    Downloads DataFrame of PDB IDs annotated with RFAM families
    :param family_id: RFAM ID
    :type family_id: str
    :return: Pandas DataFrame with information about the structures of the RFAM family
    :rtype: pd.DataFrame
    """
    # Send an HTTP GET request to retrieve the data
    response = requests.get(f'{RFAM_API_URL}/{family_id}/structures?content-type=application/json')

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract the family names from the response
        data = response.json()
        df = pd.DataFrame(data['mapping'])
    else:
        raise FamilyNotFound(
            f'Error occurred while retrieving data for family {family_id} (status code: {response.status_code})')

    return df


def RFAM_families_df(family_ids: Optional[List[str]] = None,
                     max_id: int = 4236,
                     verbose=True):
    """
    Retrieves a DataFrame of PDB IDs annotated with RFAM families
    :param family_ids: List of families to retrieve. If None, retrieves all families: RF00001, RF00002, ..., RF04236
    (we assume that RFAM family IDs are in increasing order, see: http://http.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/)
    :type family_ids: Optional[List[str]]
    :param max_id: Maximum identifier to try. If family_ids is None, it'll query RFAM families RF00001 .. f'RF{max_id:05d}'
    :type max_id: int
    :param verbose: Whether to print messages
    :type verbose: bool
    :return: Pandas DataFrame with information about the structures of each RFAM family
    :rtype: pd.DataFrame
    """
    if family_ids is None:
        family_ids = [f'RF{i:05d}' for i in range(1, max_id + 1)]

    families_df = None
    if verbose:
        print('Retrieving RFAM families ...')
    for family_id in tqdm(family_ids):
        # Retrieve DF for a single family
        try:
            df = _get_RFAM_family_df(family_id)

            # Concatenate
            if families_df is None:
                families_df = df
            else:
                families_df = pd.concat([families_df, df])
        except FamilyNotFound as e:
            if verbose:
                print(e)
            continue
    return families_df


if __name__ == '__main__':
    family_IDs = None  # ['RF10000']
    families_df = RFAM_families_df(family_IDs)
    print(families_df)
    # families_df.to_csv('RFAM_families_27062023.csv', index=False)
