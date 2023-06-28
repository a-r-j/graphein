import gzip
import os
import shutil
from pathlib import Path

import pandas as pd
import wget

from loguru import logger as log


class RFAMManager:
    """ A utility for downloading RFAM families and their PDB structure IDs."""

    def __init__(
        self,
        root_dir: str = ".",
    ):
        # Arguments
        self.root_dir = Path(root_dir)

        # Constants
        self.rfam_families_url = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/database_files/family.txt.gz"
        self.rfam_pdb_mapping_url = (
            "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.pdb.gz"
        )

        self.rfam_dir = self.root_dir / "rfam"
        if not os.path.exists(self.rfam_dir):
            os.makedirs(self.rfam_dir)

        self.rfam_families_archive_filename = Path(self.rfam_families_url).name
        self.rfam_families_filename = Path(self.rfam_families_url).stem
        self.rfam_pdb_mapping_archive_filename = Path(
            self.rfam_pdb_mapping_url
        ).name
        self.rfam_pdb_mapping_filename = Path(self.rfam_pdb_mapping_url).stem

        self.download_metadata()

    def download_metadata(self):
        """ Download metadata mapping PDB structures to RFAM families """
        self._download_rfam_families()
        self._download_rfam_pdb_mapping()

    def _download_rfam_families(self):
        """Download RFAM families from
        https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/database_files/family.txt.gz
        """
        if not os.path.exists(self.rfam_dir / self.rfam_families_filename):
            log.info("Downloading RFAM families...")
            wget.download(self.rfam_families_url, out=str(self.rfam_dir))
            log.info("Downloaded RFAM families")

        # Unzip all collected families
        if not os.path.exists(self.rfam_dir / self.rfam_families_filename):
            log.info("Unzipping RFAM sequences...")
            with gzip.open(
                self.rfam_dir / self.rfam_families_archive_filename, "rb"
            ) as f_in:
                with open(
                    self.rfam_dir / self.rfam_families_filename, "wb"
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            log.info("Unzipped RFAM families")

    def _download_rfam_pdb_mapping(self):
        """Download RFAM families from
        https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/database_files/Rfam.pdb.gz
        """
        if not os.path.exists(self.rfam_dir / self.rfam_pdb_mapping_filename):
            log.info("Downloading RFAM family - PDB structure ID mapping ...")
            wget.download(self.rfam_pdb_mapping_url, out=str(self.rfam_dir))
            log.info("Downloaded RFAM family - PDB structure ID mapping")

        # Unzip all collected mappings
        if not os.path.exists(self.rfam_dir / self.rfam_pdb_mapping_filename):
            log.info("Unzipping RFAM family - PDB structure ID mapping...")
            with gzip.open(
                self.rfam_dir / self.rfam_pdb_mapping_archive_filename, "rb"
            ) as f_in:
                with open(
                    self.rfam_dir / self.rfam_pdb_mapping_filename, "wb"
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            log.info("Unzipped RFAM family - PDB structure ID mapping")

    def _parse_rfam_families(self) -> pd.DataFrame:
        """Parse the RFAM families metadata

        :return: Pandas DataFrame with information about the RFAM families
        :rtype: pd.DataFrame
        """
        df = pd.read_csv(
            self.rfam_dir / self.rfam_families_filename,
            sep="\t",
            header=None,
            encoding="ISO-8859-1",
        )
        # Selecting accession, ID, and description
        df = df[
            [0, 1, 3]
        ]  # TODO: Could select other fields such as comment for an extended description of the family?
        df.columns = ["rfam_acc", "id", "description"]
        df = df.set_index("rfam_acc")
        return df

    def _parse_rfam_pdb_mapping(self) -> pd.DataFrame:
        """Parse the PDB IDs annotated with RFAM families

        :return: Pandas DataFrame with information about the structures of the RFAM family
        :rtype: pd.DataFrame
        """
        df = pd.read_csv(
            self.rfam_dir / self.rfam_pdb_mapping_filename, sep="\t", header=0
        )
        return df

    def parse_rfam(self) -> pd.DataFrame:
        """Parse mapping between PDB structures and RFAM families """
        family_info_df = self._parse_rfam_families()
        rfam_pdb_mapping_df = self._parse_rfam_pdb_mapping()
        df = pd.merge(
            rfam_pdb_mapping_df,
            family_info_df,
            left_on="rfam_acc",
            right_index=True,
        )
        return df


if __name__ == "__main__":
    rfam_manager = RFAMManager()
    df = rfam_manager.parse_rfam()
    print(df.head())
