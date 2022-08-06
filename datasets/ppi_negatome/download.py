"""Downloads the PPI-Negatome dataset."""
import wget


def download_dataset():
    # Download manual
    wget.download(
        "http://mips.helmholtz-muenchen.de/proj/ppi/negatome/manual.txt"
    )
    # Download manual-stringent
    wget.download(
        "http://mips.helmholtz-muenchen.de/proj/ppi/negatome/manual_stringent.txt"
    )
    # Download manual-pfam
    wget.download(
        "http://mips.helmholtz-muenchen.de/proj/ppi/negatome/manual_pfam.txt"
    )
    # Download PDB
    wget.download(
        "http://mips.helmholtz-muenchen.de/proj/ppi/negatome/pdb.txt"
    )
    # Download PDB stringent
    wget.download(
        "http://mips.helmholtz-muenchen.de/proj/ppi/negatome/pdb_stringent.txt"
    )
    # Download PDB-Pfam
    wget.download(
        "http://mips.helmholtz-muenchen.de/proj/ppi/negatome/pdb_pfam.txt"
    )
    # Download combined
    wget.download(
        "http://mips.helmholtz-muenchen.de/proj/ppi/negatome/combined.txt"
    )
    # Download combined stringent
    wget.download(
        "http://mips.helmholtz-muenchen.de/proj/ppi/negatome/combined_stringent.txt"
    )
    # Download combined stringent
    wget.download(
        "http://mips.helmholtz-muenchen.de/proj/ppi/negatome/combined_pfam.txt"
    )


if __name__ == "__main__":
    download_dataset()
