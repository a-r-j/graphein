import zipfile

import wget


def download_pcddb():
    URL = "https://pcddb.cryst.bbk.ac.uk/dl/pcddb_pcd_and_gen.zip"
    wget.download(URL)
    with zipfile.ZipFile("pcddb_pcd_and_gen.zip", "r") as h:
        h.extractall()


if __name__ == "__main__":
    download_pcddb()
