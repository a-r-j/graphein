"""
Download all EXPASY amino acid scales.

Collaborative work done by Kannan Sankar, Mei Xiao and Eric Ma.

Extracts all amino acid properties from EXPASY,
and dumps them as a CSV file.

If you use the scales that are downloaded by this script,
please make sure that you cite ProtScale.
Reference is here: https://web.expasy.org/protscale/protscale-ref.html.
Credit must be due properly.
"""

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.autonotebook import tqdm

urls = requests.get("https://web.expasy.org/protscale/")
urls_list = urls.text.split("<PRE>")[1].split("</PRE>")[0]
soup = BeautifulSoup(urls.text, "html.parser")

wanted = [
    a.get("href")
    for a in soup.body.find_all("a")
    if (len(a) >= 1)
    and ("/protscale/pscale" in a.get("href"))
    and ("protscale_help" not in a.get("href"))
]

urls = ["https://web.expasy.org" + path for path in wanted]

property_names = [
    path.split("/")[-1]
    .replace(".html", "")
    .replace(".", "_")
    .replace("-", "_")
    .lower()
    for path in wanted
]


def generate_mapping(url):
    result = requests.get(url)
    mappings = result.text.split("<PRE>")[1].split("</PRE>")[0].split("\n")
    mappings = [m for m in mappings if ":" in m]
    mappings = [re.split(":\s+", m) for m in mappings]
    mappings = {aa.upper(): float(value) for aa, value in mappings}
    return mappings


mappings = []
for i, (url, name) in enumerate(zip(urls, property_names)):
    print(name)
    mapping = generate_mapping(url)
    mapping = pd.Series(mapping, name=name)
    mappings.append(mapping)

expasy_aa_feats = pd.DataFrame(mappings)
# aa_props.to_csv("amino_acid_properties.csv")


AMINO_ACIDS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]

# Taken from: https://www.anaspec.com/html/pK_n_pl_Values_of_AminoAcids.html
PKA_COOH_ALPHA = [
    2.35,
    2.18,
    2.18,
    1.88,
    1.71,
    2.17,
    2.19,
    2.34,
    1.78,
    2.32,
    2.36,
    2.20,
    2.28,
    2.58,
    1.99,
    2.21,
    2.15,
    2.38,
    2.20,
    2.29,
]
PKA_NH3 = [
    9.87,
    9.09,
    9.09,
    9.60,
    10.78,
    9.13,
    9.67,
    9.60,
    8.97,
    9.76,
    9.60,
    8.90,
    9.21,
    9.24,
    10.60,
    9.15,
    9.12,
    9.39,
    9.11,
    9.74,
]
PKA_RGROUP = [
    7.0,
    13.2,
    13.2,
    3.65,
    8.33,
    7,
    4.25,
    7,
    5.97,
    7,
    7,
    10.28,
    7,
    7,
    7,
    7,
    7,
    7,
    10.07,
    7,
]
ISOELECTRIC_POINTS = [
    6.11,
    10.76,
    10.76,
    2.98,
    5.02,
    5.65,
    3.08,
    6.06,
    7.64,
    6.04,
    6.04,
    9.47,
    5.74,
    5.91,
    6.30,
    5.68,
    5.60,
    5.88,
    5.63,
    6.02,
]


# Other features?
pka_cooh_alpha = pd.Series(
    dict(zip(AMINO_ACIDS, PKA_COOH_ALPHA)), name="pka_cooh_alpha"
)
pka_nh3 = pd.Series(dict(zip(AMINO_ACIDS, PKA_NH3)), name="pka_nh3")
pka_rgroup = pd.Series(dict(zip(AMINO_ACIDS, PKA_RGROUP)), name="pka_rgroup")
isoelectric_points = pd.Series(
    dict(zip(AMINO_ACIDS, ISOELECTRIC_POINTS)), name="isoelectric_points"
)

basic_aa_feats = pd.DataFrame(
    [pka_cooh_alpha, pka_nh3, pka_rgroup, isoelectric_points]
)

aa_feats = pd.concat([basic_aa_feats, expasy_aa_feats])
aa_feats.to_csv("amino_acid_properties.csv")
