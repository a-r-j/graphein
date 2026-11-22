"""Tests for graphein.molecule.chembl (queries the ChEMBL API)."""

from graphein.molecule.chembl import (
    get_chembl_id_from_smiles,
    get_chembl_metadata,
    get_smiles_from_chembl,
)


def test_get_smiles_from_chembl():
    assert get_smiles_from_chembl("CHEMBL25") == "CC(=O)Oc1ccccc1C(=O)O"


def test_get_chembl_id_from_smiles():
    assert get_chembl_id_from_smiles("CC(=O)Oc1ccccc1C(=O)O") == "CHEMBL25"


def test_smiles_with_url_characters_are_encoded():
    """``#`` must not end the URL: ``C#N`` is hydrogen cyanide, not ``C``
    (methane, CHEMBL17564)."""
    assert get_chembl_id_from_smiles("C#N") == "CHEMBL183419"


def test_get_chembl_metadata():
    metadata = get_chembl_metadata("CHEMBL25")
    assert metadata["pref_name"] == "ASPIRIN"
