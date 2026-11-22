import ssl
from urllib.error import URLError

import pytest

import graphein.molecule as gm
import graphein.molecule.zinc as zinc

TEST_SMILE = "CCC[S@](=O)c1ccc2[nH]/c(=N\\C(=O)OC)[nH]c2c1"


@pytest.mark.skip("Sometimes returns None. Need to check impl to handle this.")
def test_get_smiles_from_zinc():
    smiles = gm.get_smiles_from_zinc("ZINC000000000017")
    if smiles is None:
        test_get_smiles_from_zinc()
    assert smiles == "CCC[S@](=O)c1ccc2[nH]/c(=N\\C(=O)OC)[nH]c2c1"


@pytest.mark.skip("Sometimes returns None. Need to check impl to handle this.")
def test_get_zinc_id_from_smile():
    assert gm.get_zinc_id_from_smile(TEST_SMILE) == [
        "ZINC000000000017",
        "ZINC000004095934",
    ]


@pytest.mark.skip("Sometimes returns None. Need to check impl to handle this.")
def test_batch_get_smiles_from_zinc():
    smiles = gm.batch_get_smiles_from_zinc(
        ["ZINC000000000017", "ZINC000004095934"]
    )
    assert smiles == {
        "ZINC000000000017": "CCC[S@](=O)c1ccc2[nH]/c(=N\\C(=O)OC)[nH]c2c1",
        "ZINC000004095934": "CCC[S@@](=O)c1ccc2[nH]/c(=N\\C(=O)OC)[nH]c2c1",
    }


@pytest.mark.skip("Sometimes returns None. Need to check impl to handle this.")
def test_batch_get_zinc_id_from_smile():
    assert gm.batch_get_zinc_id_from_smiles(
        [
            "CCC[S@](=O)c1ccc2[nH]/c(=N\\C(=O)OC)[nH]c2c1",
            "CCC[S@](=O)c1ccc2[nH]/c(=N\\C(=O)OC)[nH]c2c1",
        ]
    ) == {
        "CCC[S@](=O)c1ccc2[nH]/c(=N\\C(=O)OC)[nH]c2c1": [
            "ZINC000000000017",
            "ZINC000004095934",
        ]
    }


@pytest.mark.parametrize(
    "func, smilite_func",
    [
        (zinc.get_smiles_from_zinc, "get_zinc_smile"),
        (zinc.get_zinc_id_from_smile, "get_zincid_from_smile"),
    ],
)
def test_url_error_keeps_tls_verification(monkeypatch, func, smilite_func):
    """A failed request must not turn off certificate checks process-wide."""
    pytest.importorskip("smilite")

    def fail(*args, **kwargs):
        raise URLError("certificate verify failed")

    monkeypatch.setattr(zinc.smilite, smilite_func, fail)
    default = ssl._create_default_https_context
    try:
        with pytest.raises(URLError):
            func("ZINC00029323")
        assert ssl._create_default_https_context is default
    finally:
        ssl._create_default_https_context = default
