import graphein.molecule as gm

TEST_SMILE = "CCC[S@](=O)c1ccc2[nH]/c(=N\\C(=O)OC)[nH]c2c1"


def test_get_smiles_from_zinc():
    assert (
        gm.get_smiles_from_zinc("ZINC000000000017")
        == "CCC[S@](=O)c1ccc2[nH]/c(=N\\C(=O)OC)[nH]c2c1"
    )


def test_get_zinc_id_from_smile():
    assert gm.get_zinc_id_from_smile(TEST_SMILE) == [
        "ZINC000000000017",
        "ZINC000004095934",
    ]


def test_batch_get_smiles_from_zinc():
    assert gm.batch_get_smiles_from_zinc(
        ["ZINC000000000017", "ZINC000004095934"]
    ) == {
        "ZINC000000000017": "CCC[S@](=O)c1ccc2[nH]/c(=N\\C(=O)OC)[nH]c2c1",
        "ZINC000004095934": "CCC[S@@](=O)c1ccc2[nH]/c(=N\\C(=O)OC)[nH]c2c1",
    }


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
