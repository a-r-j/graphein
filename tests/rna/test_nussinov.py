from graphein.rna.nussinov import nussinov

TEST_SEQUENCE = "UUGGAGUACACAACCUGUACACUCUUUC"


def test_nussinov():
    # nussinov algo does not guarantee that the dot-bracket notation is correct.
    # There are several other ways of computing this.
    target_dot_bracket = nussinov(TEST_SEQUENCE)
    assert len(TEST_SEQUENCE) == len(target_dot_bracket)
