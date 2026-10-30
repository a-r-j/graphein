from graphein.rna.utils import read_dbn_file


def test_read_dbn_file(tmp_path):
    path = tmp_path / "test.dbn"
    path.write_text("#Name: test\n#Length: 6\nGGAUCC\n((..))\n")
    seq, db = read_dbn_file(str(path))
    assert seq.strip() == "GGAUCC"
    assert db.strip() == "((..))"
