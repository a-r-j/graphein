import pytest

from graphein.grn.parse_trrust import TRRUST_df
from graphein.utils.utils import ping

GENE_LIST = ["AATF", "MYC", "USF1", "SP1", "TP53", "DUSP1"]


@pytest.mark.skipif(
    not ping("www.grnpedia.org"), reason="TRRUST is unreachable"
)
def test_trrust_df():
    df = TRRUST_df(GENE_LIST)
    assert len(df) > 0
    assert set(df["g1"]).union(df["g2"]) <= set(GENE_LIST)
