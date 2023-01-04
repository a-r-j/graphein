from biopandas.pdb import PandasPdb

from graphein.protein.resi_atoms import RESI_THREE_TO_1
from graphein.protein.tensor.io import protein_df_to_tensor
from graphein.protein.tensor.sequence import (
    get_residue_id,
    infer_residue_types,
)


def test_infer_sequence():
    # Get Protein
    p = PandasPdb().fetch_pdb("4hhb")
    df = p.df["ATOM"]
    df.head()

    sequence = get_residue_id(df)
    sequence = "".join(
        [RESI_THREE_TO_1[res.split(":")[1]] for res in sequence]
    )

    coords = protein_df_to_tensor(df)
    inferred_sequence = infer_residue_types(coords)

    assert sequence == inferred_sequence, "Sequences do not match."
