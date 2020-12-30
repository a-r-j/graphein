"""Functions for working with Protein Structure Graphs"""
# %%
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
import torch
from functools import lru_cache


@lru_cache()
def _load_esm_model():
    import torch

    return torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")


def compute_esm_embedding(sequence: str, representation: str):
    model, alphabet = _load_esm_model()
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", sequence),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    if representation == "residue":
        return token_representations.numpy()
    elif representation == "sequence":
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(
                token_representations[i, 1 : len(seq) + 1].mean(0)
            )
        return sequence_representations[0].numpy()


if __name__ == "__main__":
    print(compute_esm_embedding("MYGVYMK", representation="sequence"))
