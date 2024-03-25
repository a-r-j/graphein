import networkx as nx
import numpy as np

from graphein.utils.dependencies import import_message

try:
    import prody
except ImportError:
    message = import_message(
        "graphein.protein.features.graph.dynamics", "prody", "prody", True
    )


def add_modes(
    g: nx.Graph, n_modes: int = 6, add_separate: bool = False
) -> nx.Graph:
    """
    Computes normal modes for a protein graph.

    Normal modes are accessible with: ``g.graph["nma_eigenvectors"]``.
    Normal mode eigenvalues are accessible with: ``g.graph["nma_eigenvalues"]``.

    :param g: Protein structure graph
    :type g: nx.Graph
    :param n_modes: Number of modes, defaults to 6
    :type n_modes: int
    :return: Graph with modes added as a graph feature
    :rtype: nx.Graph
    """
    coords = g.graph["coords"]
    anm = prody.ANM(g.graph["name"])
    anm.buildHessian(coords)
    anm.calcModes()

    # eig_vecs = torch.Tensor(anm[:n_modes].getEigvecs()).reshape(-1, n_modes, 3)
    eig_vals = anm[:n_modes].getEigvals()

    modes = np.array([anm[m].getArrayNx3() for m in range(n_modes)])
    modes = np.einsum("ijk->jki", modes)

    if add_separate:
        for i, (_, d) in enumerate(g.nodes(data=True)):
            for mode in range(n_modes):
                d[f"nma_eigenvectors_{mode}"] = modes[i, :, mode]
                d[f"nma_eigenvalues_{mode}"] = modes[i, :, mode]

    for i, (_, d) in enumerate(g.nodes(data=True)):
        d["nma_eigenvectors"] = modes[i, :, :]
        d["nma_eigenvalues"] = eig_vals

    return g
