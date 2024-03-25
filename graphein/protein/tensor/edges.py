"""Utilities for computing edges."""

from typing import Optional

from loguru import logger as log

from graphein.utils.dependencies import import_message

from .types import CoordTensor, EdgeTensor, OptTensor

try:
    import torch
except ImportError:
    message = import_message(
        "graphein.protein.tensor.edges",
        "torch",
        conda_channel="pytorch",
        pip_install=True,
    )
    log.warning(message)

try:
    from torch_geometric.nn.pool import knn_graph, radius_graph
except ImportError:
    message = import_message(
        "graphein.protein.tensor.edges",
        "torch_geometric",
        "pyg",
        pip_install=True,
    )


def compute_edges(
    x: torch.Tensor,
    edge_type: str = "knn_8",
    batch: OptTensor = None,
    **kwargs,
) -> EdgeTensor:
    """Computes edges for a given node feature (or coordinate) matrix ``x``.

    .. code-block:: python
        import torch
        import graphein.protein.tensor as gpt

        x = torch.rand(10, 3)

        gpt.compute_edges(x, "knn_8") # KNN with k = 8
        gpt.compute_edges(x, "eps_8") # Radius graph with r = 8

    :param x: Node features or positions
    :param edge_type: Str denoting type of edges in form ``{edgetype}_{value}``.
        E.g. ``"knn_8"`` for KNN with ``k=8``,
        ``"eps_6"`` for radius graph with ``r=6``.
    :returns: Edge indices
    :rtype: EdgeTensor
    """

    edge_type = edge_type.lower()
    etype = edge_type.split("_")[0]
    val = edge_type.split("_")[1]

    if etype == "knn":
        return knn_edges(x=x, k=int(val), batch=batch, **kwargs)
    elif etype == "eps":
        return radius_edges(x=x, radius=float(val), batch=batch, **kwargs)
    else:
        raise ValueError(f"Edge type {edge_type} not recognised.")


def radius_edges(
    x: torch.Tensor,
    radius: float = 8.0,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
) -> EdgeTensor:
    r"""Computes graph edges to all points within a given distance.

    .. code-block:: python

            import torch
            import graphein.protein.tensor

            x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
            batch = torch.tensor([0, 0, 0, 0])
            edge_index = gpt.radius_graph(x, r=1.5, batch=batch, loop=False)

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)

    """
    return radius_graph(
        x,
        r=radius,
        batch=batch,
        loop=loop,
        max_num_neighbors=max_num_neighbors,
        flow=flow,
        num_workers=num_workers,
    )


def knn_edges(
    x: torch.Tensor,
    k: int = 10,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    flow: str = "source_to_target",
    num_workers: int = 1,
) -> EdgeTensor:
    r"""Computes graph edges to the nearest :obj:`k` points.

    .. code-block:: python

        import torch
        import graphein.protein.tensor as gpt

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = gpt.knn_graph(x, k=2, batch=batch, loop=False)

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        cosine (boolean, optional): If :obj:`True`, will use the cosine
            distance instead of euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)
    """
    return knn_graph(
        x,
        k=k,
        batch=batch,
        loop=loop,
        flow=flow,
        num_workers=num_workers,
    )


def edge_distances(x: CoordTensor, edge_index: EdgeTensor, p: float = 2):
    """Computes the distances between edges

    :param x: Node positions
    :param edge_index: Edge indices
    :param p: The norm degree. Can be negative. Default: ``2``.``
    :returns: Edge distances
    :rtype: torch.Tensor
    """
    return torch.pairwise_distance(x[edge_index[0]], x[edge_index[1]], p=p)


def contact_map(x: CoordTensor) -> torch.Tensor:
    """Computes the contact map for a set of coordinates

    :param x: Node positions
    """
    return torch.cdist(x, x)
