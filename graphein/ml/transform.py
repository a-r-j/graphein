"""Dataset loader transform Functions for loading set"""
# Graphein
# Author: xutingfeng <xutingfeng@big.ac.cn>
# License: MIT

# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein

from torch_geometric.transforms import BaseTransform
from typing import Callable, Dict, Generator, List, Optional, Union

import torch
import torch_geometric


def attr2Tensor(data:torch_geometric.data.Data, attr:str, dtype=torch.float32)->torch_geometric.data.Data:

    """
    ``attr2Tensor`` Change specific attribution ``attr`` of ``torch_geometric.data.Data`` to Tensor. Assume these ``attr`` are [``np.array``, ``np.array``]. If the array's shape is ``(dim,)`` will be reshaped as ``(1, dim)``. Finally the shape of  the array will be ``(O_1, dim1, dim2, ...)`` or ``(1, dim)``, and will be concated at axis=0 or dim=0.

    :param data: Data to change it's attribution
    :type data: torch_geometric.data.Data
    :param attr: The name of attribution
    :type attr: str
    :param dtype: Dtype of new attr, defaults to torch.float32
    :type dtype: _type_, optional
    :raises ValueError: Passing the correct name of attribution
    :return: torch_geometric.data.Data with specific attr Tensor
    :rtype: torch_geometric.data.Data
    """
    if not hasattr(data, attr):
        raise ValueError(f"doesn't have an attr:{attr}")
    attr_data = getattr(data, attr)

    if isinstance(attr_data, list):  
        attr_data = [torch.tensor(i, dtype=dtype).unsqueeze(0) if len(i.shape) == 1 else torch.tensor(i, dtype=dtype) for i in attr_data ]

        attr_data = torch.concat(attr_data, 0)
    setattr(data, attr, attr_data)


class Reshape_Attr2Tensor(BaseTransform):
    """
    Reshape_Attr2Tensor Convert ``Data.attr``, like ``[np.array, np.array]`` to ``torch.Tensor`` 
    """
    def __init__(self, attr:Union[str, List]) -> None:
        self.attr = attr

    def __call__(self, data:torch_geometric.data.Data)->torch_geometric.data.Data:
        """
        :param data: Convert attr of ``torch_geometric.data.Data`` to ``torch.Tensor``
        :type data: torch_geometric.data.Data
        """
        def _attr2Tensor_recursive(x, attr:Union[str, List]):
            if isinstance(attr, str):
                attr2Tensor(x, attr)
                return x
            elif isinstance(attr, list):  # multiple attr will be done by this
                for i in attr:
                    x = _attr2Tensor_recursive(x, i)
                return x 
            else:
                raise ValueError(f"attr name or list of attr name")
        _attr2Tensor_recursive(data, self.attr)
        return data
