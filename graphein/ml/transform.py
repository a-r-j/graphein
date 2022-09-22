"""Dataset loader transform Functions for loading set"""
# Graphein
# Author: xutingfeng <xutingfeng@big.ac.cn>
# License: MIT

# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein


import torch 
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


def attr2Tensor(data:Data, attr:"str")->Data:
    if not hasattr(data, attr):
        raise ValueError("doesn't have an attr:{}".format(attr))
    attr_data = getattr(data, attr)

    if isinstance(attr_data, list):

        attr_data = [torch.tensor(i, dtype=torch.float64).unsqueeze(0) if len(i.shape) == 1 else torch.tensor(i, dtype=torch.float64) for i in attr_data ]

        attr_data = torch.concat(attr_data, 0)
    setattr(data, attr, attr_data)


class Reshape_Attr2Tensor(BaseTransform):
    def __init__(self, attr) -> None:
        self.attr = attr

    def __call__(self, data):

        #  do
        def _attr2Tensor_recursive(x, attr:str):
            if isinstance(attr, str):
                attr2Tensor(x, attr)
                return x
            elif isinstance(attr, list):
                for i in attr:
                    x = _attr2Tensor_recursive(x, i)
                return x 
        _attr2Tensor_recursive(data, self.attr)
        return data