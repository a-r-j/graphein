"""Types for graphein.protein.tensor."""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein


from typing import Union

from torchtyping import TensorType

AtomTensor = TensorType[-1, 37, 3]

CoordTensor = TensorType[-1, 3]

PositionTensor = Union[AtomTensor, CoordTensor]

EulerAngleTensor = TensorType[-1, 3]

QuaternionTensor = TensorType[-1, 4]

TransformTensor = TensorType[-1, 4, 4]

RotationMatrixTensor = TensorType[-1, 3, 3]

RotationTensor = Union[QuaternionTensor, RotationMatrixTensor]

DihedralTensor = Union[TensorType[-1, 3], TensorType[-1, 6]]

TorsionTensor = Union[TensorType[-1, 4], TensorType[-1, 8]]

BackboneFrameTensor = TensorType[-1, 3, 3]

ResidueFrameTensor = TensorType[3, 3]

EdgeTensor = TensorType[2, -1]

BackboneTensor = TensorType[-1, 4, 3]
