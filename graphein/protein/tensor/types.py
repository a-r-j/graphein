"""Types for graphein.protein.tensor.

Graphein provides many types for commonly used tensors of specific shapes.
"""
# Graphein
# Author: Arian Jamasb <arian@jamasb.io>
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from typing import NewType, Optional, Union

import torch
from torchtyping import TensorType

# Positions
AtomTensor = NewType("AtomTensor", TensorType[-1, 37, 3])
"""
``TensorType[-1, 37, 3]``

Tensor of atom coordinates. The first dimension is the length of the protein,
the second the number of canonical atom types. The last dimension contains the
x,y,z positions.

.. seealso:: :class:`ResidueTensor` :class:`CoordTensor`
"""

BackboneTensor = NewType("BackboneTensor", TensorType[-1, 4, 3])
"""
``TensorType[-1, 4, 3]``

Tensor of backbone atomic coordinates. The first dimension is the length of the
protein (or batch), the second dimension corresponds to ``[N, Ca, C, O]`` atoms
and the last dimension contains the x,y,z coordinates.

.. seealso:: :ref:`graphein.protein.tensor.types.AtomTensor

"""

ResidueTensor = NewType("ResidueTensor", TensorType[37, 3])
"""
``TensorType[37, 3]``

Tensor of atom coordinates for a residue. Each index in dimension one
corresponds to an atom (See: :ref:`graphein.protein.resi_atoms.ATOM_NUMBERING`).
The last dimension contains the x,y,z positions. Missing atoms are denoted by a
fill value (default = ``1e-5``).

.. seealso:: :class:`AtomTensor` :class:`CoordTensor`
"""


CoordTensor = NewType("CoordTensor", TensorType[-1, 3])
"""
``TensorType[-1, 3]``

Tensor of coordinates. The first dimension is the length of the protein
(or batch of proteins), the last dimension contains the x,y,z positions."""

PositionTensor = NewType("PositionTensor", Union[AtomTensor, CoordTensor])
"""
Union[:ref:`graphein.protein.tensor.types.AtomTensor,
:ref:`graphein.protein.tensor.types.CoordTensor]


Union of ``AtomTensor`` and ``CoordTensor``.

.. seealso:: :class:`AtomTensor`, :class:`CoordTensor`
"""

# Represenations
BackboneFrameTensor = NewType("BackboneFrameTensor", TensorType[-1, 3, 3])
"""
``TensorType[-1, 3, 3]``

Tensor of backbone frames as rotation matrices. The first dimension is the
length of the protein, the second and third dimensions specify a rotation matrix
relative to an idealised residue.

.. seealso::
    :ref:`graphein.protein.tensor.reconstruction.get_ideal_backbone_coords
"""

ResidueFrameTensor = NewType("ResidueFrameTensor", TensorType[3, 3])
"""
``TensorType[3, 3]``

.. seealso:: :class:`BackboneFrameTensor`
"""


# Rotations
EulerAngleTensor = NewType("EulerAngleTensor", TensorType[-1, 3])

QuaternionTensor = NewType("QuaternionTensor", TensorType[-1, 4])
"""
``TensorType[-1, 4]``

Tensor of quaternions. The first dimension is the length of the protein
(or batch), the second dimension is the quaternion.

.. see:: https://en.wikipedia.org/wiki/Quaternion
"""


TransformTensor = NewType("TransformTensor", TensorType[-1, 4, 4])


RotationMatrix2D = NewType("RotationMatrix2D", TensorType[2, 2])
"""
``TensorType[2, 2]``

Specifies a 2D rotation matrix.

.. seealso:: :class:`RotationMatrix3D` :class:`RotationMatrix` :class:`RotationTensor`
"""

RotationMatrix3D = NewType("RotationMatrix3D", TensorType[3, 3])
"""
``TensorType[3, 3]``

Specifies a 3D rotation matrix.

.. seealso:: :class:`RotationMatrix2D` :class:`RotationMatrix` :class:`RotationTensor`
"""

RotationMatrix = NewType(
    "RotationMatrix", Union[RotationMatrix2D, RotationMatrix3D]
)
"""
``Union[RotationMatrix3D, RotationMatrix2D]``

Specifies a rotation matrix in either 2D or 3D.

.. seealso:: :class:`RotationMatrix2D` :class:`RotationMatrix3D` :class:`RotationTensor`
"""


RotationMatrixTensor = NewType("RotationMatrixTensor", TensorType[-1, 3, 3])

RotationTensor = NewType(
    "RotationTensor", Union[QuaternionTensor, RotationMatrixTensor]
)


# Angles
DihedralTensor = NewType(
    "DihedralTensor", Union[TensorType[-1, 3], TensorType[-1, 6]]
)
"""
``Union[TensorType[-1, 3], TensorType[-1, 6]]``

Tensor of backbone dihedral angles (phi, psi, omega). Either in degrees/radians
or embedded on the unit sphere ``[cos(phi), sin(phi), cos(psi), sin(psi), ...]``

.. seealso::

    :meth:graphein.protein.tensor.angles.dihedrals,
    :meth:graphein.protein.tensor.angles.dihedrals_to_rad
    :meth:graphein.protein.tensor.plot.plot_dihedrals

"""

TorsionTensor = NewType(
    "TorsionTensor", Union[TensorType[-1, 4], TensorType[-1, 8]]
)
"""
``Union[TensorType[-1, 4], TensorType[-1, 8]]``

Tensor of sidechain torsion angles ``[chi1, chi2, chi3, chi4]``. Either in
degrees/radians or embedded on the unit sphere:
``[cos(chi1), sin(chi1), cos(chi2), sin(chi2), ...]``.

.. seealso::

    :meth:graphein.protein.tensor.angles.sidechain_torsions,
    :meth:graphein.protein.tensor.angles.torsions_to_rad

"""

BackboneFrameTensor = NewType("BackboneFrameTensor", TensorType[-1, 3, 3])
"""
``TensorType[-1, 3, 3]``

Tensor of backbone frames as rotation matrices. The first dimension is the
length of the protein, the second and third dimensions specify a rotation matrix
relative to an idealised residue.

.. seealso::

    :meth:`graphein.protein.tensor.reconstruction.get_ideal_backbone_coords`
    :meth:`graphein.protein.tensor.representation.get_backbone_frames`
"""

ResidueFrameTensor = NewType("ResidueFrameTensor", TensorType[3, 3])
"""
``TensorType[-1, 3, 3]``

.. seealso:: :class:`BackboneFrameTensor`
"""

EdgeTensor = NewType("EdgeTensor", TensorType[2, -1])


ScalarTensor = NewType("ScalarTensor", TensorType[-1])

OptTensor = NewType("OptTensor", Optional[torch.Tensor])
