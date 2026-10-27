import pytest

from graphein.protein.config import ProteinMeshConfig
from graphein.protein.meshes import convert_verts_and_face_to_mesh, create_mesh

pytest.importorskip("pymol")
pytest.importorskip("pytorch3d")


def test_create_mesh():
    config = ProteinMeshConfig(
        pymol_commands=[
            "show surface",
            "set surface_solvent, on",
            "set solvent_radius, 10000",
        ]
    )
    verts, faces, _ = create_mesh(pdb_code="3eiy", config=config)
    mesh = convert_verts_and_face_to_mesh(verts, faces)
    assert mesh.verts_packed().shape[0] == verts.shape[0]
