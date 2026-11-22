"""Tests for graphein.protein.edges.intramolecular"""

from pathlib import Path

import graphein.protein.edges.intramolecular as intra
from graphein.protein.config import GetContactsConfig


def test_get_contacts_df_without_contacts_dir_cleans_up(monkeypatch):
    """With no ``contacts_dir``, GetContacts output goes to a private
    temporary directory that is removed afterwards, and the config is left
    unchanged."""
    written = []

    def fake_run(config, file_name):
        out = config.contacts_dir / f"{Path(file_name).stem}_contacts.tsv"
        out.write_text("")
        written.append(out)

    monkeypatch.setattr(intra, "run_get_contacts", fake_run)
    monkeypatch.setattr(intra, "read_contacts_file", lambda c, f: "df")

    config = GetContactsConfig()
    config.contacts_dir = None
    assert intra.get_contacts_df(config, "/data/1abc.pdb") == "df"

    assert config.contacts_dir is None
    assert written[0].parent != Path("/tmp")
    assert not written[0].parent.exists()
