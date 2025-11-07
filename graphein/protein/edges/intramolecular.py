"""Featurization functions for graph edges."""

# Graphein
# Author: Arian Jamasb <arian@jamasb.io>, Eric Ma, Charlie Harris
# License: MIT
# Project Website: https://github.com/a-r-j/graphein
# Code Repository: https://github.com/a-r-j/graphein
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd
from loguru import logger as log

from graphein.protein.utils import download_pdb

CONTACTS_FILE_SUFFIX = "_contacts.tsv"


def peptide_bonds(G: nx.Graph) -> nx.Graph:
    """
    Adds peptide backbone to residues in each chain.

    :param G: nx.Graph protein graph.
    :type G: nx.Graph
    :returns: nx.Graph protein graph with added peptide bonds.
    :rtype: nx.Graph
    """
    log.debug("Adding peptide bonds to graph")
    # Iterate over every chain
    for chain_id in G.graph["chain_ids"]:
        # Find chain residues
        chain_residues = [
            (n, v) for n, v in G.nodes(data=True) if v["chain_id"] == chain_id
        ]

        # Iterate over every residue in chain
        for i, residue in enumerate(chain_residues):
            # Checks not at chain terminus
            if i == len(chain_residues) - 1:
                continue
            # Asserts residues are on the same chain
            cond_1 = (
                residue[1]["chain_id"] == chain_residues[i + 1][1]["chain_id"]
            )
            # Asserts residue numbers are adjacent
            cond_2 = (
                abs(
                    residue[1]["residue_number"]
                    - chain_residues[i + 1][1]["residue_number"]
                )
                == 1
            )

            # If this checks out, we add a peptide bond
            if cond_1 and cond_2:
                # Adds "peptide_bond" between current residue and the next
                if G.has_edge(i, i + 1):
                    G.edges[i, i + 1]["kind"].add("peptide_bond")
                else:
                    G.add_edge(
                        residue[0],
                        chain_residues[i + 1][0],
                        kind={"peptide_bond"},
                    )
            else:
                continue
    return G


####################################
#                                  #
#     GetContacts Interactions     #
#                                  #
####################################


def get_contacts_df(config: GetContactsConfig, pdb_path: str) -> pd.DataFrame:
    """
    Reads GetContact File and returns it as a ``pd.DataFrame``.

    :param config: GetContactsConfig object
    :type config: GetContactsConfig
    :param pdb_name: Name of PDB file. Output contacts files are named:
        ``{pdb_name}_contacts.tsv``.
    :type pdb_name: str
    :return: DataFrame of parsed GetContacts output.
    :rtype: pd.DataFrame
    """
    if not config.contacts_dir:
        config.contacts_dir = Path("/tmp/")

    contacts_file = (
        config.contacts_dir / f"{Path(pdb_path).stem}{CONTACTS_FILE_SUFFIX}"
    )

    # Check for existence of GetContacts file
    if not os.path.isfile(contacts_file):
        log.info("GetContacts file not found. Running GetContacts...")
        run_get_contacts(config, file_name=pdb_path)

    contacts_df = read_contacts_file(config, contacts_file)

    # remove temp GetContacts file
    if config.contacts_dir == "/tmp/":
        os.remove(contacts_file)

    return contacts_df


def _validate_get_contacts_installation(config: GetContactsConfig) -> None:
    """Validate GetContacts installation."""
    get_contacts_script = (
        Path(config.get_contacts_path) / "get_static_contacts.py"
    )
    if not get_contacts_script.is_file():
        raise FileNotFoundError(
            f"GetContacts installation not found at {get_contacts_script}. "
            "Please install from: https://getcontacts.github.io"
        )


def _get_or_download_pdb(
    config: GetContactsConfig, pdb_id: Optional[str], file_name: Optional[str]
) -> Path:
    """Get PDB file path, downloading if necessary."""
    if file_name and Path(file_name).exists():
        return Path(file_name)

    pdb_path = config.pdb_dir / f"{pdb_id}{'.pdb'}"
    if not pdb_path.exists():
        log.debug(f"Downloading PDB file for {pdb_id}")
        downloaded_path = download_pdb(pdb_code=pdb_id, out_dir=config.pdb_dir)
        return Path(downloaded_path)

    return pdb_path


def run_get_contacts(
    config: GetContactsConfig,
    pdb_id: Optional[str] = None,
    file_name: Optional[str] = None,
):
    """
    Runs GetContacts on a protein structure. If no file_name is provided, a
    PDB file is downloaded for the pdb_id

    :param config: GetContactsConfig object containing GetContacts parameters
    :type config: graphein.protein.config.GetContactsConfig
    :param pdb_id: 4-character PDB accession code
    :type pdb_id: str, optional
    :param file_name: PDB_name file to use, if annotations to be retrieved from
        the PDB
    :type file_name: str, optional
    """
    _validate_get_contacts_installation(config)

    if not file_name and not pdb_id:
        raise ValueError("Either file_name or pdb_id must be provided")

    # Check for existence of pdb file. If not, download it.
    pdb_path = _get_or_download_pdb(config, pdb_id, file_name)
    output_path = config.contacts_dir / (pdb_path.stem + CONTACTS_FILE_SUFFIX)

    command = f"{config.get_contacts_path}/get_static_contacts.py "
    command += f"--structure {str(pdb_path)} "
    command += f"--output {str(output_path)} "
    command += "--itypes all"  # --sele "protein"'

    log.info(f"Running GetContacts: {command}")
    subprocess.run(command, shell=True, check=True)

    if not output_path.exists():
        raise FileNotFoundError(
            f"GetContacts failed to create output file: {output_path}"
        )

    log.info(f"Computed Contacts for: {pdb_id}")


def read_contacts_file(
    config: GetContactsConfig, contacts_file: str
) -> pd.DataFrame:
    """
    Parses GetContacts file to an edge list (pd.DataFrame).

    :param config: GetContactsConfig object
        (:ref:`~graphein.protein.config.GetContactsConfig`)
    :type config: GetContactsConfig
    :param contacts_file: file name of contacts file
    :type contacts_file: str
    :return: Pandas Dataframe of edge list
    :rtype: pd.DataFrame
    """
    if not Path(contacts_file).exists():
        raise FileNotFoundError(f"Contacts file not found: {contacts_file}")

    log.debug(f"Parsing GetContacts output file at: {contacts_file}")

    contacts = []

    # Extract every contact and residue types
    with open(contacts_file, "r") as f:
        # Skip header lines
        for _ in range(2):
            next(f)

        for line in f:
            contact = line.strip().split("\t")

            interaction_type = contact[1]
            res1, res2 = contact[2], contact[3]

            # Remove atom names if not using atom granularity
            if config.granularity != "atom":
                res1 = ":".join(res1.split(":")[:3])
                res2 = ":".join(res2.split(":")[:3])

            contacts.append([res1, res2, interaction_type])

    if not contacts:
        log.warning(f"No contacts found in file: {contacts_file}")
        return pd.DataFrame(columns=["res1", "res2", "interaction_type"])

    return pd.DataFrame(
        contacts, columns=["res1", "res2", "interaction_type"]
    ).drop_duplicates()


def _get_pdb_path_from_graph(G: nx.Graph) -> Path:
    """Extract PDB path from graph metadata."""
    if G.graph.get("path"):
        return Path(G.graph["path"])

    config = G.graph.get("config")
    if not config:
        raise KeyError("Graph missing required 'config' attribute")

    return (
        config.get_contacts_config.pdb_dir
        / f"{G.graph['name']}{'.pdb' if not str(G.graph['name']).endswith('.pdb') else ''}"
    )


def add_contacts_edge(G: nx.Graph, interaction_type: str) -> nx.Graph:
    """
    Adds specific interaction types to the protein graph.

    :param G: Networkx protein graph.
    :type G: nx.Graph
    :param interaction_type: Interaction type to be added.
    :type interaction_type: str
    :return G: nx.Graph with specified interaction-based edges added.
    :rtype: nx.Graph
    """
    log.debug(f"Adding {interaction_type} edges to graph")

    # Ensure contacts_df exists
    if "contacts_df" not in G.graph:
        log.info("No 'contacts_df' found in G.graph. Running GetContacts.")
        pdb_path = _get_pdb_path_from_graph(G)
        G.graph["contacts_df"] = get_contacts_df(
            G.graph["config"].get_contacts_config, str(pdb_path)
        )

    contacts = G.graph["contacts_df"]

    # Select specific interaction type
    interactions = contacts[contacts["interaction_type"] == interaction_type]

    for _, [res1, res2, interaction_type] in interactions.iterrows():
        # Check residues are actually in graph
        if not (G.has_node(res1) and G.has_node(res2)):
            continue

        if G.has_edge(res1, res2):
            G.edges[res1, res2]["kind"].add(interaction_type)
        else:
            G.add_edge(res1, res2, kind={interaction_type})

    return G


def _create_interaction_function(interaction_code: str, description: str):
    """Factory function to create interaction edge functions."""

    def interaction_func(G: nx.Graph) -> nx.Graph:
        """
        Adds {description} to protein structure graph.

        :param G: nx.Graph to add {description} to.
        :return: nx.Graph with {description} added.
        """
        return add_contacts_edge(G, interaction_code)

    interaction_func.__name__ = description.replace(" ", "_").replace("-", "_")
    interaction_func.__doc__ = (
        f"Adds {description} to protein structure graph."
    )
    return interaction_func


# Create all interaction functions
hydrogen_bond = _create_interaction_function("hb", "hydrogen bonds")
hydrogen_bond_backbone_backbone = _create_interaction_function(
    "hbbb", "backbone to backbone hydrogen bond interactions"
)
hydrogen_bond_sidechain_backbone = _create_interaction_function(
    "hbsb", "side-chain to backbone hydrogen bond interactions"
)
hydrogen_bond_sidechain_sidechain = _create_interaction_function(
    "hbss", "side-chain to side-chain hydrogen bond interactions"
)


def hydrogen_bond(G: nx.Graph) -> nx.Graph:
    """
    Adds all types of hydrogen bonds to protein structure graph.
    Combines backbone-backbone, sidechain-backbone, and sidechain-sidechain hydrogen bonds.

    :param G: nx.Graph to add hydrogen bonds to.
    :return: nx.Graph with all hydrogen bond types added.
    """
    G = hydrogen_bond_backbone_backbone(G)
    G = hydrogen_bond_sidechain_backbone(G)
    G = hydrogen_bond_sidechain_sidechain(G)
    return G


salt_bridge = _create_interaction_function("sb", "salt bridges")
pi_cation = _create_interaction_function("pc", "pi-cation interactions")
pi_stacking = _create_interaction_function("ps", "pi-stacking interactions")
t_stacking = _create_interaction_function("ts", "t-stacking interactions")
hydrophobic = _create_interaction_function("hp", "hydrophobic interactions")
van_der_waals = _create_interaction_function(
    "vdw", "van der Waals interactions"
)
