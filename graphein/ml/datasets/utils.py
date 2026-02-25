"""
Modified from: https://github.com/rcsb/rcsb-training-resources/blob/master/example-use-cases/pdb-ligand-composition/generate_pdb_ligand_mappings.py
This script generates two mapping files (output as TSV files):

    1. A mapping between all PDB structures and the ligands present in each structure
       (`pdb-to-cc.tsv`)
    2. A mapping between all ligands in the PDB and the corresponding list of PDB
       structures in which they are present (`cc-to-pdb.tsv`)

The following types of ligands (or "chemical components") can be included in the mappings:

    - Those that exist as a nonpolymer entity (e.g., a non-covalently bound ATP ligand)
    - Those that exist as a monomer within a branched polymer entity (e.g., a subunit in a carbohydrate)
    - Those that exist as a non-standard amino acid monomer within a polymer entity
    - Those that exist as a standard amino acid monomer within a polymer entity

By default, the first three are included. You can control this using the '--chem_comp_types' argument.

Additionally, this script can generate a file with the occurrence count of chemical components
along with its name and formula (`cc-counts-extra.tsv`), using the '--generate_cc_extra_file' flag.


Requirements:
    pip install "rcsb-api>=1.4.0"

Usage:
    # Get usage details
        python3 generate_pdb_ligand_mappings.py --help

    # Generate the PDB->CC and CC->PDB files
        python3 generate_pdb_ligand_mappings.py

    # Generate the PDB->CC and CC->PDB files, plus the CC counts file
        python3 generate_pdb_ligand_mappings.py --generate_cc_extra_file

    # Generate the PDB->CC and CC->PDB files, omitting non-standard amino acids
        python3 generate_pdb_ligand_mappings.py --chem_comp_types nonpolymer branched

    # Generate the PDB->CC and CC->PDB files, including standard amino acids occurring within a polymer chain
        python3 generate_pdb_ligand_mappings.py --chem_comp_types nonpolymer branched polymer_nstd polymer_std

Output (can customize name using corresponding CLI arguments):
    pdb-to-cc.tsv
        # Format:  <pdb_id1>        <chem_comp_id_1>  <chem_comp_id_2>  ...

    cc-to-pdb.tsv
        # Format:  <chem_comp_id>   <pdb_id_1>  <pdb_id_2>  ...

    cc-counts-extra.tsv
        # Format:  <chem_comp_id>   <count>  <name>  <formula>
"""

import argparse
import time

from loguru import logger
from rcsbapi.config import config
from rcsbapi.data import ALL_STRUCTURES
from rcsbapi.data import DataQuery as Query

# Public constants for use in the Python API
ALLOWED_CHEM_COMP_TYPES = (
    "nonpolymer",
    "branched",
    "polymer_nstd",
    "polymer_std",
)

DEFAULT_CHEM_COMP_TYPES = ("nonpolymer", "branched", "polymer_nstd")

CHEMICAL_COMPONENT_TYPES_ARG_MAPPINGS = {
    "nonpolymer": "nonpolymer_entities.rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",  # Ligands in nonpolymer entities
    "branched": "branched_entities.rcsb_branched_entity_container_identifiers.chem_comp_monomers",  # Monomers in branched entities
    "polymer_nstd": "polymer_entities.rcsb_polymer_entity_container_identifiers.chem_comp_nstd_monomers",  # Non-standard monomers in polymer chains
    "polymer_std": "polymer_entities.rcsb_polymer_entity_container_identifiers.chem_comp_monomers",  # Standard monomers in polymer chains
}


def fetch_all_chem_comp_ids(chem_comp_types_to_include: list):
    """Fetch the chemical component and PDB mapping data from RCSB.org using the Data API"""

    logger.info(
        "Fetching chemical component data for {n_types} chem component field(s): {types}",
        n_types=len(chem_comp_types_to_include),
        types=chem_comp_types_to_include,
    )

    # Initialize the data query to retrieve relevant chemical component data
    query = Query(
        input_type="entries",  # Query all structure entries
        input_ids=ALL_STRUCTURES,  # Constant representing all known structures
        return_data_list=["rcsb_id"] + chem_comp_types_to_include,
    )

    # Execute the query with a progress bar
    result = query.exec(progress_bar=True)

    # Extract list of returned structure entries
    entry_chem_comp_results = result.get("data", {}).get("entries", [])

    n_entries = len(entry_chem_comp_results)
    logger.info("Fetched chemical component data for {n_entries} entries", n_entries=n_entries)

    return entry_chem_comp_results


def process_chem_comp_results_and_write_to_file(
    entry_chem_comp_results: dict,
    pdb_to_cc_output_file: str,
    cc_to_pdb_output_file: str,
    cc_extras_output_file: str,
    generate_cc_extra_file: bool,
):
    """Process the fetched Data API data to create the mapping between chemical component IDs and PDB IDs"""

    logger.info(
        "Processing {n_entries} entries into ligand mapping files",
        n_entries=len(entry_chem_comp_results),
    )

    # Dictionary to collect mapping from chem_comp_id to a set of PDB IDs
    pdb_to_chem_comp_map = {}
    chem_comp_to_pdb_map = {}

    # Iterate over all returned entries
    for entry in entry_chem_comp_results:
        pdb_id = entry.get("rcsb_id")

        # --- 1. Nonpolymer Entities ---
        # Small molecule ligands
        nonpolymer_entities = entry.get("nonpolymer_entities")
        if nonpolymer_entities:
            for nonpolymer in nonpolymer_entities:
                nonpolymer_comp = nonpolymer.get(
                    "rcsb_nonpolymer_entity_container_identifiers"
                )
                if nonpolymer_comp:
                    chem_id = nonpolymer_comp.get("nonpolymer_comp_id")
                    if chem_id:
                        pdb_to_chem_comp_map.setdefault(pdb_id, set()).add(chem_id)
                        chem_comp_to_pdb_map.setdefault(chem_id, set()).add(pdb_id)

        # --- 2. Branched Entities ---
        # Includes things like monomers in saccharides
        branched_entities = entry.get("branched_entities")
        if branched_entities:
            for branched in branched_entities:
                branched_container_identifiers = branched.get(
                    "rcsb_branched_entity_container_identifiers"
                )
                if branched_container_identifiers:
                    chem_comp_monomers = branched_container_identifiers.get(
                        "chem_comp_monomers"
                    )
                    if chem_comp_monomers:
                        for chem_id in chem_comp_monomers:
                            pdb_to_chem_comp_map.setdefault(pdb_id, set()).add(chem_id)
                            chem_comp_to_pdb_map.setdefault(chem_id, set()).add(pdb_id)

        # --- 3. Polymer Entities ---
        polymer_entities = entry.get("polymer_entities")
        if polymer_entities:
            for polymer in polymer_entities:
                polymer_entity_container_identifiers = polymer.get(
                    "rcsb_polymer_entity_container_identifiers"
                )
                if polymer_entity_container_identifiers:
                    # Includes non-standard residues within polymer chains
                    chem_comp_nstd_monomers = polymer_entity_container_identifiers.get(
                        "chem_comp_nstd_monomers"
                    )
                    if chem_comp_nstd_monomers:
                        for chem_id in chem_comp_nstd_monomers:
                            pdb_to_chem_comp_map.setdefault(pdb_id, set()).add(chem_id)
                            chem_comp_to_pdb_map.setdefault(chem_id, set()).add(pdb_id)
                    # Includes standard residues within polymer chains
                    chem_comp_poly_monomers = polymer_entity_container_identifiers.get(
                        "chem_comp_monomers"
                    )
                    if chem_comp_poly_monomers:
                        for chem_id in chem_comp_poly_monomers:
                            pdb_to_chem_comp_map.setdefault(pdb_id, set()).add(chem_id)
                            chem_comp_to_pdb_map.setdefault(chem_id, set()).add(pdb_id)

    # Write the final PDB -> CC mapping to a TSV file
    with open(pdb_to_cc_output_file, "w", encoding="utf-8") as f:
        for pdbid, ccids in pdb_to_chem_comp_map.items():
            f.write(f"{pdbid}\t{' '.join(sorted(ccids))}\n")
    logger.info(
        "PDB → CC mapping written to '{path}' for {n_pdb} PDB IDs",
        path=pdb_to_cc_output_file,
        n_pdb=len(pdb_to_chem_comp_map),
    )

    # Write the final CC -> PDB mapping to a TSV file
    with open(cc_to_pdb_output_file, "w", encoding="utf-8") as f:
        for ccid, pdb_ids in chem_comp_to_pdb_map.items():
            f.write(f"{ccid}\t{' '.join(sorted(pdb_ids))}\n")
    logger.info(
        "CC → PDB mapping written to '{path}' for {n_cc} chemical components",
        path=cc_to_pdb_output_file,
        n_cc=len(chem_comp_to_pdb_map),
    )

    # Write the chemical components extra file (if requested)
    if generate_cc_extra_file:
        logger.info(
            "Generating chemical component extras for {n_cc} components",
            n_cc=len(chem_comp_to_pdb_map),
        )
        cc_extra_tuple_list = generate_cc_extra_data(chem_comp_to_pdb_map)
        with open(cc_extras_output_file, "w", encoding="utf-8") as f:
            f.write("id\tcount\tname\tformula\n")
            for cc_id, cc_occurrence_count, cc_name, cc_formula in cc_extra_tuple_list:
                f.write(f"{cc_id}\t{cc_occurrence_count}\t{cc_name}\t{cc_formula}\n")
        logger.info(
            "CC extras written to '{path}' for {n_cc} components",
            path=cc_extras_output_file,
            n_cc=len(cc_extra_tuple_list),
        )


def generate_cc_extra_data(chem_comp_to_pdb_map):
    """Generate the chemical component extra data"""
    cc_list = list(chem_comp_to_pdb_map.keys())
    if len(cc_list) == 0:
        return []

    # Fetch extra chemical component data
    query = Query(
        input_type="chem_comps",
        input_ids=cc_list,
        return_data_list=["rcsb_id", "chem_comp.formula", "chem_comp.name"],
    )
    cc_result = query.exec(progress_bar=True)
    chem_comp_results = cc_result.get("data", {}).get("chem_comps", [])

    # Process the results into a list of tuples
    cc_extra_tuples = []
    for cc in chem_comp_results:
        cc_name, cc_formula = None, None
        cc_id = cc["rcsb_id"]
        cc_data = cc.get("chem_comp")
        if cc_data:
            cc_name = cc_data.get("name")
            if cc_name:
                cc_name = cc_name.replace("\n", "")  # strip newline characters
            cc_formula = cc_data.get("formula")
        cc_occurrence_count = len(chem_comp_to_pdb_map[cc_id])
        cc_extra_tup = (cc_id, cc_occurrence_count, cc_name, cc_formula)
        cc_extra_tuples.append(cc_extra_tup)

    cc_extra_tuples_sorted = sorted(cc_extra_tuples, key=lambda x: x[1], reverse=True)
    return cc_extra_tuples_sorted


def generate_pdb_ligand_mappings(
    chem_comp_types=None,
    pdb_to_cc_output_file: str = "pdb-to-cc.tsv",
    cc_to_pdb_output_file: str = "cc-to-pdb.tsv",
    cc_extras_output_file: str = "cc-counts-extra.tsv",
    max_concurrent_api_requests: int = 10,
    generate_cc_extra_file: bool = False,
) -> None:
    """
    High-level API for generating PDB/ligand mapping files.

    This is the preferred entrypoint when using this module from Python code.

    Parameters
    ----------
    chem_comp_types:
        Iterable of chemical component type labels to include. Must be drawn
        from ``ALLOWED_CHEM_COMP_TYPES``. If ``None``, uses ``DEFAULT_CHEM_COMP_TYPES``.
    pdb_to_cc_output_file:
        Path for the PDB → CC mapping TSV file.
    cc_to_pdb_output_file:
        Path for the CC → PDB mapping TSV file.
    cc_extras_output_file:
        Path for the CC extras TSV file.
    max_concurrent_api_requests:
        Maximum number of concurrent Data API requests.
    generate_cc_extra_file:
        If ``True``, also generate the CC extras file.
    """
    if chem_comp_types is None:
        chem_comp_types = DEFAULT_CHEM_COMP_TYPES

    # Map high-level chem component type labels to the underlying RCSB Data API
    # return fields.
    chem_comp_types_to_include = [
        CHEMICAL_COMPONENT_TYPES_ARG_MAPPINGS[cc_type] for cc_type in chem_comp_types
    ]

    # Configure the Data API client
    config.DATA_API_MAX_CONCURRENT_REQUESTS = max_concurrent_api_requests

    start = time.time()
    entry_chem_comp_data = fetch_all_chem_comp_ids(chem_comp_types_to_include)
    process_chem_comp_results_and_write_to_file(
        entry_chem_comp_data,
        pdb_to_cc_output_file,
        cc_to_pdb_output_file,
        cc_extras_output_file,
        generate_cc_extra_file,
    )
    end = time.time()
    print(f"Processing completed in {end - start:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mappings between PDB structures and ligands.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    allowed_cc_types = list(ALLOWED_CHEM_COMP_TYPES)
    cc_type_help_text = (
        "Chemical component types to include (space separated).\n"
        "Allowed values:\n"
        "  nonpolymer   - Ligands as nonpolymer entities (e.g., a non-covalently bound ATP ligand)\n"
        "  branched     - Monomers in branched entities (e.g., a subunit in a carbohydrate)\n"
        "  polymer_nstd - Non-standard monomers in polymer chains\n"
        "  polymer_std  - Standard monomers in polymer chains\n"
        "Default is %(default)s."
    )
    parser.add_argument(
        "--chem_comp_types",
        nargs="+",
        choices=allowed_cc_types,
        default=["nonpolymer", "branched", "polymer_nstd"],
        help=cc_type_help_text,
    )
    parser.add_argument(
        "--pdb_to_cc_output_file",
        default="pdb-to-cc.tsv",
        help="Mapping between all PDB structures and the ligands present in each structure (default: %(default)s).",
    )

    parser.add_argument(
        "--cc_to_pdb_output_file",
        default="cc-to-pdb.tsv",
        help="Mapping between all ligands in the PDB and the corresponding list of PDB structures in which they are present (default: %(default)s).",
    )

    parser.add_argument(
        "--cc_extras_output_file",
        default="cc-counts-extra.tsv",
        help="Tabulation of the number of PDB entries containing each chemical component, including name and formula (default: %(default)s).",
    )

    parser.add_argument(
        "--max_concurrent_api_requests",
        type=int,
        default=10,
        help="Maximum number of concurrent Data API requests (default: %(default)s).",
    )

    parser.add_argument(
        "--generate_cc_extra_file",
        action="store_true",
        default=False,
        help="Turn on generation of the chemical components extra file (see '--cc_extras_output_file').",
    )

    args = parser.parse_args()

    generate_pdb_ligand_mappings(
        chem_comp_types=args.chem_comp_types,
        pdb_to_cc_output_file=args.pdb_to_cc_output_file,
        cc_to_pdb_output_file=args.cc_to_pdb_output_file,
        cc_extras_output_file=args.cc_extras_output_file,
        max_concurrent_api_requests=args.max_concurrent_api_requests,
        generate_cc_extra_file=args.generate_cc_extra_file,
    )
