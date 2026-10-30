# AGENTS.md

Graphein builds geometric graphs of proteins, RNA, molecules and interaction
networks (`graphein/protein`, `graphein/rna`, `graphein/molecule`,
`graphein/ppi`, `graphein/grn`) and converts them to ML formats
(`graphein/ml`).

## Tests

- `uv sync --extra dev`, then `uv run pytest tests/<path>`. The pytest
  `addopts` use `--cov`, so pytest-cov (in `dev`) must be installed.
- torch, PyG, DGL and DSSP are in no extra. Tests that need them skip, or
  fail at import for the torch-only `tests/protein/tensor` and `tests/ml`.
  Run them with e.g.
  `uv run --with torch --with torch_geometric pytest tests/ml`.
- Many tests download structures from the RCSB, so they need the network.
- Test structures live in `tests/protein/test_data`.
  `input_pdb_cryst1.pdb` has insertion codes and chains in non-alphabetical
  order (L, H, A); `4hhb.pdb` is a plain four-chain structure.
- For a bug fix, write the regression test first and watch it fail.

## Style

- Format and lint with ruff (line length 79): `uv run ruff format <files>`
  and `uv run ruff check <files>`, on the files you changed. The lint rule
  set is deliberately narrow (see `pyproject.toml`).
- Docstrings use Sphinx fields: `:param x:`, `:type x:`, `:return:`,
  `:rtype:`.

## CHANGELOG

Every user-visible change gets an entry in the top `UNRELEASED` section of
`CHANGELOG.md`, under New Features, Bugfixes, Breaking Changes or Misc. Say
what was wrong for the user, why, and what changed, then link the issue and
PR, matching the existing entries.

## Pull requests

- Fill in `.github/PULL_REQUEST_TEMPLATE.md`. Under testing, give the exact
  commands and results, including suites you couldn't run and why.
- To update a contributor's fork branch, merge `master` into it so the push
  stays fast-forward.

## Protein structure gotchas

- **Row alignment.** `pdb_df` is in file order; `groupby` results are sorted.
  Match rows between dataframes by key (`chain_id`, `residue_number`,
  `insertion`), or by position within the same filtered dataframe. Matching
  by position across differently ordered frames silently puts data on the
  wrong residue.
- **Insertion codes.** Residues `52` and `52A` share `residue_number`; blank
  codes are `""`. Measure sequence separation with
  `graphein.protein.utils.sequence_positions`, which gives inserted residues
  their own positions.
- **Node ids** are `chain:RESNAME:resnum[:insertion]` for residues (e.g.
  `H:TYR:52:A`), with `:ATOM` appended at atom granularity
  (`H:TYR:52:A:CA`).
- **Edge `kind` is a set** shared by every edge function. Add edges with
  `graphein.protein.edges.distance.add_edge`, which adds to the set of an
  existing edge.
