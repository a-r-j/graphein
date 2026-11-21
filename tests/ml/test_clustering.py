"""Tests for graphein.ml.clustering"""

import os
import stat
import sys

import pytest

from graphein.ml.clustering import blast

FAKE_MAKEBLASTDB = """#!/bin/sh
touch "$4.phr" "$4.pin" "$4.psq"
"""

# Writes one hit to the file after -out, and the args it got to args.txt
FAKE_BLASTP = """#!/bin/sh
printf '%s\\n' "$@" > "$(dirname "$0")/args.txt"
while [ "$1" != "-out" ]; do shift; done
printf 'hit\\t90.0\\t10\\t1\\t0\\t1\\t10\\t1\\t10\\t1e-5\\t9\\t10\\n' > "$2"
"""


@pytest.mark.skipif(sys.platform == "win32", reason="shell scripts")
def test_blast_paths_are_not_run_by_a_shell(tmp_path):
    """Paths with spaces or shell metacharacters reach BLAST as one argument
    and are never executed."""
    tools = tmp_path / "blast; touch pwned"
    tools.mkdir()
    for name, script in (
        ("makeblastdb", FAKE_MAKEBLASTDB),
        ("blastp", FAKE_BLASTP),
    ):
        exe = tools / name
        exe.write_text(script)
        exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    database = tmp_path / "my db; touch pwned.fasta"
    database.write_text(">hit\nACDEFGHIKLMNPQRSTVWY\n")

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        results = blast(
            "ACDEFGHIKLMNPQRSTVWY",
            str(database),
            blast_path=f"{tools}/",
            tmp_folder=str(tmp_path),
        )
    finally:
        os.chdir(cwd)

    assert results == [
        ["hit", 90.0, 10.0, 1.0, 0.0, 1.0, 10.0, 1.0, 10.0, 1e-5, 9.0, 10.0]
    ]
    assert not (tmp_path / "pwned").exists()
    args = (tools / "args.txt").read_text().splitlines()
    assert args[args.index("-db") + 1] == str(database)
    # The query and output files are cleaned up
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [tools.name, database.name]
        + [database.name + e for e in (".phr", ".pin", ".psq")]
    )
