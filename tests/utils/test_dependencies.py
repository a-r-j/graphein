"""Tests for graphein.utils.dependencies"""

import ast
import re
from pathlib import Path
from typing import List, Tuple

import pytest

from graphein.utils.dependencies import import_message

import tomllib


ROOT = Path(__file__).resolve().parent.parent.parent


def _extras_packages() -> set:
    """Package names installed by ``pip install graphein[extras]``."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return {
        re.split(r"[<>=!~\[ ;]", req)[0].lower()
        for req in pyproject["project"]["optional-dependencies"]["extras"]
    }


def _import_message_calls() -> List[Tuple[str, int, str, bool]]:
    """Return ``(file, line, package, extras)`` for each ``import_message``
    call in the graphein package."""
    calls = []
    for path in sorted((ROOT / "graphein").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", getattr(func, "attr", None))
            if name != "import_message":
                continue
            kwargs = {k.arg: k.value for k in node.keywords}
            package = (
                kwargs["package"] if "package" in kwargs else node.args[1]
            )
            if "extras" in kwargs:
                extras = kwargs["extras"]
            elif len(node.args) > 4:
                extras = node.args[4]
            else:
                extras = ast.Constant(value=False)
            calls.append(
                (
                    str(path.relative_to(ROOT)),
                    node.lineno,
                    ast.literal_eval(package),
                    ast.literal_eval(extras),
                )
            )
    return calls


def test_import_message_extras():
    # The package install line depends on whether we're in a conda env, but
    # the extras line doesn't.
    msg = import_message("graphein.test", "foo", pip_install=True, extras=True)
    assert "pip install graphein[extras]" in msg

    msg = import_message("graphein.test", "foo", pip_install=True)
    assert "graphein[extras]" not in msg


def test_import_message_calls_found():
    assert len(_import_message_calls()) > 0


@pytest.mark.parametrize(
    "path,line,package,extras",
    _import_message_calls(),
    ids=lambda x: x if isinstance(x, str) else None,
)
def test_import_message_extras_flag_matches_extras_requirements(
    path, line, package, extras
):
    """``extras=True`` iff ``package`` is installed with graphein[extras]."""
    in_extras = package.lower() in _extras_packages()
    assert extras == in_extras, (
        f"{path}:{line}: import_message for {package!r} has "
        f"extras={extras}, but {package!r} is "
        f"{'' if in_extras else 'not '}in {EXTRAS_FILE.name}"
    )
