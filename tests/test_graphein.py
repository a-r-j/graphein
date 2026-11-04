"""Unit test for the graphein package."""

import sys

# Import package, test suite, and other packages as needed
import graphein  # noqa: F401


def test_sidechainnet_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "graphein" in sys.modules
