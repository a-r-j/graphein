"""
Module that defines a setup function and publishes the package to PyPI.
Use the command `python setup.py upload`.
"""

import codecs
import contextlib
import io
import os
import re
import sys
from pprint import pprint
from shutil import rmtree
from typing import List

from setuptools import Command, find_packages, setup

VERSION = None
HERE = os.path.abspath(os.path.dirname(__file__))
NAME = "graphein"

# Import the PYPI README and use it as the long-description.
# Note: this will only work if "README.md" is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


with io.open(
    os.path.join(os.path.dirname(__file__), "graphein/__init__.py"),
    encoding="utf-8",
) as f:
    for l in f:
        if not l.startswith("__version__"):
            continue
        VERSION = l.split("=")[1].strip(" \"'\n")
        break
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open
    return codecs.open(os.path.join(HERE, *parts), "r").read()


def read_requirements(*parts):
    """
    Return requirements from parts.
    Given a requirements.txt (or similar style file),
    returns a list of requirements.
    Assumes anything after a single '#' on a line is a comment, and ignores
    empty lines.
    :param parts: list of filenames which contain the installation "parts",
        i.e. submodule-specific installation requirements
    :returns: A compiled list of requirements.
    """
    requirements = []
    for line in read(*parts).splitlines():
        new_line = re.sub(  # noqa: PD005
            r"(\s*)?#.*$",  # the space immediately before the
            # hash mark, the hash mark, and
            # anything that follows it
            "",  # replace with a blank string
            line,
        )
        new_line = re.sub(  # noqa: PD005
            r"-r.*$",  # link to another requirement file
            "",  # replace with a blank string
            new_line,
        )
        new_line = re.sub(  # noqa: PD005
            r"-e \..*$",  # link to editable install
            "",  # replace with a blank string
            new_line,
        )
        # print(line, "-->", new_line)
        if new_line:  # i.e. we have a non-zero-length string
            requirements.append(new_line)
    return requirements


INSTALL_REQUIRES = read_requirements(".requirements/base.in")
EXTRA_REQUIRES = {
    "dev": read_requirements(".requirements/dev.in"),
    "extras": read_requirements(".requirements/extras.in"),
}
# Add all requires
all_requires = []
for v in EXTRA_REQUIRES.values():
    all_requires.extend(v)
EXTRA_REQUIRES["all"] = set(all_requires)

pprint(EXTRA_REQUIRES)


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options: List = []

    @staticmethod
    def status(s):
        """Print things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Publish package to PyPI."""
        with contextlib.suppress(OSError):
            self.status("Removing previous builds…")
            rmtree(os.path.join(HERE, "dist"))
        self.status("Building Source and Wheel (universal) distribution…")
        os.system(
            "{0} setup.py sdist bdist_wheel --universal".format(sys.executable)
        )

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(VERSION))
        os.system("git push --tags")

        sys.exit()


setup(
    name="graphein",
    version="1.7.7",
    description="Protein & Interactomic Graph Construction for Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Arian Jamasb",
    author_email="arian@jamasb.io",
    url="https://github.com/a-r-j/graphein",
    project_urls={
        "homepage": "https://github.com/a-r-j/graphein",
        "changelog": "https://github.com/a-r-j/graphein/blob/master/CHANGELOG.md",
        "issue": "https://github.com/a-r-j/graphein/issues",
        "documentation": "https://graphein.ai/",
    },
    packages=find_packages(),
    package_data={
        "": ["LICENSE.txt", "README.md", "requirements.txt", "*.csv"]
    },
    include_package_data=True,
    # install_requires=install_reqs,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRES,
    python_requires=">=3.7",
    license="MIT",
    platforms="any",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 5 - Production/Stable",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
    entry_points={
        "console_scripts": [
            "graphein = graphein.cli:main",
        ],
    },
)
