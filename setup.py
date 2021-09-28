import codecs
import io
import os
import re
from pprint import pprint

import versioneer
from setuptools import find_packages, setup

VERSION = None
HERE = os.path.abspath(os.path.dirname(__file__))

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
    "extras": read_requirements(".requirements/dev.in"),
}
# Add all requires
all_requires = []
for k, v in EXTRA_REQUIRES.items():
    all_requires.extend(v)
EXTRA_REQUIRES["all"] = set(all_requires)

pprint(EXTRA_REQUIRES)

setup(
    name="graphein",
    version="1.0.0"#versioneer.get_version(),
    #cmdclass=versioneer.get_cmdclass(),
    description="Machine Learning Library Extensions",
    author="Arian Jamasb",
    author_email="arian@jamasb.io",
    url="https://github.com/a-r-j/graphein",
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
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    long_description="""
Graphein is a python package for working with protein structure graphs
Contact
=============
If you have any questions or comments about graphein,
please feel free to contact me via
email: arian@jamasb.io
or Twitter: https://twitter.com/arianjamasb
This project is hosted at https://github.com/a-r-j/graphein
""",
)
