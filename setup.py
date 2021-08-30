import io
import os

import versioneer
from setuptools import find_packages, setup

VERSION = None
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

#REQUIREMENTS_FILE = os.path.join(PROJECT_ROOT, "requirements.txt")

#with open(REQUIREMENTS_FILE) as f:
#    install_reqs = f.read().splitlines()

install_reqs = ["setuptools"]
#install_reqs.append("setuptools")

setup(
    name="graphein",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
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
