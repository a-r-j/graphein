# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import plotly.io as pio
import sphinx.ext.autodoc

sphinx.ext.autodoc.NewTypeDataDocumenter.directivetype = "class"

os.environ["PLOTLY_RENDERER"] = "sphinx_gallery"

sys.path.insert(0, os.path.abspath("."))

pio.renderers.default = "sphinx_gallery"


# -- Project information -----------------------------------------------------

project = "Graphein"
author = "Arian Jamasb"
copyright = f"{datetime.datetime.now().year}, {author}"

# The full version, including alpha/beta/rc tags
release = "1.7.1"


# -- General configuration ---------------------------------------------------
master_doc = "index"
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_inline_tabs",
    "sphinxcontrib.gtagjs",
    "sphinxext.opengraph",
    "m2r2",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.napoleon",
    "sphinx_codeautolink",
    # "sphinx_autorun",
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary
nbsphinx_allow_errors = True
nbsphinx_require_js_path = (
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"
)
nbsphinx_kernel_name = "graphein"
nbsphinx_execute = "always"
# nbsphinx_execute = "never"

ogp_site_url = "https://graphein.ai/"
ogp_image = "https://graphein.ai/_static/graphein.png"

gtagjs_ids = [
    "G-ZKD1FQDEYH",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "networkx": ("https://networkx.github.io/documentation/stable/", None),
    "nx": ("https://networkx.github.io/documentation/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "np": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pd": ("https://pandas.pydata.org/docs/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "Sphinx": ("https://www.sphinx-doc.org/en/stable/", None),
    "torch": ("https://pytorch.org/docs/master/", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "torch_geometric": (
        "https://pytorch-geometric.readthedocs.io/en/latest/",
        None,
    ),
}

mathjax_path = (
    "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)
mathjax2_config = {
    "tex2jax": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "processEscapes": True,
        "ignoreClass": "document",
        "processClass": "math|output_area",
    }
}

autodoc_default_options = {
    "special-members": "__init__",
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]
source_suffix = [".rst", ".md"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/graphein.png"
html_title = f"{project} {release}"


def setup(app):
    app.add_js_file(
        "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"
    )
