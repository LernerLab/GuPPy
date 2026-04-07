project = "GuPPy"
copyright = "2024, LernerLab"
author = "LernerLab"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

html_theme = "pydata_sphinx_theme"

source_suffix = [".rst", ".md"]

myst_enable_extensions = ["colon_fence"]

html_theme_options = {
    "github_url": "https://github.com/LernerLab/GuPPy",
    "logo": {
        "text": "GuPPy",
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
