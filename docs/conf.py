project = "GuPPy"
copyright = "2024, LernerLab"
author = "LernerLab"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

html_theme = "pydata_sphinx_theme"

source_suffix = [".rst", ".md"]

myst_enable_extensions = ["attrs_block", "colon_fence", "dollarmath"]
myst_heading_anchors = 3

html_favicon = "../assets/favicon.png"

# A CSS-only static directory: registering _static/ instead would copy its 11 MB of
# screenshots a second time, alongside the copies the image directive already makes.
html_static_path = ["_css"]
html_css_files = ["brand.css"]

html_theme_options = {
    "github_url": "https://github.com/LernerLab/GuPPy",
    "header_links_before_dropdown": 7,
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/guppy-neuro/",
            "icon": "fa-brands fa-python",
        },
        {
            "name": "Issue tracker",
            "url": "https://github.com/LernerLab/GuPPy/issues",
            "icon": "fa-solid fa-circle-dot",
        },
    ],
    "logo": {
        "image_light": "../assets/GuppyMark.png",
        "image_dark": "../assets/GuppyMark.png",
        "text": "GuPPy",
        "alt_text": "GuPPy",
    },
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
