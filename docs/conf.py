# Configuration file for the Sphinx documentation builder.

import yobx

project = "yet-another-onnx-builder"
author = "yet-another-onnx-builder contributors"
release = yobx.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runpython",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "furo"
