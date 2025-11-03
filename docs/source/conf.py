import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'src')))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QDFlow'
copyright = '2025, Donovan Buterakos, Sandesh Kalantre, Joshua Ziegler, Jacob M. Taylor, Justyna P. Zwolak'
author = 'Donovan Buterakos, Sandesh Kalantre, Joshua Ziegler, Jacob M. Taylor, Justyna P. Zwolak'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinxcontrib.jquery',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True
autodoc_typehints = "none"
toc_object_entries_show_parents = 'hide'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'titles_only': True
}
html_static_path = ['_static']
