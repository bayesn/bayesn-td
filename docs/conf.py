import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# Configuration file for the Sphinx documentation builder.

# -- Project information
project = 'bayesn-td'
copyright = '2025, Matthew Grayling, Stephen Thorp, Kaisey Mandel'
author = 'Matthew Grayling, Stephen Thorp, Kaisey Mandel'
release = '0.1.0'

# -- General configuration
extensions = ['numpydoc', 'sphinx.ext.autosummary', 'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
autosummary_generate = True
numpydoc_class_members_toctree = False
