# Configuration file for the Sphinx documentation builder.

# -- Path setup
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information

project = 'edu-toolkit'
copyright = '2023, Rose E. Wang'
author = 'Rose E. Wang'

release = '0.1'
version = '0.1.0'

# -- General configuration
source_suffix = ['.rst', '.md']

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.coverage',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'