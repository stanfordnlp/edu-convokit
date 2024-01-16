# Configuration file for the Sphinx documentation builder.

# -- Path setup
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information

project = 'edu-convokit'
copyright = '2023, Rose E. Wang'
author = 'Rose E. Wang'

release = '0.1'
version = '0.1.0'

# -- General configuration
source_suffix = ['.rst', '.md']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.coverage',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = []  

# -- Options for EPUB output
epub_show_urls = 'footnote'


