# docs/conf.py

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'LightningMasterPro'
copyright = '2024, LightningMasterPro Team'
author = 'LightningMasterPro Team'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store']

# Options for HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
}

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# Source suffix
source_suffix = {
    '.rst': None,
    '.md': 'markdown',
}