import os
import sys

# -- Path setup --------------------------------------------------------------
# If your documentation documents your source code, point Sphinx to it.
# This injects your project root into the Python path so autodoc can find your modules.
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------
project = 'Your Project Name'
copyright = '2026, Your Name or Company'
author = 'Your Name'
release = '0.1.0'  # The full version, including alpha/beta/rc tags


# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here as strings.
extensions = [
    'sphinx.ext.autodoc',      # Automatically generate docs from docstrings
    'sphinx.ext.napoleon',     # Support for Google/NumPy style docstrings
    'sphinx.ext.viewcode',     # Add links to highlighted source code
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages. 
# Read the Docs provides its own theme, but 'sphinx_rtd_theme' is standard for local testing.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here.
html_static_path = ['_static']
