# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'automl_project'
copyright = '2024, Deborah Dobles & Ana Trusca'
author = 'Deborah Dobles & Ana Trusca'
release = '10-11-2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_theme_options = {
    'sidebar_includehidden': False,
    'show_related': False,
    'nosidebar': True,
}
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
