import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(1500)

project = 'irp-jm4622'
copyright = '2023, edsml-jm4622'
author = 'edsml-jm4622'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build"]
autoclass_content = "both"

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'collapse_navigation': False,
    'display_version': False,
    'navigation_depth': 6, }
