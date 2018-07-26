# -*- coding: utf-8 -*-
#
# __init__.py
#
# This module is part of skxtend.
#

"""
Initializer of skxtend.
"""

__author__ = 'Severin E. R. Langberg'
__email__ = 'Langberg91@gmail.no'
__status__ = 'Operational'


from .clustering import FuzzyCMeans
from .regression import TikhonovCV

# Imported through `import *`
__all__ = ['FuzzyCMeans', 'TikhonovCV']
