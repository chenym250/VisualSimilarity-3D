# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:08:39 2016

run with `python setup.py build_ext --inplace`

@author: chenym
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'compare in Cython',
    ext_modules = cythonize(["partialcomparison.pyx","fullcomparison.pyx"]),
    include_dirs=[numpy.get_include()]
)
