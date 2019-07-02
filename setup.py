#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy


python_module = Extension(
            'py_deserialize', ['libstrtol.c', 'pystrtol.c'],
            extra_compile_args=["-O3"],
            include_dirs=[numpy.get_include()]
            )

setup(author           = "Yu-Chia Hank Liu",
      author_email     = "yliu719@ucr.edu",
      ext_modules      = [python_module]
)
