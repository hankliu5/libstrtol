#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy


hello_module = Extension(
            'hello', ['libstrtol.c'],
            extra_compile_args=["-O3"],
            include_dirs=[numpy.get_include()]
            )

setup(name             = "numpy_c_ext_example",
      version          = "1.0",
      description      = "Example code for blog post.",
      author           = "J. David Lee",
      author_email     = "contact@crumpington.com",
      maintainer       = "contact@crumpington.com",
      url              = "https://www.crumpington.com",
      ext_modules      = [hello_module]
)
