# filename: setup.py
# This script is used to compile our Cython code into a Python extension.

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

# Define the extension module
extensions = [
    Extension(
        "fast_kmer_counter",            # The name of the module to import
        ["fast_kmer_counter.pyx"],      # The Cython source file
        include_dirs=[numpy.get_include()] # Include NumPy headers for C-level integration
    )
]

# Run the setup
setup(
    ext_modules=cythonize(extensions)
)