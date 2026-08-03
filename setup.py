# setup.py

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "motiflab_cpp",
        ["cpp/src/encoder.cpp"],
        cxx_std=17,
        extra_compile_args=["/openmp"],
    ),
]

setup(
    name="motiflab_cpp",
    version="1.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)