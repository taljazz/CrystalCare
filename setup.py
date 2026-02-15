from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "simplex5d",
        ["simplex5d.cpp"],
        # No include_dirs needed for same-directory header
    ),
]

setup(
    name="simplex5d",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)