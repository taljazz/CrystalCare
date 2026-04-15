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
    name="crystalcare-simplex5d",
    version="1.0.0",
    description="5D Simplex noise extension for CrystalCare",
    python_requires=">=3.10",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)