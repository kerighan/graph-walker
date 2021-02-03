from setuptools import setup, find_packages, Extension
from glob import glob
import pybind11
import os


compiler_args = "-Ofast -mavx2 -march=native".split()
ext_modules = [
    Extension(
        '_walker',
        sorted(glob("src/*.cpp")),
        language='c++',
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(True),],
        extra_compile_args=compiler_args)
]


setup(
    name="graph-walker",
    version="1.0.0",
    author="Maixent Chenebaux",
    author_email="max.chbx@gmail.com",
    description="Fastest library for random walks on graph",
    url="https://github.com/kerighan/graph-walker",
    packages=find_packages(),
    install_requires=["networkx", "numpy", "scipy", "pybind11"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=ext_modules
)
