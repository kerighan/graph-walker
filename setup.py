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
        extra_compile_args=compiler_args)]


setup(
    name="walker",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=ext_modules
)
