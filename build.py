from pybind11.setup_helpers import Pybind11Extension, build_ext
from glob import glob

def build(setup_kwargs):
    ext_modules = [
    Pybind11Extension(
        '_walker',
        sorted(glob("src/*.cpp")),
        language='c++',
        extra_compile_args=["-Ofast", "-std=c++11"]
    )
    ]
    setup_kwargs.update({
        "ext_modules": ext_modules,
        "cmd_class": {"build_ext": build_ext},
        "zip_safe": False,
    })
