from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import pybind11

setup(
    name='custom_module',
    ext_modules=[
        CppExtension(
            name='custom_module',  # Ensure this matches your PYBIND11_MODULE name
            sources=[
                'src/minimum.cpp', # Source files
                'src/heaviside.cpp', # Source files
                'src/elu.cpp',    # Source files
                'src/entr.cpp',    # Source files
                'pybind/bindings.cpp'  # Pybind wrapper file
            ],
            include_dirs=['include',
                         pybind11.get_include(),     # Pybind11 headers
                         pybind11.get_include(True),],  # Path to include directory
            libraries=["torch"],           # Link against PyTorch
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
