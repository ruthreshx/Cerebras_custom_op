# PyTorch Operations

# Assignment 2 & 4: Custom PyTorch Operations

## Overview
This assignment involves decomposing specific PyTorch operations using only the supported operations in the C++ API. The objective is to ensure the implementations are robust, validated through comprehensive GTest cases, and extended to Python using `pybind11`. The implementation should be well-documented and maintainable.

- Decompose the following PyTorch operations:
  - `torch.minimum`
  - `torch.heaviside`
  - `torch.special.entr`
  - `torch.nn.ELU`
  - `torch.addBMM`
  - `torch.addMM`
  - `torch.addMV`


## Project Structure

Below is the overview of the directory structure:
```
cerebras/ 
    ├── include/ 
        ├── include.h # headers of custom ops
    ├── pybind/ 
        ├── pybind.cpp # pybind11 of custom ops
    ├── src/ 
        ├── custom_ops.cpp # Implementation of custom ops
    ├── test/ 
        ├── custom_ops.py# Pytest cases for validating custom ops 
    ├── gtest/ 
        ├── custom_ops.cpp# Gtest cases for validating custom ops 
    ├── setup.py # build configuration file 
    └── README.md # Project documentation
```

### Directory Descriptions

- **include/**: Contains the headers for the custom operations.
- **pybind/**: Contains the pybind for the custom operations.
- **src/**: Contains the source code for the custom operations.
- **test/**: Contains test file to validate the functionality of the custom operations using pytest.
- **gtest/**: Contains test file to validate the functionality of the custom operations using gtest.
- **setup.py**: Defines the build configuration, including compiler options and linking the necessary libraries (such as PyTorch, etc.) to compile the C++ code.
- **README.md**: Provides documentation and instructions for the project.

## Installation

To get started with this project, follow the steps below to clone the repository and install the required dependencies.

### Clone the Repository

Run the following commands in your terminal:

```bash
git clone https://github.com/ruthreshx/Cerebras_custom_op.git
cd Cerebras
```

## 1. Install Dependencies

If the **PyTorch (Torch)** package is not installed on your system, you'll need to download the necessary **LibTorch** package for your C++ project. Here's how:

### Download the LibTorch Package

To download the **LibTorch CPU-only** version, use the following command:

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

Alternatively, you can visit the PyTorch official website to download different variants of LibTorch (either CPU or GPU-enabled).

## 2. Build the Project

Once you've installed the **LibTorch** dependencies, follow these steps to build the project:

1. Export necessary configs for setup and build the project:

   
```bash
python setup.py build
python setup.py install
python setup.py build_ext --inplace
```

## 3. Run the pytest

1. Export necessary configs for setup and build the project:

   
```bash
cd test
python -m pytest -v <file_name.py>
```

## 4. Run the gtest

1. Export necessary configs for cmake and build the project:

   
```bash
mkdir build
cd build
cmake -DPATH_TO_GTEST=/usr/src/gtest -DCMAKE_PREFIX_PATH=/home/cerebras/env/lib/python3.8/site-packages/torch/share/cmake ..
make
test_module
```
