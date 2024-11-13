cmake_minimum_required(VERSION 3.18)
project(cerebras)

# Find Torch and GTest packages
find_package(Torch REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_subdirectory(${PATH_TO_GTEST} ${CMAKE_BINARY_DIR}/gtest_build)

# Include Torch and project directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Create shared library from custom source files
add_library(custom_module SHARED 
                src/minimum.cpp
                src/heaviside.cpp
                src/elu.cpp
                src/entr.cpp
                src/addmm.cpp
                src/addmv.cpp
                src/addbmm.cpp
                )

# Link Torch libraries to the custom library
target_link_libraries(custom_module "${TORCH_LIBRARIES}")

# Create the test executable from the test source file
add_executable(test_module gtest/test_minimum.cpp gtest/test_elu.cpp gtest/test_entr.cpp gtest/test_heaviside.cpp gtest/test_addmm.cpp gtest/test_addmv.cpp gtest/test_addbmm.cpp)

# Link GTest libraries to the test executable
target_link_libraries(test_module custom_module gtest gtest_main)

# Enable testing and add the test
enable_testing()
add_test(TestModule test_module)