# Set the default build type to Debug if it's not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(NEOHOOKEAN_CUDA LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test test.cu)
add_executable(derivatives_verification derivatives_verification.cu)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Link CUDA Toolkit libraries (if needed)
target_link_libraries(test CUDA::cudart)
