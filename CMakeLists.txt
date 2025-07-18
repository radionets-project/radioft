cmake_minimum_required(VERSION 3.18...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Find CUDAToolkit and enable CUDA language if found
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(TORCH_INCLUDE_DIRS
    "${TORCH_PATH}/include"
    "${TORCH_PATH}/include/torch/csrc/api/include"
)
set(TORCH_LIBRARY_DIRS "${TORCH_PATH}/lib")

if(SKBUILD_STATE STREQUAL "editable")
    set(INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/radioft)
else()
    set(INSTALL_DIR ${SKBUILD_PROJECT_NAME})
endif()

python_add_library(_libphase_kernel MODULE
    src/radioft/cuda/kernels/cuda_phase_kernel.cu
    src/radioft/cuda/kernels/cuda_phase_kernel32.cu
    src/radioft/cuda/cuda_bindings.cpp
    WITH_SOABI
)

# Define the module name for PyTorch extension.
# This will then set the module name in
# src/radioft/cuda/cuda_bindings.cpp to _libphase_kernel:
#
# PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) <--
#
target_compile_definitions(_libphase_kernel PRIVATE TORCH_EXTENSION_NAME=_libphase_kernel)

target_include_directories(_libphase_kernel PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_directories(_libphase_kernel PRIVATE ${TORCH_LIBRARY_DIRS})
target_link_libraries(_libphase_kernel PRIVATE
    CUDA::cudart
    torch
    torch_cpu
    torch_python
    c10
)

set_target_properties(_libphase_kernel PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# Install _libphase_kernel under radioft.cuda.kernels._libphase_kernel
install(TARGETS _libphase_kernel DESTINATION ${INSTALL_DIR}/cuda/kernels)
