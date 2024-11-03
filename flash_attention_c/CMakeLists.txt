cmake_minimum_required(VERSION 3.26.4)  # Specify your minimum CMake version

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(_kernels LANGUAGES CUDA CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

file(GLOB PYTORCH_SOURCES "csrc/*.cu" "csrc/*.c" "csrc/*.cpp")
pybind11_add_module(_kernels MODULE ${PYTORCH_CPP_SOURCES} ${PYTORCH_SOURCES})

target_compile_definitions(_kernels PRIVATE -DBSK_TORCH_CHECK) # Enable Torch Tensor Dimension Check
target_compile_options(_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} Python::Python pybind11::module ${TORCH_PYTHON_LIBRARY} OpenMP::OpenMP_CXX)

