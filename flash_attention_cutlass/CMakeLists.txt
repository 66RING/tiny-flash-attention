cmake_minimum_required(VERSION 3.26)
project(cutlass CUDA CXX)

# set environment PATH for cmake
set(CMAKE_PREFIX_PATH $ENV{VIRTUAL_ENV}/lib/python3.11/site-packages/torch/share/cmake)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_COLOR_DIAGNOSTICS ON)


find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories(
    "third-party/cutlass/include"
    "include"
)

file(GLOB CUDA_SOURCE_FILES "./standalone_src/*.cu" "./csrc/*.cu")
foreach(CUDA_SOURCE_FILE ${CUDA_SOURCE_FILES})
    # NOTE: NAME_WE: name without extension
    # Extract the filename ${CUDA_SOURCE_FILE} without the extension to EXECUTABLE_NAME
    get_filename_component(EXECUTABLE_NAME ${CUDA_SOURCE_FILE} NAME_WE)
    # NOTE: string(FIND ...) to find a substring, result store in containsAten
    string(FIND ${EXECUTABLE_NAME} "aten" containsAten)

    # Create an executable for each source file
    add_executable(${EXECUTABLE_NAME} ${CUDA_SOURCE_FILE})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CXX_STANDARD 17 CUDA_ARCHITECTURES 89)
    if(containsAten GREATER -1)
        # Link torch lib for some case
        message(STATUS "Linking torch libraries to ${EXECUTABLE_NAME}")
        target_link_libraries(${EXECUTABLE_NAME} "${TORCH_LIBRARIES}")
    endif()
    # target_compile_options(${EXECUTABLE_NAME} PRIVATE -G -g)
    target_compile_options(${EXECUTABLE_NAME} PRIVATE -O3 -lineinfo)
endforeach()
