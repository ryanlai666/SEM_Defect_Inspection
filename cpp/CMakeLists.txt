cmake_minimum_required(VERSION 3.24)
# Must be set BEFORE project(): enabling CUDA otherwise caches nvcc's default (sm_52), and a later
# set(... CACHE ...) is silently ignored.  RTX 3070 = sm_86; use "75;80;86;89" for Colab T4/A100/L4.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA SM architectures")
endif()
project(semdefect_cpp LANGUAGES CXX CUDA)
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(5|6)")
  message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}: the tensor-core kNN needs sm_70+")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  # static CRT off (pybind11 module + gtest must agree); C++ exceptions on
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc> $<$<COMPILE_LANGUAGE:CXX>:/W3>)
endif()

option(SEMDEFECT_BUILD_TESTS "GoogleTest suite (kernels vs CPU reference)" ON)
option(SEMDEFECT_BUILD_PYTHON "pybind11 module semdefect_cuda" ON)
option(SEMDEFECT_BUILD_TRT_RUNNER "TensorRT C++ runner (needs TensorRT headers, see cpp/third_party/README.md)" OFF)

find_package(CUDAToolkit REQUIRED)

# One source of truth for kernels: the same .cuh files are JIT-compiled by NVRTC in Python.
set(SEMDEFECT_KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src/semdefect/kernels/cuda)

add_library(semkernels STATIC src/kernels.cu src/postprocess.cu src/cpu_reference.cpp)
target_include_directories(semkernels PUBLIC include ${SEMDEFECT_KERNEL_DIR})
target_link_libraries(semkernels PUBLIC CUDA::cudart_static)
target_compile_options(semkernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo --expt-relaxed-constexpr>)

if(SEMDEFECT_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_executable(semkernels_tests tests/test_kernels.cpp)
  target_link_libraries(semkernels_tests PRIVATE semkernels GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(semkernels_tests DISCOVERY_TIMEOUT 60)
endif()

if(SEMDEFECT_BUILD_PYTHON)
  find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(semdefect_cuda python/bindings.cpp)
  target_link_libraries(semdefect_cuda PRIVATE semkernels)
  # drop the built module next to the Python package so `import semdefect.kernels.semdefect_cuda` works
  add_custom_command(TARGET semdefect_cuda POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:semdefect_cuda> ${CMAKE_CURRENT_SOURCE_DIR}/../src/semdefect/kernels/)
endif()

if(SEMDEFECT_BUILD_TRT_RUNNER)
  set(TENSORRT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tensorrt/include CACHE PATH "TensorRT headers")
  add_executable(trt_runner runner/trt_runner.cpp)
  target_include_directories(trt_runner PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(trt_runner PRIVATE semkernels)
  if(NOT WIN32)
    target_link_libraries(trt_runner PRIVATE dl)
  endif()
endif()
