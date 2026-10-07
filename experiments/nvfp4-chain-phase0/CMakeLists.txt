cmake_minimum_required(VERSION 3.22)
project(nvfp4_chain_phase0 LANGUAGES CXX CUDA)

# P4.2 Phase-0 offline feasibility bench (design doc
# docs/research/2026-10-08-nvfp4-chained-direct-output.md §9).
# Standalone: no dependency on the h3.c main build system.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 5090: sm_120a required for CUTLASS_ARCH_MMA_SM120_SUPPORTED / UMMA.
set(CMAKE_CUDA_ARCHITECTURES 120a)

# --- CUTLASS headers ---------------------------------------------------------
# Priority: -DCUTLASS_DIR=... > $CUTLASS_DIR > ./_deps/cutlass > FetchContent.
if(NOT DEFINED CUTLASS_DIR AND DEFINED ENV{CUTLASS_DIR})
  set(CUTLASS_DIR $ENV{CUTLASS_DIR})
endif()
if(NOT DEFINED CUTLASS_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/_deps/cutlass/include/cutlass/cutlass.h")
  set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/_deps/cutlass")
endif()

if(NOT DEFINED CUTLASS_DIR)
  message(STATUS "CUTLASS_DIR not set; FetchContent-ing cutlass v4.8.0")
  include(FetchContent)
  FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v4.8.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(cutlass)
  set(CUTLASS_DIR "${cutlass_SOURCE_DIR}")
endif()

find_path(CUTLASS_INCLUDE_DIR cutlass/cutlass.h
  PATHS "${CUTLASS_DIR}/include"
  NO_DEFAULT_PATH)
find_path(CUTLASS_UTILS_INCLUDE_DIR helper.h
  PATHS "${CUTLASS_DIR}/examples/common"
  NO_DEFAULT_PATH)

if(NOT CUTLASS_INCLUDE_DIR)
  message(FATAL_ERROR "CUTLASS headers not found under CUTLASS_DIR=${CUTLASS_DIR}")
endif()

message(STATUS "Using CUTLASS include: ${CUTLASS_INCLUDE_DIR}")

find_package(CUDAToolkit REQUIRED)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-O3>")

add_executable(bench_gemm bench_gemm.cu)
add_executable(sfd_compare sfd_compare.cu)

foreach(tgt bench_gemm sfd_compare)
  target_include_directories(${tgt} PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_DIR}/tools/util/include
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${tgt} PRIVATE CUDA::cudart CUDA::cublasLt)
  # Force the CUDA 12.8 runtime cublasLt (12.4 lacks VEC16_UE4M3 / sm_120 FP4).
  set_target_properties(${tgt} PROPERTIES
    BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}"
    INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")
endforeach()
