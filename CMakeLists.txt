cmake_minimum_required(VERSION 3.18)

# Apple Silicon / Metal remains the primary path via the top-level Makefile.
# This CMake file is the Linux/CUDA stub entry (P1). It does not replace
# h3_gpu.m or build the full h3 CLI.

project(h3_cuda_stub LANGUAGES C CXX)

# A prior 5090 cache baked -Wl,-rpath,/opt/nccl/lib into EXE_LINKER_FLAGS.
# That directory ships libstdc++ 6.0.34 and steals the gold U=1 cuDNN plan.
foreach(_h3_lf CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS
               CMAKE_MODULE_LINKER_FLAGS CMAKE_CUDA_EXE_LINKER_FLAGS)
  if(DEFINED ${_h3_lf} AND "${${_h3_lf}}" MATCHES "nccl")
    message(WARNING "H3: stripping NCCL rpath from ${_h3_lf}")
    string(REGEX REPLACE "(^| )-Wl,-rpath,[^ ]*nccl[^ ]*" "" ${_h3_lf}
      "${${_h3_lf}}")
    string(STRIP "${${_h3_lf}}" ${_h3_lf})
    set(${_h3_lf} "${${_h3_lf}}" CACHE STRING "${_h3_lf}" FORCE)
  endif()
endforeach()

set(H3_GPU_BACKEND "metal" CACHE STRING "GPU backend: metal|cuda")
set_property(CACHE H3_GPU_BACKEND PROPERTY STRINGS metal cuda)

if(H3_GPU_BACKEND STREQUAL "metal")
  message(FATAL_ERROR
    "H3_GPU_BACKEND=metal: use the repository Makefile on Darwin/Apple Silicon.\n"
    "For the CUDA probe stub, reconfigure with:\n"
    "  cmake -S . -B build-cuda -DH3_GPU_BACKEND=cuda\n"
    "See docs/nvidia_build.md.")
endif()

if(NOT H3_GPU_BACKEND STREQUAL "cuda")
  message(FATAL_ERROR "H3_GPU_BACKEND must be 'metal' or 'cuda' (got '${H3_GPU_BACKEND}')")
endif()

# RTX 5090 must use CUDA 13+ (conda nvcc or /usr/local/cuda-13*). 4090 keeps
# PATH/conda 12.x when H3_REQUIRE_CUDA13 is off. Never silently pick CUDA 12
# on a 13-required configure.
option(H3_REQUIRE_CUDA13 "Require CUDA Toolkit 13+ (RTX 5090 / sm_120)" OFF)

function(_h3_nvcc_release nvcc_path out_var)
  set(_ver "")
  if(nvcc_path AND EXISTS "${nvcc_path}")
    execute_process(
      COMMAND "${nvcc_path}" --version
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" _m "${_out}")
    if(CMAKE_MATCH_1)
      set(_ver "${CMAKE_MATCH_1}")
    endif()
  endif()
  set(${out_var} "${_ver}" PARENT_SCOPE)
endfunction()

function(_h3_best_nvcc min_ver out_path out_ver)
  set(_best_path "")
  set(_best_ver "0.0")
  set(_cands "")
  if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
    list(APPEND _cands "$ENV{CONDA_PREFIX}/bin/nvcc")
  endif()
  file(GLOB _h3_conda_nvccs
    "$ENV{HOME}/miniconda3/envs/*/bin/nvcc"
    "$ENV{HOME}/miniforge3/envs/*/bin/nvcc"
    "$ENV{HOME}/anaconda3/envs/*/bin/nvcc")
  list(APPEND _cands ${_h3_conda_nvccs})
  foreach(_h3_hint
      "$ENV{CUDA_HOME}/bin/nvcc"
      "$ENV{CUDA_PATH}/bin/nvcc"
      "/usr/local/cuda-13.2/bin/nvcc"
      "/usr/local/cuda-13.1/bin/nvcc"
      "/usr/local/cuda-13.0/bin/nvcc"
      "/usr/local/cuda-13/bin/nvcc"
      "/usr/local/cuda-12.9/bin/nvcc"
      "/usr/local/cuda-12.8/bin/nvcc"
      "/usr/local/cuda/bin/nvcc")
    if(EXISTS "${_h3_hint}")
      list(APPEND _cands "${_h3_hint}")
    endif()
  endforeach()
  foreach(_h3_cand IN LISTS _cands)
    _h3_nvcc_release("${_h3_cand}" _h3_cand_ver)
    if(_h3_cand_ver VERSION_GREATER_EQUAL "${min_ver}" AND
       _h3_cand_ver VERSION_GREATER "${_best_ver}")
      set(_best_ver "${_h3_cand_ver}")
      set(_best_path "${_h3_cand}")
    endif()
  endforeach()
  set(${out_path} "${_best_path}" PARENT_SCOPE)
  set(${out_ver} "${_best_ver}" PARENT_SCOPE)
endfunction()

if(NOT CMAKE_CUDA_COMPILER)
  find_program(_h3_nvcc_path nvcc)
  _h3_nvcc_release("${_h3_nvcc_path}" _h3_path_nvcc_ver)
  set(_h3_need_upgrade FALSE)
  if(H3_REQUIRE_CUDA13 AND
     (NOT _h3_path_nvcc_ver OR _h3_path_nvcc_ver VERSION_LESS "13.0"))
    set(_h3_need_upgrade TRUE)
  elseif(NOT H3_REQUIRE_CUDA13 AND
         (NOT _h3_path_nvcc_ver OR _h3_path_nvcc_ver VERSION_LESS "12.8"))
    set(_h3_need_upgrade TRUE)
  endif()
  if(_h3_need_upgrade)
    if(H3_REQUIRE_CUDA13)
      _h3_best_nvcc("13.0" _h3_picked _h3_picked_ver)
    else()
      _h3_best_nvcc("13.0" _h3_picked _h3_picked_ver)
      if(NOT _h3_picked)
        _h3_best_nvcc("12.8" _h3_picked _h3_picked_ver)
      endif()
    endif()
    if(_h3_picked)
      set(CMAKE_CUDA_COMPILER "${_h3_picked}" CACHE FILEPATH
          "nvcc (auto-selected)" FORCE)
      message(STATUS
        "H3: PATH nvcc is ${_h3_path_nvcc_ver}; "
        "using ${_h3_picked} (${_h3_picked_ver})")
    endif()
  elseif(_h3_nvcc_path)
    # Pin the absolute path; check_language(CUDA) often misses conda nvcc on PATH.
    set(CMAKE_CUDA_COMPILER "${_h3_nvcc_path}" CACHE FILEPATH "nvcc" FORCE)
    message(STATUS "H3: using PATH nvcc ${_h3_path_nvcc_ver}: ${_h3_nvcc_path}")
  endif()
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR
    "H3_GPU_BACKEND=cuda but no CUDA compiler (nvcc) was found.\n"
    "RTX 5090: CUDA Toolkit 13+ (set H3_REQUIRE_CUDA13=ON).\n"
    "RTX 4090: CUDA 12.8+ for sm_120 fatbin, else sm_89 only.\n"
    "See docs/nvidia_build.md.")
endif()

# Must set architectures BEFORE enable_language(CUDA); otherwise CMake may
# default to a legacy arch (e.g. 52) for the first CUDA targets.
# Ada (4090 sm_89) + Blackwell (5090 sm_120).
set(_h3_cuda_archs "89")
_h3_nvcc_release("${CMAKE_CUDA_COMPILER}" _h3_nvcc_ver)
if(H3_REQUIRE_CUDA13 AND
   (NOT _h3_nvcc_ver OR _h3_nvcc_ver VERSION_LESS "13.0"))
  message(FATAL_ERROR
    "H3_REQUIRE_CUDA13=ON but nvcc is ${_h3_nvcc_ver} (${CMAKE_CUDA_COMPILER}).\n"
    "Point CMAKE_CUDA_COMPILER at CUDA 13+ (e.g. conda env nvcc).")
endif()
if(_h3_nvcc_ver VERSION_GREATER_EQUAL "12.8")
  list(APPEND _h3_cuda_archs "120")
  message(STATUS "nvcc ${_h3_nvcc_ver}: including architecture 120")
else()
  message(STATUS
    "nvcc ${_h3_nvcc_ver}: omitting architecture 120 "
    "(needs CUDA Toolkit 12.8+). Building for sm_89 only.")
endif()
set(CMAKE_CUDA_ARCHITECTURES "${_h3_cuda_archs}" CACHE STRING
    "CUDA architectures (89=Ada, 120=Blackwell)" FORCE)
message(STATUS "H3 CUDA architectures (pre-enable): ${CMAKE_CUDA_ARCHITECTURES}")

enable_language(CUDA)
# FindCUDAToolkit's find_library hits distro CUDA 11 in /usr/lib before conda
# CUDA 13. Pin search to the same prefix as nvcc.
get_filename_component(_h3_nvcc_bindir "${CMAKE_CUDA_COMPILER}" DIRECTORY)
get_filename_component(_h3_cuda_root "${_h3_nvcc_bindir}" DIRECTORY)
set(CUDAToolkit_ROOT "${_h3_cuda_root}" CACHE PATH "CUDA toolkit root matching nvcc" FORCE)
list(PREPEND CMAKE_PREFIX_PATH "${_h3_cuda_root}")
list(PREPEND CMAKE_LIBRARY_PATH
  "${_h3_cuda_root}/lib"
  "${_h3_cuda_root}/targets/x86_64-linux/lib"
  "${_h3_cuda_root}/lib64")
find_package(CUDAToolkit REQUIRED)

# Conda CUDA toolkits inject x86_64-conda-linux-gnu/sysroot -L paths whose
# libpthread.so is a linker script with absolute /lib64/... (RHEL layout).
# On Ubuntu that breaks the final link; keep system lib dirs, but PREPEND the
# toolkit lib so CUDA 13 libcudart.so.13 wins over distro CUDA 11 in /usr/lib.
set(_h3_sys_link_dirs
  "/usr/lib/x86_64-linux-gnu;/lib/x86_64-linux-gnu;/usr/lib;/lib")
if(CUDAToolkit_LIBRARY_DIR)
  set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES
    "${CUDAToolkit_LIBRARY_DIR};${_h3_sys_link_dirs}")
  set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES
    "${CUDAToolkit_LIBRARY_DIR};${_h3_sys_link_dirs}")
else()
  set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${_h3_sys_link_dirs}")
  set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "${_h3_sys_link_dirs}")
endif()

message(STATUS "H3 CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

option(H3_ENABLE_CUDNN "Link cuDNN and enable H3_HAVE_CUDNN SDPA hooks" ON)
option(H3_ENABLE_CUDNN_FRONTEND
  "Build fused SDPA via vendored cudnn_frontend (needs cuDNN 9)" ON)
set(H3_CUDNN_FRONTEND_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn_frontend/include"
  CACHE PATH "Path to cudnn_frontend headers (cudnn_frontend.h)")

add_library(h3_gpu_cuda STATIC
  h3_gpu_cuda.cu
  h3_gpu_cuda_audio.cu
  h3_gpu_cuda_vae.cu
  h3_nvfp4_cuda.cu
  h3_cuda_device_caps.c
  h3_sdpa_registry.c)
target_include_directories(h3_gpu_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(h3_gpu_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::cuda_driver ${CMAKE_DL_LIBS})
# Toolkit first so NVRTC/JIT matches gold cafe. Distro libstdc++ is added
# later on generate (DT_RPATH) so NCCL's copy cannot win. libcudart.so.13
# beats distro CUDA 11 by SONAME.
if(DEFINED CUDAToolkit_LIBRARY_DIR)
  target_link_directories(h3_gpu_cuda PUBLIC ${CUDAToolkit_LIBRARY_DIR})
  set_property(TARGET h3_gpu_cuda PROPERTY BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")
  set_property(TARGET h3_gpu_cuda PROPERTY INSTALL_RPATH "${CUDAToolkit_LIBRARY_DIR}")
  target_link_options(h3_gpu_cuda PUBLIC
    "-Wl,-rpath,${CUDAToolkit_LIBRARY_DIR}"
    "-Wl,-rpath-link,${CUDAToolkit_LIBRARY_DIR}")
endif()
target_compile_options(h3_gpu_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
set_target_properties(h3_gpu_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON)

# --- NVFP4 P4.2 Phase-1: FC1→FC2 chained direct-output (CUTLASS, sm_120a) ---
# Separate STATIC lib so the CUTLASS headers (slow to compile) and the sm_120a
# arch requirement stay isolated from the main translation units. Auto-OFF when
# CUTLASS headers are unavailable, keeping CUTLASS-less builds working.
option(H3_ENABLE_NVFP4_CHAIN
  "Build the NVFP4 FC1->FC2 chained direct-output kernel (needs CUTLASS + sm_120a)" ON)
set(H3_CUTLASS_DIR "" CACHE PATH
  "Path to a CUTLASS checkout (>= v4.x, needs Sm1xxBlockScaledOutputConfig)")
if(H3_ENABLE_NVFP4_CHAIN AND NOT H3_CUTLASS_DIR AND DEFINED ENV{CUTLASS_DIR})
  set(H3_CUTLASS_DIR "$ENV{CUTLASS_DIR}")
endif()
if(H3_ENABLE_NVFP4_CHAIN AND NOT H3_CUTLASS_DIR)
  foreach(_cand
      "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass"
      "${CMAKE_CURRENT_SOURCE_DIR}/deps/cutlass")
    if(EXISTS "${_cand}/include/cutlass/cutlass.h")
      set(H3_CUTLASS_DIR "${_cand}")
      break()
    endif()
  endforeach()
endif()
set(_h3_chain_ok OFF)
if(H3_ENABLE_NVFP4_CHAIN AND H3_CUTLASS_DIR AND
   EXISTS "${H3_CUTLASS_DIR}/include/cutlass/cutlass.h")
  set(_h3_chain_ok ON)
endif()
if(_h3_chain_ok)
  message(STATUS "NVFP4 chain (P4.2): CUTLASS at ${H3_CUTLASS_DIR}")
  add_library(h3_dit_nvfp4_chain STATIC h3_dit_nvfp4_chain.cu)
  target_include_directories(h3_dit_nvfp4_chain PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${H3_CUTLASS_DIR}/include
    ${H3_CUTLASS_DIR}/tools/util/include)
  target_compile_definitions(h3_dit_nvfp4_chain PUBLIC H3_ENABLE_NVFP4_CHAIN=1)
  target_compile_options(h3_dit_nvfp4_chain PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
  # sm_120a exposes CUTLASS_ARCH_MMA_SM120_SUPPORTED / UMMA (79b needs it).
  set_target_properties(h3_dit_nvfp4_chain PROPERTIES
    CUDA_ARCHITECTURES "120a"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(h3_dit_nvfp4_chain PUBLIC CUDA::cudart)
  target_link_libraries(h3_gpu_cuda PUBLIC h3_dit_nvfp4_chain)
  # Propagate the macro so h3_dit_w4a8.c can gate the chain dispatch.
  target_compile_definitions(h3_gpu_cuda PUBLIC H3_ENABLE_NVFP4_CHAIN=1)
else()
  message(STATUS "NVFP4 chain (P4.2): OFF (CUTLASS not found; set H3_CUTLASS_DIR)")
  # Stub TU provides h3_cutlass_nvfp4_fc1_f32_run / h3_nvfp4_swiglu_requant_dev
  # so h3_dit_w4a8.c still links when H3_ENABLE_NVFP4_CHAIN is unset (Ada/4090).
  add_library(h3_dit_nvfp4_chain_stub STATIC h3_dit_nvfp4_chain.cu)
  target_include_directories(h3_dit_nvfp4_chain_stub PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(h3_dit_nvfp4_chain_stub PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(h3_dit_nvfp4_chain_stub PUBLIC CUDA::cudart)
  target_link_libraries(h3_gpu_cuda PUBLIC h3_dit_nvfp4_chain_stub)
endif()

if(H3_ENABLE_CUDNN)
  # Prefer conda cuDNN 9 over system cuDNN 8.
  find_path(H3_CUDNN_INCLUDE cudnn.h
    PATHS
      ENV CUDNN_ROOT
      "$ENV{CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cudnn/include"
      "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cudnn/include"
      "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/include"
      "$ENV{CONDA_PREFIX}/include"
      "${CUDAToolkit_INCLUDE_DIRS}"
      /usr/include
    NO_DEFAULT_PATH)
  if(NOT H3_CUDNN_INCLUDE)
    find_path(H3_CUDNN_INCLUDE cudnn.h)
  endif()
  # pip nvidia-cudnn wheels ship libcudnn.so.9 without unversioned .so
  find_library(H3_CUDNN_LIB NAMES cudnn cudnn.so.9 libcudnn.so.9
    PATHS
      ENV CUDNN_ROOT
      "$ENV{CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cudnn/lib"
      "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cudnn/lib"
      "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/lib"
      "$ENV{CONDA_PREFIX}/lib"
      "${CUDAToolkit_LIBRARY_DIR}"
      /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH)
  if(NOT H3_CUDNN_LIB)
    find_library(H3_CUDNN_LIB NAMES cudnn cudnn.so.9)
  endif()
  # If include is cuDNN9 but lib resolved to system v8, force conda .so.9
  if(H3_CUDNN_INCLUDE AND H3_CUDNN_LIB)
    file(READ "${H3_CUDNN_INCLUDE}/cudnn_version.h" _h3_cudnn_ver_h)
    string(REGEX MATCH "#define CUDNN_MAJOR[ \t]+([0-9]+)" _h3_cudnn_maj
      "${_h3_cudnn_ver_h}")
    set(_h3_cudnn_major "${CMAKE_MATCH_1}")
    if(_h3_cudnn_major VERSION_GREATER_EQUAL 9 AND
       H3_CUDNN_LIB MATCHES "/usr/lib")
      set(_h3_sys_cudnn_lib "${H3_CUDNN_LIB}")
      unset(H3_CUDNN_LIB CACHE)
      find_library(H3_CUDNN_LIB NAMES cudnn.so.9 libcudnn.so.9
        PATHS
          "$ENV{CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cudnn/lib"
          "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cudnn/lib"
          "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/lib"
        NO_DEFAULT_PATH)
      if(NOT H3_CUDNN_LIB)
        set(H3_CUDNN_LIB "${_h3_sys_cudnn_lib}")
      endif()
    endif()
  endif()
  if(H3_CUDNN_INCLUDE AND H3_CUDNN_LIB)
    message(STATUS "cuDNN: include=${H3_CUDNN_INCLUDE} lib=${H3_CUDNN_LIB}")
    target_include_directories(h3_gpu_cuda PRIVATE ${H3_CUDNN_INCLUDE})
    target_compile_definitions(h3_gpu_cuda PRIVATE H3_HAVE_CUDNN=1)
    target_link_libraries(h3_gpu_cuda PUBLIC ${H3_CUDNN_LIB} ${CMAKE_DL_LIBS})
    get_filename_component(H3_CUDNN_LIBDIR "${H3_CUDNN_LIB}" DIRECTORY)
    set_property(TARGET h3_gpu_cuda APPEND PROPERTY BUILD_RPATH "${H3_CUDNN_LIBDIR}")
    set_property(TARGET h3_gpu_cuda APPEND PROPERTY INSTALL_RPATH "${H3_CUDNN_LIBDIR}")

    # generate preloads libnvrtc.so.13 on CUDA 13. cuDNN 9.10.x (built for
    # CUDA 12.9) still dlopens libnvrtc.so.12 to JIT sm_120 SDPA; missing
    # so.12 yields KERNEL_CFG=-1 / "No valid engine". Gold cafe rpath'd
    # pip nvidia/cuda_nvrtc. Always keep so.12 on the loader path.
    set(_h3_nvrtc12_paths
      "$ENV{CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
      "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"
      "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
      "$ENV{HOME}/opt/py310/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
      "$ENV{CONDA_PREFIX}/lib"
      "$ENV{CONDA_PREFIX}/targets/x86_64-linux/lib"
      "${CUDAToolkit_LIBRARY_DIR}"
      /usr/local/cuda/lib64)
    find_library(H3_NVRTC12_LIB NAMES libnvrtc.so.12 nvrtc.so.12
      PATHS ${_h3_nvrtc12_paths} NO_DEFAULT_PATH)
    if(NOT H3_NVRTC12_LIB)
      find_library(H3_NVRTC12_LIB NAMES nvrtc.so.12 libnvrtc.so.12)
    endif()
    if(H3_NVRTC12_LIB)
      get_filename_component(H3_NVRTC12_LIBDIR "${H3_NVRTC12_LIB}" DIRECTORY)
      message(STATUS "NVRTC12 (cuDNN JIT): ${H3_NVRTC12_LIB}")
      set_property(TARGET h3_gpu_cuda APPEND PROPERTY BUILD_RPATH "${H3_NVRTC12_LIBDIR}")
      set_property(TARGET h3_gpu_cuda APPEND PROPERTY INSTALL_RPATH "${H3_NVRTC12_LIBDIR}")
      target_link_options(h3_gpu_cuda PUBLIC "-Wl,-rpath,${H3_NVRTC12_LIBDIR}")
    else()
      message(WARNING "H3: libnvrtc.so.12 not found; cuDNN SDPA JIT may have no engine")
    endif()
    if(_h3_nvcc_ver VERSION_GREATER_EQUAL "13.0")
      find_library(H3_NVRTC_LIB NAMES libnvrtc.so.13 nvrtc.so.13 nvrtc
        PATHS
          "$ENV{CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13/lib"
          "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cu13/lib"
          "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cu13/lib"
          "$ENV{HOME}/opt/py310/lib/python3.10/site-packages/nvidia/cu13/lib"
          "$ENV{CONDA_PREFIX}/lib"
          "$ENV{CONDA_PREFIX}/targets/x86_64-linux/lib"
          "${CUDAToolkit_LIBRARY_DIR}"
        NO_DEFAULT_PATH)
      if(NOT H3_NVRTC_LIB)
        find_library(H3_NVRTC_LIB NAMES nvrtc nvrtc.so.13 libnvrtc.so.13)
      endif()
    else()
      set(H3_NVRTC_LIB "${H3_NVRTC12_LIB}")
    endif()
    if(H3_NVRTC_LIB)
      get_filename_component(H3_NVRTC_LIBDIR "${H3_NVRTC_LIB}" DIRECTORY)
      message(STATUS "NVRTC: ${H3_NVRTC_LIB}")
      set_property(TARGET h3_gpu_cuda APPEND PROPERTY BUILD_RPATH "${H3_NVRTC_LIBDIR}")
      set_property(TARGET h3_gpu_cuda APPEND PROPERTY INSTALL_RPATH "${H3_NVRTC_LIBDIR}")
      target_link_options(h3_gpu_cuda PUBLIC "-Wl,-rpath,${H3_NVRTC_LIBDIR}")
    else()
      message(STATUS "NVRTC not found at configure time; runtime will search CONDA_PREFIX")
    endif()

    if(H3_ENABLE_CUDNN_FRONTEND AND EXISTS "${H3_CUDNN_FRONTEND_DIR}/cudnn_frontend.h")
      message(STATUS "cuDNN frontend: ${H3_CUDNN_FRONTEND_DIR}")
      target_sources(h3_gpu_cuda PRIVATE h3_gpu_cuda_sdpa_cudnn.cpp)
      target_include_directories(h3_gpu_cuda PRIVATE ${H3_CUDNN_FRONTEND_DIR})
      target_compile_definitions(h3_gpu_cuda PRIVATE H3_HAVE_CUDNN_FRONTEND=1)
      target_compile_features(h3_gpu_cuda PRIVATE cxx_std_17)
      # Frontend needs the full cuDNN shared libs on the loader path.
      target_link_options(h3_gpu_cuda PUBLIC "-Wl,-rpath,${H3_CUDNN_LIBDIR}")
    else()
      message(STATUS "cuDNN frontend headers missing; fused SDPA disabled")
    endif()
  else()
    message(STATUS "cuDNN not found; building without H3_HAVE_CUDNN")
  endif()
endif()

add_executable(h3_cuda_info tools/h3_cuda_info.c)
target_link_libraries(h3_cuda_info PRIVATE h3_gpu_cuda)

add_executable(h3_sol_attn_bench
  tools/h3_sol_attn_bench.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_sol_attn_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_sol_attn_bench PRIVATE c_std_11)
target_link_libraries(h3_sol_attn_bench PRIVATE h3_gpu_cuda m)

# P3 host-only safetensors tools (no CUDA runtime needed at link time).
add_executable(h3_w4a8_probe
  tools/h3_w4a8_probe.c
  h3_safetensors.c
  h3_w4a8.c)
target_include_directories(h3_w4a8_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_w4a8_probe PRIVATE c_std_11)

add_executable(h3_dit_int8_probe
  tools/h3_dit_int8_probe.c
  h3_safetensors.c
  h3_dit_int8.c)
target_include_directories(h3_dit_int8_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_int8_probe PRIVATE c_std_11)

add_executable(h3_dit_nvfp4_probe
  tools/h3_dit_nvfp4_probe.c
  h3_safetensors.c
  h3_dit_nvfp4.c
  h3_nvfp4.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_dit_nvfp4_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_nvfp4_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_nvfp4_probe PRIVATE h3_gpu_cuda m)

add_executable(h3_nvfp4_int8_sidecar
  tools/h3_nvfp4_int8_sidecar.c
  h3_safetensors.c
  h3_dit_nvfp4.c
  h3_nvfp4.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_nvfp4_int8_sidecar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_nvfp4_int8_sidecar PRIVATE c_std_11)
target_link_libraries(h3_nvfp4_int8_sidecar PRIVATE h3_gpu_cuda m)

add_executable(h3_st_inventory
  tools/h3_st_inventory.c
  h3_safetensors.c
  h3_w4a8.c)
target_include_directories(h3_st_inventory PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_st_inventory PRIVATE c_std_11)

# P4 step-1: CPU W4A8 dequant + FP32 GEMM probe (no CUDA runtime).
add_executable(h3_w4a8_matmul_probe
  tools/h3_w4a8_matmul_probe.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_w4a8_matmul_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_w4a8_matmul_probe PRIVATE c_std_11)
target_link_libraries(h3_w4a8_matmul_probe PRIVATE m)

# P4 step-2: act ConvRot equivalence + cuBLAS FP32 GEMM on real W4A8 linear.
add_executable(h3_w4a8_cuda_probe
  tools/h3_w4a8_cuda_probe.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_w4a8_cuda_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_w4a8_cuda_probe PRIVATE c_std_11)
target_link_libraries(h3_w4a8_cuda_probe PRIVATE h3_gpu_cuda m)

# P4 step-3: multi-block smoke + CPU INT8 act/GEMM vs FP32 (no Tensor Core yet).
add_executable(h3_w4a8_blocks_probe
  tools/h3_w4a8_blocks_probe.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_w4a8_blocks_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_w4a8_blocks_probe PRIVATE c_std_11)
target_link_libraries(h3_w4a8_blocks_probe PRIVATE m)

# P4 step-4: CUDA INT8 GEMM (cuBLASLt / GemmEx / kernel) vs CPU INT8 reference.
add_executable(h3_w4a8_int8_cuda_probe
  tools/h3_w4a8_int8_cuda_probe.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_w4a8_int8_cuda_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_w4a8_int8_cuda_probe PRIVATE c_std_11)
target_link_libraries(h3_w4a8_int8_cuda_probe PRIVATE h3_gpu_cuda m)

# P4 step-5: device W4A8 unpack → resident INT8 + GEMM vs host.
add_executable(h3_w4a8_device_probe
  tools/h3_w4a8_device_probe.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_w4a8_device_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_w4a8_device_probe PRIVATE c_std_11)
target_link_libraries(h3_w4a8_device_probe PRIVATE h3_gpu_cuda m)

# P4 step-6: 50-block INT8 residency / window + DiT-facing cache API.
add_executable(h3_w4a8_cache_probe
  tools/h3_w4a8_cache_probe.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c
  h3_dit_nvfp4.c
  h3_nvfp4.c)
target_include_directories(h3_w4a8_cache_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_w4a8_cache_probe PRIVATE c_std_11)
target_link_libraries(h3_w4a8_cache_probe PRIVATE h3_gpu_cuda m)

# P4 step-7/8: DiT W4A8 linear wiring (bypass) + block-0 probe.
add_executable(h3_dit_w4a8_block_probe
  tools/h3_dit_w4a8_block_probe.c
  h3_dit_w4a8.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c
  h3_dit_nvfp4.c
  h3_nvfp4.c)
target_include_directories(h3_dit_w4a8_block_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_block_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_block_probe PRIVATE h3_gpu_cuda m pthread)

# P4 M1: multi-block chain.
add_executable(h3_dit_w4a8_layers_probe
  tools/h3_dit_w4a8_layers_probe.c
  h3_dit_w4a8.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c
  h3_dit_nvfp4.c
  h3_nvfp4.c)
target_include_directories(h3_dit_w4a8_layers_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_layers_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_layers_probe PRIVATE h3_gpu_cuda m pthread)

# P4 M2: patch → layers → final (video).
add_executable(h3_dit_w4a8_forward_probe
  tools/h3_dit_w4a8_forward_probe.c
  h3_dit_w4a8.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c
  h3_dit_nvfp4.c
  h3_nvfp4.c)
target_include_directories(h3_dit_w4a8_forward_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_forward_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_forward_probe PRIVATE h3_gpu_cuda m pthread)

# P1 FP MAE gates: merged QKV GEMM vs split3, fused QKNorm+RoPE vs unfused.
add_executable(h3_dit_w4a8_p1_fp_probe
  tools/h3_dit_w4a8_p1_fp_probe.c
  h3_dit_w4a8.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c
  h3_dit_nvfp4.c
  h3_nvfp4.c)
target_include_directories(h3_dit_w4a8_p1_fp_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_p1_fp_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_p1_fp_probe PRIVATE h3_gpu_cuda m pthread)

# P4 M3 / E2: Euler sample loop (+ optional text context).
add_executable(h3_dit_w4a8_sample_probe
  tools/h3_dit_w4a8_sample_probe.c
  h3_dit_w4a8.c
  h3_nvfp4.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c
  h3_dit_nvfp4.c)
target_include_directories(h3_dit_w4a8_sample_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_sample_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_sample_probe PRIVATE h3_gpu_cuda m pthread)

# P4 M4: Video VAE INT8+scale dequant smoke (host-only).
add_executable(h3_vae_int8_probe
  tools/h3_vae_int8_probe.c
  h3_vae_int8.c
  h3_safetensors.c
  h3_w4a8_dequant.c
  h3_w4a8.c)
target_include_directories(h3_vae_int8_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_vae_int8_probe PRIVATE c_std_11)
target_link_libraries(h3_vae_int8_probe PRIVATE m)

# P4 M4 / E1: TE NVFP4 header + dequant + encode (CUDA GEMM when available).
add_executable(h3_nvfp4_probe
  tools/h3_nvfp4_probe.c
  h3_nvfp4.c
  h3_token_ids.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_nvfp4_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_nvfp4_probe PRIVATE c_std_11)
target_link_libraries(h3_nvfp4_probe PRIVATE h3_gpu_cuda m)

# E4: end-to-end generate smoke (DiT + optional TE/VAE).
add_executable(h3_w4a8_generate_probe
  tools/h3_w4a8_generate_probe.c
  h3_dit_w4a8.c
  h3_nvfp4.c
  h3_vae_int8.c
  h3_vae_int8_cuda.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c
  h3_dit_nvfp4.c)
target_include_directories(h3_w4a8_generate_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_w4a8_generate_probe PRIVATE c_std_11)
target_link_libraries(h3_w4a8_generate_probe PRIVATE h3_gpu_cuda m pthread)

# G4: formal CUDA bypass T2V generate CLI (not Metal h3_generate).
add_executable(h3_cuda_generate
  tools/h3_cuda_generate.c
  h3_dit_w4a8.c
  h3_dit_int8.c
  h3_dit_nvfp4.c
  h3_nvfp4.c
  h3_token_ids.c
  h3_ffmpeg.c
  h3_vae_int8.c
  h3_vae_int8_cuda.c
  h3_vae_encode_f32.c
  h3_vision_cuda.c
  h3_multimodal_cuda.c
  h3_audio_vae_cuda.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_step_cache.c
  h3_generate_preset.c
  h3_taomate.c
  h3_lora.c
  h3_fewstep_contract.c
  h3_env_contract.c
  h3_sampler.c
  h3_tae.c
  h3_tae_cuda.cu
  h3_te_cache.c
  h3_ulysses.c
  h3_selflift.c
  h3_latent_upscaler.c
  h3_latent_upscaler.cu
  h3_vae_prewarm.c
  h3_av_sync.c
  h3_waffle.c
  h3_run_report.c
  h3_kpi_guard.c)
target_include_directories(h3_cuda_generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_cuda_generate PRIVATE c_std_11)
target_compile_definitions(h3_cuda_generate PRIVATE H3_TAE_HAVE_CUDA=1)
target_link_libraries(h3_cuda_generate PRIVATE h3_gpu_cuda m pthread)

add_executable(h3_lbh_oracle
  tools/h3_lbh_oracle.c
  h3_latent_upscaler.c
  h3_latent_upscaler.cu
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_lbh_oracle PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_lbh_oracle PRIVATE c_std_11)
target_link_libraries(h3_lbh_oracle PRIVATE h3_gpu_cuda m pthread)
# Default OFF: linking libnccl pulls $ORIGIN libstdc++ 6.0.34 into generate
# and drops the gold cuDNN plan. a2a test still links NCCL when present.
# Opt-in H3_LINK_NCCL_IN_GENERATE=ON for U>1 experiments (U=1 cuDNN then
# expected broken). --ulysses default stays 1.
option(H3_LINK_NCCL_IN_GENERATE
  "Link libnccl into h3_cuda_generate (breaks U=1 gold cuDNN on NCCL-bundled libstdc++)"
  OFF)
# Pin generate to the distro libstdc++ unless this build opted into NCCL
# (needs CXXABI_1.3.15). U=1 gold cafe stays on the distro copy.
if(NOT H3_LINK_NCCL_IN_GENERATE AND EXISTS "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
  target_link_libraries(h3_cuda_generate PRIVATE
    "-Wl,--no-as-needed"
    "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    "-Wl,--as-needed")
endif()
find_package(OpenMP)
if(OpenMP_C_FOUND)
  target_link_libraries(h3_cuda_generate PRIVATE OpenMP::OpenMP_C)
endif()
find_path(H3_NCCL_INCLUDE_DIR nccl.h
  HINTS /usr/include /usr/local/include /usr/local/cuda/include
        /usr/include/x86_64-linux-gnu
        "$ENV{NCCL_ROOT}/include" "$ENV{NCCL_HOME}/include")
find_library(H3_NCCL_LIBRARY nccl
  HINTS /usr/lib /usr/local/lib /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
        "$ENV{NCCL_ROOT}/lib" "$ENV{NCCL_HOME}/lib")
if(H3_NCCL_LIBRARY AND H3_NCCL_INCLUDE_DIR AND H3_LINK_NCCL_IN_GENERATE)
  target_sources(h3_cuda_generate PRIVATE h3_ulysses_comm.cu)
  target_include_directories(h3_cuda_generate PRIVATE ${H3_NCCL_INCLUDE_DIR})
  target_link_libraries(h3_cuda_generate PRIVATE ${H3_NCCL_LIBRARY})
  # NCCL wheel/prefix needs CXXABI_1.3.15. Gold stub never takes this branch.
  get_filename_component(_h3_nccl_libdir "${H3_NCCL_LIBRARY}" DIRECTORY)
  if(EXISTS "${_h3_nccl_libdir}/libstdc++.so.6")
    target_link_libraries(h3_cuda_generate PRIVATE
      "-Wl,--no-as-needed"
      "${_h3_nccl_libdir}/libstdc++.so.6"
      "-Wl,--as-needed")
    message(STATUS
      "H3: NCCL generate links ${_h3_nccl_libdir}/libstdc++.so.6 (not gold)")
  endif()
  message(STATUS
    "H3: generate links NCCL ${H3_NCCL_LIBRARY} (U=1 cuDNN may have no engine)")
else()
  target_sources(h3_cuda_generate PRIVATE h3_ulysses_comm_stub.c)
  if(H3_NCCL_LIBRARY AND H3_NCCL_INCLUDE_DIR)
    message(STATUS
      "H3: generate uses NCCL stub (gold U=1 cuDNN); set "
      "H3_LINK_NCCL_IN_GENERATE=ON to link libnccl")
  else()
    message(STATUS "H3: NCCL not found; --ulysses>1 stubbed")
  endif()
endif()
if(H3_NCCL_LIBRARY AND H3_NCCL_INCLUDE_DIR)
  add_executable(h3_ulysses_test_a2a tests/test_h3_ulysses_a2a.cu h3_ulysses.c
                 h3_ulysses_comm.cu)
  target_include_directories(h3_ulysses_test_a2a PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${H3_NCCL_INCLUDE_DIR})
  target_link_libraries(h3_ulysses_test_a2a PRIVATE ${H3_NCCL_LIBRARY}
                        CUDA::cudart)
  set_target_properties(h3_ulysses_test_a2a PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Never put the raw NCCL prefix on generate's RPATH (bundled libstdc++).
# DT_RPATH + system first so LD_LIBRARY_PATH cannot steal libstdc++.
function(h3_pin_loader_rpath tgt)
  if(NOT TARGET "${tgt}")
    return()
  endif()
  # Gold cafe: pip cuda_nvrtc (so.12) + pip cuDNN for SDPA JIT, then so.13 /
  # toolkit for generate preload + cudart, then distro libstdc++. Never NCCL.
  set(_rpath)
  if(H3_NVRTC12_LIBDIR)
    list(APPEND _rpath "${H3_NVRTC12_LIBDIR}")
  endif()
  if(H3_CUDNN_LIBDIR)
    list(APPEND _rpath "${H3_CUDNN_LIBDIR}")
  endif()
  if(H3_NVRTC_LIBDIR)
    list(APPEND _rpath "${H3_NVRTC_LIBDIR}")
  endif()
  if(CUDAToolkit_LIBRARY_DIR)
    list(APPEND _rpath "${CUDAToolkit_LIBRARY_DIR}")
  endif()
  list(APPEND _rpath "/usr/lib/x86_64-linux-gnu" "/lib/x86_64-linux-gnu")
  set_property(TARGET ${tgt} PROPERTY SKIP_BUILD_RPATH TRUE)
  set_property(TARGET ${tgt} PROPERTY BUILD_WITH_INSTALL_RPATH TRUE)
  set_property(TARGET ${tgt} PROPERTY INSTALL_RPATH "${_rpath}")
  set_property(TARGET ${tgt} PROPERTY INSTALL_RPATH_USE_LINK_PATH FALSE)
  set(_link_rpath)
  foreach(_d IN LISTS _rpath)
    list(APPEND _link_rpath "LINKER:-rpath,${_d}")
  endforeach()
  # DT_RPATH (not RUNPATH): engine dlopen("libnvrtc.so.12") sees the exe path.
  target_link_options(${tgt} BEFORE PRIVATE
    "-Wl,--disable-new-dtags" ${_link_rpath})
endfunction()
h3_pin_loader_rpath(h3_cuda_generate)

# Optional Python embeds for pluggable SDPA backends:
#   --sdpa-backend sol_cute  (NVlabs SM89 CuTe)
#   --sdpa-backend sage3     (thu-ml SageAttention3 Blackwell FP4)
find_package(Python3 COMPONENTS Development)
if(Python3_FOUND)
  add_library(h3_sol_cute SHARED tools/h3_sol_cute_embed.c)
  target_include_directories(h3_sol_cute PRIVATE ${Python3_INCLUDE_DIRS})
  target_link_libraries(h3_sol_cute PRIVATE ${Python3_LIBRARIES})
  set_target_properties(h3_sol_cute PROPERTIES
    OUTPUT_NAME h3_sol_cute
    PREFIX "lib")
  message(STATUS "libh3_sol_cute: Python ${Python3_VERSION}")

  add_library(h3_sage3 SHARED tools/h3_sage3_embed.c)
  target_include_directories(h3_sage3 PRIVATE ${Python3_INCLUDE_DIRS})
  target_link_libraries(h3_sage3 PRIVATE ${Python3_LIBRARIES})
  set_target_properties(h3_sage3 PROPERTIES
    OUTPUT_NAME h3_sage3
    PREFIX "lib")
  message(STATUS "libh3_sage3: Python ${Python3_VERSION}")
else()
  message(STATUS "libh3_sol_cute / libh3_sage3 skipped (Python3 Development not found)")
endif()

# Diagnose prompt→condition→DiT velocity distinguishability.
add_executable(h3_condition_probe
  tools/h3_condition_probe.c
  h3_dit_w4a8.c
  h3_nvfp4.c
  h3_token_ids.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c
  h3_dit_nvfp4.c)
target_include_directories(h3_condition_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_condition_probe PRIVATE c_std_11)
target_link_libraries(h3_condition_probe PRIVATE h3_gpu_cuda m pthread)

# T2AV timing: official shape math + phased TE/DiT/VAE wall times on CUDA bypass.
add_executable(h3_t2av_timing_probe
  tools/h3_t2av_timing_probe.c
  h3_dit_w4a8.c
  h3_nvfp4.c
  h3_vae_int8.c
  h3_vae_int8_cuda.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c
  h3_dit_nvfp4.c)
target_include_directories(h3_t2av_timing_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_t2av_timing_probe PRIVATE c_std_11)
target_link_libraries(h3_t2av_timing_probe PRIVATE h3_gpu_cuda m pthread)

# Au1: CUDA audio 1D ops vs CPU reference.
add_executable(h3_audio_ops_probe
  tools/h3_audio_ops_probe.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_audio_ops_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_audio_ops_probe PRIVATE c_std_11)
target_link_libraries(h3_audio_ops_probe PRIVATE h3_gpu_cuda m)

# Au2: CUDA AudioVAE decode (Comfy FP32).
add_executable(h3_audio_vae_cuda_probe
  tools/h3_audio_vae_cuda_probe.c
  h3_audio_vae_cuda.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_audio_vae_cuda_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_audio_vae_cuda_probe PRIVATE c_std_11)
target_link_libraries(h3_audio_vae_cuda_probe PRIVATE h3_gpu_cuda m)

# Au6: CUDA AudioVAE encode.
add_executable(h3_audio_vae_cuda_encode_probe
  tools/h3_audio_vae_cuda_encode_probe.c
  h3_audio_vae_cuda.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_audio_vae_cuda_encode_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_audio_vae_cuda_encode_probe PRIVATE c_std_11)
target_link_libraries(h3_audio_vae_cuda_encode_probe PRIVATE h3_gpu_cuda m)

# G7: minimal numeric quality gate.
add_executable(h3_quality_gate_probe
  tools/h3_quality_gate_probe.c
  h3_audio_vae_cuda.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_quality_gate_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_quality_gate_probe PRIVATE c_std_11)
target_link_libraries(h3_quality_gate_probe PRIVATE h3_gpu_cuda m)

# Au3/Au4: DiT audio patch + AV Euler smoke.
add_executable(h3_dit_w4a8_audio_probe
  tools/h3_dit_w4a8_audio_probe.c
  h3_dit_w4a8.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c
  h3_dit_nvfp4.c
  h3_nvfp4.c)
target_include_directories(h3_dit_w4a8_audio_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_audio_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_audio_probe PRIVATE h3_gpu_cuda m pthread)

add_executable(h3_dit_w4a8_cond_test
  tests/test_h3_dit_w4a8_cond.c
  h3_dit_w4a8.c
  h3_dit_int8.c
  h3_dit_nvfp4.c
  h3_nvfp4.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_lora.c)
target_include_directories(h3_dit_w4a8_cond_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_cond_test PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_cond_test PRIVATE h3_gpu_cuda m pthread)

add_executable(h3_step_cache_test
  tests/test_h3_step_cache.c
  h3_step_cache.c)
target_include_directories(h3_step_cache_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_step_cache_test PRIVATE c_std_11)
target_link_libraries(h3_step_cache_test PRIVATE m)

add_executable(h3_taomate_test
  tests/test_h3_taomate.c
  h3_taomate.c)
target_include_directories(h3_taomate_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_taomate_test PRIVATE c_std_11)
target_link_libraries(h3_taomate_test PRIVATE m)

add_executable(h3_generate_preset_test
  tests/test_h3_generate_preset.c
  h3_generate_preset.c)
target_include_directories(h3_generate_preset_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_generate_preset_test PRIVATE c_std_11)

add_executable(h3_selflift_test
  tests/test_h3_selflift.c
  h3_selflift.c)
target_include_directories(h3_selflift_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_selflift_test PRIVATE c_std_11)
target_link_libraries(h3_selflift_test PRIVATE m)

add_executable(h3_lora_test
  tests/test_h3_lora.c
  h3_lora.c
  h3_safetensors.c)
target_include_directories(h3_lora_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_lora_test PRIVATE c_std_11)
target_link_libraries(h3_lora_test PRIVATE m)

add_executable(h3_fewstep_contract_test
  tests/test_h3_fewstep_contract.c
  h3_fewstep_contract.c)
target_include_directories(h3_fewstep_contract_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_fewstep_contract_test PRIVATE c_std_11)
target_link_libraries(h3_fewstep_contract_test PRIVATE m)

add_executable(h3_sampler_test
  tests/test_h3_sampler.c
  h3_sampler.c)
target_include_directories(h3_sampler_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_sampler_test PRIVATE c_std_11)
target_link_libraries(h3_sampler_test PRIVATE m)

add_executable(h3_ulysses_test
  tests/test_h3_ulysses.c
  h3_ulysses.c)
target_include_directories(h3_ulysses_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_ulysses_test PRIVATE c_std_11)

add_executable(h3_te_cache_test
  tests/test_h3_te_cache.c
  h3_te_cache.c)
target_include_directories(h3_te_cache_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_te_cache_test PRIVATE c_std_11)

add_executable(h3_tae_test
  tests/test_h3_tae.c
  h3_tae.c
  h3_safetensors.c)
target_include_directories(h3_tae_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_tae_test PRIVATE c_std_11)
target_link_libraries(h3_tae_test PRIVATE m)

add_executable(h3_vae_rope_test
  tests/test_h3_vae_rope.c
  h3_vae_int8.c
  h3_safetensors.c
  h3_w4a8_dequant.c
  h3_w4a8.c)
target_include_directories(h3_vae_rope_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_vae_rope_test PRIVATE c_std_11)
target_link_libraries(h3_vae_rope_test PRIVATE m)

add_executable(h3_vae_resident_test
  tests/test_h3_vae_resident.c
  h3_vae_int8.c
  h3_safetensors.c
  h3_w4a8_dequant.c
  h3_w4a8.c)
target_include_directories(h3_vae_resident_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_vae_resident_test PRIVATE c_std_11)
target_link_libraries(h3_vae_resident_test PRIVATE m)

add_executable(h3_vae_layout_test
  tests/test_h3_vae_layout.c
  h3_vae_int8.c
  h3_safetensors.c
  h3_w4a8_dequant.c
  h3_w4a8.c)
target_include_directories(h3_vae_layout_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_vae_layout_test PRIVATE c_std_11)
target_link_libraries(h3_vae_layout_test PRIVATE m)

add_executable(h3_sage3_require_test
  tests/test_h3_sage3_require.c)
target_include_directories(h3_sage3_require_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_sage3_require_test PRIVATE c_std_11)

add_executable(h3_sol_cute_backend_test
  tests/test_h3_sol_cute_backend.c)
target_include_directories(h3_sol_cute_backend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_sol_cute_backend_test PRIVATE c_std_11)

add_executable(h3_kpi_guard_test
  tests/test_h3_kpi_guard.c
  h3_kpi_guard.c)
target_include_directories(h3_kpi_guard_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_kpi_guard_test PRIVATE c_std_11)

add_executable(h3_dit_act_f16_test
  tests/test_h3_dit_act_f16.c)
target_include_directories(h3_dit_act_f16_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_act_f16_test PRIVATE c_std_11)
target_link_libraries(h3_dit_act_f16_test PRIVATE m)

add_executable(h3_dit_fused_epi_test
  tests/test_h3_dit_fused_epi.c)
target_include_directories(h3_dit_fused_epi_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_fused_epi_test PRIVATE c_std_11)
target_link_libraries(h3_dit_fused_epi_test PRIVATE m)

add_executable(h3_epi_fuse_sl_host_test
  tests/test_epi_fuse_sl_host.c)
target_include_directories(h3_epi_fuse_sl_host_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_epi_fuse_sl_host_test PRIVATE c_std_11)
target_link_libraries(h3_epi_fuse_sl_host_test PRIVATE m)

add_executable(h3_epi_fuse_ink_host_test
  tests/test_epi_fuse_ink_host.c)
target_include_directories(h3_epi_fuse_ink_host_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_epi_fuse_ink_host_test PRIVATE c_std_11)
target_link_libraries(h3_epi_fuse_ink_host_test PRIVATE m)

add_executable(h3_nvfp4_act_convrot_test
  tests/test_h3_nvfp4_act_convrot.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_safetensors.c)
target_include_directories(h3_nvfp4_act_convrot_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_nvfp4_act_convrot_test PRIVATE c_std_11)
target_link_libraries(h3_nvfp4_act_convrot_test PRIVATE m)

add_executable(h3_nvfp4_sf_bytes_test
  tests/test_h3_nvfp4_sf_bytes.c)
target_include_directories(h3_nvfp4_sf_bytes_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_nvfp4_sf_bytes_test PRIVATE c_std_11)
target_compile_definitions(h3_nvfp4_sf_bytes_test PRIVATE TEST_H3_NVFP4_SF_BYTES_STANDALONE)

add_executable(h3_w4a8_window_test
  tests/test_h3_w4a8_window.c)
target_include_directories(h3_w4a8_window_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_w4a8_window_test PRIVATE c_std_11)

add_executable(h3_vae_prewarm_test
  tests/test_h3_vae_prewarm.c)
target_include_directories(h3_vae_prewarm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_vae_prewarm_test PRIVATE c_std_11)

add_executable(h3_av_sync_test
  tests/test_h3_av_sync.c
  h3_av_sync.c
  h3_ffmpeg.c)
target_include_directories(h3_av_sync_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_av_sync_test PRIVATE c_std_11)
target_link_libraries(h3_av_sync_test PRIVATE m pthread)

add_executable(h3_waffle_test
  tests/test_h3_waffle.c
  h3_waffle.c)
target_include_directories(h3_waffle_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_waffle_test PRIVATE c_std_11)

add_executable(h3_ffmpeg_args_test
  tests/test_h3_ffmpeg_args.c
  h3_ffmpeg.c)
target_include_directories(h3_ffmpeg_args_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_ffmpeg_args_test PRIVATE c_std_11)
target_link_libraries(h3_ffmpeg_args_test PRIVATE pthread)
add_executable(h3_ffmpeg_stream_test
  tests/test_h3_ffmpeg_stream.c
  h3_ffmpeg.c)
target_include_directories(h3_ffmpeg_stream_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_ffmpeg_stream_test PRIVATE c_std_11)
target_link_libraries(h3_ffmpeg_stream_test PRIVATE pthread)

add_executable(h3_run_report_test
  tests/test_h3_run_report.c
  h3_run_report.c)
target_include_directories(h3_run_report_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_run_report_test PRIVATE c_std_11)

add_executable(h3_waffle_probe
  tools/h3_waffle_probe.c
  h3_waffle.c)
target_include_directories(h3_waffle_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_waffle_probe PRIVATE c_std_11)

add_executable(h3_vae_encode_f32_test
  tests/test_h3_vae_encode_f32.c
  h3_vae_encode_f32.c
  h3_safetensors.c)
target_include_directories(h3_vae_encode_f32_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_vae_encode_f32_test PRIVATE c_std_11)
target_link_libraries(h3_vae_encode_f32_test PRIVATE h3_gpu_cuda m)
find_package(OpenMP)
if(OpenMP_C_FOUND)
  target_link_libraries(h3_vae_encode_f32_test PRIVATE OpenMP::OpenMP_C)
endif()

add_executable(h3_multimodal_cuda_test
  tests/test_h3_multimodal_cuda.c
  h3_multimodal_cuda.c
  h3_vision_cuda.c
  h3_nvfp4.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c)
target_include_directories(h3_multimodal_cuda_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_multimodal_cuda_test PRIVATE c_std_11)
target_link_libraries(h3_multimodal_cuda_test PRIVATE h3_gpu_cuda m)
if(OpenMP_C_FOUND)
  target_link_libraries(h3_multimodal_cuda_test PRIVATE OpenMP::OpenMP_C)
endif()

install(TARGETS h3_cuda_info h3_sol_attn_bench h3_w4a8_probe h3_dit_int8_probe h3_dit_nvfp4_probe h3_st_inventory h3_w4a8_matmul_probe
                h3_w4a8_cuda_probe h3_w4a8_blocks_probe h3_w4a8_int8_cuda_probe
                h3_w4a8_device_probe h3_w4a8_cache_probe
                h3_dit_w4a8_block_probe h3_dit_w4a8_layers_probe
                h3_dit_w4a8_forward_probe h3_dit_w4a8_sample_probe
                h3_t2av_timing_probe
                h3_vae_int8_probe h3_nvfp4_probe h3_w4a8_generate_probe
                h3_cuda_generate
                h3_audio_ops_probe h3_audio_vae_cuda_probe
                h3_audio_vae_cuda_encode_probe h3_quality_gate_probe
                h3_dit_w4a8_audio_probe
                h3_dit_w4a8_cond_test
                h3_vae_encode_f32_test
                h3_vae_rope_test
                h3_vae_resident_test
                h3_vae_layout_test
                h3_sage3_require_test
                h3_sol_cute_backend_test
                h3_kpi_guard_test
                h3_w4a8_window_test
                h3_dit_act_f16_test
                h3_dit_fused_epi_test
                h3_vae_prewarm_test
                h3_av_sync_test
                h3_waffle_test
                h3_ffmpeg_args_test
                h3_ffmpeg_stream_test
                h3_run_report_test
                h3_waffle_probe
                h3_multimodal_cuda_test
                h3_step_cache_test
                h3_generate_preset_test
                h3_taomate_test
                h3_selflift_test
                h3_lora_test
                h3_fewstep_contract_test
                h3_sampler_test
                h3_ulysses_test
                h3_te_cache_test
                h3_tae_test
  RUNTIME DESTINATION bin)
