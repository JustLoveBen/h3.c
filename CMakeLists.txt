cmake_minimum_required(VERSION 3.18)

# Apple Silicon / Metal remains the primary path via the top-level Makefile.
# This CMake file is the Linux/CUDA stub entry (P1). It does not replace
# h3_gpu.m or build the full h3 CLI.

project(h3_cuda_stub LANGUAGES C CXX)

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

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR
    "H3_GPU_BACKEND=cuda but no CUDA compiler (nvcc) was found.\n"
    "Install CUDA Toolkit (12.8+ recommended for sm_120) and ensure nvcc is on PATH.\n"
    "See docs/nvidia_build.md.")
endif()

# Must set architectures BEFORE enable_language(CUDA); otherwise CMake may
# default to a legacy arch (e.g. 52) for the first CUDA targets.
# Ada (4090) + Blackwell (5090-class). Older toolkits cannot target 120.
# CMAKE_CUDA_COMPILER_VERSION is often unset until after enable_language, so
# parse `nvcc --version` here.
set(_h3_cuda_archs "89")
set(_h3_nvcc_ver "")
execute_process(
  COMMAND "${CMAKE_CUDA_COMPILER}" --version
  OUTPUT_VARIABLE _h3_nvcc_out
  ERROR_VARIABLE _h3_nvcc_err
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "release ([0-9]+\\.[0-9]+)" _h3_nvcc_match "${_h3_nvcc_out}")
if(CMAKE_MATCH_1)
  set(_h3_nvcc_ver "${CMAKE_MATCH_1}")
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
find_package(CUDAToolkit REQUIRED)

# Conda CUDA toolkits inject x86_64-conda-linux-gnu/sysroot -L paths whose
# libpthread.so is a linker script with absolute /lib64/... (RHEL layout).
# On Ubuntu that breaks the final link; keep only real system lib dirs.
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES
  "/usr/lib/x86_64-linux-gnu;/lib/x86_64-linux-gnu;/usr/lib;/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES
  "/usr/lib/x86_64-linux-gnu;/lib/x86_64-linux-gnu;/usr/lib;/lib")

message(STATUS "H3 CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

option(H3_ENABLE_CUDNN "Link cuDNN and enable H3_HAVE_CUDNN SDPA hooks" ON)
option(H3_ENABLE_CUDNN_FRONTEND
  "Build fused SDPA via vendored cudnn_frontend (needs cuDNN 9)" ON)
set(H3_CUDNN_FRONTEND_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn_frontend/include"
  CACHE PATH "Path to cudnn_frontend headers (cudnn_frontend.h)")

add_library(h3_gpu_cuda STATIC h3_gpu_cuda.cu h3_gpu_cuda_audio.cu h3_nvfp4_cuda.cu)
target_include_directories(h3_gpu_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(h3_gpu_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt ${CMAKE_DL_LIBS})
# Ensure -lcudadevrt/-lcudart_static resolve under conda CUDA (no /usr/local/cuda).
if(DEFINED CUDAToolkit_LIBRARY_DIR)
  target_link_directories(h3_gpu_cuda PUBLIC ${CUDAToolkit_LIBRARY_DIR})
endif()
target_compile_options(h3_gpu_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
set_target_properties(h3_gpu_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  POSITION_INDEPENDENT_CODE ON)

if(H3_ENABLE_CUDNN)
  # Prefer conda cuDNN 9 over system cuDNN 8.
  find_path(H3_CUDNN_INCLUDE cudnn.h
    PATHS
      ENV CUDNN_ROOT
      "$ENV{CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cudnn/include"
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
      unset(H3_CUDNN_LIB CACHE)
      find_library(H3_CUDNN_LIB NAMES cudnn.so.9 libcudnn.so.9
        PATHS
          "$ENV{CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cudnn/lib"
          "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn/lib"
        NO_DEFAULT_PATH)
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

    # cuDNN frontend JIT dlopens libnvrtc.so.12. Bake the conda/pip path into
    # rpath so generate does not depend on a forgotten LD_LIBRARY_PATH.
    find_library(H3_NVRTC_LIB NAMES libnvrtc.so.12 nvrtc.so.12 nvrtc
      PATHS
        "$ENV{CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
        "$ENV{CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"
        "$ENV{CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
        "$ENV{CONDA_PREFIX}/lib"
        "$ENV{CONDA_PREFIX}/targets/x86_64-linux/lib"
        "${CUDAToolkit_LIBRARY_DIR}"
        /usr/local/cuda/lib64
      NO_DEFAULT_PATH)
    if(NOT H3_NVRTC_LIB)
      find_library(H3_NVRTC_LIB NAMES nvrtc nvrtc.so.12 libnvrtc.so.12)
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
  h3_dit_int8.c)
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
  h3_dit_int8.c)
target_include_directories(h3_dit_w4a8_block_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_block_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_block_probe PRIVATE h3_gpu_cuda m)

# P4 M1: multi-block chain.
add_executable(h3_dit_w4a8_layers_probe
  tools/h3_dit_w4a8_layers_probe.c
  h3_dit_w4a8.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c)
target_include_directories(h3_dit_w4a8_layers_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_layers_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_layers_probe PRIVATE h3_gpu_cuda m)

# P4 M2: patch → layers → final (video).
add_executable(h3_dit_w4a8_forward_probe
  tools/h3_dit_w4a8_forward_probe.c
  h3_dit_w4a8.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c)
target_include_directories(h3_dit_w4a8_forward_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_forward_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_forward_probe PRIVATE h3_gpu_cuda m)

# P4 M3 / E2: Euler sample loop (+ optional text context).
add_executable(h3_dit_w4a8_sample_probe
  tools/h3_dit_w4a8_sample_probe.c
  h3_dit_w4a8.c
  h3_nvfp4.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c
  h3_dit_int8.c)
target_include_directories(h3_dit_w4a8_sample_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_sample_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_sample_probe PRIVATE h3_gpu_cuda m)

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
  h3_dit_int8.c)
target_include_directories(h3_w4a8_generate_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_w4a8_generate_probe PRIVATE c_std_11)
target_link_libraries(h3_w4a8_generate_probe PRIVATE h3_gpu_cuda m)

# G4: formal CUDA bypass T2V generate CLI (not Metal h3_generate).
add_executable(h3_cuda_generate
  tools/h3_cuda_generate.c
  h3_dit_w4a8.c
  h3_dit_int8.c
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
  h3_w4a8_cache.c)
target_include_directories(h3_cuda_generate PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_cuda_generate PRIVATE c_std_11)
target_link_libraries(h3_cuda_generate PRIVATE h3_gpu_cuda m pthread)
find_package(OpenMP)
if(OpenMP_C_FOUND)
  target_link_libraries(h3_cuda_generate PRIVATE OpenMP::OpenMP_C)
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
  h3_dit_int8.c)
target_include_directories(h3_condition_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_condition_probe PRIVATE c_std_11)
target_link_libraries(h3_condition_probe PRIVATE h3_gpu_cuda m)

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
  h3_dit_int8.c)
target_include_directories(h3_t2av_timing_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_t2av_timing_probe PRIVATE c_std_11)
target_link_libraries(h3_t2av_timing_probe PRIVATE h3_gpu_cuda m)

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
  h3_dit_int8.c)
target_include_directories(h3_dit_w4a8_audio_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_audio_probe PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_audio_probe PRIVATE h3_gpu_cuda m)

add_executable(h3_dit_w4a8_cond_test
  tests/test_h3_dit_w4a8_cond.c
  h3_dit_w4a8.c
  h3_dit_int8.c
  h3_safetensors.c
  h3_w4a8.c
  h3_w4a8_dequant.c
  h3_w4a8_cache.c)
target_include_directories(h3_dit_w4a8_cond_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(h3_dit_w4a8_cond_test PRIVATE c_std_11)
target_link_libraries(h3_dit_w4a8_cond_test PRIVATE h3_gpu_cuda m)

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

install(TARGETS h3_cuda_info h3_sol_attn_bench h3_w4a8_probe h3_dit_int8_probe h3_st_inventory h3_w4a8_matmul_probe
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
                h3_multimodal_cuda_test
  RUNTIME DESTINATION bin)
