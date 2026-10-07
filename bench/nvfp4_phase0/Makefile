# R-D Phase 0: cuBLASLt NVFP4 vs INT8 probe — build for 238 (RTX 5090 sm_120).
#
# Requires CUDA >= 12.8 (cuBLASLt FP4 APIs). On 238 the production toolchain is
# the pip-installed CUDA 13 stack (same as bench/cutlass_evt):
#
#   NVCC ?= /home/yongbin/miniconda3/envs/yongbin/bin/nvcc
#   CU13 ?= /home/yongbin/miniconda3/envs/yongbin/lib/python3.11/site-packages/nvidia/cu13
#
# Override on the command line, e.g.:
#   make NVCC=/usr/local/cuda/bin/nvcc CU13=/usr/local/cuda

NVCC    ?= /home/yongbin/miniconda3/envs/yongbin/bin/nvcc
CU13    ?= /home/yongbin/miniconda3/envs/yongbin/lib/python3.11/site-packages/nvidia/cu13
ARCH    ?= sm_120a
CUDART  ?= /usr/local/cuda

BIN = bench_cublaslt_fp4
TEST = test_fp4_e2e
ROOT = /mnt/nvme1n1/workspace/h3_c
H3LIB = $(ROOT)/build-rel-verify/libh3_gpu_cuda.a

all: $(BIN)

$(BIN): bench_cublaslt_fp4.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) \
	    -I$(CU13)/include -I$(CUDART)/include \
	    $< -L$(CU13)/lib -lcublasLt -lcudart \
	    -Xlinker -rpath -Xlinker $(CU13)/lib \
	    -o $@

# Phase 1 correctness gate (self-contained): reimplements the production
# quantize kernel verbatim + cuBLASLt FP4 GEMM vs CPU reference. No h3 lib.
$(TEST): test_fp4_e2e.cu
	$(NVCC) -O3 -std=c++17 -arch=$(ARCH) \
	    -I$(CU13)/include -I$(CUDART)/include \
	    $< -L$(CU13)/lib -lcublasLt -lcudart \
	    -Xlinker -rpath -Xlinker $(CU13)/lib \
	    -o $@

test: $(TEST)
	./$(TEST)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN) $(TEST) *.o

.PHONY: all run test clean
