// diag_m_limit: find the max M for which cuBLASLt FP4 (E2M1 + VEC16_UE4M3,
// TN, BF16 out) returns a native heuristic on sm_120. Heuristic-only: no real
// data buffers are dereferenced (scale pointers point at a tiny allocation).
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CKL(x) do { cublasStatus_t _s = (x); if (_s != CUBLAS_STATUS_SUCCESS) { \
    printf("  cublas call %s -> st=%d\n", #x, (int)_s); } } while (0)

static int probe(cublasLtHandle_t lt, int m, int n, int k, const void* dsf) {
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    cublasLtMatmulMatrixScale_t vec16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t sc32f = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    cublasLtPointerMode_t pm = CUBLASLT_POINTER_MODE_DEVICE;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &vec16, sizeof(vec16));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &vec16, sizeof(vec16));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, &sc32f, sizeof(sc32f));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &sc32f, sizeof(sc32f));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dsf, sizeof(dsf));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dsf, sizeof(dsf));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pm, sizeof(pm));

    cublasLtMatrixLayout_t la, lb, lc;
    cublasStatus_t sa = cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, k, n, k);
    cublasStatus_t sb = cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, k, m, k);
    cublasStatus_t sc = cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, n, m, n);
    if (sa || sb || sc) {
        printf("m=%6d n=%6d k=%6d  layoutCreate st a=%d b=%d c=%d\n", m, n, k,
               (int)sa, (int)sb, (int)sc);
        cublasLtMatmulDescDestroy(desc);
        return -1;
    }
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = 256ull * 1024 * 1024;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[8];
    int returned = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 8, heur, &returned);
    printf("m=%6d n=%6d k=%6d  heuristic st=%d returned=%d\n", m, n, k, (int)st, returned);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(desc);
    return returned;
}

int main() {
    cublasLtHandle_t lt;
    cublasLtCreate(&lt);
    void* dsf;
    cudaMalloc(&dsf, 1024 * 1024);
    const int fc1_n = 21504, fc1_k = 5376;
    int ms[] = {4096, 8192, 16384, 32768, 36864, 37788, 37888, 40960, 65536};
    printf("== fc1 n=%d k=%d, sweep M ==\n", fc1_n, fc1_k);
    for (size_t i = 0; i < sizeof(ms)/sizeof(ms[0]); ++i)
        probe(lt, ms[i], fc1_n, fc1_k, dsf);
    printf("== alignment microprobe around 37788 (fc1) ==\n");
    int ms2[] = {37760, 37788, 37888, 38016};
    for (size_t i = 0; i < sizeof(ms2)/sizeof(ms2[0]); ++i)
        probe(lt, ms2[i], fc1_n, fc1_k, dsf);
    cudaFree(dsf);
    cublasLtDestroy(lt);
    return 0;
}
