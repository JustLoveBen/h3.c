#ifndef H3_DIT_INT8_OPT_H
#define H3_DIT_INT8_OPT_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LoRA F32 GEMM on a side stream overlapping INT8 ConvRot+GEMM.
 * Opt-in: H3_DIT_LORA_OVERLAP=1. KPI-B hot waffle exploded; default OFF. */
static inline int h3_dit_lora_overlap_wanted(void) {
    const char *e = getenv("H3_DIT_LORA_OVERLAP");
    return e && e[0] == '1' && e[1] == '\0';
}

/* cuBLASLt INT8 workspace. Default 256 MiB (was 64). Clamp 32–1024. */
static inline size_t h3_dit_int8_lt_workspace_bytes(void) {
    const char *e = getenv("H3_INT8_LT_WORKSPACE_MIB");
    long v = 256;
    if (e && e[0]) {
        v = strtol(e, NULL, 10);
        if (v < 32)
            v = 32;
        if (v > 1024)
            v = 1024;
    }
    return (size_t)v * 1024ull * 1024ull;
}

/* INT8 acc budget. Default 1536 MiB. 0 = ignore budget (TILE_M only). */
static inline size_t h3_dit_int8_gemm_acc_mib(void) {
    const char *e = getenv("H3_INT8_GEMM_ACC_MIB");
    long v;
    if (!e || !e[0])
        return 1536;
    if (e[0] == '0' && e[1] == '\0')
        return 0;
    v = strtol(e, NULL, 10);
    if (v < 0)
        return 0;
    if (v > 8192)
        v = 8192;
    return (size_t)v;
}

/* H3_INT8_GEMM_TILE_M: unset = 4096 (§14); "0" = full M; else aligned cap.
 * Adaptive (unset=no cap, full M if acc fits) measured 2026-09-30: checksum
 * matched §14 but hot dit ≈39.8s and hires window 28 vs 31–33. Default OFF. */
static inline int h3_dit_int8_gemm_tile_m_explicit(void) {
    const char *e = getenv("H3_INT8_GEMM_TILE_M");
    long v;
    if (!e || !e[0])
        return 4096;
    if (e[0] == '0' && e[1] == '\0')
        return 0;
    v = strtol(e, NULL, 10);
    if (v < 256)
        v = 256;
    if (v > 65536)
        v = 65536;
    return (int)((v + 15) & ~15);
}

/* Rows per INT8 GEMM tile. Default 4096. TILE_M=0 → full M (may cut window). */
static inline size_t h3_dit_int8_gemm_tile_rows(size_t m, size_t n) {
    int tile_ex;
    size_t acc_mib, cap, budget_rows;
    if (m == 0 || n == 0)
        return m;
    tile_ex = h3_dit_int8_gemm_tile_m_explicit();
    if (tile_ex == 0)
        return m;
    cap = (size_t)tile_ex;
    acc_mib = h3_dit_int8_gemm_acc_mib();
    budget_rows = m;
    if (acc_mib > 0 && n > 0) {
        size_t bytes = acc_mib * 1024ull * 1024ull;
        budget_rows = bytes / (n * sizeof(int32_t));
        budget_rows &= ~(size_t)15u;
        if (budget_rows < 256u)
            budget_rows = 256u;
        if (budget_rows > m)
            budget_rows = m;
    }
    if (budget_rows < cap)
        cap = budget_rows;
    if (cap > m)
        cap = m;
    if (cap < 256u && m >= 256u)
        cap = 256u;
    return cap;
}

static inline size_t h3_dit_int8_gemm_acc_bytes(size_t m, size_t n) {
    size_t rows = h3_dit_int8_gemm_tile_rows(m, n);
    if (n > 0 && rows > ((size_t)-1 / n) / sizeof(int32_t))
        return (size_t)-1;
    return rows * n * sizeof(int32_t);
}

#ifdef __cplusplus
}
#endif

#endif
