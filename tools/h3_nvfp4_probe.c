/* E1: TE NVFP4 header + dequant + embed lookup / synth condition.
 *   cmake --build build-cuda -j --target h3_nvfp4_probe
 *   ./build-cuda/h3_nvfp4_probe --tokens 8 FILE.safetensors
 */
#include "h3_nvfp4.h"
#include "h3_nvfp4_cuda.h"
#include "h3_gpu_cuda.h"
#include "h3_token_ids.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int all_finite(const float *x, size_t n) {
    size_t i;
    for (i = 0; i < n; i++)
        if (!isfinite(x[i])) return 0;
    return 1;
}

static float absmax(const float *x, size_t n) {
    float m = 0.0f;
    size_t i;
    for (i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > m) m = a;
    }
    return m;
}

static double elapsed_ms(clock_t t0, clock_t t1) {
    return 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
    const char *path = NULL;
    const char *base = "model.layers.0.self_attn.q_proj";
    const char *token_canary = NULL;
    const char *token_ids_file = NULL;
    size_t max_rows = 8;
    size_t tokens = 8;
    unsigned nlay = 1;
    unsigned repeat_linear = 0;
    char error[512];
    h3_nvfp4_store store;
    h3_nvfp4_linear lin;
    h3_nvfp4_embed emb;
    h3_cuda *cuda = NULL;
    float *w = NULL, *cond = NULL;
    int32_t *ids = NULL;
    size_t rows = 0, out_tok = 0, i;
    int rc = 1;
    int iarg;

    memset(&store, 0, sizeof(store));
    for (iarg = 1; iarg < argc; iarg++) {
        if (!strcmp(argv[iarg], "--base") && iarg + 1 < argc) {
            base = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--rows") && iarg + 1 < argc) {
            max_rows = (size_t)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--tokens") && iarg + 1 < argc) {
            tokens = (size_t)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--layers") && iarg + 1 < argc) {
            nlay = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--repeat-linear") && iarg + 1 < argc) {
            repeat_linear = (unsigned)strtoul(argv[++iarg], NULL, 10);
        } else if (!strcmp(argv[iarg], "--token-canary") && iarg + 1 < argc) {
            token_canary = argv[++iarg];
        } else if (!strcmp(argv[iarg], "--token-ids-file") && iarg + 1 < argc) {
            token_ids_file = argv[++iarg];
        } else if (argv[iarg][0] == '-') {
            fprintf(stderr, "unknown option: %s\n", argv[iarg]);
            return 2;
        } else {
            path = argv[iarg];
        }
    }
    if (!path) {
        fprintf(stderr,
                "usage: %s [--base NAME] [--rows N] [--tokens T] [--layers L] "
                "[--repeat-linear N] [--token-canary fox|cinematic] "
                "[--token-ids-file F] "
                "FILE.safetensors\n",
                argv[0]);
        return 2;
    }

    if (!h3_nvfp4_store_open(path, &store, error, sizeof(error))) {
        fprintf(stderr, "open failed: %s\n", error);
        return 1;
    }
    cuda = h3_cuda_create(0, error, sizeof(error));
    if (cuda) {
        (void)h3_nvfp4_cuda_enable(cuda, error, sizeof(error));
        printf("te_backend: %s\n",
               h3_nvfp4_cuda_enabled() ? "cublas" : "host_gemm");
    } else {
        printf("te_backend: host_gemm (cuda unavailable: %s)\n", error);
    }
    printf("path: %s\n", path);
    printf("tensors: %zu  format=%s\n", store.header.tensor_count,
           store.quant.format[0] ? store.quant.format : "(none)");
    printf("scope: NVFP4 dequant + configurable-layer TE encode (G1); "
           "CUDA token path via --token-ids-file/--token-canary; "
           "HF tokenizer.json BPE / visual tower still Metal-only\n");

    if (!h3_nvfp4_find_linear(&store, base, &lin, error, sizeof(error))) {
        fprintf(stderr, "find_linear failed: %s\n", error);
        goto cleanup;
    }
    printf("linear: %s  U8[%zu,%zu]→F32[%zu,%zu]  groups=%zu gs=%zu\n",
           lin.base_name, lin.n, lin.k / 2, lin.n, lin.k, lin.groups,
           lin.group_size);
    w = h3_nvfp4_dequant_rows_f32(&store, &lin, max_rows, &rows, error,
                                  sizeof(error));
    if (!w) {
        fprintf(stderr, "dequant failed: %s\n", error);
        goto cleanup;
    }
    printf("dequant_rows: %zu absmax=%.6g finite=%d\n", rows,
           absmax(w, rows * lin.k), all_finite(w, rows * lin.k));
    if (repeat_linear > 0) {
        float *x = (float *)malloc(tokens * lin.k * sizeof(float));
        float *y = (float *)malloc(tokens * lin.n * sizeof(float));
        unsigned rep;
        if (!x || !y) {
            fprintf(stderr, "OOM repeat_linear buffers\n");
            goto cleanup;
        }
        for (i = 0; i < tokens * lin.k; i++)
            x[i] = (float)((int)(i % 29) - 14) * (1.0f / 29.0f);
        for (rep = 0; rep < repeat_linear; rep++) {
            clock_t t0 = clock();
            if (!h3_nvfp4_linear_f32(&store, &lin, x, tokens, y, error,
                                     sizeof(error))) {
                free(x);
                free(y);
                fprintf(stderr, "repeat_linear failed: %s\n", error);
                goto cleanup;
            }
            printf("repeat_linear: pass=%u tokens=%zu out_n=%zu ms=%.1f "
                   "absmax=%.6g\n",
                   rep + 1, tokens, lin.n, elapsed_ms(t0, clock()),
                   absmax(y, tokens * lin.n));
        }
        free(x);
        free(y);
    }

    if (!h3_nvfp4_find_embed(&store, &emb, error, sizeof(error))) {
        fprintf(stderr, "find_embed failed: %s\n", error);
        goto cleanup;
    }
    printf("embed: vocab=%zu dim=%zu\n", emb.vocab, emb.dim);
    ids = NULL;
    {
        size_t n_ids = 0;
        if (token_ids_file) {
            if (!h3_token_ids_load_file(token_ids_file, &ids, &n_ids, error,
                                        sizeof(error))) {
                fprintf(stderr, "token-ids-file: %s\n", error);
                goto cleanup;
            }
        } else if (token_canary) {
            if (!h3_token_ids_canary(token_canary, &ids, &n_ids, error,
                                     sizeof(error))) {
                fprintf(stderr, "token-canary: %s\n", error);
                goto cleanup;
            }
        } else {
            ids = (int32_t *)malloc(tokens * sizeof(int32_t));
            if (!ids) {
                fprintf(stderr, "OOM\n");
                goto cleanup;
            }
            for (i = 0; i < tokens; i++) ids[i] = (int32_t)(1 + i);
            n_ids = tokens;
        }
        if (!h3_token_ids_fit(&ids, &n_ids, tokens, error, sizeof(error))) {
            fprintf(stderr, "token fit: %s\n", error);
            goto cleanup;
        }
        printf("te_tokens: n=%zu source=%s\n", n_ids,
               token_ids_file ? "file" : token_canary ? "canary" : "seq");
    }
    cond = h3_nvfp4_make_condition_f32(&store, ids, tokens, 0, NULL, &out_tok,
                                       error, sizeof(error));
    if (!cond) {
        fprintf(stderr, "make_condition failed: %s\n", error);
        goto cleanup;
    }
    printf("condition_embed: shape=[%zu,%d] absmax=%.6g finite=%d\n", out_tok,
           H3_NVFP4_TEXT_DIM, absmax(cond, out_tok * H3_NVFP4_TEXT_DIM),
           all_finite(cond, out_tok * H3_NVFP4_TEXT_DIM));
    free(cond);
    cond = NULL;
    {
        h3_nvfp4_encode_stats est;
        const char *el = getenv("H3_NVFP4_TE_LAYERS");
        if (el && el[0]) nlay = (unsigned)strtoul(el, NULL, 10);
        if (nlay < 1) nlay = 1;
        cond = h3_nvfp4_encode_layers_f32(&store, ids, tokens, nlay, &est,
                                          error, sizeof(error));
        if (!cond) {
            fprintf(stderr, "encode_layers failed: %s\n", error);
            goto cleanup;
        }
        printf("condition_te: layers=%u shape=[%zu,%d] absmax=%.6g finite=%d "
               "embed_ms=%.1f layers_ms=%.1f total_ms=%.1f cuda=%d "
               "embed_cache=%d\n",
               est.n_layers, tokens, H3_NVFP4_TEXT_DIM,
               absmax(cond, tokens * H3_NVFP4_TEXT_DIM), est.all_finite,
               est.embed_ms, est.layers_ms, est.total_ms, est.used_cuda,
               est.embed_cache_hit);
        out_tok = tokens;
    }

    if (all_finite(w, rows * lin.k) &&
        all_finite(cond, out_tok * H3_NVFP4_TEXT_DIM)) {
        printf("summary: nvfp4 dequant+TE_layers_encode → PASS\n");
        printf("limit: HF tokenizer.json BPE / visual / default-50 still "
               "optional (--layers / H3_NVFP4_TE_LAYERS; product rec ≤8 @16tok)\n");
        rc = 0;
    } else {
        printf("summary: FAIL\n");
    }

cleanup:
    free(w);
    free(cond);
    free(ids);
    h3_nvfp4_cuda_disable();
    h3_cuda_free(cuda);
    h3_nvfp4_store_free(&store);
    return rc;
}
