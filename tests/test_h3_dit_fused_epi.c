/* CPU: fused DiT post-GEMM epilogue vs separate kernels. No CUDA. */
#include "h3_dit_fused_epi.h"
#include "h3_dit_int8_opt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;

static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}

static float maxabs_diff(const float *a, const float *b, size_t n) {
    size_t i;
    float m = 0.0f;
    for (i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

static void test_env(void) {
    unsetenv("H3_DIT_FUSED_EPILOGUE");
    expect(h3_dit_fused_epilogue_wanted() == 1, "fused epi default on");
    setenv("H3_DIT_FUSED_EPILOGUE", "0", 1);
    expect(h3_dit_fused_epilogue_wanted() == 0, "fused epi 0");
    setenv("H3_DIT_FUSED_EPILOGUE", "1", 1);
    expect(h3_dit_fused_epilogue_wanted() == 1, "fused epi 1");
    unsetenv("H3_DIT_FUSED_EPILOGUE");

    unsetenv("H3_DIT_EPI_FUSE");
    expect(h3_dit_epi_fuse_wanted() == 0, "epi fuse default off");
    setenv("H3_DIT_EPI_FUSE", "1", 1);
    expect(h3_dit_epi_fuse_wanted() == 1, "epi fuse 1");
    setenv("H3_DIT_EPI_FUSE", "0", 1);
    expect(h3_dit_epi_fuse_wanted() == 0, "epi fuse 0");
    unsetenv("H3_DIT_EPI_FUSE");
}

static void test_int8_opt_env(void) {
    unsetenv("H3_DIT_LORA_OVERLAP");
    expect(h3_dit_lora_overlap_wanted() == 0, "lora overlap default off");
    setenv("H3_DIT_LORA_OVERLAP", "1", 1);
    expect(h3_dit_lora_overlap_wanted() == 1, "lora overlap 1");
    setenv("H3_DIT_LORA_OVERLAP", "0", 1);
    expect(h3_dit_lora_overlap_wanted() == 0, "lora overlap 0");
    unsetenv("H3_DIT_LORA_OVERLAP");

    unsetenv("H3_INT8_LT_WORKSPACE_MIB");
    expect(h3_dit_int8_lt_workspace_bytes() == 256ull * 1024ull * 1024ull,
           "lt ws default 256MiB");
    setenv("H3_INT8_LT_WORKSPACE_MIB", "64", 1);
    expect(h3_dit_int8_lt_workspace_bytes() == 64ull * 1024ull * 1024ull,
           "lt ws 64");
    unsetenv("H3_INT8_LT_WORKSPACE_MIB");

    unsetenv("H3_INT8_GEMM_ACC_MIB");
    unsetenv("H3_INT8_GEMM_TILE_M");
    expect(h3_dit_int8_gemm_acc_mib() == 1536, "acc mib default 1536");
    expect(h3_dit_int8_gemm_tile_rows(39000, 5376) == 4096,
           "unset TILE_M = 4096 (out/fc2)");
    expect(h3_dit_int8_gemm_tile_rows(39000, 28672) == 4096,
           "unset TILE_M = 4096 (fc1)");
    setenv("H3_INT8_GEMM_TILE_M", "4096", 1);
    expect(h3_dit_int8_gemm_tile_rows(39000, 28672) == 4096,
           "TILE_M=4096 hard cap");
    expect(h3_dit_int8_gemm_tile_rows(39000, 5376) == 4096,
           "TILE_M=4096 out/fc2 also capped");
    setenv("H3_INT8_GEMM_TILE_M", "0", 1);
    expect(h3_dit_int8_gemm_tile_rows(39000, 28672) == 39000,
           "TILE_M=0 full M");
    unsetenv("H3_INT8_GEMM_TILE_M");
    setenv("H3_INT8_GEMM_ACC_MIB", "0", 1);
    expect(h3_dit_int8_gemm_tile_rows(39000, 5376) == 4096,
           "acc 0 + unset TILE_M = 4096");
    unsetenv("H3_INT8_GEMM_ACC_MIB");
}

static void test_gate_adaln(void) {
    const size_t rows = 4, width = 8, slots = 6;
    const size_t gate_slot = 2, shift_slot = 3, scale_slot = 4;
    const float eps = 1e-5f;
    float residual[4 * 8], branch[4 * 8], norm_w[8], mod[4 * 6 * 8];
    float gated_ref[4 * 8], norm_ref[4 * 8], gated[4 * 8], normed[4 * 8];
    uint32_t map[4];
    size_t i, r, c;

    for (i = 0; i < rows * width; i++) {
        residual[i] = (float)((int)i % 7) * 0.1f - 0.3f;
        branch[i] = (float)((int)i % 5) * 0.05f - 0.1f;
    }
    for (c = 0; c < width; c++)
        norm_w[c] = 0.9f + 0.01f * (float)c;
    for (i = 0; i < rows * slots * width; i++)
        mod[i] = ((float)((int)i % 11) - 5.0f) * 0.02f;
    for (r = 0; r < rows; r++)
        map[r] = (uint32_t)r;

    for (r = 0; r < rows; r++) {
        const float *gate = mod + r * slots * width + gate_slot * width;
        const float *shift = mod + r * slots * width + shift_slot * width;
        const float *scale = mod + r * slots * width + scale_slot * width;
        float sum = 0.0f, inv;
        for (c = 0; c < width; c++) {
            float v = residual[r * width + c] + branch[r * width + c] * gate[c];
            gated_ref[r * width + c] = v;
            sum += v * v;
        }
        inv = 1.0f / sqrtf(sum / (float)width + eps);
        for (c = 0; c < width; c++) {
            float n = gated_ref[r * width + c] * inv * norm_w[c];
            norm_ref[r * width + c] = n * (1.0f + scale[c]) + shift[c];
        }
    }

    h3_dit_gate_adaln_f32(residual, branch, norm_w, mod, map, rows, width,
                          slots, gate_slot, shift_slot, scale_slot, eps, gated,
                          normed);
    expect(maxabs_diff(gated, gated_ref, rows * width) < 1e-6f,
           "gate_adaln gated");
    expect(maxabs_diff(normed, norm_ref, rows * width) < 1e-6f,
           "gate_adaln normed");
}

static void test_dequant_swiglu(void) {
    const size_t rows = 3, width = 6;
    int32_t acc[3 * 12];
    float act[3], sc[12], fused[3 * 12], out_ref[3 * 6], out[3 * 6];
    size_t i, r, c;

    for (i = 0; i < rows * width * 2; i++)
        acc[i] = (int32_t)((int)i % 17) - 8;
    for (i = 0; i < width * 2; i++)
        sc[i] = 0.01f + 0.001f * (float)i;
    for (r = 0; r < rows; r++)
        act[r] = 0.02f + 0.001f * (float)r;
    for (r = 0; r < rows; r++) {
        for (c = 0; c < width * 2; c++)
            fused[r * width * 2 + c] =
                act[r] * sc[c] * (float)acc[r * width * 2 + c];
        for (c = 0; c < width; c++) {
            float g = fused[r * width * 2 + c];
            float u = fused[r * width * 2 + width + c];
            out_ref[r * width + c] = (g / (1.0f + expf(-g))) * u;
        }
    }
    h3_dit_dequant_swiglu_f32(acc, act, 0, sc, rows, width, NULL, out);
    expect(maxabs_diff(out, out_ref, rows * width) < 1e-5f, "dequant_swiglu");
}

static void test_dequant_gate(void) {
    const size_t rows = 3, width = 8, slots = 6, gate_slot = 5;
    int32_t acc[3 * 8];
    float act[3], sc[8], residual[3 * 8], mod[3 * 6 * 8];
    float br[3 * 8], out_ref[3 * 8], out[3 * 8];
    uint32_t map[3];
    size_t i, r, c;

    for (i = 0; i < rows * width; i++) {
        acc[i] = (int32_t)((int)i % 13) - 6;
        residual[i] = (float)((int)i % 5) * 0.1f;
    }
    for (c = 0; c < width; c++)
        sc[c] = 0.008f + 0.0005f * (float)c;
    for (r = 0; r < rows; r++) {
        act[r] = 0.03f;
        map[r] = (uint32_t)r;
    }
    for (i = 0; i < rows * slots * width; i++)
        mod[i] = ((float)((int)i % 9) - 4.0f) * 0.05f;
    for (r = 0; r < rows; r++) {
        const float *gate = mod + r * slots * width + gate_slot * width;
        for (c = 0; c < width; c++) {
            br[r * width + c] = act[r] * sc[c] * (float)acc[r * width + c];
            out_ref[r * width + c] =
                residual[r * width + c] + br[r * width + c] * gate[c];
        }
    }
    h3_dit_dequant_gate_f32(acc, act, 0, sc, residual, mod, map, rows, width,
                            slots, gate_slot, NULL, out);
    expect(maxabs_diff(out, out_ref, rows * width) < 1e-5f, "dequant_gate");
}

/* Product turbo LoRA: Y = dequant(acc) + 0.75 * delta, then gate / SwiGLU.
 * Fused epilogue must add the already-scaled delta (scale lives in LoRA GEMM). */
static void test_dequant_gate_lora_075(void) {
    const size_t rows = 3, width = 8, slots = 6, gate_slot = 5;
    const float scale = 0.75f;
    int32_t acc[3 * 8];
    float act[3], sc[8], residual[3 * 8], mod[3 * 6 * 8];
    float delta[3 * 8], delta_s[3 * 8], out_ref[3 * 8], out[3 * 8];
    uint32_t map[3];
    size_t i, r, c;

    for (i = 0; i < rows * width; i++) {
        acc[i] = (int32_t)((int)i % 13) - 6;
        residual[i] = (float)((int)i % 5) * 0.1f;
        delta[i] = ((float)((int)i % 7) - 3.0f) * 0.04f;
        delta_s[i] = scale * delta[i];
    }
    for (c = 0; c < width; c++)
        sc[c] = 0.008f + 0.0005f * (float)c;
    for (r = 0; r < rows; r++) {
        act[r] = 0.03f;
        map[r] = (uint32_t)r;
    }
    for (i = 0; i < rows * slots * width; i++)
        mod[i] = ((float)((int)i % 9) - 4.0f) * 0.05f;
    for (r = 0; r < rows; r++) {
        const float *gate = mod + r * slots * width + gate_slot * width;
        for (c = 0; c < width; c++) {
            float br = act[r] * sc[c] * (float)acc[r * width + c] +
                       delta_s[r * width + c];
            out_ref[r * width + c] = residual[r * width + c] + br * gate[c];
        }
    }
    h3_dit_dequant_gate_f32(acc, act, 0, sc, residual, mod, map, rows, width,
                            slots, gate_slot, delta_s, out);
    expect(maxabs_diff(out, out_ref, rows * width) < 1e-5f,
           "dequant_gate lora 0.75");
}

static void test_dequant_swiglu_lora_075(void) {
    const size_t rows = 3, width = 6;
    const float scale = 0.75f;
    int32_t acc[3 * 12];
    float act[3], sc[12], delta[3 * 12], delta_s[3 * 12];
    float out_ref[3 * 6], out[3 * 6];
    size_t i, r, c;

    for (i = 0; i < rows * width * 2; i++) {
        acc[i] = (int32_t)((int)i % 17) - 8;
        delta[i] = ((float)((int)i % 5) - 2.0f) * 0.03f;
        delta_s[i] = scale * delta[i];
    }
    for (i = 0; i < width * 2; i++)
        sc[i] = 0.01f + 0.001f * (float)i;
    for (r = 0; r < rows; r++)
        act[r] = 0.02f + 0.001f * (float)r;
    for (r = 0; r < rows; r++) {
        for (c = 0; c < width; c++) {
            float g = act[r] * sc[c] * (float)acc[r * width * 2 + c] +
                      delta_s[r * width * 2 + c];
            float u = act[r] * sc[width + c] *
                          (float)acc[r * width * 2 + width + c] +
                      delta_s[r * width * 2 + width + c];
            out_ref[r * width + c] = (g / (1.0f + expf(-g))) * u;
        }
    }
    h3_dit_dequant_swiglu_f32(acc, act, 0, sc, rows, width, delta_s, out);
    expect(maxabs_diff(out, out_ref, rows * width) < 1e-5f,
           "dequant_swiglu lora 0.75");
}

int main(void) {
    test_env();
    test_int8_opt_env();
    test_gate_adaln();
    test_dequant_swiglu();
    test_dequant_gate();
    test_dequant_gate_lora_075();
    test_dequant_swiglu_lora_075();
    return fails ? 1 : (printf("test_h3_dit_fused_epi: PASS\n"), 0);
}
