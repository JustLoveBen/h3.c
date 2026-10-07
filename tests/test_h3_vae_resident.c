/* tests/test_h3_vae_resident.c
 * Defaults: VAE weight cache ON, tile batch 4 (Comfy _decode_tile_row).
 * H3_VAE_WEIGHT_CACHE=0 / H3_VAE_TILE_BATCH=N override. */
#include "h3_vae_int8.h"

#include <stdio.h>
#include <stdlib.h>

static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}

int main(void) {
    unsetenv("H3_VAE_WEIGHT_CACHE");
    expect(h3_vae_int8_weight_cache_wanted() == 1, "cache default on");
    setenv("H3_VAE_WEIGHT_CACHE", "0", 1);
    expect(h3_vae_int8_weight_cache_wanted() == 0, "cache 0 disables");
    setenv("H3_VAE_WEIGHT_CACHE", "1", 1);
    expect(h3_vae_int8_weight_cache_wanted() == 1, "cache 1 on");
    unsetenv("H3_VAE_TILE_BATCH");
    unsetenv("H3_VAE_FLASH_SDPA");
    expect(h3_vae_int8_tile_batch_want() == 4, "tile batch default 4");
    setenv("H3_VAE_TILE_BATCH", "1", 1);
    expect(h3_vae_int8_tile_batch_want() == 1, "tile batch 1");
    setenv("H3_VAE_TILE_BATCH", "9", 1);
    expect(h3_vae_int8_tile_batch_want() == 8, "tile batch clamp 8");
    unsetenv("H3_VAE_REPEAT_RELEASE");
    expect(h3_vae_int8_repeat_release_mode() == H3_VAE_REPEAT_DEVICE,
           "repeat device default");
    setenv("H3_VAE_REPEAT_RELEASE", "keep", 1);
    expect(h3_vae_int8_repeat_release_mode() == H3_VAE_REPEAT_KEEP,
           "repeat keep");
    setenv("H3_VAE_REPEAT_RELEASE", "full", 1);
    expect(h3_vae_int8_repeat_release_mode() == H3_VAE_REPEAT_FULL,
           "repeat full");
    setenv("H3_VAE_REPEAT_RELEASE", "device", 1);
    expect(h3_vae_int8_repeat_release_mode() == H3_VAE_REPEAT_DEVICE,
           "repeat device");
    unsetenv("H3_VAE_REPEAT_RELEASE");

    unsetenv("H3_VAE_GEMM_F16_DIRECT");
    expect(h3_vae_int8_gemm_f16_direct() == 1, "gemm direct default on");
    setenv("H3_VAE_GEMM_F16_DIRECT", "0", 1);
    expect(h3_vae_int8_gemm_f16_direct() == 0, "gemm direct 0");
    setenv("H3_VAE_GEMM_F16_DIRECT", "1", 1);
    expect(h3_vae_int8_gemm_f16_direct() == 1, "gemm direct 1");
    unsetenv("H3_VAE_GEMM_F16_DIRECT");

    unsetenv("H3_VAE_FUSED_EPILOGUE");
    expect(h3_vae_int8_fused_epilogue() == 1, "fused epilogue default on");
    setenv("H3_VAE_FUSED_EPILOGUE", "0", 1);
    expect(h3_vae_int8_fused_epilogue() == 0, "fused epilogue 0");
    unsetenv("H3_VAE_FUSED_EPILOGUE");

    unsetenv("H3_VAE_ACT_F16");
    expect(h3_vae_int8_act_f16() == 1, "act f16 default on");
    setenv("H3_VAE_ACT_F16", "0", 1);
    expect(h3_vae_int8_act_f16() == 0, "act f16 0");
    setenv("H3_VAE_ACT_F16", "1", 1);
    expect(h3_vae_int8_act_f16() == 1, "act f16 1");
    unsetenv("H3_VAE_ACT_F16");

    unsetenv("H3_VAE_CUDA_GRAPH");
    unsetenv("H3_VAE_FLASH_SDPA");
    expect(h3_vae_int8_cuda_graph_wanted() == 0, "graph unset = 0");
    setenv("H3_VAE_FLASH_SDPA", "1", 1);
    expect(h3_vae_int8_cuda_graph_wanted() == 0, "graph not follow flash");
    setenv("H3_VAE_CUDA_GRAPH", "1", 1);
    expect(h3_vae_int8_cuda_graph_wanted() == 1, "graph 1");
    setenv("H3_VAE_CUDA_GRAPH", "0", 1);
    expect(h3_vae_int8_cuda_graph_wanted() == 0, "graph 0");
    unsetenv("H3_VAE_CUDA_GRAPH");
    unsetenv("H3_VAE_FLASH_SDPA");
    return fails ? 1 : (printf("test_h3_vae_resident: PASS\n"), 0);
}
