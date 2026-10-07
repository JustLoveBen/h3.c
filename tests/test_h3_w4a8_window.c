/* CPU: max_blocks_fit arithmetic + sage-drop policy (R4). No CUDA. */
#include "h3_sdpa.h"
#include "h3_w4a8_window.h"

#include <stdio.h>

static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}

int main(void) {
    const size_t gib = (size_t)1 << 30;
    const size_t mib = (size_t)1 << 20;
    const size_t bpb = 400u * mib; /* ~400MiB / INT8 block */
    unsigned fit;

    /* 768P-like: 1.5GiB free, 1.5GiB reserve, 3GiB sage → window=1 */
    fit = h3_w4a8_cache_max_blocks_fit_calc(1536u * mib, 1536u * mib, 3u * gib,
                                            bpb, 50);
    expect(fit == 1, "768p with sage+reserve → 1");

    /* Same free after dropping sage: usable += 3GiB → 3GiB/400MiB = 7 */
    fit = h3_w4a8_cache_max_blocks_fit_calc(1536u * mib + 3u * gib, 1536u * mib,
                                            0, bpb, 50);
    expect(fit == 7, "drop sage → window=7");

    /* 480P-like: 12GiB free, 1.5GiB reserve, 0.8GiB sage */
    fit = h3_w4a8_cache_max_blocks_fit_calc(12u * gib, 1536u * mib, 800u * mib,
                                            bpb, 50);
    expect(fit >= 20, "480p keeps large window");

    fit = h3_w4a8_cache_max_blocks_fit_calc(20u * gib, 8u * gib, 0, bpb, 50);
    expect(fit == 30, "clamp to n_blocks=50 but 12GiB/400MiB=30");

    fit = h3_w4a8_cache_max_blocks_fit_calc(40u * gib, 8u * gib, 0, bpb, 50);
    expect(fit == 50, "clamp to n_blocks");

    expect(h3_w4a8_cache_max_blocks_fit_calc(8u * gib, 1u * gib, 0, 0, 50) == 0,
           "bpb=0 → 0");

    expect(h3_w4a8_cache_should_drop_sage_pool(1, 10, 8, 0) == 1,
           "fake hit + window=1 → drop");
    expect(h3_w4a8_cache_should_drop_sage_pool(1, 10, 8, 1) == 0,
           "true hit keeps pool");
    expect(h3_w4a8_cache_should_drop_sage_pool(20, 22, 8, 0) == 0,
           "window already ≫1");
    expect(h3_w4a8_cache_should_drop_sage_pool(1, 1, 8, 0) == 0,
           "drop would not grow fit");

    /* 32GB 5090 lift: drop INT8+WS+low-res QKVO first. ~28GiB free.
     * WS-first 9GiB + 1.5GiB reserve. Sol QKVO 2.09GiB as attn-pool.
     * leftover / 400MiB must keep window=32 (do not shrink). */
    {
        const size_t free28 = 28u * gib;
        const size_t ws9 = 9u * gib;
        const size_t reserve = 1536u * mib;
        size_t qkvo = h3_sol_cute_qkvo_bytes(39172, 56, 128);
        size_t usable;
        unsigned fit_qkvo_after_int8, fit_if_qkvo_before_int8;
        expect(qkvo > 2u * gib && qkvo < 3u * gib, "hires QKVO ~2.1GiB");
        usable = free28 - ws9;
        fit_if_qkvo_before_int8 = h3_w4a8_cache_max_blocks_fit_calc(
            usable, reserve, qkvo, bpb, 50);
        fit_qkvo_after_int8 = h3_w4a8_cache_max_blocks_fit_calc(
            usable, reserve, 0, bpb, 50);
        expect(fit_qkvo_after_int8 >= 32, "INT8-first keeps window>=32");
        expect(fit_if_qkvo_before_int8 >= 32,
               "even QKVO-first still fits 32 on 28-9GiB");
        /* Compile spike / torch reserved ~4GiB on top of QKVO pins window. */
        fit_if_qkvo_before_int8 = h3_w4a8_cache_max_blocks_fit_calc(
            usable, reserve, qkvo + 4u * gib, bpb, 50);
        expect(fit_if_qkvo_before_int8 < 32,
               "compile+QKVO before INT8 would drop window — warmup first");
    }

    return fails ? 1 : (printf("test_h3_w4a8_window: PASS\n"), 0);
}
