#ifndef H3_W4A8_WINDOW_H
#define H3_W4A8_WINDOW_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pure arithmetic: how many INT8 blocks fit after reserve + sage pool.
 * usable = free_now + already_resident_bytes (or free after drop).
 * Returns at least 1 when bytes_per_block > 0 so LRU never goes to 0. */
static inline unsigned h3_w4a8_cache_max_blocks_fit_calc(
    size_t usable_bytes, size_t reserve_bytes, size_t sage_pool_bytes,
    size_t bytes_per_block, unsigned n_blocks) {
    size_t overhead;
    size_t budget;
    unsigned fit;
    if (bytes_per_block == 0)
        return 0;
    overhead = reserve_bytes;
    if (sage_pool_bytes > ((size_t)-1) - overhead)
        return 1;
    overhead += sage_pool_bytes;
    if (usable_bytes <= overhead)
        return 1;
    budget = usable_bytes - overhead;
    fit = (unsigned)(budget / bytes_per_block);
    if (fit < 1)
        fit = 1;
    if (n_blocks > 0 && fit > n_blocks)
        fit = n_blocks;
    return fit;
}

/* R4: drop sage pool when it is what pins window below min, unless sage3
 * already true-hit (then keep pool and shrink reserve instead). */
static inline int h3_w4a8_cache_should_drop_sage_pool(unsigned fit_with_sage,
                                                      unsigned fit_without_sage,
                                                      unsigned min_window,
                                                      int sage_true_hit) {
    if (min_window < 2)
        min_window = 2;
    if (sage_true_hit)
        return 0;
    if (fit_with_sage >= min_window)
        return 0;
    return fit_without_sage > fit_with_sage;
}

#ifdef __cplusplus
}
#endif

#endif
