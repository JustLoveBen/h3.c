#ifndef H3_WAFFLE_H
#define H3_WAFFLE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 16px-period block-edge energy / interior energy on RGB24.
 * Official tiled VAE + deblock16 ≈ 1.0; full-frame waffle ≈ 2.6. */
double h3_waffle_b16_rgb24(const uint8_t *rgb, int width, int height);

/* Minimum semantic-energy gate (DMAD-noise metric-failure fix, see
 * docs/research/2026-10-07-dmad-noise-diagnosis.md): pure-noise output
 * concentrates energy in 16px block-mean jumps with flat interiors, so the
 * edge/inner gradient ratio lands in the 0.8-1.3 "good" band and produces
 * false low scores. NOTE: raw inner gradient energy does NOT separate the
 * classes (flat-but-real kpia content scores lower than heavy mosaic noise).
 * The separating statistic is the 16px between-block / within-block luma
 * variance ratio, calibrated on 864x480 f0/f60/f119 samples (18 frames):
 *   normal (kpia-vae-pipeline r1/r2): 17.5 .. 146.8
 *   DMAD noise (C_full_critic / D_kijai_rank39 r1/r2): 1.20 .. 7.57
 * Gate = 10.0, placed just above the noise cluster (max 7.57) rather than at
 * the geometric midpoint: kpia r1 f0 (a real, human-reviewed portrait frame
 * whose large flat skin regions make 16px blocks low-frequency self-similar)
 * sits at 17.5, inside the raw gap, so a midpoint threshold (24.0) would
 * false-positive it. INVALID verdicts on borderline real frames are more
 * damaging than late triggering on noise. */
#define H3_WAFFLE_B16_MIN_BLOCK_RATIO 10.0

/* Same metric as h3_waffle_b16_rgb24, plus gate observability:
 *   block_ratio (optional out): between/within 16px block luma variance.
 *   low_energy  (optional out): 1 when block_ratio < gate (b16 INVALID). */
double h3_waffle_b16_gated_rgb24(const uint8_t *rgb, int width, int height,
                                 double *block_ratio, int *low_energy);

/* Print waffle: frame= b16= for f0/24/48/90/119 (clamped to nframes-1).
 * Frames below the energy gate are marked INVALID_LOWENERGY; the b16 value
 * itself is left untouched so existing regex/golden parsing keeps working. */
void h3_waffle_log_rgb24(const uint8_t *rgb, int nframes, int width, int height);

#ifdef __cplusplus
}
#endif

#endif
