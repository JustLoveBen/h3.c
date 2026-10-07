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
 * damaging than late triggering on noise.
 *
 * 768P AI-generated content recalibration (2026-10-07, f0/f60/f119 of
 * verify_768p {dit_nvfp4_w4a4 p02/p03/p05/p07, w4a4_e2e p01, gold_e2e p01}
 * plus review-2026-10-07 rh-vdn-comfy p01/p02/p03/p05/p07; 33 frames,
 * 1344x768 / 1376x768):
 *   normal 768P AI:  0.1 .. 8.8
 * The same DMAD noise frames upscaled to 768P (signature is resolution-
 * invariant; block-mean vs within-block variance survives scaling):
 *   DMAD noise @768P: 1.3 .. 9.2
 * The distributions overlap, so NO fixed block_ratio gate separates 768P
 * normal from noise. The gate is therefore resolution-adaptive: at >=768p
 * it falls back to degenerate-frame detection only (block_ratio < 0.02,
 * i.e. an essentially constant frame) and the ungated b16 edge/inner
 * ratio remains the quality signal there (upscaled DMAD noise still
 * scores b16 1.2-1.4 vs 3.7-31.7 for the w4a4 768P runs above); at <768p
 * the original DMAD gate (10.0) is unchanged. */
#define H3_WAFFLE_B16_MIN_BLOCK_RATIO 10.0
#define H3_WAFFLE_B16_MIN_BLOCK_RATIO_HIRES 0.02
#define H3_WAFFLE_B16_HIRES_MIN_HEIGHT 768

static inline double h3_waffle_b16_gate(int width, int height) {
    (void)width;
    return (height >= H3_WAFFLE_B16_HIRES_MIN_HEIGHT)
               ? H3_WAFFLE_B16_MIN_BLOCK_RATIO_HIRES
               : H3_WAFFLE_B16_MIN_BLOCK_RATIO;
}

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
