#ifndef H3_SAMPLER_H
#define H3_SAMPLER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Denoise sampler for CUDA generate (fewstep / DMAD).
 *
 * velocity_euler — default; Comfy FLOW+CONST ≡ Metal:
 *   X += (σ − σ′) · v   with raw DiT velocity v.
 *
 * lcm_renoise — ComfyUI SamplerLCM + ModelSamplingDiscreteFlow CONST
 *   (scheduler "simple"):
 *   x0 = X + σ · v
 *   if σ′ == 0: X = x0
 *   else:       X = (1 − σ′) · x0 + σ′ · ε,  ε ~ N(0,1)
 * Matches DMAD author graph (lcm + simple); NOT Euler.
 *
 * See docs/research/2026-10-26-dmad-lcm-renoise-sampler.md
 */

typedef enum {
    H3_SAMPLER_VELOCITY_EULER = 0,
    H3_SAMPLER_LCM_RENOISE = 1
} h3_sampler_id;

/* Parse CLI / H3_SAMPLER token. Unknown → VELOCITY_EULER and fills err. */
h3_sampler_id h3_sampler_parse(const char *name, char *err, size_t err_size);

/* CLI NULL → H3_SAMPLER env. Empty/unset → velocity_euler. */
h3_sampler_id h3_sampler_resolve(const char *cli, char *err, size_t err_size);

const char *h3_sampler_name(h3_sampler_id id);

/*
 * One LCM + simple re-noise step on host float buffers.
 * sample: in/out latent X (length count).
 * velocity: raw DiT v (same length).
 * noise: N(0,1) length count; required when sigma_next > 1e-8, else ignored
 *        (may be NULL when sigma_next ≈ 0).
 * Returns 1 on success, 0 and fills error on bad args.
 */
int h3_sampler_lcm_renoise_step_f32(float *sample, const float *velocity,
                                    size_t count, float sigma, float sigma_next,
                                    const float *noise, char *error,
                                    size_t error_size);

#ifdef __cplusplus
}
#endif

#endif /* H3_SAMPLER_H */
