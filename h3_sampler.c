#include "h3_sampler.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int fail_msg(char *err, size_t err_size, const char *msg) {
    if (err && err_size) {
        snprintf(err, err_size, "%s", msg ? msg : "sampler error");
        err[err_size - 1] = '\0';
    }
    return 0;
}

h3_sampler_id h3_sampler_parse(const char *name, char *err, size_t err_size) {
    if (err && err_size)
        err[0] = '\0';
    if (!name || !name[0])
        return H3_SAMPLER_VELOCITY_EULER;
    if (!strcmp(name, "velocity_euler") || !strcmp(name, "euler") ||
        !strcmp(name, "VelocityEuler"))
        return H3_SAMPLER_VELOCITY_EULER;
    if (!strcmp(name, "lcm_renoise") || !strcmp(name, "lcm") ||
        !strcmp(name, "lcm+re-noise") || !strcmp(name, "lcm_simple") ||
        !strcmp(name, "LCM") || !strcmp(name, "lcm+simple"))
        return H3_SAMPLER_LCM_RENOISE;
    if (err && err_size)
        snprintf(err, err_size,
                 "unknown sampler '%s' (want velocity_euler|lcm_renoise)",
                 name);
    return H3_SAMPLER_VELOCITY_EULER;
}

h3_sampler_id h3_sampler_resolve(const char *cli, char *err, size_t err_size) {
    const char *src = cli;
    if (err && err_size)
        err[0] = '\0';
    if (!src || !src[0]) {
        src = getenv("H3_SAMPLER");
        if (!src || !src[0])
            return H3_SAMPLER_VELOCITY_EULER;
    }
    {
        char perr[256];
        h3_sampler_id id;
        perr[0] = '\0';
        id = h3_sampler_parse(src, perr, sizeof(perr));
        if (perr[0]) {
            if (err && err_size) {
                snprintf(err, err_size, "%s", perr);
                err[err_size - 1] = '\0';
            }
            return H3_SAMPLER_VELOCITY_EULER;
        }
        return id;
    }
}

const char *h3_sampler_name(h3_sampler_id id) {
    switch (id) {
    case H3_SAMPLER_VELOCITY_EULER:
        return "velocity_euler";
    case H3_SAMPLER_LCM_RENOISE:
        return "lcm_renoise";
    default:
        return "unknown";
    }
}

int h3_sampler_lcm_renoise_step_f32(float *sample, const float *velocity,
                                    size_t count, float sigma, float sigma_next,
                                    const float *noise, char *error,
                                    size_t error_size) {
    size_t i;
    int to_clean = (sigma_next <= 1e-8f);

    if (error && error_size)
        error[0] = '\0';
    if (!sample || !velocity || count == 0)
        return fail_msg(error, error_size, "lcm_renoise: null/empty buffer");
    if (!isfinite(sigma) || !isfinite(sigma_next) || sigma < 0.0f ||
        sigma_next < 0.0f)
        return fail_msg(error, error_size, "lcm_renoise: bad sigma");
    if (!to_clean && !noise)
        return fail_msg(error, error_size,
                        "lcm_renoise: noise required when sigma_next > 0");

    /* Comfy sample_lcm + CONST: denoised = X + σ·v (raw v).
     * Then noise_scaling: (1−σ′)·x0 + σ′·ε  (or x0 when σ′=0). */
    for (i = 0; i < count; i++) {
        float x0 = sample[i] + sigma * velocity[i];
        if (to_clean)
            sample[i] = x0;
        else
            sample[i] = (1.0f - sigma_next) * x0 + sigma_next * noise[i];
    }
    return 1;
}
