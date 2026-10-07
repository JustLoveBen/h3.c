/* CPU: AdaLN-range F16 prescale. No CUDA.
 * F32 AdaLN/SwiGLU absmax can exceed 65504, and ConvRot Hadamard
 * amplifies ∞-norm by sqrt(gs). Unscale restores INT8 act_scale. */
#include "h3_dit_act_f16.h"

#include <math.h>
#include <stdio.h>

static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}

int main(void) {
    const int gs = 32;
    const float f16_max = 65504.0f;
    float s, room, amp;
    float act_f16, act_corr;

    expect(h3_dit_act_f16_pre_scale(1.0f, gs) == 1.0f, "small absmax → 1");
    expect(h3_dit_act_f16_pre_scale(0.0f, gs) == 1.0f, "zero → 1");
    expect(h3_dit_act_f16_pre_scale(1e5f, 3) == 1.0f, "gs<4 → 1");

    /* Naive F16 cast of AdaLN-scale values overflows. */
    expect(1e5f > f16_max, "1e5 exceeds F16 max");
    expect(1e5f * sqrtf((float)gs) > f16_max, "ConvRot amp also overflows");

    s = h3_dit_act_f16_pre_scale(1e5f, gs);
    expect(s > 0.0f && s < 1.0f, "large AdaLN absmax scales down");
    amp = sqrtf((float)gs);
    room = f16_max * 0.96f;
    expect(1e5f * s * amp <= room + 1e-3f, "scaled*sqrt(gs) fits F16");

    /* INT8 dequant identity: act_scale_f16 / S == absmax/127. */
    act_f16 = (1e5f * s) / 127.0f;
    act_corr = act_f16 / s;
    expect(fabsf(act_corr - (1e5f / 127.0f)) < 1e-4f, "unscale restores act_scale");

    s = h3_dit_act_f16_pre_scale(1e5f, gs);
    expect(!isnan(s) && !isinf(s), "scale itself finite");

    return fails ? 1 : (printf("test_h3_dit_act_f16: PASS\n"), 0);
}
