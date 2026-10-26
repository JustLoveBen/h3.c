/* Host-only: print frozen PDMD/DMAD student_grid σ tables + recompute fill.
 * Usage: ./dump_fewstep_student_grid
 * Refresh tests/fixtures/fewstep_student_grid_golden.h from this output when
 * the formula changes (should be rare).
 */
#include "h3_fewstep_contract.h"
#include "tests/fixtures/fewstep_student_grid_golden.h"

#include <stdio.h>

static void print_row(const char *label, const float *v, int n) {
    int i;
    printf("%s:", label);
    for (i = 0; i < n; i++)
        printf(" %.9g", (double)v[i]);
    printf("\n");
}

static void check_fill_vs_golden(const char *name, int nfe, float shift,
                                 const float *golden, int glen) {
    float got[16];
    int i, bad = 0;
    if (glen != nfe + 1) {
        printf("FAIL %s glen=%d nfe+1=%d\n", name, glen, nfe + 1);
        return;
    }
    h3_fewstep_student_grid_fill(nfe, shift, got);
    for (i = 0; i < glen; i++) {
        float d = got[i] - golden[i];
        if (d > 1e-7f || d < -1e-7f) {
            printf("FAIL %s[%d] fill=%.9g golden=%.9g\n", name, i,
                   (double)got[i], (double)golden[i]);
            bad = 1;
        }
    }
    if (!bad)
        printf("ok fill≡golden %s nfe=%d shift=%.3g\n", name, nfe,
               (double)shift);
}

int main(void) {
    printf("# fewstep student_grid schedule (host float32)\n");
    print_row("pdmd4/dmad4 video s12", H3_FEWSTEP_GOLDEN_PDMD4_VIDEO_S12,
              H3_FEWSTEP_GOLDEN_PDMD4_VIDEO_S12_LEN);
    print_row("pdmd4 audio s3", H3_FEWSTEP_GOLDEN_PDMD4_AUDIO_S3,
              H3_FEWSTEP_GOLDEN_PDMD4_AUDIO_S3_LEN);
    print_row("dmad4 audio s2", H3_FEWSTEP_GOLDEN_DMAD4_AUDIO_S2,
              H3_FEWSTEP_GOLDEN_DMAD4_AUDIO_S2_LEN);
    print_row("pdmd2 video s12", H3_FEWSTEP_GOLDEN_PDMD2_VIDEO_S12,
              H3_FEWSTEP_GOLDEN_PDMD2_VIDEO_S12_LEN);
    print_row("pdmd2 audio s3", H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S3,
              H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S3_LEN);
    print_row("pdmd2 audio s6", H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S6,
              H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S6_LEN);

    check_fill_vs_golden("pdmd4_video", 4, 12.0f,
                         H3_FEWSTEP_GOLDEN_PDMD4_VIDEO_S12,
                         H3_FEWSTEP_GOLDEN_PDMD4_VIDEO_S12_LEN);
    check_fill_vs_golden("pdmd4_audio", 4, 3.0f,
                         H3_FEWSTEP_GOLDEN_PDMD4_AUDIO_S3,
                         H3_FEWSTEP_GOLDEN_PDMD4_AUDIO_S3_LEN);
    check_fill_vs_golden("dmad4_audio", 4, 2.0f,
                         H3_FEWSTEP_GOLDEN_DMAD4_AUDIO_S2,
                         H3_FEWSTEP_GOLDEN_DMAD4_AUDIO_S2_LEN);
    check_fill_vs_golden("pdmd2_video", 2, 12.0f,
                         H3_FEWSTEP_GOLDEN_PDMD2_VIDEO_S12,
                         H3_FEWSTEP_GOLDEN_PDMD2_VIDEO_S12_LEN);
    check_fill_vs_golden("pdmd2_audio3", 2, 3.0f,
                         H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S3,
                         H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S3_LEN);
    check_fill_vs_golden("pdmd2_audio6", 2, 6.0f,
                         H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S6,
                         H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S6_LEN);
    return 0;
}
