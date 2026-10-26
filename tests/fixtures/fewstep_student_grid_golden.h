/* Frozen student_grid σ tables for PDMD/DMAD contracts (float32).
 * Formula: linspace(1,0,nfe+1) then SD3 shift s*t/(1+(s-1)*t); last=0.
 * Numbers come from host float32 fill (tools/dump_fewstep_student_grid).
 * Do not hand-edit; regenerate the dump tool and refresh this file.
 * Consumed by h3_fewstep_contract.c and tests/test_h3_fewstep_contract.c.
 */
#ifndef H3_FEWSTEP_STUDENT_GRID_GOLDEN_H
#define H3_FEWSTEP_STUDENT_GRID_GOLDEN_H

#ifdef __cplusplus
extern "C" {
#endif

/* pdmd4/dmad4 video shift=12 (nfe=4, len=5) */
#define H3_FEWSTEP_GOLDEN_PDMD4_VIDEO_S12_LEN 5
static const float H3_FEWSTEP_GOLDEN_PDMD4_VIDEO_S12[5] = {
    1.0f,
    0.972972989f,
    0.923076928f,
    0.800000012f,
    0.0f
};

/* pdmd4 audio shift=3 (nfe=4, len=5) */
#define H3_FEWSTEP_GOLDEN_PDMD4_AUDIO_S3_LEN 5
static const float H3_FEWSTEP_GOLDEN_PDMD4_AUDIO_S3[5] = {
    1.0f,
    0.899999976f,
    0.75f,
    0.5f,
    0.0f
};

/* dmad4 audio shift=2 (nfe=4, len=5) */
#define H3_FEWSTEP_GOLDEN_DMAD4_AUDIO_S2_LEN 5
static const float H3_FEWSTEP_GOLDEN_DMAD4_AUDIO_S2[5] = {
    1.0f,
    0.857142866f,
    0.666666687f,
    0.400000006f,
    0.0f
};

/* pdmd2 video shift=12 (nfe=2, len=3) */
#define H3_FEWSTEP_GOLDEN_PDMD2_VIDEO_S12_LEN 3
static const float H3_FEWSTEP_GOLDEN_PDMD2_VIDEO_S12[3] = {
    1.0f,
    0.923076928f,
    0.0f
};

/* pdmd2 audio paper shift=3 (nfe=2, len=3) */
#define H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S3_LEN 3
static const float H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S3[3] = {
    1.0f,
    0.75f,
    0.0f
};

/* pdmd2 audio look-feel shift=6 (nfe=2, len=3) */
#define H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S6_LEN 3
static const float H3_FEWSTEP_GOLDEN_PDMD2_AUDIO_S6[3] = {
    1.0f,
    0.857142866f,
    0.0f
};

#ifdef __cplusplus
}
#endif

#endif /* H3_FEWSTEP_STUDENT_GRID_GOLDEN_H */
