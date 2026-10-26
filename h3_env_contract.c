/* C061: shared H3_* env contract dump + P0 visual checks (host-safe). */
#include "h3_env_contract.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *name;
    const char *role;
    /* NULL/info | unset | unset_or_0 | arm | arm_file */
    const char *rule;
} h3_env_entry;

/* Keep in sync with tools/print_h3_env_contract.py CONTRACT. */
static const h3_env_entry k_contract[] = {
    {"H3_DIT_NVFP4", "DiT NVFP4 experimental arm (1=on)", "arm"},
    {"H3_DIT_NVFP4_FILE", "NVFP4 sidecar safetensors path", "arm_file"},
    {"H3_DIT_NVFP4_CHAIN", "NVFP4 FC1→FC2 chain WIP (P3)", "unset"},
    {"H3_DIT_NVFP4_CHAIN_FC1_F32", "chain FC1 f32 diag", "unset"},
    {"H3_DIT_NVFP4_CHAIN_DIAG", "chain diagnostics", "unset"},
    {"H3_SDPA_CUDNN_CUDA_GRAPH", "cuDNN SDPA CUDA graph (P0 default off)",
     "unset_or_0"},
    {"H3_TEXT_H_LOAD",
     "load refined text_h (velocity/md5 only; not default visual)", "info"},
    {"H3_TE_PROMPT_CACHE",
     "opt-in TE prompt emb disk cache dir (--te-cache / C062)", "info"},
    {"H3_TE_CACHE_REQUIRE_HIT",
     "1=MISS is FAIL_ENV when TE cache on (warm first)", "info"},
    {"H3_DIT_HOST_STAGE_TE", "DiT host-stage overlap with TE", "info"},
    {"H3_DIT_HOST_STAGE_WAIT_BLOCKS", "host-stage wait blocks", "info"},
    {"H3_VAE_PREWARM_DEV", "VAE device prewarm during DiT", "info"},
    {"H3_FEWSTEP_CONTRACT", "fewstep contract id (pdmd4/pdmd2/dmad4/...)",
     "info"},
    {"H3_FEWSTEP_PRESET", "one-shot preset binding steps/cfg/shift/sampler",
     "info"},
    {"H3_GEN_MEM_TRACE", "per-phase mem trace (forbid during wall-clock KPI)",
     "info"},
    {"H3_SKIP_PPM", "skip PPM dumps", "info"},
    {"H3_DIAG_W_CKSUM", "weight checksum diag", "unset"},
    {"H3_DIAG_BLK0", "block0 diag", "unset"},
    {"H3_DUMP_LATENT", "dump latent once (gen/score split)", "info"},
    {"H3_REPLAY_LATENT", "replay dumped latent", "info"},
    {"H3_DUMP_DIR", "oracle dump dir (TE/velocity; not gold retune)", "info"},
    {"H3_DUMP_ONLY", "1=stop after step0 forward dump", "info"},
    {"H3_LIFECYCLE_TRACE", "1=print C064 lifecycle whitelist/transitions", "info"},
    {"H3_FORBID_HOT_SYNC", "1=abort on H3_SYNC_HOT device sync (C070)", "info"},
    {"H3_CUDA_L2_PIN", "1=pin ConvRot Hadamard in persisting L2 (C017 bit-id)", "info"},
    {"H3_ROPE_PROFILE", "1=log dit/vae rope ms + memo hit/miss", "info"},
    {"H3_ROPE_MEMO", "0=disable RoPE host memo (fair KPI); default on", "info"},
    {"H3_TE_CACHE_AUTO", "1=auto te_cache under out/ when --repeat>=2 (C062)", "info"},
};

static const char *fmt_val(const char *val, char *buf, size_t buf_size) {
    size_t n;
    if (!val)
        return "<unset>";
    if (!val[0])
        return "<empty>";
    n = strlen(val);
    if (n <= 96)
        return val;
    if (buf_size < 97)
        return val;
    memcpy(buf, val, 93);
    buf[93] = '.';
    buf[94] = '.';
    buf[95] = '.';
    buf[96] = '\0';
    return buf;
}

static int is_set(const char *val) { return val != NULL && val[0] != '\0'; }

static int is_on(const char *val) { return val != NULL && val[0] == '1'; }

static int sdpa_graph_wanted(void) {
    const char *e = getenv("H3_SDPA_CUDNN_CUDA_GRAPH");
    return is_on(e) ? 1 : 0;
}

static const char *norm_arm(const char *arm) {
    if (arm && arm[0] && strcmp(arm, "nvfp4") == 0)
        return "nvfp4";
    return "int8";
}

/* Append one violation; returns 0 if buffer exhausted. */
static int add_violation(char *err, size_t err_size, size_t *used,
                         const char *msg) {
    size_t n;
    if (!err || err_size == 0 || !used)
        return 0;
    n = strlen(msg);
    if (*used + n + 2 >= err_size)
        return 0;
    if (*used) {
        err[(*used)++] = ';';
        err[(*used)++] = ' ';
    }
    memcpy(err + *used, msg, n);
    *used += n;
    err[*used] = '\0';
    return 1;
}

static int check_one(const h3_env_entry *e, const char *arm, char *msg,
                     size_t msg_size) {
    const char *val = getenv(e->name);
    const char *rule = e->rule ? e->rule : "info";
    if (!strcmp(rule, "info"))
        return 1;
    if (!strcmp(rule, "unset")) {
        if (is_set(val)) {
            snprintf(msg, msg_size, "%s must be unset for P0 visual", e->name);
            return 0;
        }
        return 1;
    }
    if (!strcmp(rule, "unset_or_0")) {
        if (!is_set(val) || (val[0] == '0' && val[1] == '\0'))
            return 1;
        snprintf(msg, msg_size, "%s must be unset or 0 for P0 visual", e->name);
        return 0;
    }
    if (!strcmp(rule, "arm")) {
        if (!strcmp(arm, "int8")) {
            if (is_on(val)) {
                snprintf(msg, msg_size, "%s=1 but --arm int8", e->name);
                return 0;
            }
        } else if (!strcmp(arm, "nvfp4")) {
            if (!is_on(val)) {
                snprintf(msg, msg_size, "%s must be 1 for --arm nvfp4", e->name);
                return 0;
            }
        }
        return 1;
    }
    if (!strcmp(rule, "arm_file")) {
        if (!strcmp(arm, "int8")) {
            if (is_set(val)) {
                snprintf(msg, msg_size, "%s should be unset for --arm int8",
                         e->name);
                return 0;
            }
        } else if (!strcmp(arm, "nvfp4")) {
            if (!is_set(val)) {
                snprintf(msg, msg_size, "%s should be set for --arm nvfp4",
                         e->name);
                return 0;
            }
        }
        return 1;
    }
    snprintf(msg, msg_size, "unknown rule %s for %s", rule, e->name);
    return 0;
}

int h3_env_contract_expect_p0_visual(void) {
    const char *e = getenv("H3_EXPECT_P0_VISUAL");
    return is_on(e) ? 1 : 0;
}

int h3_env_contract_check_p0(const char *arm, char *err, size_t err_size) {
    size_t i, used = 0;
    char one[192];
    const char *a = norm_arm(arm);
    if (err && err_size)
        err[0] = '\0';
    for (i = 0; i < sizeof(k_contract) / sizeof(k_contract[0]); i++) {
        if (!check_one(&k_contract[i], a, one, sizeof(one))) {
            if (!add_violation(err, err_size, &used, one) && err && err_size)
                break;
        }
    }
    return used == 0 ? 1 : 0;
}

void h3_env_contract_fprint(FILE *out, const h3_env_contract_cli *cli) {
    size_t i;
    char trunc[97];
    const char *arm = norm_arm(cli ? cli->arm : NULL);
    int expect_p0 = h3_env_contract_expect_p0_visual();
    const char *th = getenv("H3_TEXT_H_LOAD");
    int te_live = !(th && th[0]);
    if (!out)
        out = stdout;

    fprintf(out,
            "# H3_ENV_CONTRACT  arm=%s  expect_p0=%d\n",
            arm, expect_p0);
    for (i = 0; i < sizeof(k_contract) / sizeof(k_contract[0]); i++) {
        const char *raw = getenv(k_contract[i].name);
        const char *fv = fmt_val(raw, trunc, sizeof(trunc));
        const char *rule =
            k_contract[i].rule ? k_contract[i].rule : "info";
        fprintf(out, "env %s=%s  # %s rule=%s\n", k_contract[i].name, fv,
                k_contract[i].role, rule);
    }
    fprintf(out,
            "cli seed=0x%llx skip_audio=%d fewstep_preset=%s "
            "fewstep_contract=%s sampler=%s sdpa=%s sdpa_graph=%d "
            "te_path=%s\n",
            (unsigned long long)(cli ? cli->seed : 0ull),
            cli ? cli->skip_audio : 0,
            (cli && cli->fewstep_preset && cli->fewstep_preset[0])
                ? cli->fewstep_preset
                : "none",
            (cli && cli->fewstep_contract && cli->fewstep_contract[0])
                ? cli->fewstep_contract
                : "none",
            (cli && cli->sampler && cli->sampler[0]) ? cli->sampler : "none",
            (cli && cli->sdpa && cli->sdpa[0]) ? cli->sdpa : "unset",
            sdpa_graph_wanted(), te_live ? "live" : "H3_TEXT_H_LOAD");
    if (expect_p0) {
        char err[512];
        if (h3_env_contract_check_p0(arm, err, sizeof(err)))
            fprintf(out, "# PROTO_ENV_OK arm=%s\n", arm);
        else
            fprintf(out, "# PROTO_ENV_FAIL arm=%s reason=%s\n", arm,
                    err[0] ? err : "unknown");
    }
    fflush(out);
}
