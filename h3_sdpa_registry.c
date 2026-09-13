#include "h3_sdpa.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const h3_sdpa_backend_desc g_table[] = {
    {H3_SDPA_ID_CUDNN, "cudnn", 1, 0, 0},
    {H3_SDPA_ID_SAGE3, "sage3", 0, 1, 0},
    {H3_SDPA_ID_SOL_CUTE, "sol_cute", 0, 1, 0},
    {H3_SDPA_ID_FLASH, "flash", 0, 0, 0},
    {H3_SDPA_ID_TC, "tc", 0, 0, 0},
    {H3_SDPA_ID_SAGE2, "sage2", 0, 0, 1},
    {H3_SDPA_ID_SOL, "sol", 0, 0, 1},
};

static int g_cached_valid;
static h3_sdpa_backend_id g_cached_id = H3_SDPA_ID_CUDNN;
static int g_logged;

const h3_sdpa_backend_desc *h3_sdpa_backend_table(size_t *count) {
    if (count) *count = sizeof(g_table) / sizeof(g_table[0]);
    return g_table;
}

const char *h3_sdpa_backend_id_name(h3_sdpa_backend_id id) {
    size_t i, n = sizeof(g_table) / sizeof(g_table[0]);
    for (i = 0; i < n; i++) {
        if (g_table[i].id == id) return g_table[i].name;
    }
    return "unknown";
}

const char *h3_sdpa_normalize_name(const char *request) {
    if (!request || !request[0] || !strcmp(request, "auto")) return "auto";
    if (!strcmp(request, "cudnn") || !strcmp(request, "fused")) return "cudnn";
    if (!strcmp(request, "sage3") || !strcmp(request, "sageattn3") ||
        !strcmp(request, "sage_attn3") || !strcmp(request, "sageattention3"))
        return "sage3";
    if (!strcmp(request, "sol_cute") || !strcmp(request, "sol-cute") ||
        !strcmp(request, "sol-engine") || !strcmp(request, "cute_sm89"))
        return "sol_cute";
    if (!strcmp(request, "flash") || !strcmp(request, "cuda_flash"))
        return "flash";
    if (!strcmp(request, "tc") || !strcmp(request, "wmma")) return "tc";
    if (!strcmp(request, "sage2") || !strcmp(request, "sage2++") ||
        !strcmp(request, "sage"))
        return "sage2";
    if (!strcmp(request, "sol") || !strcmp(request, "sol-attn") ||
        !strcmp(request, "solattn"))
        return "sol";
    return request;
}

static h3_sdpa_backend_id h3_sdpa_id_from_norm(const char *n) {
    if (!strcmp(n, "cudnn") || !strcmp(n, "auto")) return H3_SDPA_ID_CUDNN;
    if (!strcmp(n, "sage3")) return H3_SDPA_ID_SAGE3;
    if (!strcmp(n, "sol_cute")) return H3_SDPA_ID_SOL_CUTE;
    if (!strcmp(n, "flash")) return H3_SDPA_ID_FLASH;
    if (!strcmp(n, "tc")) return H3_SDPA_ID_TC;
    if (!strcmp(n, "sage2")) return H3_SDPA_ID_SAGE2;
    if (!strcmp(n, "sol")) return H3_SDPA_ID_SOL;
    return H3_SDPA_ID_UNKNOWN;
}

int h3_sdpa_resolve(const char *request, const h3_cuda_device_caps *caps,
                    h3_sdpa_backend_id *out_id, int *available, char *reason,
                    size_t reason_size) {
    const char *norm = h3_sdpa_normalize_name(request);
    h3_sdpa_backend_id id;
    int avail = 1;

    if (!out_id) return 0;
    id = h3_sdpa_id_from_norm(norm);
    *out_id = id;
    if (available) *available = 1;

    if (id == H3_SDPA_ID_UNKNOWN) {
        if (available) *available = 0;
        if (reason && reason_size)
            snprintf(reason, reason_size,
                     "unknown SDPA backend '%s' (want cudnn|sage3|sol_cute)",
                     request ? request : "");
        return 0;
    }

    if (!strcmp(norm, "auto")) {
        if (reason && reason_size)
            snprintf(reason, reason_size,
                     "auto→cudnn (Tier-0 quality default; never auto sage3/sol)");
        if (caps && !caps->has_cudnn_sdpa) {
            avail = 0;
            if (available) *available = 0;
            if (reason && reason_size)
                snprintf(reason, reason_size,
                         "auto→cudnn but build lacks H3_HAVE_CUDNN");
            return 0;
        }
        if (available) *available = 1;
        return 1;
    }

    switch (id) {
    case H3_SDPA_ID_CUDNN:
        if (caps && !caps->has_cudnn_sdpa) {
            avail = 0;
            if (reason && reason_size)
                snprintf(reason, reason_size, "cudnn requested but not built");
        } else if (reason && reason_size) {
            snprintf(reason, reason_size, "Tier-0 quality default (cudnn)");
        }
        break;
    case H3_SDPA_ID_SAGE3:
        if (caps && !caps->has_sage3_plugin) {
            avail = 0;
            if (reason && reason_size)
                snprintf(reason, reason_size,
                         "sage3 Tier-1: set H3_SAGE3_LIB to libh3_sage3.so "
                         "(tried %s); also H3_SAGE3_ROOT",
                         caps->sage3_lib_tried[0] ? caps->sage3_lib_tried
                                                  : "libh3_sage3.so");
        } else if (caps && !caps->has_fp4_tc) {
            /* Still allow try (plugin may refuse); warn in reason. */
            if (reason && reason_size)
                snprintf(reason, reason_size,
                         "sage3 Tier-1 on non-FP4 SM (sm_%d%d); expect fail/"
                         "fallback — prefer 5090/sm_120",
                         caps->info.compute_major, caps->info.compute_minor);
        } else if (reason && reason_size) {
            snprintf(reason, reason_size,
                     "Tier-1 sage3 (thu-ml FP4); quality opt-in, not default");
        }
        break;
    case H3_SDPA_ID_SOL_CUTE:
        if (caps && !caps->has_sol_cute_plugin) {
            avail = 0;
            if (reason && reason_size)
                snprintf(reason, reason_size,
                         "sol_cute Tier-1: set H3_SOL_CUTE_LIB to "
                         "libh3_sol_cute.so (tried %s); also H3_SOL_ATTN_ROOT",
                         caps->sol_cute_lib_tried[0] ? caps->sol_cute_lib_tried
                                                     : "libh3_sol_cute.so");
        } else if (reason && reason_size) {
            snprintf(reason, reason_size,
                     "Tier-1 sol_cute (official CuTe Sol-Attn); visual gate "
                     "open — not quality default");
        }
        break;
    case H3_SDPA_ID_SAGE2:
        if (reason && reason_size)
            snprintf(reason, reason_size,
                     "FORBIDDEN as product: in-repo sage2 reference (not "
                     "Sage3)");
        break;
    case H3_SDPA_ID_SOL:
        if (reason && reason_size)
            snprintf(reason, reason_size,
                     "FORBIDDEN as product: nested-loop sol (not sol_cute)");
        break;
    case H3_SDPA_ID_FLASH:
        if (reason && reason_size)
            snprintf(reason, reason_size, "legacy hand-roll flash (debug)");
        break;
    case H3_SDPA_ID_TC:
        if (reason && reason_size)
            snprintf(reason, reason_size, "legacy WMMA tc (often slower)");
        break;
    default:
        break;
    }

    if (available) *available = avail;
    return avail ? 1 : 0;
}

void h3_sdpa_cache_invalidate(void) {
    g_cached_valid = 0;
    g_logged = 0;
}

h3_sdpa_backend_id h3_sdpa_cached_id(void) {
    const char *e;
    h3_sdpa_backend_id id;
    int avail = 1;
    char reason[256];
    if (g_cached_valid) return g_cached_id;
    e = getenv("H3_CUDA_SDPA_BACKEND");
    if (!h3_sdpa_resolve(e && e[0] ? e : "auto", NULL, &id, &avail, reason,
                         sizeof(reason))) {
        /* Unknown → safe cudnn */
        id = H3_SDPA_ID_CUDNN;
    }
    g_cached_id = id;
    g_cached_valid = 1;
    return g_cached_id;
}

void h3_sdpa_log_selection(const h3_cuda_device_caps *caps) {
    const char *e = getenv("H3_CUDA_SDPA_BACKEND");
    h3_sdpa_backend_id id;
    int avail = 1;
    char reason[320];
    char caps_line[384];
    if (g_logged) return;
    g_logged = 1;
    if (!h3_sdpa_resolve(e && e[0] ? e : "auto", caps, &id, &avail, reason,
                         sizeof(reason))) {
        /* keep id from resolve (may be UNKNOWN→treat as cudnn for log) */
        if (id == H3_SDPA_ID_UNKNOWN) id = H3_SDPA_ID_CUDNN;
    }
    g_cached_id = id;
    g_cached_valid = 1;
    if (caps)
        h3_cuda_caps_format(caps, caps_line, sizeof(caps_line));
    else
        snprintf(caps_line, sizeof(caps_line), "(caps not queried)");
    fprintf(stderr,
            "[h3] sdpa_select request=%s resolved=%s available=%d reason=%s | "
            "%s\n",
            (e && e[0]) ? e : "auto", h3_sdpa_backend_id_name(id), avail,
            reason, caps_line);
    fflush(stderr);
    if (!avail && (id == H3_SDPA_ID_SAGE3 || id == H3_SDPA_ID_SOL_CUTE)) {
        fprintf(stderr,
                "[h3] sdpa_select: Tier-1 backend missing plugin — will not "
                "silently use nested-loop/sage2 fake; expect fallback to "
                "cudnn or hard fail on first hit\n");
        fflush(stderr);
    }
    if (id == H3_SDPA_ID_SAGE2 || id == H3_SDPA_ID_SOL) {
        fprintf(stderr,
                "[h3] sdpa_select: WARNING %s is forbidden as a product "
                "backend (reference / fake nested-loop)\n",
                h3_sdpa_backend_id_name(id));
        fflush(stderr);
    }
}
