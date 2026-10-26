/* Host-only: C064 DeferredRelease whitelist table sanity. */
#include "h3_lifecycle_whitelist.h"

#include <stdio.h>
#include <string.h>

static int g_fail;

static void expect_ok(const char *name, int cond) {
    if (!cond) {
        fprintf(stderr, "FAIL %s\n", name);
        g_fail = 1;
    }
}

int main(void) {
    int i, saw_te = 0, saw_nvfp4 = 0, saw_vae_dev = 0;
    char buf[8192];
    FILE *mem;

    expect_ok("n>=8", H3_LIFECYCLE_WHITELIST_N >= 8);
    expect_ok("sync_n>=3", H3_LIFECYCLE_ALLOWED_SYNC_N >= 3);

    for (i = 0; i < H3_LIFECYCLE_WHITELIST_N; i++) {
        const h3_lifecycle_whitelist_row *r = &k_h3_lifecycle_whitelist[i];
        expect_ok("object", r->object && r->object[0]);
        expect_ok("produce", r->produce && r->produce[0]);
        expect_ok("release", r->release && r->release[0]);
        expect_ok("event", r->event_dep && r->event_dep[0]);
        expect_ok("overlap_range",
                  r->safe_overlap == 0 || r->safe_overlap == 1 ||
                      r->safe_overlap == -1);
        if (strstr(r->object, "text_emb")) {
            saw_te = 1;
            expect_ok("text_emb no defer", r->safe_overlap == 0);
        }
        if (strstr(r->object, "NVFP4")) {
            saw_nvfp4 = 1;
            expect_ok("nvfp4 no overlap", r->safe_overlap == 0);
        }
        if (strstr(r->object, "device-W")) {
            saw_vae_dev = 1;
            expect_ok("vae device gated", r->safe_overlap == -1);
        }
    }
    expect_ok("saw text_emb", saw_te);
    expect_ok("saw NVFP4 host_stage", saw_nvfp4);
    expect_ok("saw VAE device-W", saw_vae_dev);

    mem = tmpfile();
    expect_ok("tmpfile", mem != NULL);
    if (mem) {
        size_t n;
        h3_lifecycle_whitelist_fprint(mem);
        rewind(mem);
        n = fread(buf, 1, sizeof(buf) - 1, mem);
        buf[n] = '\0';
        fclose(mem);
        expect_ok("marker", strstr(buf, "# H3_LIFECYCLE_WHITELIST") != NULL);
        expect_ok("row", strstr(buf, "lifecycle_wl:") != NULL);
        expect_ok("sync", strstr(buf, "lifecycle_sync_ok:") != NULL);
    }

    /* Trace off by default */
    unsetenv("H3_LIFECYCLE_TRACE");
    expect_ok("trace off", h3_lifecycle_trace_wanted() == 0);
    setenv("H3_LIFECYCLE_TRACE", "1", 1);
    expect_ok("trace on", h3_lifecycle_trace_wanted() == 1);
    unsetenv("H3_LIFECYCLE_TRACE");

    if (g_fail) {
        fprintf(stderr, "test_h3_lifecycle_whitelist: FAILED\n");
        return 1;
    }
    printf("ok test_h3_lifecycle_whitelist\n");
    return 0;
}
