#include "h3_ulysses.h"

#include <stdio.h>
#include <string.h>

static int u_fail(char *err, size_t err_size, const char *msg) {
    if (err && err_size)
        snprintf(err, err_size, "%s", msg ? msg : "ulysses error");
    return 0;
}

int h3_ulysses_validate(int heads, int packed_align, int tp, int ulysses,
                        int ring, char *err, size_t err_size) {
    if (err && err_size)
        err[0] = '\0';
    if (tp < 1 || ulysses < 1 || ring < 1 || heads < 1 || packed_align < 1)
        return u_fail(err, err_size, "ulysses: non-positive dim");
    if (ulysses != 1 && ulysses != 2 && ulysses != 4 && ulysses != 8) {
        char buf[160];
        snprintf(buf, sizeof(buf),
                 "ulysses=%d illegal (legal {1,2,4,8}; packed align)", ulysses);
        return u_fail(err, err_size, buf);
    }
    if (heads % tp != 0)
        return u_fail(err, err_size, "ulysses: heads % tp != 0");
    if ((heads / tp) % ulysses != 0)
        return u_fail(err, err_size, "ulysses: (heads/tp) % U != 0");
    if (packed_align % (ulysses * ring) != 0) {
        char buf[160];
        snprintf(buf, sizeof(buf),
                 "ulysses: packed align %d %% (U=%d * ring=%d) != 0",
                 packed_align, ulysses, ring);
        return u_fail(err, err_size, buf);
    }
    return 1;
}

size_t h3_ulysses_pad_seq(size_t seq, int align) {
    size_t a;
    if (align < 1)
        return seq;
    a = (size_t)align;
    return (seq + a - 1u) / a * a;
}

size_t h3_ulysses_local_rows(size_t seq_pad, int ulysses, int rank) {
    size_t u, base, rem;
    if (ulysses < 1)
        return seq_pad;
    u = (size_t)ulysses;
    base = seq_pad / u;
    rem = seq_pad % u;
    if (rank < 0 || (size_t)rank >= u)
        return 0;
    return base + ((size_t)rank < rem ? 1u : 0u);
}
