/* Host-only: dump frozen fewstep contract table as JSON.
 * Usage:
 *   ./dump_fewstep_contracts              # pretty → stdout
 *   ./dump_fewstep_contracts --compact
 *   ./dump_fewstep_contracts --check PATH  # compare to fixture (ignore ws)
 */
#include "h3_fewstep_contract.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void strip_ws(const char *in, char *out, size_t out_sz) {
    size_t j = 0;
    for (; *in && j + 1 < out_sz; in++) {
        if (!isspace((unsigned char)*in))
            out[j++] = *in;
    }
    out[j] = '\0';
}

static char *slurp(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    long sz;
    char *buf;
    size_t n;
    if (!f)
        return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    sz = ftell(f);
    if (sz < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[n] = '\0';
    if (n_out)
        *n_out = n;
    return buf;
}

static char *dump_to_string(int pretty, size_t *n_out) {
    char tmp[] = "/tmp/h3_fewstep_contracts_XXXXXX";
    int fd = mkstemp(tmp);
    FILE *tf;
    char *got;
    long sz;
    size_t n;
    if (fd < 0)
        return NULL;
    tf = fdopen(fd, "w+");
    if (!tf) {
        close(fd);
        unlink(tmp);
        return NULL;
    }
    h3_fewstep_contracts_json_fprint(tf, pretty);
    fflush(tf);
    if (fseek(tf, 0, SEEK_END) != 0) {
        fclose(tf);
        unlink(tmp);
        return NULL;
    }
    sz = ftell(tf);
    rewind(tf);
    if (sz < 0) {
        fclose(tf);
        unlink(tmp);
        return NULL;
    }
    got = (char *)malloc((size_t)sz + 1);
    if (!got) {
        fclose(tf);
        unlink(tmp);
        return NULL;
    }
    n = fread(got, 1, (size_t)sz, tf);
    got[n] = '\0';
    fclose(tf);
    unlink(tmp);
    if (n_out)
        *n_out = n;
    return got;
}

int main(int argc, char **argv) {
    int compact = 0;
    const char *check_path = NULL;
    int i;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--compact"))
            compact = 1;
        else if (!strcmp(argv[i], "--check") && i + 1 < argc)
            check_path = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr, "usage: %s [--compact] [--check PATH]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }
    if (check_path) {
        char *got = dump_to_string(1, NULL);
        char *want = slurp(check_path, NULL);
        char *got_s;
        char *want_s;
        size_t gl, wl;
        if (!got) {
            fprintf(stderr, "FAIL: cannot dump contracts\n");
            return 1;
        }
        if (!want) {
            fprintf(stderr, "FAIL: cannot read %s\n", check_path);
            free(got);
            return 1;
        }
        gl = strlen(got);
        wl = strlen(want);
        got_s = (char *)malloc(gl + 1);
        want_s = (char *)malloc(wl + 1);
        if (!got_s || !want_s) {
            free(got);
            free(want);
            free(got_s);
            free(want_s);
            return 1;
        }
        strip_ws(got, got_s, gl + 1);
        strip_ws(want, want_s, wl + 1);
        if (strcmp(got_s, want_s) != 0) {
            fprintf(stderr,
                    "FAIL: dump != fixture %s (whitespace-insensitive)\n",
                    check_path);
            free(got);
            free(want);
            free(got_s);
            free(want_s);
            return 1;
        }
        printf("ok dump≡fixture %s\n", check_path);
        free(got);
        free(want);
        free(got_s);
        free(want_s);
        return 0;
    }
    h3_fewstep_contracts_json_fprint(stdout, compact ? 0 : 1);
    return 0;
}
