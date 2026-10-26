/* tests/test_h3_te_cache.c */
#include "h3_te_cache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static int fails;
static void expect(int c, const char *m) {
    if (!c) {
        fprintf(stderr, "FAIL %s\n", m);
        fails++;
    }
}
int main(void) {
    int32_t ids[4] = {11, 22, 33, 44};
    char hex[32], hex2[32], err[128];
    char dir[] = "/tmp/h3_te_cache_test_XXXXXX";
    float *got = NULL;
    float emb[4 * H3_TE_CACHE_TEXT_DIM];
    size_t i;
    expect(h3_te_cache_key(ids, 4, 50, "/no/such/te", hex, sizeof hex, err,
                           sizeof err) == 1,
           "key");
    expect(strlen(hex) == 16, "hex len");
    ids[0] = 12;
    expect(h3_te_cache_key(ids, 4, 50, "/no/such/te", hex2, sizeof hex2, err,
                           sizeof err) == 1,
           "key2");
    expect(strcmp(hex, hex2) != 0, "ids change key");
    ids[0] = 11;
    expect(h3_te_cache_key(ids, 4, 49, "/no/such/te", hex2, sizeof hex2, err,
                           sizeof err) == 1,
           "key layers");
    expect(strcmp(hex, hex2) != 0, "layers change key");
    if (!mkdtemp(dir)) {
        fprintf(stderr, "FAIL mkdtemp\n");
        return 1;
    }
    for (i = 0; i < sizeof(emb) / sizeof(emb[0]); i++)
        emb[i] = (float)i * 0.001f;
    expect(h3_te_cache_save(dir, hex, emb, 4, 50, err, sizeof err) == 1, "save");
    expect(h3_te_cache_load(dir, hex, 4, 50, &got, err, sizeof err) == 1, "load");
    expect(got && memcmp(got, emb, sizeof(emb)) == 0, "roundtrip");
    free(got);
    got = NULL;
    expect(h3_te_cache_load(dir, hex, 4, 49, &got, err, sizeof err) == 0,
           "layers mismatch");
    free(got);
    {
        char cmd[256];
        snprintf(cmd, sizeof cmd, "rm -rf %s", dir);
        (void)system(cmd);
    }
    return fails ? 1 : (printf("test_h3_te_cache: PASS\n"), 0);
}
