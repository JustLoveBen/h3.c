/* tests/test_h3_tae.c — TAE opt-in contract + fixture decode (host). */
#include "h3_tae.h"

#include <stdint.h>
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

typedef struct {
    const char *name;
    int oc, ic, kh, kw;
    int has_b;
} fix_t;

static int write_u64_le(FILE *fp, uint64_t v) {
    unsigned char b[8];
    int i;
    for (i = 0; i < 8; i++)
        b[i] = (unsigned char)((v >> (8 * i)) & 0xff);
    return fwrite(b, 1, 8, fp) == 8;
}

static int write_fixture(const char *path) {
    /* Minimal TAEHV taeh3-shaped zeros. Enough for open + tiny decode. */
    static const fix_t spec[] = {
        {"decoder.1.weight", 256, 24, 3, 3, 1},
        {"decoder.3.conv.0.weight", 256, 512, 3, 3, 1},
        {"decoder.3.conv.2.weight", 256, 256, 3, 3, 1},
        {"decoder.3.conv.4.weight", 256, 256, 3, 3, 1},
        {"decoder.4.conv.0.weight", 256, 512, 3, 3, 1},
        {"decoder.4.conv.2.weight", 256, 256, 3, 3, 1},
        {"decoder.4.conv.4.weight", 256, 256, 3, 3, 1},
        {"decoder.5.conv.0.weight", 256, 512, 3, 3, 1},
        {"decoder.5.conv.2.weight", 256, 256, 3, 3, 1},
        {"decoder.5.conv.4.weight", 256, 256, 3, 3, 1},
        {"decoder.7.conv.weight", 256, 256, 1, 1, 0},
        {"decoder.8.weight", 128, 256, 3, 3, 0},
        {"decoder.9.conv.0.weight", 128, 256, 3, 3, 1},
        {"decoder.9.conv.2.weight", 128, 128, 3, 3, 1},
        {"decoder.9.conv.4.weight", 128, 128, 3, 3, 1},
        {"decoder.10.conv.0.weight", 128, 256, 3, 3, 1},
        {"decoder.10.conv.2.weight", 128, 128, 3, 3, 1},
        {"decoder.10.conv.4.weight", 128, 128, 3, 3, 1},
        {"decoder.11.conv.0.weight", 128, 256, 3, 3, 1},
        {"decoder.11.conv.2.weight", 128, 128, 3, 3, 1},
        {"decoder.11.conv.4.weight", 128, 128, 3, 3, 1},
        {"decoder.13.conv.weight", 256, 128, 1, 1, 0},
        {"decoder.14.weight", 64, 128, 3, 3, 0},
        {"decoder.15.conv.0.weight", 64, 128, 3, 3, 1},
        {"decoder.15.conv.2.weight", 64, 64, 3, 3, 1},
        {"decoder.15.conv.4.weight", 64, 64, 3, 3, 1},
        {"decoder.16.conv.0.weight", 64, 128, 3, 3, 1},
        {"decoder.16.conv.2.weight", 64, 64, 3, 3, 1},
        {"decoder.16.conv.4.weight", 64, 64, 3, 3, 1},
        {"decoder.17.conv.0.weight", 64, 128, 3, 3, 1},
        {"decoder.17.conv.2.weight", 64, 64, 3, 3, 1},
        {"decoder.17.conv.4.weight", 64, 64, 3, 3, 1},
        {"decoder.19.conv.weight", 128, 64, 1, 1, 0},
        {"decoder.20.weight", 64, 64, 3, 3, 0},
        {"decoder.22.weight", 12, 64, 3, 3, 1},
    };
    const int nspec = (int)(sizeof(spec) / sizeof(spec[0]));
    char json[16384];
    size_t off = 0;
    int i, pos;
    FILE *fp;
    char *zeros = NULL;
    size_t zcap = 0;

    pos = snprintf(json, sizeof(json), "{");
    for (i = 0; i < nspec; i++) {
        size_t ne =
            (size_t)spec[i].oc * (size_t)spec[i].ic * (size_t)spec[i].kh *
            (size_t)spec[i].kw;
        size_t nb = ne * 4;
        pos += snprintf(json + pos, sizeof(json) - (size_t)pos,
                        "%s\"%s\":{\"dtype\":\"F32\",\"shape\":[%d,%d,%d,%d],"
                        "\"data_offsets\":[%zu,%zu]}",
                        i ? "," : "", spec[i].name, spec[i].oc, spec[i].ic,
                        spec[i].kh, spec[i].kw, off, off + nb);
        off += nb;
        if (spec[i].has_b) {
            char bname[80];
            size_t bb = (size_t)spec[i].oc * 4;
            snprintf(bname, sizeof(bname), "%s", spec[i].name);
            /* weight → bias */
            {
                char *dot = strrchr(bname, '.');
                if (dot && !strcmp(dot, ".weight"))
                    memcpy(dot, ".bias", 6);
            }
            pos += snprintf(json + pos, sizeof(json) - (size_t)pos,
                            ",\"%s\":{\"dtype\":\"F32\",\"shape\":[%d],"
                            "\"data_offsets\":[%zu,%zu]}",
                            bname, spec[i].oc, off, off + bb);
            off += bb;
        }
        if (pos < 0 || (size_t)pos >= sizeof(json) - 8)
            return 0;
    }
    pos += snprintf(json + pos, sizeof(json) - (size_t)pos, "}");
    fp = fopen(path, "wb");
    if (!fp) return 0;
    if (!write_u64_le(fp, (uint64_t)pos) ||
        fwrite(json, 1, (size_t)pos, fp) != (size_t)pos) {
        fclose(fp);
        return 0;
    }
    zcap = 256 * 512 * 9 * 4;
    zeros = (char *)calloc(zcap, 1);
    if (!zeros) {
        fclose(fp);
        return 0;
    }
    {
        size_t written = 0;
        while (written < off) {
            size_t chunk = off - written;
            if (chunk > zcap) chunk = zcap;
            if (fwrite(zeros, 1, chunk, fp) != chunk) {
                free(zeros);
                fclose(fp);
                return 0;
            }
            written += chunk;
        }
    }
    free(zeros);
    fclose(fp);
    return 1;
}

int main(void) {
    char err[256];
    char fix[128];
    void *ctx = NULL;
    h3_vae_int8_frames fr;
    float latent[24 * 1 * 2 * 2];
    int i;

    expect(h3_tae_allow_opt_in(0, "official", NULL, err, sizeof err) == 1,
           "official gold ok");
    expect(h3_tae_allow_opt_in(0, "tae", "/tmp/x.safetensors", err,
                              sizeof err) == 0,
           "gold+tae fail");
    expect(strstr(err, "不是金线") != NULL, "gold err 金线");
    expect(h3_tae_allow_opt_in(1, "tae", NULL, err, sizeof err) == 0,
           "turbo tae no path");
    expect(strstr(err, "不静默") != NULL, "no silent official");
    expect(h3_tae_allow_opt_in(1, "tae", "/tmp/fake.safetensors", err,
                              sizeof err) == 1,
           "turbo4 explicit tae allowed");
    expect(h3_tae_open(NULL, &ctx, err, sizeof err) == 0, "open null");
    expect(h3_tae_open("/no/such/tae.safetensors", &ctx, err, sizeof err) == 0,
           "open missing");

    snprintf(fix, sizeof(fix), "/tmp/h3_tae_fix_%d.safetensors", (int)getpid());
    expect(write_fixture(fix) == 1, "write fixture");
    expect(h3_tae_open(fix, &ctx, err, sizeof err) == 1, "open fixture");
    if (ctx) {
        memset(latent, 0, sizeof(latent));
        for (i = 0; i < 24 * 2 * 2; i++)
            latent[i] = 0.01f * (float)((i % 7) - 3);
        memset(&fr, 0, sizeof(fr));
        expect(h3_tae_decode_frames(ctx, latent, 1, 2, 2, 1, &fr, err,
                                    sizeof err) == 1,
               "tiny decode");
        expect(fr.frames >= 1, "frames");
        expect(fr.height == 32 && fr.width == 32, "spatial 16x (2→32)");
        expect(fr.rgb != NULL, "rgb");
        if (fr.rgb) {
            int finite = 1;
            size_t n = (size_t)fr.frames * 32 * 32 * 3;
            size_t k;
            for (k = 0; k < n; k++)
                if (fr.rgb[k] != fr.rgb[k]) finite = 0;
            expect(finite, "finite rgb");
            free(fr.rgb);
        }
        h3_tae_close(ctx);
    } else {
        expect(0, "ctx after open");
    }
    unlink(fix);
    return fails ? 1 : (printf("test_h3_tae: PASS\n"), 0);
}
