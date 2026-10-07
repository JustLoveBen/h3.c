#include "h3_lora.h"

#include "h3_safetensors.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int lora_fail(char *err, size_t err_size, const char *msg) {
    if (err && err_size)
        snprintf(err, err_size, "%s", msg ? msg : "lora error");
    return 0;
}

int h3_lora_name_is_vsa(const char *tensor_name) {
    if (!tensor_name)
        return 0;
    if (strstr(tensor_name, "vsa") || strstr(tensor_name, "VSA"))
        return 1;
    if (strstr(tensor_name, "to_gate_compress"))
        return 1;
    if (strstr(tensor_name, "sparse"))
        return 1;
    if (strstr(tensor_name, "sm100a") || strstr(tensor_name, "sm_100"))
        return 1;
    return 0;
}

static int parse_block_index(const char *name, unsigned *block) {
    const char *p;
    char *end = NULL;
    unsigned long v;

    p = strstr(name, "blocks.");
    if (p) {
        p += 7;
        if (!isdigit((unsigned char)*p))
            return 0;
        v = strtoul(p, &end, 10);
        if (!end || (*end != '.' && *end != '_'))
            return 0;
        if (v >= (unsigned long)H3_LORA_N_BLOCKS)
            return 0;
        *block = (unsigned)v;
        return 1;
    }
    p = strstr(name, "blocks_");
    if (p) {
        p += 7;
        if (!isdigit((unsigned char)*p))
            return 0;
        v = strtoul(p, &end, 10);
        if (!end || (*end != '_' && *end != '.'))
            return 0;
        if (v >= (unsigned long)H3_LORA_N_BLOCKS)
            return 0;
        *block = (unsigned)v;
        return 1;
    }
    return 0;
}

static const char *parse_suffix(const char *name) {
    /* More specific tokens first. */
    if (strstr(name, "qkv_proj") || strstr(name, "to_qkv") ||
        strstr(name, "attn_qkv") || strstr(name, "attn.to_qkv"))
        return "qkv";
    if (strstr(name, "out_proj") || strstr(name, "to_out") ||
        strstr(name, "attn_out"))
        return "out";
    if (strstr(name, "mlp.fc1") || strstr(name, "mlp_fc1") ||
        strstr(name, "ff.net.0") || strstr(name, "ff_net_0") ||
        strstr(name, "mlp.0") || strstr(name, "mlp_0"))
        return "fc1";
    if (strstr(name, "mlp.fc2") || strstr(name, "mlp_fc2") ||
        strstr(name, "ff.net.2") || strstr(name, "ff_net_2") ||
        strstr(name, "mlp.2") || strstr(name, "mlp_2"))
        return "fc2";
    return NULL;
}

static int parse_ab(const char *name, int *is_a) {
    const char *p;
    if (strstr(name, "lora_A") || strstr(name, "lora_down")) {
        *is_a = 1;
        return 1;
    }
    if (strstr(name, "lora_B") || strstr(name, "lora_up")) {
        *is_a = 0;
        return 1;
    }
    /* TaoMate: *.lora_a / *.lora_b — do not match lora_alpha. */
    p = strstr(name, ".lora_a");
    if (p && (p[7] == '\0' || p[7] == '.')) {
        *is_a = 1;
        return 1;
    }
    p = strstr(name, ".lora_b");
    if (p && (p[7] == '\0' || p[7] == '.')) {
        *is_a = 0;
        return 1;
    }
    return 0;
}

int h3_lora_parse_name(const char *name, unsigned *block, const char **suffix,
                       int *is_a, char *err, size_t err_size) {
    unsigned b = 0;
    const char *suf;
    int ab = 0;

    if (err && err_size)
        err[0] = '\0';
    if (!name || !block || !suffix || !is_a)
        return lora_fail(err, err_size, "lora parse: null args");
    if (h3_lora_name_is_vsa(name))
        return lora_fail(err, err_size, "lora parse: vsa tensor");
    /* larryvrh v4 also ships token_refiner.blocks.* with the same linear
     * suffixes. Those are not DiT blocks; matching them as blocks.N collides
     * with the 50-block backbone ("lora duplicate A"). */
    if (strstr(name, "token_refiner"))
        return lora_fail(err, err_size, "lora parse: token_refiner");
    if (!parse_ab(name, &ab))
        return lora_fail(err, err_size, "lora parse: not A/B");
    if (!parse_block_index(name, &b))
        return lora_fail(err, err_size, "lora parse: no block index");
    suf = parse_suffix(name);
    if (!suf)
        return lora_fail(err, err_size, "lora parse: unknown linear suffix");
    *block = b;
    *suffix = suf;
    *is_a = ab;
    return 1;
}

int h3_lora_parse_tensor(const char *name, int *scope, unsigned *block,
                         const char **suffix, int *is_a, char *err,
                         size_t err_size) {
    unsigned b = 0;
    const char *suf;
    int ab = 0;
    const char *p;
    char *end = NULL;
    unsigned long v;

    if (err && err_size)
        err[0] = '\0';
    if (!name || !scope || !block || !suffix || !is_a)
        return lora_fail(err, err_size, "lora parse: null args");
    *scope = H3_LORA_SCOPE_DIT;
    if (h3_lora_name_is_vsa(name))
        return lora_fail(err, err_size, "lora parse: vsa tensor");
    if (!parse_ab(name, &ab))
        return lora_fail(err, err_size, "lora parse: not A/B");
    p = strstr(name, "token_refiner.blocks.");
    if (p) {
        p += strlen("token_refiner.blocks.");
        if (!isdigit((unsigned char)*p))
            return lora_fail(err, err_size, "lora parse: refiner index");
        v = strtoul(p, &end, 10);
        if (!end || (*end != '.' && *end != '_'))
            return lora_fail(err, err_size, "lora parse: refiner index");
        if (v >= (unsigned long)H3_LORA_N_REFINER)
            return lora_fail(err, err_size, "lora parse: refiner index");
        suf = parse_suffix(name);
        if (!suf)
            return lora_fail(err, err_size, "lora parse: unknown linear suffix");
        *scope = H3_LORA_SCOPE_REFINER;
        *block = (unsigned)v;
        *suffix = suf;
        *is_a = ab;
        return 1;
    }
    if (strstr(name, "token_refiner"))
        return lora_fail(err, err_size, "lora parse: token_refiner");
    if (!parse_block_index(name, &b))
        return lora_fail(err, err_size, "lora parse: no block index");
    suf = parse_suffix(name);
    if (!suf)
        return lora_fail(err, err_size, "lora parse: unknown linear suffix");
    *block = b;
    *suffix = suf;
    *is_a = ab;
    return 1;
}

static h3_lora_pair *pair_for(h3_lora_block *blk, const char *suffix) {
    if (!blk || !suffix)
        return NULL;
    if (!strcmp(suffix, "qkv"))
        return &blk->qkv;
    if (!strcmp(suffix, "out"))
        return &blk->out;
    if (!strcmp(suffix, "fc1"))
        return &blk->fc1;
    if (!strcmp(suffix, "fc2"))
        return &blk->fc2;
    return NULL;
}

static int expected_nk(const char *suffix, int *n, int *k) {
    /* DiT W4A8 geometry. */
    if (!strcmp(suffix, "qkv")) {
        *n = 21504;
        *k = 5376;
        return 1;
    }
    if (!strcmp(suffix, "out")) {
        *n = 5376;
        *k = 7168;
        return 1;
    }
    if (!strcmp(suffix, "fc1")) {
        *n = 28672;
        *k = 5376;
        return 1;
    }
    if (!strcmp(suffix, "fc2")) {
        *n = 5376;
        *k = 14336;
        return 1;
    }
    return 0;
}

void h3_lora_free(h3_lora_set *set) {
    unsigned i;
    if (!set)
        return;
    for (i = 0; i < H3_LORA_N_BLOCKS; i++) {
        free(set->blocks[i].qkv.A);
        free(set->blocks[i].qkv.B);
        free(set->blocks[i].out.A);
        free(set->blocks[i].out.B);
        free(set->blocks[i].fc1.A);
        free(set->blocks[i].fc1.B);
        free(set->blocks[i].fc2.A);
        free(set->blocks[i].fc2.B);
    }
    for (i = 0; i < H3_LORA_N_REFINER; i++) {
        free(set->refiner[i].qkv.A);
        free(set->refiner[i].qkv.B);
        free(set->refiner[i].out.A);
        free(set->refiner[i].out.B);
        free(set->refiner[i].fc1.A);
        free(set->refiner[i].fc1.B);
        free(set->refiner[i].fc2.A);
        free(set->refiner[i].fc2.B);
    }
    memset(set, 0, sizeof(*set));
}

static int assign_ab(h3_lora_pair *p, int is_a, float *data, const h3_st_tensor *t,
                     const char *suffix, char *err, size_t err_size) {
    int exp_n = 0, exp_k = 0;
    int r, n, k;
    if (!p || !data || !t)
        return lora_fail(err, err_size, "lora assign: null");
    if (!expected_nk(suffix, &exp_n, &exp_k))
        return lora_fail(err, err_size, "lora assign: bad suffix");
    if (t->ndim != 2) {
        free(data);
        return lora_fail(err, err_size, "lora tensor ndim != 2");
    }
    if (is_a) {
        /* A [r, K] */
        r = (int)t->shape[0];
        k = (int)t->shape[1];
        if (k != exp_k) {
            free(data);
            snprintf(err, err_size, "lora A K=%d expected %d (%s)", k, exp_k,
                     suffix);
            return 0;
        }
        if (p->A) {
            free(data);
            return lora_fail(err, err_size, "lora duplicate A");
        }
        if (p->B && p->r && p->r != r) {
            free(data);
            return lora_fail(err, err_size, "lora A/B rank mismatch");
        }
        p->A = data;
        p->r = r;
        p->k = k;
        if (!p->n)
            p->n = exp_n;
    } else {
        /* B [N, r] */
        n = (int)t->shape[0];
        r = (int)t->shape[1];
        if (n != exp_n) {
            free(data);
            snprintf(err, err_size, "lora B N=%d expected %d (%s)", n, exp_n,
                     suffix);
            return 0;
        }
        if (p->B) {
            free(data);
            return lora_fail(err, err_size, "lora duplicate B");
        }
        if (p->A && p->r && p->r != r) {
            free(data);
            return lora_fail(err, err_size, "lora A/B rank mismatch");
        }
        p->B = data;
        p->r = r;
        p->n = n;
        if (!p->k)
            p->k = exp_k;
    }
    return 1;
}

int h3_lora_load(const char *path, float scale, h3_lora_set *out, char *err,
                 size_t err_size) {
    return h3_lora_load_flags(path, scale, 0u, out, err, err_size);
}

int h3_lora_load_flags(const char *path, float scale, unsigned flags,
                       h3_lora_set *out, char *err, size_t err_size) {
    h3_st_header hdr;
    size_t i;
    int complete = 0;
    int want_refiner = (flags & H3_LORA_REQUIRE_REFINER) ? 1 : 0;

    if (err && err_size)
        err[0] = '\0';
    if (!path || !path[0])
        return lora_fail(err, err_size, "lora: empty path (非金线)");
    memset(&hdr, 0, sizeof(hdr));
    if (!h3_st_read_header(path, &hdr, err, err_size)) {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "lora cannot open '%s' (非金线；缺文件硬失败): %s", path,
                 err && err[0] ? err : "cannot read safetensors");
        if (out)
            memset(out, 0, sizeof(*out));
        return lora_fail(err, err_size, buf);
    }
    if (!out) {
        h3_st_free_header(&hdr);
        return lora_fail(err, err_size, "lora load: null out");
    }
    memset(out, 0, sizeof(*out));
    snprintf(out->path, sizeof(out->path), "%s", path);
    out->scale = scale;

    for (i = 0; i < hdr.tensor_count; i++) {
        const h3_st_tensor *t = &hdr.tensors[i];
        unsigned block = 0;
        const char *suffix = NULL;
        int is_a = 0;
        int scope = H3_LORA_SCOPE_DIT;
        char perr[192];
        float *data;
        size_t elems = 0;
        h3_dtype dt = H3_DTYPE_UNKNOWN;
        h3_lora_pair *pair;

        if (!t->name)
            continue;
        if (h3_lora_name_is_vsa(t->name)) {
            out->ignored_vsa_tensors++;
            fprintf(stderr, "h3_lora: ignore VSA tensor %s\n", t->name);
            continue;
        }
        if (strstr(t->name, ".alpha") || strstr(t->name, "lora_magnitude") ||
            strstr(t->name, "lora_alpha"))
            continue;
        perr[0] = '\0';
        if (!h3_lora_parse_tensor(t->name, &scope, &block, &suffix, &is_a, perr,
                                  sizeof(perr))) {
            fprintf(stderr, "h3_lora: skip unrecognized %s (%s)\n", t->name,
                    perr);
            continue;
        }
        if (scope == H3_LORA_SCOPE_REFINER && !want_refiner) {
            fprintf(stderr, "h3_lora: skip token_refiner %s (not required)\n",
                    t->name);
            continue;
        }
        data = h3_st_load_f32(&hdr, t->name, &elems, &dt, err, err_size);
        if (!data) {
            h3_lora_free(out);
            h3_st_free_header(&hdr);
            return 0;
        }
        pair = pair_for(scope == H3_LORA_SCOPE_REFINER ? &out->refiner[block]
                                                       : &out->blocks[block],
                        suffix);
        if (!assign_ab(pair, is_a, data, t, suffix, err, err_size)) {
            h3_lora_free(out);
            h3_st_free_header(&hdr);
            return 0;
        }
    }
    h3_st_free_header(&hdr);

    for (i = 0; i < H3_LORA_N_BLOCKS; i++) {
        const h3_lora_block *b = &out->blocks[i];
        const h3_lora_pair *ps[4];
        int p;
        ps[0] = &b->qkv;
        ps[1] = &b->out;
        ps[2] = &b->fc1;
        ps[3] = &b->fc2;
        for (p = 0; p < 4; p++) {
            if (!ps[p]->A || !ps[p]->B || ps[p]->r < 1) {
                char buf[256];
                const char *names[] = {"qkv", "out", "fc1", "fc2"};
                snprintf(buf, sizeof(buf),
                         "lora 非金线：缺 blocks.%zu.attn/mlp %s A/B（需 50×4 对）",
                         i, names[p]);
                h3_lora_free(out);
                return lora_fail(err, err_size, buf);
            }
            complete++;
        }
    }
    out->n_pairs = complete;
    if (want_refiner) {
        int rcomplete = 0;
        for (i = 0; i < H3_LORA_N_REFINER; i++) {
            const h3_lora_block *b = &out->refiner[i];
            const h3_lora_pair *ps[4];
            int p;
            ps[0] = &b->qkv;
            ps[1] = &b->out;
            ps[2] = &b->fc1;
            ps[3] = &b->fc2;
            for (p = 0; p < 4; p++) {
                if (!ps[p]->A || !ps[p]->B || ps[p]->r < 1) {
                    char buf[256];
                    snprintf(buf, sizeof(buf),
                             "lora 非金线：缺 token_refiner.blocks.%zu A/B", i);
                    h3_lora_free(out);
                    return lora_fail(err, err_size, buf);
                }
                rcomplete++;
            }
        }
        out->n_refiner_pairs = rcomplete;
    }
    return 1;
}

int h3_lora_apply_f32(const h3_lora_pair *p, float scale, const float *x,
                      float *y, size_t m, char *err, size_t err_size) {
    size_t i, j, t;
    float *tmp;
    if (!p || !p->A || !p->B || !x || !y || m == 0 || p->r < 1 || p->n < 1 ||
        p->k < 1)
        return lora_fail(err, err_size, "lora apply: bad args");
    tmp = (float *)calloc(m * (size_t)p->r, sizeof(float));
    if (!tmp)
        return lora_fail(err, err_size, "lora apply: OOM");
    /* T[m,r] = X[m,k] @ A[r,k]^T */
    for (i = 0; i < m; i++) {
        for (t = 0; t < (size_t)p->r; t++) {
            float s = 0.0f;
            const float *xi = x + i * (size_t)p->k;
            const float *at = p->A + t * (size_t)p->k;
            size_t k;
            for (k = 0; k < (size_t)p->k; k++)
                s += xi[k] * at[k];
            tmp[i * (size_t)p->r + t] = s;
        }
    }
    /* Y[m,n] += scale * T[m,r] @ B[n,r]^T */
    for (i = 0; i < m; i++) {
        for (j = 0; j < (size_t)p->n; j++) {
            float s = 0.0f;
            const float *ti = tmp + i * (size_t)p->r;
            const float *bj = p->B + j * (size_t)p->r;
            for (t = 0; t < (size_t)p->r; t++)
                s += ti[t] * bj[t];
            y[i * (size_t)p->n + j] += scale * s;
        }
    }
    free(tmp);
    return 1;
}
