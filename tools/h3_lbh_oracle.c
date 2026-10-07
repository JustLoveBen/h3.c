/* Layer/kernel F32 oracle for LatentResizer3D vs PyTorch. Not a generate path. */
#include "h3_gpu_cuda.h"
#include "h3_latent_upscaler.h"
#include "h3_latent_upscaler_internal.h"

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *msg) {
    fprintf(stderr, "h3_lbh_oracle: %s\n", msg ? msg : "fail");
    return 1;
}

static float *read_f32(const char *path, size_t n) {
    FILE *f;
    float *p;
    size_t got;
    f = fopen(path, "rb");
    if (!f)
        return NULL;
    p = (float *)malloc(n * sizeof(float));
    if (!p) {
        fclose(f);
        return NULL;
    }
    got = fread(p, sizeof(float), n, f);
    fclose(f);
    if (got != n) {
        free(p);
        return NULL;
    }
    return p;
}

static int write_f32(const char *path, const float *p, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f)
        return 0;
    if (fwrite(p, sizeof(float), n, f) != n) {
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

static int need_cuda(h3_cuda **cuda, char *err, size_t err_size) {
    *cuda = h3_cuda_create(0, err, err_size);
    return *cuda != NULL;
}

static int cmd_conv3d(int argc, char **argv) {
    int n, ic, oc, t, h, w, kd, kh, kw, pt, ph, pw, groups;
    size_t nin, nw, nb, nout;
    float *in = NULL, *wt = NULL, *bias = NULL, *out = NULL;
    float *d_in = NULL, *d_w = NULL, *d_b = NULL, *d_o = NULL;
    h3_cuda *cuda = NULL;
    char err[256];
    int rc = 1;
    if (argc != 17)
        return fail("conv3d n ic oc t h w kd kh kw padt padh padw groups in w b out");
    n = atoi(argv[0]);
    ic = atoi(argv[1]);
    oc = atoi(argv[2]);
    t = atoi(argv[3]);
    h = atoi(argv[4]);
    w = atoi(argv[5]);
    kd = atoi(argv[6]);
    kh = atoi(argv[7]);
    kw = atoi(argv[8]);
    pt = atoi(argv[9]);
    ph = atoi(argv[10]);
    pw = atoi(argv[11]);
    groups = atoi(argv[12]);
    nin = (size_t)n * (size_t)ic * (size_t)t * (size_t)h * (size_t)w;
    nw = (size_t)oc * (size_t)(ic / groups) * (size_t)kd * (size_t)kh *
         (size_t)kw;
    nb = (size_t)oc;
    nout = (size_t)n * (size_t)oc * (size_t)t * (size_t)h * (size_t)w;
    if (!need_cuda(&cuda, err, sizeof(err)))
        return fail(err);
    h3_ups_bind_cuda(cuda);
    in = read_f32(argv[13], nin);
    wt = read_f32(argv[14], nw);
    bias = read_f32(argv[15], nb);
    out = (float *)malloc(nout * sizeof(float));
    if (!in || !wt || !bias || !out) {
        fail("conv3d read/OOM");
        goto done;
    }
    if (cudaMalloc((void **)&d_in, nin * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_w, nw * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_b, nb * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_o, nout * sizeof(float)) != cudaSuccess) {
        fail("conv3d cudaMalloc");
        goto done;
    }
    cudaMemcpy(d_in, in, nin * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, wt, nw * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, bias, nb * sizeof(float), cudaMemcpyHostToDevice);
    if (!h3_ups_conv3d_nchw(d_in, d_o, d_w, d_b, n, ic, oc, t, h, w, kd, kh, kw,
                            pt, ph, pw, groups, err, sizeof(err))) {
        fail(err);
        goto done;
    }
    cudaMemcpy(out, d_o, nout * sizeof(float), cudaMemcpyDeviceToHost);
    if (!write_f32(argv[16], out, nout))
        fail("conv3d write");
    else {
        printf("ok conv3d nout=%zu\n", nout);
        rc = 0;
    }
done:
    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_b);
    cudaFree(d_o);
    free(in);
    free(wt);
    free(bias);
    free(out);
    h3_cuda_free(cuda);
    return rc;
}

static int cmd_gn(int argc, char **argv) {
    int n, c, t, h, w, groups;
    size_t ntot;
    float *x = NULL, *gamma = NULL, *beta = NULL, *d_x = NULL, *d_g = NULL,
          *d_b = NULL;
    h3_cuda *cuda = NULL;
    char err[256];
    int rc = 1;
    if (argc != 10)
        return fail("gn n c t h w groups in gamma beta out");
    n = atoi(argv[0]);
    c = atoi(argv[1]);
    t = atoi(argv[2]);
    h = atoi(argv[3]);
    w = atoi(argv[4]);
    groups = atoi(argv[5]);
    ntot = (size_t)n * (size_t)c * (size_t)t * (size_t)h * (size_t)w;
    if (!need_cuda(&cuda, err, sizeof(err)))
        return fail(err);
    x = read_f32(argv[6], ntot);
    gamma = read_f32(argv[7], (size_t)c);
    beta = read_f32(argv[8], (size_t)c);
    if (!x || !gamma || !beta) {
        fail("gn read");
        goto done;
    }
    if (cudaMalloc((void **)&d_x, ntot * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_g, (size_t)c * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_b, (size_t)c * sizeof(float)) != cudaSuccess) {
        fail("gn cudaMalloc");
        goto done;
    }
    cudaMemcpy(d_x, x, ntot * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, gamma, (size_t)c * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, beta, (size_t)c * sizeof(float), cudaMemcpyHostToDevice);
    if (!h3_ups_groupnorm(d_x, d_g, d_b, n, c, t, h, w, groups, 1e-5f, err,
                          sizeof(err))) {
        fail(err);
        goto done;
    }
    cudaMemcpy(x, d_x, ntot * sizeof(float), cudaMemcpyDeviceToHost);
    if (!write_f32(argv[9], x, ntot))
        fail("gn write");
    else {
        printf("ok gn ntot=%zu\n", ntot);
        rc = 0;
    }
done:
    cudaFree(d_x);
    cudaFree(d_g);
    cudaFree(d_b);
    free(x);
    free(gamma);
    free(beta);
    h3_cuda_free(cuda);
    return rc;
}

static int cmd_trilinear(int argc, char **argv) {
    int n, c, t, ih, iw, oh, ow;
    size_t nin, nout;
    float *in = NULL, *out = NULL, *d_in = NULL, *d_o = NULL;
    h3_cuda *cuda = NULL;
    char err[256];
    int rc = 1;
    if (argc != 9)
        return fail("trilinear n c t ih iw oh ow in out");
    n = atoi(argv[0]);
    c = atoi(argv[1]);
    t = atoi(argv[2]);
    ih = atoi(argv[3]);
    iw = atoi(argv[4]);
    oh = atoi(argv[5]);
    ow = atoi(argv[6]);
    nin = (size_t)n * (size_t)c * (size_t)t * (size_t)ih * (size_t)iw;
    nout = (size_t)n * (size_t)c * (size_t)t * (size_t)oh * (size_t)ow;
    if (!need_cuda(&cuda, err, sizeof(err)))
        return fail(err);
    in = read_f32(argv[7], nin);
    out = (float *)malloc(nout * sizeof(float));
    if (!in || !out) {
        fail("trilinear read/OOM");
        goto done;
    }
    if (cudaMalloc((void **)&d_in, nin * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&d_o, nout * sizeof(float)) != cudaSuccess) {
        fail("trilinear cudaMalloc");
        goto done;
    }
    cudaMemcpy(d_in, in, nin * sizeof(float), cudaMemcpyHostToDevice);
    if (!h3_ups_trilinear(d_in, d_o, n, c, t, ih, iw, oh, ow, err,
                          sizeof(err))) {
        fail(err);
        goto done;
    }
    cudaMemcpy(out, d_o, nout * sizeof(float), cudaMemcpyDeviceToHost);
    if (!write_f32(argv[8], out, nout))
        fail("trilinear write");
    else {
        printf("ok trilinear nout=%zu\n", nout);
        rc = 0;
    }
done:
    cudaFree(d_in);
    cudaFree(d_o);
    free(in);
    free(out);
    h3_cuda_free(cuda);
    return rc;
}

static int cmd_whead(int argc, char **argv) {
    h3_latent_upscaler *u = NULL;
    char err[256];
    float head[8];
    int i;
    if (argc != 1)
        return fail("whead ckpt");
    if (!h3_latent_upscaler_open(argv[0], &u, err, sizeof(err)))
        return fail(err);
    if (!h3_latent_upscaler_conv_in_head(u, head)) {
        h3_latent_upscaler_close(u);
        return fail("whead");
    }
    printf("conv_in.weight[0:8]");
    for (i = 0; i < 8; i++)
        printf(" %.8g", (double)head[i]);
    printf("\n");
    h3_latent_upscaler_close(u);
    return 0;
}

static int cmd_stage(int argc, char **argv) {
    int stage, t, h, w, oh, ow, oc = 0, ot = 0, oh2 = 0, ow2 = 0;
    size_t nin, nout_cap;
    float *in = NULL, *mean = NULL, *stdv = NULL, *out = NULL;
    h3_latent_upscaler *u = NULL;
    h3_cuda *cuda = NULL;
    char err[512];
    if (argc != 11)
        return fail("stage N ckpt mean std t h w oh ow in out");
    stage = atoi(argv[0]);
    t = atoi(argv[4]);
    h = atoi(argv[5]);
    w = atoi(argv[6]);
    oh = atoi(argv[7]);
    ow = atoi(argv[8]);
    nin = 24ull * (size_t)t * (size_t)h * (size_t)w;
    nout_cap = 512ull * (size_t)t * (size_t)(oh > h ? oh : h) *
               (size_t)(ow > w ? ow : w);
    if (!h3_latent_upscaler_open(argv[1], &u, err, sizeof(err)))
        return fail(err);
    if (!need_cuda(&cuda, err, sizeof(err))) {
        h3_latent_upscaler_close(u);
        return fail(err);
    }
    in = read_f32(argv[9], nin);
    mean = read_f32(argv[2], 24);
    stdv = read_f32(argv[3], 24);
    out = (float *)malloc(nout_cap * sizeof(float));
    if (!in || !mean || !stdv || !out) {
        fail("stage read/OOM");
        free(in);
        free(mean);
        free(stdv);
        free(out);
        h3_cuda_free(cuda);
        h3_latent_upscaler_close(u);
        return 1;
    }
    if (!h3_latent_upscaler_oracle(u, cuda, stage, in, t, h, w, oh, ow, mean,
                                   stdv, out, &oc, &ot, &oh2, &ow2, err,
                                   sizeof(err))) {
        fprintf(stderr, "oracle: %s\n", err);
        free(in);
        free(mean);
        free(stdv);
        free(out);
        h3_cuda_free(cuda);
        h3_latent_upscaler_close(u);
        return 1;
    }
    if (!write_f32(argv[10], out, (size_t)oc * (size_t)ot * (size_t)oh2 *
                                      (size_t)ow2)) {
        fail("stage write");
        free(in);
        free(mean);
        free(stdv);
        free(out);
        h3_cuda_free(cuda);
        h3_latent_upscaler_close(u);
        return 1;
    }
    printf("ok stage=%d out=%d,%d,%d,%d\n", stage, oc, ot, oh2, ow2);
    free(in);
    free(mean);
    free(stdv);
    free(out);
    h3_cuda_free(cuda);
    h3_latent_upscaler_close(u);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2)
        return fail("conv3d|gn|trilinear|whead|stage ...");
    if (strcmp(argv[1], "conv3d") == 0)
        return cmd_conv3d(argc - 2, argv + 2);
    if (strcmp(argv[1], "gn") == 0)
        return cmd_gn(argc - 2, argv + 2);
    if (strcmp(argv[1], "trilinear") == 0)
        return cmd_trilinear(argc - 2, argv + 2);
    if (strcmp(argv[1], "whead") == 0)
        return cmd_whead(argc - 2, argv + 2);
    if (strcmp(argv[1], "stage") == 0)
        return cmd_stage(argc - 2, argv + 2);
    return fail("unknown cmd");
}
