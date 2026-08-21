#include "h3_gpu_cuda_audio.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, size_t error_size, const char *msg) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s", msg ? msg : "h3_cuda_audio error");
}

static void set_error_cuda(char *error, size_t error_size, const char *op,
                           cudaError_t st) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s: %s", op ? op : "cuda",
             cudaGetErrorString(st));
}

static int fail_cuda(char *error, size_t error_size, const char *op,
                     cudaError_t st) {
    if (st == cudaSuccess) return 1;
    set_error_cuda(error, error_size, op, st);
    return 0;
}

__global__ void k_weight_norm(const float *vector, const float *magnitude,
                              float *out, uint32_t outer, uint32_t inner) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= outer) return;
    uint32_t base = row * inner;
    float square = 0.0f;
    for (uint32_t i = 0; i < inner; i++) {
        float v = vector[base + i];
        square = fmaf(v, v, square);
    }
    float scale = magnitude[row] * rsqrtf(square);
    for (uint32_t i = 0; i < inner; i++)
        out[base + i] = vector[base + i] * scale;
}

__global__ void k_conv1d(const float *in, const float *weight, const float *bias,
                         float *out, uint32_t batch, uint32_t length,
                         uint32_t in_ch, uint32_t out_ch, uint32_t kernel,
                         uint32_t padding, uint32_t dilation) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = batch * length * out_ch;
    if (tid >= total) return;
    uint32_t oc = tid % out_ch;
    uint32_t t = (tid / out_ch) % length;
    uint32_t b = tid / (out_ch * length);
    float sum = bias ? bias[oc] : 0.0f;
    for (uint32_t k = 0; k < kernel; k++) {
        int src = (int)t + (int)(k * dilation) - (int)padding;
        if (src < 0 || src >= (int)length) continue;
        for (uint32_t ic = 0; ic < in_ch; ic++) {
            float x = in[(b * length + (uint32_t)src) * in_ch + ic];
            float w = weight[(oc * in_ch + ic) * kernel + k];
            sum = fmaf(x, w, sum);
        }
    }
    out[tid] = sum;
}

__global__ void k_conv1d_stride(const float *in, const float *weight,
                                const float *bias, float *out, uint32_t batch,
                                uint32_t in_len, uint32_t out_len,
                                uint32_t in_ch, uint32_t out_ch,
                                uint32_t kernel, uint32_t stride,
                                uint32_t padding, uint32_t dilation) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = batch * out_len * out_ch;
    if (tid >= total) return;
    uint32_t oc = tid % out_ch;
    uint32_t t = (tid / out_ch) % out_len;
    uint32_t b = tid / (out_ch * out_len);
    float sum = bias ? bias[oc] : 0.0f;
    for (uint32_t k = 0; k < kernel; k++) {
        int src = (int)(t * stride) + (int)(k * dilation) - (int)padding;
        if (src < 0 || src >= (int)in_len) continue;
        for (uint32_t ic = 0; ic < in_ch; ic++) {
            float x = in[(b * in_len + (uint32_t)src) * in_ch + ic];
            float w = weight[(oc * in_ch + ic) * kernel + k];
            sum = fmaf(x, w, sum);
        }
    }
    out[tid] = sum;
}

/* im2col: rows = batch*out_len, cols = in_ch*kernel (OIK flatten order). */
__global__ void k_im2col1d(const float *in, float *col, uint32_t batch,
                           uint32_t in_len, uint32_t out_len, uint32_t in_ch,
                           uint32_t kernel, uint32_t stride, uint32_t padding,
                           uint32_t dilation) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t cols = in_ch * kernel;
    uint32_t total = batch * out_len * cols;
    if (tid >= total) return;
    uint32_t col_i = tid % cols;
    uint32_t ot = (tid / cols) % out_len;
    uint32_t b = tid / (cols * out_len);
    uint32_t k = col_i % kernel;
    uint32_t ic = col_i / kernel;
    int src = (int)(ot * stride) + (int)(k * dilation) - (int)padding;
    float v = 0.0f;
    if (src >= 0 && src < (int)in_len)
        v = in[(b * in_len + (uint32_t)src) * in_ch + ic];
    col[tid] = v;
}

__global__ void k_snake1d(const float *in, const float *alpha, float *out,
                          uint32_t channels, uint32_t total) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    float a = alpha[tid % channels];
    float x = in[tid];
    float wave = sinf(a * x);
    out[tid] = x + wave * wave / (a + 1e-9f);
}

__global__ void k_layer_norm(const float *in, const float *weight,
                             const float *bias, float *out, uint32_t width,
                             float epsilon) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    extern __shared__ float sm[];
    float *buf = sm;
    const float *x = in + (size_t)row * width;
    float local = 0.0f;
    for (uint32_t i = tid; i < width; i += blockDim.x) local += x[i];
    buf[tid] = local;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    float mean = buf[0] / (float)width;
    __syncthreads();
    local = 0.0f;
    for (uint32_t i = tid; i < width; i += blockDim.x) {
        float c = x[i] - mean;
        local = fmaf(c, c, local);
    }
    buf[tid] = local;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    float inv = rsqrtf(buf[0] / (float)width + epsilon);
    for (uint32_t i = tid; i < width; i += blockDim.x)
        out[(size_t)row * width + i] =
            (x[i] - mean) * inv * weight[i] + bias[i];
}

__global__ void k_qkv_split(const float *qkv, const float *q_bias,
                            const float *k_bias, const float *v_bias, float *q,
                            float *k, float *v, uint32_t width,
                            uint32_t total) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    uint32_t col = tid % width;
    uint32_t row = tid / width;
    uint32_t base = row * width * 3;
    q[tid] = qkv[base + col] + q_bias[col];
    k[tid] = qkv[base + width + col] + k_bias[col];
    v[tid] = qkv[base + width * 2 + col] + v_bias[col];
}

__global__ void k_sdpa_causal(const float *Q, const float *K, const float *V,
                              float *O, uint32_t batch, uint32_t seq,
                              uint32_t heads, uint32_t head_dim, float scale) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = batch * seq * heads;
    if (tid >= total) return;
    uint32_t h = tid % heads;
    uint32_t t = (tid / heads) % seq;
    uint32_t b = tid / (heads * seq);
    const float *q =
        Q + (((size_t)b * seq + t) * heads + h) * head_dim;
    float m = -1e30f;
    float l = 0.0f;
    float acc[256];
    if (head_dim > 256) return; /* guarded by host */
    for (uint32_t d = 0; d < head_dim; d++) acc[d] = 0.0f;
    for (uint32_t s = 0; s <= t; s++) {
        const float *krow =
            K + (((size_t)b * seq + s) * heads + h) * head_dim;
        const float *vrow =
            V + (((size_t)b * seq + s) * heads + h) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++)
            dot = fmaf(q[d], krow[d], dot);
        float score = dot * scale;
        float m_new = score > m ? score : m;
        float alpha = (m > -1e29f) ? expf(m - m_new) : 0.0f;
        float p = expf(score - m_new);
        l = alpha * l + p;
        for (uint32_t d = 0; d < head_dim; d++)
            acc[d] = acc[d] * alpha + p * vrow[d];
        m = m_new;
    }
    float inv = 1.0f / fmaxf(l, 1e-20f);
    float *o = O + (((size_t)b * seq + t) * heads + h) * head_dim;
    for (uint32_t d = 0; d < head_dim; d++) o[d] = acc[d] * inv;
}

__global__ void k_attn_pool(const float *attended, float *out, uint32_t batch,
                            uint32_t length, uint32_t heads, uint32_t head_dim,
                            uint32_t output_dim) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = batch * length * output_dim;
    if (tid >= total) return;
    uint32_t column = tid % output_dim;
    uint32_t row = tid / output_dim;
    uint32_t pool = head_dim / output_dim;
    float sum = 0.0f;
    for (uint32_t head = 0; head < heads; head++) {
        uint32_t base =
            (row * heads + head) * head_dim + column * pool;
        for (uint32_t item = 0; item < pool; item++)
            sum += attended[base + item];
    }
    out[tid] = sum / (float)(heads * pool);
}

__global__ void k_geglu(const float *gate, const float *linear, float *out,
                        uint32_t elements) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) return;
    float x = gate[tid];
    float cube = x * x * x;
    float gelu = 0.5f * x *
                 (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * cube)));
    out[tid] = gelu * linear[tid];
}

static int g_audio_conv_backend = 0; /* 0=auto, 1=naive, 2=gemm */

void h3_cuda_audio_set_conv_backend(const char *backend) {
    if (!backend || !*backend || !strcmp(backend, "auto"))
        g_audio_conv_backend = 0;
    else if (!strcmp(backend, "naive"))
        g_audio_conv_backend = 1;
    else if (!strcmp(backend, "gemm"))
        g_audio_conv_backend = 2;
}

const char *h3_cuda_audio_conv_backend(void) {
    if (g_audio_conv_backend == 1) return "naive";
    if (g_audio_conv_backend == 2) return "gemm";
    const char *e = getenv("H3_AUDIO_CONV_BACKEND");
    if (e && !strcmp(e, "naive")) return "naive";
    if (e && !strcmp(e, "gemm")) return "gemm";
    return "auto";
}

static int want_gemm_conv(uint32_t batch, uint32_t out_len, uint32_t in_ch,
                          uint32_t out_ch, uint32_t kernel) {
    const char *mode = h3_cuda_audio_conv_backend();
    uint64_t flops =
        (uint64_t)batch * out_len * out_ch * in_ch * kernel;
    if (!strcmp(mode, "naive")) return 0;
    if (!strcmp(mode, "gemm")) return 1;
    /* Auto: prefer GEMM when im2col+sgemm beats naive scalar loops. */
    return flops >= 256ull * 1024ull;
}

static int conv1d_gemm(h3_cuda *cuda, float *d_out, const float *d_in,
                       const float *d_weight, const float *d_bias,
                       uint32_t batch, uint32_t in_len, uint32_t out_len,
                       uint32_t in_ch, uint32_t out_ch, uint32_t kernel,
                       uint32_t stride, uint32_t padding, uint32_t dilation,
                       char *error, size_t error_size) {
    static float *d_col_ws = NULL;
    static size_t d_col_cap = 0;
    size_t rows = (size_t)batch * out_len;
    size_t cols = (size_t)in_ch * kernel;
    size_t need = rows * cols;
    cudaError_t st;
    if (need > d_col_cap) {
        if (d_col_ws) cudaFree(d_col_ws);
        d_col_ws = NULL;
        d_col_cap = 0;
        st = cudaMalloc((void **)&d_col_ws, need * sizeof(float));
        if (st != cudaSuccess) {
            set_error_cuda(error, error_size, "im2col alloc", st);
            return 0;
        }
        d_col_cap = need;
    }
    {
        size_t total = need;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        k_im2col1d<<<blocks, threads>>>(d_in, d_col_ws, batch, in_len, out_len,
                                        in_ch, kernel, stride, padding,
                                        dilation);
        if (!fail_cuda(error, error_size, "im2col", cudaGetLastError()))
            return 0;
    }
    if (!h3_cuda_gemm_f32_dd(cuda, d_col_ws, d_weight, rows, out_ch, cols, d_out,
                             NULL, error, error_size))
        return 0;
    if (d_bias &&
        !h3_cuda_bias_add_f32_dev(cuda, d_out, d_bias, rows, out_ch, error,
                                  error_size))
        return 0;
    return 1;
}

__global__ void k_conv_transpose1d(const float *in, const float *weight,
                                   const float *bias, float *out,
                                   uint32_t batch, uint32_t in_len,
                                   uint32_t out_len, uint32_t in_ch,
                                   uint32_t out_ch, uint32_t kernel,
                                   uint32_t stride, uint32_t padding) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = batch * out_len * out_ch;
    if (tid >= total) return;
    uint32_t oc = tid % out_ch;
    uint32_t t = (tid / out_ch) % out_len;
    uint32_t b = tid / (out_ch * out_len);
    float sum = bias ? bias[oc] : 0.0f;
    for (uint32_t k = 0; k < kernel; k++) {
        int numerator = (int)t + (int)padding - (int)k;
        if (numerator < 0 || (numerator % (int)stride) != 0) continue;
        int src = numerator / (int)stride;
        if (src < 0 || src >= (int)in_len) continue;
        for (uint32_t ic = 0; ic < in_ch; ic++) {
            float x = in[(b * in_len + (uint32_t)src) * in_ch + ic];
            float w = weight[(ic * out_ch + oc) * kernel + k];
            sum = fmaf(x, w, sum);
        }
    }
    out[tid] = sum;
}

__device__ float upsample_at(const float *input, const float *filter,
                            uint32_t length, uint32_t channels, uint32_t batch,
                            uint32_t channel, int up_time) {
    int raw = up_time + 15;
    float result = 0.0f;
    for (int k = 0; k < 12; k++) {
        int numerator = raw - k;
        if (numerator < 0 || (numerator & 1)) continue;
        int padded = numerator / 2;
        int source = padded - 5;
        if (source < 0) source = 0;
        if (source >= (int)length) source = (int)length - 1;
        uint32_t idx =
            (batch * length + (uint32_t)source) * channels + channel;
        result = fmaf(input[idx], 2.0f * filter[k], result);
    }
    return result;
}

__global__ void k_alias_free_snake(const float *input, const float *alpha_log,
                                   const float *beta_log,
                                   const float *up_filter,
                                   const float *down_filter, float *output,
                                   uint32_t batch, uint32_t length,
                                   uint32_t channels) {
    uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t time = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t b = blockIdx.z;
    if (channel >= channels || time >= length || b >= batch) return;
    float alpha = expf(alpha_log[channel]);
    float beta = expf(beta_log[channel]);
    float result = 0.0f;
    for (int down_k = 0; down_k < 12; down_k++) {
        int up_time = (int)(time * 2) + down_k - 5;
        if (up_time < 0) up_time = 0;
        if (up_time >= (int)(length * 2)) up_time = (int)(length * 2) - 1;
        float upsampled = upsample_at(input, up_filter, length, channels, b,
                                      channel, up_time);
        float sine = sinf(alpha * upsampled);
        float activated = upsampled + sine * sine / (beta + 1e-9f);
        result = fmaf(activated, down_filter[down_k], result);
    }
    output[(b * length + time) * channels + channel] = result;
}

__global__ void k_add_scaled(const float *left, const float *right, float *out,
                             float left_scale, float right_scale,
                             uint32_t elements) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements)
        out[i] = left[i] * left_scale + right[i] * right_scale;
}

__global__ void k_clip(const float *in, float *out, uint32_t elements, float lo,
                       float hi) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) {
        float v = in[i];
        out[i] = fminf(hi, fmaxf(lo, v));
    }
}

int h3_cuda_audio_weight_norm_f32(h3_cuda *cuda, float *d_out,
                                  const float *d_vector,
                                  const float *d_magnitude, uint32_t outer,
                                  uint32_t inner, char *error,
                                  size_t error_size) {
    if (!cuda || !d_out || !d_vector || !d_magnitude || !outer || !inner) {
        set_error(error, error_size, "invalid weight_norm args");
        return 0;
    }
    int threads = 128;
    int blocks = (int)((outer + threads - 1) / threads);
    k_weight_norm<<<blocks, threads>>>(d_vector, d_magnitude, d_out, outer,
                                       inner);
    return fail_cuda(error, error_size, "weight_norm launch",
                     cudaGetLastError());
}

int h3_cuda_audio_conv1d_f32(h3_cuda *cuda, float *d_out, const float *d_in,
                             const float *d_weight, const float *d_bias,
                             uint32_t batch, uint32_t length, uint32_t in_ch,
                             uint32_t out_ch, uint32_t kernel, uint32_t padding,
                             uint32_t dilation, char *error,
                             size_t error_size) {
    if (!cuda || !d_out || !d_in || !d_weight || !batch || !length || !in_ch ||
        !out_ch || !kernel || !dilation) {
        set_error(error, error_size, "invalid conv1d args");
        return 0;
    }
    if (want_gemm_conv(batch, length, in_ch, out_ch, kernel))
        return conv1d_gemm(cuda, d_out, d_in, d_weight, d_bias, batch, length,
                           length, in_ch, out_ch, kernel, 1, padding, dilation,
                           error, error_size);
    uint32_t total = batch * length * out_ch;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    k_conv1d<<<blocks, threads>>>(d_in, d_weight, d_bias, d_out, batch, length,
                                  in_ch, out_ch, kernel, padding, dilation);
    return fail_cuda(error, error_size, "conv1d launch", cudaGetLastError());
}

int h3_cuda_audio_conv1d_stride_f32(h3_cuda *cuda, float *d_out,
                                    const float *d_in, const float *d_weight,
                                    const float *d_bias, uint32_t batch,
                                    uint32_t in_length, uint32_t out_length,
                                    uint32_t in_ch, uint32_t out_ch,
                                    uint32_t kernel, uint32_t stride,
                                    uint32_t padding, uint32_t dilation,
                                    char *error, size_t error_size) {
    if (!cuda || !d_out || !d_in || !d_weight || !batch || !in_length ||
        !out_length || !in_ch || !out_ch || !kernel || !stride || !dilation) {
        set_error(error, error_size, "invalid conv1d_stride args");
        return 0;
    }
    if (want_gemm_conv(batch, out_length, in_ch, out_ch, kernel))
        return conv1d_gemm(cuda, d_out, d_in, d_weight, d_bias, batch, in_length,
                           out_length, in_ch, out_ch, kernel, stride, padding,
                           dilation, error, error_size);
    uint32_t total = batch * out_length * out_ch;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    k_conv1d_stride<<<blocks, threads>>>(
        d_in, d_weight, d_bias, d_out, batch, in_length, out_length, in_ch,
        out_ch, kernel, stride, padding, dilation);
    return fail_cuda(error, error_size, "conv1d_stride launch",
                     cudaGetLastError());
}

int h3_cuda_audio_conv_transpose1d_f32(h3_cuda *cuda, float *d_out,
                                       const float *d_in, const float *d_weight,
                                       const float *d_bias, uint32_t batch,
                                       uint32_t length, uint32_t in_ch,
                                       uint32_t out_ch, uint32_t kernel,
                                       uint32_t stride, uint32_t padding,
                                       char *error, size_t error_size) {
    if (!cuda || !d_out || !d_in || !d_weight || !batch || !length || !in_ch ||
        !out_ch || !kernel || !stride) {
        set_error(error, error_size, "invalid conv_transpose1d args");
        return 0;
    }
    uint64_t expanded =
        (uint64_t)(length - 1) * stride + kernel - 2ull * padding;
    if (expanded == 0 || expanded > UINT32_MAX) {
        set_error(error, error_size, "conv_transpose length overflow");
        return 0;
    }
    uint32_t out_len = (uint32_t)expanded;
    uint32_t total = batch * out_len * out_ch;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    k_conv_transpose1d<<<blocks, threads>>>(d_in, d_weight, d_bias, d_out,
                                            batch, length, out_len, in_ch,
                                            out_ch, kernel, stride, padding);
    return fail_cuda(error, error_size, "conv_transpose1d launch",
                     cudaGetLastError());
}

int h3_cuda_audio_alias_free_snake_f32(h3_cuda *cuda, float *d_out,
                                       const float *d_in, const float *d_alpha,
                                       const float *d_beta,
                                       const float *d_up_filter,
                                       const float *d_down_filter,
                                       uint32_t batch, uint32_t length,
                                       uint32_t channels, char *error,
                                       size_t error_size) {
    if (!cuda || !d_out || !d_in || !d_alpha || !d_beta || !d_up_filter ||
        !d_down_filter || !batch || !length || !channels) {
        set_error(error, error_size, "invalid alias_free_snake args");
        return 0;
    }
    dim3 block(8, 16, 1);
    dim3 grid((channels + block.x - 1) / block.x,
              (length + block.y - 1) / block.y, batch);
    k_alias_free_snake<<<grid, block>>>(d_in, d_alpha, d_beta, d_up_filter,
                                        d_down_filter, d_out, batch, length,
                                        channels);
    return fail_cuda(error, error_size, "alias_free_snake launch",
                     cudaGetLastError());
}

int h3_cuda_audio_add_scaled_f32(h3_cuda *cuda, float *d_out,
                                 const float *d_left, const float *d_right,
                                 float left_scale, float right_scale,
                                 uint32_t elements, char *error,
                                 size_t error_size) {
    if (!cuda || !d_out || !d_left || !d_right || !elements) {
        set_error(error, error_size, "invalid add_scaled args");
        return 0;
    }
    int threads = 256;
    int blocks = (int)((elements + threads - 1) / threads);
    k_add_scaled<<<blocks, threads>>>(d_left, d_right, d_out, left_scale,
                                      right_scale, elements);
    return fail_cuda(error, error_size, "add_scaled launch",
                     cudaGetLastError());
}

int h3_cuda_audio_clip_f32(h3_cuda *cuda, float *d_out, const float *d_in,
                           uint32_t elements, float lo, float hi, char *error,
                           size_t error_size) {
    if (!cuda || !d_out || !d_in || !elements) {
        set_error(error, error_size, "invalid clip args");
        return 0;
    }
    int threads = 256;
    int blocks = (int)((elements + threads - 1) / threads);
    k_clip<<<blocks, threads>>>(d_in, d_out, elements, lo, hi);
    return fail_cuda(error, error_size, "clip launch", cudaGetLastError());
}

int h3_cuda_audio_copy_f32(h3_cuda *cuda, float *d_dst, const float *d_src,
                           size_t elements, char *error, size_t error_size) {
    if (!cuda || !d_dst || !d_src || !elements) {
        set_error(error, error_size, "invalid audio copy args");
        return 0;
    }
    cudaError_t st =
        cudaMemcpy(d_dst, d_src, elements * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    return fail_cuda(error, error_size, "audio copy", st);
}

int h3_cuda_audio_snake1d_f32(h3_cuda *cuda, float *d_out, const float *d_in,
                              const float *d_alpha, uint32_t batch,
                              uint32_t length, uint32_t channels, char *error,
                              size_t error_size) {
    if (!cuda || !d_out || !d_in || !d_alpha || !batch || !length || !channels) {
        set_error(error, error_size, "invalid snake1d args");
        return 0;
    }
    uint32_t total = batch * length * channels;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    k_snake1d<<<blocks, threads>>>(d_in, d_alpha, d_out, channels, total);
    return fail_cuda(error, error_size, "snake1d launch", cudaGetLastError());
}

int h3_cuda_audio_layer_norm_f32(h3_cuda *cuda, float *d_out, const float *d_in,
                                 const float *d_weight, const float *d_bias,
                                 uint32_t rows, uint32_t width, float epsilon,
                                 char *error, size_t error_size) {
    if (!cuda || !d_out || !d_in || !d_weight || !d_bias || !rows || !width) {
        set_error(error, error_size, "invalid layer_norm args");
        return 0;
    }
    int threads = 256;
    if (threads > (int)width) {
        threads = 1;
        while (threads * 2 <= (int)width && threads < 256) threads *= 2;
    }
    k_layer_norm<<<rows, threads, (size_t)threads * sizeof(float)>>>(
        d_in, d_weight, d_bias, d_out, width, epsilon);
    return fail_cuda(error, error_size, "layer_norm launch",
                     cudaGetLastError());
}

int h3_cuda_audio_linear_f32(h3_cuda *cuda, float *d_out, const float *d_in,
                             const float *d_weight, const float *d_bias,
                             uint32_t rows, uint32_t in_dim, uint32_t out_dim,
                             char *error, size_t error_size) {
    if (!cuda || !d_out || !d_in || !d_weight || !rows || !in_dim || !out_dim) {
        set_error(error, error_size, "invalid linear args");
        return 0;
    }
    if (!h3_cuda_gemm_f32_dd(cuda, d_in, d_weight, rows, out_dim, in_dim, d_out,
                             NULL, error, error_size))
        return 0;
    if (d_bias &&
        !h3_cuda_bias_add_f32_dev(cuda, d_out, d_bias, rows, out_dim, error,
                                  error_size))
        return 0;
    return 1;
}

int h3_cuda_audio_qkv_split_f32(h3_cuda *cuda, float *d_q, float *d_k,
                                float *d_v, const float *d_qkv,
                                const float *d_q_bias, const float *d_k_bias,
                                const float *d_v_bias, uint32_t batch,
                                uint32_t length, uint32_t heads,
                                uint32_t head_dim, char *error,
                                size_t error_size) {
    if (!cuda || !d_q || !d_k || !d_v || !d_qkv || !d_q_bias || !d_k_bias ||
        !d_v_bias || !batch || !length || !heads || !head_dim) {
        set_error(error, error_size, "invalid qkv_split args");
        return 0;
    }
    uint32_t width = heads * head_dim;
    uint32_t total = batch * length * width;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    k_qkv_split<<<blocks, threads>>>(d_qkv, d_q_bias, d_k_bias, d_v_bias, d_q,
                                     d_k, d_v, width, total);
    return fail_cuda(error, error_size, "qkv_split launch", cudaGetLastError());
}

int h3_cuda_audio_sdpa_causal_f32(h3_cuda *cuda, float *d_out, const float *d_q,
                                  const float *d_k, const float *d_v,
                                  uint32_t batch, uint32_t sequence,
                                  uint32_t heads, uint32_t head_dim,
                                  float scale, char *error,
                                  size_t error_size) {
    if (!cuda || !d_out || !d_q || !d_k || !d_v || !batch || !sequence ||
        !heads || !head_dim) {
        set_error(error, error_size, "invalid sdpa_causal args");
        return 0;
    }
    if (head_dim > 256) {
        set_error(error, error_size, "sdpa_causal head_dim>256 unsupported");
        return 0;
    }
    uint32_t total = batch * sequence * heads;
    int threads = 128;
    int blocks = (int)((total + threads - 1) / threads);
    k_sdpa_causal<<<blocks, threads>>>(d_q, d_k, d_v, d_out, batch, sequence,
                                       heads, head_dim, scale);
    return fail_cuda(error, error_size, "sdpa_causal launch",
                     cudaGetLastError());
}

int h3_cuda_audio_attention_pool_f32(h3_cuda *cuda, float *d_out,
                                     const float *d_attended, uint32_t batch,
                                     uint32_t length, uint32_t heads,
                                     uint32_t head_dim, uint32_t output_dim,
                                     char *error, size_t error_size) {
    if (!cuda || !d_out || !d_attended || !batch || !length || !heads ||
        !head_dim || !output_dim || (head_dim % output_dim) != 0) {
        set_error(error, error_size, "invalid attention_pool args");
        return 0;
    }
    uint32_t total = batch * length * output_dim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    k_attn_pool<<<blocks, threads>>>(d_attended, d_out, batch, length, heads,
                                     head_dim, output_dim);
    return fail_cuda(error, error_size, "attention_pool launch",
                     cudaGetLastError());
}

int h3_cuda_audio_geglu_f32(h3_cuda *cuda, float *d_out, const float *d_gate,
                            const float *d_linear, uint32_t elements,
                            char *error, size_t error_size) {
    if (!cuda || !d_out || !d_gate || !d_linear || !elements) {
        set_error(error, error_size, "invalid geglu args");
        return 0;
    }
    int threads = 256;
    int blocks = (int)((elements + threads - 1) / threads);
    k_geglu<<<blocks, threads>>>(d_gate, d_linear, d_out, elements);
    return fail_cuda(error, error_size, "geglu launch", cudaGetLastError());
}
