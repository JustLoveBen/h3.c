#include "h3_gpu_cuda_vae.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static void set_error(char *error, size_t error_size, const char *msg) {
    if (!error || !error_size) return;
    snprintf(error, error_size, "%s", msg ? msg : "h3_cuda_vae error");
}

static cudaStream_t vae_st(h3_cuda *cuda) {
    return (cudaStream_t)h3_cuda_op_stream(cuda);
}

__global__ void h3_vae_bias_residual_scale_kernel(
    float *__restrict__ x, const float *__restrict__ branch,
    const float *__restrict__ bias, const float *__restrict__ scale,
    size_t rows, size_t width) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * width;
    size_t c;
    float v;
    if (idx >= total) return;
    c = idx % width;
    v = branch[idx];
    if (bias) v += bias[c];
    x[idx] += v * scale[c];
}

__global__ void h3_vae_bias_silu_mul_kernel(const float *__restrict__ fused,
                                            const float *__restrict__ bias,
                                            float *__restrict__ out,
                                            size_t rows, size_t width) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * width;
    size_t r, c;
    float g, u;
    if (idx >= total) return;
    r = idx / width;
    c = idx % width;
    g = fused[r * width * 2 + c];
    u = fused[r * width * 2 + width + c];
    if (bias) {
        g += bias[c];
        u += bias[width + c];
    }
    out[idx] = (g / (1.0f + expf(-g))) * u;
}

__global__ void h3_vae_rms_f32_to_f16_kernel(const float *__restrict__ x,
                                             const float *__restrict__ w,
                                             __half *__restrict__ out,
                                             size_t rows, size_t width,
                                             float eps) {
    size_t r = (size_t)blockIdx.x;
    if (r >= rows) return;
    const float *xr = x + r * width;
    __half *yr = out + r * width;
    float sum = 0.0f;
    for (size_t c = threadIdx.x; c < width; c += blockDim.x)
        sum += xr[c] * xr[c];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    __shared__ float warp_sums[32];
    __shared__ float ssum;
    int warp = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        int nwarps = ((int)blockDim.x + 31) >> 5;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        ssum = total;
    }
    __syncthreads();
    float inv = rsqrtf(ssum / (float)width + eps);
    for (size_t c = threadIdx.x; c < width; c += blockDim.x)
        yr[c] = __float2half(xr[c] * inv * w[c]);
}

__global__ void h3_vae_bias_add_f16_kernel(__half *__restrict__ y,
                                           const float *__restrict__ bias,
                                           size_t rows, size_t width) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * width;
    if (idx >= total) return;
    y[idx] = __float2half(__half2float(y[idx]) + bias[idx % width]);
}

__global__ void h3_vae_bias_silu_mul_f16_kernel(
    const __half *__restrict__ fused, const float *__restrict__ bias,
    __half *__restrict__ out, size_t rows, size_t width) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * width;
    size_t r, c;
    float g, u;
    if (idx >= total) return;
    r = idx / width;
    c = idx % width;
    g = __half2float(fused[r * width * 2 + c]);
    u = __half2float(fused[r * width * 2 + width + c]);
    if (bias) {
        g += bias[c];
        u += bias[width + c];
    }
    out[idx] = __float2half((g / (1.0f + expf(-g))) * u);
}

__global__ void h3_vae_bias_residual_scale_f16_kernel(
    float *__restrict__ x, const __half *__restrict__ branch,
    const float *__restrict__ bias, const float *__restrict__ scale,
    size_t rows, size_t width) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * width;
    size_t c;
    float v;
    if (idx >= total) return;
    c = idx % width;
    v = __half2float(branch[idx]);
    if (bias) v += bias[c];
    x[idx] += v * scale[c];
}

/* Same 256/64 split-half geometry as h3_cuda_qkv_rope_f16_kernel; F16 QKV. */
__global__ void h3_vae_qkv_rope_f16_in_kernel(
    const __half *__restrict__ qkv, const float *__restrict__ q_norm,
    const float *__restrict__ k_norm, const float *__restrict__ rope_cos,
    const float *__restrict__ rope_sin, __half *__restrict__ query,
    __half *__restrict__ key, __half *__restrict__ value, size_t seq,
    size_t seq_padded, size_t heads, size_t head_dim, size_t rope_half,
    float epsilon, int grouped, int split_half) {
    size_t row = (size_t)blockIdx.y;
    size_t head = (size_t)blockIdx.x;
    if (head >= heads) return;
    if (seq_padded < seq) seq_padded = seq;
    if (row >= seq_padded) return;
    __half *q_out = query + (row * heads + head) * head_dim;
    __half *k_out = key + (row * heads + head) * head_dim;
    __half *v_out = value + (row * heads + head) * head_dim;
    if (row >= seq) {
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            q_out[d] = __float2half(0.0f);
            k_out[d] = __float2half(0.0f);
            v_out[d] = __float2half(0.0f);
        }
        return;
    }
    size_t inner = heads * head_dim;
    const __half *qkv_row = qkv + row * inner * 3;
    const float *c_row = rope_cos + row * rope_half;
    const float *s_row = rope_sin + row * rope_half;
    const __half *q_base;
    const __half *k_base;
    const __half *v_base;
    if (grouped) {
        q_base = qkv_row + head * head_dim * 3;
        k_base = q_base + head_dim;
        v_base = k_base + head_dim;
    } else {
        q_base = qkv_row + head * head_dim;
        k_base = q_base + inner;
        v_base = q_base + inner * 2;
    }
    __shared__ float sq[128], sk[128];
    float q_sum = 0.0f, k_sum = 0.0f;
    for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float qv = __half2float(q_base[d]);
        float kv = __half2float(k_base[d]);
        q_sum += qv * qv;
        k_sum += kv * kv;
        v_out[d] = v_base[d];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        q_sum += __shfl_down_sync(0xffffffffu, q_sum, offset);
        k_sum += __shfl_down_sync(0xffffffffu, k_sum, offset);
    }
    __shared__ float qk[2];
    __shared__ float wq[8], wk[8];
    int warp = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);
    if (lane == 0) {
        wq[warp] = q_sum;
        wk[warp] = k_sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float tq = 0.0f, tk = 0.0f;
        int nwarps = ((int)blockDim.x + 31) >> 5;
        for (int w = 0; w < nwarps; w++) {
            tq += wq[w];
            tk += wk[w];
        }
        qk[0] = tq;
        qk[1] = tk;
    }
    __syncthreads();
    float q_inv = rsqrtf(qk[0] / (float)head_dim + epsilon);
    float k_inv = rsqrtf(qk[1] / (float)head_dim + epsilon);

    if (rope_half > 0 && split_half) {
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float q0 = __half2float(q_base[d]) * q_inv * q_norm[d];
            float k0 = __half2float(k_base[d]) * k_inv * k_norm[d];
            if (d < rope_half) {
                size_t pair = d + rope_half;
                float q1 = __half2float(q_base[pair]) * q_inv * q_norm[pair];
                float k1 = __half2float(k_base[pair]) * k_inv * k_norm[pair];
                float c = c_row[d];
                float s = s_row[d];
                q0 = q0 * c - q1 * s;
                k0 = k0 * c - k1 * s;
            } else if (d < rope_half * 2) {
                size_t pair = d - rope_half;
                float q1 = __half2float(q_base[pair]) * q_inv * q_norm[pair];
                float k1 = __half2float(k_base[pair]) * k_inv * k_norm[pair];
                float c = c_row[pair];
                float s = s_row[pair];
                q0 = q0 * c + q1 * s;
                k0 = k0 * c + k1 * s;
            }
            q_out[d] = __float2half(q0);
            k_out[d] = __float2half(k0);
        }
    } else if (rope_half > 0) {
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            sq[d] = __half2float(q_base[d]) * q_inv * q_norm[d];
            sk[d] = __half2float(k_base[d]) * k_inv * k_norm[d];
        }
        __syncthreads();
        for (size_t p = threadIdx.x; p < rope_half; p += blockDim.x) {
            size_t i0 = p * 2u;
            size_t i1 = i0 + 1u;
            float c = c_row[p];
            float s = s_row[p];
            float q0 = sq[i0], q1 = sq[i1];
            float k0 = sk[i0], k1 = sk[i1];
            sq[i0] = q0 * c - q1 * s;
            sq[i1] = q0 * s + q1 * c;
            sk[i0] = k0 * c - k1 * s;
            sk[i1] = k0 * s + k1 * c;
        }
        __syncthreads();
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            q_out[d] = __float2half(sq[d]);
            k_out[d] = __float2half(sk[d]);
        }
    } else {
        for (size_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            q_out[d] =
                __float2half(__half2float(q_base[d]) * q_inv * q_norm[d]);
            k_out[d] =
                __float2half(__half2float(k_base[d]) * k_inv * k_norm[d]);
        }
    }
}

__global__ void h3_vae_f16_unpad_kernel(const __half *__restrict__ src,
                                        __half *__restrict__ dst, size_t n_batch,
                                        size_t seq, size_t padded,
                                        size_t inner) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n_batch * seq * inner;
    size_t bi, s, c;
    if (idx >= total) return;
    c = idx % inner;
    s = (idx / inner) % seq;
    bi = idx / (seq * inner);
    dst[idx] = src[bi * padded * inner + s * inner + c];
}

int h3_cuda_vae_bias_residual_scale_f32_dev(
    h3_cuda *cuda, float *d_x, const float *d_branch, const float *d_bias,
    const float *d_scale, size_t rows, size_t width, char *error,
    size_t error_size) {
    size_t total;
    cudaStream_t st;
    if (!cuda || !d_x || !d_branch || !d_scale || rows == 0 || width == 0) {
        set_error(error, error_size, "invalid vae bias_residual");
        return 0;
    }
    total = rows * width;
    st = vae_st(cuda);
    h3_vae_bias_residual_scale_kernel<<<(unsigned)((total + 255) / 256), 256, 0,
                                        st>>>(d_x, d_branch, d_bias, d_scale,
                                              rows, width);
    if (cudaGetLastError() != cudaSuccess) {
        set_error(error, error_size, "vae bias_residual kernel");
        return 0;
    }
    return 1;
}

int h3_cuda_vae_bias_silu_mul_f32_dev(h3_cuda *cuda, const float *d_fused,
                                      const float *d_bias, size_t rows,
                                      size_t width, float *d_out, char *error,
                                      size_t error_size) {
    size_t total;
    cudaStream_t st;
    if (!cuda || !d_fused || !d_out || rows == 0 || width == 0) {
        set_error(error, error_size, "invalid vae bias_silu");
        return 0;
    }
    total = rows * width;
    st = vae_st(cuda);
    h3_vae_bias_silu_mul_kernel<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(
        d_fused, d_bias, d_out, rows, width);
    if (cudaGetLastError() != cudaSuccess) {
        set_error(error, error_size, "vae bias_silu kernel");
        return 0;
    }
    return 1;
}

int h3_cuda_vae_rms_norm_f32_to_f16_dev(h3_cuda *cuda, const float *d_x,
                                        const float *d_w, size_t rows,
                                        size_t width, float eps,
                                        void *d_out_half, char *error,
                                        size_t error_size) {
    if (!cuda || !d_x || !d_w || !d_out_half || rows == 0 || width == 0) {
        set_error(error, error_size, "invalid vae rms f16");
        return 0;
    }
    h3_vae_rms_f32_to_f16_kernel<<<(unsigned)rows, 256, 0, vae_st(cuda)>>>(
        d_x, d_w, (__half *)d_out_half, rows, width, eps);
    if (cudaGetLastError() != cudaSuccess) {
        set_error(error, error_size, "vae rms f16 kernel");
        return 0;
    }
    return 1;
}

int h3_cuda_vae_bias_add_f16_dev(h3_cuda *cuda, void *d_y_half,
                                 const float *d_bias, size_t rows, size_t width,
                                 char *error, size_t error_size) {
    size_t total;
    if (!cuda || !d_y_half || !d_bias || rows == 0 || width == 0) {
        set_error(error, error_size, "invalid vae bias f16");
        return 0;
    }
    total = rows * width;
    h3_vae_bias_add_f16_kernel<<<(unsigned)((total + 255) / 256), 256, 0,
                                 vae_st(cuda)>>>((__half *)d_y_half, d_bias,
                                                 rows, width);
    if (cudaGetLastError() != cudaSuccess) {
        set_error(error, error_size, "vae bias f16 kernel");
        return 0;
    }
    return 1;
}

int h3_cuda_vae_bias_silu_mul_f16_dev(h3_cuda *cuda, const void *d_fused_half,
                                      const float *d_bias, size_t rows,
                                      size_t width, void *d_out_half,
                                      char *error, size_t error_size) {
    size_t total;
    if (!cuda || !d_fused_half || !d_out_half || rows == 0 || width == 0) {
        set_error(error, error_size, "invalid vae silu f16");
        return 0;
    }
    total = rows * width;
    h3_vae_bias_silu_mul_f16_kernel<<<(unsigned)((total + 255) / 256), 256, 0,
                                      vae_st(cuda)>>>(
        (const __half *)d_fused_half, d_bias, (__half *)d_out_half, rows,
        width);
    if (cudaGetLastError() != cudaSuccess) {
        set_error(error, error_size, "vae silu f16 kernel");
        return 0;
    }
    return 1;
}

int h3_cuda_vae_bias_residual_scale_f16_dev(
    h3_cuda *cuda, float *d_x, const void *d_branch_half, const float *d_bias,
    const float *d_scale, size_t rows, size_t width, char *error,
    size_t error_size) {
    size_t total;
    if (!cuda || !d_x || !d_branch_half || !d_scale || rows == 0 ||
        width == 0) {
        set_error(error, error_size, "invalid vae residual f16");
        return 0;
    }
    total = rows * width;
    h3_vae_bias_residual_scale_f16_kernel<<<(unsigned)((total + 255) / 256), 256,
                                            0, vae_st(cuda)>>>(
        d_x, (const __half *)d_branch_half, d_bias, d_scale, rows, width);
    if (cudaGetLastError() != cudaSuccess) {
        set_error(error, error_size, "vae residual f16 kernel");
        return 0;
    }
    return 1;
}

extern "C" int h3_cuda_vae_launch_qkv_rope_f16_in(
    const void *d_qkv_half, const float *d_q_norm, const float *d_k_norm,
    const float *d_rope_cos, const float *d_rope_sin, void *d_q, void *d_k,
    void *d_v, size_t seq, size_t seq_padded, size_t heads, size_t head_dim,
    size_t rope_half, float epsilon, int grouped, int split_half, void *stream,
    char *error, size_t error_size) {
    dim3 grid((unsigned)heads, (unsigned)seq_padded, 1);
    cudaStream_t st = stream ? (cudaStream_t)stream : (cudaStream_t)0;
    if (!d_qkv_half || !d_q_norm || !d_k_norm || !d_rope_cos || !d_rope_sin ||
        !d_q || !d_k || !d_v || seq == 0 || heads == 0 || head_dim == 0) {
        set_error(error, error_size, "invalid vae rope f16_in");
        return 0;
    }
    h3_vae_qkv_rope_f16_in_kernel<<<grid, 128, 0, st>>>(
        (const __half *)d_qkv_half, d_q_norm, d_k_norm, d_rope_cos, d_rope_sin,
        (__half *)d_q, (__half *)d_k, (__half *)d_v, seq, seq_padded, heads,
        head_dim, rope_half, epsilon, grouped, split_half);
    if (cudaGetLastError() != cudaSuccess) {
        set_error(error, error_size, "vae rope f16_in kernel");
        return 0;
    }
    return 1;
}

extern "C" int h3_cuda_vae_launch_f16_unpad(const void *d_src, void *d_dst,
                                            size_t n_batch, size_t seq,
                                            size_t padded, size_t inner,
                                            void *stream, char *error,
                                            size_t error_size) {
    size_t total;
    cudaStream_t st = stream ? (cudaStream_t)stream : (cudaStream_t)0;
    if (!d_src || !d_dst || n_batch == 0 || seq == 0 || padded < seq ||
        inner == 0) {
        set_error(error, error_size, "invalid vae f16 unpad");
        return 0;
    }
    if (padded == seq && n_batch == 1) {
        if (cudaMemcpyAsync(d_dst, d_src, seq * inner * 2,
                            cudaMemcpyDeviceToDevice, st) != cudaSuccess) {
            set_error(error, error_size, "vae f16 unpad memcpy");
            return 0;
        }
        return 1;
    }
    total = n_batch * seq * inner;
    h3_vae_f16_unpad_kernel<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(
        (const __half *)d_src, (__half *)d_dst, n_batch, seq, padded, inner);
    if (cudaGetLastError() != cudaSuccess) {
        set_error(error, error_size, "vae f16 unpad kernel");
        return 0;
    }
    return 1;
}
