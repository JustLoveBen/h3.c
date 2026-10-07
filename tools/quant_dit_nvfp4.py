#!/usr/bin/env python3
"""NVFP4 W4A4 offline weight quantization for the H3 DiT (Phase 1 P1.3).

Source: minimax_h3_fl2va_pruned_int8_convrot.safetensors
  Per DiT linear: weight [N,K] I8 (ConvRot baked) + weight_scale [N,1] F32.
Target: NVFP4 packed E2M1 [N,K/2] U8 + swizzled UE4M3 SF [N,K/16] + F32 scale2.

Math: y = W @ x  where W_f32 = I8 * per_row_scale. We requantize W_f32 into
per-16-element micro-blocks: W_f32[n, g*16+j] ~= E2M1(nib) * UE4M3(SF[n,g]) * 1.
ConvRot stays baked (rotation preserves per-block geometry; FP4 uses the same
Hadamard-rotated weights, so act-side rotation is unchanged).

SF swizzle = layout measured empirically on sm_120 cuBLASLt
(bench/nvfp4_phase0/diag_sf_layout.cu single-marker probe), MUST match the
runtime quantize kernel h3_cuda_quantize_act_nvfp4_kernel:
  tile = (r//128)*ncb + (c//4);  off = tile*512 + (r%128)*4 + (c%4)
  (ncb = (K/16)//4; K%64==0). 128 rows major, 4 mblock cols minor.

Usage:
  python quant_dit_nvfp4.py SRC.safetensors DST.safetensors [--blocks N]
Only blocks.{0..N-1} linears are converted; all other tensors copied through.
"""
import argparse
import numpy as np
import ml_dtypes  # registers bfloat16 dtype so numpy safe_open can read bf16 tensors
from safetensors import safe_open

try:
    import torch
    from safetensors.torch import save_file as st_save_file
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    from safetensors.numpy import save_file as st_save_file

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def f32_to_ue4m3(v):
    v = np.asarray(v, dtype=np.float32)
    out = np.zeros(v.shape, dtype=np.uint8)
    pos = v > 0
    vv = np.where(pos, v, 1.0)
    f, e = np.frexp(vv)          # v = f * 2^e, f in [0.5,1)
    E = e - 1 + 7
    m = np.rint((f * 2.0 - 1.0) * 8.0).astype(np.int32)
    carry = m == 8
    E = np.where(carry, E + 1, E)
    m = np.where(carry, 0, m)
    E = np.clip(E, 0, 14)
    norm = ((E << 3) | m).astype(np.uint8)
    sub = np.clip(np.rint(np.ldexp(vv, 6) * 8.0), 0, 7).astype(np.uint8)
    out[pos] = np.where((e - 1 + 7)[pos] <= 0, sub[pos], norm[pos])
    return out


def ue4m3_to_f32(b):
    b = np.asarray(b, dtype=np.uint8).astype(np.int32)
    e = (b >> 3) & 0xF
    m = b & 0x7
    sub = np.ldexp(m.astype(np.float32) / 8.0, -6)
    norm = np.ldexp(1.0 + m.astype(np.float32) / 8.0, e - 7)
    return np.where(e == 0, sub, norm)


def quant_linear(w_i8, per_row_scale):
    """w_i8 [N,K] int8, per_row_scale [N,1] f32 -> (packed [N,K/2] u8,
    sf_swz [sf_elems] u8)."""
    N, K = w_i8.shape
    assert K % 64 == 0, f"K={K} must be %64==0"
    nc = K // 16
    ncb = nc // 4
    W = w_i8.astype(np.float32) * per_row_scale.astype(np.float32)  # [N,K]
    Wb = W.reshape(N, nc, 16)
    amax = np.abs(Wb).max(axis=2)                       # [N,nc]
    scale = np.where(amax > 0, amax / 6.0, 1.0)
    sf = f32_to_ue4m3(scale)                            # [N,nc] u8
    sf = np.maximum(sf, 1)                              # clamp UE4M3 underflow: min nonzero = m1/e0 (2^-9)
    scale_r = ue4m3_to_f32(sf).reshape(N, nc, 1)
    q = Wb / scale_r                                    # [-?..?]
    a = np.abs(q)
    mag = np.zeros_like(a, dtype=np.uint8)
    for thr, val in [(0.25, 0), (0.75, 1), (1.25, 2), (1.75, 3),
                     (2.5, 4), (3.5, 5), (5.0, 6)]:
        pass
    # nearest E2M1 magnitude. E2M1 has no NaN encoding; mag=7 (bits 0111) is the
    # valid max value 6.0, so clamp to 7 (not 6).
    mag = np.digitize(a, [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]).astype(np.uint8)
    mag = np.clip(mag, 0, 7)
    nib = (np.where(q < 0, 8, 0).astype(np.uint8) | mag).astype(np.uint8)  # [N,nc,16]
    nib = nib.reshape(N, K)
    even = nib[:, 0::2]
    odd = nib[:, 1::2]
    packed = ((even << 4) | (odd & 0xF)).astype(np.uint8)   # [N,K/2] hi_first
    # swizzle SF [N,nc] -> blocked, measured sm_120 layout:
    # off = tile*512 + (r%128)*4 + (c%4), tile = (r//128)*ncb_rd + c//4
    # cuBLASLt reads a padded K-extent of ceil(ncb/32)*32 tile-cols per 128-row
    # tile (2048 K elems each); for K%2048!=0 the tail must exist and be 0
    # (a stale byte >= 0xFE is UE4M3 NaN and poisons the accumulator).
    ncb_rd = (ncb + 31) & ~31
    rb = (N + 127) // 128
    sf_pad = np.zeros((rb * 128, ncb_rd * 4), dtype=np.uint8)
    sf_pad[:N, :nc] = sf
    out = np.zeros(rb * ncb_rd * 512, dtype=np.uint8)
    r_idx = np.arange(rb * 128)
    c_idx = np.arange(ncb_rd * 4)
    tile = (r_idx // 128)[:, None] * ncb_rd + (c_idx // 4)[None, :]
    off = tile * 512 + (r_idx % 128)[:, None] * 4 + (c_idx % 4)[None, :]
    out[off.ravel()] = sf_pad.ravel()
    return packed, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--blocks", type=int, default=50)
    args = ap.parse_args()

    framework = "pt" if _HAS_TORCH else "numpy"
    LINEAR_SUFFIXES = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"]
    f = safe_open(args.src, framework=framework)
    keys = list(f.keys())
    out_tensors = {}
    n_conv = 0

    def to_np(t):
        if _HAS_TORCH:
            if t.dtype == torch.bfloat16:
                return t.float().numpy()
            return t.numpy()
        return t

    for k in keys:
        base = k[:-len(".weight")] if k.endswith(".weight") else None
        is_dit_lin = False
        if base:
            for suf in LINEAR_SUFFIXES:
                if base.endswith(suf) and base.startswith("blocks."):
                    try:
                        bidx = int(base.split(".")[1])
                    except ValueError:
                        bidx = -1
                    if 0 <= bidx < args.blocks:
                        is_dit_lin = True
                    break
        if is_dit_lin and f"{base}.weight_scale" in keys:
            w = to_np(f.get_tensor(k))
            sc = to_np(f.get_tensor(f"{base}.weight_scale"))
            packed, sf = quant_linear(w, sc)
            if _HAS_TORCH:
                out_tensors[f"{base}.weight_fp4"] = torch.from_numpy(packed)
                out_tensors[f"{base}.weight_sf"] = torch.from_numpy(sf)
                out_tensors[f"{base}.weight_scale2"] = torch.from_numpy(
                    np.array([1.0], np.float32))
            else:
                out_tensors[f"{base}.weight_fp4"] = packed
                out_tensors[f"{base}.weight_sf"] = sf
                out_tensors[f"{base}.weight_scale2"] = np.array([1.0], np.float32)
            n_conv += 1
            if n_conv % 20 == 0:
                print(f"  converted {n_conv} linears ...", flush=True)
        else:
            t = f.get_tensor(k)
            if _HAS_TORCH:
                out_tensors[k] = t if isinstance(t, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(to_np(t)))
            else:
                out_tensors[k] = to_np(t)
    st_save_file(out_tensors, args.dst)
    print(f"DONE: converted {n_conv} DiT linears -> {args.dst}")


if __name__ == "__main__":
    main()
