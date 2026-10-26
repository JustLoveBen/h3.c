#!/usr/bin/env python3
"""Re-swizzle NVFP4 weight SF from linear (r%128)*4+cc to 32x4 swizzle in-place
(or to a new file). Does not re-quantize — only permutes SF bytes.

Old (wrong): off = tile*512 + (r%128)*4 + (c%4)
New (sm_120 cuBLASLt): off = tile*512 + 16*((r%128)%32) + 4*((r%128)//32) + (c%4)
"""
import argparse, json, struct, sys
from pathlib import Path
import numpy as np

def load_st(path):
    with open(path, 'rb') as f:
        hl = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hl))
        data_start = 8 + hl
        tensors = {}
        for k, v in header.items():
            if k == '__metadata__':
                continue
            off0, off1 = v['data_offsets']
            f.seek(data_start + off0)
            tensors[k] = (v, f.read(off1 - off0))
        meta = header.get('__metadata__', {})
        return header, tensors, meta, data_start

def old_off(r, c, ncb_rd):
    tile = (r // 128) * ncb_rd + (c // 4)
    return tile * 512 + (r % 128) * 4 + (c % 4)

def new_off(r, c, ncb_rd):
    r128 = r % 128
    tile = (r // 128) * ncb_rd + (c // 4)
    return tile * 512 + 16 * (r128 % 32) + 4 * (r128 // 32) + (c % 4)

def reswizzle_blob(blob: bytes, N: int, K: int) -> bytes:
    nc = K // 16
    ncb = nc // 4
    ncb_rd = (ncb + 31) & ~31
    rb = (N + 127) // 128
    expect = rb * ncb_rd * 512
    if len(blob) != expect:
        raise SystemExit(f'SF size mismatch: got {len(blob)} expect {expect} (N={N} K={K})')
    src = np.frombuffer(blob, dtype=np.uint8)
    dst = np.zeros_like(src)
    # Only logical [0:N, 0:nc] carry values; padded region stays 0.
    for r in range(N):
        for c in range(nc):
            dst[new_off(r, c, ncb_rd)] = src[old_off(r, c, ncb_rd)]
    return dst.tobytes()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    header, tensors, meta, _ = load_st(args.src)
    n_fix = 0
    for k, (v, raw) in list(tensors.items()):
        if not k.endswith('.weight_sf'):
            continue
        base = k[:-len('.weight_sf')]
        fp4_k = base + '.weight_fp4'
        if fp4_k not in tensors:
            raise SystemExit(f'missing {fp4_k}')
        fp4_shape = tensors[fp4_k][0]['shape']  # [N, K/2]
        N, Kh2 = fp4_shape
        K = Kh2 * 2
        new_raw = reswizzle_blob(raw, N, K)
        tensors[k] = (v, new_raw)
        n_fix += 1
        print(f'reswizzled {k}: N={N} K={K} bytes={len(new_raw)}')
    # Rewrite safetensors
    # Rebuild with contiguous data_offsets
    keys = [k for k in header.keys() if k != '__metadata__']
    # Preserve original key order from header
    out_header = {}
    if meta:
        out_header['__metadata__'] = meta
    blobs = []
    cursor = 0
    for k in keys:
        v, raw = tensors[k]
        meta_t = dict(v)
        meta_t['data_offsets'] = [cursor, cursor + len(raw)]
        out_header[k] = meta_t
        blobs.append(raw)
        cursor += len(raw)
    hdr = json.dumps(out_header, separators=(',', ':')).encode('utf-8')
    # safetensors requires 8-byte aligned header
    pad = (8 - (len(hdr) % 8)) % 8
    hdr = hdr + b' ' * pad
    with open(args.dst, 'wb') as f:
        f.write(struct.pack('<Q', len(hdr)))
        f.write(hdr)
        for b in blobs:
            f.write(b)
    print(f'wrote {args.dst} ({n_fix} SF tensors, data={cursor} bytes)')

if __name__ == '__main__':
    main()
