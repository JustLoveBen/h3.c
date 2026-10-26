#!/usr/bin/env python3
"""Re-layout NVFP4 weight SF to the sm_120 cuBLASLt layout proven by e2e:
  tile = (r//128)*ncb + (c//4)          # stride=ncb (NOT ceil32/ncb_rd)
  off  = tile*512 + 16*((r%128)%32) + 4*((r%128)//32) + (c%4)  # 32x4

Accepts either prior buggy layout as input:
  - linear+ncb_rd  (original quant)
  - 32x4+ncb_rd    (post first reswizzle)
and writes tight 32x4+ncb (size rb*ncb*512).
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

def off_linear_ncb_rd(r, c, ncb_rd):
    tile = (r // 128) * ncb_rd + (c // 4)
    return tile * 512 + (r % 128) * 4 + (c % 4)

def off_32x4_ncb_rd(r, c, ncb_rd):
    r128 = r % 128
    tile = (r // 128) * ncb_rd + (c // 4)
    return tile * 512 + 16 * (r128 % 32) + 4 * (r128 // 32) + (c % 4)

def off_32x4_ncb(r, c, ncb):
    r128 = r % 128
    tile = (r // 128) * ncb + (c // 4)
    return tile * 512 + 16 * (r128 % 32) + 4 * (r128 // 32) + (c % 4)

def convert_blob(blob: bytes, N: int, K: int, src_layout: str) -> bytes:
    nc = K // 16
    ncb = nc // 4
    ncb_rd = (ncb + 31) & ~31
    rb = (N + 127) // 128
    expect_rd = rb * ncb_rd * 512
    expect_ncb = rb * ncb * 512
    src = np.frombuffer(blob, dtype=np.uint8)
    if src_layout == 'auto':
        if len(blob) == expect_rd:
            # distinguish linear vs 32x4 by checking a probe row
            # If already 32x4+ncb (tight), len == expect_ncb
            src_layout = '32x4_ncb_rd'  # most recent buggy
        elif len(blob) == expect_ncb:
            # already tight — assume needs 32x4 from linear tight OR already good;
            # treat as linear_ncb (rare) by checking; default identity 32x4_ncb
            src_layout = '32x4_ncb'
        else:
            raise SystemExit(f'SF size {len(blob)} not in ({expect_ncb}, {expect_rd}) N={N} K={K}')
    dst = np.zeros(expect_ncb, dtype=np.uint8)
    if src_layout == '32x4_ncb':
        if len(blob) != expect_ncb:
            raise SystemExit('32x4_ncb size mismatch')
        return blob  # already final
    if len(blob) != expect_rd:
        raise SystemExit(f'{src_layout} expect {expect_rd} got {len(blob)}')
    reader = off_linear_ncb_rd if src_layout == 'linear_ncb_rd' else off_32x4_ncb_rd
    for r in range(N):
        for c in range(nc):
            dst[off_32x4_ncb(r, c, ncb)] = src[reader(r, c, ncb_rd)]
    return dst.tobytes()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--src-layout', default='auto',
                    choices=['auto', 'linear_ncb_rd', '32x4_ncb_rd', '32x4_ncb'])
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
        fp4_shape = tensors[fp4_k][0]['shape']
        N, Kh2 = fp4_shape
        K = Kh2 * 2
        new_raw = convert_blob(raw, N, K, args.src_layout)
        # update dtype size in header meta if shape encoded as flat
        meta_t = dict(v)
        # keep shape if present; many SF tensors are 1-D
        if 'shape' in meta_t:
            meta_t = dict(meta_t)
            # flatten size changes — encode as 1-D byte vector
            meta_t['shape'] = [len(new_raw)]
        tensors[k] = (meta_t, new_raw)
        n_fix += 1
        print(f'converted {k}: N={N} K={K} {len(raw)} -> {len(new_raw)}')
    keys = [k for k in header.keys() if k != '__metadata__']
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
