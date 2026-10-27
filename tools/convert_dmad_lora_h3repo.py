#!/usr/bin/env python3
"""Convert ZhengmingYu/DMAD Diffusers-layout LoRA → h3repo layout.

Observed key layout (ZhengmingYu full_critic, 624 tensors):
  transformer_blocks.{i}.attn.to_{q,k,v,out.0}.lora.{down,up}.weight
  transformer_blocks.{i}.ff.net.{0.proj,2}.lora_{A,B}.weight
  token_refiner.refiner_blocks.{i}.… (same)
No leading `transformer.` prefix. Attn uses down/up; FFN uses A/B.

Geometry matches tools/convert_pdmd_lora_h3repo.py:
  - Q/K/V → contiguous qkv_proj (A concat, B block-diag, rank 384)
  - FC1 B: (up,gate) → (gate,up) half-swap
  - metadata h3_load_refiner=1
"""
from __future__ import annotations

import argparse
import os
import re

import torch
from safetensors.torch import load_file, save_file


def convert(raw_path: str, out_path: str, source_label: str) -> None:
    raw = load_file(raw_path)
    out: dict[str, torch.Tensor] = {}

    def pick(prefix: str, module: str):
        candidates = [
            (f"{prefix}.{module}.lora.down.weight", f"{prefix}.{module}.lora.up.weight"),
            (f"{prefix}.{module}.lora_A.weight", f"{prefix}.{module}.lora_B.weight"),
        ]
        for a, b in candidates:
            if a in raw and b in raw:
                return raw[a].contiguous(), raw[b].contiguous()
        raise KeyError(f"missing LoRA A/B for {prefix}.{module}")

    main_idx = sorted(
        {int(m.group(1)) for k in raw if (m := re.match(r"transformer_blocks\.(\d+)\.", k))}
    )
    ref_idx = sorted(
        {int(m.group(1)) for k in raw if (m := re.match(r"token_refiner\.refiner_blocks\.(\d+)\.", k))}
    )
    if not main_idx:
        raise SystemExit("no transformer_blocks.* keys found")

    def fuse_qkv(prefix: str, out_prefix: str) -> None:
        Aq, Bq = pick(prefix, "attn.to_q")
        Ak, Bk = pick(prefix, "attn.to_k")
        Av, Bv = pick(prefix, "attn.to_v")
        A = torch.cat([Aq, Ak, Av], dim=0)
        o, r = Bq.shape
        Bd = torch.zeros((o * 3, r * 3), dtype=Bq.dtype)
        Bd[0:o, 0:r] = Bq
        Bd[o : 2 * o, r : 2 * r] = Bk
        Bd[2 * o : 3 * o, 2 * r : 3 * r] = Bv
        out[f"{out_prefix}.attn.qkv_proj.lora_A.weight"] = A
        out[f"{out_prefix}.attn.qkv_proj.lora_B.weight"] = Bd
        Ao, Bo = pick(prefix, "attn.to_out.0")
        out[f"{out_prefix}.attn.out_proj.lora_A.weight"] = Ao
        out[f"{out_prefix}.attn.out_proj.lora_B.weight"] = Bo

    def mlp(prefix: str, out_prefix: str) -> None:
        A1, B1 = pick(prefix, "ff.net.0.proj")
        half = B1.shape[0] // 2
        B1 = torch.cat([B1[half:], B1[:half]], dim=0)
        out[f"{out_prefix}.mlp.fc1.lora_A.weight"] = A1
        out[f"{out_prefix}.mlp.fc1.lora_B.weight"] = B1
        A2, B2 = pick(prefix, "ff.net.2")
        out[f"{out_prefix}.mlp.fc2.lora_A.weight"] = A2
        out[f"{out_prefix}.mlp.fc2.lora_B.weight"] = B2

    for i in main_idx:
        p = f"transformer_blocks.{i}"
        fuse_qkv(p, f"blocks.{i}")
        mlp(p, f"blocks.{i}")
    for i in ref_idx:
        p = f"token_refiner.refiner_blocks.{i}"
        fuse_qkv(p, f"token_refiner.blocks.{i}")
        mlp(p, f"token_refiner.blocks.{i}")

    leftover = [
        k
        for k in raw
        if not re.match(
            r"(transformer_blocks|token_refiner\.refiner_blocks)\.\d+\."
            r"(attn\.to_(q|k|v|out\.0)|ff\.net\.(0\.proj|2))\.(lora\.(down|up)|lora_[AB])\.weight$",
            k,
        )
    ]
    if leftover:
        raise SystemExit(f"unhandled keys ({len(leftover)}): {leftover[:12]}")

    meta = {
        "format": "pt",
        "lora_scale": "1.0",
        "h3_load_refiner": "1",
        "h3_convert": "dmad Diffusers (no transformer. prefix; attn down/up + ff A/B) → qkv block-diag; fc1 B swap; refiner",
        "source": source_label,
    }
    to_save = {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_file(to_save, out_path, metadata=meta)
    print(
        f"wrote {out_path} keys={len(to_save)} bytes={os.path.getsize(out_path)} "
        f"blocks={len(main_idx)} refiner={len(ref_idx)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-label", default="")
    args = ap.parse_args()
    convert(args.inp, args.out, args.source_label or os.path.basename(args.inp))


if __name__ == "__main__":
    main()
