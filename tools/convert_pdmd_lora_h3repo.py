#!/usr/bin/env python3
"""Convert pdmd2026 Diffusers-layout PDMD LoRA → h3repo layout.

Verified round-trip against pdmd_2NFE_lora_model_0.h3repo.safetensors (maxdiff 0.0):
  - Q/K/V → contiguous qkv_proj: A concat on axis0 (3r,in); B block-diag (3out,3r)
  - FC1 B: (up,gate) → (gate,up) half-swap on out dim
  - out_proj / fc2: copy
  - token_refiner.refiner_blocks.* → token_refiner.blocks.* (same rules)
  - metadata: h3_load_refiner=1, lora_scale=1.0, bf16

Usage:
  python3 tools/convert_pdmd_lora_h3repo.py \\
    --in path/to/lora_model_0.safetensors \\
    --out path/to/pdmd_XNFE_lora_model_0.h3repo.safetensors \\
    [--source-label "..."] [--verify-against gold.h3repo.safetensors]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file


def convert(raw_path: str, out_path: str, source_label: str, verify_against: str | None) -> None:
    raw = load_file(raw_path)
    out: dict[str, torch.Tensor] = {}
    main_idx = sorted(
        {int(m.group(1)) for k in raw if (m := re.match(r"transformer\.transformer_blocks\.(\d+)\.", k))}
    )
    ref_idx = sorted(
        {
            int(m.group(1))
            for k in raw
            if (m := re.match(r"transformer\.token_refiner\.refiner_blocks\.(\d+)\.", k))
        }
    )

    def get(prefix: str, name: str) -> torch.Tensor:
        return raw[f"{prefix}.{name}"].contiguous()

    def fuse_qkv(prefix: str, out_prefix: str) -> None:
        Aq, Ak, Av = [get(prefix, f"attn.to_{x}.lora_A.weight") for x in "qkv"]
        Bq, Bk, Bv = [get(prefix, f"attn.to_{x}.lora_B.weight") for x in "qkv"]
        A = torch.cat([Aq, Ak, Av], dim=0)
        o, r = Bq.shape
        Bd = torch.zeros((o * 3, r * 3), dtype=Bq.dtype)
        Bd[0:o, 0:r] = Bq
        Bd[o : 2 * o, r : 2 * r] = Bk
        Bd[2 * o : 3 * o, 2 * r : 3 * r] = Bv
        out[f"{out_prefix}.attn.qkv_proj.lora_A.weight"] = A
        out[f"{out_prefix}.attn.qkv_proj.lora_B.weight"] = Bd
        out[f"{out_prefix}.attn.out_proj.lora_A.weight"] = get(prefix, "attn.to_out.0.lora_A.weight")
        out[f"{out_prefix}.attn.out_proj.lora_B.weight"] = get(prefix, "attn.to_out.0.lora_B.weight")

    def mlp(prefix: str, out_prefix: str) -> None:
        A1 = get(prefix, "ff.net.0.proj.lora_A.weight")
        B1 = get(prefix, "ff.net.0.proj.lora_B.weight")
        half = B1.shape[0] // 2
        B1 = torch.cat([B1[half:], B1[:half]], dim=0)  # (up,gate)->(gate,up)
        out[f"{out_prefix}.mlp.fc1.lora_A.weight"] = A1
        out[f"{out_prefix}.mlp.fc1.lora_B.weight"] = B1
        out[f"{out_prefix}.mlp.fc2.lora_A.weight"] = get(prefix, "ff.net.2.lora_A.weight")
        out[f"{out_prefix}.mlp.fc2.lora_B.weight"] = get(prefix, "ff.net.2.lora_B.weight")

    for i in main_idx:
        p = f"transformer.transformer_blocks.{i}"
        o = f"blocks.{i}"
        fuse_qkv(p, o)
        mlp(p, o)
    for i in ref_idx:
        p = f"transformer.token_refiner.refiner_blocks.{i}"
        o = f"token_refiner.blocks.{i}"
        fuse_qkv(p, o)
        mlp(p, o)

    leftover = [
        k
        for k in raw
        if not (
            re.match(
                r"transformer\.transformer_blocks\.\d+\.attn\.to_(q|k|v|out\.0)\.lora_[AB]\.weight$",
                k,
            )
            or re.match(
                r"transformer\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)\.lora_[AB]\.weight$",
                k,
            )
            or re.match(
                r"transformer\.token_refiner\.refiner_blocks\.\d+\.attn\.to_(q|k|v|out\.0)\.lora_[AB]\.weight$",
                k,
            )
            or re.match(
                r"transformer\.token_refiner\.refiner_blocks\.\d+\.ff\.net\.(0\.proj|2)\.lora_[AB]\.weight$",
                k,
            )
        )
    ]
    if leftover:
        raise SystemExit(f"unhandled keys ({len(leftover)}): {leftover[:8]}")

    if verify_against:
        gold = load_file(verify_against)
        if set(out.keys()) != set(gold.keys()):
            raise SystemExit(
                f"key mismatch +{len(set(out)-set(gold))} -{len(set(gold)-set(out))}"
            )
        maxd = 0.0
        worst = None
        for k in out:
            d = float((out[k].float() - gold[k].float()).abs().max())
            if d > maxd:
                maxd = d
                worst = k
        print(f"VERIFY vs {verify_against}: maxdiff={maxd} worst={worst}")
        if maxd > 0:
            raise SystemExit("verify failed")

    meta = {
        "format": "pt",
        "lora_scale": "1.0",
        "h3_load_refiner": "1",
        "h3_convert": "pdmd contiguous qkv block-diag; fc1 B (up,gate)->(gate,up); refiner included",
        "source": source_label,
    }
    to_save = {k: v.to(torch.bfloat16).contiguous() for k, v in out.items()}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_file(to_save, out_path, metadata=meta)
    print(f"wrote {out_path} keys={len(to_save)} bytes={os.path.getsize(out_path)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-label", default="")
    ap.add_argument("--verify-against", default=None)
    args = ap.parse_args()
    label = args.source_label or os.path.basename(args.inp)
    convert(args.inp, args.out, label, args.verify_against)


if __name__ == "__main__":
    main()
