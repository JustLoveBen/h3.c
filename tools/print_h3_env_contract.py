#!/usr/bin/env python3
"""Print effective H3_* env contract for P0 / fewstep / lifecycle gates.

Inspired by DeepInfer's environ.py dump pattern (see
docs/research/2026-10-26-deepinfer-deeper-pass.md §1.11): one place to see
what the shell *actually* has set before a generate run.

Offline only — does not talk to the GPU binary. Exit 0 on print; with
--expect-p0-visual exit 1 if any P0-forbidden var is set "on".

Usage:
  python3 tools/print_h3_env_contract.py
  python3 tools/print_h3_env_contract.py --expect-p0-visual
  python3 tools/print_h3_env_contract.py --expect-p0-visual --arm nvfp4
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

# (name, role, p0_visual_rule)
# p0_visual_rule: None/"info" = informational; "unset" = must be unset/empty;
# "unset_or_0" = unset/empty/0; "arm"/"arm_file" = depends on --arm
Entry = Tuple[str, str, Optional[str]]

CONTRACT: List[Entry] = [
    ("H3_DIT_NVFP4", "DiT NVFP4 experimental arm (1=on)", "arm"),
    ("H3_DIT_NVFP4_FILE", "NVFP4 sidecar safetensors path", "arm_file"),
    ("H3_DIT_NVFP4_KEEP_INT8",
     "Hybrid no-retrain: CSV of linears kept INT8 (qkv,out,fc1,fc2,attn,mlp,all)",
     "optional"),
    ("H3_DIT_NVFP4_HOST_DEQUANT",
     "DISARM: Sep-13 NVFP4→INT8 host dequant bypass (default OFF)", "unset"),
    ("H3_DIT_NVFP4_CHAIN", "DISARM/WIP: NVFP4 FC1→FC2 chain (unset)", "unset"),
    ("H3_DIT_NVFP4_CHAIN_FC1_F32", "chain FC1 f32 diag", "unset"),
    ("H3_DIT_NVFP4_CHAIN_DIAG", "chain diagnostics", "unset"),
    ("H3_SDPA_CUDNN_CUDA_GRAPH", "cuDNN SDPA CUDA graph (P0 default off)", "unset_or_0"),
    ("H3_TEXT_H_LOAD", "load refined text_h (velocity/md5 only; not default visual)", "info"),
    ("H3_TE_PROMPT_CACHE", "opt-in TE prompt emb disk cache dir (--te-cache / C062)", "info"),
    ("H3_TE_CACHE_REQUIRE_HIT", "1=MISS is FAIL_ENV when TE cache on (warm first)", "info"),
    ("H3_DIT_HOST_STAGE_TE", "DiT host-stage overlap with TE", "info"),
    ("H3_DIT_HOST_STAGE_WAIT_BLOCKS", "host-stage wait blocks", "info"),
    ("H3_VAE_PREWARM_DEV", "VAE device prewarm during DiT", "info"),
    ("H3_FEWSTEP_CONTRACT", "fewstep contract id (pdmd4/pdmd2/dmad4/...)", "info"),
    ("H3_FEWSTEP_PRESET", "one-shot preset binding steps/cfg/shift/sampler", "info"),
    ("H3_GEN_MEM_TRACE", "per-phase mem trace (forbid during wall-clock KPI)", "info"),
    ("H3_SKIP_PPM", "skip PPM dumps", "info"),
    ("H3_DIAG_W_CKSUM", "weight checksum diag", "unset"),
    ("H3_DIAG_BLK0", "block0 diag", "unset"),
    ("H3_DUMP_LATENT", "dump latent once (gen/score split)", "info"),
    ("H3_REPLAY_LATENT", "replay dumped latent", "info"),
    ("H3_DUMP_DIR", "oracle dump dir (TE/velocity; not gold retune)", "info"),
    ("H3_DUMP_ONLY", "1=stop after step0 forward dump", "info"),
    ("H3_LIFECYCLE_TRACE", "1=print C064 lifecycle whitelist/transitions", "info"),
    ("H3_FORBID_HOT_SYNC", "1=abort on H3_SYNC_HOT device sync (C070)", "info"),
    ("H3_CUDA_L2_PIN", "1=pin ConvRot Hadamard in persisting L2 (C017 bit-id)", "info"),
    ("H3_ROPE_PROFILE", "1=log dit/vae rope ms + memo hit/miss", "info"),
    ("H3_ROPE_MEMO", "0=disable RoPE host memo (fair KPI); default on", "info"),
    ("H3_TE_CACHE_AUTO", "1=auto te_cache under out/ when --repeat>=2 (C062)", "info"),
    ("H3_VRAM_LEDGER", "1=C040 cudaMemGetInfo VRAM buckets (after_cuda..after_vae)", "info"),
]


def raw(name: str) -> Optional[str]:
    return os.environ.get(name)


def is_on(val: Optional[str]) -> bool:
    """Match common C getenv patterns: present and first char == '1'."""
    return bool(val) and val[0] == "1"


def is_set(val: Optional[str]) -> bool:
    return val is not None and val != ""


def fmt_val(val: Optional[str]) -> str:
    if val is None:
        return "<unset>"
    if val == "":
        return "<empty>"
    if len(val) > 96:
        return val[:93] + "..."
    return val


def check_rule(name: str, rule: Optional[str], arm: str) -> Optional[str]:
    """Return violation message or None."""
    val = raw(name)
    if rule is None or rule == "info":
        return None
    if rule == "unset":
        if is_set(val):
            return f"{name} must be unset for P0 visual (got {fmt_val(val)!r})"
        return None
    if rule == "unset_or_0":
        if not is_set(val) or val == "0":
            return None
        return f"{name} must be unset or 0 for P0 visual (got {fmt_val(val)!r})"
    if rule == "arm":
        if arm == "int8":
            if is_on(val):
                return f"{name}=1 but --arm int8"
        elif arm == "nvfp4":
            if not is_on(val):
                return f"{name} must be 1 for --arm nvfp4 (got {fmt_val(val)!r})"
        return None
    if rule == "arm_file":
        if arm == "int8":
            if is_set(val):
                return f"{name} should be unset for --arm int8 (got {fmt_val(val)!r})"
        elif arm == "nvfp4":
            if not is_set(val):
                return f"{name} should be set for --arm nvfp4"
        return None
    return f"unknown rule {rule!r} for {name}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--expect-p0-visual",
        action="store_true",
        help="exit 1 if P0-forbidden env is active",
    )
    ap.add_argument(
        "--arm",
        choices=("int8", "nvfp4"),
        default="int8",
        help="expected DiT arm when --expect-p0-visual (default int8)",
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="emit a markdown table (for META.txt paste)",
    )
    args = ap.parse_args()

    violations: List[str] = []
    rows = []
    for name, role, rule in CONTRACT:
        val = raw(name)
        rows.append((name, fmt_val(val), role, rule or "info"))
        if args.expect_p0_visual:
            msg = check_rule(name, rule, args.arm)
            if msg:
                violations.append(msg)

    if args.markdown:
        print("| env | effective | role | p0_rule |")
        print("|---|---|---|---|")
        for name, fv, role, rule in rows:
            print(f"| `{name}` | `{fv}` | {role} | {rule} |")
    else:
        print(
            f"# H3 env contract  arm={args.arm}  "
            f"expect_p0={int(args.expect_p0_visual)}"
        )
        width = max(len(r[0]) for r in rows)
        for name, fv, role, rule in rows:
            flag = ""
            if args.expect_p0_visual:
                msg = check_rule(name, rule, args.arm)
                flag = "  OK" if msg is None else "  FAIL"
            print(f"{name:<{width}}  {fv:<48}  # {role}{flag}")

    if args.expect_p0_visual and violations:
        print("", file=sys.stderr)
        print("P0 visual contract violations:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(
            "Hint: DeepInfer-style bool trap — prefer unset or 0/1; "
            "do not rely on 'off'/'false' unless the C path parses them.",
            file=sys.stderr,
        )
        return 1

    if args.expect_p0_visual:
        print("# PROTO_ENV_OK arm=" + args.arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
