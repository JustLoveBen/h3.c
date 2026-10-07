#!/usr/bin/env python3
"""Encode evals/768p prompts to Qwen token-id files for h3_cuda_generate.

CUDA generate has no --prompt text path. This writes space-separated ids that
h3_token_ids_load_file can parse. Does not silently truncate.

Resolution of tokenizer.json (first hit wins):
  1. --tokenizer / H3_TOKENIZER_JSON
  2. $H3_MODELS/tokenizer.json
  3. $H3_MODELS/FL2VA/tokenizer/tokenizer.json
  4. <repo>/MiniMax-H3/FL2VA/tokenizer/tokenizer.json

Requires the `tokenizers` package (or transformers). No passwords. No cafe ids.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def find_tokenizer_json(explicit: str | None) -> Path:
    cands = []
    if explicit:
        cands.append(Path(explicit))
    env = os.environ.get("H3_TOKENIZER_JSON")
    if env:
        cands.append(Path(env))
    models = os.environ.get("H3_MODELS")
    if models:
        cands.append(Path(models) / "tokenizer.json")
        cands.append(Path(models) / "FL2VA" / "tokenizer" / "tokenizer.json")
    repo = Path(__file__).resolve().parents[1]
    cands.append(repo / "MiniMax-H3" / "FL2VA" / "tokenizer" / "tokenizer.json")
    for p in cands:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "tokenizer.json not found. Set H3_TOKENIZER_JSON or pass --tokenizer. "
        "Do not fall back to cafe ids or --token-canary for 768p eval."
    )


def load_encode(tok_json: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit(
            "need Python package `tokenizers` (or transformers). "
            f"import failed: {exc}"
        ) from exc
    tok = Tokenizer.from_file(str(tok_json))

    def encode(text: str) -> list[int]:
        return list(tok.encode(text).ids)

    return encode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-id", required=True, help="e.g. p01_sprint")
    ap.add_argument(
        "--prompts-dir",
        default="",
        help="default: <repo>/evals/768p/prompts",
    )
    ap.add_argument("--out", default="", help="default: evals/768p/ids/<id>.ids")
    ap.add_argument("--tokenizer", default="", help="path to tokenizer.json")
    ap.add_argument("--text-file", default="", help="override prompt file")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else (
        repo / "evals" / "768p" / "prompts"
    )
    src = Path(args.text_file) if args.text_file else (
        prompts_dir / f"{args.prompt_id}.txt"
    )
    if not src.is_file():
        print(f"prompt file missing: {src}", file=sys.stderr)
        return 2
    text = src.read_text(encoding="utf-8").strip()
    if not text:
        print(f"empty prompt: {src}", file=sys.stderr)
        return 2
    try:
        tok_json = find_tokenizer_json(args.tokenizer or None)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    encode = load_encode(tok_json)
    ids = encode(text)
    if not ids:
        print("tokenizer produced 0 ids", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else (
        repo / "evals" / "768p" / "ids" / f"{args.prompt_id}.ids"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(" ".join(str(i) for i in ids) + "\n", encoding="utf-8")
    print(f"wrote {out} tokens={len(ids)} tokenizer={tok_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
