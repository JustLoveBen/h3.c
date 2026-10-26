#!/usr/bin/env python3
"""Offline protocol checker for P0 INT8 vs NVFP4 visual-gate generate.log files.

Also recognizes T3 fewstep log lines from --fewstep-preset / --fewstep-contract:
  fewstep_preset: ...
  fewstep_contract: ...
  fewstep_bound: ...
  sampler: selected=...

Hard gates (P0 int8|nvfp4):
  - sdpa_graph: wanted=0
  - no NVFP4 chain (H3_DIT_NVFP4_CHAIN / FC1->FC2 chained / chain scratch)
  - shape: line with sdpa= and seed=0x...
  - sigma schedule print
  - optional --expect-seed / --expect-skip-audio / --expect-live-te
  - recognizes inline "# H3_ENV_CONTRACT" from generate (C061);
    --expect-env-contract requires the marker
  - recognizes te_prompt_cache HIT/MISS/SKIP (C062);
    --expect-te-cache-hit requires a HIT line

Does not run the model. Exit 0 = protocol OK; 1 = fail; 2 = usage error.
See docs/research/2026-10-26-p0-visual-gate-runbook.md and
docs/research/2026-10-26-fewstep-preset.md.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_FEWSTEP_STEPS = {"pdmd4": 4, "pdmd2": 2, "dmad4": 4}
_FEWSTEP_SAMPLER = {
    "pdmd4": "velocity_euler",
    "pdmd2": "velocity_euler",
    "dmad4": "lcm_renoise",
}
_FEWSTEP_AUDIO = {"pdmd4": 3.0, "pdmd2": 3.0, "dmad4": 2.0}


def _read(path: Path) -> str:
    return path.read_text(errors="replace")


def _parse_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def _norm_seed(s: str) -> str:
    s = s.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    return s.lstrip("0") or "0"



def check_env_contract(
    text: str, *, require: bool = False
) -> tuple[list[str], list[str]]:
    """Recognize C061 inline # H3_ENV_CONTRACT block (soft by default)."""
    errs: list[str] = []
    warnings: list[str] = []
    m = re.search(r"^# H3_ENV_CONTRACT\b.*$", text, re.M)
    if not m:
        if require:
            errs.append("missing # H3_ENV_CONTRACT block (C061 inline)")
        return errs, warnings
    if not re.search(
        r"^cli seed=0x[0-9a-fA-F]+\s+skip_audio=\d+\s+fewstep_preset=\S+\s+"
        r"fewstep_contract=\S+\s+sampler=\S+\s+sdpa=\S+\s+sdpa_graph=\d+\s+"
        r"te_path=(live|H3_TEXT_H_LOAD)\s*$",
        text,
        re.M,
    ):
        warnings.append(
            "# H3_ENV_CONTRACT present but cli seed=... line missing/malformed"
        )
    m_ok = re.search(r"^# PROTO_ENV_OK\s+arm=(\S+)\s*$", text, re.M)
    m_fail = re.search(r"^# PROTO_ENV_FAIL\b", text, re.M)
    if m_fail:
        errs.append("H3_ENV_CONTRACT PROTO_ENV_FAIL present")
    if re.search(
        r"^# H3_ENV_CONTRACT\s+arm=\S+\s+expect_p0=1\s*$", text, re.M
    ):
        if not m_ok and not m_fail:
            warnings.append(
                "expect_p0=1 but neither PROTO_ENV_OK nor PROTO_ENV_FAIL"
            )
    return errs, warnings


def check_te_prompt_cache(
    text: str, *, require_hit: bool = False
) -> tuple[list[str], list[str]]:
    """Recognize C062 te_prompt_cache HIT/MISS/SKIP lines (soft by default).

    Grep-friendly shapes (generate):
      te_prompt_cache: dir=... require_hit=0|1 (opt-in; gold default off)
      te_prompt_cache: HIT dir=... key=... tokens=N layers=N dim=N
      te_prompt_cache: MISS wrote dir=... key=... tokens=N layers=N dim=N
      te_prompt_cache: MISS dir=... key=none ... reason=no_key
      te_prompt_cache: SKIP dir=... reason=vision_span
      te_prompt_cache: REQUIRE_HIT failed ...
    """
    errs: list[str] = []
    warnings: list[str] = []
    has_any = bool(re.search(r"^te_prompt_cache:", text, re.M))
    hit = bool(
        re.search(
            r"^te_prompt_cache:\s*HIT\s+dir=\S+\s+key=[0-9a-f]+\s+"
            r"tokens=\d+\s+layers=\d+\s+dim=\d+\s*$",
            text,
            re.M,
        )
    )
    miss = bool(re.search(r"^te_prompt_cache:\s*MISS\b", text, re.M))
    skip = bool(
        re.search(r"^te_prompt_cache:\s*SKIP\b.*vision_span", text, re.M)
    )
    req_fail = bool(
        re.search(r"^te_prompt_cache:\s*REQUIRE_HIT failed", text, re.M)
    )
    if req_fail:
        errs.append("te_prompt_cache REQUIRE_HIT failed (C062 hard gate)")
    if require_hit:
        if not has_any:
            errs.append(
                "missing te_prompt_cache lines (--expect-te-cache-hit)"
            )
        elif skip:
            warnings.append(
                "te_prompt_cache SKIP vision_span "
                "(HIT not applicable)"
            )
        elif not hit:
            errs.append(
                "te_prompt_cache HIT required "
                "(--expect-te-cache-hit; warm cache first)"
            )
        if miss and not hit:
            # redundant with above but keeps message explicit
            if "te_prompt_cache HIT required" not in errs[-1:]:
                errs.append("te_prompt_cache MISS while HIT required")
    elif has_any and miss and not hit:
        warnings.append(
            "te_prompt_cache MISS (ok for cold; set H3_TE_CACHE_REQUIRE_HIT "
            "or --expect-te-cache-hit for warm KPI)"
        )
    return errs, warnings


def check_fewstep(text: str, expect: str | None) -> tuple[list[str], list[str]]:
    """Validate fewstep_* / sampler lines.

    expect:
      None  — if any fewstep lines present, soft-check consistency
      none  — require absence of fewstep_preset / fewstep_bound
      pdmd4|pdmd2|dmad4 — require matching preset/bound/contract/sampler
    """
    errs: list[str] = []
    warnings: list[str] = []

    m_preset = re.search(
        r"^fewstep_preset:\s*id=(\S+)\s+steps=(\d+)\s+cfg=([0-9.eE+-]+)\s+"
        r"video_shift=([0-9.eE+-]+)\s+audio_shift=([0-9.eE+-]+)\s+"
        r"sampler=(\S+)\s+contract=(\S+)\s*$",
        text,
        re.M,
    )
    m_bound = re.search(
        r"^fewstep_bound:\s*contract=(\S+)\s+steps=(\d+)\s+cfg=([0-9.eE+-]+)\s+"
        r"video_shift=([0-9.eE+-]+)\s+audio_shift=([0-9.eE+-]+)\s+"
        r"sampler=(\S+)\s+speed_preset=(\S+)\s*$",
        text,
        re.M,
    )
    m_contract = re.search(
        r"^fewstep_contract:\s*id=(\S+)\s+steps=(\d+)\s+sampler=(\S+)\s+"
        r"video_shift=([0-9.eE+-]+)\s+audio_shift=([0-9.eE+-]+)",
        text,
        re.M,
    )
    m_sampler = re.search(
        r"^sampler:\s*selected=(\S+)\s+\(explicit=(\d+)\s+fewstep=(\S+)\)\s*$",
        text,
        re.M,
    )
    m_sigma_ok = re.search(r"^fewstep_contract:\s*sigma_ok\b", text, re.M)

    any_few = bool(m_preset or m_bound or m_contract)

    if expect == "none":
        if m_preset or m_bound:
            errs.append(
                "fewstep_preset/fewstep_bound present but --expect-fewstep=none"
            )
        return errs, warnings

    if expect in _FEWSTEP_STEPS:
        want = expect
        if not m_bound and not m_preset:
            errs.append(
                f"missing fewstep_preset / fewstep_bound for --expect-fewstep={want}"
            )
        if not m_contract:
            errs.append(f"missing fewstep_contract: id=... for {want}")
        if not m_sampler:
            errs.append(f"missing sampler: selected=... fewstep=... for {want}")

        def _id_ok(got: str | None, label: str) -> None:
            if got is None:
                return
            if got != want:
                errs.append(f"{label} id/contract={got!r} (want {want})")

        if m_preset:
            _id_ok(m_preset.group(1), "fewstep_preset")
            _id_ok(m_preset.group(7), "fewstep_preset.contract")
            steps = int(m_preset.group(2))
            if steps != _FEWSTEP_STEPS[want]:
                errs.append(
                    f"fewstep_preset steps={steps} (want {_FEWSTEP_STEPS[want]})"
                )
            samp = m_preset.group(6)
            if samp != _FEWSTEP_SAMPLER[want]:
                errs.append(
                    f"fewstep_preset sampler={samp!r} "
                    f"(want {_FEWSTEP_SAMPLER[want]})"
                )
            a = _parse_float(m_preset.group(5))
            v = _parse_float(m_preset.group(4))
            if v is not None and abs(v - 12.0) > 0.051:
                errs.append(f"fewstep_preset video_shift={v} (want 12)")
            if a is not None and abs(a - _FEWSTEP_AUDIO[want]) > 0.051:
                errs.append(
                    f"fewstep_preset audio_shift={a} "
                    f"(want {_FEWSTEP_AUDIO[want]})"
                )

        if m_bound:
            _id_ok(m_bound.group(1), "fewstep_bound")
            steps = int(m_bound.group(2))
            if steps != _FEWSTEP_STEPS[want]:
                errs.append(
                    f"fewstep_bound steps={steps} (want {_FEWSTEP_STEPS[want]})"
                )
            samp = m_bound.group(6)
            if samp != _FEWSTEP_SAMPLER[want]:
                errs.append(
                    f"fewstep_bound sampler={samp!r} "
                    f"(want {_FEWSTEP_SAMPLER[want]})"
                )
            a = _parse_float(m_bound.group(5))
            v = _parse_float(m_bound.group(4))
            if v is not None and abs(v - 12.0) > 0.051:
                errs.append(f"fewstep_bound video_shift={v} (want 12)")
            if a is not None and abs(a - _FEWSTEP_AUDIO[want]) > 0.051:
                errs.append(
                    f"fewstep_bound audio_shift={a} "
                    f"(want {_FEWSTEP_AUDIO[want]})"
                )
            speed = m_bound.group(7)
            if speed in ("turbo4", "taomate", "fast4"):
                errs.append(
                    f"fewstep_bound speed_preset={speed!r} "
                    "(fewstep forbids sigma-rewriting presets)"
                )

        if m_contract:
            _id_ok(m_contract.group(1), "fewstep_contract")
            csamp = m_contract.group(3)
            want_c = (
                "lcm+re-noise"
                if want == "dmad4"
                else "velocity_euler"
            )
            if csamp != want_c and csamp != _FEWSTEP_SAMPLER[want]:
                errs.append(
                    f"fewstep_contract sampler={csamp!r} (want {want_c})"
                )

        if m_sampler:
            sel = m_sampler.group(1)
            fs = m_sampler.group(3)
            if fs != want:
                errs.append(f"sampler fewstep={fs!r} (want {want})")
            if sel != _FEWSTEP_SAMPLER[want]:
                errs.append(
                    f"sampler selected={sel!r} "
                    f"(want {_FEWSTEP_SAMPLER[want]})"
                )

        if not m_sigma_ok:
            warnings.append(
                "no fewstep_contract: sigma_ok (run may have failed before "
                "schedule print, or older binary)"
            )

        if re.search(r"dit_nvfp4:\s*H3_DIT_NVFP4=1", text):
            errs.append(
                "fewstep arm log has H3_DIT_NVFP4=1 "
                "(cross-arm forbidden unless H3_FEWSTEP_ALLOW_NVFP4)"
            )

        return errs, warnings

    # expect is None: soft recognition when lines present
    if any_few:
        if m_bound and m_preset:
            if m_bound.group(1) != m_preset.group(1):
                errs.append(
                    f"fewstep_bound contract={m_bound.group(1)!r} != "
                    f"fewstep_preset id={m_preset.group(1)!r}"
                )
        if m_bound and m_sampler:
            if m_bound.group(1) != m_sampler.group(3):
                warnings.append(
                    f"fewstep_bound contract={m_bound.group(1)!r} vs "
                    f"sampler fewstep={m_sampler.group(3)!r}"
                )
        if m_bound:
            speed = m_bound.group(7)
            if speed in ("turbo4", "taomate", "fast4"):
                errs.append(
                    f"fewstep_bound speed_preset={speed!r} "
                    "(fewstep forbids sigma-rewriting presets)"
                )
        if m_preset and not m_bound:
            warnings.append("fewstep_preset without fewstep_bound")
        if m_bound and not m_contract:
            warnings.append("fewstep_bound without fewstep_contract print")
    return errs, warnings


def check_chain_off(text: str) -> list[str]:
    errs: list[str] = []
    if re.search(r"H3_DIT_NVFP4_CHAIN\s*=\s*1", text):
        errs.append("NVFP4 chain enabled (H3_DIT_NVFP4_CHAIN=1)")
    if re.search(r"dit_nvfp4:.*CHAIN=1", text):
        errs.append("dit_nvfp4 CHAIN=1 line present (P0 forbids chain)")
    if re.search(r"FC1->FC2 chained", text):
        errs.append("NVFP4 chain path active (FC1->FC2 chained)")
    if re.search(r"dit_nvfp4:\s*chain scratch", text):
        errs.append("dit_nvfp4 chain scratch line (P0 forbids chain)")
    return errs


def check(
    text: str,
    arm: str,
    expect_fewstep: str | None,
    *,
    expect_seed: str | None = None,
    expect_skip_audio: bool = False,
    expect_live_te: bool = False,
    expect_te_load: bool = False,
    expect_env_contract: bool = False,
    expect_te_cache_hit: bool = False,
) -> list[str]:
    errs: list[str] = []
    warnings: list[str] = []

    # Graph must be off
    m = re.search(r"sdpa_graph:\s*wanted=(\d+)", text)
    if not m:
        errs.append("missing: sdpa_graph: wanted=N")
    elif m.group(1) != "0":
        errs.append(f"sdpa_graph wanted={m.group(1)} (need 0 for P0)")

    errs.extend(check_chain_off(text))

    # Sigma schedule (always for generate completion)
    if not re.search(r"sigma\[0\.\.", text):
        errs.append("missing: sigma[0..N] schedule print")

    # shape: banner — sdpa + seed (P0 + fewstep)
    m_shape = re.search(
        r"^shape:.*\bsdpa=(\S+).*?\bseed=(0x[0-9a-fA-F]+)",
        text,
        re.M,
    )
    if not m_shape:
        # tolerate seed before sdpa ordering variants
        m_shape = re.search(
            r"^shape:.*\bseed=(0x[0-9a-fA-F]+).*?\bsdpa=(\S+)",
            text,
            re.M,
        )
        if m_shape:
            seed_got, sdpa_got = m_shape.group(1), m_shape.group(2)
        else:
            seed_got, sdpa_got = None, None
            if arm in ("int8", "nvfp4", "fewstep"):
                errs.append("missing: shape: ... sdpa=... seed=0x... banner")
    else:
        sdpa_got, seed_got = m_shape.group(1), m_shape.group(2)

    if expect_seed and seed_got:
        if _norm_seed(seed_got) != _norm_seed(expect_seed):
            errs.append(
                f"seed={seed_got} (want {expect_seed} via --expect-seed)"
            )
    elif expect_seed and not seed_got:
        errs.append(f"missing seed for --expect-seed={expect_seed}")

    # skip-audio / dual sigma
    m_phase_audio = re.search(
        r"^phase_ms:.*?audio=([0-9.eE+-]+)", text, re.M
    )
    has_dual = bool(re.search(r"adaln_dual_sigma:.*video_shift=", text))
    if expect_skip_audio:
        if m_phase_audio:
            av = _parse_float(m_phase_audio.group(1))
            if av is not None and av > 1.0:
                errs.append(
                    f"phase_ms audio={av} (want ~0 for --expect-skip-audio)"
                )
        else:
            warnings.append(
                "no phase_ms audio= field (--expect-skip-audio; run may "
                "have aborted before summary)"
            )
        # skip-audio → pack_audio=0 → no adaln_dual_sigma; do not require it
        if has_dual:
            warnings.append(
                "adaln_dual_sigma present with --expect-skip-audio "
                "(unexpected audio pack)"
            )
    else:
        # Legacy: require dual-sigma when audio likely packed; else warn only
        if not has_dual:
            if arm == "fewstep":
                warnings.append(
                    "missing adaln_dual_sigma (ok if --skip-audio; "
                    "fewstep gold usually needs audio shift print)"
                )
            elif arm in ("int8", "nvfp4"):
                warnings.append(
                    "missing adaln_dual_sigma "
                    "(P0 visual gate uses --skip-audio; "
                    "pass --expect-skip-audio to soft-skip this)"
                )

    # TE path gates
    has_override = bool(
        re.search(r"^text_h:\s*override loaded from\s+", text, re.M)
    )
    has_text_ck = bool(re.search(r"text_h checksum=0x[0-9a-fA-F]+", text))
    if expect_live_te and has_override:
        errs.append(
            "text_h: override loaded present but --expect-live-te "
            "(unset H3_TEXT_H_LOAD for visual gate)"
        )
    if expect_te_load and not has_override:
        errs.append(
            "missing text_h: override loaded (--expect-te-load for "
            "velocity/md5 diagnostics with H3_TEXT_H_LOAD)"
        )
    if arm in ("int8", "nvfp4") and not has_text_ck and not has_override:
        warnings.append(
            "no text_h checksum= line (confirm TE / refine ran)"
        )

    if arm == "int8":
        if re.search(r"dit_nvfp4:\s*H3_DIT_NVFP4=1", text):
            errs.append("INT8 arm log contains dit_nvfp4: H3_DIT_NVFP4=1")
        if not re.search(r"dit_format:\s*INT8", text):
            warnings.append("no dit_format: INT8 line (confirm arm manually)")
    elif arm == "nvfp4":
        if not re.search(r"dit_nvfp4:\s*H3_DIT_NVFP4=1", text):
            errs.append("NVFP4 arm missing dit_nvfp4: H3_DIT_NVFP4=1")
        if not re.search(r"dit_nvfp4:\s*sidecar=", text):
            warnings.append(
                "no dit_nvfp4: sidecar= line (confirm H3_DIT_NVFP4_FILE)"
            )
        if re.search(r"dit_format:\s*INT8", text) and not re.search(
            r"dit_format:\s*NVFP4", text
        ):
            warnings.append(
                "dit_format: INT8 without dit_format: NVFP4 "
                "(confirm NVFP4 arm banner)"
            )
    elif arm == "fewstep":
        if expect_fewstep is None or expect_fewstep == "none":
            errs.append(
                "--arm fewstep requires --expect-fewstep pdmd4|pdmd2|dmad4"
            )
    else:
        errs.append(f"unknown --arm {arm!r} (want int8|nvfp4|fewstep)")

    # P0 arms: warn if T3 fewstep lines leak in (arm isolation)
    if arm in ("int8", "nvfp4") and expect_fewstep is None:
        if re.search(r"^fewstep_bound:", text, re.M) or re.search(
            r"^fewstep_preset:", text, re.M
        ):
            warnings.append(
                "fewstep_preset/bound present on P0 arm log "
                "(量化臂与少步臂应隔离；用 --expect-fewstep 显式验收 T3)"
            )

    fe, fw = check_fewstep(text, expect_fewstep)
    errs.extend(fe)
    warnings.extend(fw)

    ee, ew = check_env_contract(text, require=expect_env_contract)
    errs.extend(ee)
    warnings.extend(ew)
    te, tw = check_te_prompt_cache(
        text, require_hit=expect_te_cache_hit
    )
    errs.extend(te)
    warnings.extend(tw)

    mrc = re.search(r"^RUN_RC=(\d+)\s*$", text, re.M)
    if mrc and mrc.group(1) != "0":
        errs.append(f"RUN_RC={mrc.group(1)}")

    if not re.search(r"phase_ms:", text):
        warnings.append("missing phase_ms (run may have aborted before summary)")

    for w in warnings:
        print(f"WARN: {w}", file=sys.stderr)
    return errs


def _self_test() -> int:
    """Host-only fixture checks (no GPU)."""
    base = """sdpa_graph: wanted=0
shape: 864x480 5s → latent 1x30x54  dit_layers=40 te_layers=0 vae_blocks=0 vae_repeats=0 steps=8 reuse=0 magcache=0 step_cache=off/0 token_red=0 sdpa=cudnn attn_window=0 dist_attn=off overlap=0 cfg=1 seed=0x76800002 context_budget=0 (default 0; hard_max 0) preset=none ulysses=1 vae_decode=on
sigma[0..4]=1.000 ...
dit_format: INT8 ConvRot (theme-follow path)
text_condition: refiner=1 tokens=78 text_mod_row=1(default_TEXT) (refiner QKV follows H3_DIT_QKV_GROUPED; default contiguous) text_h checksum=0xbca4036316f71294 n=419328
phase_ms: te=1.0 dit=1.0 lift_switch=0.0 vae=1.0 vae_load=0.0 vae_blocks=0.0 audio=0.0 ppm=0.0 mux=0.0 total=1.0 dit_evals=8 dit_skips=0 hot=0 clip=1/1
RUN_RC=0
"""
    # P0 INT8 clean + skip-audio + live TE + seed
    e = check(
        base,
        "int8",
        None,
        expect_seed="0x76800002",
        expect_skip_audio=True,
        expect_live_te=True,
    )
    assert not e, e

    # Wrong seed
    e = check(base, "int8", None, expect_seed="0x76800001")
    assert any("seed=" in x for x in e), e

    # text_h override vs live-te
    override = base + "\ntext_h: override loaded from /tmp/c_te_refined.f32 (419328 floats)\n"
    e = check(override, "int8", None, expect_live_te=True)
    assert any("override" in x for x in e), e
    e = check(override, "int8", None, expect_te_load=True)
    assert not e, e

    # chain hard fails
    chain = base + "\ndit_nvfp4: H3_DIT_NVFP4_CHAIN=1 — FC1->FC2 chained\n"
    e = check(chain, "int8", None)
    assert any("chain" in x.lower() or "CHAIN" in x for x in e), e

    # skip-audio: audio wall too high
    loud = base.replace("audio=0.0", "audio=12000.0")
    e = check(loud, "int8", None, expect_skip_audio=True)
    assert any("audio=" in x for x in e), e

    # fewstep pdmd4 happy path
    pdmd = base + """
fewstep_preset: id=pdmd4 steps=4 cfg=1 video_shift=12 audio_shift=3 sampler=velocity_euler contract=pdmd4
fewstep_contract: id=pdmd4 steps=4 sampler=velocity_euler video_shift=12 audio_shift=3 cfg=1 preset=none nvfp4=0 lora=(none)
sampler: selected=velocity_euler (explicit=0 fewstep=pdmd4)
fewstep_bound: contract=pdmd4 steps=4 cfg=1 video_shift=12 audio_shift=3 sampler=velocity_euler speed_preset=none
fewstep_contract: sigma_ok student_grid(nfe=4, video_shift=12, audio_shift=3)
adaln_dual_sigma: video_shift=12 audio_shift=3
"""
    e = check(pdmd, "fewstep", "pdmd4")
    assert not e, e

    # dmad4 wants lcm_renoise
    dmad_bad = base + """
fewstep_preset: id=dmad4 steps=4 cfg=1 video_shift=12 audio_shift=2 sampler=velocity_euler contract=dmad4
fewstep_contract: id=dmad4 steps=4 sampler=velocity_euler video_shift=12 audio_shift=2 cfg=1 preset=none nvfp4=0 lora=x
sampler: selected=velocity_euler (explicit=0 fewstep=dmad4)
fewstep_bound: contract=dmad4 steps=4 cfg=1 video_shift=12 audio_shift=2 sampler=velocity_euler speed_preset=none
"""
    e = check(dmad_bad, "fewstep", "dmad4")
    assert any("sampler" in x for x in e), e

    dmad_ok = base + """
fewstep_preset: id=dmad4 steps=4 cfg=1 video_shift=12 audio_shift=2 sampler=lcm_renoise contract=dmad4
fewstep_contract: id=dmad4 steps=4 sampler=lcm+re-noise video_shift=12 audio_shift=2 cfg=1 preset=none nvfp4=0 lora=x
sampler: selected=lcm_renoise (explicit=0 fewstep=dmad4)
fewstep_bound: contract=dmad4 steps=4 cfg=1 video_shift=12 audio_shift=2 sampler=lcm_renoise speed_preset=none
fewstep_contract: sigma_ok student_grid(nfe=4, video_shift=12, audio_shift=2)
"""
    e = check(dmad_ok, "fewstep", "dmad4")
    assert not e, e

    turbo = pdmd.replace("speed_preset=none", "speed_preset=turbo4")
    e = check(turbo, "fewstep", "pdmd4")
    assert any("turbo4" in x for x in e), e

    e = check(pdmd, "int8", "none")
    assert any("expect-fewstep=none" in x for x in e), e

    # nvfp4 arm needs dit_nvfp4 lines
    nv = base.replace("dit_format: INT8 ConvRot (theme-follow path)",
                      "dit_format: NVFP4 (非金线；dequant→INT8 GEMM)")
    nv += "\ndit_nvfp4: H3_DIT_NVFP4=1 — FP4 cuBLASLt GEMM track\n"
    nv += "dit_nvfp4: sidecar=/models/dit_nvfp4.safetensors tensors=1332\n"
    e = check(nv, "nvfp4", None, expect_skip_audio=True, expect_live_te=True)
    assert not e, e

    # C061 inline env contract recognition
    env_blk = base + """
# H3_ENV_CONTRACT  arm=int8  expect_p0=0
env H3_DIT_NVFP4=<unset>  # DiT NVFP4 experimental arm (1=on) rule=arm
cli seed=0x76800002 skip_audio=1 fewstep_preset=none fewstep_contract=none sampler=velocity_euler sdpa=cudnn sdpa_graph=0 te_path=live
"""
    e = check(env_blk, "int8", None, expect_env_contract=True)
    assert not e, e
    e = check(base, "int8", None, expect_env_contract=True)
    assert any("H3_ENV_CONTRACT" in x for x in e), e
    # soft: without --expect-env-contract, missing block is OK
    e = check(base, "int8", None)
    assert not any("H3_ENV_CONTRACT" in x for x in e), e

    # C062 te_prompt_cache recognition
    te_hit = (
        base
        + "\nte_prompt_cache: dir=/tmp/te require_hit=0 "
        + "(opt-in; gold default off)\n"
        + "te_prompt_cache: HIT dir=/tmp/te key=0123456789abcdef "
        + "tokens=16 layers=4 dim=5120\n"
    )
    e = check(te_hit, "int8", None, expect_te_cache_hit=True)
    assert not e, e
    te_miss = (
        base
        + "\nte_prompt_cache: MISS wrote dir=/tmp/te key=0123456789abcdef "
        + "tokens=16 layers=4 dim=5120\n"
    )
    e = check(te_miss, "int8", None, expect_te_cache_hit=True)
    assert e and any("HIT required" in x for x in e), e
    e = check(te_miss, "int8", None)  # soft: warning only
    assert not e, e

    print("ok p0_protocol_check_log self-test")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, nargs="?", help="path to generate.log")
    ap.add_argument(
        "--arm",
        choices=("int8", "nvfp4", "fewstep"),
        help="expected DiT / acceptance arm",
    )
    ap.add_argument(
        "--expect-fewstep",
        choices=("none", "pdmd4", "pdmd2", "dmad4"),
        default=None,
        help="require/forbid fewstep_preset+bound+sampler lines "
        "(T3 offline check; default: soft-recognize if present)",
    )
    ap.add_argument(
        "--expect-seed",
        default=None,
        help="require shape: seed=0x... match (e.g. 0x76800002)",
    )
    ap.add_argument(
        "--expect-skip-audio",
        action="store_true",
        help="P0 visual gate: require phase_ms audio≈0; do not require "
        "adaln_dual_sigma",
    )
    ap.add_argument(
        "--expect-live-te",
        action="store_true",
        help="FAIL if text_h: override loaded (visual gate default)",
    )
    ap.add_argument(
        "--expect-te-load",
        action="store_true",
        help="FAIL if H3_TEXT_H_LOAD override line missing (md5 diagnostics)",
    )
    ap.add_argument(
        "--expect-env-contract",
        action="store_true",
        help="require # H3_ENV_CONTRACT marker (C061)",
    )
    ap.add_argument(
        "--expect-te-cache-hit",
        action="store_true",
        help="require te_prompt_cache HIT line (C062 warm KPI)",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="run host fixtures and exit (no log file)",
    )
    args = ap.parse_args()
    if args.expect_live_te and args.expect_te_load:
        ap.error("--expect-live-te and --expect-te-load are mutually exclusive")
    if args.self_test:
        return _self_test()
    if args.log is None or args.arm is None:
        ap.error("log and --arm are required unless --self-test")
    if not args.log.is_file():
        print(f"usage error: not a file: {args.log}", file=sys.stderr)
        return 2
    errs = check(
        _read(args.log),
        args.arm,
        args.expect_fewstep,
        expect_seed=args.expect_seed,
        expect_skip_audio=args.expect_skip_audio,
        expect_live_te=args.expect_live_te,
        expect_te_load=args.expect_te_load,
        expect_env_contract=args.expect_env_contract,
        expect_te_cache_hit=args.expect_te_cache_hit,
    )
    if errs:
        for e in errs:
            print(f"FAIL: {e}", file=sys.stderr)
        print("PROTO_FAIL", file=sys.stderr)
        return 1
    print("PROTO_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
