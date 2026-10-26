# C073 — dump-once / score-many conventions

Date: 2026-10-27 (Asia/Shanghai). Offline conventions for gen/score split.

## Rule (one line)

**Generate (or dump) once under a frozen config; score / waffle / A-B many times on that artifact. Do not retune gold while scoring.**

## Env

| Env | Role |
|---|---|
| `H3_DUMP_LATENT=PATH` | After DiT final latent (clip1), write `H3LAT1` file and **exit before VAE** |
| `H3_REPLAY_LATENT=PATH` | Skip TE+DiT; load dump into VAE-only path (bit-fixed A/B) |
| `H3_DUMP_DIR=DIR` | Oracle dumps (TE/velocity etc.); not a gold retune lever |
| `H3_DUMP_ONLY=1` | Stop after step0 forward dump (diagnostics) |

Defaults: all off. Listed in `# H3_ENV_CONTRACT` as `info` (not P0-forbidden).

## File format (`H3_DUMP_LATENT`)

- Header (64 bytes ASCII): `H3LAT1 <t> <h> <w> <n>\n` padded in fixed 64-byte record
- Payload: `n` float32, layout packed DiT latent `[T, H/2, W/2, 96]` (same as `latent.raw` semantics)
- `n` must equal `t * (h/2) * (w/2) * 96`

Helper: `python3 tools/dump_once_score_many_check.py PATH` (header + n check; optional `--log`).

## Suggested directories

```
OUT/
  META.txt          # seed, arm, sdpa_graph, fewstep, binary, prompt id
  generate.log
  latent_dump.h3lat # from H3_DUMP_LATENT (name arbitrary)
  score/
    waffle_*.txt
    replay_async_off.log
    replay_async_on.log
```

Copy env contract into META (`print_h3_env_contract.py --markdown`).

## Log markers (grep)

Dump path:

```
dump_latent: clip=1 wrote N f32 (T×H×W) checksum=0x... → PATH
# H3_DUMP_ONCE path=PATH ... shape=TxHxW layout=THW96
dump_once_score_many: mode=dump path=PATH do_not_retune_gold=1
dump_latent: done; exiting before clip 1 VAE
```

Replay path:

```
replay_latent: PATH → T×H×W (...)
dump_once_score_many: mode=replay_geometry path=PATH ...
replay_latent: loaded N f32 checksum=0x...
# H3_SCORE_MANY mode=replay path=PATH checksum=0x... n=N
dump_once_score_many: mode=replay path=PATH do_not_retune_gold=1
```

## Do not

- Change gold seed / sdpa_graph / fewstep / TE path while iterating score scripts.
- Treat `H3_DUMP_DIR` oracle files as a new gold source without META.
- Open SDPA CUDA graph “just for this score run” without META title (P0 runbook).
- Commit NVFP4 chain WIP into a score folder and call it gold.

## Related

- P0 runbook: [`2026-10-26-p0-visual-gate-runbook.md`](2026-10-26-p0-visual-gate-runbook.md)
- Waffle same-config: [`2026-10-26-waffle-same-config-gate.md`](2026-10-26-waffle-same-config-gate.md)
- Catalog **C073**
