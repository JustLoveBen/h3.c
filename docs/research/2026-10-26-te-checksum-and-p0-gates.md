# TE checksum helper + P0 log hard gates（2026-10-26）

离线开发项（无 GPU）：把 P0 对照纪律落到可执行的 host 工具，并明确 fewstep-preset 与错误 audio shift 的拒绝文案。

## 1. `tools/te_f32_checksum.py`

对 `H3_DUMP_DIR` 写出的 `c_te_raw.f32` / `c_te_refined.f32` 算与 `h3_dit_w4a8_checksum_f32` 同构的 FNV-1a64，便于：

- 两次 live TE 是否漂了（`c_te_raw` 偶发不确定债）；
- 确认 `H3_TEXT_H_LOAD` 注入的 refine dump 身份。

```bash
python3 tools/te_f32_checksum.py --guide
python3 tools/te_f32_checksum.py out/.../c_te_refined.f32
python3 tools/te_f32_checksum.py a.f32 b.f32   # MATCH / DIFF
```

| 场景 | TE 路径 |
| --- | --- |
| P0 视觉门 | **live TE**；`unset H3_TEXT_H_LOAD`；日志检加 `--expect-live-te` |
| 1-step velocity md5 / DiT 侧对照 | dump 一次 refined → `H3_TEXT_H_LOAD=.../c_te_refined.f32`；日志检 `--expect-te-load` |
| waffle 金线 | **禁止** live 与 load 混成一条 |

## 2. `tools/p0_protocol_check_log.py` 新增硬门

| 旋钮 | 作用 |
| --- | --- |
| `--expect-seed 0x76800002` | `shape:` 行 seed 必须匹配 |
| `--expect-skip-audio` | `phase_ms audio≈0`；**不**要求 `adaln_dual_sigma`（skip-audio 时本就不打印） |
| `--expect-live-te` | 有 `text_h: override loaded` → FAIL |
| `--expect-te-load` | 缺 override 行 → FAIL（诊断臂） |
| chain 加强 | `CHAIN=1` / `FC1->FC2 chained` / `chain scratch` 任一即 FAIL |

视觉门推荐：

```bash
python3 tools/p0_protocol_check_log.py --arm int8 \
  --expect-seed 0x76800002 --expect-skip-audio --expect-live-te \
  "$OUT/generate.log"
```

## 3. fewstep-preset × 错误 `H3_AUDIO_SIGMA_SHIFT`

`--fewstep-preset` 仍用 `setenv(..., 0)`（不覆盖已有 env）。若 env 已设且与 preset 目标不符（pdmd→3，dmad→2），**提前 FAIL**，文案点明冲突与应设值（`h3_fewstep_preset_audio_env_compatible`）。避免「preset 打印 audio_shift=2、实际 schedule 仍是 3、再在合同层才炸」的含糊路径。

## 4. `tools/timing_append_row.py`

从单份 `generate.log` 抽出可粘贴进 `2026-10-09-generation-timing.md` 的一行；**不**自动改账本、**不**算加速比。仅在 PROTO_OK 之后使用。
