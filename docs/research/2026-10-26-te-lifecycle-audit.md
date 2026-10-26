# TE / DiT / VAE 生命周期审计（DeepInfer 取舍 B）

> **日期：** 2026-10-26。  
> **性质：** P2 可选工程笔记——只审计峰值与竞态检查点，**不改热路径分支**。  
> **对应：** [`2026-10-26-deepinfer-to-h3c-takeaways.md`](2026-10-26-deepinfer-to-h3c-takeaways.md) §2.2 B（异步重叠）；[`2026-10-26-followup-dev-plan.md`](2026-10-26-followup-dev-plan.md) checklist #7。  
> **主入口：** `tools/h3_cuda_generate.c`（CUDA CLI）。

---

## 1. 结论（一页能带走的）

| 阶段 | 谁占显存 / 主机 | 该何时释放或等待 | 已知坑 |
| --- | --- | --- | --- |
| TE open + encode | TE packed H2D + layer 激活；可选 disk→host DiT stage | encode / refine 后立刻 `free(text_emb)` + `h3_nvfp4_store_free` | 与 DiT GPU prefetch 重叠会抢 VRAM（默认 `latent_t≥32` 跳过 overlap） |
| DiT host_stage | 仅 disk→pinned host（无 GPU） | INT8：等前 N 块 ready 即可开 DiT；**NVFP4：必须 join 后再进 DiT** | NVFP4 dequant 与 upload 重叠曾 UAF/SIGSEGV |
| DiT step0 upload | INT8 `resident_all` ≈18 GiB（50×~386 MiB/block）+ LoRA A/B | VAE device-W 不能抢在 upload 前占位 | PDMD 2NFE：预热踢在 step0 → OOM（已修 gate） |
| VAE prewarm | mmap + hostreg；可选 ~1.8 GiB device W | 默认 last-NFE；2-step 会踢在首步 | `vae_prewarm_want_dev_for_dit`：resident=0 时强制 host-only |

**不要做：** 引入 DeepInfer 调度器；在热路径加多 shape plan；把 VAE device prefill 当默认无门控。

---

## 2. 时间线（代码锚点）

均在 `tools/h3_cuda_generate.c`，辅以 `h3_vae_prewarm.c` / `h3_te_cache.*`。

```
te_open / te_encode_begin
  ├─（可选）H3_DIT_HOST_STAGE_TE：packed hook → dit_host_stage_th
  │         disk→host，printf "dit_host_stage: kick …"
  ├─（可选）H3_DIT_TE_OVERLAP_BLOCKS：GPU prefetch（latent_t<32）
  ├─ te_prompt_cache HIT → 立即 kick host_stage（hook 不会打）
  └─ live encode: h3_nvfp4_encode_layers(_multimodal)_f32
        或 H3_TEXT_H_LOAD 诊断覆盖 refined text_h

refine: h3_dit_w4a8_refine_text_f32(dit, text_emb → text_h)
  │
  ├─ free(text_emb); h3_nvfp4_store_free(&te_store)
  │     printf "released TE after encode (before DiT; …)"
  ├─ NVFP4: pthread_join(host_stage)   // 禁止 overlap
  └─ INT8: wait_first want=H3_DIT_HOST_STAGE_WAIT_BLOCKS(默认2)
           其余 disk→host 与 step1 重叠

DiT steps…
  └─ kick_vae_prewarm（默认 H3_GEN_VAE_PREWARM=on）
        h3_vae_prewarm_th: open + madvise + hostreg + optional device W
        want_dev = vae_prewarm_want_dev_for_dit(...)
```

### 2.1 TE 释放

- `free(text_emb)` 后 `text_h` 留在 host，供 DiT conditioning。
- `h3_nvfp4_store_free(&te_store)` + `h3_nvfp4_cuda_disable()`：给后续 weight window 腾 VRAM（注释写明 24GB 场景）。
- 磁盘 prompt cache：`h3_te_cache_{key,load,save}`（`h3_te_cache.c`）；HIT 不跑 packed hook，需手动 kick host_stage。
- 生产债：`c_te_raw` 偶发不确定 → 速度场对照优先 `H3_TEXT_H_LOAD` / refine 路径（见 P0 纪律）。

### 2.2 DiT 上传与 host_stage

| 旋钮 | 默认 | 作用 |
| --- | --- | --- |
| `H3_DIT_HOST_STAGE_TE` | on（`=0` 关） | TE 期间 disk→host stage |
| `H3_DIT_HOST_STAGE_WAIT_BLOCKS` | 2 | INT8 开 DiT 前最少 ready 块数 |
| `H3_DIT_TE_OVERLAP_BLOCKS` | 全窗口上限 / 受 cache window 裁 | TE 期间 GPU prefetch；`latent_t≥32` 默认跳过 |

金线 INT8 允许 host_stage 与 step1 重叠；NVFP4 host dequant 会 realloc 同槽，**必须 join**（日志：`dit_host_stage: NVFP4 joined before DiT`）。

### 2.3 VAE 预热

| 旋钮 | 默认 | 作用 |
| --- | --- | --- |
| `H3_GEN_VAE_PREWARM` | on（`=0` 关） | 是否起 prewarm 线程 |
| `H3_VAE_PREWARM_DEV` | auto | `0` host-only；`1` 强制 device W；auto：空闲 ≥ ~6 GiB 才 prefill |

门控函数：

1. `vae_prewarm_want_dev` — 6 GiB free 门。  
2. `vae_prewarm_want_dev_for_dit` — **INT8 `resident=0` 且 window>0 → 强制 host-only**（除非 `H3_VAE_PREWARM_DEV=1`）。  
   线程体：`h3_vae_prewarm_th`（`h3_vae_prewarm.c`）。

---

## 3. PDMD / 少步 OOM 检查清单（跑前 / 跑后）

来自 [`2026-10-09-pdmd-dmad-loras.md`](2026-10-09-pdmd-dmad-loras.md) §7–8 的复盘，固化成可勾项：

**跑前**

- [ ] `nvidia-smi` 无残留大进程；合同 `pdmd2|pdmd4|dmad4` 与 `--fewstep-preset` 一致。  
- [ ] 未叠 `H3_DIT_NVFP4`（除非 `H3_FEWSTEP_ALLOW_NVFP4=1`）。  
- [ ] 未叠 `--preset turbo4`（会改 sigma）。  
- [ ] 2NFE：确认日志会出现  
  `vae_prewarm_dev: INT8 resident=0 window=… → host-only prewarm`  
  （若强制 `H3_VAE_PREWARM_DEV=1`，视为自担 OOM 风险）。

**跑中关键日志（顺序）**

1. `te_encode:` / `te_prompt_cache:`  
2. `released TE after encode (before DiT; …)`  
3. `dit_host_stage:`（kick / wait_first / NVFP4 joined）  
4. `dit_int8_cache_refit:` / `dit_int8_cache_freeze:`（`free_giB`、`resident`）  
5. `vae_th: created at=dit_step… want_dev=`  
6. `vae_prewarm_dev:`（host-only vs device W）  
7. 若 OOM：`h3_cuda_w4a8_weight_create_from_i8` / `cudaMalloc w_i8`（`h3_gpu_cuda.cu`）

**归因口诀**

- 已见 `released TE after encode`，但 step0 `cudaMalloc w_i8` 失败 → **多半是 VAE device-W 与 resident_all 竞态**，不是 TE 没释放。  
- `after_te` free 比 video-only 低数 GiB → 常见是 **LoRA fp32 A/B + audio pack**，不是 TE 残留。  
- 8-step 在 freeze 后踢预热，6 GiB 门通常已 host-only；**2-step 才是竞态高发**。

---

## 4. 与 DeepInfer B 的边界

可借鉴：生命周期意识、峰值记账、CPU–GPU 错峰（host_stage / prewarm）。  
不借鉴：DeepInfer 调度器代码、continuous batching、多 shape JIT。  
本仓已有重叠（TE∥host_stage、tail NFE∥VAE）；审计目标是 **别回退已知门控**，不是再加一层框架。

---

## 5. 变更记录

| 日期 | 说明 |
| --- | --- |
| 2026-10-26 | 初稿：锚到 `h3_cuda_generate.c` / `h3_vae_prewarm.*` / `h3_te_cache.*`；收录 PDMD 2NFE 预热 OOM 检查清单。 |
| 2026-10-27 | C064：白名单落成 `h3_lifecycle_whitelist.h`；说明 [`2026-10-27-lifecycle-deferred-release.md`](2026-10-27-lifecycle-deferred-release.md)；`H3_LIFECYCLE_TRACE` 仅调试打印，**不改重叠**。 |

---

## DeepInfer 旁证（2026-10-26 deeper-pass）

来源：[`2026-10-26-deepinfer-deeper-pass.md`](2026-10-26-deepinfer-deeper-pass.md) §1.3 / §1.12。

**DeferredRelease 白名单：** 代码表见 `h3_lifecycle_whitelist.h`（C064）；说明 [`2026-10-27-lifecycle-deferred-release.md`](2026-10-27-lifecycle-deferred-release.md)。允许「多活到下一阶段 event 之后再 free」的对象须显式列出，避免过早 free（UAF）或过晚 free（顶 resident）。审计建议（与代码表对齐）：

| 对象 | 释放时机 | 可否延后到 DiT step0 launch 后 |
|---|---|---|
| TE `text_emb` scratch | encode 返回后立刻 | 否（立刻） |
| TE store（权重页） | refine 完成后 `store_free` | 否（立刻，除非实测需延后） |
| DiT host-stage pin 缓冲 | wait_blocks / TE join 后 | 可与 DiT 首 kernel 重叠，但须有 event |
| VAE device-W | DiT 结束后 / 非 2NFE 抢占路径 | 见既有 NVFP4/2NFE 门控 |

**允许的 host sync 点（白名单）：** TE join、VAE wait、显式 nsys 边界、进程退出。  
**禁止（热路径）：** DiT 层间遗留的 `cudaDeviceSynchronize` / 阻塞 `cudaMemcpy` 当默认路径。代码级 `H3_FORBID_HOT_SYNC` 守卫 = P3，不挡主线。

