# h3.c 5090 极速 SKU Implementation Plan

> using writing-plans
>
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> 本轮文档已锁定文件图与任务切分。**禁止改产品默认。禁止 commit（除非执行会话被明确要求）。禁止把密码写入 repo。禁止把未实现的 LTX Super Accel 做成实现任务。**

**Goal:** 在不改金线默认的前提下，落地独立极速档 `--preset turbo4`（4-step 匹配 LoRA + Sage3），并在 8×5090 上按 TP=1 实现可叠加的 Ulysses（默认 U=1）。

**Architecture:** 两套 CLI 配方互不污染。金线仍是 INT8 8-step + `sdpa=cudnn`、cache/sage/sol/ulysses 关。`turbo4` 只组合已有开关并强制 LoRA；LoRA 是新模块，不塞进 `h3_cuda_generate.c`。Ulysses 是新 `h3_ulysses*` + NCCL A2A，插在现有 `block_device_compute` 的 QKV/SDPA 两侧。TAE 成片另文件、仅 turbo opt-in。TE 去 vision / 常驻已在 T2VA 路径落地，本计划标 KEEP。

**Tech Stack:** 本仓 C/CUDA `h3_cuda_generate`；safetensors LoRA（PEFT / Kohya）；`h3_sdpa` registry 的 `sage3`；NCCL（单进程 `ncclCommInitAll`）；8×5090 `build-cuda-sm120` + CUDA 13.2 / sm_120。对照仓（FastVideo / larryvrh）只作权来源，不进金线。

**Spec:**

| 文档 | 本文怎么用 |
| --- | --- |
| [2026-09-11-h3c-5090-turbo-sku.md](./2026-09-11-h3c-5090-turbo-sku.md) | 极速配方、质量底线、单卡/8 卡组合、落地顺序 T0–T4 |
| [2026-09-11-h3c-next-opt-plan.md](./2026-09-11-h3c-next-opt-plan.md) | 金线 Ulysses 合同（校验/pad/A2A/MAE 门）；P3 `fast4` **并入** 本文 `turbo4`，不另开第二套 preset |
| [2026-09-11-h3c-accel-principle-matrix.md](./2026-09-11-h3c-accel-principle-matrix.md) | 杠杆：turbo = 1（LoRA）+ 2（Sage3）[+ 4 Ulysses][+ 5 TAE]；金线只加 4 |
| [2026-09-11-h3c-pcie-multigpu-lessons.md](./2026-09-11-h3c-pcie-multigpu-lessons.md) | 无 NVLink：先 TP=1；U∈{2,4,8}；不要 `NCCL_P2P_DISABLE` |
| [docs/nvidia_build.md](../../nvidia_build.md) | 8×5090 探测、sm_120、Sage3 插件环境 |

## Global Constraints

- 金线默认：**INT8 ConvRot + `sdpa=cudnn` + 8-step（产品命令显式 `--steps 8 --cfg 1`）+ cache/sage/sol/ulysses 关**。CLI 源码缺省仍是 `sdpa="cudnn"`、step-cache OFF、无 `--preset`。**turbo 不得改这些默认。**
- 真 `--seconds`；禁止用 `--frames` 冒充产品时长。
- 密码不入库。8×5090：`H3_SSH_5090x8=user@89.185.31.140`；密码仅会话变量 `H3_SSH_PASS_5090x8`。旧单卡 5090：`H3_SSH_5090`。
- 极速质量底线：主体可辨、禁花屏、有声；口型可偏。MAE 记录、**不设金线阈值**。未蒸馏 `--steps 4`（无匹配 LoRA）禁止当产品。
- 同测 warm **n≥2，推荐 3**（冷 1 + warm 3，报表用 warm）。
- 无 NVLink。先 **TP=1**；合法 Ulysses ∈ {1,2,4,8}；**U=7 非法**。
- 未实现的 LTX Super Accel / TAEHV / `cute_sm100` / 6.85s **不进本计划实现任务**。
- 公开倍数（12.2× / 14× / 22.2×）禁止写成相对本仓 66s。推断秒数必须标 **推断**。
- 根目录 `/scripts` 被 gitignore；新 harness 放 `tools/`。rsync **禁止 `--delete`**。
- next-opt 的 `--preset fast4` **不要并行实现**；需要时在 help 写「已更名为 turbo4」。

---

## 摸底结论（执行前锁定）

仓库现状（2026-09-11）：

| 轴 | 状态 |
| --- | --- |
| `tools/h3_cuda_generate.c` | 无 `--preset` / `--lora` / `--ulysses`。`sdpa` 缺省 `"cudnn"`；`steps` 缺省 **2**（冒烟，产品金线命令显式 8）；`seconds` 缺省 5。Sage3 / Tea / Mag / FB 已是 opt-in。 |
| `h3_sdpa.h` + `h3_sdpa_registry.c` | `auto`→cudnn；sage3 缺插件硬失败。KEEP。 |
| `h3_step_cache*` | 默认 OFF。turbo 主路径必须保持关。KEEP。 |
| `h3_dit_w4a8*` | 单卡 `block_device_compute`：INT8 GEMM → `h3_cuda_qkv_rope_sdpa_f16_dev(..., heads=56, ...)`。无 SP。线性后缀 `attn.qkv_proj` / `attn.out_proj` / `mlp.fc1` / `mlp.fc2`。 |
| 权重 / LoRA | `h3_safetensors` + `h3_w4a8` 只加载骨干。**无产品 LoRA 加载器。** |
| TE | T2VA（无 `--ref-image` / keyframe）**不跑 vision**。`H3_TE_MMAP_POPULATE` / `H3_TE_PREFAULT` / packed `cudaHostRegister` 已在 `h3_nvfp4*`。 |
| TAE | **无**成片路径。金线是 fp16 VAE tile+deblock。 |
| 多卡 | **无** `ulysses` / NCCL。 |
| CMake | `h3_cuda_generate` 不链 nccl。nvcc 候选有 13.1/13.0/`/usr/local/cuda`，**缺显式 `cuda-13.2` 路径**（`/usr/local/cuda` 在 8× 机可命中 13.2）。 |
| 8×5090 | `docs/nvidia_build.md`：8×32GB、无 NVLink、CUDA 13.2、NUMA 0=GPU0–3。工作区/权重尚未作为产品约定存在。 |

金线产品命令（对照，任何任务结束后必须仍能这样跑）：

```sh
# GPU0. 不要传 mag/tea/fb/sage/sol/preset/ulysses。
export CUDA_VISIBLE_DEVICES=0
./build-cuda-sm120/h3_cuda_generate \
  --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO" \
  --out "$OUT" --mux "$OUT/out.mp4" \
  --width 864 --height 480 --seconds 5 --steps 8 --cfg 1 \
  --sdpa-backend cudnn --seed 0x11 \
  --token-ids-file "$CAFE_IDS"
```

---

## File map（分解锁死）

| 路径 | 职责 | 为何不塞进 generate |
| --- | --- | --- |
| **Create** `h3_generate_preset.h` / `h3_generate_preset.c` | 解析 `gold` / `turbo4`；填充 `h3_generate_preset_cfg`；缺 LoRA 硬失败 | generate 已 ~4000 行；preset 必须可单测 |
| **Create** `tests/test_h3_generate_preset.c` | 无 GPU：gold 零改写、turbo4 组合、未知名、无 LoRA | TDD |
| **Modify** `tools/h3_cuda_generate.c` | 只接线：`--preset` / `--lora` / `--lora-scale` / help；**不改** `sdpa="cudnn"` 等缺省 | 组合层 |
| **Create** `h3_lora.h` / `h3_lora.c` | 读 safetensors；PEFT `lora_A/B` 与 Kohya `lora_down/up`；映射到 `blocks.{i}.attn.qkv_proj` 等；缺失硬失败 | 加载器是独立合同 |
| **Modify** `h3_dit_w4a8.c` `block_device_compute` | INT8 GEMM 之后加 LoRA residual（U=1 零 A2A） | 唯一算子插点 |
| **Create** `tests/test_h3_lora.c` | 无 GPU：解析夹具、缺键、scale、名字映射 | TDD |
| **Create** `h3_ulysses.h` / `h3_ulysses.c` | 头数/64 对齐 fail-fast；`S_pad`；local rows | **不用 `h3_sp*`**（与 next-opt 同名，避免第二套符号） |
| **Create** `h3_ulysses_comm.h` / `h3_ulysses_comm.cu` | NCCL init/destroy；packed QKV 一次 A2A + attn-out A2A；U=2 可选 IPC | 通信 |
| **Create** `tests/test_h3_ulysses.c` | 无 GPU：U=1/2/4/8 合法，U=7/U=3/TP4+U4 非法，cafe pad=16000 | TDD |
| **Create** `tests/test_h3_ulysses_a2a.cu` | 2 GPU 往返 | 有 GPU |
| **Create** `h3_tae.h` / `h3_tae.c`（Part B） | 轻量视频 decode；**不改** `h3_vae_int8*` 金线 | 另文件 |
| **KEEP** `h3_vision_cuda.*` / T2VA 分支 | 无 ref/keyframe 不跑 vision | 不去重写 TE |
| **KEEP** `h3_nvfp4_cuda.cu` TE pin | `H3_TE_MMAP_POPULATE` / `H3_TE_PREFAULT` | 常驻已有 |
| **Create** `tools/h3_5090x8_sync.sh` | 源码 rsync，无 `--delete`，无密码字面量 | harness |
| **Create** `out/verify_fast_sku/` | 极速 KPI（禁写入金线 66s 表） | 另册 |
| **Create** `out/verify_5090x8_gold/` · `out/verify_5090_sp{2,4,8}/` | 金线复现 / Ulysses 另册 | 另册 |
| **Modify** `CMakeLists.txt` | 新 test target；generate 链 preset/lora；Ulysses+nccl；nvcc 候选补 `cuda-13.2` | 构建 |
| **不改** | `h3_sdpa` 默认 resolve；`--steps` 缺省 2；cache 缺省；Metal 路径 | 金线 |

数值合同（LoRA × INT8，写进 `h3_lora.h` 注释）：

```
禁止把 LoRA merge 进金线 INT8 文件。
运行时 residual（每条匹配线性，在 INT8 ConvRot GEMM 之后）：
  Y = GEMM_i8(X, W)
  Y += lora_scale * (X @ A^T) @ B^T
    PEFT:  A [r, K], B [N, r]
    Kohya: lora_down = A, lora_up = B
VSA / gate / sm100a 键：忽略并 log，5090 不接训练 VSA。
缺 `blocks.{0..49}` 的 qkv/out/fc1/fc2 任一条所需 A/B：load 硬失败。
```

`turbo4` 组合（只填「调用方未显式传入」的槽）：

```
steps = 4
sdpa  = sage3          # 缺插件：沿用现有 resolve 硬失败，禁止 fallback cudnn
cfg   = 1              # 除非 --cfg 已显式
step_cache = OFF；拒绝 mag/tea/fb/reuse>1
必须 --lora PATH（或 H3_TURBO_LORA）；缺失硬失败，stderr 含「非金线」
sigma = 现有 build_serving_sigmas(4, video_shift=12 / audio_shift=3)
        （DMD2 1000,750,500,250→0 对应未移位 base；H3 serving 仍走 shift。
         若 CHOICE.md 证明学生用了自定义表，再加 --sigma-schedule，默认不要发明第二套。）
```

不传 `--preset` ≡ 今日行为。`--preset gold` 是显式 no-op。

---

## 任务目录

**Part A — 单卡 turbo（P0）**

1. Task A1: `--preset turbo4` CLI（steps=4, sage3；默认仍 cudnn 8）
2. Task A2: LoRA 加载器（FastH3 / larryvrh；缺失硬失败）
3. Task A3: 单卡 5090 / 8×5090 GPU0 环境 + cafe 5s warm×3 金线 vs turbo
4. Task A4: 极速 KPI 报告模板

**Part B — 选做加速**

5. Task B1: TAE 成片 opt-in（新文件；仅 turbo）
6. Task B2: TE 常驻 / 去 vision — **KEEP**（核实现，不新写加速）

**Part C — 多卡 Ulysses（可后于 A）**

7. Task C1: 校验器 + pad 64（fail-fast 头约束）
8. Task C2: NCCL A2A + DiT 插点 + `--ulysses`（默认 1）
9. Task C3: U=2 同 NUMA cafe MAE/目视/有声门
10. Task C4: U=4 同 NUMA，再 U=8 跨 NUMA

依赖：A1 ‖ A2 的解析单测；A2 的 DiT 挂钩需要 A1 的 `--lora` 接线可同 PR。A3 依赖 A1+A2。A4 可与 A1 并行写骨架，填数依赖 A3。B1 依赖 A1（拒绝在 gold 上开 TAE）。B2 不阻塞。C 不依赖 turbo 过门；金线 Ulysses 与 turbo Ulysses **禁止混表**。

---

## Part A — 单卡 turbo（P0）

### Task A1: `--preset turbo4` CLI

**Files:**

- Create: `h3_generate_preset.h`, `h3_generate_preset.c`
- Create: `tests/test_h3_generate_preset.c`
- Modify: `tools/h3_cuda_generate.c`（argv 增加 `--preset` / `--lora` / `--lora-scale`；增加 `steps_explicit` / `sdpa_explicit`；help）
- Modify: `CMakeLists.txt`（`h3_generate_preset_test` 无 GPU；`h3_cuda_generate` 链 `h3_generate_preset.c`）

**Interfaces:**

```c
/* h3_generate_preset.h */
typedef enum {
    H3_PRESET_NONE = 0, /* 未传 --preset，与今日一致 */
    H3_PRESET_GOLD,
    H3_PRESET_TURBO4
} h3_preset_id;

typedef struct h3_generate_preset_in {
    const char *preset_name; /* NULL 或 "" → NONE */
    const char *lora_path;   /* --lora 或 NULL */
    int steps_explicit;
    int steps;
    int sdpa_explicit;
    const char *sdpa;        /* 调用方现有指针，可改写为内部静态名 */
    int cfg_explicit;
    float cfg_scale;
    float magcache_thresh;
    int step_cache_kind;     /* h3_step_cache_kind 的 int 值 */
    int reuse_interval;      /* effective reuse; 1 = 全评 */
} h3_generate_preset_in;

typedef struct h3_generate_preset_out {
    h3_preset_id id;
    int steps;
    const char *sdpa;
    float cfg_scale;
    const char *lora_path;
    float lora_scale;        /* 默认 1.0；本任务只透传，A2 使用 */
    int non_gold;            /* turbo4 = 1 */
} h3_generate_preset_out;

/* 0 = 失败（填 err）。NONE/GOLD 不改 steps/sdpa/cfg。 */
int h3_generate_preset_apply(const h3_generate_preset_in *in,
                             h3_generate_preset_out *out,
                             char *err, size_t err_size);
```

- [ ] **Step 1: 写失败单测**（符号尚不存在）

```c
/* tests/test_h3_generate_preset.c */
#include "h3_generate_preset.h"
#include <stdio.h>
#include <string.h>
static int fails;
static void expect(int c, const char *m) {
    if (!c) { fprintf(stderr, "FAIL %s\n", m); fails++; }
}
int main(void) {
    h3_generate_preset_in in;
    h3_generate_preset_out out;
    char err[256];
    memset(&in, 0, sizeof in);
    in.steps = 2; in.sdpa = "cudnn"; in.cfg_scale = 5.0f; in.reuse_interval = 1;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1, "none");
    expect(out.id == H3_PRESET_NONE, "id none");
    expect(out.steps == 2 && strcmp(out.sdpa, "cudnn") == 0, "none no rewrite");
    expect(out.non_gold == 0, "none is gold path");

    in.preset_name = "gold";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1, "gold");
    expect(out.steps == 2 && strcmp(out.sdpa, "cudnn") == 0, "gold no rewrite");

    in.preset_name = "turbo4";
    in.lora_path = NULL;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0, "turbo no lora");
    expect(strstr(err, "非金线") != NULL, "stderr 非金线");

    in.lora_path = "/tmp/fake.safetensors";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1, "turbo ok");
    expect(out.id == H3_PRESET_TURBO4 && out.non_gold == 1, "turbo id");
    expect(out.steps == 4, "steps 4");
    expect(strcmp(out.sdpa, "sage3") == 0, "sage3");
    expect(out.cfg_scale == 1.0f, "cfg 1");

    in.steps_explicit = 1; in.steps = 8;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1, "override steps");
    expect(out.steps == 8, "explicit steps kept");

    in.steps_explicit = 0; in.steps = 2;
    in.sdpa_explicit = 1; in.sdpa = "cudnn";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 1, "override sdpa");
    expect(strcmp(out.sdpa, "cudnn") == 0, "explicit sdpa kept");

    in.sdpa_explicit = 0; in.sdpa = "cudnn";
    in.magcache_thresh = 0.08f;
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0, "turbo+mag");
    in.magcache_thresh = 0.0f;
    in.preset_name = "fast4";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0, "fast4 rejected");
    in.preset_name = "nope";
    expect(h3_generate_preset_apply(&in, &out, err, sizeof err) == 0, "unknown");
    return fails ? 1 : (printf("test_h3_generate_preset: PASS\n"), 0);
}
```

- [ ] **Step 2: 红**

```sh
cmake --build build-cuda-sm120 -j --target h3_generate_preset_test
./build-cuda-sm120/h3_generate_preset_test
```

Expected: 链接失败或 FAIL（符号不存在）。

- [ ] **Step 3: 最小实现** `h3_generate_preset_apply`；generate 在 argv 循环后调用。help 增加：

```
  --preset gold|turbo4
      gold（可省略）= 今日金线路径，不改 steps/backend/cache。
      turbo4 = 非金线极速档：steps=4 + sage3 + 要求 --lora；cache 关。
      未蒸馏 --steps 4 而无 LoRA 不是产品。缺省不传 preset。
  --lora FILE.safetensors   turbo4 必填；金线禁止依赖此文件
  --lora-scale S            默认 1.0
```

argv 伪代码（插在现有循环，**不要**改 `const char *sdpa = "cudnn"` 初始值）：

```c
const char *preset_name = NULL;
const char *lora_path = NULL;
float lora_scale = 1.0f;
int steps_explicit = 0, sdpa_explicit = 0;
/* 在 --steps / --sdpa-backend 分支里置 explicit=1 */
else if (!strcmp(argv[iarg], "--preset") && iarg + 1 < argc)
    preset_name = argv[++iarg];
else if (!strcmp(argv[iarg], "--lora") && iarg + 1 < argc)
    lora_path = argv[++iarg];
```

解析完成后：

```c
h3_generate_preset_in pin = {
    .preset_name = preset_name,
    .lora_path = lora_path,
    .steps_explicit = steps_explicit,
    .steps = steps,
    .sdpa_explicit = sdpa_explicit,
    .sdpa = sdpa,
    .cfg_explicit = cfg_explicit,
    .cfg_scale = cfg_scale,
    .magcache_thresh = magcache_thresh,
    .step_cache_kind = (int)step_cache.kind,
    .reuse_interval = effective_reuse, /* 若尚未算，用 reuse_explicit ? reuse : 1 */
};
h3_generate_preset_out pout;
if (!h3_generate_preset_apply(&pin, &pout, error, sizeof error)) {
    fprintf(stderr, "h3: %s\n", error);
    return 1;
}
steps = pout.steps;
sdpa = pout.sdpa;
cfg_scale = pout.cfg_scale;
if (pout.non_gold)
    fprintf(stderr, "h3: preset=turbo4 非金线（MAE 金线不适用）\n");
```

本任务 **不加载 LoRA**。`--preset turbo4 --lora /nonexistent` 允许 preset 解析成功（文件存在性在 A2）。无 `--lora` 必须在 A1 就失败。

- [ ] **Step 4: 绿**

```sh
cmake --build build-cuda-sm120 -j --target h3_generate_preset_test h3_cuda_generate
./build-cuda-sm120/h3_generate_preset_test
./build-cuda-sm120/h3_cuda_generate --help 2>&1 | grep -F turbo4
./build-cuda-sm120/h3_cuda_generate --preset turbo4 --dit x --vae y --out /tmp/x ; echo exit:$?
```

Expected: 单测 PASS；help 含「非金线」；无 `--lora` 非零退出；stderr 含 `非金线`。

**验收：** 不传 preset 时 argv 行为与改前一致（缺省 cudnn / 无 sage）。turbo 不改 `sdpa="cudnn"` 初始化。`--preset fast4` 失败并提示用 turbo4。

---

### Task A2: LoRA 加载器

**Files:**

- Create: `h3_lora.h`, `h3_lora.c`
- Create: `tests/test_h3_lora.c`
- Create: `tests/fixtures/h3_lora_tiny.safetensors`（单测用：1 个 block、r=2、随机数；由测试生成或仓库内极小夹具）
- Create: `out/verify_fast_sku/CHOICE.md`（选 FastH3 **或** larryvrh，三行理由；**只落地一条**）
- Modify: `h3_dit_w4a8.h` / `h3_dit_w4a8.c`（ctx 挂 `const h3_lora_set *`；`block_device_compute` 四条线性后 residual）
- Modify: `tools/h3_cuda_generate.c`（turbo 路径 `h3_lora_load`；失败硬退出；金线 `lora_path==NULL` 不调用）
- Modify: `CMakeLists.txt`（`h3_lora_test`；generate 链 `h3_lora.c`）

**Interfaces:**

```c
/* h3_lora.h */
enum { H3_LORA_N_BLOCKS = 50 };

typedef struct h3_lora_pair {
    float *A; /* [r, K] row-major */
    float *B; /* [N, r] */
    int r, n, k;
} h3_lora_pair;

typedef struct h3_lora_block {
    h3_lora_pair qkv, out, fc1, fc2;
} h3_lora_block;

typedef struct h3_lora_set {
    char path[768];
    float scale;
    h3_lora_block blocks[H3_LORA_N_BLOCKS];
    int n_pairs;
    int ignored_vsa_tensors;
} h3_lora_set;

int h3_lora_name_is_vsa(const char *tensor_name);

/* Map HF/PEFT/Kohya name → block_index + suffix in {qkv,out,fc1,fc2} + AB.
 * 0 = 不识别（调用方：VSA 则 ignore，其它未知键 log+continue 或按合同 fail）。 */
int h3_lora_parse_name(const char *name, unsigned *block, const char **suffix,
                       int *is_a, char *err, size_t err_size);

int h3_lora_load(const char *path, float scale, h3_lora_set *out,
                 char *err, size_t err_size);
void h3_lora_free(h3_lora_set *set);

/* Host F32 residual: y[M,N] += scale * (x[M,K] @ A[r,K]^T) @ B[N,r]^T */
int h3_lora_apply_f32(const h3_lora_pair *p, float scale,
                      const float *x, float *y, size_t m,
                      char *err, size_t err_size);
```

名字映射（必须覆盖两条公开权，用 CHOICE 后的真实 header 再钉死一层）：

```
blocks.0.attn.qkv_proj.lora_A.weight
blocks.0.attn.to_qkv.lora_A.weight
diffusion_model.blocks.0.attn.qkv_proj.lora_down.weight
lora_unet_blocks_0_attn_qkv_proj.lora_down.weight
→ block=0, suffix=qkv, is_a=1
同理 out_proj / to_out.0 / attn.out_proj → out
mlp.fc1 / ff.net.0 / mlp.0 → fc1
mlp.fc2 / ff.net.2 → fc2
含 vsa / to_gate_compress / sparse：h3_lora_name_is_vsa=1，不进 pairs
```

产品 50 block：load 结束时 `n_pairs` 必须 == 50*4*2 张量（200 对 A/B），否则硬失败。单测夹具可以用 `#define` 降到 1 block（`H3_LORA_TEST_BLOCKS=1` 或 load 参数 `min_blocks` 仅测试用）。**产品 `h3_lora_load` 不暴露 min_blocks。** 测试直接调 `h3_lora_parse_name` + 手建 pair 测 `h3_lora_apply_f32`。

- [ ] **Step 1: CHOICE.md**（可在隔离 Python/FastVideo 跑 cafe，**不进 C**）

在 `out/verify_fast_sku/CHOICE.md` 写三行：选 α FastH3 或 β larryvrh v4；权的远端/HF 路径（**不要本机密码**）；人眼是否过「主体可辨 ∩ 有声」。人眼不过 → **停 A2 产品挂钩**，换权，不要拧 Mag。

- [ ] **Step 2: 写失败单测** `tests/test_h3_lora.c`

```c
#include "h3_lora.h"
#include <stdio.h>
#include <string.h>
static int fails;
static void expect(int c, const char *m) {
    if (!c) { fprintf(stderr, "FAIL %s\n", m); fails++; }
}
int main(void) {
    unsigned b; const char *suf; int is_a; char err[128];
    expect(h3_lora_parse_name("blocks.3.attn.qkv_proj.lora_A.weight",
                              &b, &suf, &is_a, err, sizeof err) == 1, "peft qkv A");
    expect(b == 3 && strcmp(suf, "qkv") == 0 && is_a == 1, "qkv A fields");
    expect(h3_lora_parse_name("blocks.3.attn.qkv_proj.lora_B.weight",
                              &b, &suf, &is_a, err, sizeof err) == 1, "peft qkv B");
    expect(is_a == 0, "B");
    expect(h3_lora_name_is_vsa("blocks.0.attn.to_gate_compress.weight") == 1, "vsa");
    expect(h3_lora_load("/no/such.safetensors", 1.0f, NULL, err, sizeof err) == 0,
           "missing file");
    expect(strstr(err, "非金线") != NULL || strstr(err, "cannot") != NULL,
           "missing file err");
    /* apply: r=1, M=2, K=2, N=2, A=[1,1], B=[1,1] → delta all-ones * scale */
    {
        h3_lora_pair p;
        float A[2] = {1.f, 1.f}, B[2] = {1.f, 1.f};
        float x[4] = {1.f, 0.f, 0.f, 1.f};
        float y[4] = {0, 0, 0, 0};
        p.A = A; p.B = B; p.r = 1; p.k = 2; p.n = 2;
        expect(h3_lora_apply_f32(&p, 2.0f, x, y, 2, err, sizeof err) == 1, "apply");
        expect(y[0] == 2.f && y[1] == 2.f, "row0");
    }
    return fails ? 1 : (printf("test_h3_lora: PASS\n"), 0);
}
```

- [ ] **Step 3: 红 → 实现 parse/load/apply → 绿**

```sh
cmake --build build-cuda-sm120 -j --target h3_lora_test
./build-cuda-sm120/h3_lora_test
```

Expected: 先 FAIL，后 `test_h3_lora: PASS`。

- [ ] **Step 4: 接入 DiT + generate**

`h3_dit_w4a8` 增加 `const h3_lora_set *lora`（NULL = 今日路径）。在 `block_device_compute` 里，`h3_cuda_gemm_i8_dequant_f32_w_act_dev` 成功后，若 `ctx->lora`：对 qkv/out/fc1/fc2 调 device 等价 residual（可用现有 cuBLAS F32：`X[M,K] × A^T[K,r] → T[M,r]`，再 `T × B^T[r,N]`）。**U=1 且 lora==NULL 时指令流与今日一致（零 extra kernel）。**

generate：仅当 `pout.lora_path` 非空时 `h3_lora_load`；失败 `return 1` 且 stderr 含路径与「非金线」。金线命令不碰 `h3_lora_load`。

- [ ] **Step 5: 产品权冒烟（可在 A3 机上）**

```sh
./build-cuda-sm120/h3_cuda_generate --preset turbo4 --lora "$LORA" \
  --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO" \
  --out /tmp/turbo_smoke --seconds 1 --steps 4 --cfg 1 --seed 0x11 \
  --token-ids-file "$CAFE_IDS"
# 缺文件：
./build-cuda-sm120/h3_cuda_generate --preset turbo4 --dit "$DIT" --vae "$VAE" \
  --out /tmp/x ; echo exit:$?
```

Expected: 缺 LoRA 非零；有 LoRA 短片能出（1s 允许）。`--lora /missing.safetensors` 硬失败，禁止静默跳过。

**验收：** 金线 INT8 文件字节不变。VSA 键被忽略。未选第二套加载器。

---

### Task A3: 8×5090 GPU0 环境 + cafe warm×3 对照

**Files:**

- Create: `tools/h3_5090x8_sync.sh`
- Modify: `CMakeLists.txt`（nvcc 候选补 `"/usr/local/cuda-13.2/bin/nvcc"`；若 configure 已命中 `/usr/local/cuda`→13.2 则只在 HOST.md 记一笔）
- Create: `out/verify_5090x8_gold/HOST.md` · `MODELS.md`（无密码）
- Create: `out/verify_5090x8_gold/REPORT.md` 与 `out/verify_fast_sku/cafe_turbo4/` 原始日志目录

**Interfaces:** 无新 C API。SSH 只认 `H3_SSH_5090x8` / `H3_SSH_PASS_5090x8`。也可用旧单卡 `H3_SSH_5090` 当 GPU0 对照，但 **8 卡机 GPU0 是产品锚**。

- [ ] **Step 1: 连通（不拷模型）**

```sh
export H3_SSH_5090x8=user@89.185.31.140
# export H3_SSH_PASS_5090x8=...    # 仅当前 shell；禁止 echo 进文件
ssh -p 22 "$H3_SSH_5090x8" \
  'nvidia-smi -L; nvidia-smi nvlink -s; echo PATH_NVCC=$(command -v nvcc); \
   ls /usr/local/cuda/bin/nvcc; nvidia-smi topo -m | head'
```

Expected: 8× RTX 5090；nvlink 不支持；nvcc 在 `/usr/local/cuda/bin`。写 `HOST.md`：卡数、驱动、CUDA 13.2、无 NVLink、NUMA、磁盘。禁止写密码。

- [ ] **Step 2: sync 源码（无 `--delete`）**

```sh
#!/usr/bin/env bash
# tools/h3_5090x8_sync.sh — source tree only. Never rsync --delete. Never print passwords.
set -euo pipefail
HOST="${H3_SSH_5090x8:-user@89.185.31.140}"
REMOTE_ROOT="${H3_REMOTE_ROOT:-/home/user/h3_c}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ssh -p 22 "$HOST" "mkdir -p '$REMOTE_ROOT'"
rsync -az --exclude '.git/' --exclude 'out/' --exclude 'models/' \
  --exclude 'third_party/accel_survey/FastVideo/' \
  --exclude 'third_party/accel_survey/vdn-minimax-h3/' \
  --exclude 'third_party/accel_survey/MiniMax-H3-MultiGPU-Lightning/' \
  -e "ssh -p 22" "$ROOT/" "$HOST:$REMOTE_ROOT/"
echo "synced source -> $HOST:$REMOTE_ROOT (no --delete, no models)"
```

远端 `ls` 须见 `CMakeLists.txt` `h3_dit_w4a8.c` `tools/h3_cuda_generate.c`。

- [ ] **Step 3: sm_120 构建 + 权重点名拷贝**

```sh
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
cd /home/user/h3_c
cmake -S . -B build-cuda-sm120 -DH3_GPU_BACKEND=cuda -DH3_REQUIRE_CUDA13=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-cuda-sm120 -j --target \
  h3_cuda_info h3_cuda_generate h3_quality_gate_probe \
  h3_step_cache_test h3_generate_preset_test h3_lora_test
./build-cuda-sm120/h3_cuda_info
./build-cuda-sm120/h3_step_cache_test
./build-cuda-sm120/h3_generate_preset_test
./build-cuda-sm120/h3_lora_test
```

Expected: nvcc 13.2.x；arch 含 **120**；单测 PASS。权重从已验证的单卡 5090 **点名** INT8 DiT / fp16 VAE / TE / audio VAE / cafe `prompt.ids` + CHOICE 的一条 LoRA。禁止整树 BF16。`MODELS.md` 只写远端路径与 `H3_MODELS=...`。

Sage3（turbo 用）：按 `docs/nvidia_build.md` 导出 `H3_SAGE3_LIB` / `H3_SAGE3_ROOT`；缺插件时 turbo **必须硬失败**，不要改去 cudnn 凑秒数。

- [ ] **Step 4: 金线 cafe 冷 1 + warm×3（GPU0）**

```sh
export CUDA_VISIBLE_DEVICES=0
# i=1..4；报表只用 2..4
./build-cuda-sm120/h3_cuda_generate \
  --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO" \
  --out "$GOLD/run_$i" --mux "$GOLD/run_$i/out.mp4" \
  --width 864 --height 480 --seconds 5 --steps 8 --cfg 1 \
  --sdpa-backend cudnn --seed 0x11 \
  --token-ids-file "$CAFE_IDS"
```

门：`ffprobe` 视频+音频、时长 5.0s；日志 `sdpa=cudnn`、无 mag/tea/sage、无 preset。warm DiT ~31–32s 量级（新机允许 e2e ±15%）。失败 **停止**，不要跑 turbo 对外报秒数。

- [ ] **Step 5: turbo4 同 prompt/seed 冷 1 + warm×3**

```sh
export CUDA_VISIBLE_DEVICES=0
# Sage3 env 见 nvidia_build.md；缺则硬失败
./build-cuda-sm120/h3_cuda_generate \
  --preset turbo4 --lora "$LORA" \
  --dit "$DIT" --vae "$VAE" --te "$TE" --audio-vae "$AUDIO" \
  --out "$TURBO/run_$i" --mux "$TURBO/run_$i/out.mp4" \
  --width 864 --height 480 --seconds 5 --cfg 1 --seed 0x11 \
  --token-ids-file "$CAFE_IDS"
```

不要传 tea/mag。日志须含 `preset=turbo4` 或 stderr「非金线」、`sdpa=sage3`、`steps=4`。人眼抽 f0/24/48/90/119：主体可辨、无花屏；有声。口型可偏。MAE 用 `h3_quality_gate_probe --ppm-a gold.ppm --ppm-b turbo.ppm` **只记录**，sage 量级 ~27 预期可放行极速，Mag 历史 52–68 当 FAIL。

**验收：** 两套配方都有 warm n=3。金线表（`out/verify_accel_matrix/REPORT.md` §F）一行不改。秒数标已测/推断。回滚测试：再跑 Step 4 命令，行为仍是 cudnn 8-step。

---

### Task A4: 极速 KPI 报告模板

**Files:**

- Create: `out/verify_fast_sku/REPORT.md`（模板 + A3 填数）
- 不修改 `out/verify_accel_matrix/REPORT.md` 的 66s 行

**Interfaces:** 无代码。目录独立。

- [ ] **Step 1: 落模板**（A3 之前可写空表）

```markdown
# 极速 SKU cafe（非金线）

尺子：主体可辨 ∩ 禁花屏 ∩ 有声可用；口型可偏。MAE 记录不设金线阈值。
配方：`--preset turbo4` + 一条 LoRA + sage3 + 官方 VAE；cache 关。
对照金线：同机 GPU0 `--steps 8 --cfg 1 --sdpa-backend cudnn`（另册 `out/verify_5090x8_gold/`）。
warm n=3。真 `--seconds 5`。

| 臂 | e2e mean±std | DiT | TE | VAE | 有声 | 花屏 | 主体 | MAE 五帧 | 性质 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 金线 cudnn 8 |  |  |  |  |  |  |  | 基线 | 已测 / 未测 |
| turbo4 |  |  |  |  |  |  |  |  | 已测 / 未测 |

抽帧：f0 / 24 / 48 / 90 / 119。
禁止：把 32–42s 写成已测（未跑之前）；除进 66s 的公开倍数；未蒸馏 steps=4。
```

- [ ] **Step 2: 用 A3 日志填表**；性质列标 **已测**。推断区间若引用 turbo-sku §4.1，必须写「推断」。

**验收：** 独立目录；金线 66s 行未改；密码/本机绝对路径默认值未入库（远端路径可写在 MODELS.md）。

---

## Part B — 选做加速

### Task B1: TAE 成片 opt-in

**Files:**

- Create: `h3_tae.h`, `h3_tae.c`（视频 latent → RGB；权路径 `--tae`）
- Modify: `tools/h3_cuda_generate.c`：`--vae-decode official|tae`，默认 `official`；`tae` 仅当 `preset=turbo4`，否则硬失败「TAE 不是金线成片」
- Modify: `CMakeLists.txt`：generate 链 `h3_tae.c`；可选 `h3_tae_test`（夹具 decode 形状）
- **不改** `h3_vae_int8_cuda.c` 金线 tile+deblock
- 音频继续 `h3_audio_vae_cuda`（禁止 TinyVAE PCM）

**Interfaces:**

```c
/* h3_tae.h */
int h3_tae_open(const char *path, /* 实现定义的 ctx */ void **ctx,
                char *err, size_t err_size);
void h3_tae_close(void *ctx);
/* latent: 本仓 VAE latent 布局（与 official decode 输入相同）。 */
int h3_tae_decode_f32(void *ctx, const float *latent,
                      int t, int h, int w, uint8_t *rgb_out,
                      char *err, size_t err_size);
```

金线 `--vae-decode` 省略 = official。turbo 未传 = 仍 official（A 的墙钟仍含 ~8s VAE）。本任务是 P1 选做：单卡要再砍 VAE 才做。

- [ ] **Step 1: CLI 合同单测/手测**（无 TAE 权也可）

```sh
./build-cuda-sm120/h3_cuda_generate --vae-decode tae \
  --dit "$DIT" --vae "$VAE" --out /tmp/x --steps 8 --cfg 1
# Expected: 非零，stderr 含「不是金线」或「仅 turbo」
./build-cuda-sm120/h3_cuda_generate --preset turbo4 --lora "$LORA" \
  --vae-decode tae --dit "$DIT" --vae "$VAE" --out /tmp/x
# 无 --tae 权：硬失败，不静默 official
```

- [ ] **Step 2: 实现 decode**；权缺失硬失败。形状与 864×480×120 对齐（或文档写清 TAE 空间尺寸并 resample，禁止悄悄改 `--seconds`）。

- [ ] **Step 3: turbo+TAE cafe 至少 warm n=2**；VAE 段应下降；仍有声；花屏/色偏允许但主体可辨。写入 `out/verify_fast_sku/tae/`，**不进**金线表。

**验收：** 默认 decode 仍 official。无 LTX/TAEHV 任务。

---

### Task B2: TE 常驻 / 去 vision — KEEP

**Files:** 只读核对；必要时 generate 打一行 log（可选，禁止改 TE 数值）。

摸底：

| 项 | 代码 | 本任务 |
| --- | --- | --- |
| T2VA 不跑 Qwen vision | `h3_cuda_generate.c`：`official_kf` / `n_ref_images` / `ref_video_path` 才 `h3_vision_encode_f32` | **KEEP** |
| TE mmap populate | `h3_nvfp4.c` `H3_TE_MMAP_POPULATE` 默认 on | **KEEP** |
| Prefault + packed pin | `h3_nvfp4_cuda.cu` `H3_TE_PREFAULT` / `cudaHostRegister` | **KEEP** |
| 同 prompt 缓存 | **无** | 本计划 **不做**（4-step+TAE 后若 TE 仍 ~12s 再另开任务） |

- [ ] **Step 1: 在 A3 金线/turbo 日志确认** T2VA cafe 无 `vision encode` 墙钟；TE 段仍 ~9–12s 量级。在 `out/verify_fast_sku/REPORT.md` 写「TE KEEP：vision off / pin on；未做 prompt cache」。

- [ ] **Step 2: 不要**为「对称」切 encoder DP；不要在 T2VA 误开 `--ref-image`。

**验收：** 无 TE 行为差。不把「砍 TE」写成已交付加速。

---

## Part C — 多卡 Ulysses（可后于 A）

设计锁死（与 next-opt P1 相同，执行时不许改）：

1. `tp_size=1`，`ulysses∈{1,2,4,8}`，`ring=1`。INT8 权每卡一份。
2. 切 packed 行；attn 前 A2A → 全序列、`H/U` 头；attn 后 A2A 还原。MLP 留在 `S_pad/U`。
3. QKV **一次** destination-major 集体。
4. `heads % tp == 0`；`(heads/tp) % U == 0`；`64 % (U*R) == 0`。非法 fail-fast。
5. cafe S≈15973 → `S_pad=16000`。RoPE pad 行 dummy；SDPA 后切回逻辑 S。
6. TE / VAE / mux 在 device 0。
7. U=2 先同 NUMA（`CUDA_VISIBLE_DEVICES=0,1`）。
8. 单进程 `ncclCommInitAll`；不要 `NCCL_P2P_DISABLE`。U=2 可加 IPC，失败退 NCCL。
9. `--ulysses` 默认 **1**；N=1 不 init NCCL。

金线 Ulysses 用 cudnn；turbo 再挂 U 时 **对照是同配方单卡 turbo**，禁止用金线片子给 turbo U=2 算 MAE 过门。

### Task C1: 校验器 + pad 64

**Files:**

- Create: `h3_ulysses.h`, `h3_ulysses.c`, `tests/test_h3_ulysses.c`
- Modify: `CMakeLists.txt`（`h3_ulysses_test`，不链 CUDA）

**Interfaces:**

```c
/* h3_ulysses.h */
enum { H3_ULYSSES_PACKED_ALIGN = 64 };

typedef struct h3_ulysses_cfg {
    int tp_size;    /* P1: 1 */
    int ulysses;    /* 1,2,4,8 */
    int ring;       /* 1 */
    int rank;
    int world_size;
} h3_ulysses_cfg;

int h3_ulysses_validate(int heads, int packed_align, int tp, int ulysses,
                        int ring, char *err, size_t err_size);
size_t h3_ulysses_pad_seq(size_t seq, int align);
size_t h3_ulysses_local_rows(size_t seq_pad, int ulysses, int rank);
```

- [ ] **Step 1: 失败测试**

```c
/* tests/test_h3_ulysses.c */
#include "h3_ulysses.h"
#include <stdio.h>
static int fails;
static void expect(int c, const char *m) {
    if (!c) { fprintf(stderr, "FAIL %s\n", m); fails++; }
}
int main(void) {
    char err[128];
    expect(h3_ulysses_validate(56, 64, 1, 1, 1, err, sizeof err) == 1, "U1");
    expect(h3_ulysses_validate(56, 64, 1, 2, 1, err, sizeof err) == 1, "U2");
    expect(h3_ulysses_validate(56, 64, 1, 4, 1, err, sizeof err) == 1, "U4");
    expect(h3_ulysses_validate(56, 64, 1, 8, 1, err, sizeof err) == 1, "U8");
    expect(h3_ulysses_validate(56, 64, 1, 7, 1, err, sizeof err) == 0, "U7");
    expect(h3_ulysses_validate(56, 64, 1, 3, 1, err, sizeof err) == 0, "U3");
    expect(h3_ulysses_validate(56, 64, 2, 4, 1, err, sizeof err) == 1, "TP2+U4");
    expect(h3_ulysses_validate(56, 64, 4, 4, 1, err, sizeof err) == 0, "TP4+U4");
    expect(h3_ulysses_pad_seq(15973, 64) == 16000, "cafe pad");
    expect(h3_ulysses_local_rows(16000, 2, 0) == 8000, "r0");
    expect(h3_ulysses_local_rows(16000, 2, 1) == 8000, "r1");
    return fails ? 1 : (printf("test_h3_ulysses: PASS\n"), 0);
}
```

- [ ] **Step 2: 红 → 实现规则 → 绿**

```
heads % tp == 0
(heads / tp) % ulysses == 0
packed_align % (ulysses * ring) == 0
```

```sh
cmake --build build-cuda-sm120 -j --target h3_ulysses_test
./build-cuda-sm120/h3_ulysses_test
```

Expected: PASS。**失败：** 抄 FastVideo 只查 `heads%sp` 而放行 U=7。

---

### Task C2: NCCL A2A + DiT 插点 + CLI

**Files:**

- Create: `h3_ulysses_comm.h`, `h3_ulysses_comm.cu`, `tests/test_h3_ulysses_a2a.cu`
- Modify: `h3_dit_w4a8.c` `block_device_compute`（U=1 走现路径，**零 extra A2A**）
- Modify: `tools/h3_cuda_generate.c`：`--ulysses N` 默认 1；N>1 忽略单 `--device`，用 `0..N-1`；非法 N 在 validate 失败
- Modify: `CMakeLists.txt`：`find_library(NCCL nccl)`；generate 链 ulysses + comm + nccl；`h3_ulysses_test_a2a`

**Interfaces:**

```c
/* h3_ulysses_comm.h — device pointers. dtype_bytes = 4（本仓 GEMM 后 F32 QKV）或 2。 */
int h3_ulysses_comm_init(h3_ulysses_cfg *cfg, const int *device_ids, int n_dev,
                         char *err, size_t err_size);
void h3_ulysses_comm_destroy(h3_ulysses_cfg *cfg);
int h3_ulysses_a2a_qkv(h3_ulysses_cfg *cfg, const void *d_in, void *d_out,
                       size_t seq_local, size_t heads, size_t head_dim,
                       int dtype_bytes, char *err, size_t err_size);
int h3_ulysses_a2a_attn_out(h3_ulysses_cfg *cfg, const void *d_in, void *d_out,
                            size_t seq_local, size_t heads, size_t head_dim,
                            int dtype_bytes, char *err, size_t err_size);
```

插点：

```
QKV GEMM on local rows [S_local, 3*H*D]
→ (+ LoRA residual if turbo)
→ (U>1) A2A QKV → [S_pad, 3*(H/U)*D]
→ rope+sdpa heads=H/U, seq=S_pad   /* 金线保持 cudnn；turbo 可 sage3 */
→ (U>1) A2A out → [S_local, H*D]
→ out-proj + MLP on local rows
```

- [ ] **Step 1: 2 卡合成 A2A 往返**（S_pad=64、H=56、D=8 或 D=128 的缩小张量）。整数填充 max-abs==0。

```sh
CUDA_VISIBLE_DEVICES=0,1 ./build-cuda-sm120/h3_ulysses_test_a2a
```

Expected: PASS。日志 `peer_access=0/1`；不要设 `NCCL_P2P_DISABLE`。

- [ ] **Step 2: `--ulysses 1` 短冒烟**（`--seconds 1 --steps 2`）与改前一致：`sdpa=cudnn`、无 NCCL init。

- [ ] **Step 3: `--ulysses 2` 短冒烟出片；`--ulysses 7` 非零，stderr 含 `packed align`。help 写实验性、默认 1。**

**验收：** 默认路径无 NCCL。U=2 不自动开 sage3。不从 TP2 起步。

---

### Task C3: U=2 同 NUMA 质量门

**Files:** `out/verify_5090_sp2/REPORT.md`

对照：Task A3 的 `out/verify_5090x8_gold/` 同 seed（**金线 ⊗ U=2**）。另：若测 turbo⊗U=2，对照是单卡 turbo，目录 `out/verify_fast_sku/sp2/`。

```sh
export CUDA_VISIBLE_DEVICES=0,1
./build-cuda-sm120/h3_cuda_generate ... --ulysses 2 --sdpa-backend cudnn \
  --steps 8 --cfg 1 --seconds 5 --seed 0x11 --token-ids-file "$CAFE_IDS" \
  --out /home/user/h3_c/out/verify_5090_sp2/run_$i
```

| 门 | 过 |
| --- | --- |
| 视觉 | `h3_quality_gate_probe --ppm-a gold.ppm --ppm-b sp2.ppm` 抽五帧；目视。Ulysses 是置换：MAE 落到 sage3~27 / sol~35 **当实现 bug**，不要改阈值 |
| 有声 | ffprobe 双流；5.0s；波形非静音 |
| 墙钟 | warm×3；报 e2e **和** DiT。DiT 相对单卡 ~31.4s 应明显下降。禁止写「2 卡 → 33s e2e」 |

- [ ] **Step 1: GT 复用 A3 金线片子**
- [ ] **Step 2: U=2 warm×3；填表**
- [ ] **Step 3: 不过门 → `--ulysses` 保持默认 1；禁止叠 sage/tea 把秒数做漂亮**

**验收：** 过门才宣称支持 U=2。数字不除 66s。

---

### Task C4: U=4 然后 U=8

**Files:** `out/verify_5090_sp4/REPORT.md` · `out/verify_5090_sp8/REPORT.md`

- [ ] **Step 1: U=4** `CUDA_VISIBLE_DEVICES=0,1,2,3`（NUMA 0）。同 C3 门。填 `k_eff = DiT_1 / DiT_4`（未测不算）。

- [ ] **Step 2: U=8** GPU0–7。先比 **同配方** U=8 vs U=4 的 DiT。跨 NUMA 若更慢：记「480p 甜点 U=4，U=8 留 768p」。不要先加 TP。仅当 U=8 显存/实现爆了才允许试 TP2+U4，且必须 C3+U=4 已过门。

- [ ] **Step 3: 书面决策** TE 不做；VAE tile 并行仅当仍用官方 VAE 且 e2e 被 VAE 托住（有 TAE 则降优先级）。768p 单卡常驻另目录，OOM 记缺口。

**验收：** `--ulysses` 默认仍 1。失败拓扑不写进「已支持」。

---

## 明确不做（执行者硬停止）

1. 改 generate 缺省 backend / steps / mag / tea / fb / sage / sol / ulysses。
2. 未蒸馏 `--steps 4` 当产品；4-step LoRA merge 进金线 INT8 文件。
3. Mag / FB；4-step 上 Tea；turbo 默认开 sol_cute。
4. TAE 当金线成片。
5. LTX / Super Accel / `cute_sm100` / TAEHV / 6.85s 写 5090。
6. VDN 65GB hybrid；从 TP2 起步；U=7；`NCCL_P2P_DISABLE`。
7. 12.2× / 14× / 22.2× 写成相对 66s；估时标已测。
8. 密码、本机绝对路径默认值入库。
9. 无多卡实现却对外承诺 8 卡秒数。
10. 实现 `h3_sp*` 第二套命名。
11. 并行 FastH3 + larryvrh 两套加载器。
12. `--preset fast4` 作为第二条产品档（已并入 turbo4）。

---

## 验收与回滚（总表）

| 任务 | 验收 | 回滚 | 默认变？ |
| --- | --- | --- | --- |
| A1 | preset 单测 PASS；无 preset = 金线 CLI | 删 `--preset` 接线 | 否 |
| A2 | lora 单测 PASS；缺文件硬失败；INT8 文件未改 | generate 不调用 load | 否 |
| A3 | GPU0 金线 warm×3 + turbo warm×3 | 不用远端工作区 | 否 |
| A4 | `out/verify_fast_sku/REPORT.md` 另册 | 删报告 | 否 |
| B1 | `--vae-decode tae` 在 gold 硬失败 | 默认 official | 否 |
| B2 | KEEP 记录 | 无代码 | 否 |
| C1 | U=7 被拒；pad=16000 | 不链 generate | 否 |
| C2 | U=1 无 NCCL；U=2 短片 | 默认 U=1 | 否 |
| C3 | MAE∩有声 vs 单卡金线 | 默认 U=1 | 否 |
| C4 | U=4/8 诚实数字 | 失败拓扑不宣称 | 否 |

全局回滚（每阶段结束）：

```sh
CUDA_VISIBLE_DEVICES=0 ./build-cuda-sm120/h3_cuda_generate ... \
  --sdpa-backend cudnn --steps 8 --cfg 1 --seconds 5 --seed 0x11
# 日志：sdpa=cudnn，无 mag/tea/sage/sol，无 ulysses>1，无 preset
```

---

## Spec 覆盖自检

| Spec 要求 | 任务 |
| --- | --- |
| `--preset turbo4`；不传 = 金线 | A1 |
| 一条 4-step LoRA；缺权硬失败；禁 merge INT8 | A2 |
| Sage3 在 turbo 内；缺插件硬失败 | A1 组合 + 现有 `h3_sdpa_resolve` |
| 极速 KPI 有声/花屏/主体；MAE 不冒充金线 | A3、A4 |
| 未蒸馏 steps=4 禁止当产品 | A1 help + Global Constraints |
| 单卡 5090 / 8× GPU0 cafe warm×3 | A3 |
| TAE opt-in 仅 turbo | B1 |
| TE 去 vision / pin | B2 KEEP |
| Ulysses 校验/pad/NCCL/U=2 门/U=4/8 | C1–C4 |
| 先 TP=1；无 NVLink；U=7 非法 | C1、C2 |
| Super Accel / LTX 不实现 | 明确不做 |
| turbo 不改默认 | 每任务验收 |

占位符扫描：无 TBD 实现步骤。`CHOICE.md` 的权路径由执行时写入，不是本文占位实现。

---

## 执行交接

计划已写入 `docs/superpowers/plans/2026-09-11-h3c-5090-turbo-implementation.md`。

本轮 **只产出方案**；不改 generate 默认、不跑 GPU、不 commit、不写密码。

**执行选项：**

1. **Subagent-Driven（推荐）** — 每 Task 新 subagent + 两段 review
2. **Inline Execution** — 本会话按 Part A → B → C checkpoint

Which approach?
