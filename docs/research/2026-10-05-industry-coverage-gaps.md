# H3 视频 DiT 保质量加速：业内覆盖盲区核查（只读）

> 来源：子代理 `f6c0cb20-2c47-4694-a35c-fce4a92eb424` 交付（只读）。核查基准：480P hot 16.4s（DiT 10.37s INT8 98–99% 峰值 + sol_cute；VAE 4.96s），turbo4 4-step，golden=8-step。判定标准：非步数缩减、无损或感知无损。

## 盲区 1：Token 级缩减（ToMe / 剪枝 / 聚类）— ✅ 真实缺口（部分）

先验报告只覆盖注意力稀疏化（保留全部 token、稀疏化注意力矩阵），漏了「减少 token 数量」这一支。但直接 ToMe 已证伪，唯一可用是 **AsymRnR**。

| 方法 | 无损性 | 实测加速 | 质量影响 | H3 适用性 |
|---|---|---|---|---|
| ToMe / ToMeSD（直接合并） | ❌ 有损 | ~1.4× | CogVideoX VBench 81.5→79.8（−3.4）模糊/像素化；与 3D RoPE 不兼容 | 不适用，排除 |
| **AsymRnR**（ICML 2025，非对称 K/V 缩减+恢复） | ⚠️ 感知无损（非逐像素） | HunyuanVideo **1.45×**、CogVideoX 1.38×；**FastVideo-Hunyuan 6-step 蒸馏模型 >24%（~1.32×）** | VBench 83.2→83.5（+0.3 略升） | **最相关证据：显式在步数蒸馏模型上验证过，与 turbo4 同构** |

关键：AsymRnR 只合并冗余 K/V token、保留全部 Q，注意力后恢复原序列长度，绕开 ToMe 对 3D RoPE 的破坏。是 token 级缩减族里唯一有感知无损证据的方法。

**对 H3 的现实折算（泼冷水）：**
- 1.45× 是全注意力模型端到端数。我们 DiT 已叠加 INT8 + sol_cute 稀疏，K/V 缩减与稀疏注意力作用在同一计算上，**收益不可叠加**，两层近似误差相乘。
- K/V 缩减只加速注意力，不碰 FFN/线性层（INT8 GEMM 已 98–99% 峰值）。480P 短片注意力占 DiT 比例中等，现实期望约 **1.05–1.15× DiT（≈0.5–1.5s）**，且必须过 golden 8-step 对比——非 bit-exact。
- 结论：真实缺口，但属「近似类二级杠杆」，不是无损主路径。

## 盲区 2：Radial / Video-SPAR 等视频 DiT 专用稀疏注意力 — ⚠️ 枚举缺口，非战略缺口

先验报告覆盖 sol/LoSA/NSA/MoBA/SpargeAttn，2025–2026 这族还有未枚举的：

| 方法 | 加速（端到端） | 质量 | 备注 |
|---|---|---|---|
| Radial Attention（NeurIPS 2025） | 1.8–1.9× | 与 dense 几乎一致 | 静态 O(n log n) mask，无需在线 profiling |
| Sparse VideoGen (SVG) | 2.33×/2.28×/1.51× | PSNR 最高 29 | 空间/时间头在线分类 |
| SVG2 | 2.30×/1.89× | PSNR 30/26 | 语义感知置换 + top-p 预算 |
| Sparse-vDiT | 1.58–1.85× | Wan2.1 PSNR 仅 22.59 —— **非近无损** | 离线 per-head 模式搜索 |
| VORTA | 1.76× | VBench 无损 | 路由式稀疏注意力 |
| VSA（可训练） | 1.7× | 需训练 | 超出 training-free 范围 |
| RAPID（CVPR 2026） | 1.53–2.01× | 与 SVG/Radial 持平 | 跨步复用一次性 mask |
| HEART（2026） | XAttention 1.67→1.85× | 质量同时改善 | 头级自适应 mask 复用，是插件不是新 kernel |

**判定**：都属「近似稀疏注意力」同一类别——先验报告已覆盖该类别并选定 sol_cute + LoSA 路线。2026 LoSA 自报 HunyuanVideo **3.2× 仅损失 0.02 VBench**，已是该族 SOTA 且在已覆盖清单。Radial/SVG2 可作为 sol_cute 不及预期时的备选，值得补一行枚举。**无一家在当前证据下改变路线选择，但注意两条机制差异**：RAPID（跨步复用一次性 mask）与 HEART（头级自适应 mask 复用，是插件不是新 kernel）的机制与 sol_cute 不同，在 sol 调窗不及预期时值得单独评估，不宜与纯 mask 类方法一并关闭。另注意：这些数字均来自 720p/长视频大模型，注意力占比高；480P 短片端到端收益会系统性缩水。

## 盲区 3：vLLM/SGLang 式服务优化 — ⚠️ 当前产品形态下 N/A；形态扩展后需重估

- Continuous batching：多请求吞吐，单请求产品形态下 N/A；**若产品形态扩展到批量/并发（roadmap §1 有批量抽卡场景），continuous batching 需重估**。
- Paged attention：管理 LLM 自回归 KV cache；视频 DiT 无 KV cache 累积，N/A。
- Prefix caching：只在跨请求重复 prompt 前缀时有效，单发 N/A；批量抽卡同 prompt 变 seed 场景下前缀缓存同样需重估。
- 量化感知服务：已有 INT8。
- 旁证：vLLM-Omni 对「质量优先单请求」的推荐就是 baseline + enforce-eager（1.00×）；无损选项只有多 GPU 并行（单 5090 不可用）；Cache-DiT 明确 lossy；torch.compile 预热后 ~1.00×。
- 唯一沾边：vLLM-Omni 的 MP4 编码优化（x264 ultrafast 省 ~1.4s，SSIM 0.96）——与我们已有 ffmpeg/编码工程同向；**caveat：该 SSIM 0.96 未过我们的质量地板 A/B**——本仓自有 x264 测试判定 ultrafast 画质损失明显、不推荐用于正式档（仅 T1 预览档 opt-in 可接受）。

**判定**：在当前单请求产品形态下，先验报告对 sglang/vLLM 的浅覆盖已足够、无真实缺口；但「无缺口」依赖产品形态假设——批量/并发形态落地时本盲区需重开。

## 盲区 4：Diffusion 编译栈 — ❌ 无超越自定义 kernel 的无损收益；5090 支持已就绪

- **torch.compile / Inductor**：PyTorch 2.7+ cu128 已支持 sm_120；Triton sm_120 segfault 已在 3.5.1+ 修复。但对已手写 CUDA kernel 且 GEMM 在 98–99% 峰值的栈，compile 只能融合逐元素算子——正是我们已识别的 QKV-merge/RoPE fusion，**殊途同归**。Inductor 融合会改数值顺序，非 bit-exact。
- **TensorRT**：10.8+ 支持 RTX 50 系；FP4 FLUX 比 FP16 快 >2× 但 FP4 非无损。TRT 的 SDXL 1.7–1.95× 主要来自 INT8/FP8 量化——我们已有峰值 INT8 GEMM，TRT 不会更快。
- **OneDiff/nexfort**：停留 torch 2.1–2.4 / SD3 / Latte 时代，无现代视频 DiT 与 sm_120 证据。**过时，排除**。
- **stable-fast**：UNet 时代，不适用视频 DiT。**排除**。
- **ONNX Runtime**：无超越自定义 kernel 空间。

**判定**：无缺口。编译栈能给的 lossless 融合收益，等价于先验报告已列行动项的 QKV-merge + RoPE fusion + CUDA Graph。

## 盲区 5：2026 SOTA 兜底 — ⚠️ 无类别性遗漏；缓存族新工作「疑似同族、待查一手数据」

- 缓存族新工作**逐项判定**（质量数据均未在本仓查一手值，下述 PSNR 22–26 为 TeaCache/MagCache 同族旧值，不代表新工作自身实测）：
  - **EasyCache（2.1–3.3×）**：疑似与 TeaCache/MagCache 同族（训练-free 跨步特征缓存），自身质量数据**未查**；若确为同族近似则预期同档 PSNR 22–26、非感知无损，但**零本仓实验，不宜直接封档**。
  - **ProfilingDiT**：疑似同族（profiling 引导的缓存），自身质量数据**未查**，同上待证。
  - 结论：两项均「疑似同族、质量证据不足、待查一手数据」——而非「已拒绝，结论不变」。旧族的拒绝结论不能无实验地外推到新工作；如需封档，先查一手 VBench/PSNR 或做小规模 A/B。
- PipeDiT（1.06–4.02× 无损）是多 GPU 流水线，单 5090 不适用。
- Jenga（生成中动态删 token）属盲区 1 家族，无独立无损证据。
- 2026 综述确认版图仍是五族：稀疏注意力 / 缓存 / 蒸馏 / 量化 / 系统优化——与先验报告覆盖一致。

## 最终判定

**填补这些盲区不改变先验核心结论。**

1. **唯一真实缺口是 AsymRnR**（token 级 K/V 缩减，感知无损、在 6-step 蒸馏模型上验证过），及 Radial/SVG2/VORTA 作稀疏注意力备选的枚举补全。建议补录，但标注「近似类二级杠杆，需过 golden 8-step 评估，480P 短片现实期望 ≤1.15× DiT」。
2. 核心结论依然成立：**480P 4-step 处于 pareto 前沿**；真正无损空间仍在工程侧——**VAE decode 的族#6 浪费子集（d2h 串行尾 ~1.6s + FillFunctor ~0.7s ≈ 2.3s，最大无损靶子）+ QKV-merge + RoPE fusion + CUDA Graph**。
3. 量化直觉：DiT 已 98–99% INT8 峰值，注意力侧任何近似方法边际收益被压到 1 秒量级且有质量风险；而 VAE 4.96s 中属零成本工程浪费的 ~2.3s 是确定性、无质量代价的目标。**优先级排序不应被动摇。**
4. **VAE 归因修正（依据 `2026-10-05-acceleration-mechanisms.md` 机理族框架）：** 本节先前提及的「VAE 4.96s 整块是无损靶子」口径需收紧——4.96s 中只有 ~2.3s（d2h 串行尾 + FillFunctor）是族#6 零成本浪费（bit-identical 可达）；vae_blocks 主体 ~2.6s 为 **fp16 未量化真实计算**（族#5 未收割，有 VAE 量化误差质量杠杆）。VAE int8 融合 decode 是族#5×#6 纠缠项，须做误差结构分析 + waffle A/B（同 DiT INT8 流程），不得按零风险工程项对待；T3-c cold-clip 间歇高噪疑似该量化误差的既有症状。
