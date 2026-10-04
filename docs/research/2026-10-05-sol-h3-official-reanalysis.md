# sol-h3 / NVIDIA Sol-Engine 官方一手重分析

> 来源：子代理 `5b096d05-c052-455c-b7a0-5947be01addf`（只读）。一手源：H3-OnDevice 官方页、Sol-Engine 技术报告 arXiv:2606.23743、Sol-Attn 论文 arXiv:2607.24027 全文、238 上官方 `sol_attn/` 源码、官方文档页；社区：awesome-minimax-H3（17 篇 Reddit 实测汇总）、Saganaki22/ComfyUI-sol-attn + BENCHMARKS、KingGore/ComfyUI_sol-attn_Blackwell。

## 一、官方柱图真实形状：5 技术族 × 3 部署层级

Sol-Engine 报告 §4.2 钉死五族：①跨步缓存 ②稀疏注意力 ③token 剪枝 ④量化 ⑤核融合。**H3-OnDevice 部署只用 3 族（融合核+TeaCache+Sol-Attn）**——token 剪枝与量化在近无损合同下官方主动未用。5090 阶梯（768p·124f·50-step·BF16 stock）：Baseline 1045.4s → +核优化（融合核+**graph capture**+VAE 常驻 BF16）971.6s(1.08×) → +Sol-Attn(τ=1) 708.3s(1.48×) → +TeaCache **231.2s(4.52×)**。**最大单步=跨步缓存 3.07×**。Spark 同族 1.39×（含 graph capture）→3.92×。

## 二、纠偏我们现有认知（重要）

1. **CUDA Graph 非独立柱但确在官方栈**（Spark kernel 阶段写明 graph capture，是 1.39× vs 5090 1.08× 差距的重要成分）——我们完全没有，确认缺失。
2. **官方柱图还有 token pruning/量化两族，H3 保轨迹部署主动没用**——不是漏，是官方近无损合同下只选 3/5 柱。
3. **Sol-Attn 官方没有「window」概念**——我们 480p 全 hit 且 checksum 变化来自官方算法本身的近似（proxy-score reuse），社区也定性为「近似需 A/B」非 bit-exact。社区「lossless」说法仅指 all-exact 模式（tau 极端化）。
4. **我们只用了一半官方 MMDiT 接线**：官方 `make_mmdit_sol_attn_dispatch`=sink 保文本 K/V 被精确看到 **+ 文本 query 行整个走 dense + 前 2 层 dense + 前 10 步 dense**。我们接了 sink（K/V 侧），**没有文本 query 行 dense、dense 层表（`FIRST_DENSE_*=0` 全稀疏）**——这是最值得重新评估的采用项，也是 sol_cute 臂能否往产品默认挪的关键 A/B。
5. **我们 fallback 自研核有 `neigh<=1` 邻块保留，官方 cute_sm120 核没有**——480p 全走官方真核（fallback=0）不受影响，但自研 nested-loop fallback 与官方语义不等价，报数不能叫 sol。

## 三、Sol-Attn 官方机制（论文+源码钉死）

块粒度 64；逐 query 阈值 τᵢ=μᵢ+βσᵢ（thresh_type 有 exact/diag，5090 官方用 diag）；在线 softmax 内路由；**丢弃块不丢——低于阈值的块用零阶 Taylor（pooled key 近似整块 exp score）补偿进 softmax 分子分母**（相对 top-k 高稀疏不崩的关键）。**是近似算法非精确算法**（给 PSNR/SSIM/LPIPS，从不声称 bit-exact）。配方惯例前 ~20% 步 dense warmup + 首层 dense。目标序列 16K–128K 甜区。

## 四、社区独立评判（多源一致）

1. **收益强依赖序列长度，短序列近乎无效**：RTX PRO 6000 实测 480p 仅 2.5%（39.9→38.9s），与我们 480p ~20% 方向一致偏乐观；官方甜区 768p packed(seq~38676)，我们 480p(seq~15973) 已近收益下限。Saganaki22 基准：2K tokens Sol 比 Sage 慢 0.67×，32K/65K 1.44×。
2. **相对 Sage 的 kernel 级优势真实但温和**（5090 1.38–1.44×）；**社区没有独立复现官方 e2e 大倍数**（官方倍数主成分是 cache，社区用 Turbo LoRA 替代——共识「Turbo 加速远大于 Sol-Attn」）。
3. **质量判定：近似、需 A/B、稀疏模式有人看出差异**；没人独立验证 NVIDIA 的 LPIPS 0.293；社区普遍 two-stage 工作流（draft 加速/final 关加速）。
4. **已知局限**：Triton 参考实现长序列 3–17× 慢且偶发 OOM；官方 CuTe 核只放 SM89/90/100/120 仅前向 BF16/D=128 硬合同；CuTe DSL 版本钉死部署脆。

## 五、sol 做了而我们没采用、应重新考虑清单（按机制排序）

| # | 项 | 官方依据 | 状态/预期 |
|---|---|---|---|
| 1 | **torch.compile 区域捕获/CUDA Graph** | Spark 1.39× 含 graph capture | 缺失，纯无损，粗估 DiT 侧 3–8% |
| 2 | **QKV merge**（3 瘦 GEMM→1 宽） | Pillar-1 明列 | 缺失，「免费的钱」 |
| 3 | **fused QKNorm+RoPE** | Pillar-1 成分 | 缺失，需 checksum 门 |
| 4 | **官方 MMDiT 接线另一半：文本 query 行 dense + dense 层表** | 官方近无损背书来源 | **缺失，需 A/B（sol_cute 臂能否挪产品默认的关键）** |
| 5 | **GEMM+GELU 融合** | Pillar-1 成分 | 部分/待核 |
| 6 | ~~TeaCache 跨步缓存~~ | 官方最大单项 3.07× | **维持不采用**：turbo4 已榨干步数冗余；且 Sol-H3 官方 4-step 版也不用 cache 而用蒸馏——官方自己的 4-step 栈印证我们判断 |
| 7 | ~~Token pruning~~ | 官方 H3 部署自己也没用 | 不采用 |

## 六、结论性回答「是否已榨干 sol-h3」：没有

已榨干=注意力稀疏柱（真核已接、用法基本正确）；**没榨干=Pillar-1 的一半（QKV-merge/QKNorm+RoPE/GEMM+GELU/graph capture）+ 官方 MMDiT 接线的质量护栏（dense 层表/文本行 dense）**。但校准预期：官方 5090 阶梯里 Pillar-1 全部只有 1.08×，最大头是 cache（对 turbo4 不适用）——**剩余可榨空间在 DiT 侧约 5–15% 量级，不是倍数级**。真正需修正认知的：(a) Sol-Attn 是近似算法，官方社区都未为默认稀疏模式承诺 bit-exact；(b) 官方 4-step 栈（Sol-H3）恰恰也不用 cache 而用蒸馏，与我们 turbo4 路线同构，大方向没误判；(c) 「window」概念属 INT8 权重窗，与 Sol-Attn 无关，两份叙事勿混。
