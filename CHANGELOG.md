# Changelog

All milestone releases of **FlashProve-Next** are documented here.

---

## [v0.2.0] — 2026-11-16 — Stage 0: Protocol Theory, Lean 4 Proofs & Hardware Calibration

### English
- **Technical Report (`paper/flashprove-next.pdf`):** *FlashProve-Next: Verifiable Inference for Production LLM APIs*—presents a publicly verifiable, zero-knowledge proof protocol for closed-weight language model APIs that leaves production serving unchanged (native floating point, random sampling, FlashAttention, CUDA Graph, continuous batching, TP/PP/EP, speculative decoding), hides architecture and private context, never exhausts its weight commitment, and checks a fixed cost-weighted sample ($3 \times 105 = 315$ draws for $Q=2^{64}, \varepsilon \le 2^{-40}$) rather than re-proving every layer.
- **Protocol Specifications (`specs/01`–`10`):** Ten self-contained specifications covering the versioned response relation, lifetime soundness, explicit-leakage zero knowledge, completeness & failure state machine, production serving & API semantics, rounding-contract tolerance & cheat catalog, 30 assumptions & Lean correspondence, related work, one-time weight commitment, and long-context KV history commitment.
- **Standalone Lean 4.29.1 + Mathlib Library (`lean/`):** **18 modules and 194 axiom-audited declarations** (`0 sorry`, standard axioms only), proving the end-to-end protocol-to-`RoundByRound` lifetime soundness theorem, all 16 sample-count table cells, sort-free bucketed sampling completeness and zero-slack circuit checks, causal multi-round masking uniformity, Libra bijections, pairwise MoE routing, coupled speculative decoding, and algebraic identities for MLA, chunked linear attention, and PP/EP.
- **Hardware Calibration Suite (`benchmarks/certified/`):** Raw JSON results and tables on `NVIDIA RTX 4070 Ti SUPER 16 GB`: **0 / 392,928** false rejections on honest `Qwen3-0.6B` and `Llama-3.2-1B-Instruct` blocks, detection of PyTorch's default `bf16` split-K prefill contract violation, **97.3%–100%** bad-cost detection across 8 cost-saving cheats, and 4k–32k sparse-attention tail bounds on `Qwen3-4B`.

### 中文
- **英文技术论文（`paper/flashprove-next.pdf`）：**《*FlashProve-Next: Verifiable Inference for Production LLM APIs*》——提出面向生产级闭源大模型 API 的零知识推理证明协议：完全不改动线上推理引擎（照常运行原生浮点、随机采样、FlashAttention、CUDA Graph、连续批处理、多卡并行与投机解码），不泄露标准 API 之外的模型结构与隐藏提示词/思维链，模型权重一次承诺永不耗尽，并通过按成本随机抽查（默认 315 次，覆盖 $2^{64}$ 次哈希查询下 $\le 2^{-40}$ 防伪造界）替代全层重证。
- **协议与安全规范（`specs/01`–`10`）：** 10 份独立规范文档，涵盖单次回复证明陈述、全生命周期防伪造、零知识信息边界、零浮点误拒与故障状态机、主流架构与商业 API 兼容、算子舍入误差传播与分桶采样、30 项前提假设与 Lean 对应表、相关工作对比、一次性模型承诺及长上下文 KV 历史承诺。
- **独立完整 Lean 4.29.1 + Mathlib 证明库（`lean/`）：** **18 个模块、194 条定理声明**（零 `sorry`，仅依赖三条标准公理），机器验证了全生命周期防刷可靠性主定理、16 格抽查参数表、分桶温度/top-$p$ 采样完备性、多轮盲化均匀性、MoE 路由、耦合投机解码及 MLA / 分块线性注意力 / 多卡并行等价性。
- **真实模型硬件实测数据（`benchmarks/certified/`）：** 收录 `RTX 4070 Ti SUPER 16 GB` 上 `Qwen3-0.6B` 与 `Llama-3.2-1B-Instruct` 共 **392,928 个诚实算子块零误拒**、PyTorch 默认 `bf16` split-K 精度违约实证、**8 类省算力作弊 97.3%–100% 违规成本检出率**，以及 `Qwen3-4B` 长上下文稀疏注意力尾部上界原始数据。
