# FlashProve-Next

**面向生产级大模型 API 的可验证推理。** 作者：宋亮（Liang Song）

[![CI](https://github.com/s-JoL/FlashProve-Next/actions/workflows/ci.yml/badge.svg)](https://github.com/s-JoL/FlashProve-Next/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/版本-v1.0.0-0f766e)](#版本演进)
[![Lean 4](https://img.shields.io/badge/Lean_4.29.1-28_模块_·_696_条审计声明-2563eb)](lean/THEOREM.md)
[![License](https://img.shields.io/badge/License-Apache_2.0-slategray)](LICENSE)

[English](README.md) · 简体中文 · [双语项目主页](https://s-jol.github.io/FlashProve-Next/) · [审阅指南 (`REVIEW.md`)](REVIEW.md) · [英文论文 (PDF)](paper/flashprove-next.pdf) · [规范全文 (`01`–`11`)](specs/README.zh-CN.md)

---

## 要解决什么问题

调用闭源大模型 API 时，用户看不到接口后面到底跑了什么：高峰期有没有切到更小的模型，宣传的 `bf16` 权重有没有降成 `int4`，有没有跳层、截断上下文，或者改短隐藏系统提示词来省思考步数。只看返回文本，低比例掺假几乎无法发现。

FlashProve-Next 给 API 回复附带一份可验证的零知识掩码证明，同时守住两条线上约束：

1. **不改现有推理引擎。** 不做定点或整数替身模型。线上照常跑原生浮点（`bf16`/`fp8`/`mxfp4` 配合 `fp32` 矩阵乘累加）、温度与 top-$p$ 随机采样、FlashAttention、CUDA Graph、连续批处理和前缀缓存，证明在流式回复结束后异步生成。
2. **保护线上机密。** 证明隐藏模型权重、隐藏系统提示词、隐藏思维链（`<think>` CoT）原文、中间激活与多轮加密状态。`v1.0.0` 中稠密全回复档位仍按 profile 暴露粗粒度几何（层数/宽度档），跨架构隐藏连线表留待 `v2`。

---

## 在任意笔记本 CPU 上体验（无需 GPU，无需模型权重）

### 1. 零依赖 Rust `FPN2` 验证器（`8 MiB` 容器约 `10 ms`）

用零第三方密码依赖、禁用 `unsafe` 的 Rust CLI（`core/`）与 AST 零耦合的独立 Python 参考验证器（`verifier/python/`）核验严格定长 `8,388,608 B` 的 `FPN2` 容器（`fpn2-llm-std-v1`，`Q = 72`，64 行参考电路）：

```bash
pip install numpy   # Python 演示脚本需要；Rust fpn-verify 本身零依赖
cargo build --release --manifest-path core/Cargo.toml --bin fpn-verify
python demo/run_v1_relation2_demo.py
```

在空闲 Apple M1 CPU 上，诚实 `8 MiB` 容器热运行验证约 `10 ms`（Rust）/ `110 ms`（Python），8 项负用例（缺/错 `--circuit-root`、缺/错 `--stmt-digest`、非零填充、截断、门伪造 `E_SUMCHECK`、范围伪造 `E_GKR_ROOT`）在双验证器下返回一致拒绝码，且重证共享 0 个 64 字节私有块。

### 2. 真实 `Qwen3-0.6B` API 证明样本包（`demo/run_demo.py`）

核验随仓附带的真实 `Qwen/Qwen3-0.6B`（28 层、词表 151,936）在 `vLLM 0.33` 上生成的样本包（Stage 1 `FPNZK1` 格式，每包约 `10.1 MiB`，脚本自动强制 `CUDA_VISIBLE_DEVICES=""`，M1 CPU 全套约 `30–145 s`，视 CPU 负载而定）：

```bash
pip install -e ".[tokenizers]"
python -m flashprove_next.build_ext   # 可选 C++ 有限域加速；无编译器自动回退纯 NumPy
python demo/run_demo.py
```

包含 3 条诚实回复（`VERIFIED`：隐藏系统提示词 + 隐藏 CoT、跨轮加密状态链 `St_prev -> St_new`、严格 JSON Schema 约束解码）和 6 项攻击拦截（`REJECTED`：在线 `int4` 量化降级、未传 `--allow-reduced-security`、篡改回复 token、重放随机数、篡改 `St_prev`、私自塞入 Schema）。样本包采用轻量的 `qwen3-0.6b-demo`（`4 + 2` 抽查）；生产档使用 `NI-315`（`210 + 105`）或 `ExtSeed-153`（`102 + 51`）。

---

## 先说清验证路径与 Lean 覆盖范围（`v1.0.0`）

在 `v1.0.0` 中，零依赖 Rust 二进制验证的是自包含 `FPN2` 容器（关系 1–6），而包含 12 轮交互提交的完整 LLM 回复（`FPNZK2`）目前仍走 Python 验证器：

| 格式 | 证明对象 | 证明器 | 验证器 | Lean 4 覆盖情况 |
|---|---|---|---|---|
| **`FPN2`**（二进制定长） | 自包含关系 1–6：关系 2（`8 MiB` 64 行演示）、关系 4 AIR、关系 5–6（`32×1024` 瓦片，含真在线 `vLLM` 30B MoE） | `prover/fpn_prover/backend/` | 零依赖 Rust `core/`（`fpn-verify`，`10–24 ms`）与脱钩 Python `verifier/python/` | 关系 1、3–5 可执行字节验证器（`131/131` 黄金向量全过）+ 条件式 `Fpn2EndToEnd`（[`THEOREM.md`](lean/THEOREM.md) §2.4）；关系 2/6 尚无 Lean |
| **`FPNZK2`**（`8 MiB` 过渡期全零填充 JSON） | 完整 12 轮交互回复（`Qwen3-0.6B ExtSeed-153`、`Qwen3-4B dev 4+2`、`14/14` 作弊矩阵） | `flashprove_next/reply_zk.py` → `folded_engine.py`（ZK 掩码；全模拟器待补） | 隔离纯 CPU Python `verifier/fpn_verify.py`（0.6B `10.8 s` / 4B `7.2 s`，与证明器共用 Python 栈） | 仅覆盖协议级逐轮可靠性、采样与盲化模块，尚无可执行 Lean 验证器 |
| **`FPNZK1`**（Stage 1 gzip） | 历史 `v0.3.0` `NI-315`（`149.8 MB`）及随仓笔记本样本（`demo/run_demo.py`） | 旧版 `flashprove_next/zk_flat.py`（`--backend legacy`） | 隔离纯 CPU Python `verifier/fpn_verify.py` | 仅覆盖协议级逐轮可靠性、采样与盲化模块 |

---

## 与 FlashProve 的取舍对比

Anthropic 的 [FlashProve](https://github.com/anthropics/FlashProve) 证明了无需 TEE 也能对 400B 闭源 MoE 做零知识验证，其路线是逐层精确证明一个 4-bit 确定性整数替身模型。FlashProve-Next 则选择保留原生浮点生产推理栈：

| 对比维度 | FlashProve (2026-11) | FlashProve-Next |
|---|---|---|
| **安全保证** | 4-bit 整数替身模型的全层精确执行 | 原生浮点轨迹的成本加权抽查阈值（$\sigma = \rho = 0.5$）：非交互 `NI-315`（$\varepsilon \le 2^{-40}, Q \le 2^{64}$）或外部种子 `ExtSeed-153`（$\varepsilon \le 2^{-20}, Q_{\rm samp} \le 2^{30}$，须预记录 `C_pre`） |
| **承诺对象** | 4-bit 确定性整数替身模型 | 线上实际加载的 `bf16`/`fp8`/`mxfp4` 权重、逐算子舍入合约、隐藏提示词与采样密钥 |
| **线上推理** | 贪心或固定种子确定性整数逐层执行 | 原生浮点（`fp32` 矩阵乘归约）、温度/top-$p$ 随机采样、FlashAttention、CUDA Graph、连续批处理、前缀缓存 |
| **隐私范围** | 只藏权重；架构公开；无隐藏提示词、CoT 与多轮状态 | 权重、隐藏提示词、隐藏 CoT、中间激活与跨轮状态（`v1.0.0` 稠密全回复按 profile 公开粗粒度几何，跨架构全隐藏归入 `v2`） |
| **承诺寿命** | 688,127 个批次后耗尽 | 永不耗尽的静态 RS 承诺 + 电路内隐藏列打开（`HCO`，[`specs/09`](specs/09-model-commitment.md)） |
| **出证成本结构** | 逐层全量重证 | 固定抽查数（`315` 或 `153` 次），与模型层数 $L$ 解耦 |

### 四项核心构造

1. **永不耗尽的静态 RS 权重承诺（`HCO`，[`specs/09`](specs/09-model-commitment.md)）。** 权重、半径矩阵、隐藏提示词与采样密钥上线时一次性承诺进 `model_id` 并附设置证明 $\pi_{\rm setup}$；每次回复证明在电路内以新鲜盲化打开命中的矩阵瓦片，出证次数不会耗尽承诺。
2. **局部浮点舍入合约与谱系计费（[`specs/06`](specs/06-tolerance-and-cheats.md)）。** 如果整数替身与原生浮点在哪怕极少数请求上分叉，诚实服务“能否出证”这 1 比特本身就会泄露隐藏激活（论文 §9.4）。我们在被抽查算子上直接从引擎自己承诺的输入出发做局部参考检验，误差半径保持在单算子尺度，再配合消费端谱系计费（`Rule CA`）堵住填零跳层漏洞。
3. **免排序分桶采样与 $\mathbb F_{p^3}$ 约束解码（[`BucketSampling.lean`](lean/FlashProveNext/BucketSampling.lean)、[`ConstrainedMask.lean`](lean/FlashProveNext/ConstrainedMask.lean)）。** 用 32 词分桶与整数核夹逼（`sure`/`poss`）替代全词表排序，并用多线性 Schwartz–Zippel（逃逸率 $\le 2^{-186}$）在 $O(k_2 Q_{\rm pad})$ 格内绑定 151,936 词表的 JSON Schema 字节自动机掩码。
4. **零 Phase-2 折叠后端（`multi_gkr` + `multi_pcs` + `folded_engine`）。** 用批处理 ZK 掩码 FracLogUp-GKR（层-0 采用独立 `(0, d_L)/(0, d_R)` 分母盲化）与多尺度分层注入 rate-$1/8$ coset-4 折叠 ZK-Basefold，彻底省去 Stage 1 的 `2.36B` 格 Phase-2 LogUp 倒数承诺。

---

## 实测数据汇总（`v1.0.0`）

以下每条数字都对应仓库 [`benchmarks/`](benchmarks/) 下的公开 JSON 回执，复现驱动脚本在 `bench/` 与 `benchmarks/prover/`。

### 1. 全模型回复证明与作弊矩阵

| 测试项 | 硬件 | 格数（`Ph-2`） | 容器大小 | GPU 证明 / 总计 | 峰值 RAM / VRAM | CPU 验证 | 公开回执 |
|---|---|---|---|---|---|---|---|
| `Qwen3-0.6B` `ExtSeed-153`（`Q=128`，唯一译码 $\le 2^{-106}$） | RTX 6000D 84 GB | `1.47G (0)` | `8 MiB` `FPNZK2` | `450.2 / 569.8 s` | `99.7 / 55.0 GiB` | `10.76 s` (Py) | [`qwen3_06b_extseed153_q128`](benchmarks/prover/results/qwen3_06b_extseed153_q128_rtx6000d.json) |
| `Qwen3-0.6B` `ExtSeed-153`（`Q=100`，`A-PCS-fold`） | RTX 6000D 84 GB | `1.47G (0)` | `8 MiB` `FPNZK2` | `461.1 / 571.0 s` | `100.1 / 54.3 GiB` | `10.73 s` (Py) | [`qwen3_06b_extseed153_v1`](benchmarks/prover/results/qwen3_06b_extseed153_v1_rtx6000d.json) |
| `Qwen3-0.6B` `8+4` 作弊矩阵（`Q=100`，1 诚实 + 13 攻击） | RTX 6000D 84 GB | — | `8 MiB` `FPNZK2` | `14 / 14` 通过 | — | 精确拒绝码 | [`qwen3_06b_cheating_matrix`](benchmarks/prover/results/qwen3_06b_cheating_matrix_rtx6000d.json) |
| `Qwen3-4B` 36 层、40.2 亿参数（开发档 `4+2`，`Q=100`） | RTX 6000D 84 GB | `259M (0)` | `8 MiB` `FPNZK2` | `160.3 / 273.0 s` | `58.2 / 47.8 GiB` | `7.18 s` (Py) | [`qwen3_4b_mc4_mg2_v1`](benchmarks/prover/results/qwen3_4b_mc4_mg2_v1_rtx6000d.json) |
| `Qwen3-0.6B` Stage 1 `NI-315`（`1.51 B/格` 流式，含 `116` 隐藏 CoT） | RTX 4070 Ti S 16 GB | `5.25G (2.36G)` | `149.8 MB` `FPNZK1` | `3,752 / 3,878 s` | `21.0 / 11.3 GiB` | `12.02 s` (Py) | [`qwen3_06b_reply_315`](benchmarks/prover/results/qwen3_06b_reply_315.json) |
| `Qwen3-0.6B` Stage 1 设置证明 $\pi_{\rm setup}$（第 i–vi 项） | RTX 4070 Ti S 16 GB | `8.67G` | `353.0 MB` | `3,623.7 s` | `12.2 / 4.2 GiB` | `59.91 s` (Py) | [`qwen3_06b_mcsetup`](benchmarks/prover/results/qwen3_06b_mcsetup.json) |

口径说明：
- `14/14` 作弊矩阵包含在真实 `Qwen3-0.6B` 上修改全部 196 个线性层权重并跑完 28 层前向、以及 group-32 `int4` 量化 196 个线性层权重并跑完 28 层前向对照原 `bf16` 承诺的真实攻击，其余覆盖跳层、截断 KV、篡改隐藏提示词/可见回复/`St_prev`、缺/换 `C_pre` 与 `c_ext`、非零填充、降级 `Q = 4` 以及 `4 MiB` vs `8 MiB` 封套长度不符。
- 上述两次 `ExtSeed-153` 运行在到达 `</think>` 前即达到输出长度上限（`n_reason = 0`）；隐藏 CoT（`n_reason = 116`）在 Stage 1 `qwen3_06b_reply_315.json` 与 `demo/run_demo.py` 中实测通过。
- 单流 `Qwen3-0.6B ExtSeed-153` 总出证（`569.8 s`）约为单流推理（`5.54 s`）的 `103×`；把集群总出证算力压在服务算力 `100%` 以内依赖概率抽审威慑（`p_audit <= 1%`，定义见 [`specs/02-soundness.md`](specs/02-soundness.md) §10）。

### 2. 前沿模型标定、异步 `vLLM` 采集、MoE 关系 6 与多轮状态（RTX 6000D）

- **五模型 Blackwell 全算子标定（0 诚实误拒）：** 覆盖 `Qwen3-4B`、`Llama-3.2-1B-Instruct`、`Qwen3.5-0.8B`、`Qwen3-30B-A3B-FP8` 与 `gpt-oss-20b`——`0 / 1,846,272` GEMM+MoE 专家块（`863,232` GEMM + `983,040` MoE）与 `0 / 73,728` 路由器诚实 token（[`frontier_calibrate`](benchmarks/certified/results/frontier_calibrate_rtx6000d.json)；单 token 路由器交换检出率因零间隔 top-$k$ 平局为 `11.5%`/`38.0%`），以及 `0 / 1.96M` 非 GEMM 行（RMSNorm、GQA Softmax、SWA + sinks、分块 64 WY Gated DeltaNet，[`non_gemm_calibrate`](benchmarks/certified/results/non_gemm_calibrate_rtx6000d.json)）。早期 Ada `bf16` 标定见 [`benchmarks/certified/`](benchmarks/certified/README.md)。
- **真在线 `vLLM 0.33` `Qwen3-30B-A3B-FP8` 零明文 Relation 6（[`live_vllm_rel6_zk_v1`](benchmarks/prover/results/live_vllm_rel6_zk_v1_rtx6000d.json)）：** 临时密钥 AES-256-CTR 惰性分块承诺将留存缓冲压降 `192×`（30B MoE `25.9 GiB -> 135.0 MiB`，进程峰值 RSS `VmHWM` 增量 `0.0 MiB`）、预承诺 `45.1 s`，64 片真实瓦片（`64/64` 整数保真）出证 `1.49 s`，**Rust `fpn-verify` 耗时 `23.4 ms` `VERIFIED`**，`0 B` 明文激活披露，伪造全拒。*（范围说明：Relation 6 打开 `16.8 MB` 公开权重瓦片，且未将 `X`/`Y` 绑至 GPU 捕获/服务输出，属零明文激活组件里程碑，非闭源权重全回复证明。）*
- **零同步异步采集与 20 轮增量状态：** 开启 `R32` 数值配置仅耗 `1.2%–2.7%` `vLLM` 吞吐（[`vllm_serving_overhead`](benchmarks/prover/results/vllm_serving_overhead_qwen3_06b.json)）。零同步异步 CUDA-graph 采集（`serve/fpn_serve/`）在 `Qwen3-4B` `TIER_B` 6 格下 0 额外同步、0 溢出，相对 `numerics_r32` 的端到端吞吐变化为 `-1.15%`（`b1, c256`）、`-3.80%`（`b4, c256`）、`-7.23%`（`b8, c256`）、`-4.90%`（`b1, c8192`）、`-11.25%`（`b4, c8192`）与 `-13.14%`（`b8, c8192`）（[`async_capture_qwen3_4b`](benchmarks/prover/results/async_capture_qwen3_4b_rtx6000d.json)）。20 轮增长型 `vLLM` 会话（`18,506` tokens，`95.8%` 前缀命中）通过 `PageStore` 去重 `10.5×`，第 20 轮增量 `St_i` 耗时 `0.75–0.92 s`（相对单线程 `28–34×` / 多线程 `9.9–13.6×`，[`agentic_20turn`](benchmarks/prover/results/agentic_20turn_qwen3_06b_rtx6000d.json)）。

### 3. Lean 4 形式化（`lean/`，28 模块，696 条公理审计声明，零 `sorry`）

在 Lean `4.29.1` + Mathlib 上通过 `lean/check.sh` 编译并完成 `#print axioms` 审计（禁用 `native_decide`，仅标准三公理）：
- **19 个协议模块（219 条声明）：** 覆盖逐轮可靠性组合（`RoundByRound`、`LifetimeSoundness`、`ProtocolBridge`）、16 格抽查表、多列 LogUp、分桶采样（`BucketSampling`）、$\mathbb F_{p^3}$ 约束掩码绑定（`ConstrainedMask`）、因果多轮盲化（`MaskJoint`）以及 MoE/MLA/线性注意力/投机解码恒等式。
- **9 个 `FPN2` 模块（477 条声明）：** 不依赖 Mathlib 的可执行字节验证器（`VerifyBytes`、`Fpn2Rel3`–`5`，通过全部 `131/131` 黄金向量与 `26/26` 多瓦片测试）、具体 Goldilocks $\mathbb F_{p^3}$ 可靠性（`Fpn2Soundness`）、6 类 IR 算子翻译校验（`Fpn2Gadgets`）以及在 `verifyBytesRel4`/`5` 上的条件式 `Fpn2EndToEnd` 实例化（前提与 `Delta` 边界见 [`lean/THEOREM.md`](lean/THEOREM.md) §2.4）。

### 4. 明确边界与 `v2` 方向

- **全回复接入单一 Rust 验证器：** 将 `reply_zk` 的 12 轮交互提交下沉为自包含 `CircuitDesc`。
- **折叠后端内存流式化与 4B+ 生产档：** 恢复约 `1.5 B/格` 流式内存（把 `100 GiB` 峰值压回 ~`21 GiB`），补齐 4B 及 30B+ 闭源全回复生产抽查档。
- **形式化与 ZK 证明补齐：** 补全多尺度分层注入的 Johnson 邻近 RBR（`A-PCS-fold`，`Q >= 128` 已在唯一译码区绕开）、折叠后端全转录本 ZK 模拟器、关系 2/6 的 Lean 验证器，以及 Python 电路到 `CompiledCS` 的机械提取。
- **跨架构隐藏：** 用统一隐藏连线表原语槽取代按 profile 公开的稠密几何。

---

## 版本演进

| 版本 | 主要内容 | 状态 |
|---|---|---|
| **Stage 0 (`v0.2.0`)** | 英文论文、规范 `01`–`10`、18 个 Lean 4 模块、Ada `bf16` 容差与作弊标定 | 已发布 |
| **Stage 1 (`v0.3.0`)** | 流式 $\mathbb F_{p^3}$ 证明器（`1.51 B/格`）、`Qwen3-0.6B` 设置与 `NI-315` 证明、CPU Demo、`vLLM 0.33` 插件、第 19 个 Lean 模块 | 已发布 |
| **`v1.0.0`** | 零 Phase-2 折叠后端（`8 MiB` `FPNZK2`、`Q=128` 唯一译码）、零依赖 Rust `fpn-core`/`fpn-verify` + 脱钩 Python 参考（`specs/11` 关系 1–6）、28 个 Lean 4 模块（696 条审计声明）、RTX 6000D `Qwen3-0.6B`/`4B`/`30B-A3B-FP8` 与五模型标定 | **当前版本** |
| **`v2`** | 12 轮全回复入 Rust 单一 CLI、折叠后端流式内存、跨架构隐藏连线表、Lean 关系 2/6 | 规划中 |

---

## 本地复验命令

```bash
# 1. 零依赖 Rust FPN2 关系 2 CPU 演示（8 MiB, Q=72, 64 行参考电路）
pip install numpy
cargo build --release --manifest-path core/Cargo.toml --bin fpn-verify
python demo/run_v1_relation2_demo.py

# 2. 真实 Qwen3-0.6B Stage 1 样本包与快速 CPU 检查
pip install -e ".[tokenizers]"
python -m flashprove_next.build_ext
python demo/run_demo.py
FPN_CI_FAST=1 python -m tests.test_fast_cpu

# 3. 核验全部 28 个 Lean 4.29.1 模块（696 条审计声明，零 sorry，仅标准三公理）
(cd lean && lake exe cache get && bash check.sh)

# 4. 重新编译英文论文 PDF（零报错、零警告）
python3 paper/md2tex.py paper/flashprove-next.md paper/flashprove-next.tex
(cd paper && tectonic flashprove-next.tex)

# 5. 全套 Rust 验证器测试、脱钩 Python 验证器测试与黄金向量一致性检查
cargo test --manifest-path core/Cargo.toml --release
PYTHONPATH=verifier/python python -m unittest discover -s verifier/python/tests
python specs/vectors/generate.py --check
```
