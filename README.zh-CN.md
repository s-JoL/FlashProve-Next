# FlashProve-Next

**面向生产级大模型 API 的可验证推理**

[![CI](https://github.com/s-JoL/FlashProve-Next/actions/workflows/ci.yml/badge.svg)](https://github.com/s-JoL/FlashProve-Next/actions/workflows/ci.yml)
[![Stage](https://img.shields.io/badge/当前版本-Stage_1_端到端证明器与_vLLM_演示-0f766e)](#分阶段发布路线)
[![Lean 4](https://img.shields.io/badge/Lean_4.29.1-19_模块_·_219_条声明-2563eb)](lean/THEOREM.md)
[![License](https://img.shields.io/badge/License-Apache_2.0-slategray)](LICENSE)

[English](README.md) · 简体中文 · [双语项目主页](https://s-jol.github.io/FlashProve-Next/) · [审阅指南 (`REVIEW.md`)](REVIEW.md) · [规范中文导读 (`specs/`)](specs/README.zh-CN.md) · [英文技术论文 (PDF)](paper/flashprove-next.pdf)

---

## 这是什么

调用闭源大模型 API 时，客户端看不到接口背后到底跑了什么：高峰期有没有暗中切到便宜的小模型？宣传的 `bf16` 权重有没有被压成 `int4`？有没有跳过中间层、截断历史上下文，或者改短隐藏的系统提示词来省思考步数？单凭返回的文字，低比例掺假几乎无法分辨。

**FlashProve-Next** 给每一条 API 回复外挂一份公开可验证的零知识证明，让任何人都能在普通笔记本 CPU 上核实“这条回复确实出自官方公布的那套真实模型配置”，同时守住两条线上生产底线：

1. **不改动现有推理引擎**：不发明定点或整数替身模型，不限制采样和加速算子。线上照常运行原生浮点（`bf16`/`fp8` 配合 `fp32` 矩阵乘累加）、温度与 top-$p$ 随机采样、FlashAttention、CUDA Graph、连续批处理和前缀缓存；证明在流式回复结束后由后台异步生成（当前参考采集与出证开销实测见下文第 1 节）。
2. **不多露一点商业秘密**：除标准商业 API 原本就会返回的字段外，完整隐藏模型权重、层数、宽度、注意力与 MoE 结构、部署精度、隐藏系统提示词、隐藏思维链（`<think>` CoT）原文以及多轮加密状态。

---

## 在任意笔记本纯 CPU 上核验真实 `Qwen3-0.6B` 证明（无需显卡、无需模型权重）

克隆仓库后敲三行命令，即可在任意笔记本 CPU 上独立验证由**完整未裁切 `Qwen/Qwen3-0.6B`**（28 层、词表 151,936）在 **`vLLM 0.33`（开启 CUDA Graph）** 上真实生成的零知识证明包（脚本自动设置 `CUDA_VISIBLE_DEVICES=""`）：

```bash
pip install -e ".[tokenizers]"
python -m flashprove_next.build_ext   # 可选编译 C++ 有限域加速；无编译器时自动回退纯 NumPy
python demo/run_demo.py
```

`demo/run_demo.py` 调用独立验证器 [`verifier/fpn_verify.py`](verifier/fpn_verify.py) 在每条数秒内依次核验：
- **带隐藏系统提示词与隐藏 `<think>` 思维链的诚实首轮回复** $\rightarrow$ **`VERIFIED`**（验证方全程看不到权重、层数或隐藏思考原文）；
- **跨轮绑定加密思考历史的诚实第 2 轮对话回复**（`St_prev -> St_new`） $\rightarrow$ **`VERIFIED`**；
- **带严格 JSON Schema 结构化约束解码的诚实回复** $\rightarrow$ **`VERIFIED`**（本地编译最小字节自动机，并在 $\mathbb F_{p^3}$ 上核验全词表掩码绑定）；
- **服务端偷偷把 `bf16` 权重降级成 `int4` 量化后强行生成的证明** $\rightarrow$ **`REJECTED`**；
- **降级防护、篡改与重放拦截**（不传 `--allow-reduced-security` 默认拒收演示档、改一个可见回复 token、换用新随机数重放旧证明、篡改跨轮 `St_prev`、服务端私自塞入未请求的 Schema） $\rightarrow$ **全部 `REJECTED`**。

*（注：随仓样本包使用轻量的 `qwen3-0.6b-demo` 演示档——4 次计算 + 2 次生成抽查，单包约 10.1 MiB——以便快速克隆与秒级体验；`fpn-verify` 默认只接受全安全生产档，必须显式传入 `--allow-reduced-security` 才允许核验演示档。完整的 $2^{-40}$ 全生命周期可靠性使用 `qwen3-0.6b-sec315` 档，共 $210 + 105 = 315$ 次抽查。）*

---

## 与 FlashProve 的设计对比

Anthropic 的 **[FlashProve](https://github.com/anthropics/FlashProve)** 验证了在 400B 闭源模型上无需硬件飞地（TEE）即可生成零知识证明。其“确定性整数替身 + 全层逐算子重证”的路线在走向在线商业 API 时留下了几处结构性缺口：

| 对比维度 | FlashProve (2026-11) | FlashProve-Next |
|---|---|---|
| **承诺对象** | 4-bit 确定性整数替身模型；线上原生浮点推理会偏离其精确匹配策略 | 线上实际加载的真实权重（`bf16`/`fp8`）、算子浮点舍入合约、隐藏提示词与采样密钥 |
| **线上推理方式** | 仅限贪心或固定种子确定性整数逐层执行 | 零侵入生产推理（原生浮点、随机采样、FlashAttention、CUDA Graph、连续批处理、前缀缓存照常跑） |
| **隐私边界** | 只藏权重；层数、宽度、MoE 结构公开；无隐藏提示词、隐藏 CoT 或多轮状态 | 权重、层数、宽度、MoE/MLA/线性注意力结构、精度、隐藏提示词、隐藏 CoT 与跨轮状态全部隐藏 |
| **承诺寿命** | 掩码权重表最多用 688,127 个批次就彻底耗尽，换新 ID 后无法公开关联 | 出证次数不会耗尽承诺（静态编码一次 + 电路内秘密打开列；可靠性覆盖全生命周期 $Q \le 2^{64}$ 次查询） |
| **出证开销结构** | 逐层全量重证（单张 H100 耗时约为 8×H100 FP8 vLLM 预填充的 71–93 倍） | 按算力成本加权随机抽查（默认三池共 315 次，覆盖 $2^{64}$ 次查询下 $\le 2^{-40}$ 防伪造界），与总层数解耦 |

---

## 目前已完成的实测与机检成果（`Stage 1 · v0.3.0`）

我们严格区分**已在真卡与真实模型上跑通的闭环**、**已由 Lean 4 机器证明的定理**，以及**当前原型的真实开销差距**：

### 1. 完整未裁切 `Qwen3-0.6B` 端到端证明器与 `vLLM 0.33` 在线原型（单张 `RTX 4070 Ti SUPER 16 GB` + `25 GB RAM`）
- **全安全参数 315 次抽查零知识回复证明（$\mathbb F_{p^3}$ 扩域）**：覆盖 `210` 次计算抽查 + `105` 次采样抽查，总规模 **52.46 亿格**（含隐藏系统提示词、用户请求、116 个隐藏 `<think>` CoT token 与可见回复）。通过位打包紧凑见证与按需 AES-256-CTR 盲化重生成，主机内存从 `49.7 B/格`（外推 ~260 GB）压至 **`1.51 B/格`**（**峰值主机内存 `21.00 GiB`，`0 字节`磁盘溢写**，跨槽合批将电路向量数从 ~152 万压至 `6,887`；完整分阶段计时见 [`benchmarks/prover/`](benchmarks/prover/README.md)）。GPU 出证耗时 `3,877.7 s`（证明体积 `149.8 MB`），独立纯 CPU 验证 **`12.02 s` 通过**。
- **一次性模型承诺与 86.7 亿格设置证明 $\pi_{\rm setup}$**：完整 5.96 亿参数静态承诺耗时 **`94.2 s`**（仅存 `7.62 GB` 行系数，零落盘码字），设置证明覆盖第 (i)–(vi) 项共 **86.7 亿格**，GPU 耗时 `3,623.7 s`，独立 CPU 验证 **`59.9 s` 通过**。
- **零侵入 `vLLM 0.33` 在线闭环与 33 项白盒伪造/篡改测试**：通过 `vLLM` 的 `@register_model_loader` 接口与图内自定义算子 `torch.ops.fpn.port` 接入，兼容 `FULL_AND_PIECEWISE` CUDA Graph（重放阶段 **0 次 Python 回调**）。开启满足舍入合约的 `fp32` 矩阵乘累加（`R32`）与 `fp32` 输出头在 batch 1–32 下仅带来 **`1.2%–2.7%`** 吞吐损耗（当前 Stage 1 参考采集器采用每步同步 D2H 拷贝，使解码吞吐下降 **`53%–85%`**，留待 Stage 2 改为双缓冲异步流）。在线跑通 9 组商业 API 场景，且 **`test_reply_zk.py`（21 项）与 `test_api_zk.py`（12 项）共 33 项白盒伪造与篡改测试及在线 `int4` 偷工减料服务端全部被拒绝（`REJECTED`）**。

### 2. 已由 Lean 4 机器检验的定理（`lean/`，19 个模块、219 条声明，零 `sorry`）
在 Lean `4.29.1` + Mathlib 上编译通过，仅依赖三条标准公理：
- **全生命周期逐轮可靠性组合定理（`RoundByRound`、`LifetimeSoundness`、`ProtocolBridge`）**：不依赖“作弊不划算”这类经济假设，证明只要协议满足给定的逐轮分解规格（`ProtocolRBRSpec`）且各轮逃逸密度满足上界（三池共 315 次抽查漏检率 $\le 2^{-105}$、代数与列查询逃逸率 $\le 2^{-106}$），则在 $Q \le 2^{64}$ 次自适应查询下坏接受事件概率不超过 $2^{-40}$（`fpn_protocol_lifetime_soundness`），并精确核验全部 16 格抽查参数表与 $\mathbb F_{p^3}$ / $p^4$ 算术界。（具体 Python 电路满足 `ProtocolRBRSpec` 以及 Basefold 邻近可靠性 `A-PCS` 由论文数学论证与白盒测试支撑。）
- **免排序分桶采样与 $\mathbb F_{p^3}$ 约束解码掩码绑定（`BucketSampling`、`ConstrainedMask`）**：证明分桶采样对任意温度与 top-$p$ 都不会误拒诚实采样，且边界词对的电路检查与原定义完全等价；机检了在 $O(k_2 Q_{\rm pad})$ 格内绑定 JSON Schema 字节自动机词表掩码的多线性 Schwartz–Zippel 定理（错误掩码蒙混通过的概率 $\le 2^{-186}$）。
- **因果多轮零知识盲化与多架构代数等价性（`MaskJoint`、`Gadgets`、`Speculative`、`LinearAttn`、`ArchGeneral`）**：覆盖前缀可测多轮转录均匀分布、Libra 双射、MoE 成对路由、耦合投机解码以及吸收式 MLA、分块线性注意力与流水线/专家并行恒等式。

### 3. 原生浮点容差与防作弊实测（`benchmarks/certified/`）
- 在 `Qwen3-0.6B`（249,384 块）与 `Llama-3.2-1B-Instruct`（143,544 块）上实现**诚实原生 `bf16` 零误拒**，抓出 PyTorch 默认 `bf16` split-K 预填充高达 $39\times$ 的精度违约，对 8 类常见省算力作弊检出 **97.3%–100%** 违规成本，并标定 `Qwen3-4B` 在 4k–32k 上下文下丢弃低注意力权重键所能省下的算力上界（记录标量 LSE 后降至 **5.9%–36.6%**）。

### 4. 当前原型的真实开销差距与 Stage 2 目标
- **出证与在线采集速度**：当前单张消费级 RTX 4070 Ti SUPER 上全安全 315 抽查出证耗时 `3,877.7 s`（约为单路推理的 ~1,009 倍，其中约 `3,000 s` 花在 Basefold 多项式打开与 $2^{24}\dots 2^{26}$ 流式求和上）；Stage 1 参考采集器的每步同步 D2H 拷贝使解码吞吐下降 `53%–85%`（而 `R32` 数值配置本身仅耗 `1.2%–2.7%`）。Stage 2 将通过双缓冲异步 CUDA 拷贝流与多项式打开批处理压降开销。
- **跨架构电路统一**：Stage 2 将把当前面向稠密 `GQA + QK-Norm`（$h=1$ 键产生者追责）的电路泛化为由隐藏配置表连线的统一原语槽与 $h=3$ 子层追责，在同一验证器下通吃 MoE、MLA、滑窗与混合线性注意力。

---

## 分阶段发布路线

| 阶段 | 发布内容 | 状态 |
|---|---|---|
| **Stage 0 (`v0.2.0`)** | 英文技术论文、10 份协议规范（`01`–`10`）、Lean 4.29.1 证明库、真实模型容差与作弊标定、中英双语主页 | **已发布** |
| **Stage 1 (`v0.3.0`)** | $\mathbb F_{p^3}$ 流式跨槽证明器（`1.51 B/格`）、完整 `Qwen3-0.6B` 设置证明与 315 抽查回复证明、独立纯 CPU 验证器（`fpn-verify`）、离线笔记本验证 Demo、无侵入 `vLLM 0.33` 闭环原型、第 19 个 Lean 模块（`ConstrainedMask`） | **已发布** |
| **Stage 2** | 隐藏连线表统一原语槽（覆盖稀疏 MoE / MLA / 滑窗 / 线性注意力）、双缓冲异步采集压向 $\le 5\%$ 在线开销与打开批处理降本、开放字节规范与测试向量 | *规划中* |

---

## 本地复验命令

```bash
# 1. 核验真实 Qwen3-0.6B 证明样本包并运行纯 CPU 快速验证测试（无需 GPU）
python -m flashprove_next.build_ext
python demo/run_demo.py
FPN_CI_FAST=1 python -m tests.test_fast_cpu

# 2. 核验全部 19 个 Lean 4.29.1 模块（零 sorry，仅标准公理）
cd lean && lake exe cache get && bash check.sh

# 3. 重新编译英文技术论文 PDF（零报错、零版面警告）
python3 paper/md2tex.py paper/flashprove-next.md paper/flashprove-next.tex
cd paper && tectonic flashprove-next.tex
```
