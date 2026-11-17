# Spec 09: Static Reed–Solomon Model Commitment & Hidden Column Openings (`fpn-modelcom-1.0`)

**Specification ID:** `fpn-modelcom-1.0` | **Role:** Non-exhausting one-time Reed–Solomon weight and radius commitment, single-transcript setup proof $\pi_{\mathrm{setup}}$, in-circuit hidden configuration opening, and unbounded lifetime zero knowledge

---

## Overview and Design Space

A production verifiable-inference deployment commits to a model checkpoint once and serves millions to trillions of responses under that single `model_id`. Three fundamental constraints govern how weights can be committed and read inside per-response proofs:

1. **Static codeword exhaustion under public openings (Lemma 2.1 `[Paper]`).** In any hash-based polynomial commitment (Basefold, FRI, WHIR, Ligero) where the verifier inspects opened symbols of a *static* codeword in public transcripts, the union of revealed coordinates per row grows across proofs: after $T$ proofs opening $m_q$ random columns of a rate-$\vartheta$ Reed–Solomon matrix $\mathbf M \in \mathbb F_p^{n_r \times n}$, the expected number of distinct revealed positions per row is $n(1 - (1 - m_q/n)^T)$. Once this reaches the message-plus-blinding length $(k+b) = \vartheta n$ ($T \lesssim (k+b)/m_q$), MDS interpolation recovers the secret weights completely. Adding fresh per-proof masks (`T4`) does not prevent exhaustion if static Merkle leaves are opened in public. Unbounded lifetime zero knowledge therefore requires either homomorphically re-randomizable commitments or **keeping static codeword symbols strictly inside freshly blinded zero-knowledge circuits**.
2. **Work proportional to touched weights (Lemma 2.3 `[Paper]`).** Evaluating a multilinear polynomial over all model weights at a non-boolean matrix-index point touches $\Omega(|W|)$ parameters on every proof, forcing sparse MoE proofs to scan inactive experts. Scaling prover work strictly with **touched** tiles requires boolean selection over hidden tile indices.
3. **Hidden architectural coordinates (`[Paper]`).** Because layer count, width, and active MoE expert indices are protected under $\mathcal L_{\mathrm{API}}$, opening weights at a public point $(m, \boldsymbol\xi, \boldsymbol\zeta)$ would directly leak which matrix $m$ was sampled.

**Selected construction: Static Reed–Solomon commitment + in-circuit Hidden Column Openings (`HCO`, §4).** Model weights, per-row decode scales, exact radius weights $\mathbf W^r$, and stable configuration tables are arranged into a $R_{\max} \times n_c$ matrix, encoded row-wise with a rate-$1/4$ Reed–Solomon code, and committed column-wise into salted Poseidon2 Merkle trees ($\mathsf C_W, \mathsf C_{\mathrm{raw}}$) accompanied by a one-time zero-knowledge setup proof $\pi_{\mathrm{setup}}$. In each response proof, all sampled GEMM products reduce in two rounds to hidden tile claims at a single shared evaluation point $(\boldsymbol\xi, \boldsymbol\zeta)$, combine via a hidden sparse vector $\boldsymbol\alpha$ into one linear combination row $\mathbf u = \boldsymbol\alpha^\top \mathbf W$, and verify against $m_q = 123$ public random column indices whose **column contents and Merkle paths remain hidden witnesses inside the outer zero-knowledge circuit**. Static codeword symbols never enter any public transcript, achieving **zero wear across unbounded lifetime proofs** with $O(\text{touched weights})$ prover work.

*(Alternative candidates evaluated and deferred: Epoch re-randomization leaks request volume via epoch-rotation frequency or runs out of capacity under traffic spikes; lattice homomorphic commitments such as Ajtai + Greyhound/LaBRADOR/LatticeFold avoid wear-out and remain promising for Stage 2 research, whereas discrete-log Hyrax/Dory commitments sacrifice post-quantum security and require leaving the 64-bit Goldilocks stack.)*

---

## 1. Formal Static Exhaustion & Selection Lemmas

### 1.1 Static Codeword Exhaustion `[Paper]`

**Lemma 2.1.** Let $\mathbf M \in \Sigma^{n_r \times n}$ be a fixed codeword matrix committed once, where row $i$ is the rate-$\vartheta = (k+b)/n$ Reed–Solomon encoding of $k$ secret weight elements $\mathbf w_i$ and $b$ commitment-time blinding elements $\mathbf g_i$. If each proof reveals $\mathbf M_{:, \mathcal Q}$ at $m_q$ uniform random column indices $\mathcal Q$ in its public transcript, then after $T$ proofs the expected number of distinct revealed column coordinates per row is $n\bigl(1 - (1 - m_q / n)^T\bigr)$. Once $\ge k + b$ distinct coordinates are revealed (occurring in $T \lesssim (k+b)/m_q$ proofs), MDS unique interpolation determines $(\mathbf w_i, \mathbf g_i)$ uniquely.

**Corollary 2.2.** Fresh per-opening linear masks ($F' = \sum \epsilon^i f_i + \epsilon^k h$) do not avert Lemma 2.1 whenever the verifier checks consistency against static leaves in the clear. To eliminate lifetime opening budgets without homomorphic re-randomization, static Merkle leaves and paths must be verified exclusively as hidden witnesses inside a freshly blinded outer proof.

---

## 2. Construction: Matrix Layout, `model_id`, and Joint Setup Proof $\pi_{\mathrm{setup}}$

### 2.1 Dual-Half Matrix Layout (`v2`) and Radius Matrix `[Def]`

Let $T_o \times T_i$ be the public tile shape ($1024 \times 1024$, $n_c = T_o T_i = 2^{20}$ for the `Qwen3-0.6B` profile) and $n_r = R_{\max}$ the public row upper bound ($R_{\max} = 2048$). Every weight tensor (attention projections, MLP matrices, routed/shared MoE experts, token embeddings, and LM head) is partitioned into $T_o \times T_i$ tiles:

- **Lower half $[0, R_{\max}/2)$ — Canonical Decode ($\mathbf W^{\mathrm{cd}}$) & Metadata:**
  - Row `0` (`SCALE`): stores the signed block-floating-point decode exponent $G_i \in [12, 28)$ of every row $i$;
  - Row `1` (`PN`): stores exact per-tile-row squared Euclidean norms $\mathrm{PN}[i T_o + r] = \sum_{c < T_i} \mathbf W^{\mathrm{cd}}[i, r T_i + c]^2$;
  - Vector rows: 1D gain vectors (`RMSNorm` / `q_norm` / `k_norm` weights);
  - $N_{\mathrm{cr}}$ stable configuration rows (§3);
  - Weight tile rows $\mathbf W_t^{\mathrm{cd}}$, where each entry is $\mathrm{cd} = \mathrm{trunc}(\sigma \cdot 2^{E + G_t})$ under the **maximal** integer exponent $G_t \in [12, 28)$ keeping $\max_k |\mathrm{cd}_k| < 2^{22}$ (`[Lean: ArchGeneral.sbfp_shift_unique]`). Enforcing maximality ($\exists k, |\mathrm{cd}_k| \ge 2^{21}$ unless the row is identically `0x0000`) prevents a provider from covertly quantizing weights by under-scaling $G_t$;
  - Row $R_{\max}/2 - 1$: all-zero padding row for unused claim slots.
- **Upper half $[R_{\max}/2, R_{\max})$ — Certified Radius Matrix ($\mathbf W^r$):**
  - Certified GEMM error propagation ([Spec 06](06-tolerance-and-cheats.md) §2.2 Row 1) requires both $Y = \mathbf W^{\mathrm{cd}} X$ and $A = \mathbf W^r Z$, where

    $$
    \mathbf W_t^r = \bigl|\mathbf W_t^{\mathrm{cd}}\bigr| + \kappa_t\, m_t, \qquad m_t = \bigl[\,w \cdot 2^{G_t} \notin \mathbb Z\,\bigr], \qquad \kappa_t = \bigl\lceil (1 + \gamma_n)/\gamma_n \bigr\rceil.
    $$

    Because $|\cdot|$ is non-linear, $\mathbf W_t^r$ is committed directly at partner row $R_{\max}/2 + t$ (with partner zero row $R_{\max}-1$) and opened in the same HCO pass. On `Qwen3-0.6B`, only $0.04\%\text{–}0.05\%$ of `bf16` entries have $m_t = 1$, keeping radii nearly $5\times$ tighter than a coarse $2^{-G_t}\|x\|_1$ bound `[Meas]`.
- **Raw bitstring matrix $\mathbf W^{\mathrm{raw}}$:** Stores the exact deployed `bf16`/`fp8` codes at matching tile coordinates (zero on metadata and radius rows).
- **Reed–Solomon encoding & column trees:** Each row of $\mathbf W$ (and $\mathbf W^{\mathrm{raw}}$) is Reed–Solomon encoded at rate $\vartheta = 1/4$ to length $n = 4n_c = 2^{22}$. Column $j$ is hashed with a fresh 256-bit salt $\varsigma_j$ via Poseidon2 sponge $H_{\mathrm{in}}(\texttt{dom}_{\mathrm{col}}, j, \varsigma_j, R_{\max}, \mathbf M_{:,j})$ into fixed-depth column Merkle roots $\mathsf C_W$ and $\mathsf C_{\mathrm{raw}}$. With master salt $\varsigma_0$ and volatile configuration root $\mathsf C_{\mathrm{cfg}}$ (§3):

$$
\mathsf{model\_id} = H_{\mathrm{in}}\bigl(\texttt{dom}_{\mathrm{mid}},\;\varsigma_0,\;\mathsf C_W,\;\mathsf C_{\mathrm{raw}},\;\mathsf C_{\mathrm{cfg}}\bigr).
$$

---

### 2.2 Single-Transcript Setup Proof $\pi_{\mathrm{setup}}$ (Zero Correlated-Agreement Premise) `[Def + Paper]`

Published once alongside `model_id`, $\pi_{\mathrm{setup}}$ proves in zero knowledge that:
1. **(i) Joint Reed–Solomon proximity:** Every row of $\mathsf C_W$ and $\mathsf C_{\mathrm{raw}}$ agrees with the RS encoding of its committed message row on one **common** column set of density $\ge 1 - \delta_{\mathrm{ca}} = 0.70$;
2. **(ii) Format legality & maximal exponent:** Every raw entry is a finite legal code of the declared format, $|\mathrm{cd}| < 2^{22}$, and $G_i$ is maximal on every non-zero row;
3. **(iii) Canonical decode consistency:** $\mathbf W^{\mathrm{cd}} = \mathrm{CD}(\mathbf W^{\mathrm{raw}})$ under `SCALE` row exponents $G_i$;
4. **(iv) Exact row norms:** `PN` equals the exact squared row norms of $\mathbf W^{\mathrm{cd}}$;
5. **(v) Exact radius rows:** $\mathbf W_{R_{\max}/2 + t} = |\mathbf W_t^{\mathrm{cd}}| + \kappa_t m_t$ for all tile rows $t$, and zero elsewhere;
6. **(vi) Well-formed stable configuration rows:** Strictly increasing lookup keys, `SCALE`/$\kappa_t$ agreement, valid derived score-exponent windows, and sentinel padding (§3).

**Two-pass joint-column architecture.** Independent per-block column draws would certify only per-block agreement sets whose intersection across blocks could be empty. Instead, $\pi_{\mathrm{setup}}$ runs as **one unified Fiat–Shamir transcript** partitioned into $B_{\mathrm{setup}} = \lceil R_{\max}/7\rceil$ row-block sub-proofs plus one hidden-column sub-proof to bound prover memory:
- **Phase 1 (Message commitment across all blocks):** Each block $b$ commits its 7 message rows $(\mathbf W_b^{\mathrm{cd}}, \mathbf W_b^{\mathrm{raw}})$, maximality witnesses, partial norms, and a random zero-sum link scalar mask $m_b$ ($\sum_x m_x = 0$).
- **Joint column draw:** Only after **all** $B_{\mathrm{setup}}$ Phase-1 roots are fixed does $\mathrm{seed}_1 = H_{\mathrm{out}}(\text{stmt}, \text{all Phase-1 roots})$ draw one shared set $\mathcal Q^{\mathrm{s}}$ of $m_q^{\mathrm{s}} = 206$ column indices.
- **Hidden-column sub-proof & bilinear link:** One column sub-proof opens the $2\times 206$ salted columns of $\mathsf C_W$ and $\mathsf C_{\mathrm{raw}}$ in-circuit from `model_id`; $\mathrm{seed}_2$ draws row/column batching challenges $(r, \beta)$, and the verifier checks $\sum_x T_x = 0$ over the masked link scalars $T_x = m_x + S_x$.
- **Direct soundness without `A-CA` `[Paper; Lean: ProtocolBridge.setup_col_esc_le]`:** Because all message rows are committed prior to $\mathcal Q^{\mathrm{s}}$, if the set $D$ of columns where *any* row disagrees with its committed message encoding has size $|D| \ge 0.3n$, $\mathcal Q^{\mathrm{s}}$ misses $D$ with probability at most $0.7^{206} \le 2^{-106}$. Otherwise $|A| = n - |D| > 0.7n > (1+\vartheta)n/2 = 0.625n$, so every row uniquely decodes to its committed message; a hit on $D$ survives the $(r, \beta)$ bilinear link with probability $\le 2 / |\mathbb E|$.
- **Two-pass AES-256-CTR mask regeneration (`A-PRG`):** Because $\mathcal Q^{\mathrm{s}}$ depends on all Phase-1 roots, the prover evaluates Phase 1 of each block twice using an ephemeral per-block AES-256-CTR key $K_b$ (erased upon completion) rather than spilling gigabytes of Phase-1 masks to disk, making $\pi_{\mathrm{setup}}$ computationally zero-knowledge.

---

## 3. Per-Response Hidden Column Opening (`HCO`) & Configuration Opening

### 3.1 Two-Round Tile Reduction and Hidden Column Check `[Def + Paper]`

Within each response proof's Fiat–Shamir transcript ([Spec 02](02-soundness.md) §4.1):
1. **Two-round shared-point tile reduction:** Given committed block operands/outputs, the verifier draws a public row point $\boldsymbol\xi \in \mathbb E^{\log_2 T_o}$; the prover commits folded rows $w_{g,a}^h(b, i') = \sum_{o'} \mathrm{eq}(\boldsymbol\xi, o')\mathbf W_{t(g,a,b)}^h(o', i')$ ($h \in \{\mathrm{cd}, r\}$) tied to activations via per-GEMM zero-sum checks (multiplying per-slot scale factors $2^{G_{\mathrm{hi}} - G_j}$ when tiles of one tensor have distinct row exponents); the verifier draws column point $\boldsymbol\zeta \in \mathbb E^{\log_2 T_i}$; and the prover commits $J_{\max}$ hidden claims $c_j = \widetilde{\mathbf W}_{t_j}(\boldsymbol\xi, \boldsymbol\zeta)$ with paired radius indices $t_{j + J/2} = t_j + R_{\max}/2$ (escape $\le 2\log_2 T / |\mathbb E|$). Token embeddings and batched LM-head products enter the same reduction.
2. **Weighted LogUp row combination:** Challenge $\gamma_{\mathrm{hco}} \in \mathbb E$ defines the sparse hidden combination $\alpha_i = \sum_{j < J_{\max}} \gamma_{\mathrm{hco}}^j [t_j = i]$, enforced via the partial-fraction identity $\sum_{i < n_r} \frac{\alpha_i}{\lambda - i} = \sum_{j < J_{\max}} \frac{\gamma_{\mathrm{hco}}^j}{\lambda - t_j}$ (false $\boldsymbol\alpha$ escape $\le (n_r + J_{\max}) / |\mathbb E|$). The prover commits $\mathbf u = \boldsymbol\alpha^\top \mathbf W \in \mathbb E^{n_c}$ ($O(\text{touched rows} \cdot n_c)$ work) and proves $\sum_j \gamma_{\mathrm{hco}}^j c_j = \tilde{\mathbf u}(\boldsymbol\xi, \boldsymbol\zeta)$.
3. **In-circuit hidden column check (`m_q = 123`):** Public random column positions $\mathcal Q$ ($|\mathcal Q| = 123$) are drawn; the prover supplies $\mathbf M_{:,j}$, salts $\varsigma_j$, and Merkle paths ($j \in \mathcal Q$) as **hidden outer-circuit witnesses**, verifies them in-circuit against $\mathsf C_W$ opened from `model_id`, and checks $\sum_{j \in \mathcal Q} \beta_{\mathrm{col},j}\mathrm{RS}(\mathbf u)[j] = \sum_{j \in \mathcal Q} \beta_{\mathrm{col},j}\langle \boldsymbol\alpha, \mathbf M_{:,j}\rangle$.

**Lemma 5.1 (HCO Column Soundness) `[Paper; Lean: ProtocolBridge.hco_col_esc_le]`.** Given the common agreement set $|A| \ge 0.7n$ certified by $\pi_{\mathrm{setup}}$, any $\mathbf u \ne \boldsymbol\alpha^\top \mathbf W_*$ agrees with $\boldsymbol\alpha^\top \mathrm{RS}(\mathbf W_*)$ on at most $\vartheta n = 0.25n$ columns inside $A$ plus at most $0.3n$ columns outside $A$, escaping $m_q = 123$ independent column draws with probability at most $(0.25 + 0.30)^{123} = 0.55^{123} \approx 2^{-106.09} \le 2^{-106}$.

---

### 3.2 In-Circuit Configuration Opening (`M^st` + `M^vol`) `[Def + Paper]`

To prevent a dishonest prover from substituting another committed tile of identical shape (e.g., swapping layer $\ell_1$'s weight tile into layer $\ell_2$) while keeping architecture metadata secret and avoiding full weight re-encoding when prompts or keys rotate:
- **Stable tables $M^{\mathrm{st}}$** (tile map $\mathsf{Tt}: (\nu, e, a, b) \mapsto (t, G_t, \kappa_t)$, vector table $\mathsf{Vt}$, layer table $\mathsf{Lt}$ holding layer kind, SWA window, sinks, dimensions, RoPE, $\epsilon$, score-exponent window, and contract tier, plus graph/cost/tokenizer tables) reside in $N_{\mathrm{cr}}$ configuration rows of $\mathbf W$. In each proof, the hidden table columns $T_k$ used by LogUp are bound to configuration row $r$ at the shared HCO point via $c_r = \sum_{k : r_k = r}\sum_{i < n_{t,k}} \mathrm{eq}((\boldsymbol\xi, \boldsymbol\zeta), o_k + i) T_k(i)$ (adding zero extra challenge rounds and $\approx 3\text{–}4\times 10^5$ data cells). Key strict monotonicity verified once in $\pi_{\mathrm{setup}}$ item (vi) guarantees a unique tile index $t_j$ for every sampled slot.
- **Volatile configuration $M^{\mathrm{vol}}$** ($K_{\mathrm{prf}}$, sampling parameters, and hidden prompt/template tokens padded to $P_{\max}$) is committed in $\mathsf C_{\mathrm{cfg}}$ and opened in full per proof via an in-circuit Poseidon2 sponge ($\approx 715$ permutations at $P_{\max} = 2^{14}$), feeding $K_{\mathrm{prf}}$ directly into the sampling PRF and prompt tokens into rendering checks.

**Theorem 6.1 (Unbounded Lifetime Zero Knowledge) `[Paper]`.** Because every static column, path, combination vector $\boldsymbol\alpha$, folded row $\mathbf u$, and configuration table is freshly blinded inside each response proof's witness columns, the multi-proof simulation advantage over $N_\pi$ proofs contains **no opening-budget degradation term** in $N_\pi$.

---

## 4. Measured Component Costs & Frontier Scaling `[Meas + Est]`

### 4.1 Full `Qwen3-0.6B` Measurements over $\mathbb F_{p^3}$ (`RTX 4070 Ti SUPER 16 GB` + `Intel i7-14700F`)

| Component | Committed Cells | Proving Time | Proof / Store Size | CPU Verification | Notes |
|---|---|---|---|---|---|
| **One-time static commitment** ($\mathsf C_W, \mathsf C_{\mathrm{raw}}, \mathsf C_{\mathrm{cfg}} \to \mathsf{model\_id}$, $R_{\max}=2048$ with $\mathbf W^r$ & `CFG0`/`CFG1`) | — | **$94.2\text{ s}$** (`W`: $31.3\text{ s}$, `RAW`: $34.7\text{ s}$) | **$7.62\text{ GB}$** coeff store ($0\text{ B}$ materialized codeword) | — | $574$ active rows ($G_i \in [14, 25]$) + $569$ radius rows; peak host RSS $12.0\text{ GiB}$ |
| **Complete one-time setup proof $\pi_{\mathrm{setup}}$** (items (i)–(vi), 147 blocks + 206-col joint sub-proof) | **$8.67\text{ G}$** (`8,673,610,624`) | **$3{,}623.7\text{ s}$** ($60.4\text{ min}$, two passes) | **$353.0\text{ MB}$** | **$59.9\text{ s}$** (i7-14700F) / **$63.7\text{ s}$** (Apple M1) | Peak VRAM pool $3.54\text{ GiB}$, host RSS $12.2\text{ GiB}$; all 9 tile/exponent tampering modes rejected |
| **Per-response `HCO` + `WSum` inside full 315-draw reply proof** (`16,384` claim slots, `123` hidden cols) | **$12.07\text{ M}$** (`HCO`) + **$34.46\text{ M}$** (`WSum`) | Included in $3{,}877.7\text{ s}$ full 5.25G-cell reply proof | $149.8\text{ MB}$ total reply proof | **$12.02\text{ s}$** (full reply proof) | `int4`-requantized weights and tampered configuration rows rejected $100\%$ |

### 4.2 Analytical Scaling to 70B Dense & 400B Sparse MoE `[Est]`

Choosing $n_c^* \approx \sqrt{9.5\, m_q N_{\max}}$ balances the $\mathbf u$ column against in-circuit column hashing:
- **70B Dense ($N_{\max} = 2^{37}$, $n_c = 2^{23}\text{–}2^{24}$):** $\approx 5\times 10^7$ HCO data cells per proof; prover $\mathbf u$ + tile folding takes $\approx 20\text{–}60\text{ ms}$ on H100-class bandwidth ($\approx 1\%\text{–}2\%$ of a 1,000-token reply).
- **400B Sparse MoE ($N_{\max} = 2^{39}$, $\approx 20\text{B}$ active parameters, $n_c = 2^{24}$):** $\approx 1.1\times 10^8$ HCO data cells per proof; prover $\mathbf u$ touches **only active expert rows** ($\approx 10\text{–}20\text{ ms}$ on H100-class bandwidth).
