# Spec 10: Per-Request KV Matrix Commitment, Hidden-Selection Sumcheck & Length Tiers (`fpn-tracecom-1.0`)

**Specification ID:** `fpn-tracecom-1.0` | **Role:** Eliminating the long-context attention circuit bottleneck via post-reply Reed–Solomon KV matrix commitments, pre-commitment fixed-point lifts, hidden-selection sumcheck reductions, and public power-of-two length tiers

---

## Overview and Motivation

Static weight commitments ([Spec 09](09-model-commitment.md)) reduce per-proof model-weight authentication on `Qwen3-0.6B` to $12.07\text{ M}$ `HCO` cells (with $\mathbf W^r$), and sort-free bucketed sampling ([Spec 06](06-tolerance-and-cheats.md) §3) reduces vocabulary checks by an order of magnitude. On realistic multi-thousand-token contexts, however, a naive per-leaf Merkle trace commitment encounters a severe **long-context attention circuit wall** ([Spec 05](05-computation-model.md) §5.1.1):
- Opening $T_{\mathrm{pad}}$ historical KV sponge leaves inside the circuit for each of the $840$ attention slots in a default $h=3$ lineage budget costs $\approx (1.5\text{–}2.0)\times 10^4$ cells per key, and evaluating per-element `bf16` KV decodings plus componentwise $|v_j - o|$ differences inside the attention gadget costs a measured $\approx 1.27\times 10^5$ cells per key (`Qwen3-0.6B`), totaling $\approx 0.49\text{–}0.51\text{ T}$ cells at $4\text{k}$ context and $\approx 3.9\text{–}4.0\text{ T}$ cells at $32\text{k}$ context.

Two structural observations unlock a compact solution (`TC`):
1. **Zero cross-proof wear-out on per-request commitments (Lemma 2.1 `[Paper]`).** Unlike static model weights, a request's KV history matrix $\mathbf H$ belongs to a single API response and is opened against only one Fiat–Shamir transcript ($T = 1$ in [Spec 09](09-model-commitment.md) Lemma 2.1). Verified in-circuit column openings keep proof size small ($\sim 1\text{ MB}$ rather than $25\text{ MB}$ of clear Ligero columns) with zero cross-request leakage, even when prefix-cache values repeat across requests.
2. **Resolving mixed row/point reads via Hidden-Selection Sumcheck (Proposition 2.2 & Lemma 2.3 `[Paper]`).** Attention consumers read entire key/value prefixes across token rows, whereas `QKV` producers point-read single `(position, layer)` slices at hidden intra-row offsets—breaking the rank-1 shared-point structure used by [Spec 09](09-model-commitment.md). Applying a **Hidden-Selection Sumcheck** reduces any batch of claims $\sum_j \gamma^j \widetilde H(P_j)$ with partially hidden boolean coordinates to a **single evaluation $\widetilde H(r)$ at one public random point $r = (r_{\mathrm{row}}, r_{\mathrm{col}})$**. The subsequent row combination $\mathbf u_H = \mathrm{eq}(r_{\mathrm{row}}, \cdot)^\top \mathbf H$ uses public weights, requiring neither a hidden $\boldsymbol\alpha$ vector nor a weighted LogUp argument.

Combining **(i) post-reply asynchronous RS matrix commitment**, **(ii) pre-commitment fixed-point lifting (`D-5`)**, **(iii) sumcheck-based $PV$ and quadratic radius bounds (`D-9`)**, and **(iv) public power-of-two length tiers $T_{\mathrm{pad}} = 2^{\lceil\log_2 T\rceil}$ (`D-8`)** drops the context-dependent circuit slope by **more than two orders of magnitude**—down to $\approx 80\text{–}130$ cells per `(slot, key)` plus $6.6\text{–}14.8\text{ M}$ opening cells ($\approx 3.3\text{–}3.5\text{ G}$ total cells at $4\text{k}$ and $5.2\text{–}6.6\text{ G}$ at $32\text{k}$ `[Est]`), while removing $100\%$ of Poseidon2 hashing from the synchronous online serving path.

---

## 1. Foundational Lemmas

### 1.1 Single-Opening Non-Exhaustion `[Paper]`

**Lemma 2.1.** Because each response builds its own matrix $\mathbf H$ committed under a fresh 256-bit root salt $\varsigma_R$ and opens it once inside a freshly masked outer proof, $\mathbf H$ undergoes $T = 1$ opening across its lifetime. Even when two requests share identical prefix-cached KV values, re-encoding $\mathbf H$ per request (or reusing internal prover sponge states that never enter transcripts) leaks zero information across proofs under the release-order hybrid of [Spec 03](03-zero-knowledge.md) §4.3.

### 1.2 Hidden-Selection Sumcheck `[Paper]`

**Lemma 2.3.** Let $\widetilde H$ be the $n$-variate multilinear extension of $\mathbf H \in \mathbb F_p^{N_H}$ ($n = \log_2 N_H$), and let $\{c_j = \widetilde H(P_j)\}_{j < J_H}$ be evaluation claims where each point $P_j \in \mathbb E^n$ mixes hidden boolean witness coordinates (such as hidden layer $\ell_j$ or token offset $\tau_j$) and public challenge coordinates. For Fiat–Shamir batching challenge $\gamma_{\mathrm{hs}} \in \mathbb E$, define $A(x) = \sum_{j < J_H} \gamma_{\mathrm{hs}}^j \mathrm{eq}(P_j, x)$. Then:

$$
\sum_{j < J_H} \gamma_{\mathrm{hs}}^j c_j = \sum_{x \in \{0,1\}^n} A(x)\,H(x).
$$

Running an $n$-round degree-2 Libra-masked sumcheck ending at public random point $r = (r_{\mathrm{row}}, r_{\mathrm{col}}) \in \mathbb E^n$ reduces the batch to a single identity $A(r)\cdot \widetilde H(r) = v_{\mathrm{fin}}$, where $A(r) = \sum_{j < J_H} \gamma_{\mathrm{hs}}^j \prod_{b < n} \mathrm{eq}_1(P_j[b], r_b)$ is evaluated in-circuit from the committed hidden coordinates ($\approx 2n$ cells per claim) and $\widetilde H(r) = \tilde{\mathbf u}_H(r_{\mathrm{col}})$ for $\mathbf u_H = \mathrm{eq}(r_{\mathrm{row}}, \cdot)^\top \mathbf H$. Soundness escape is at most $(J_H - 1)/|\mathbb E| + 2n / |\mathbb E|$.

---

## 2. The `TC` Construction

### 2.1 Object Partitioning, Matrix Layout, and Fixed-Point Lift `[Def]`

1. **History vs. point objects:** Carried states read across multiple positions—KV cache entries $k, v(\tau, \ell)$, `MLA` compressed latents $(c^{\mathrm{KV}}, k^R)(\tau, \ell)$, `LREC` chunk-boundary states $S(\ell, m)$, `KVCast` outputs, and multimodal encoder patch KV—are placed in the per-request **history matrix** $\mathbf H$. Single-consumer intra-token boundary tensors ($h, x^{\mathrm{attn}}, q, o, h^{\mathrm{mid}}, x^{\mathrm{mlp}}, a, x^{\mathrm{fn}}$) remain in the salted sponge Merkle tree $\mathrm{rt}_{\mathrm{obj}}$ ([Spec 01](01-relation.md) §4.2).
2. **Public length tier & chunked rows (`D-3`, `D-8`):** Standard API usage counters (`input_tokens + output_tokens`, including `reasoning_tokens`) already belong to $\mathcal L_{\mathrm{API}}$, determining the public power-of-two tier $T_{\mathrm{pad}} = 2^{\lceil\log_2 T\rceil}$ with zero extra leakage. For profile history width bound $V_{\mathrm{hist}} = 2^v \ge L_{\max} P_{\max} G_{\max} d_{\max}$ and chunk size $T_c$:

   $$
   n_c = T_c\,V_{\mathrm{hist}}, \qquad n_r = T_{\mathrm{pad}} / T_c, \qquad N_H = n_r n_c = T_{\mathrm{pad}}\,V_{\mathrm{hist}},
   $$

   where row $c$ packs token chunk $[c T_c, (c+1)T_c)$ and intra-row columns index $(\tau', \ell, \mathsf{port}, g, i)$ in binary.
3. **Pre-commitment fixed-point lift (`D-5`):** Outside the circuit prior to encoding, each history value is lifted to a signed integer $\mathrm{Lift}_{\nu,\mathsf{port}}(v) = \mathrm{round}_0(v\cdot 2^{F_{\nu,\mathsf{port}}}) \in (-2^{b_H}, 2^{b_H})$ ($b_H \le 27$ so $d_h \cdot 2^{2b_H} < p/2$ prevents modular wrap-around, `[Lean: BoundedLift.signedLift_unique_of_bounded]`). Values $\ge 2^{8 - F}$ lift losslessly; smaller values incur additive lift error $\le \eta_{\mathrm{lift}} = 2^{-F}$, which is added directly to consumer input box radii $\mathbf r_x$ ([Spec 06](06-tolerance-and-cheats.md) §2.2).
4. **Reed–Solomon encoding & trace root:** Each of the $n_r$ rows is Reed–Solomon encoded at rate $\vartheta_H = 1/2$ to length $n = 2n_c$, and columns are hashed via Poseidon2 sponge $H_{\mathrm{in}}(\texttt{dom}_{\mathrm{hcol}}, j, \mathbf M_{:,j})$ into Merkle root $\mathrm{rt}_H$. The salted per-request trace commitment is:

   $$
   R = H_{\mathrm{in}}\bigl(\texttt{dom}_{\mathrm{root}},\;\varsigma_R,\;\mathrm{rt}_{\mathrm{obj}},\;\mathrm{rt}_H,\;T_{\mathrm{pad}}\bigr).
   $$

---

### 2.2 Asynchronous Post-Reply Commitment Schedule (`D-1`) `[Def]`

Because knowledge soundness requires only that $R$ be fixed before sampling challenge $\tau_{\mathrm{samp}}$ ([Spec 02](02-soundness.md) §2.1), online serving performs **zero Poseidon2 hashing**:
- **Online serving:** Read-only hooks copy accepted KV pages and boundary tensors asynchronously to host memory over a profile-reserved PCIe bandwidth budget ($\approx 115\text{ KB/tok}$ for `Qwen3-0.6B` KV, $<4\%$ of PCIe 4.0 x16 at 7,574 tok/s).
- **Background proving:** After the response stream finishes, the prover worker lifts $\mathbf H$, runs row NTTs and column Poseidon2 sponges ($\approx 0.22\text{ s}$ at $4\text{k}$, $1.8\text{ s}$ at $32\text{k}$ on `RTX 4070 Ti SUPER`), forms $R$, and begins the Fiat–Shamir transcript, releasing $(R, \pi)$ at the fixed tier deadline $\Delta_{\mathrm{rel}}(T_{\mathrm{pad}})$.

---

### 2.3 Sumcheck Attention Reads, Quadratic Radii (`D-9`), and Column Opening `[Def + Paper]`

For all sampled and lineage attention slots $j$ (sharing batch challenges $\tau_{\mathrm{att}}$):
1. **Attention score & output sumchecks:** Score tensor $z_j(\tau, g, \varrho) = \sum_i q_j(g, \varrho, i) K_{\ell_j}(\tau, g, i)$ reduces over head dimension $i$ to a hidden claim $\widetilde H(P_j^K)$ (only $\ell_j$ hidden), and output $o_j(g, \varrho, i) = \sum_\tau p_j(\tau, g, \varrho) V_{\ell_j}(\tau, g, i)$ reduces over key position $\tau$ to $\widetilde H(P_j^V)$—removing all per-dimension KV decoding and $PV$ inner products from the circuit.
2. **Sumcheck-expressible certified radius bounds (`D-9`):** The three componentwise absolute-value sums in `AttnFA` ([Spec 06](06-tolerance-and-cheats.md) §2.2 Row 7) are upper-bounded cleanly by quadratic sumchecks over $\mathbf H$ (preserving $100\%$ honest completeness by Jensen and Cauchy–Schwarz):
   - $\sum_\tau p_\tau |v_{\tau,i} - o_i| \le \sqrt{(1 + \epsilon_p)\sum_\tau p_\tau(v_{\tau,i} - o_i)^2}$;
   - $\sum_\tau |v_{\tau,i}| \le \sqrt{n\sum_\tau v_{\tau,i}^2}$;
   - $|q|^\top |k_\tau| \le \|q\|_2 \|k_\tau\|_2$.
   Only per-key scalar scores $z$, exponential lookups, probabilities $p$, and window masks remain in-circuit ($\approx 80\text{–}130$ data cells per `(slot, key)` on `Qwen3-0.6B`).
3. **Producer point-reads & architectural generality:** `QKV` producer slots bind their emitted KV slice at shared `(group, dim)` challenges via a single claim in $\mathsf{CL}_H$ (with hidden `(chunk, offset, layer)` coordinates). `MLA` latents $(c^{\mathrm{KV}}, k^R)$ (`576` values/tok/layer on DeepSeek-V3), `SWA` + attention sinks, multimodal patch KV, and `LREC` chunk boundary states all share the identical matrix interface.
4. **Proximity combination & `268`-column opening (`D-6`):** Following Lemma 2.3's reduction to $\mathbf u_H = \mathrm{eq}(r_{\mathrm{row}}, \cdot)^\top \mathbf H$, challenge $\tau_{\mathrm{prox}}$ defines proximity combination $\mathbf u_{\mathrm{px}} = \mathbf s^\top \mathbf H$ ($\mathbf s = (s^c)_{c < n_r}$), and $\tau_{\mathrm{col}}^H$ draws $m_q^H = 268$ public column indices $\mathcal Q_H$ verified in-circuit against $\mathrm{rt}_H$:

   $$
   \sum_{j \in \mathcal Q_H} \beta_j\bigl(\mathrm{RS}(\mathbf u_H)[j] + \mu\,\mathrm{RS}(\mathbf u_{\mathrm{px}})[j]\bigr) = \sum_{j \in \mathcal Q_H} \beta_j \sum_{c < n_r}\bigl(\mathrm{eq}(r_{\mathrm{row}}, c) + \mu s^c\bigr)\mathbf M_{c,j}.
   $$

---

## 3. Soundness, Zero Knowledge, and Cost Profile

- **Soundness (`A-AROM` + `A-CA`) `[Paper; Lean: ProtocolBridge.tc_col_esc_le]`:** Straight-line extraction recovers columns of $\mathrm{rt}_H$ prior to $\tau_{\mathrm{samp}}$ and uniquely decodes $\mathbf H_*$ whenever $\mathbf M$ lies within $\delta_H = 0.24$ of the interleaved RS code. Under correlated agreement (`A-CA`, error $< 2^{-157}$), a matrix $\delta_H$-far from the code passes $m_q^H = 268$ queries with probability $< (1 - 0.24)^{268} = 0.76^{268} \approx 2^{-106.1} \le 2^{-106}$, and a false claim on a close matrix passes with probability $\le (\vartheta_H + \delta_H)^{268} = 0.74^{268} \approx 2^{-116.4} \le 2^{-106}$ (`tc_col_esc_le`).
- **Zero knowledge `[Paper]`:** Every column of $\mathsf{CL}_H$, $\mathbf u_H$, $\mathbf u_{\mathrm{px}}$, and $\mathbf M_{:, \mathcal Q_H}$ carries fresh `T1` blinding, sumchecks carry Libra `T2` masks, and all matrix dimensions depend solely on public $(T_{\mathrm{pad}}, \mathrm{Prof})$.

### Estimated Circuit & Prover Scaling (`Qwen3-0.6B`, Default Budget `840` Attention Slots) `[Est]`

| Context Tier $T_{\mathrm{pad}}$ | Geometry $(T_c, n_c, n_r)$ | `TC` Opening Cells | In-Circuit Attention Arithmetic | Total Proof Data Cells (`TC`) | Naive Sponge-Leaf Baseline |
|---|---|---|---|---|---|
| `1,024` | $(8, 2^{19}, 128)$ | $3.8\text{ M}$ | $0.07\text{–}0.11\text{ G}$ | **$\approx 3.1\text{ G}$** | $\approx 120\text{–}130\text{ G}$ |
| `4,096` | $(16, 2^{20}, 256)$ | $6.6\text{ M}$ | $0.28\text{–}0.45\text{ G}$ | **$\approx 3.3\text{–}3.5\text{ G}$** | $\approx 490\text{–}510\text{ G}$ |
| `32,768` | $(32, 2^{21}, 1024)$ | $14.8\text{ M}$ | $2.2\text{–}3.6\text{ G}$ | **$\approx 5.2\text{–}6.6\text{ G}$** | $\approx 3.9\text{–}4.0\text{ T}$ |
| `131,072` | $(64, 2^{22}, 2048)$ | $28.4\text{ M}$ | $8.8\text{–}14.3\text{ G}$ | **$\approx 12\text{–}17\text{ G}$** | $> 15\text{ T}$ |

*(Measured Stage 1 realization `[Meas: Qwen3-0.6B, RTX 4070 Ti SUPER 16 GB]`: Under Stage 1's $h=1$ key-producer lineage (`210` attention slots + `210` `N1 + KV + EMB` key-producer checks) in the full 315-draw `Qwen3-0.6B` response proof at $T_{\mathrm{pad}} = 256$, encoding and committing $\mathbf H$ takes **$3.78\text{ s}$** on GPU, the two-phase `BatchedAttn` sumcheck across all `210` compute draws completes in **$5.8\text{ s}$** by folding each layer's KV cache once, and the `HSel` sumcheck over all `16,384` history-matrix evaluation claims completes in **$9.5\text{ s}$**, fitting the entire `5.25 G`-cell response proof within `21.0 GiB` host RAM and `10.48 GiB` VRAM pool; verified independently at $T_{\mathrm{pad}} \in \{1024, 4096\}$ as well. Stage 2 optimizations include prefix sponge-state caching `P1'`, static shared-prefix commitments `P2`, and native-hash column leaves.)*
