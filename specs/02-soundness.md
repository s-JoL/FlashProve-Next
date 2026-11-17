# Spec 02: Lifetime Knowledge Soundness & Sampling Budget (`fpn-soundness-1.0`)

**Specification ID:** `fpn-soundness-1.0` | **Role:** Straight-line witness extraction, round-by-round doomed-state soundness analysis, and $\mathbb F_{p^3}$ spot-check parameter budget

---

## Overview and Architectural Motivation

In verifiable closed-weight LLM inference, the adversary for soundness is the API provider itself. A malicious provider holds every secret in the system—model weights, hidden system prompts, sampling PRF keys, and tenant traffic—and can evaluate Fiat–Shamir hash challenges offline across trillions of candidate transcripts, silently discarding unfavourable draws before returning a single response to a client. Soundness must therefore hold adaptively across an entire service lifetime of $Q = 2^{64}$ challenge queries with zero reliance on economic rationality, hardware attestation, or honest-majority assumptions.

`fpn-soundness-1.0` establishes lifetime knowledge soundness for the cost-stratified relation $\mathcal R_{\Pi,\sigma,\rho}$ of [Spec 01](01-relation.md) through five core components:

1. **Zero-rewinding straight-line extraction.** Because trajectory leaves absorb complete serialized boundary tensors under the Poseidon2 sponge $H_{\rm in}$, the knowledge extractor $\mathcal E_{\rm SL}$ reconstructs the extended witness $w_R$ directly from the adversary's lifetime hash query log $\mathcal T$ with zero rewinding and zero additional oracle queries (§2). Unqueried local leaves map to $\bot$ and count at full weight toward bad-block defect fractions, avoiding the exponential $Q/\varepsilon$ rewinding blowup and underconstrained preimage spaces that afflict compressive algebraic digests.
2. **Round-by-round doomed-state soundness.** Multi-round Fiat–Shamir protocols cannot be analyzed by conditioning on a fixed spot-check prefix and taking a union bound over prefixes: an adversary could spend $1$ query to find a lucky prefix and $Q - 1$ queries grinding downstream sumcheck or PCS folding challenges. We define an explicit **doomed-state predicate** $\mathsf D_\tau(s)$ across every protocol round—stratified sampling, Hidden Column Opening (HCO), multi-column LogUp, Libra masking, sumcheck, claim batching, and Basefold folding/queries—charging at most $\varepsilon_\tau$ per distinct first-time challenge query (`[Lean: RoundByRound.rbr_lifetime_soundness]`, `[Lean: ProtocolBridge.fpn_protocol_lifetime_soundness]`).
3. **Tight multi-column LogUp bound and $\mathbb F_{p^3}$ challenge field.** Tuple-level lookup arguments with $w$ columns, $n_v$ lookups, and $n_t$ distinct table entries exhibit an exact false-acceptance bound of $(n_t(w-1) + n_v + n_t)/|\mathbb E|$ (`[Lean: RoundByRound.logup_multicol_prob]`), proved tight via root-of-unity collision families (§6.3). Because realistic LLM slot tables and codewords reach numerators up to $2^{52}$, Goldilocks quadratic extension $\mathbb F_{p^2}$ ($|\mathbb F_{p^2}| < 2^{128}$) falls short of the per-round target $2^{-106}$ (`[Lean: RoundByRound.field_p2_iff]`), motivating the cubic extension $\mathbb F_{p^3}$ ($|\mathbb F_{p^3}| > 2^{191.99}$, covering numerators up to $2^{85}$, `[Lean: RoundByRound.field_p3_bits]`).
4. **Independent per-response transcripts and stratified weighted sampling.** Each response derives its Fiat–Shamir transcript independently with no cross-request batch root, while sampling cost units proportionally across $|\Pi|$ public strata with bounded $192$-bit integer mapping bias $a_{\rm map} \le 2^{-107}$ (`[Lean: RoundByRound.multi_pool_map_le_max]`, `[Lean: RoundByRound.amap_default]`).
5. **Closed-form sample-count budget.** Under round-by-round composition, per-round algebraic and mapping errors combine via $\max(a_{\rm map}, \bar\varepsilon_{\rm alg})$ rather than summing across rounds (`[Lean: RoundByRound.total_le_rbr]`), yielding exact machine-checked sample counts $k(s, Q, \varepsilon_{\rm sp})$ across all $16$ cells of the $(\sigma, \rho) \times (Q, \varepsilon)$ parameter table (§8).

---

## 1. Security Model and Knowledge Soundness Target

`[Def]` **Adversary capabilities.** A malicious provider $\mathcal A$ is a probabilistic polynomial-time algorithm holding all model weights, metadata, sampling PRF keys, AEAD wrapping keys, and tenant inputs. $\mathcal A$ adaptively chooses model commitments (including adversarially crafted configurations), client responses, commitments, salts, and submission orderings; $\mathcal A$ may silently abort and retry proof generation arbitrarily many times within its query budget and collude with arbitrary corrupted tenants. Client verifiers and public `profile` parameters are fixed and untrusted-provider-safe.

`[Def]` **Unified cryptographic hash model ([Spec 01](01-relation.md) §1).**
- **External oracle $H_{\rm out}$** (used strictly outside the circuit for PCS Merkle trees and Fiat–Shamir challenges) is modeled as a **classical Random Oracle** (`A-ROM`) instantiated with an independent BLAKE3 or SHA-3 instance sharing no internal permutation with Poseidon2.
- **Internal sponge permutation $P$** (width-12 Goldilocks Poseidon2 with capacity $4$ field elements) is modeled across all specifications as a single **ideal invertible permutation** providing forward and inverse oracles $(P, P^{-1})$. Both the unkeyed commitment sponge $H_{\rm in}$ (`model_id`, weight column trees $\mathsf C_W, \mathsf C_{\rm raw}$, configuration root $\mathsf C_{\rm cfg}$, trajectory root $R$, state anchors $\mathsf{St}_i$) and the keyed sampling sponge $\mathsf{PRF}_{K_{\rm prf}}$ ([Spec 03](03-zero-knowledge.md) §5.4) are domain- and IV-separated constructions over $P$:
  - *Zero knowledge ([Spec 03](03-zero-knowledge.md)):* Hiding of salted sponge outputs follows from sponge indifferentiability in the ideal permutation model (Bertoni–Daemen–Peeters–Van Assche, EUROCRYPT 2008), and PRF security follows from keyed-sponge bounds. The zero-knowledge simulator programs only $H_{\rm out}$ and never programs or observes $P$.
  - *Binding (§2.2, [Spec 05](05-computation-model.md) Lemma 1.3):* Sponge collisions and post-use preimage discovery are bounded directly in terms of $P^{\pm 1}$ query counts.
  - *Straight-line extractability (§2):* Because $H_{\rm in}$ is simultaneously evaluated inside algebraic circuit constraints, bridging ideal-permutation query observability to concrete Poseidon2 constraints requires one explicit heuristic assumption:

`[Assume]` **Premise `A-AROM` (Arithmetized Random-Permutation Consistency):** (a) The in-circuit Poseidon2 gadget is an exact arithmetization of $P$ (part of `P-compile`); and (b) in any accepted proof, every input–output pair $(x, P(x))$ appearing in the extracted witness opening paths (trajectory leaves and Merkle paths, HCO weight column paths, `model_id` and $\mathsf C_{\rm cfg}$ openings, and state anchors) equals a query–answer pair recorded in $\mathcal A$'s $P^{\pm 1}$ query log prior to the corresponding Fiat–Shamir challenge query, except with negligible probability $\Delta_{\rm arom}(Q_{\rm tot})$.

This assumption follows the Arithmetized Random Oracle Model (AROM) framework formalized by Chen, Chiesa, Gur, O'Connor, and Spooner (*Proof-Carrying Data from Arithmetized Random Oracles*, EUROCRYPT 2023, ePrint 2023/587), used here purely to state `A-AROM` cleanly without invoking any PCD theorem. No proof step simultaneously treats $P$ as both black-box and white-box: zero knowledge uses only ideal-permutation hiding without `A-AROM`, whereas knowledge extraction invokes `A-AROM` on the fixed permutation (§2.4 also describes a pure-ROM trajectory alternative `R3`). Quantum random-oracle (`QROM`) adversaries lie outside scope (§9).

`[Def]` **Lifetime query budgets.**
- $Q$ counts all distinct first-time queries to **Fiat–Shamir challenge domains** of $H_{\rm out}$ across the entire service lifetime—including salt grinding, `model_id` rotations, discarded proof attempts, offline precomputations, unpublished candidates, and challenges evaluated by the verifier on submitted proofs (eliminating any "free final guess").
- $Q_{\rm tot} \ge Q$ counts all distinct first-time queries across $P^{\pm 1}$ and all domains of $H_{\rm out}$, including honest Poseidon2 leaf hashing across served tokens. Separating $Q_{\rm tot}$ from $Q$ is essential: an honest high-throughput provider may evaluate $\sim 2^{60}$ sponge permutations over its lifetime without consuming challenge-grinding budget, and $Q_{\rm tot}$ enters solely into the $2^{-86}$ collision term $\Delta_{\rm SL}$ (§2.2).
- Default lifetime targets are $Q = 2^{64}$, $Q_{\rm tot} \le 2^{80}$, and total lifetime failure probability $\varepsilon = 2^{-40}$.

`[Def]` **Straight-line knowledge soundness with partial witnesses.** Plain existential soundness is vacuous for compressed commitments because short Merkle roots admit exponentially many preimages (§2.3). Knowledge soundness requires extracting [Spec 01](01-relation.md)'s extended witness $w_R$ directly from $\mathcal A$'s query log, assigning $\bot$ to any local trajectory leaf lacking a queried preimage and charging its enclosing blocks via [Spec 01](01-relation.md) §4.3.

---

## 2. Straight-Line Witness Extractor

### 2.1 Definition of $\mathcal E_{\rm SL}$

`[Def]` Given the chronologically ordered table $\mathcal T$ of all lifetime $P^{\pm 1}$ and $H_{\rm out}$ queries/answers (observed passively without rewinding or programming) and an accepted pair $(X, \pi)$, the deterministic straight-line extractor $\mathcal E_{\rm SL}(\mathcal T, X, \pi)$ outputs an extended witness $w_R$ as follows:

1. **Model configuration and weights (extracted once per `model_id`).** Invert `model_id` in $\mathcal T$ to recover $(\varsigma_0, \mathsf C_W, \mathsf C_{\rm raw}, \mathsf C_{\rm cfg})$; invert $\mathsf C_{\rm cfg}$ to recover volatile metadata $M^{\rm vol}$; and traverse the $H_{\rm in}$ Merkle trees under $\mathsf C_W$ and $\mathsf C_{\rm raw}$ in $\mathcal T$ to recover each column $(\varsigma_j, \mathbf M_{:,j})$, marking unqueried columns as erasures. Because `model_id`'s accepted setup proof $\pi_{\rm setup}$ ([Spec 09](09-model-commitment.md) §2.2) certifies that all rows of both matrices agree with Reed–Solomon encodings of committed message rows on a common column set of density $\ge 1 - \delta_{\rm ca} = 0.7 > (1 + \vartheta)/2$ (with setup escape probability accounted in §6.2), and because discovering a preimage for an erased column after `model_id` publication triggers event `B2` (§2.2), unique Reed–Solomon decoding uniquely determines $W_* = \mathrm{Dec}(\mathbf M)$—including all stable metadata tables stored in configuration rows ([Spec 09](09-model-commitment.md) §3.2)—prior to any response challenge. If `model_id` or $\mathsf C_{\rm cfg}$ lacks a preimage or unique decoding fails, the global configuration extracts to $\bot$ and $E(X, w_R) = \mathrm{false}$.
2. **Local trajectory objects.** Invert $R$ in $\mathcal T$ to recover $(\varsigma_R, \text{tree root})$ and traverse the $D_{\rm trace}$-depth $H_{\rm in}$ tree to recover $(\mathsf{salt}_a, v_a)$ at each address $a \in \mathcal A_{\rm buf}$, outputting $\bot$ wherever an internal node or leaf preimage is absent from $\mathcal T$.
3. **Global witness columns.** Invert the $H_{\rm out}$ Merkle tree of $\mathsf{cm}_{\rm glob}$ in $\mathcal T$ (marking unqueried leaves as erasures) and apply unique Reed–Solomon decoding within radius $(1 - \vartheta'_{\rm bf})/2$ to recover the global polynomials; if the erasure/error pattern lies outside the unique-decoding radius, set the global column to $\bot$ ($E(X, w_R) = \mathrm{false}$).
4. **Multi-turn state anchors.** Invert $\mathsf{St}_{\rm prev}$ and $\mathsf{St}_{\rm new}$ in $\mathcal T$ to recover their openings (or $\bot$, falsifying $E$).

`[Def]` **Prefix preservation as a theorem.** Whenever collision events `B1`–`B3` (§2.2) do not occur, running $\mathcal E_{\rm SL}$ on the full end-of-lifetime log $\mathcal T$ produces the exact same extended witness $w_R$ for $(X, \pi)$ as running $\mathcal E_{\rm SL}$ on the prefix log $\mathcal T_{\prec \tau_{\rm samp}}$ recorded at the instant the sampling challenge $\tau_{\rm samp}$ for $X$ is first queried. Thus all stratum defect ratios $d^{(j)}_C(X, w_R)$ and $d_G(X, w_R)$ are fixed before $\tau_{\rm samp}$ is sampled.

### 2.2 Extractor Complexity and Collision Bound

`[Paper]` **Complexity.** Indexing $\mathcal T$ takes $O(Q_{\rm tot} \log Q_{\rm tot})$ time; extracting one proof performs $O(|w_{\max}| \cdot D)$ table lookups plus polynomial-time Reed–Solomon unique decoding. **Additional oracle queries: $0$**, with zero rewinding, zero oracle programming, and runtime independent of $\mathcal A$'s acceptance probability.

`[Paper]` **Straight-line collision events.** Count $Q_{\rm tot}$ directly in primitive $P^{\pm 1}$ permutation calls and $H_{\rm out}$ queries rather than multi-block sponge calls. Both digest outputs and sponge capacity states occupy $4$ Goldilocks elements of domain size $p^4$, where $2^{255} < p^4 < 2^{256}$ (`[Lean: ProtocolBridge.goldilocks_p4_bounds]`, specifically $p^4 = 2^{256}(1 - 2^{-32} + 2^{-64})^4 \approx 2^{256}(1 - 2^{-30})$). Let $\alpha$ upper-bound the number of digest fields referenced inside a single call's input ($\alpha < 2^9$):
- **(B1) Digest collision:** Two distinct queries produce identical $4$-element digest outputs ($\le Q_{\rm tot}^2 / (2 p^4)$).
- **(B2) Preimage-after-use:** A fresh query answer equals a digest value that already appeared inside an earlier query's input without being a prior query output ($\le \alpha Q_{\rm tot}^2 / p^4$).
- **(B3) Sponge capacity collision:** A fresh $P$ or $P^{-1}$ answer hits a $4$-element capacity state that already appeared in an earlier call or fixed IV ($\le Q_{\rm tot}^2 / p^4$, accounting for inverse $P^{-1}$ queries).

Summing `(B1)`–`(B3)` under lazy sampling yields

$$
\Delta_{\rm SL}(Q_{\rm tot}) \le \frac{Q_{\rm tot}^2}{2 p^4} + \frac{\alpha Q_{\rm tot}^2}{p^4} + \frac{Q_{\rm tot}^2}{p^4} \le \frac{(\alpha + 2)\,Q_{\rm tot}^2}{p^4}.
$$

When none of `(B1)`–`(B3)` occurs, every sponge digest traces uniquely backward along its absorption chain in $\mathcal T$, every valid Merkle opening in an accepted proof matches $\mathcal T$, and every preimage under $R$, $\mathsf{cm}_{\rm glob}$, and `model_id` was queried strictly before $\tau_{\rm samp}$. For default parameters $Q_{\rm tot} \le 2^{80}$ and $\alpha < 2^9$,

$$
\Delta_{\rm SL}(Q_{\rm tot}) \le \frac{(2^9 + 2)\cdot 2^{160}}{p^4} < 2^{-86.99} \le 2^{-86}
$$

(`[Lean: ProtocolBridge.delta_SL_p4_default]`), consuming a negligible fraction of the auxiliary budget $\bar\Delta \le 2^{-42}$.

### 2.3 Why Compressive Algebraic Leaves Fail Knowledge Soundness

`[Paper]` Replacing full-value sponge leaves $\ell_a$ with compressive linear algebraic digests such as Byte-SIS $D_b = A v_b + B r_b \bmod q$ breaks lifetime knowledge soundness for three structural reasons:
1. *Vacuous existential soundness:* Under compression, every digest $D_b$ admits exponentially many valid short preimages $(v_b, r_b)$, making "there exists a consistent trajectory" trivially true even when the provider never computed one.
2. *Unobservability to straight-line extraction:* Computing $D_b = A v_b + B r_b \bmod q$ makes zero oracle calls, so $\mathcal E_{\rm SL}$ cannot observe $(v_b, r_b)$ for any unsampled block.
3. *Infeasibility of rewinding at $Q = 2^{64}$:* Forking/rewinding extractors require $\Omega(Q/\varepsilon)$ executions—utterly intractable at $Q = 2^{64}$—and divide the SIS collision reduction advantage by $Q$, wiping out standard lattice security margins.

Full-value sponge absorption under $H_{\rm in}$ resolves all three obstacles simultaneously.

### 2.4 Commitment Instantiations

`[Def]` Three architectural instantiations clarify the role of `A-AROM`:
- **Instantiation `R2` (Primary architecture, uses `A-AROM`):** Trajectory leaves/trees, weight column trees ($\mathsf C_W, \mathsf C_{\rm raw}$ checked in-circuit via HCO), $\mathsf C_{\rm cfg}$, and $\mathsf{St}_i$ use Poseidon2 $H_{\rm in}$ opened inside the zero-knowledge circuit, while PCS Merkle trees and Fiat–Shamir challenges use $H_{\rm out}$. `R2` preserves sublinear per-response prover complexity without leaking sampled block addresses, KV access patterns, or weight column indices.
- **Instantiation `R3` (Pure-ROM trajectory alternative):** All trajectory tensors are committed inside a single global blinded ZK-PCS oracle ($H_{\rm out}$ Merkle tree) and fetched by sampled blocks via private LogUp lookups. `R3` eliminates `A-AROM` on the per-request trajectory ($\Delta_{\rm arom}^{\rm trace} = 0$) at the cost of prover work linear in total trajectory size. Static model commitments in [Spec 09](09-model-commitment.md) have no pure-ROM counterpart: opening static weight codewords externally under $H_{\rm out}$ leaks codeword symbols across proofs and exhausts zero knowledge after finitely many responses ([Spec 09](09-model-commitment.md) Lemma 2.1), so HCO always relies on in-circuit $H_{\rm in}$ (`A-AROM`).
- **Infeasible `R1` (External $H_{\rm out}$ trajectory paths):** Verifying trajectory Merkle paths outside the circuit reveals sampled slot addresses and attention KV read indices in the public transcript, violating [Spec 03](03-zero-knowledge.md).

---

## 3. Adaptive Lifetime Knowledge Soundness Experiment

`[Def]` The multi-instance adaptive security game $\mathsf{LifeKS}_{\mathcal A}(Q, Q_{\rm tot})$ proceeds as follows:
1. Initialize versioned public parameters, independent oracles $(H_{\rm out}, P^{\pm 1})$, and stateful client nonce/anchor logs; grant $\mathcal A$ full malicious-provider capabilities.
2. $\mathcal A$ adaptively queries the oracles, constructs candidate responses/proofs, discards failed attempts, and submits pairs $(X, \pi)$ to the verifier, making at most $Q$ distinct challenge-domain $H_{\rm out}$ queries and at most $Q_{\rm tot}$ total distinct oracle queries.
3. For each distinct `model_id`, the verifier checks $\pi_{\rm setup}$ once (§4.1); for each submitted response, the verifier checks canonical encodings, version tags, client nonce/anchor consistency, and the proof $\pi$ (counting any new verifier-evaluated challenge queries toward $Q$).
4. For every accepted response $(X, \pi)$, compute $w_R = \mathcal E_{\rm SL}(\mathcal T, X, \pi)$.
5. $\mathcal A$ wins if **any** accepted pair across the service lifetime satisfies the bad-acceptance event (`[Lean: ProtocolBridge.ExtractedBad]`):

$$
\mathsf{BadAccept}(X, w_R) \;:\iff\; \neg E(X, w_R) \;\lor\; \neg \mathrm{Chain}(X, w_R) \;\lor\; \max_{1 \le j \le |\Pi|} d^{(j)}_C(X, w_R) \ge \sigma \;\lor\; d_G(X, w_R) \ge \rho.
$$

`[Def]` Here $d^{(j)}_C$ is the lineage-weighted bad-atom fraction of [Spec 01](01-relation.md) §4.5 and $d_G$ is the generation-step violation fraction of [Spec 01](01-relation.md) §5.4. Because $\mathcal E_{\rm SL}$ is total over extended witnesses—mapping unqueried local leaves into $d^{(j)}_C$ and unqueried global objects into $\neg E$—no separate "$w_R = \bot$" clause is needed. Multi-turn conversations are covered by external finite induction across turns ([Spec 01](01-relation.md) §6.4) within the **same** lifetime budget $Q$, without granting an extra $\varepsilon$ per turn.

---

## 4. Fiat–Shamir Transcript Ordering and Stratified Sampling

### 4.1 Per-Response and Model-Setup Transcripts

`[Def]` **Per-response transcript.** Every response generates an independent Fiat–Shamir transcript with zero cross-request batch challenges (background prover daemons may share read-only weight buffers in memory while blinding and hashing each response independently, [Spec 03](03-zero-knowledge.md) §2.3). Rounds execute in strict sequence:

1. **Initial commitment $m_0 = (X_0, R, \mathsf{cm}_{\rm glob})$:** $\mathsf{cm}_{\rm glob}$ commits all global witness columns—`model_id` opening $(\varsigma_0, \mathsf C_W, \mathsf C_{\rm raw}, \mathsf C_{\rm cfg})$, metadata $M_{\rm meta}$ (volatile $\mathsf{Enc}(M^{\rm vol})$, $K_{\rm prf}$, and table-side columns of stable tables, [Spec 09](09-model-commitment.md) §3.2), token sequence $u$, slot table $\mathsf{Sl}$ (validity, node IDs, MoE routes, costs, prefix sums, stratum tags, billing-unit IDs), automaton columns $q^{\rm fsm}, q^{\rm stop}$, generation-step table, and state-anchor openings.
2. *(List-decoding regime $L > 1$ only)* **Out-of-domain challenge $\tau_{\rm ood}$:** Prover replies with evaluations of global polynomials at $\tau_{\rm ood}$ (§6.4).
3. **Spot-check challenge $\tau_{\rm samp}$:** Derives stratified compute and generation sample seeds (§4.2).
4. **Local witness commitment $m_1 = \mathsf{cm}_{\rm loc}$:** Commits local witness columns for all blocks in $\mathrm{Lin}_h(a)$ of every sampled atom $a$ and all sampled generation steps $t$ (opened boundary tensors, internal block intermediates, and error radii), followed by $\tau'_{\rm ood}$ if $L > 1$.
5. **Hidden Column Opening (HCO) rounds ([Spec 09](09-model-commitment.md) §3.1):**
   - *5a. Two-stage tile reduction:* Row challenge $\tau_\xi$ fixes shared row evaluation point $\boldsymbol\xi$, prover commits folded rows $w^h_{g,a}$ ($h \in \{{\rm cd}, r\}$); column challenge $\tau_\zeta$ fixes shared column point $\boldsymbol\zeta$, prover commits claim column $\mathsf{CL} = (t_j, c_j)_{j < J}$ ($J = R_{\max}$, paired across halves $t_{j + J/2} = t_j + R_{\max}/2$) and stable configuration-row claims $c_r$ ($m_{\rm cl}$).
   - *5b. Linear combination $\gamma_{\rm hco}$:* Prover commits hidden multiplicity vector $\boldsymbol\alpha$ ($\alpha_i = \sum_j \gamma_{\rm hco}^j [t_j = i]$) and combined weight row $\mathbf u = \boldsymbol\alpha^\top \mathbf W^{\rm cd}$ ($m_{\mathbf u}$).
   - *5c. Column query $\tau_{\rm col}$:* Selects $m_q$ public Reed–Solomon column indices $\mathcal Q$; prover commits queried static columns $\mathbf M_{:,\mathcal Q}$, column salts, and $H_{\rm in}$ Merkle sibling paths ($m_{\rm col}$), verified in-circuit against $\mathsf C_W$.
   - *5d. Column-check batching $\beta_{\rm col}$:* Batches the $m_q$ column inner-product checks, followed by round-by-round sumcheck challenges.
6. **Lookup challenge $\tau_{\rm lu}$:** Draws $(\gamma, \lambda)$ challenges for all LogUp instances (including tile table `L5'`, layer table `L11`, vector table `L12`, slot/automaton/range tables); prover commits auxiliary inverse columns.
7. **Constraint reduction rounds:** Zero-check random evaluation points, Libra mask combination $\mu_{\rm L}$, round-by-round sumcheck challenges, and evaluation-claim batching challenges.
8. **Basefold proximity rounds:** Folding challenges $\chi_1, \dots, \chi_{m_{\rm f}}$ interleaved with folded oracle commitments, concluding with query challenge $\tau_{\rm qry}$ opening $s$ codeword positions.

`[Def]` **Model-level setup transcript ($\pi_{\rm setup}$, [Spec 09](09-model-commitment.md) §2.2).** Generated once per `model_id` release across $B_{\rm setup} = \lceil R_{\max}/7 \rceil$ row-chunk sub-proofs and one column sub-proof linked by a single Fiat–Shamir chain: Phase 1 commits message rows $(\mathbf W^{\rm cd}_b, \mathbf W^{\rm raw}_b)$, auxiliary columns, and masks across all chunks $\longrightarrow$ $\mathrm{seed}_1 = H_{\rm out}(\text{statement}, \text{all Phase-1 roots})$ selects $m^{\rm s}_q = 206$ shared column indices $\mathcal Q^{\rm s}$ $\longrightarrow$ the column sub-proof commits the queried full columns, salts, and $H_{\rm in}$ paths (verified in-circuit against $\mathsf C_W, \mathsf C_{\rm raw}$, and `model_id`) $\longrightarrow$ $\mathrm{seed}_2$ derives row weights $r$, column weights $\beta$, and linkage points $(z_G, z_N)$ $\longrightarrow$ each sub-proof executes zero-check, sumcheck, LogUp, and Basefold rounds and outputs linkage scalar $T_x$ (verifier checks $\sum_x T_x = 0$). All first-time challenge queries in $\pi_{\rm setup}$ count toward the same lifetime budget $Q$.

Every challenge is hash-chained via $c_\tau = H_{\rm out}(\texttt{dom}_\tau \Vert c_{\rm prev} \Vert \mathsf{Enc}(m_{\rm new}))$, with $c_{\rm samp}$ explicitly absorbing $v_{\rm rel}$, `profile`, and every field of $X_0$ (including complete response $y$, $N_c, N_s$, state anchors, $\sigma, \rho, \Pi$, and sample counts).

### 4.2 Stratified Cost-Weighted Sampling

`[Def]` Expanding $\tau_{\rm samp}$ under domain-separated $H_{\rm out}$ calls yields $\sum_{j=1}^{|\Pi|} k_{1,j} + k_2$ independent $n_{\rm smp}$-bit unsigned integers ($n_{\rm smp}$ denotes sampling bit-width, distinct from lineage depth $h$). For sample $i$ in compute stratum $j$ and sample $i$ in the generation pool, the circuit enforces

$$
J_{C,j,i} = \left\lfloor \frac{U_{j,i}\,C_j}{2^{n_{\rm smp}}} \right\rfloor \in \{0, \dots, C_j - 1\}, \qquad J_{G,i} = \left\lfloor \frac{U'_i\,N}{2^{n_{\rm smp}}} \right\rfloor \in \{0, \dots, N - 1\},
$$

via multi-limb Euclidean division $U_{j,i} C_j = J_{C,j,i} 2^{n_{\rm smp}} + \mathrm{rem}$ with range-checked limbs $0 \le \mathrm{rem} < 2^{n_{\rm smp}}$ (`[Lean: StdAlgOps.divmod_unique]`). Each $J_{C,j,i}$ locates the unique slot $s$ whose stratum prefix-sum interval $[P_{j,s}, P_{j,s} + c_s)$ contains $J_{C,j,i}$, resolves the exact cost atom $a$ inside $s$ ([Spec 05](05-computation-model.md) §1.5 `E5`), and opens all blocks in $\mathrm{Chk}(U)$ across $U \in \mathrm{Lin}_h(a)$ ([Spec 01](01-relation.md) §4.5).

- **Fixed circuit shape:** The circuit unconditionally unrolls $\sum_j k_{1,j}$ atom-lineage slots and $k_2$ generation-step slots with replacement (duplicate hits are proved repeatedly without deduplication or secret-dependent rejection loops).
- **Empty-stratum bypass:** Each stratum carries a private Boolean flag enforcing $(C_j = 0) \lor (0 \le J_{C,j,i} < C_j)$ (`[Lean: RoundByRound.multi_pool_map_le]`), leaking nothing because $N = 0 \iff C = 0$ is already determined by public branches ([Spec 01](01-relation.md) §6.3).
- **Default bit-width:** $n_{\rm smp} = 192$, subject to $\log_2(\sum_j k_{1,j} C_{\max} + k_2 N_{\max}) \le 85$.

### 4.3 Finite Seed-Mapping Bias

`[Lean: LifetimeSoundness.bucketCount_bounds]` `[Lean: LifetimeSoundness.bucket_prob_close]` `[Lean: RoundByRound.multi_pool_map_le_max]` Under the integer floor map $U \mapsto \lfloor U C / 2^{n_{\rm smp}} \rfloor$, every unit's selection probability deviates from $1/C$ by strictly less than $2^{-n_{\rm smp}}$. Across $|\Pi|$ independent compute strata and the generation pool, the additive mapping bias satisfies

$$
a_{\rm map} \le \frac{\sum_{j=1}^{|\Pi|} k_{1,j} C_{\max} + k_2 N_{\max}}{2^{n_{\rm smp}}}.
$$

When $n_{\rm smp} = 192$ and the numerator does not exceed $2^{85}$, $a_{\rm map} \le 2^{-107}$ (`[Lean: RoundByRound.amap_default]`).

---

## 5. Spot-Check Miss Lemmas

`[Lean: LifetimeSoundness.two_pool_all_good_le]` `[Lean: RoundByRound.multi_pool_max]` Fix any committed trajectory $w_R$. Suppose stratum $j$ draws $k_{1,j}$ cost units uniformly with replacement (where a unit is good iff its atom satisfies $V_a = 0$) and the generation pool draws $k_2$ steps uniformly with replacement from $G$. If $\max_j d^{(j)}_C(X, w_R) \ge \sigma \lor d_G(X, w_R) \ge \rho$, then the probability that **all** drawn samples across all non-empty pools are good is at most

$$
\max\Bigl\{\max_{1 \le j \le |\Pi|} (1 - \sigma)^{k_{1,j}},\;(1 - \rho)^{k_2}\Bigr\}.
$$

*Proof.* At least one stratum $j^\star$ satisfies $d^{(j^\star)}_C \ge \sigma$ (forcing $C_{j^\star} > 0$) or the generation pool satisfies $d_G \ge \rho$ (forcing $N > 0$). Conditioning on that single violated pool already bounds the joint all-good event by $(1 - \sigma)^{k_{1,j^\star}}$ or $(1 - \rho)^{k_2}$, requiring no independence assumption between pools and avoiding loose union sums. Cost-unit weighting is essential: unweighted uniform block sampling fails whenever an adversary concentrates cheats in high-FLOP blocks. $\square$

*(Hypergeometric sampling without replacement satisfies $\binom{C - D}{k}/\binom{C}{k} \le (1 - D/C)^k$, `[Lean: LifetimeSoundness.two_pool_wor_le]`, but production circuits sample with replacement to preserve fixed secret-independent circuit topology.)*

---

## 6. Round-by-Round Soundness Analysis

### 6.1 Doomed States and Protocol Reduction Obligations

`[Def]` At the instant of the $\tau$-th challenge query along a transcript, the **state** $s$ consists of the round tag, the full message prefix, and all objects extracted from $\mathcal T$ up to that query; $s$ is completely determined before the random oracle samples the challenge answer. Each round defines a **doomed-state predicate** $\mathsf D_\tau(s)$:

- **Sampling round ($\tau_{\rm samp}$):** $\mathsf D_{\rm samp}(s) \iff \mathsf{BadAccept}(X, w_R)$ holds on the extracted witness $w_R$.
- **Post-sampling local commitment ($m_1$):** The active claim set becomes

$$
\mathrm{Cl}_1 = E(X, w_R) \;\land\; \mathrm{Chain}(X, w_R) \;\land\; \bigwedge_{\text{sampled } a} (V_a = 0) \;\land\; \bigwedge_{\text{sampled } t} S_t \;\land\; (\text{committed PCS oracles within distance } \delta_{\rm bf}),
$$

and $\mathsf D_1(s) \iff \neg \mathrm{Cl}_1$ (`[Lean: ProtocolBridge.compiled_cs_doomed_of_bad_sample]`).
- **HCO rounds (§4.1 Step 5, [Spec 09](09-model-commitment.md) §3.1):** After $\boldsymbol\xi$, the claim asserts $Y^h = \mathbf W^h_* X^h$ across checked GEMMs and exact row folding at $\boldsymbol\xi$; after $\boldsymbol\zeta$, $c_j = \widetilde W_{*,t_j}(\boldsymbol\xi, \boldsymbol\zeta)$ for all $j < J$; after $\gamma_{\rm hco}$, $\sum_j \gamma_{\rm hco}^j c_j = \tilde{\mathbf u}(\boldsymbol\xi, \boldsymbol\zeta)$ and $\mathbf u = \boldsymbol\alpha^\top \mathbf W^{\rm cd}_*$; after $\tau_{\rm col}$, $\mathrm{RS}(\mathbf u)[j] = \langle \boldsymbol\alpha, \mathbf M_{:,j} \rangle$ and $\mathsf C_W$-consistency for all $j \in \mathcal Q$; after $\beta_{\rm col}$, the batched column-verification sumcheck claim.
- **Subsequent algebraic and PCS rounds:** $\mathrm{Cl}_\tau$ tracks the reduced sumcheck, LogUp, Libra, batching, and Basefold proximity claims, with $\mathsf D_\tau(s) \iff \neg \mathrm{Cl}_\tau$.
- **Model setup transcript ($\pi_{\rm setup}$):** $\mathsf D_\tau(s)$ holds whenever the committed setup matrices violate any condition `(i)`–`(vi)` of [Spec 09](09-model-commitment.md) §2.2 (at $\mathrm{seed}_1$, inconsistent column set $|D| \ge \delta_{\rm ca} n = 0.3 n$ or an invalid row constraint), with each round's escape charged to the same $Q \bar\varepsilon_{\rm rbr}$ budget.

`[Def]` Four protocol-level properties connect the constraint system to the round-by-round state machine (`[Lean: ProtocolBridge.ProtocolRBRSpec]`):
- **(P-final):** The verifier deterministically rejects every complete transcript whose terminal state is doomed.
- **(P-compile):** Extracted columns satisfy the compiled algebraic constraints iff the semantic claims hold, and each round reduction preserves falsehood outside its escape set (`[Lean: ProtocolBridge.CompiledCS]`).
- **(P-loc):** [Spec 01](01-relation.md) §4.5 Locality Condition `(L)` holds, binding sampled atom lineages to $R$ and HCO claims to $W_*$.
- **(P-rbr):** From any doomed state $s$ at round $\tau$ and any next prover message, a fresh uniform challenge lands in the escape set $\mathrm{Esc}(s)$ with probability at most $\varepsilon_\tau$.

### 6.2 Per-Round Escape Probability Budget

| Protocol Round | Escape Bound $\varepsilon_\tau$ (from any doomed state) | Verification Status |
|---|---|---|
| Spot-check $\tau_{\rm samp}$ | $0$ if $\neg(E \land \mathrm{Chain})$; else $\le \max\{\max_j (1-\sigma)^{k_{1,j}}, (1-\rho)^{k_2}\} + a_{\rm map}$ | `[Lean: RoundByRound.multi_pool_map_le_max]` |
| Out-of-domain binding $\tau_{\rm ood}$ ($L > 1$ only) | $\binom{L}{2} D_{\rm ood} / |\mathbb E|$ | `[Paper]` Schwartz–Zippel |
| Zero-check random evaluation point | $n_{\rm var} d / |\mathbb E|$ | `[Paper]` Schwartz–Zippel |
| Sumcheck round $r$ | $d_r / |\mathbb E|$ | `[Lean: Sumcheck.agree_card_le]`, `[Lean: ProtocolBridge.sumcheck_round_esc_le]` |
| Multi-column LogUp $\tau_{\rm lu}$ ($w$ cols, $n_v$ queries, $n_t$ rows) | $\bigl(n_t(w - 1) + n_v + n_t\bigr) / |\mathbb E|$ | `[Lean: RoundByRound.logup_multicol_prob]` (§6.3) |
| Libra mask combination $\mu_{\rm L}$ | $1 / |\mathbb E|$ | `[Paper]` |
| Claim batching ($N_{\rm cl}$ claims via powers) | $(N_{\rm cl} - 1) / |\mathbb E|$ | `[Paper]` |
| Basefold folding round $j$ | $\varepsilon_{\rm fold}(n_j) \le n_j / |\mathbb E|$ (unique-decoding regime) | `[Assume]` (`A-PCS`), `[Lean: RoundByRound.fold_p3_bits]` |
| Basefold query $\tau_{\rm qry}$ ($s$ positions) | $(1 - \delta_{\rm bf})^s$; for $\delta_{\rm bf} = 0.355, s = 168$: $0.645^{168} \le 2^{-106}$ | `[Assume]` (`A-PCS`), `[Lean: ProtocolBridge.basefold_qry_esc_le]` |
| HCO two-stage tile reduction $\tau_\xi, \tau_\zeta$ | $\log_2 T / |\mathbb E|$ per stage ($2 \log_2 T / |\mathbb E|$ total, $T = T_o = T_i$) | `[Paper]` ([Spec 09](09-model-commitment.md) §3.1) |
| HCO claim combination $\gamma_{\rm hco}$ | $J_{\max} / |\mathbb E|$ | `[Paper]` |
| HCO multiplicity LogUp ($\boldsymbol\alpha$, $w = 1$) | $(n_r + J_{\max}) / |\mathbb E|$ | `[Lean: LogUp.logup_sound_prob]` |
| HCO column query $\tau_{\rm col}$ ($m_q$ columns) | $(\vartheta + \delta_{\rm ca})^{m_q}$; at $\vartheta = 1/4, \delta_{\rm ca} = 0.3, m_q = 123$: $0.55^{123} \le 2^{-106}$ | `[Lean: ProtocolBridge.hco_col_esc_le]` ([Spec 09](09-model-commitment.md) Lemma 5.1) |
| HCO column-check batching $\beta_{\rm col}$ | $m_q / |\mathbb E|$ | `[Paper]` |
| `TC` history-matrix column query $\tau_{\rm col}^H$ ($m_q^H = 268$) | $(1 - \delta_H)^{m_q^H}$; at $\delta_H = 0.24$: $0.76^{268} \le 2^{-106}$ | `[Lean: ProtocolBridge.tc_col_esc_le]` ([Spec 10](10-trace-commitment.md) §3) |
| Constrained-decoding mask MLE check ($\mathbf r_v \in \mathbb E^{v_b}$) | $v_b / |\mathbb E|$; for $v_b \le 32$ over $\mathbb F_{p^3}$: $\le 2^{-186}$ | `[Lean: ConstrainedMask.constrained_mask_goldilocks]` ([Spec 05](05-computation-model.md) §4.2) |
| $\pi_{\rm setup}$ joint column query $\mathrm{seed}_1$ ($m^{\rm s}_q$ cols) | $(1 - \delta_{\rm ca})^{m^{\rm s}_q}$; at $\delta_{\rm ca} = 0.3, m^{\rm s}_q = 206$: $0.7^{206} \le 2^{-106}$ | `[Lean: ProtocolBridge.setup_col_esc_le]` ([Spec 09](09-model-commitment.md) §2.2) |
| $\pi_{\rm setup}$ linkage $\mathrm{seed}_2$ | $2 / |\mathbb E|$ (bilinear column diff) and $\le 20 / |\mathbb E|$ ($\mathrm{eq}(z, \cdot)$ linkage) | `[Paper]` ([Spec 09](09-model-commitment.md) §2.2) |

`[Def]` Parallel sub-claims sharing a single challenge round (such as $k_{1,j}$ sampled block sumchecks folded under one challenge) do **not** multiply $\varepsilon_\tau$ by the sub-claim count: if at least one sub-claim is false, escaping doomed status requires that specific sub-claim to evaluate to true. Per-row LogUp inverse constraints $\psi_r(\lambda + f_\gamma(a_r)) = 1$ and zero-sum checks are accounted in the zero-check and sumcheck rows. Define $\bar\varepsilon_{\rm alg} = \max_{\tau \neq \tau_{\rm samp}} \varepsilon_\tau$ evaluated at public `profile` upper bounds. Note that $\bar\varepsilon_{\rm alg} \le 2^{-106}$ is tightly governed by the HCO column query ($0.55^{123} \approx 2^{-106.09}$) and $\pi_{\rm setup}$ column query ($0.7^{206} \approx 2^{-106.00}$); adjusting $\vartheta$ or $\delta_{\rm ca}$ requires recomputing $m_q$ and $m^{\rm s}_q$ accordingly.

### 6.3 Multi-Column LogUp Soundness

`[Def]` Consider query tuples $a_1, \dots, a_{n_v} \in \mathbb E^w$, pairwise distinct table tuples $t_1, \dots, t_{n_t} \in \mathbb E^w$, and prover-claimed multiplicities $m \in \mathbb E^{n_t}$ with $n_v < \mathrm{char}(\mathbb E)$ (`[Lean: LogUp.natCast_ne_zero_of_lt]`). The tuple-level lookup relation holds iff every $a_r$ belongs to $\{t_1, \dots, t_{n_t}\}$ and each $t_i$ occurs in $(a_r)_{r=1}^{n_v}$ with exact multiplicity $m_i$. Using polynomial fingerprint $f_\gamma(x) = \sum_{c=0}^{w-1} \gamma^c x_c$ of degree $\le w - 1$, LogUp checks $\sum_{r=1}^{n_v} (\lambda + f_\gamma(a_r))^{-1} = \sum_{i=1}^{n_t} m_i (\lambda + f_\gamma(t_i))^{-1}$.

**Lemma 6.1 (Tight Multi-Column LogUp Escape Bound)** `[Lean: RoundByRound.multicol_badgamma_card_le]` `[Lean: RoundByRound.logup_multicol_card_le]` `[Lean: RoundByRound.logup_multicol_prob]`: *Whenever the tuple-level lookup statement is false,*

$$
\Pr_{\gamma, \lambda \leftarrow \mathbb E}\bigl[\lambda \text{ is not a pole} \;\land\; \text{the harmonic identity holds}\bigr] \le \frac{n_t(w - 1) + n_v + n_t}{|\mathbb E|}.
$$

*Proof.* Fix $\gamma \in \mathbb E$ and merge table tuples sharing the same fingerprint value by summing their claimed multiplicities, yielding $n'_t \le n_t$ distinct scalar fingerprints.
1. If the induced scalar fingerprint multiplicity statement is false, clearing denominators produces a nonzero univariate polynomial in $\lambda$ of degree $\le n_v + n'_t \le n_v + n_t$ (`[Lean: LogUp.logup_bad_card_le]`), leaving at most $n_v + n_t$ non-pole roots $\lambda$. (Poles make inverse constraints unsatisfiable and cause rejection, affecting completeness/ZK retry rather than soundness.)
2. Suppose the tuple-level statement is false while the scalar fingerprint statement holds under $\gamma$:
   - *Case A (some $a_{r^\star} = x^\star$ is absent from the table):* If no table tuple shared $f_\gamma(x^\star)$, the net multiplicity of $f_\gamma(x^\star)$ would equal its positive query count in $[1, n_v]$, which is nonzero modulo $\mathrm{char}(\mathbb E)$. Hence $f_\gamma(x^\star) = f_\gamma(t_i)$ for some $i \in \{1, \dots, n_t\}$. Since $x^\star \neq t_i$, each $i$ yields a nonzero polynomial of degree $\le w - 1$ in $\gamma$ (`[Lean: LogUp.fingerprint_collision_card_le]`), giving at most $n_t(w - 1)$ bad values of $\gamma$.
   - *Case B (all queries lie in the table, but some $t_{i^\star}$ has true count $\neq m_{i^\star}$):* For the merged scalar multiplicity of $f_\gamma(t_{i^\star})$ to cancel this discrepancy, some $i \neq i^\star$ must satisfy $f_\gamma(t_{i^\star}) = f_\gamma(t_i)$, yielding at most $(n_t - 1)(w - 1)$ bad values of $\gamma$.

Combining both steps bounds bad $(\gamma, \lambda)$ pairs without an $O(n^2)$ pairwise union bound. $\square$

`[Paper]` `[Meas]` **Exact tightness.** For $w = 2$ and $K = n_t \mid (|\mathbb E| - 1)$ with primitive $K$-th root of unity $\omega$ and $\alpha^{2K} \neq 1$, setting $A = \{(\alpha \omega^r, \alpha^{-1} \omega^{-r})\}_{r=0}^{K-1}$ and $T = \{(\alpha^{-1} \omega^r, \alpha \omega^{-r})\}_{r=0}^{K-1}$ produces identical fingerprint multisets for **all $K = n_t(w - 1)$** choices $\gamma \in \{\alpha^2 \omega^{2s}\}_{s=0}^{K-1}$ while $A \cap T = \varnothing$ (exhaustively verified over prime fields $p \in \{97, 193, 257, 7681\}$ for $K \le 32$ on Apple M1). Thus the factor $n_t$ in $n_t(w - 1)$ is optimal. (Using $w - 1$ independent challenges $f(x) = x_0 + \sum_{c \ge 1} \gamma_c x_c$ lowers the term to $n_t / |\mathbb E|$ while retaining $n_t$; drawing $r$ independent $(\gamma, \lambda)$ pairs within the **same** Fiat–Shamir round achieves $\varepsilon_{\rm lu}^r$.)

### 6.4 Basefold Proximity, Decoding Regime, and the $\mathbb F_{p^3}$ Challenge Field

`[Assume]` **Premise `A-PCS`:** Zero-knowledge Basefold (Zeilberger–Chen–Fisch, CRYPTO 2024; cf. FRI round-by-round knowledge soundness in Block et al., ASIACRYPT 2023, and Reed–Solomon proximity gaps in Ben-Sasson et al., FOCS 2020) instantiated over Reed–Solomon foldable codes of base rate $\vartheta_{\rm bf}$, proximity parameter $\delta_{\rm bf}$, query count $s$, and extension field $\mathbb E$ satisfies the per-round folding and query escape bounds of §6.2 under two explicit parameter rules:
1. *MDS distance:* Using Reed–Solomon foldable codes provides exact MDS minimum distance $1 - \vartheta'_{\rm bf}$ (switching to random foldable codes would require recalculating $\delta_{\rm bf}$ and $s$ from that code family's proved distance bound).
2. *Blinding rate overhead:* Zero-knowledge blinding expands message length from $k$ to $k + k_{\rm bl}$, raising effective rate to $\vartheta'_{\rm bf} = (k + k_{\rm bl})/n > \vartheta_{\rm bf}$. Unique decoding requires $\delta_{\rm bf} < (1 - \vartheta'_{\rm bf})/2$. For $\vartheta_{\rm bf} = 1/4$ plus $1/32$ blinding overhead, $(1 - \vartheta'_{\rm bf})/2 \approx 0.359$; setting $\delta_{\rm bf} = 0.355$ gives $s = \lceil 106 / \log_2(1/0.645) \rceil = 168$ queries so that $(1 - \delta_{\rm bf})^s = 0.645^{168} \le 2^{-106}$ (`[Lean: ProtocolBridge.basefold_qry_esc_le]`).

 FlashProve-Next operates in the **unique-decoding regime ($L = 1$)**, avoiding list-decoding multiplicity factors and out-of-domain pre-sampling rounds $\tau_{\rm ood}$.

`[Lean: RoundByRound.field_p2_iff]` `[Lean: RoundByRound.field_p3_bits]` **Necessity of the cubic extension $\mathbb F_{p^3}$.** Achieving $\bar\varepsilon_{\rm alg} \le 2^{-106}$ requires every algebraic numerator $D_\tau$ to satisfy $D_\tau \le 2^{-106} |\mathbb E|$:
- Over Goldilocks quadratic extension $\mathbb F_{p^2}$ ($p = 2^{64} - 2^{32} + 1$, $p^2 < 2^{128}$), $D_\tau / p^2 \le 2^{-106}$ holds **if and only if** $D_\tau \le 2^{22} - 1$ (`[Lean: RoundByRound.field_p2_iff]`, `[Lean: RoundByRound.codeword_p2_insufficient]`).
- Over Goldilocks cubic extension $\mathbb F_{p^3}$ ($p^3 > 2^{191.99}$), $D_\tau / p^3 \le 2^{-106}$ holds for all $D_\tau \le 2^{85}$ (`[Lean: RoundByRound.field_p3_bits]`).

Under the unified `profile` lookup caps of [Spec 05](05-computation-model.md) §1.5 ($n_v, n_t \le 2^{48}, w \le 8$), multi-column LogUp numerators reach $n_t(w - 1) + n_v + n_t < 2^{52}$, and codeword lengths reach $n_j \le 2^{42}$ (with `Qwen3`-scale slot tables already around $n_t \sim 2^{29}$ and codewords $\sim 2^{32} \gg 2^{22}$). Therefore `fpn-soundness-1.0` mandates $\mathbb E = \mathbb F_{p^3}$ across all algebraic, HCO, `TC`, constrained-mask, and $\pi_{\rm setup}$ challenge rounds. *(Implementation status: `flashprove_next/field.py` and `zk_flat.py` instantiate $\mathbb E = \mathbb F_p[u]/(u^3 - 2)$ (`EDIM = 3`) and `T_QUERIES = 168` across both $\pi_{\rm setup}$ and `reply_zk`.)*

---

## 7. Main Lifetime Knowledge Soundness Theorem

**Theorem 7.1 (Round-by-Round Lifetime Knowledge Soundness)** `[Lean: RoundByRound.rbr_lifetime_soundness]` `[Lean: RoundByRound.rbr_lifetime_soundness_sum]` `[Lean: ProtocolBridge.fpn_protocol_lifetime_soundness]`: *Under `A-ROM`, `A-AROM` (or pure-ROM trajectory instantiation `R3`), `A-PCS`, `P-final`, `P-compile`, `P-loc`, and `P-rbr`, every adaptive lifetime adversary $\mathcal A$ making at most $Q$ challenge-domain queries and $Q_{\rm tot}$ total oracle queries satisfies*

$$
\boxed{\Pr\bigl[\mathsf{BadAccept}\bigr] \;\le\; \Delta(Q_{\rm tot}) + \mathbb E\!\left[\sum_{q=1}^{Q} \varepsilon_{\tau(q)}\right] \;\le\; \Delta(Q_{\rm tot}) + Q \cdot \bar\varepsilon_{\rm rbr},} \tag{1}
$$

*where*

$$
\bar\varepsilon_{\rm rbr} = \max\!\left\{\max\!\left\{\max_{1 \le j \le |\Pi|} (1 - \sigma)^{k_{1,j}},\;(1 - \rho)^{k_2}\right\} + a_{\rm map},\;\bar\varepsilon_{\rm alg}\right\}, \qquad \Delta(Q_{\rm tot}) = \Delta_{\rm SL}(Q_{\rm tot}) + \Delta_{\rm arom}(Q_{\rm tot}).
$$

**Corollary 7.2 (Additive-Form Envelope)** `[Lean: RoundByRound.total_le_rbr]`: *Since $\max(m + a_{\rm map}, \bar\varepsilon_{\rm alg}) \le m + \max(a_{\rm map}, \bar\varepsilon_{\rm alg})$ for $m \ge 0$,*

$$
\Pr\bigl[\mathsf{BadAccept}\bigr] \;\le\; \Delta(Q_{\rm tot}) + Q \cdot \left[\max\!\left\{\max_{1 \le j \le |\Pi|} (1 - \sigma)^{k_{1,j}},\;(1 - \rho)^{k_2}\right\} + \bar a\right], \qquad \bar a = \max\bigl(a_{\rm map},\;\bar\varepsilon_{\rm alg}\bigr).
$$

*Proof.* Fix $\mathcal A$'s internal random tape and sample oracle answers lazily.
1. Bound straight-line collision events `(B1)`–`(B3)` by $\Delta_{\rm SL}(Q_{\rm tot})$ (§2.2) and `A-AROM` exceptions by $\Delta_{\rm arom}(Q_{\rm tot})$.
2. Outside $\Delta(Q_{\rm tot})$, consider any accepted pair $(X, \pi)$ for which $\mathsf{BadAccept}(X, w_R)$ holds. By prefix preservation (§2.1), the state at $\tau_{\rm samp}$ is doomed; by `P-final`, acceptance implies the terminal state of $(X, \pi)$ is not doomed. Walking the transcript rounds yields a unique **first** round where the state transitions from doomed to non-doomed (`[Lean: ProtocolBridge.first_escape_of_doomed_to_clear]`). Mapping that round's challenge to its first-time oracle query $q \in \{1, \dots, Q\}$ shows $\mathsf{BadAccept} \implies \exists\,q : \mathsf D(s_q) \land c_q \in \mathrm{Esc}(s_q)$ (`[Lean: ProtocolBridge.hpath_of_protocol_spec]`).
3. Because $s_q$ depends strictly on query answers prior to $q$, the fresh answer $c_q$ is uniform and independent of $s_q$, giving $\Pr[c_q \in \mathrm{Esc}(s_q) \mid \mathsf D(s_q)] \le \varepsilon_{\tau(q)}$ by `P-rbr` (§5 and §6.2).
4. Summing over the $Q$ adaptive queries via `[Lean: Grinding.adaptive_grind_real]` and `[Lean: RoundByRound.rbr_lifetime_soundness]` yields Equation (1). $\square$

Theorem 7.1 relies on **zero** economic, hardware-pricing, or honest-compute-cost premises.

---

## 8. Explicit Sample Counts and Parameter Table

`[Def]` Select public auxiliary upper bounds $\bar\Delta, \bar a$ satisfying $\varepsilon_{\rm aux} = \bar\Delta + Q \bar a < \varepsilon$, and set the spot-check miss budget to $\varepsilon_{\rm sp} = \varepsilon - \varepsilon_{\rm aux}$. For any defect threshold $0 < s < 1$, define

$$
\boxed{k(s, Q, \varepsilon_{\rm sp}) = \left\lceil \frac{\ln(Q / \varepsilon_{\rm sp})}{-\ln(1 - s)} \right\rceil, \qquad k_{1,j} = k(\sigma, Q, \varepsilon_{\rm sp}), \qquad k_2 = k(\rho, Q, \varepsilon_{\rm sp}).} \tag{2}
$$

`[Lean: LifetimeSoundness.kSuff_iff_kReq_le]` Equation (2) is the **exact least integer** $k$ satisfying $Q(1 - s)^k \le \varepsilon_{\rm sp}$ (for $s = 1$, $k = 1$ suffices on non-empty pools).

`[Def]` **Canonical budget split.** Allocate $\varepsilon_{\rm sp} = \varepsilon / 2$, $\bar\Delta \le \varepsilon / 4$, and $\bar a = \max(a_{\rm map}, \bar\varepsilon_{\rm alg}) \le \varepsilon / (4 Q)$, so $\bar\Delta + Q(m + \bar a) \le \varepsilon / 4 + \varepsilon / 2 + \varepsilon / 4 = \varepsilon$ (`[Lean: RoundByRound.default_budget_rbr]`). At the default lifetime target $(Q = 2^{64}, \varepsilon = 2^{-40})$, this sets $\bar\Delta \le 2^{-42}$ and $\bar a \le 2^{-106}$, yielding $k(0.5, 2^{64}, 2^{-41}) = 105$ (`[Lean: LifetimeSoundness.k105_sufficient]`, whereas $k = 104$ is strictly insufficient: `[Lean: LifetimeSoundness.k104_insufficient]`, `[Lean: RoundByRound.k104_fails_rbr]`).

Every cell below lists **$(k_{1,j}, k_2)$**—draws per compute stratum and draws in the generation pool, with replacement—machine-checked across all $16$ entries (`[Lean: LifetimeSoundness.table_Q30_e20]`, `[Lean: LifetimeSoundness.table_Q30_e40]`, `[Lean: LifetimeSoundness.table_Q64_e20]`, `[Lean: LifetimeSoundness.table_Q64_e40]`):

| $\sigma$ | $\rho$ | $Q = 2^{30}, \varepsilon = 2^{-20}$ | $Q = 2^{30}, \varepsilon = 2^{-40}$ | $Q = 2^{64}, \varepsilon = 2^{-20}$ | $Q = 2^{64}, \varepsilon = 2^{-40}$ |
|---|---|---|---|---|---|
| $0.5$ | $0.5$ | $(51, 51)$ | $(71, 71)$ | $(85, 85)$ | $(105, 105)$ |
| $0.5$ | $0.3$ | $(51, 100)$ | $(71, 138)$ | $(85, 166)$ | $(105, 205)$ |
| $0.5$ | $0.2$ | $(51, 159)$ | $(71, 221)$ | $(85, 265)$ | $(105, 327)$ |
| $0.5$ | $0.1$ | $(51, 336)$ | $(71, 468)$ | $(85, 560)$ | $(105, 691)$ |
| $0.3$ | $0.5$ | $(100, 51)$ | $(138, 71)$ | $(166, 85)$ | $(205, 105)$ |
| $0.3$ | $0.3$ | $(100, 100)$ | $(138, 138)$ | $(166, 166)$ | $(205, 205)$ |
| $0.3$ | $0.2$ | $(100, 159)$ | $(138, 221)$ | $(166, 265)$ | $(205, 327)$ |
| $0.3$ | $0.1$ | $(100, 336)$ | $(138, 468)$ | $(166, 560)$ | $(205, 691)$ |
| $0.2$ | $0.5$ | $(159, 51)$ | $(221, 71)$ | $(265, 85)$ | $(327, 105)$ |
| $0.2$ | $0.3$ | $(159, 100)$ | $(221, 138)$ | $(265, 166)$ | $(327, 205)$ |
| $0.2$ | $0.2$ | $(159, 159)$ | $(221, 221)$ | $(265, 265)$ | $(327, 327)$ |
| $0.2$ | $0.1$ | $(159, 336)$ | $(221, 468)$ | $(265, 560)$ | $(327, 691)$ |
| $0.1$ | $0.5$ | $(336, 51)$ | $(468, 71)$ | $(560, 85)$ | $(691, 105)$ |
| $0.1$ | $0.3$ | $(336, 100)$ | $(468, 138)$ | $(560, 166)$ | $(691, 205)$ |
| $0.1$ | $0.2$ | $(336, 159)$ | $(468, 221)$ | $(560, 265)$ | $(691, 327)$ |
| $0.1$ | $0.1$ | $(336, 336)$ | $(468, 468)$ | $(560, 560)$ | $(691, 691)$ |

Under default two-stratum partitioning $\Pi_2$ and $\sigma = \rho = 0.5$ at $(Q = 2^{64}, \varepsilon = 2^{-40})$, total draws equal $2 \times 105 + 105 = 315$ (`[Lean: RoundByRound.three_pool_default]`, `[Lean: RoundByRound.rbr_default_lifetime]`).

`[Paper]` **Verification of auxiliary bounds at default parameters:**
- $\Delta_{\rm SL} \le 2^{-86}$ (`[Lean: ProtocolBridge.delta_SL_p4_default]`), leaving nearly the entire $2^{-42}$ slack for $\Delta_{\rm arom}$;
- $a_{\rm map} \le 2^{-107}$ at $n_{\rm smp} = 192$ (`[Lean: RoundByRound.amap_default]`);
- $\bar\varepsilon_{\rm alg} \le 2^{-106}$ over $\mathbb F_{p^3}$: lookup numerators $< 2^{52} \le 2^{85}$ (`[Lean: RoundByRound.field_p3_bits]`), HCO column queries $0.55^{123} \le 2^{-106}$ (`[Lean: ProtocolBridge.hco_col_esc_le]`), setup column queries $0.7^{206} \le 2^{-106}$ (`[Lean: ProtocolBridge.setup_col_esc_le]`), Basefold folding $n_j / p^3 \le 2^{-149}$ for $n_j \le 2^{42}$ (`[Lean: RoundByRound.fold_p3_bits]`), and Basefold queries $0.645^{168} \le 2^{-106}$ (`[Lean: ProtocolBridge.basefold_qry_esc_le]`).

`[Def]` **Operational semantics of $Q$.** $Q$ is a cryptographic resource cap on total lifetime challenge queries; key rotations, `model_id` updates, tenant partitioning, or log truncation do **not** reset $Q$. All per-round escape bounds are evaluated using public `profile` upper bounds rather than hidden model dimensions.

---

## 9. Optional Micro-PoW, PRF Independence, and Post-Quantum Boundary

`[Def]` **Optional sampling-round micro-PoW.** If a sampling query $\tau_{\rm samp}$ is valid only when a disjoint bit-slice of the same $H_{\rm out}$ output satisfies a ticket predicate of density $\tau_{\rm pow} = 2^{-d}$, each sampling query's escape probability drops to $\tau_{\rm pow}(\max_\cdot m + a_{\rm map})$ (`[Lean: Grinding.prod_frac_le]`), reducing required draws to $k \ge \lceil \ln(Q \tau_{\rm pow} / \varepsilon_{\rm sp}) / (-\ln(1 - s)) \rceil$ (saving $\approx d$ draws at $\sigma = 0.5$ in exchange for $2^d$ honest hashes per proof). The baseline protocol sets $\tau_{\rm pow} = 1$ (zero PoW), avoiding PoW latency variance.

`[Def]` **PRF independence.** Keyed-sponge PRF security is never used in Theorem 7.1: spot-check challenges derive exclusively from $H_{\rm out}$, so $\Delta$ contains no PRF distinguishing term.

`[Open]` **Post-quantum (`QROM`) boundary.** Quantum superposition queries invalidate classical lazy-sampling query logging and `A-AROM` straight-line extraction (requiring Zhandry compressed-oracle techniques and dedicated post-quantum Fiat–Shamir analysis); substituting $Q \mapsto Q^2$ does not automatically lift Theorem 7.1 to the `QROM`.

---

## 10. Lean 4 Mechanization Coverage and Cryptographic Boundary

All combinatorial, algebraic, field-sizing, and round-by-round composition steps of `fpn-soundness-1.0` are axiom-checked with zero `sorry` in [`lean/FlashProveNext/`](../lean/THEOREM.md):

| Soundness Obligation | Mechanized Lean 4 Theorems | Explicit External Premises |
|---|---|---|
| Two-pool & multi-stratum miss bounds | `LifetimeSoundness.two_pool_all_good_le`, `LifetimeSoundness.two_pool_wor_le`, `RoundByRound.multi_pool_max`, `RoundByRound.three_pool_default` | None |
| $192$-bit integer seed-mapping bias | `LifetimeSoundness.bucketCount_bounds`, `LifetimeSoundness.two_pool_map_le_max`, `RoundByRound.multi_pool_map_le_max`, `RoundByRound.amap_default` | Multi-limb division constraint compilation (`P-compile`) |
| Sumcheck & multi-column LogUp escape | `SumcheckSound.sumcheck_sound_real`, `ProtocolBridge.sumcheck_round_esc_le`, `LogUp.logup_bad_card_le`, `RoundByRound.multicol_badgamma_card_le`, `RoundByRound.logup_multicol_prob` | Concrete circuit degree check (`P-compile`) |
| Goldilocks $p^4$ straight-line & $\mathbb F_{p^3}$ sizing | `ProtocolBridge.goldilocks_p4_bounds`, `ProtocolBridge.delta_SL_p4_default`, `RoundByRound.field_p2_iff`, `RoundByRound.field_p3_bits`, `RoundByRound.fold_p3_bits` | `A-ROM` ($H_{\rm out}$), `A-AROM` (Poseidon2 $P^{\pm 1}$) |
| HCO, $\pi_{\rm setup}$, TC, & Basefold query escape | `ProtocolBridge.hco_col_esc_le`, `ProtocolBridge.setup_col_esc_le`, `ProtocolBridge.tc_col_esc_le`, `ProtocolBridge.basefold_qry_esc_le` | `A-PCS` (Basefold proximity gap & folding), `A-CA` (`TC` only) |
| Constrained-decoding mask MLE binding | `ConstrainedMask.constrained_mask_check_sound`, `ConstrainedMask.constrained_mask_goldilocks` | None (one-hot embed `Q^2 \ll p` via `StdAlgOps.intCast_inj_small`) |
| Sample-count optimality & 16-cell table | `LifetimeSoundness.kSuff_iff_kReq_le`, `LifetimeSoundness.table_Q30_e20`..`table_Q64_e40`, `LifetimeSoundness.k105_sufficient`, `RoundByRound.k104_fails_rbr` | None |
| End-to-end RBR bridge & lifetime theorem | `RoundByRound.rbr_lifetime_soundness`, `RoundByRound.rbr_default_lifetime`, `ProtocolBridge.compiled_cs_doomed_of_bad_sample`, `ProtocolBridge.hpath_of_protocol_spec`, `ProtocolBridge.fpn_protocol_lifetime_soundness` | Instantiation of `ProtocolRBRSpec` (`P-ext`, `P-compile`, `P-loc`, `P-final`) on concrete circuit code |
