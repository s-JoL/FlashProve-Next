# Spec 01: Versioned API Response Relation (`fpn-relation-1.0`)

**Specification ID:** `fpn-relation-1.0` | **Role:** Core protocol relation (public statement, private witness, immutable object table, cost-stratified pools, and acyclic multi-turn state chain)

---

## Overview and Architectural Motivation

Production closed-weight LLM APIs execute native floating-point kernels (`bf16`, `fp8`, `FlashAttention`), randomized temperature and nucleus sampling, hidden system prompts, private chain-of-thought (`CoT`) reasoning, tool calls, prefix caching, and multi-turn encrypted state. Verifying such an API without trusted hardware, surrogate integer models, or architectural leakage requires a mathematical relation that binds the exact bytes returned to the client to the committed production model configuration while tolerating benign floating-point non-associativity and hiding every proprietary internal parameter.

`fpn-relation-1.0` defines the versioned NP-style relation $\mathcal R_{\Pi,\sigma,\rho}(X,w)$ verified by FlashProve-Next. Seven structural choices govern its design:

1. **Global structural enforcement vs. local block defect accounting.** A deterministic global predicate $E(X,w)$ checks every global witness object (configuration opening, token stream, slot enumeration table, constrained-decoding and stop automata, and multi-turn state anchors). Local activation and carried-state objects are never checked by $E$: if a local leaf lacks a hash preimage ($\bot$) or encodes an out-of-range or malformed tensor, that fault sets the enclosing block's defect indicator $V_b = 1$ and contributes the block's full computational cost toward the threshold fraction. Model weights have no local $\bot$ branch because they are globally fixed once at `model_id` publication via a zero-knowledge setup proof $\pi_{\rm setup}$ ([Spec 09](09-model-commitment.md)).
2. **One-hop local reference with carried-state certification.** Each computation block maps the exact inputs committed by the serving engine to a deterministic integer-encoded reference output, reference carried state, and input-dependent certified error radii. The block tolerance predicate $\mathrm{Tol}_{M,b}$ simultaneously checks both the block output and any newly emitted carried state (KV cache entries or recurrent state chunks) without chaining a shadow fixed-point model across layers.
3. **Zero-degree-of-freedom wiring over full-value sponge leaves.** Every cross-block boundary tensor resides in an immutable object table indexed by a public injective address map $\mathsf{Addr}$. Producer and consumer ports are wired by exact address equality, forcing both endpoints to open the identical leaf of the salted sponge Merkle root $R$. Leaves absorb full serialized tensor values under an in-circuit Poseidon2 sponge $H_{\rm in}$, eliminating underconstrained compressive algebraic digests.
4. **Public cost stratification.** Computation blocks are partitioned into strata by a public rule $\Pi$—by default $\Pi_2$, which separates input-column blocks from generation-column blocks. Without stratification, a 10,000-token prompt paired with a 500-token completion concentrates roughly $95\%$ of total FLOPs in prefill, allowing a dishonest provider to run an inexpensive substitute model on every generated token while keeping the aggregate bad-cost fraction near $0.05 < \sigma$. Stratification forces generation columns to pass their own independent threshold check.
5. **Dual-nonce freshness and replay isolation.** Each request pairs a fresh client nonce $N_c$ ($\ge 128$ bits, preventing cross-request proof replay by the provider) with a fresh per-attempt server random value $N_s$ ($256$ bits, preventing chosen-nonce common-random-number probing by clients without introducing cross-tenant monotonic counters). Every response derives its Fiat–Shamir spot-check challenges independently, with no cross-request batch root.
6. **Acyclic multi-turn state anchors.** Multi-turn sessions chain state through a public hiding commitment $\mathsf{St}_i$ that commits only the core API response $y_i^{\rm core}$—excluding the proof envelope $\mathsf{env}_i$ and encrypted state carrier $\mathsf{blob}_i$. This breaks the circular dependency $\mathsf{St}_i \to \mathsf{blob}_i \to y_i \to \mathsf{St}_i$ while binding hidden reasoning and tool history across turns via finite external induction.
7. **Byte-stream first-match stop semantics.** Stopping conditions are formalized as a deterministic byte-level Aho–Corasick automaton over the stop-eligible output byte stream, paired with pre-sampling token/context budget guards and explicit public branches for empty ($N=0$) and interrupted responses.

Evidence tags used throughout this specification follow the repository convention: `[Def]` for protocol definitions, `[Paper]` for paper-proved mathematical claims, `[Lean: Module.decl]` for axiom-checked Lean 4 theorems in [`lean/THEOREM.md`](../lean/THEOREM.md), `[Assume]` for explicit cryptographic or operational premises, `[Meas]` for hardware measurements, `[Est]` for analytical cost estimates, and `[Open]` for unmechanized or unmeasured items.

---

## 1. Objects of the Relation and Model Commitment

`[Def]` For a fixed statement and witness, let $B$ denote the set of active computation blocks, $G$ the set of logical generation steps, and $\mathrm{Chain}$ the global structural chain predicate. A production model configuration $M$ specifies both the deployed parameters and the set of permitted numerical implementations:

$$
M = \bigl(W,\;\mathsf{dtype},\;\mathsf{graph},\;\mathsf{Tok},\;\mathsf{Pre},\;\mathsf{Render},\;\mathsf{Inject},\;\mathsf{Sample},\;K_{\rm prf},\;\mathsf{MaskCompile},\;\mathsf{Stop},\;\mathsf{Cost},\;\mathsf{Tol},\;\mathsf{Caps}\bigr).
$$

`[Def]` Weights $W$ and deployed numerical formats $\mathsf{dtype}$ are encoded from their exact deployed bitstrings, block/channel scaling factors, and canonical interpretation rules. Execution graphs, parallelism-relevant templates, hidden system prompts, tool-rendering templates, and reasoning-effort rules all belong to the committed configuration. Write $M = (M_{\rm meta}, W)$, where $M_{\rm meta}$ comprises everything outside $W$ (operator graph, rounding contracts, cost table, implementation templates, port table, private tile table $\mathsf{Tt}$, range certificate, rendering and sampling rules, and $K_{\rm prf}$).

To amortize static metadata across proofs ([Spec 09](09-model-commitment.md) §4.7), $M_{\rm meta}$ is partitioned by update frequency:
- **Stable tables**—which change only when weights change (tile table $\mathsf{Tt}$, vector table, layer parameter table, operator graph and cost table, rounding contracts and templates, range certificate, and tokenizer tables)—are encoded into dedicated configuration rows of the weight matrix $\mathbf W$ under $\mathsf C_W$. Each response proof binds the lookup-table columns used in-circuit to $\mathsf C_W$ via evaluation-claim slots at the shared Hidden Column Opening (HCO) point.
- **Volatile configuration** $M^{\rm vol}$—which may rotate without changing weights ($K_{\rm prf}$, sampling parameters, and hidden prompt/template injections)—is committed under a fixed-length sponge digest $\mathsf C_{\rm cfg}$ and opened in full inside each response circuit.

`[Def]` **Dual hash separation.** The protocol employs two distinct cryptographic hash primitives ([Spec 02](02-soundness.md) §1, [Spec 03](03-zero-knowledge.md) §4):
- $H_{\rm in}$ is an algebraic sponge hash evaluated inside the zero-knowledge circuit (Poseidon2 over the Goldilocks field, width $12$, capacity $4$ elements, output $\ge 256$ bits). It instantiates every commitment opened in-circuit: `model_id`, the weight column Merkle trees, the configuration root $\mathsf C_{\rm cfg}$, the per-request trace root $R$, and multi-turn state anchors $\mathsf{St}_i$. Whenever uncommitted or unopened local objects enter the relation, their leaves must absorb the **complete serialized value** via $H_{\rm in}$; compressive algebraic digests (such as Byte-SIS $D = Av + Br$) are prohibited as binding trajectory leaves ([Spec 02](02-soundness.md) §2.3).
- $H_{\rm out}$ is evaluated strictly outside the circuit by the verifier for Polynomial Commitment Scheme (PCS) Merkle trees and Fiat–Shamir challenge derivation. It must be instantiated as an **independent BLAKE3 or SHA-3 instance sharing no permutation with Poseidon2**: the zero-knowledge simulator programs $H_{\rm out}$ as a classical random oracle while analyzing $H_{\rm in}$ and the sampling PRF in the ideal permutation model for Poseidon2 ([Spec 03](03-zero-knowledge.md) Theorem 4.1 `(A1)` and §5.4).

`[Def]` **Static model commitment `model_id`.** `model_id` is a salted hiding commitment rather than a hash of a human-readable model name ([Spec 09](09-model-commitment.md) §4.1). Model weights are committed once per release: the deployed raw bitstrings $\mathbf W^{\rm raw}$ and their canonical integer decodings $\mathbf W^{\rm cd} = \mathrm{CD}(\mathbf W^{\rm raw})$ ([Spec 05](05-computation-model.md) §1.1) are arranged into tiles of public shape $T_o \times T_i$, encoded row-wise with a Reed–Solomon code of rate $\vartheta$, and committed column-wise into salted $H_{\rm in}$ Merkle trees whose $j$-th leaf is $H_{\rm in}(\texttt{dom}_{\rm col}, j, \varsigma_j, \mathbf M_{:,j})$. This yields column roots $\mathsf C_W$ (canonical decodings) and $\mathsf C_{\rm raw}$ (raw bitstrings). The volatile configuration root is $\mathsf C_{\rm cfg} = H_{\rm in}(\texttt{dom}_{\rm cfg}, v_{\rm rel}, v_{\rm cap}, v_{\rm profile}, K_{\rm prf}, \mathsf{Enc}_{P_{\max}}(M^{\rm vol}))$. Together:

$$
\mathsf{model\_id} = H_{\rm in}\bigl(\texttt{dom}_{\rm mid},\;\varsigma_0,\;\mathsf C_W,\;\mathsf C_{\rm raw},\;\mathsf C_{\rm cfg}\bigr),
$$

where the master salt $\varsigma_0$ and every column salt $\varsigma_j$ contain at least $256$ fresh random bits, regenerated on every publication (including volatile-only configuration rotations, [Spec 09](09-model-commitment.md) §4.4).

The release record includes a one-time zero-knowledge setup proof $\pi_{\rm setup}$ ([Spec 09](09-model-commitment.md) §4.1) certifying that: (i) all rows of both codeword matrices agree with the Reed–Solomon encodings of the committed message rows in $\pi_{\rm setup}$ on a common column set of density $\ge 1 - \delta_{\rm ca}$; (ii) $\mathbf W^{\rm cd} = \mathrm{CD}(\mathbf W^{\rm raw})$ holds entrywise under the maximal valid per-row signed block-floating-point exponent (`[Lean: ArchGeneral.sbfp_shift_unique]`) and raw bitstrings are valid encodings of the declared $\mathsf{dtype}$; (iii) row-norm and scaling rows are consistent; (iv) radius rows satisfy $\mathbf W^r = |\mathbf W^{\rm cd}| + \kappa m$; and (v) stable configuration rows have unique keys and consistent derived parameters. The weight tensor in the response relation is **defined** as the unique Reed–Solomon decoding $W_* = \mathrm{Dec}(\mathbf M)$, which is fixed at `model_id` publication prior to any response challenge ([Spec 02](02-soundness.md) §2.1, [Spec 09](09-model-commitment.md) §5.1). Verifiers check $\pi_{\rm setup}$ once per `model_id`. Tile dimensions $T_o, T_i$, maximum row count $R_{\max}$, code rate $\vartheta$, column query count $m_q$, per-proof weight claim cap $J_{\max}$, configuration row count $N_{\rm cr}$, stable table capacities, and volatile configuration length $P_{\max}$ are public profile constants.

`[Def]` The protocol version fixes all byte encodings, constraint semantics, rounding tolerances, cost tables, stratification rules $\Pi$ (§4.4), and cryptographic parameters. Public parameters are uniquely determined by the `model_id` release record and public profile without disclosing secret architectural dimensions. Rotating hidden prompt injections or $K_{\rm prf}$ produces a new `model_id` bound to its activation epoch.

`[Assume]` Binding `model_id` to "the provider's official flagship model" relies on the client obtaining `model_id` from the provider's designated public release channel; cryptography proves consistency with that published commitment rather than auditing unannounced internal assets.

---

## 2. Canonical Encoding and Request Normalization

`[Def]` $\mathsf{Enc}$ is an injective byte encoding with explicit domain tags, type tags, and length prefixes:
- Unsigned integers use fixed endianness and fixed bit-width; signed integers use a unique canonical representation; Booleans take values strictly in $\{0, 1\}$.
- Finite-field elements are represented by canonical integers in $[0, p)$; non-canonical representatives are rejected.
- Floating-point values are serialized by raw IEEE/FP8 bit patterns, with versioned rules fixing the semantics of `NaN`, `±Inf`, and `±0`.
- Syntactic encoding validity is distinct from numerical admissibility: encoding validity of global objects belongs to $E(X,w)$, whereas malformed or non-finite local objects trigger $V_b = 1$ (§4.3).

`[Def]` Public request canonicalization $x = \mathsf{Canon}_{v_{\rm cap}}(x_{\rm pub})$ enforces six rules:
1. Parse against the frozen versioned API schema $v_{\rm cap}$, rejecting duplicate keys, ambiguous union variants, and ill-formed UTF-8/byte sequences.
2. Distinguish omitted fields, `null`, empty strings `""`, and empty arrays `[]`, populating defaults strictly per $v_{\rm cap}$.
3. Sort object keys lexicographically while preserving order-significant arrays (conversation messages, content blocks, tool declarations, stop sequences).
4. Preserve exact Unicode code-point sequences without silent NFC/NFD normalization, whitespace trimming, or newline rewriting.
5. Bind multimodal inputs (images, audio, documents) to exact decoded media bytes and preprocessor version $\mathsf{Pre}_M$, rather than mutable external URLs.
6. Bind message roles, tool-call identifiers, JSON schemas, service tiers, token/thinking budgets, stop sequences, cache controls, and multi-turn state fields.

`[Def]` Binary objects already returned or hashed by the public API may be bound via their public cryptographic digests; if an API accepts external references, the private witness must supply the resolved bytes and resolution binding. `[Assume]` Authenticity of external web/URI resources at request time lies outside the model inference relation.

`[Def]` **Typed segmented rendering and tokenization.** To prevent prompt-injection or boundary-merging ambiguities, the renderer emits a sequence of typed byte segments, and the tokenizer $\mathsf{Tok}_M$ (comprising vocabulary, special tokens, BPE/unigram merge rules, byte fallback, and template version) acts independently on each segment:

$$
\mathsf{Render}_M(x,\;z_{\rm inj},\;\mathsf{hist}_{\rm prev}) = (\varsigma_1,\dots,\varsigma_m), \qquad \varsigma_j = (\mathsf{type}_j,\;\mathsf{bytes}_j), \qquad u_{\rm in} = \big\Vert_{j=1}^{m} \mathsf{Tok}^{\mathsf{type}_j}_M(\mathsf{bytes}_j).
$$

Segment types distinguish at least control tokens, user/assistant text, multimodal placeholders, tool outputs, and replayed hidden history:
- Special control token IDs can be emitted only by control segments with fixed IDs; text-segment tokenization is forbidden from producing control token IDs (under escaping rules fixed by $\mathsf{Tok}_M$), and BPE merges never cross segment boundaries.
- Multimodal segments invoke $\mathsf{Pre}_M$ deterministically on raw media bytes to fix placeholder token counts and encoder patch inputs ([Spec 05](05-computation-model.md) §2.5).
- In-turn server-side tools form alternating segments—generation segment $G^{(1)}$, server tool return segment $z_{\rm tool}$, generation segment $G^{(2)}$, and so forth—where each tool return segment is tokenized and prefilled as a non-generated input segment ([Spec 05](05-computation-model.md) §4.4).
- Pre-assembled token arrays are never accepted as unconstrained free witnesses.

`[Def]` **Response payload vs. proof envelope.** Every API response splits into a core payload and a proof envelope, $y = (y^{\rm core}, \mathsf{env})$:
- $y^{\rm core}$ is the canonical serialization of standard API content blocks (visible text, tool calls, structured outputs, refusals), finish reason, token usage counters, and echoed stop sequence. Transport-layer chunking does not alter $y^{\rm core}$, though semantically meaningful streaming event orderings and terminal markers defined by $v_{\rm cap}$ are included.
- $\mathsf{env}$ carries the public fields added by FlashProve-Next: server randomness $N_s$ (§5.5), new multi-turn state anchor $\mathsf{St}_{\rm new}$, fixed-length encrypted state carrier $\mathsf{blob}$ (§6.4), and `proof_id`.

---

## 3. Public Statement and Private Witness

`[Def]` The semantic public view of an inference turn is

$$
X_0 = \bigl(v_{\rm rel},\;\mathsf{model\_id},\;x_{\rm pub},\;y,\;\ell_{\rm api},\;\mathsf{profile},\;N_c,\;N_s,\;e,\;\mathsf{sess},\;\mathsf{St}_{\rm prev},\;\mathsf{St}_{\rm new},\;\sigma,\;\rho\bigr),
$$

where $\ell_{\rm api}$ contains strictly the metadata already returned by the unmodified commercial API (input/output token counts, `reasoning_tokens`, cache-hit counts, `service_tier`), $N_c$ is the client nonce inside $x_{\rm pub}$, $N_s$ is the server random value, $e$ is the key epoch, and $(\mathsf{sess}, \mathsf{St}_{\rm prev}, \mathsf{St}_{\rm new})$ govern the multi-turn state chain (§6.4). All protocol versions, spot-check thresholds $(\sigma, \rho)$, stratification rules $\Pi$, and circuit dimensions are public deterministic functions of `model_id` and `profile` (or explicit client-chosen fields inside $x_{\rm pub}$).

`[Def]` The full cryptographic statement verified by the proof system is

$$
X = \bigl(X_0,\;R,\;\mathsf{cm}_{\rm glob}\bigr),
$$

where $R$ is the hiding Merkle root of the per-request trajectory object table (§4.1) and $\mathsf{cm}_{\rm glob}$ is the PCS commitment to all global witness columns ([Spec 02](02-soundness.md) §4.1). The retrieval handle `proof_id` binds $X$ if included in verification and encodes no secret-dependent index. All envelope fields ($R, \mathsf{cm}_{\rm glob}, \mathsf{St}_{\rm new}, N_s, \mathsf{blob}$, blinding randomness, and proof bytes) are simulated without secret inputs under the zero-knowledge simulator of [Spec 03](03-zero-knowledge.md), with release timing governed by the state machine of [Spec 04](04-completeness.md).

`[Def]` The private witness is the tuple

$$
w = \bigl(M,\;\omega_M,\;z_{\rm inj},\;z_{\rm think},\;z_{\rm tool},\;u,\;\mathcal O,\;\mathsf{Sl},\;q^{\rm fsm},\;q^{\rm stop},\;\mathsf{hist}_{\rm prev},\;\omega_{\rm prev},\;\mathsf{hist}_{\rm cur},\;r_{\rm cur},\;\omega_{\rm enc}\bigr),
$$

whose components and binding obligations are specified below:

| Witness Component | `[Def]` Contents and Binding Obligations |
|---|---|
| $M, \omega_M$ | Model configuration preimage and `model_id` opening $\omega_M = (\varsigma_0, \{\varsigma_j\}, \mathbf M^{\rm cd}, \mathbf M^{\rm raw}, \text{opening of }\mathsf C_{\rm cfg})$; fixed to match `model_id`, with weights globally constrained by $\pi_{\rm setup}$ (§1). |
| $z_{\rm inj}$ | Active hidden prompt/template injections and derivation witness from committed templates and request state. |
| $z_{\rm think}$ | Hidden reasoning (`CoT`) tokens, content-block type tags, visibility flags, and logical ordering. |
| $z_{\rm tool}$ | Tool call invocations, raw tool-return bytes, call IDs, in-turn round indices, and origin tags. |
| $u$ | Complete logical token sequence across all input segments and generated segments. |
| $\mathcal O$ | Trajectory object table mapping every buffer address $a \in \mathcal A_{\rm buf}$ to its serialized value $v_a$ and leaf salt $\mathsf{salt}_a$ (§4.1). |
| $\mathsf{Sl}$ | Private slot table encoding validity bits, sequence positions, graph nodes, MoE routing sets, costs, stratum prefix sums, stratum tags, and billing-unit IDs ([Spec 05](05-computation-model.md) §1.5). |
| $q^{\rm fsm}, q^{\rm stop}$ | State columns of the constrained-decoding automaton (§5.2) and byte-level stop automaton (§6.2). |
| $\mathsf{hist}_{\rm prev}, \omega_{\rm prev}$ | Prior-turn hidden history and cryptographic opening of $\mathsf{St}_{\rm prev}$ (§6.4). |
| $\mathsf{hist}_{\rm cur}, r_{\rm cur}$ | Current-turn hidden history passed forward and $256$-bit blinding randomness for $\mathsf{St}_{\rm new}$. |
| $\omega_{\rm enc}$ | Canonical encoding, range-check, mask, padding, and index-consistency auxiliary witnesses. |

`[Def]` **Extended witness domain.** To support straight-line extraction without rewinding ([Spec 02](02-soundness.md) §2), the value domain of local objects in $\mathcal O$ is extended to $\mathrm{Val} \cup \{\bot\}$. Honest witnesses never contain $\bot$; $\bot$ appears exclusively when the straight-line extractor encounters a trajectory leaf whose preimage was never queried to $H_{\rm in}$ during the service lifetime. Every predicate below is total over extended witnesses (§4.3). Conversely, if any **global** object (`model_id` opening, $M_{\rm meta}$, uniquely decoded weights $W_*$, token sequence $u$, slot table $\mathsf{Sl}$, automaton columns $q^{\rm fsm}, q^{\rm stop}$, or state-anchor openings) extracts to $\bot$ or fails unique decoding, $E(X,w)$ evaluates to `false`. Model weights have no local $\bot$ branch because $W_*$ is globally extracted and validated at `model_id` publication ([Spec 09](09-model-commitment.md) §5.1–§5.2) and accessed per response strictly via HCO evaluation claims $\mathrm{Wcl}(b)$ (§4.2).

`[Def]` **In-turn tool outputs.** All tool return bytes are committed inside $R$ and $\mathsf{cm}_{\rm glob}$ prior to deriving the spot-check challenge. Client-visible tool returns must match the public request/response bytes verbatim; private server-side tool returns enter as committed external input segments rather than model-generated tokens, while the model's tool invocation name, arguments, and post-tool continuations remain subject to generation and chain checks. `[Def]` The relation certifies model execution given those tool bytes, not the external factual accuracy of third-party tools or outputs of separate uncommitted auxiliary models.

---

## 4. Immutable Object Table, Computation Blocks, Tolerance, and Stratified Cost

### 4.1 Object Table and Zero-Freedom Wiring

`[Def]` **Trajectory objects.** Every cross-block boundary tensor is an immutable object: actual quantized or native floating-point GEMM/attention operands (codewords paired with scaling factors), block outputs, residual-stream states, newly produced carried states (KV cache page entries, MLA compressed latent vectors, or versioned recurrent state chunks), and final-layer normalized activations $x^{\rm fn}$. **Vocabulary logits are not trajectory objects**: they are neither materialized into $R$ nor hashed online. For each spot-checked generation step, reference logits and error radii are recomputed inside the circuit by the `HEAD` block from the committed $x^{\rm fn}$ object and committed LM-head weight tiles via the shared-challenge tile-product sumcheck (§5.4, [Spec 05](05-computation-model.md) §1.3, [Spec 09](09-model-commitment.md) §4.3).

Each logical object is identified by a tuple $\kappa = (\mathsf{seg}, t, s, \mathsf{port}, \mathsf{ver})$ recording segment index, sequence position, producing slot, output port, and state version. A public, deterministic, injective address function $\mathsf{Addr}: \kappa \mapsto a \in \mathcal A_{\rm buf}$ maps object keys (or fixed-size $\lambda_c$-element sub-chunks thereof, [Spec 05](05-computation-model.md) §1.4) into a buffer address space whose capacity $|\mathcal A_{\rm buf}|$ and Merkle depth $D_{\rm trace}$ depend solely on `profile`.

`[Def]` **Salted sponge trajectory commitment.** The trajectory root $R$ is a position-binding hiding vector commitment over $\mathcal A_{\rm buf}$:

$$
R = H_{\rm in}\Bigl(\texttt{dom}_{\rm root},\;\varsigma_R,\;\mathsf{MT}_{H_{\rm in}}\bigl((\ell_a)_{a \in \mathcal A_{\rm buf}}\bigr)\Bigr), \qquad \ell_a = H_{\rm in}\bigl(\texttt{dom}_{\rm leaf},\;a,\;\mathsf{salt}_a,\;\mathsf{Enc}(v_a)\bigr),
$$

where unused addresses take a fixed canonical padding leaf, each active leaf absorbs the complete encoded value $v_a$ with a fresh $\ge 256$-bit $\mathsf{salt}_a$, and $\varsigma_R$ is a fresh $256$-bit **root salt** drawn at `COMMIT` time and shared across retry attempts of the same request ([Spec 04](04-completeness.md) `R10`). Including $\varsigma_R$ is essential for zero knowledge on legal empty executions ($N = 0, B = \varnothing$, §6.3): without a root salt, an all-padding tree would collapse $R$ to a public constant distinguishable from the simulator's uniform string with advantage $\approx 1$, whereas $\varsigma_R$ ensures $R$ remains pseudorandom/indistinguishable regardless of how many leaves are padded ([Spec 03](03-zero-knowledge.md) §2.3). Prefix-cached KV pages reused across requests retain their original salted leaves at identical canonical addresses, whose preimages were already queried when first computed.

`[Def]` **Zero-freedom wiring $\mathsf{Wire}$.** Input port addresses of every active block are computed deterministically inside the circuit from the committed graph $\mathsf{graph}$ and slot table $\mathsf{Sl}$: whenever a graph edge connects output port $\mathrm{out}_{s_1}$ of slot $s_1$ to input port $\mathrm{in}_{s_2}$ of slot $s_2$,

$$
\mathsf{Addr}(\mathrm{in}_{s_2}) = \mathsf{Addr}(\kappa_{\mathrm{out}_{s_1}}).
$$

Position binding of $R$ forces both slots to read the exact same leaf and preimage, leaving zero prover degrees of freedom on inter-block wires. Carried states are linearized by version: each KV entry is written once ($\mathsf{ver}=0$), and each chunkwise recurrent state boundary increments $\mathsf{ver} \mapsto \mathsf{ver}+1$ from a unique producing slot. Format conversions (`Cast` or activation/KV quantization) are forbidden on wires and must execute as the producer block's trailing operator so that committed objects match actual downstream operands ([Spec 06](06-tolerance-and-cheats.md) Corollary 6.3).

### 4.2 Block Relation: Local Reference, Certified Radii, and Carried State

`[Def]` For each active slot $b \in B$, write its inputs as $\mathrm{In}(b) = (a_b, q^{\rm in}_b)$ and outputs as $\mathrm{Out}(b) = (o_b, q^{\rm out}_b)$, where $q^{\rm in}_b$ collects historical carried-state objects read by $b$ (all KV entries in the attention window or the prior recurrent state version) and $q^{\rm out}_b$ collects carried-state objects newly emitted by $b$. Let $\mathrm{Wcl}(b)$ denote the block's multilinear weight-tile evaluation claims $c_j = \widetilde W_{*,t_j}(\boldsymbol\xi, \boldsymbol\zeta)$ paired with hidden tile indices $t_j$ ([Spec 09](09-model-commitment.md) §4.2, §5.1), and define $\mathrm{Ref}(b) = \mathrm{In}(b) \cup \mathrm{Out}(b) \cup \mathrm{Wcl}(b)$. For the vocabulary head block $\mathrm{HEAD}(t)$, $\mathrm{Out}(\mathrm{HEAD}(t)) = \varnothing$.

`[Def]` The **certified local reference function** ([Spec 06](06-tolerance-and-cheats.md) §2) is a deterministic integer-encoded operator

$$
\bigl(\widehat o_b,\;\widehat q^{\rm out}_b,\;\mathbf r_b,\;\varrho_b\bigr) = F^{\sharp}_{M,b}\bigl(a_b,\;q^{\rm in}_b\bigr)
$$

(`[Lean: StdAlg.trace_exists]`, `[Lean: StdAlg.trace_unique]`), which starts from the **exact engine inputs committed in $R$**, computes a single-hop reference output and reference carried state, and propagates the committed per-operator rounding contract along the block's internal DAG to produce input-dependent componentwise box radii $\mathbf r_b$ and $\ell_2$ ball radius $\varrho_b$ covering every coordinate of $(o_b, q^{\rm out}_b)$.

`[Def]` The **block tolerance predicate** is parameterized jointly by inputs and outputs:

$$
\mathrm{Tol}_{M,b}\bigl(\mathrm{In}(b);\;\mathrm{Out}(b)\bigr) \iff \mathrm{BB}\bigl((o_b, q^{\rm out}_b),\;(\widehat o_b, \widehat q^{\rm out}_b);\;\mathbf r_b,\;\varrho_b\bigr) \;\land\; \mathrm{Out}(b) \in \mathrm{Cap}_b \;\land\; \mathrm{Fin}\bigl(\mathrm{In}(b), \mathrm{Out}(b)\bigr) \;\land\; \mathrm{In}(b) \in \mathrm{Rng}_b,
$$

where $\mathrm{BB}(\hat v, v; \mathbf r, \varrho) \iff \sum_i (|\hat v_i - v_i| - r_i)_+^2 \le \varrho^2$ is the box-plus-ball certificate (`[Lean: StdAlgOps.l2_rel_int]`), and $\mathrm{Cap}_b$, $\mathrm{Fin}$, and $\mathrm{Rng}_b$ enforce range-certificate output caps, finiteness, and overflow-free integer embedding domains (`[Lean: BoundedLift.signedLift_unique_of_bounded]`, [Spec 06](06-tolerance-and-cheats.md) §2.1).

`[Def]` **Block defect indicator.** Each active block $b \in B$ has a binary defect indicator

$$
V_b = \begin{cases}
1, & \text{if } \exists\,\text{object} \in \mathrm{In}(b) \cup \mathrm{Out}(b) \text{ with value } \bot \text{ or non-canonical/ill-typed encoding}, \\
\mathbf 1\bigl[\neg \mathrm{Tol}_{M,b}(\mathrm{In}(b);\;\mathrm{Out}(b))\bigr], & \text{otherwise}.
\end{cases}
$$

`[Def]` **Block-level locality.** $V_b$ depends exclusively on committed global data ($M_{\rm meta}$, $W_*$, and row $b$ of $\mathsf{Sl}$) together with $\mathrm{Ref}(b)$ (`[Lean: StdAlg.verdict_determined]`). When $b$ is opened in-circuit, the local gadget opens every object in $\mathrm{In}(b) \cup \mathrm{Out}(b)$ with its $H_{\rm in}$ Merkle authentication path to $R$, while $\mathrm{Wcl}(b)$ is bound via HCO to $\mathsf C_W$ (opened in-circuit from `model_id`). For $\mathrm{HEAD}(t)$, $F^\sharp_{M,\mathrm{HEAD}(t)}(x^{\rm fn}(t))$ produces reference logits $z_t$ and radii $\mathbf r^{\rm lm}_t$ directly in-circuit; $V_{\mathrm{HEAD}(t)} = 1$ iff $x^{\rm fn}(t)$ is $\bot$ or malformed, whereas LM-head numerical or sampling deviations enter $\rho$ via the generation predicate $S_t$ (§5.4), which depends solely on global data and $\mathrm{Ref}(\mathrm{HEAD}(\mathrm{pos}(t)))$.

### 4.3 Semantics of $\bot$ and Malformed Local Values

`[Def]` An unqueried trajectory leaf ($\bot$), a `NaN`/`Inf` float, an out-of-range activation, or a non-canonical local byte encoding sets $V_b = 1$ on every block referencing that leaf and charges those blocks' full computational weight toward the stratum bad-cost ratio; it **never** falsifies the global predicate $E(X,w)$. Consequently, planting random unqueried digests in unopened leaves creates no soundness gap outside the spot-check threshold: if the affected blocks' lineage-weighted cost stays below $\sigma$, the threshold relation explicitly tolerates them; if their cost reaches $\sigma$, stratified sampling hits a doomed lineage with the probability proved in [Spec 02](02-soundness.md), forcing the prover to exhibit an unqueried $H_{\rm in}$ preimage. Because an attention block reads all KV objects in its causal/sliding window, a single $\bot$ KV leaf marks all subsequent attention blocks in that layer bad, making the relation strictly more conservative without affecting completeness: honest executions under `H-contract` produce zero $\bot$, non-finite, or out-of-range values ([Spec 04](04-completeness.md) Corollary 3.1).

### 4.4 Cost Accounting, Complete Enumeration, and Stratification

`[Def]` For fixed $(X, w)$, the active block set $B = \{s \in \mathcal U : \mathsf{val}_s = 1\}$ is determined by the private slot table $\mathsf{Sl}$ embedded inside a fixed-size public slot universe $\mathcal U = \{0, \dots, U_{\rm slot}-1\}$ determined solely by `profile`. Generic slot indices reveal neither layer count, hidden width, expert count, nor active MoE routing. All slot validity bits, node identities, MoE routes, costs, and stratum prefix sums belong to global columns committed in $\mathsf{cm}_{\rm glob}$ **before** the sampling challenge is derived ([Spec 02](02-soundness.md) §4.1), preventing post-challenge adaptive graph selection. Data-dependent MoE layers commit the selected $K_e$-expert execution graph up front in $\mathsf{Sl}$, while routing-score compliance is checked locally by the router block's $V_b$ ([Spec 05](05-computation-model.md) §2.3, [Spec 06](06-tolerance-and-cheats.md) §3.6).

`[Def]` Every active block $b$ carries a positive integer logical workload cost $c_b = \mathsf{Cost}_M(\nu_b, e_b) \in \mathbb N_{>0}$ looked up from the committed cost table ([Spec 05](05-computation-model.md) §1.5 `E3`), with total cost $C = \sum_{b \in B} c_b \le C_{\max}$. Provers cannot reprice blocks, omit active nodes, duplicate honest blocks, or assign zero cost.

`[Def]` **No prefix-cache or history exemptions.** Logical computation for the current turn encompasses all input positions (including prefix-cache hits, decrypted replayed hidden reasoning $\mathsf{hist}_{\rm prev}$, and in-turn tool return segments) plus all generated positions. Every such block enters $B$ with full logical weight $c_b$, preventing bad blocks in earlier turns or cached pages from escaping inspection. Conversely, speculative draft tokens rejected by the target model are excluded from $u$, $B$, and $G$ ([Spec 05](05-computation-model.md) §3.5) so a prover cannot pad denominators with discarded draft work.

`[Def]` **Public stratification $\Pi$.** Each protocol version fixes a public stratification rule $\Pi$ computed in-circuit from $\mathsf{Sl}$ that maps every active slot to a stratum $j \in \{1, \dots, |\Pi|\}$, partitioning $B = \bigsqcup_{j=1}^{|\Pi|} B_j$ with stratum totals $C_j = \sum_{b \in B_j} c_b$ and $C = \sum_j C_j$. Setting $|\Pi| = 1$ recovers a single unstratified computation pool.

`[Def]` **Default two-stratum rule $\Pi_2$.** Under the default rule $\Pi_2$, all slots at sequence position $p$ belong to the **generation stratum** ($j = 2$) if position $p$'s final activations produce logits consumed by a generation step in $G$, and to the **input stratum** ($j = 1$) otherwise. `[Paper]` Under $|\Pi|=1$, a request with $10{,}000$ prompt tokens and $500$ generated tokens places $\approx 95\%$ of total FLOPs in prompt prefill; a provider substituting an unverified tiny model for all $500$ decode steps incurs $d_C \approx 0.05 < \sigma$ while passing every sampling check $S_t$ relative to those corrupted generation columns. Under $\Pi_2$, decode columns have bad fraction $d^{(2)}_C = 1 \ge \sigma$ in the generation stratum and are caught with overwhelming probability. With default thresholds $\sigma = \rho = 0.5$ at $(Q = 2^{64}, \varepsilon = 2^{-40})$, $\Pi_2$ requires $k_{1,1} = k_{1,2} = k_2 = 105$ draws across the input stratum, generation stratum, and generation-step pool—$315$ spot checks in total (`[Lean: RoundByRound.three_pool_default]`, [Spec 02](02-soundness.md) §8).

`[Def]` **Empty executions and unit spaces.** $N = 0 \implies B = \varnothing$ ($C = 0$), and $N > 0 \implies C > 0$ (§6.3). For each stratum $j$, the weighted cost-unit space $I_{C,j} = \{(a, i) : a \in A_j,\; 0 \le i < c_a\}$ of cardinality $|I_{C,j}| = C_j$ drives exact cost-proportional sampling ([Spec 02](02-soundness.md) §4.2), where each unit uniquely identifies its enclosing block and cost atom ([Spec 05](05-computation-model.md) §1.5 `E5`).

### 4.5 Billing Units, Cost Atoms, Lineage, and Thresholds

Fine-grained per-GEMM observation is necessary for tight floating-point error certificates, but naive per-GEMM cost accounting suffers from a zero-operand vulnerability: skipping a transformer layer and emitting zeros leaves downstream zero-input GEMMs algebraically valid while charging only the inexpensive producer-tail normalization ([Spec 05](05-computation-model.md) §1.6, [Spec 06](06-tolerance-and-cheats.md) §5.1). FlashProve-Next closes this gap via **fine-grained block verification, coarse sublayer billing units, and consumer-side lineage accountability**. This subsection is the canonical definition of the threshold bad-cost fractions referenced throughout [Specs 02](02-soundness.md), [05](05-computation-model.md), and [06](06-tolerance-and-cheats.md).

`[Def]`
1. **Cost atoms.** Each block $b \in B$ decomposes by public rule into positive-weight cost atoms $a \in A(b)$ with $c_a \in \mathbb N_{>0}$ and $\sum_{a \in A(b)} c_a = c_b$. An attention block $\mathrm{ATT}(t, \ell)$ splits its core quadratic work into one **key atom** $(t, \tau, \ell)$ of weight $c^{\rm key}$ per attended key position $\tau$, plus one **self atom** of weight $c^{\rm self}$ ([Spec 05](05-computation-model.md) §1.5 `E3`, `E5`). Letting $b(a)$ denote the host block of atom $a$ and $A_j = \bigcup_{b \in B_j} A(b)$, we have $\sum_{a \in A_j} c_a = C_j$.
2. **Billing units and operator-ownership check sets.** A billing unit $U$ groups a logical sublayer of operators (for a dense layer, the attention sublayer $\mathsf A(t,\ell)$ and MLP sublayer $\mathsf M(t,\ell)$; for MoE, MLA, and recurrent layers see [Spec 05](05-computation-model.md) §1.6). Its check set is defined by **operator ownership**:

$$
\mathrm{Chk}(U) = \bigl\{b \in B : \mathrm{ops}(b) \cap U \neq \varnothing\bigr\}, \qquad V_U = \max_{b \in \mathrm{Chk}(U)} V_b.
$$

Because pre-attention normalization $\mathrm{N1}_\ell$ executes at the tail of $\mathrm{DOWN}(t, \ell-1)$, $\mathrm{DOWN}(t, \ell-1) \in \mathrm{Chk}(\mathsf A(t, \ell))$. Every atom $a$ has an owning unit $U(a)$ satisfying $b(a) \in \mathrm{Chk}(U(a))$.
3. **Bounded-depth lineage.** For a public profile constant $h$ (default $h = 3$, [Spec 06](06-tolerance-and-cheats.md) §5.1.6), $\mathrm{Lin}_h(a)$ is the backward closure of $U(a)$ up to depth $h$ along the public inter-unit dependency edges ([Spec 05](05-computation-model.md) §1.6); for an attention key atom $(t, \tau, \ell)$, $\mathrm{Lin}_h(a)$ additionally includes the depth-$h$ closure of the unit producing $(k,v)(\tau, \ell)$. Bounded depth $h = 3$ prices zero-chain and KV-zeroing attacks at full sublayer/consumer FLOPs while keeping per-draw verification constant-sized; arbitrary-depth prefix-collapse constructions require $h = \infty$ and lie outside the threat model scope.
4. **Bad atoms and stratum defect fractions.** Define (`[Lean: ProtocolBridge.AtomBad]`, `[Lean: ProtocolBridge.atomBad_of_mem_lineage]`)

$$
V_a = \max_{U \in \mathrm{Lin}_h(a)} V_U, \qquad d^{(j)}_C(X, w) = \begin{cases} \displaystyle \frac{\sum_{a \in A_j} c_a V_a}{C_j}, & C_j > 0, \\ 0, & C_j = 0, \end{cases} \qquad d_C(X, w) = \frac{\sum_{a} c_a V_a}{C} \quad (C > 0).
$$

`[Paper]` Since $U(a) \in \mathrm{Lin}_h(a)$ and $b(a) \in \mathrm{Chk}(U(a))$, we have $V_a \ge V_{U(a)} \ge V_{b(a)}$, so $d^{(j)}_C \ge d^{(j),\rm blk}_C := \sum_{b \in B_j} c_b V_b / C_j$. Because $d_C = \sum_j (C_j / C) d^{(j)}_C$ is a convex combination, $\max_j d^{(j)}_C \ge d_C \ge d^{\rm blk}_C$. Enforcing $d^{(j)}_C < \sigma$ across all strata is strictly stronger than an unstratified block-level threshold (recovering it when $|\Pi|=1$, $h=0$, and $|\mathrm{Chk}(U)|=1$) while preserving completeness because honest executions have $V_b = 0$ everywhere.
5. **Locality Condition `(L)` (`P-loc` in [Spec 02](02-soundness.md)).** Let $\mathrm{Ref}(\mathrm{Lin}_h(a)) = \bigcup_{U \in \mathrm{Lin}_h(a)} \bigcup_{b \in \mathrm{Chk}(U)} \mathrm{Ref}(b)$. Then $V_a$ depends solely on committed global data and $\mathrm{Ref}(\mathrm{Lin}_h(a))$. Whenever atom $a$ is sampled, the circuit opens every object in $\mathrm{Ref}(\mathrm{Lin}_h(a))$ across at most $B_h$ blocks (padded to the public profile bound $B_h$, [Spec 05](05-computation-model.md) §1.6). Accepting a sampled atom whose lineage contains a $\bot$-induced bad block requires exhibiting an unqueried $H_{\rm in}$ preimage (`[Lean: ProtocolBridge.compiled_cs_doomed_of_bad_sample]`).

---

## 5. Token Generation, Constrained Decoding, and Keyed Bucketed Sampling

### 5.1 Logical Generation Steps

`[Def]` Each model release commits a public sampling algorithm and deterministic schema-to-mask compilation rule; secret programmable sampling policies are disallowed, while the sampling PRF key $K_{\rm prf}$, hidden prompt prefix, and raw logits remain private.

`[Def]` The generation step set $G = \{0, \dots, N-1\}$ indexes every target-model generated token in strict logical generation order across all generation segments of the turn. $G$ includes hidden `CoT` tokens, visible assistant tokens, generated tool-call arguments, model-generated refusal tokens, end-of-turn control tokens, and tokens completing a stop sequence that are subsequently trimmed from visible text. $G$ excludes prompt tokens, hidden prompt injections, tool return segments, and speculative draft tokens rejected by target verification. Every target token accepted during coupled speculative decoding occupies a unique index $t \in G$ (`[Lean: Speculative.specLoop_eq_seqGen]`).

### 5.2 Constrained Decoding Automaton

`[Def]` Structured output and strict tool-argument schemas compile deterministically into a byte/token automaton with state column $q^{\rm fsm}_t$ ([Spec 05](05-computation-model.md) §4.2):

$$
q^{\rm fsm}_0 = \mathsf{Init}_M\bigl(\mathsf{schema}(x)\bigr), \qquad A_t = \mathsf{Allow}_M(q^{\rm fsm}_t), \qquad q^{\rm fsm}_{t+1} = \delta_{\rm fsm}(q^{\rm fsm}_t,\;u_t).
$$

`[Def]` $\mathrm{Chain}$ enforces $u_t \in A_t$ and the valid transition $(q^{\rm fsm}_t, u_t, q^{\rm fsm}_{t+1})$ at **every** generation step $t \in G$, permitting `EOS` exclusively in accepting states. Schema-mask violations are never relaxed into $\rho$: structured outputs must be syntactically valid at every step. If $A_t = \varnothing$, execution follows the versioned failure/refusal transition rather than selecting an unconstrained token.

### 5.3 Keyed Bucketed Sampling Noise

`[Def]` Randomized decoding uses the sorting-free **bucketed keyed sampling rule** $\mathsf{Samp}^{\rm bk}_{\beta_s}$ (`fpn-samp-bk-v3`, [Spec 06](06-tolerance-and-cheats.md) §3.1) with public bucket width $\beta_s$ (default $\beta_s = 32$). The per-turn sampling context is

$$
\mathsf{ctx} = \mathsf{Enc}\bigl(\texttt{dom}_\xi,\;v_{\rm rel},\;e,\;\mathsf{dig}(x^{\rm core}_{\rm pub}),\;N_c,\;N_s,\;\mathsf{sess},\;i\bigr),
$$

where $i$ is the conversation turn index. For generation phase/role tag $d_t$, step $t \in G$ derives two domain-separated Poseidon2 keyed-sponge PRF streams (`bkt` and `inv`):

$$
g_{t,b} = \mathsf G_{b \bmod 16}\!\Bigl(\mathsf{PRF}_{K_{\rm prf}}\bigl(\mathsf{Enc}(\mathsf{ctx},\;d_t,\;t,\;\texttt{bkt},\;\lfloor b/16 \rfloor)\bigr)\Bigr), \qquad U_t = \mathsf U\!\Bigl(\mathsf{PRF}_{K_{\rm prf}}\bigl(\mathsf{Enc}(\mathsf{ctx},\;d_t,\;t,\;\texttt{inv})\bigr)\Bigr),
$$

for bucket index $b \in [0, \lceil V_{\max} / \beta_s \rceil)$. Each width-12 Poseidon2 PRF call outputs $8$ Goldilocks field elements; $\mathsf G_r$ splits the low $48$ bits of each element into two $24$-bit indices (covering $16$ buckets per call) to look up the committed discrete Gumbel table for Gumbel-max bucket selection, while $\mathsf U$ extracts the low $48$ bits of the first output element as a $48$-bit fixed-point uniform variate for intra-bucket inverse-CDF selection ([Spec 03](03-zero-knowledge.md) §5.4, [Spec 06](06-tolerance-and-cheats.md) §3.1). Because noise depends solely on $(\mathsf{ctx}, d_t, t, b)$ and never on ungenerated tokens or draft proposals, coupled speculative decoding produces identical trajectories to sequential decoding (`[Lean: Speculative.specLoop_eq_seqGen]`).

### 5.4 Generation Predicate

`[Def]` At each generation step $t \in G$, the generation compliance indicator and defect fraction are

$$
S_t = \mathrm{Samp}^{\rm bk}_{\beta_s}\bigl(z_t,\;\mathbf r^{\rm lm}_t,\;A_t,\;(g_{t,b})_b,\;U_t;\;u_t\bigr), \qquad d_G(X, w) = \begin{cases} \displaystyle \frac{1}{N} \sum_{t=0}^{N-1} \mathbf 1[\neg S_t], & N > 0, \\ 0, & N = 0, \end{cases}
$$

where $\mathrm{Samp}^{\rm bk}_{\beta_s}$ is the certified relaxed bucketed sampling relation (`[Lean: BucketSampling.bucket_sampling_complete]`, `[Lean: BucketSampling.boundary_pair_sound_complete]`, [Spec 06](06-tolerance-and-cheats.md) §3.2) incorporating witness-free tight nucleus envelopes (`[Lean: BucketSampling.nucleus_sandwich]`) and sampler rounding contracts. Reference logits and per-coordinate radii $(z_t, \mathbf r^{\rm lm}_t) = F^\sharp_{M,\mathrm{HEAD}(\mathrm{pos}(t))}(x^{\rm fn}(\mathrm{pos}(t)))$ are **recomputed inside the circuit** from the committed final-norm activation $x^{\rm fn}(\mathrm{pos}(t))$ and HCO-bound LM-head tiles, eliminating any separate "sampling-only logits" witness. Under `H-contract`, honest fp32 engine logits $\hat z_t$ satisfying $|\hat z_t - z_t| \le \mathbf r^{\rm lm}_t$ always pass $S_t$ (`[Lean: BucketSampling.bucket_sampling_complete]`); if $x^{\rm fn}(\mathrm{pos}(t))$ is $\bot$ or malformed, $S_t = \mathrm{false}$. The denominator $N = |G|$ counts all target-generated tokens, including hidden `CoT` tokens.

### 5.5 Nonce Protocol and Server Randomness

`[Def]` **Client freshness nonce $N_c$.** Clients sample a fresh $\ge 128$-bit nonce $N_c$ per request inside $x_{\rm pub}$. $N_c$ is mandatory for soundness: without $N_c$, a provider receiving a repeated prompt could reuse a prior $(N_s, y, \pi)$ tuple verbatim and pass verification with zero new computation.

`[Def]` **Per-attempt server randomness $N_s$.** For every request attempt (and any retry without an API idempotency key), the provider draws a fresh $256$-bit CSPRNG value $N_s$ returned publicly in $\mathsf{env}$. Global or sharded monotonic counters are prohibited because counter differences across consecutive responses leak cross-tenant request volume. Across $N_{\rm req}$ requests in an epoch, $N_s$ values collide with probability at most $N_{\rm req}^2 / 2^{257}$ regardless of how a malicious client chooses $N_c$, neutralizing chosen-nonce common-random-number logit-probing attacks ([Spec 03](03-zero-knowledge.md) Lemma 5.1).

`[Def]` **Provider $N_s$ grinding boundary and optional `nonce-ticket` mode.** After computing prefix logits, a key-holding provider could evaluate inexpensive PRF calls across multiple candidate $N_s$ values to bias output token selection. Such selection still delivers a fully computed, rule-compliant trajectory for the published $N_s$ and lies outside the core compute-integrity boundary. Deployments requiring pre-committed server randomness enable the optional `nonce-ticket` capability: the provider issues a hiding commitment $T = H_{\rm in}(\texttt{dom}_{\rm tkt}, N_s)$ prior to seeing the request (e.g., in the prior turn's $\mathsf{env}$), the request echoes $T$, and $\mathrm{Chain}$ verifies $H_{\rm in}(\texttt{dom}_{\rm tkt}, N_s) = T$, leaving observable abort ([Spec 04](04-completeness.md)) as the provider's only degree of freedom.

`[Def]` Stateful clients verify that $N_c$ and $\mathsf{St}_{\rm prev}$ match their local session record and advance their stored state anchor; stateless third-party verifiers check cryptographic binding relative to the statement's anchors without global anti-replay state. `[Assume]` Keyed-sponge PRF pseudorandomness holds for honestly generated secret $K_{\rm prf}$ and is invoked exclusively in the zero-knowledge proof ([Spec 03](03-zero-knowledge.md)), never in soundness.

---

## 6. Global Chain, Stop Automaton, and Multi-Turn State Chain

### 6.1 Conjuncts of $\mathrm{Chain}(X,w)$

`[Def]` The global structural predicate $\mathrm{Chain}(X,w)$ is the conjunction of six deterministic global circuit checks, none of which is subject to $(\sigma, \rho)$ spot-check slack:

1. **Request and segment integrity:** Canonicalization $x = \mathsf{Canon}_{v_{\rm cap}}(x_{\rm pub})$, typed segmented rendering and tokenization (§2), hidden prompt injections, and tool call/return byte sequences and ordering match.
2. **Complete slot-table enumeration:** Monotone validity prefix, deterministic graph-topological stepping, legal start/terminal nodes, committed cost lookup, stratum tags, prefix sums with $C \le C_{\max}$, and deterministic port wiring $\mathsf{Addr}$ (§4.1, [Spec 05](05-computation-model.md) Lemma 1.1).
3. **Token provenance and grammar compliance:** Input tokens match rendered segments; the $t$-th generated token equals $u_{\mathrm{pos}(t)+1}$ and satisfies the constrained-decoding transition $(q^{\rm fsm}_t, u_t, q^{\rm fsm}_{t+1})$ with $u_t \in A_t$ at every step (§5.2).
4. **Visible projection and usage counters:** $\mathsf{Visible}(u, z_{\rm think}, z_{\rm tool}) = y^{\rm core}$, and all public token/cache counters in $\ell_{\rm api}$ match exact private counts.
5. **Stop-automaton exactness:** The byte-level stop state column $q^{\rm stop}_t$ and public finish reason satisfy §6.2, with empty or interrupted responses following §6.3.
6. **Multi-turn state chain:** State anchors $\mathsf{St}_{\rm prev}$ and $\mathsf{St}_{\rm new}$ satisfy §6.4, and $H_{\rm in}(\texttt{dom}_{\rm tkt}, N_s) = T$ holds when `nonce-ticket` is active.

### 6.2 Byte-Level First-Match Stop Automaton

`[Def]` Let $\mathsf{bs} = (\mathsf{bs}_1, \mathsf{bs}_2, \dots)$ be the canonical output byte stream over stop-eligible content blocks (fixed by $v_{\rm cap}$, defaulting to visible text blocks while excluding hidden `CoT` and tool arguments), formed by concatenating generated token bytes in logical order. Given request stop sequences $s_1, \dots, s_m$, define the earliest match endpoint and longest matching sequence ending there:

$$
e^\star = \min\bigl\{e \ge 1 : \exists\,\ell \in \{1,\dots,m\},\;\mathsf{bs}_{e - |s_\ell| + 1 .. e} = s_\ell\bigr\}, \qquad s^\star = \arg\max_{s_\ell : \mathsf{bs}_{e^\star - |s_\ell| + 1 .. e^\star} = s_\ell} |s_\ell|.
$$

Let $t^\star \in G$ be the generation step whose token emits byte $e^\star$. An Aho–Corasick automaton state column $q^{\rm stop}_t$ tracks byte-by-byte transitions across intra-token hits, cross-token boundary hits, and overlapping stop strings ([Spec 05](05-computation-model.md) §4.3). Generation halts at $t^\star$; visible text is truncated immediately before the start byte $e^\star - |s^\star| + 1$ (which may lie in an earlier token); token $u_{t^\star}$ remains in $u$ and $G$ for full spot-check verification; and the response reports finish reason `stop_sequence` echoing $s^\star$ (with mid-UTF-8-codepoint truncation and token-accounting rules fixed by $v_{\rm cap}$).

`[Def]` **Deterministic single-step event priority.** Within generation step $t$, stopping conditions follow a strict total order:
0. **Pre-sampling guard (prior to sampling $u_t$):** If cumulative turn output tokens (including hidden `CoT`) reach the output budget (including `max_tokens = 0` at $t = 0$) or sequence position reaches the context limit, $u_t$ is not sampled and generation terminates with `max_tokens` / context-limit finish reason.
1. **Endogenous post-sampling control token:** Sampling end-of-turn $\to$ `end_turn`; closing a client tool call $\to$ `tool_use`; reaching in-turn server tool iteration cap $\to$ `pause_turn`; refusal control marker $\to$ `refusal`.
2. **Stop-sequence byte match in $u_t$:** First byte match $e^\star$ reached inside $u_t$ $\to$ `stop_sequence`.
3. **Otherwise:** Advance to step $t + 1$.

Because special control tokens emit zero stop-eligible text bytes, priorities 1 and 2 are mutually exclusive; checking budgets prior to step $t+1$ ensures that if the final budgeted token simultaneously completes a stop sequence, `stop_sequence` takes precedence. Neither continuing past $t^\star$ nor halting early without a satisfied guard is valid, and model-generated refusal tokens undergo identical compute and sampling checks.

### 6.3 Empty Responses, External Classifiers, and Incomplete Streams

`[Def]` An empty generation set $N = 0$ (which forces $B = \varnothing$ and $C = 0$) is valid exclusively in three publicly decidable branches:
- **(a)** The pre-sampling guard triggers at $t = 0$ (e.g., `max_tokens = 0`);
- **(b)** A public policy finish code declares that an external pre-inference safety classifier intercepted the request before invoking the committed target model (external classifiers are separate uncommitted models and are never misrepresented as target-model refusals); or
- **(c)** Client cancellation or stream interruption occurs prior to the first generated token under the public state machine of [Spec 04](04-completeness.md) §7.

In all three branches, validity is a public function of $(x_{\rm pub}, y)$ leaking zero private state, the proof checks normalization and $\mathrm{Chain}$, all leaves of $R$ are canonical padding leaves blinded by the fresh root salt $\varsigma_R$ (§4.1), and the non-empty terminal-slot constraint in [Spec 05](05-computation-model.md) §1.5 `E2` is inactive ($\mathsf{val}_0 = 0$). Mid-stream `content_filter` cutoffs, cancellations, and timeouts follow [Spec 04](04-completeness.md) §7, certifying the exact prefix of target-model steps $G$ and blocks $B$ actually delivered.

### 6.4 Acyclic Multi-Turn State Chain

`[Def]` **Session identifier and genesis anchor.** A conversation derives $\mathsf{sess} = H_{\rm in}(\texttt{dom}_{\rm sess}, N_c^{(1)})$ from the first turn's client nonce (or binds an API-level conversation ID per $v_{\rm cap}$), with publicly computable genesis anchor $\mathsf{St}_0 = H_{\rm in}(\texttt{dom}_{\rm gen}, v_{\rm rel}, \mathsf{sess})$.

`[Def]` **Turn-$i$ anchor and acyclic envelope construction.** Let $\mathsf{hist}_i$ encode the hidden turn-$i$ state carried forward to subsequent turns (hidden `CoT` blocks and private server tool outputs per $v_{\rm cap}$) and let $r_i$ be $\ge 256$ bits of fresh randomness. The turn-$i$ state anchor commits strictly to the core payload $y^{\rm core}_i$:

$$
\mathsf{St}_i = H_{\rm in}\Bigl(\texttt{dom}_{\rm st},\;v_{\rm rel},\;\mathsf{model\_id}_i,\;\mathsf{sess},\;i,\;\mathsf{St}_{i-1},\;H_{\rm in}(x^{\rm core}_i),\;H_{\rm in}(y^{\rm core}_i),\;H_{\rm in}(\mathsf{hist}_i),\;r_i\Bigr).
$$

Response construction proceeds acyclically in five steps:

$$
y^{\rm core}_i \;\longrightarrow\; \mathsf{hist}_i \;\longrightarrow\; \mathsf{St}_i \;\longrightarrow\; \mathsf{blob}_i = \mathsf{AEAD.Enc}_{K_{\rm wrap}}\bigl(\mathsf{aad}_i,\;(\mathsf{hist}_i, r_i)\bigr) \;\longrightarrow\; \mathsf{env}_i \;\longrightarrow\; y_i = (y^{\rm core}_i, \mathsf{env}_i),
$$

with associated data $\mathsf{aad}_i = \mathsf{Enc}(\mathsf{sess}, i, \mathsf{St}_i, \text{API-authenticated tenant ID})$. Because $\mathsf{St}_i$ precedes $\mathsf{blob}_i$ and $\mathsf{env}_i$, the commitment DAG is strictly acyclic, while the full response $y_i$ (including $\mathsf{blob}_i$) is bound in $X_0$. Turn $i$'s $\mathrm{Chain}$ verifies $\mathsf{St}_{\rm new} = \mathsf{St}_i$ inside the circuit. AEAD encryption itself is **not** proved in-circuit: $\mathsf{blob}_i$ is purely an encrypted transport carrier whose corruption affects only next-turn completeness, never soundness.

`[Def]` **Turn-$(i+1)$ anchor consumption.** Statement $X_0$ at turn $i+1$ binds $\mathsf{St}_{\rm prev} = \mathsf{St}_i$ (echoed by the client or retrieved via `previous_response_id` and verified client-side). The witness opens $\mathsf{St}_{\rm prev}$ inside $\mathrm{Chain}$ to verify:
- **(a)** Matching $\mathsf{sess}$, strictly incremented turn counter $i \mapsto i+1$, and `model_id` transition permitted by $v_{\rm cap}$;
- **(b)** Consistency of public message history in $x_{i+1}$ with $H_{\rm in}(x^{\rm core}_i)$ and $H_{\rm in}(y^{\rm core}_i)$ (preserving the exact sequence of visible/`thinking`/`redacted_thinking`/`reasoning` blocks); and
- **(c)** Consistency of $\mathsf{hist}_i$ with $H_{\rm in}(\mathsf{hist}_i)$ and verbatim segmented rendering of $\mathsf{hist}_i$ into turn $(i+1)$'s input prefix (entering the input stratum at full cost weight $c_b$ under §4.4).

`[Def]` **External finite induction without in-circuit recursion.** Turn $(i+1)$'s proof certifies consistency relative to $\mathsf{St}_{\rm prev}$. Full multi-turn chain validity follows by verifying each individual turn proof $1, \dots, i+1$ on the CPU verifier and checking $\mathsf{St}_{\rm prev}^{(1)} = \mathsf{St}_0$ and $\mathsf{St}_{\rm new}^{(r)} = \mathsf{St}_{\rm prev}^{(r+1)}$. All turn proofs share the single service-lifetime query budget $Q$ ([Spec 02](02-soundness.md) §3), avoiding in-circuit recursive proof verification and in-circuit random-oracle assumptions. Stateful clients that record their latest verified anchor obtain rollback and fork resistance; cross-tenant anchor splicing is blocked both at the API layer (via tenant-bound $\mathsf{aad}_i$ under `[Assume]` API authentication) and at the relation layer (via $\mathsf{sess}$ and history-digest checks).

---

## 7. Assembled Relations and Verification Semantics

`[Def]` Let $E(X, w)$ denote canonical encoding and global structural validity: $\omega_M$ opens `model_id` consistently, $\mathsf C_{\rm cfg}$ opens to $M^{\rm vol}$, $W_*$ decodes uniquely under the verified $\pi_{\rm setup}$ (§1), public request/response canonicalizations hold, and all global columns ($u$, $\mathsf{Sl}$, $q^{\rm fsm}$, $q^{\rm stop}$, and state-anchor openings) are canonically encoded within `profile` bounds with zero $\bot$ entries. Local trajectory objects in $\mathcal O$ are never inspected by $E(X,w)$; they enter strictly through $V_b$ (propagated to $V_a$) and $S_t$.

The **strict all-blocks relation** and **cost-stratified threshold relation** are

$$
\mathcal R_{\rm all}(X, w) = E(X, w) \;\land\; \mathrm{Chain}(X, w) \;\land\; \bigwedge_{b \in B} (V_b = 0) \;\land\; \bigwedge_{t \in G} S_t,
$$

$$
\boxed{\mathcal R_{\Pi,\sigma,\rho}(X, w) = E(X, w) \;\land\; \mathrm{Chain}(X, w) \;\land\; \max_{1 \le j \le |\Pi|} d^{(j)}_C(X, w) < \sigma \;\land\; d_G(X, w) < \rho.}
$$

`[Def]` Thresholds $0 < \sigma, \rho \le 1$ are public rational numbers checked via exact integer cross-multiplication; $d^{(j)}_C(X,w)$ is the lineage-weighted bad-atom fraction of §4.5 (`[Lean: ProtocolBridge.ExtractedBad]`), satisfying $\mathcal R_{\Pi,\sigma,\rho}(X,w) \implies d^{\rm blk}_C(X,w) \le d_C(X,w) < \sigma$. Rather than opening every block of the execution, FlashProve-Next spot-checks stratified random samples and proves in [Spec 02](02-soundness.md) (`[Lean: ProtocolBridge.fpn_protocol_lifetime_soundness]`) that any lifetime adversary causing verification to accept a statement whose straight-line extracted extended witness $w_R$ violates $\mathcal R_{\Pi,\sigma,\rho}(X, w_R)$ succeeds with probability at most $\varepsilon$ (default $2^{-40}$ across $Q \le 2^{64}$ lifetime challenge queries).

`[Def]` **Operational meaning and explicit scope.** Verification establishes that the provider holds—in the sense of straight-line extraction from its lifetime hash queries ([Spec 02](02-soundness.md) §2)—a committed trajectory under `model_id` producing $y$ whose global chain is exact and whose per-stratum computation defects and generation-step violations lie strictly below $(\sigma, \rho)$. It does not certify physical GPU provenance, external tool truthfulness, provider dollar expenditure, or output distribution unbiasedness against $N_s$ grinding by the key holder (§5.5); under adaptive zero knowledge ([Spec 03](03-zero-knowledge.md)), the verifier learns nothing beyond this validity guarantee and standard API outputs. `[Open]` While the abstract compiled-constraint-system to round-by-round soundness bridge is mechanized in `ProtocolBridge.lean`, end-to-end mechanical extraction of the byte-level circuit builder from Lean remains outside scope.
