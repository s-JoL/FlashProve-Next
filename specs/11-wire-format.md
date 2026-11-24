# Spec 11: `FPN2` Canonical Wire Format, Profiles & Raw-Byte Fiat–Shamir Transcript (`fpn-wire-2.0`)

**Specification ID:** `fpn-wire-2.0` | **Role:** Byte-exact definition of the `FPN2` proof container, the canonical profile encoding that pins it, the Fiat–Shamir transcript that consumes its raw wire bytes, and the normative verifier check order shared by the Rust `fpn-core` verifier, the detached NumPy reference verifier, and (M4) the Lean `verifyBytes` executable.

Golden vectors: [`specs/vectors/`](vectors/). Implementations: `core/` (Rust, `fpn-core`, CLI `fpn-verify`) and `verifier/python/fpn_verify_ref/` (pure Python + optional NumPy). Both MUST agree on every golden vector, including the exact reject code. Relation 3 (§5.4) is the first end-to-end proof body: ZK-masked FracLogUp-GKR + fused two-round folded ZK-Basefold. Relation 4 (§5.5) adds a Plonkish arithmetic circuit (zerocheck + copy constraints) over the same machinery. Relation 5 (§5.6) proves full-width public-weight GEMM tiles with one inner-product sumcheck and the same batched opening. Relation 6 (§5.7) is relation 5 with the activations `X` moved out of the private witness into a separate pre-committed, ZK-blinded vector `X_com` whose root `x_root` the verifier pins, closing relation 5's case-(c) gap.

---

## 0. Design Rules

1. **Fixed size.** Every container for a given profile has exactly `C_p` bytes. Every field position is a pure function of the profile; there are **no cleartext inner length headers**, **no compression**, and **no self-describing framing**. Two proofs under the same profile are byte-for-byte the same length regardless of the hidden statement contents.
2. **Canonical bytes only.** Every byte string has exactly one accepted encoding. Non-canonical field elements, nonzero reserved bits, and nonzero padding are rejected, never normalized.
3. **Raw wire bytes in Fiat–Shamir.** The transcript absorbs the exact container bytes (section by section) with 1-byte domain tags. No JSON, no re-serialization, no language-specific object encoding ever enters a transcript.
4. **Fail closed.** The first failing check in the normative order (§6) determines the single reject code. A verifier never "repairs" input.
5. **No silent downgrade.** Profiles carry a security tier; reduced-tier profiles are rejected unless the caller explicitly opts in (`--allow-reduced-security`).

All integers are **little-endian**. `u8/u16/u32/u64` are unsigned integers of the stated width. `‖` is byte concatenation. `le_k(x)` is the `k`-byte little-endian encoding of `x`.

---

## 1. Field Encodings

`P = 2^64 − 2^32 + 1` (Goldilocks). `E = F_p[u]/(u^3 − 2)` (the cubic extension used by `flashprove_next/field.py`; limb 0 is the constant term).

| Type | Size | Encoding | Canonical iff |
|---|---|---|---|
| `Fp` | 8 B | `le_8(x)` | `x < P` |
| `Fp3` | 24 B | `le_8(a0) ‖ le_8(a1) ‖ le_8(a2)` for `a = a0 + a1·u + a2·u^2` | every limb `< P` |
| `Digest` | 32 B | four `Fp` limbs `d0..d3` (a Poseidon2 output, §3) | every limb `< P` |
| `Bytes32` | 32 B | opaque bytes | always |

Decoders MUST reject `x ≥ P` (in particular `x = P`, `x = P + 1`, `x = 2^64 − 1`) with `E_NONCANONICAL`.

Extension multiplication (reference formulas, identical to `field.py::hmul`):

```
c0 = a0 b0 + 2 (a1 b2 + a2 b1)
c1 = a0 b1 + a1 b0 + 2 a2 b2
c2 = a0 b2 + a1 b1 + a2 b0
```

Inverse via the norm map (`field.py::hinv`): `t0 = a0² − 2 a1 a2`, `t1 = 2 a2² − a0 a1`, `t2 = a1² − a0 a2`, `N = a0 t0 + 2 (a2 t1 + a1 t2)`, `a^{-1} = (t0, t1, t2) · N^{-1}`; the inverse of zero is an error.

`lagrange_eval(g_0..g_d, r)`: the unique degree-`≤ d` polynomial with `g(j) = g_j` for `j = 0..d`, evaluated at `r ∈ E`. `eq_pt(a, b) = Π_i (a_i b_i + (1 − a_i)(1 − b_i))`.

---

## 2. BLAKE2b

BLAKE2b exactly as RFC 7693, with the full 64-byte parameter block: `digest_length = nn`, `key_length = 0`, `fanout = 1`, `depth = 1`, `leaf_length = 0`, `node_offset = 0`, `node_depth = 0`, `inner_length = 0`, `salt = 0^16`, `personal = 16 bytes` (zero if absent). Implementations MUST NOT depend on a third-party crypto crate; `fpn-core` vendors its own RFC 7693 implementation (`core/src/blake2b.rs`). The Python reference uses the standard-library `hashlib.blake2b`, giving an independent cross-check.

- `B256(m) = BLAKE2b(nn = 32, personal = 0^16, m)` — used for `profile_id`.
- `H_fs(m) = BLAKE2b(nn = 64, personal = "FPN2-FS-v2" ‖ 0^6, m)` — the transcript hash.

---

## 3. Poseidon2 (Goldilocks, width 12)

Bit-exact with `flashprove_next/poseidon2.py` (Plonky3 `p3_goldilocks` width-12 instance): `t = 12`, S-box `x^7`, `R_F = 8` (4 initial + 4 final full rounds), `R_P = 22`; external layer `M_E = circ(2·M4, M4, M4)` with `M4 = circ(2, 3, 1, 1)` applied once before round 0 and after every full round; internal layer `M_I = J + diag(DIAG)`; round constants `RC_INITIAL`, `RC_INTERNAL` (added to lane 0 only), `RC_FINAL`. Constants are reproduced verbatim in both implementations and pinned by the `poseidon2_permute` golden vectors.

- `compress(l, r)`: state `= l[0..4] ‖ r[0..4] ‖ 0 ‖ 0 ‖ 0 ‖ 1` (lane 11 = `TAG_NODE = 1`); output lanes `0..4`.
- `sponge_blocks(blocks, dom)`: lanes 10 = `dom mod P`, 11 = `TAG_SPONGE = 2`, all others 0; for each 8-element block add it into lanes `0..8` and permute; output lanes `0..4` (`poseidon2.py::sponge`).
- `sponge_msg(msg, dom)`: the injective length-in-capacity sponge (`state_zk.py::sponge_trace`): lane 8 = `len(msg) mod P`, lane 10 = `dom`, lane 11 = `2`; `msg` is zero-padded to `max(1, ⌈len/8⌉)` blocks.
- Merkle trees are binary over `2^n` leaf digests; `node = compress(left, right)`. An authentication path lists siblings from the leaf level upward; at level `k`, if bit `k` of the index is 1 the running digest is the right child.

---

## 4. Fiat–Shamir Transcript `Fpn2Transcript`

State `st ∈ {0,1}^512`. Every operation hashes the previous state with a 1-byte tag:

| Operation | Tag | State update / output |
|---|---|---|
| `init(label)` | `0x00` | `st ← H_fs(0x00 ‖ le_8(|label|) ‖ label)` |
| `absorb_bytes(b)` | `0x01` | `st ← H_fs(st ‖ 0x01 ‖ le_8(|b|) ‖ b)` |
| `absorb_fp(x)` | `0x02` | `st ← H_fs(st ‖ 0x02 ‖ le_8(8) ‖ le_8(x))` (`x` canonical) |
| `absorb_fp3(a)` | `0x03` | `st ← H_fs(st ‖ 0x03 ‖ le_8(24) ‖ enc_Fp3(a))` (`a` canonical) |
| `squeeze_bytes(n)` | `0x10` | `squeeze(0x10, n)` |
| `squeeze_fp3()` | `0x11` | `o = squeeze(0x11, 48)`; limb `k = int_le(o[16k .. 16k+16]) mod P` |
| `squeeze_u192()` | `0x12` | `o = squeeze(0x12, 24)`; returns `int_le(o)` |

`squeeze(tag, n)`: output `= (H_fs(st ‖ tag ‖ le_8(n) ‖ le_8(0)) ‖ H_fs(st ‖ tag ‖ le_8(n) ‖ le_8(1)) ‖ …)[0 .. n]`, then ratchet `st ← H_fs(st ‖ 0x7F ‖ tag ‖ le_8(n))`. The 128-bit reduction in `squeeze_fp3` has statistical distance `≤ 3·2^-64` from uniform on `E`.

The container transcript label is `"FPN2/transcript/v2"` (18 ASCII bytes).

---

## 5. Container Layout

### 5.1 Header (64 bytes, all profiles)

| Offset | Size | Field | Rule |
|---|---|---|---|
| 0 | 4 | `magic` | `"FPN2"` (`46 50 4E 32`) — else `E_MAGIC` |
| 4 | 2 | `version: u16` | `= 2` — else `E_VERSION` |
| 6 | 2 | `flags: u16` | `= 0` — else `E_FLAGS` |
| 8 | 32 | `profile_id` | `= B256(canon_profile_bytes)` of a registered profile — else `E_PROFILE_ID` |
| 40 | 24 | `reserved` | all zero — else `E_RESERVED` |

### 5.2 Canonical Profile Bytes (128 bytes)

A profile carries **no architecture fields** (no layer count, width, head counts, expert counts); only public upper bounds and protocol constants. Exception (disclosed): a relation-5 / relation-6 profile's `max_hidden` is the exact GEMM tile width `K` (1024 for both registered profiles), so a relation-5 / -6 receipt header reveals `K` and, through `n_vars`, the number of tiles.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | `magic = "FPN2PROF"` |
| 8 | 2 | `profile_format: u16 = 1` |
| 10 | 32 | `name`: ASCII `[a-z0-9.-]`, ≥ 1 char, zero-padded (no nonzero byte after the first zero) |
| 42 | 1 | `security_tier: u8` (`0 = REDUCED`, `1 = STANDARD`) |
| 43 | 1 | `relation_id: u8` (`1 = HARNESS_SQ_SUMCHECK`, `2 = LLM_INFERENCE_V2`, `3 = FOLDED_GKR_LOOKUP`, `4 = FOLDED_GKR_AIR`, `5 = FOLDED_VEC_MATMUL`, `6 = FOLDED_VEC_BOUND`) |
| 44 | 2 | `security_bits: u16` (target soundness bits) |
| 46 | 4 | `max_layers: u32` (scale upper bound) |
| 50 | 4 | `max_hidden: u32` (scale upper bound; relation 5: the GEMM tile width `K`) |
| 54 | 4 | `max_seq_len: u32` |
| 58 | 4 | `max_vocab: u32` |
| 62 | 4 | `max_new_tokens: u32` |
| 66 | 1 | `n_stop: u8` (`≤ 8`) |
| 67 | 32 | `stop_ids: [u32; 8]` (slots `≥ n_stop` are zero) |
| 99 | 1 | `n_vars: u8` |
| 100 | 1 | `sc_degree: u8` |
| 101 | 2 | `n_queries: u16` |
| 103 | 8 | `c_p: u64` (exact total container length) |
| 111 | 17 | reserved, zero |

`profile_id = B256(canon_profile_bytes)`. The registry (`core/src/profile.rs`, `fpn_verify_ref/profile.py`) **pins** each `profile_id` as a hex constant; implementations MUST self-test that the recomputed hash equals the pinned value.

Registered profiles:

| Name | Tier | Relation | `C_p` | Status |
|---|---|---|---|---|
| `fpn2-harness-sq-n4` | `REDUCED` (0) | 1 | 2048 | Verifiable now; requires `--allow-reduced-security` (§7) |
| `fpn2-llm-std-v1` | `STANDARD` (1) | 2 | 8 388 608 | End-to-end folded Relation 2 (§6): verified with `--circuit-desc` (self-contained `CircuitDesc`, pinned `--circuit-root` & `--stmt-digest`, `queries >= 72`); without `--circuit-desc`, rejects with `E_BACKEND_UNAVAILABLE` |
| `fpn2-folded-gkr-v1` | `STANDARD` (1) | 3 | 163 840 | End-to-end: ZK-masked FracLogUp-GKR + fused two-round folded ZK-Basefold (§5.4); `n_vars = 12`, `sc_degree = 3`, `n_queries = 72`; id `e238db30…e986` |
| `fpn2-folded-gkr-n8` | `REDUCED` (0) | 3 | 6 144 | Same relation at `n_vars = 8`, `n_queries = 2` (carries most negative golden vectors); id `d7a3d169…c147` |
| `fpn2-folded-air-v1` | `STANDARD` (1) | 4 | 262 144 | End-to-end Plonkish circuit (§5.5): `n_vars = 14` (512 gate rows / variables), `sc_degree = 3`, `n_queries = 72`; id `f6c6b4ca…0a97` |
| `fpn2-folded-air-n12` | `REDUCED` (0) | 4 | 16 384 | Same relation at `n_vars = 12` (128 rows / variables), `n_queries = 2` (carries most relation-4 negative vectors); id `5f661ec4…ed7d` |
| `fpn2-folded-air-n20-v1` | `STANDARD` (1) | 4 | 524 288 | Production-scale relation 4: `n_vars = 20` (32 768 gate rows / variables, 64× `fpn2-folded-air-v1`), `sc_degree = 3`, `n_queries = 72`; id `966dec06…a60d` |
| `fpn2-folded-vec-n16-v1` | `STANDARD` (1) | 5 | 393 216 | Full-width GEMM tiles (§5.6): `n_vars = 16`, `max_hidden = K = 1024`, one `32 × 1024` tile (32 768 multiply-adds), `sc_degree = 3`, `n_queries = 72`; id `a48dfb0a…1327` |
| `fpn2-folded-vec-n22-v1` | `STANDARD` (1) | 5 | 524 288 | Same relation at `n_vars = 22`: 64 tiles of `32 × 1024` (2 097 152 multiply-adds), `n_queries = 72`; id `ba720668…b3ef` |
| `fpn2-folded-vec-bound-n16-v1` | `STANDARD` (1) | 6 | 393 216 | Relation 5 with pre-committed activations (§5.7): `n_vars = 16`, `max_hidden = K = 1024`, one `32 × 1024` tile, `sc_degree = 3`, `n_queries = 72`; id `9492aa70…472e` |
| `fpn2-folded-vec-bound-n22-v1` | `STANDARD` (1) | 6 | 1 048 576 | Same relation at `n_vars = 22`: 64 tiles of `32 × 1024`, `n_queries = 72`; the body (529 120 B) does not fit relation 5's 512 KiB, so `C_p = 1 MiB`; id `83bab0b6…1745` |

**Provisional security bits & PCS regime.** The `fpn2-llm-std-v1` scale bounds, stop-token ids, `security_bits = 100`, and `n_queries = 72` floor (at rate 1/8) use provisional list-decoding / Johnson-regime pricing rather than the unique-decoding bound (`(9/16)^72 ≈ 2^-60`, `(9/16)^100 ≈ 2^-83`; achieving `<= 2^-106` under unique decoding alone at rate 1/8 requires `queries >= 128`, matching the `0.645^168 <= 2^-106` unique-decoding budget of Stage 1 rate 1/4). A full round-by-round analysis of multi-size layer injection under Johnson proximity remains an explicit assumption (`A-PCS-fold`).

### 5.3 Body for `relation_id = 1` (`HARNESS_SQ_SUMCHECK`)

With `n = n_vars`, `d = sc_degree`, `Q = n_queries`, sections follow the header back-to-back:

| Section | Size | Contents |
|---|---|---|
| `STMT` | 120 | `R_tab: Digest` ‖ `N_s: Bytes32` ‖ `St_new: Digest` ‖ `S: Fp3` |
| `SUMCHECK` | `n·(d+1)·24` | for round `i = 0..n`: `g_i(0), g_i(1), …, g_i(d)` as `Fp3` |
| `FINAL` | 24 | `e: Fp3` |
| `OPENINGS` | `Q·(8 + 32n)` | for query `q`: `f_q: Fp` ‖ `sib_0 … sib_{n-1}: Digest` |
| `PAD` | `C_p − 64 − Σ` | all zero — else `E_PADDING` |

For `fpn2-harness-sq-n4` (`n = 4, d = 2, Q = 4`): `STMT` 64..184, `SUMCHECK` 184..472, `FINAL` 472..496, `OPENINGS` 496..1040, `PAD` 1040..2048.

**Relation.** The prover holds `f: {0,1}^n → F_p`. `R_tab` is the Merkle root over leaves `L_x = sponge_msg([f(x)], DOM_LEAF = 64)`, `x = Σ_k x_k 2^k` (variable 0 is the least-significant bit). The claim is `S = Σ_x f(x)^2` (embedded in `E`). Round `i` binds variable `i`.

**Transcript schedule.**

```
T = init("FPN2/transcript/v2")
T.absorb_bytes(header[0..64])
T.absorb_bytes(STMT bytes)
claim = S
for i in 0..n:
    T.absorb_bytes(round_i bytes)              # (d+1)·24 raw wire bytes
    check g_i(0) + g_i(1) == claim             # else E_SUMCHECK
    r_i = T.squeeze_fp3();  claim = lagrange_eval(g_i, r_i)
T.absorb_bytes(FINAL bytes)
check claim == e · e                           # else E_FINAL
for q in 0..Q:
    idx_q = int_le(T.squeeze_bytes(8)) mod 2^n
    check MerklePath(R_tab, sponge_msg([f_q], 64), idx_q, sib_q)   # else E_MERKLE
```

`N_s` (256-bit server randomness) and `St_new` (multi-turn state anchor) are bound only through the transcript in this relation.

**Scope (honest statement).** Relation 1 is a verification *harness*: it exercises every primitive (canonical decoding, raw-byte transcript, sumcheck, Lagrange evaluation, Poseidon2 Merkle anchors) end-to-end, but the binding `e = f̃(r)` between the final sumcheck point and the committed table is **not** enforced (that is the polynomial-commitment opening, delivered by the folded Basefold backend in Relations 2–6). It is therefore **not** a sound argument for the relation, which is exactly why its profile has tier `REDUCED` and is rejected by default.

### 5.4 Body for `relation_id = 3` (`FOLDED_GKR_LOOKUP`)

A real end-to-end argument: a ZK-masked fractional-LogUp GKR whose leaf claims and mask evaluations are all
discharged by **one** fused two-round folded ZK-Basefold opening of a single committed vector. Reference prover:
`prover/fpn_prover/backend/zk_lookup.py` (GKR in `frac_logup_gkr.py`, PCS in `folded_pcs.py`, kernels in
`kernels/goldilocks_ntt.cu`); verifiers: `core/src/folded_gkr.rs`, `verifier/python/fpn_verify_ref/folded_gkr.py`.

#### 5.4.1 Parameters and relation

From the profile: `n = n_vars` (even, `8 ≤ n ≤ 28`, matching Goldilocks 2-adicity `8·2^n = 2^(n+3) | P − 1`), `sc_degree = 3` (GKR round degree), `Q = n_queries`
(`1 ≤ Q ≤ 4096`); otherwise the relation is not wired (`E_BACKEND_UNAVAILABLE`). Derived: `s = n − 3`,
`Q4 = 2^(s−2)`, codeword length `L = 8 · 2^n` (rate 1/8), `T = n/2` fused folds, `ω = 7^((P−1)/L)`,
`ι = 7^((P−1)/4)`.

**Statement.** The prover knows `F ∈ E^(2^n)` committed under `root_0` (§5.4.5) whose query block
`q ∈ E^(Q4)` (§5.4.2) satisfies `q_j ∈ T = {0, 1, …, Q4 − 1}` for every `j` (a private range/lookup check
against the public table `T_t = t`). For `fpn2-folded-gkr-v1`: 128 private values range-checked into `[0, 128)`.

#### 5.4.2 Committed vector `F` (2^n cells, cell index bits `0..n−1`)

* **Data subcube** `D = {b : b_1 = b_3 = 1}` (`2^(n−2)` cells), addressed by *data coordinates* whose bits
  map, low to high, to cell bits `0, 2, 4, 5, …, n−1`. Aligned data-coordinate blocks:
  `[0, Q4)` `q` (queries) · `[Q4, 2Q4)` `m` (multiplicities `m_t = #{j : q_j = t}`) ·
  `[2Q4, 2Q4 + Q4/2)` `N1` · `[2Q4 + Q4/2, 3Q4)` `E1` · `[3Q4, 3Q4 + Q4/2)` `N3` · `[3Q4 + Q4/2, 4Q4)` `E3`
  (dummy-quarter numerator masks / denominators) · `[2^s, 2^s + 2^(s−1))` `A` · `[2^s + 2^(s−1), 2^(s+1))` `D`
  (additive mask numerators / denominators).
* **Libra coefficients** `g_{k,i,e}` (layer `k = 1..s`, univariate `i < k`, coefficient `e < 4`): slot
  `2k(k−1) + 4i + e`, deposited into the free bits of the subcube `{b_1 = b_3 = 0}`.
* **Every other cell** is reserved uniformly random blinding (non-doubling: `3·2^(n−2) − 2s(s+1)` cells; 2892 for
  `n = 12`). The blinding meets all four patterns of every index-bit pair, so its monomial up-sets reach every
  coefficient class of every folded layer (a clustered blinding region fails the rank check of §5.4.6).

#### 5.4.3 GKR leaves (`2^(s+1)`, lowest variable = real/mask bit)

Real leaf `j < 2^s` sits at `2j`; with quarter `qd = j >> (s−2)`, `t = j mod Q4`, `h = Q4/2`,
`t' = t mod h` and `σ = +1` for `t < h`, `−1` otherwise:
`qd = 0: (1, β − q_t)` · `qd = 1: (σ N1_{t'}, E1_{t'})` · `qd = 2: (−m_t, β − t)` · `qd = 3: (σ N3_{t'}, E3_{t'})`.
Mask leaf `v < 2^s` sits at `2v + 1`: `(A_v, D_v)` for `v < 2^(s−1)` and `(−A_{v'}, D_{v'})`, `v' = v − 2^(s−1)`,
otherwise. Mask pairs and dummy pairs cancel structurally (the same committed cell enters with both signs), so
`Σ num/den = Σ_j 1/(β − q_j) − Σ_t m_t/(β − t)`, which is `0` for all but a negligible set of `β` iff the
statement holds. The fraction tree combines adjacent pairs `(n, d) ⊕ (n', d') = (n d' + n' d, d d')`; its root must
be `(0, root_den)` with `root_den ≠ 0` (a zero leaf denominator propagates to `root_den = 0`).

#### 5.4.4 Masked GKR layers and the batched linear claim

Layer `k = 0..s` proves the level with `2^k` nodes at point `pt` (`k` coordinates) from its `2^(k+1)` children.
For `k ≥ 1` the sumcheck runs on `eq(pt, b)·(n(b) + α d(b)) + λ·G_k(b)` with the Libra mask
`G_k(x) = Σ_{i<k} g_{k,i}(x_i)`, `deg g = 3`; `G_sum = 2^(k−1) Σ_i (g_i(0) + g_i(1))` is sent before `λ` and
`g_r = G_k(r)` after the rounds. The final check is `claim = eq(pt, r)·(n + α d)(children) + λ·g_r`; the next point
is `(branch, r_0, …, r_{k−1})` and the next claim mixes the children with `branch`.

After layer `s` the verifier holds `(num, den)` at `ρ = (ρ_0, ρ_1..ρ_s)`. With `real = 1 − ρ_0`,
`low = ρ_1..ρ_{s−2}`, `low' = ρ_1..ρ_{s−3}`, `π = 1 − 2ρ_{s−2}`, `ra = ρ_1..ρ_{s−1}`,
`eqQ(qd) = eq((ρ_{s−1}, ρ_s), bits(qd))` and `T̃(low) = Σ_i 2^i low_i`:

```
num_pub = real·eqQ(0)            den_pub = real·(eqQ(0)·β + eqQ(2)·(β − T̃(low)))
y_num   = −real·eqQ(2)·m̃(low) + real·π·(eqQ(1)·Ñ1(low') + eqQ(3)·Ñ3(low')) + ρ_0·(1 − 2ρ_s)·Ã(ra)
y_den   = −real·eqQ(0)·q̃(low) + real·eqQ(1)·Ẽ1(low') + real·eqQ(3)·Ẽ3(low') + ρ_0·D̃(ra)
Y = (num − num_pub) + μ (den − den_pub) + Σ_{k=1..s} μ^(k+1) g_r(k) + μ^(s+2) z_blind
```

where `z_blind = Σ_{b ∉ D} F(b)·u(b)`, `u(b) = eq(2, b) + eq(3, b)` (two product terms with distinct slopes, so the
blinding part of every degree-2 PCS round polynomial spans all quadratics). `Y = Σ_b F(b) w(b)` for a weight
vector `w` made of subcube blocks `c·eq(z, b_free)` (fixed bits contribute `r_bit` or `1 − r_bit` to `w̃(r)`) and
the sparse Libra cells `μ^(k+1) r_{k,i}^e`; `w̃(r)` costs `O(n·(s² + 1))`.

#### 5.4.5 Fused two-round folded ZK-Basefold

`c = Möbius(F)` (monomial coefficients of `F̃`) is read as `f(X) = Σ_i c_i X^i` and encoded as `f(ω^j)`, `j < L`.
A coset leaf `j < L/4` holds `(f(ω^j), f(ω^(j+L/4)), f(ω^(j+2L/4)), f(ω^(j+3L/4)))`:
`leaf = B256_M(0x00 ‖ 4 × enc_Fp3)`, `node = B256_M(0x01 ‖ left ‖ right)` with
`B256_M = BLAKE2b(nn = 32, personal = "FPN2-MERKLE-v1" ‖ 0^2)`; `root_0` is the root over the `L/4` leaves.

Basefold sumcheck of `Y = Σ_b F(b) w(b)` (round polynomial degree 2, sent as `h(0), h(1), h(2)`), lowest variable
first. Every two rounds `(r_a, r_b) = (r_{2t}, r_{2t+1})` fold the layer-`t` codeword (`L_t = L/4^t`,
`ω_t = ω^(4^t)`) by 4:

```
v_k = f_t(x ι^k), x = ω_t^j:   s_e = Σ_k ι^(−ek) v_k,   f_e = s_e · x^(−e) / 4
f_{t+1}(x^4) = f_0 + r_a f_1 + r_b f_2 + r_a r_b f_3
```

(= two binary Basefold folds). Layers `1 .. T−1` are committed (Merkle roots on the wire); layer `T` is the
constant `F̃(r)`, sent as `FINAL`.

#### 5.4.6 Zero knowledge (scope)

* **GKR.** Every layer sumcheck with `k ≥ 1` rounds carries a fresh Libra mask committed inside `F` before any
  challenge; partial harmonic sums of intermediate tree levels are additionally hidden by the additive mask leaves
  (every node above the leaves mixes in an independent random fraction). The leaf-layer children
  `n_l, d_l` are the real-leaf numerator/denominator MLEs at `ρ'`; the dummy quarters add `π·Ñ(low')` and
  `Ẽ(low')` to them, so `n_l` no longer exposes `m̃(low)` (an earlier zero-numerator design did; the
  regression test `test_last_layer_children_do_not_reveal_multiplicities` keeps that extraction failing). The
  dummy denominators also randomize the left/right ratio of the root children.
* **PCS.** Every value the verifier sees from the opening is a linear function of `F` for fixed challenges:
  `obs = A·s + B·r` (`s` = data and Libra cells, `r` = reserved blinding). The prover-side rank check
  `folded_pcs.require_joint_mask_ext` certifies `im A ⊆ im B` over `E` (every `A(x − x')` is absorbed by a change of
  the reserved cells), using the closed-form observation rows `folded_pcs.observation_rows`.
* **Honest scope.** The rank certificate is computed exactly for test parameters (`n = 8, Q = 2` and
  `n = 10, Q ∈ {4, 8, 16, 24}` pass; `n = 8, Q = 4` fails and is therefore not claimed). At profile scale
  (`n = 12, Q = 72`: 1765 observation rows, 1204 secret vs 2892 blinding cells) an offline exact elimination over
  `E` on one honest proof's challenges gave `rank(B) = rank(A|B) = 693` (certified for that challenge outcome;
  not recomputed in CI and not a proof for every challenge outcome). A full simulation proof of the masked
  fractional GKR (joint distribution of all revealed layer evaluations) is not part of this specification. The
  soundness checks below do not depend on any ZK property.
* **Security bits (provisional).** `security_bits = 100` for `Q = 72` at rate 1/8 assumes per-query proximity
  error `≈ √ρ ≈ 2^−1.5` (Johnson-radius regime, giving `≈ 108` bits from queries) with all sumcheck/GKR/Fiat–Shamir
  terms below `2^−150` over `E` (`|E| ≈ 2^192`); it is not derived from a full round-by-round analysis and, like
  `fpn2-llm-std-v1`, will be re-pinned when one is.

#### 5.4.7 Wire layout

All `Fp3` fields are canonical (`E_NONCANONICAL` otherwise); digests are opaque 32-byte strings.

| Section | Size | Contents |
|---|---|---|
| `STMT` | 64 | `root_0: Bytes32` ‖ `N_s: Bytes32` |
| `GKR_ROOT` | 48 | `root_num: Fp3` ‖ `root_den: Fp3` |
| `GKR_L0` | 96 | children `n_l, d_l, n_r, d_r` |
| `GKR_Lk`, `k = 1..s` | `24 + 96k + 120` | `G_sum` ‖ `k` rounds × `g(0..3)` ‖ children (4) ‖ `g_r` |
| `PCS` | `24 + 72n + 32(T−1) + 24` | `z_blind` ‖ for `t < T`: round `2t` ‖ round `2t+1` ‖ (`root_{t+1}` if `t < T−1`) ‖ `FINAL` |
| `QUERIES` | `Q · Σ_{t<T} (96 + 32(n + 1 − 2t))` | per query, per layer `t`: 4 coset values ‖ `n + 1 − 2t` siblings |
| `PAD` | rest | zero |

`fpn2-folded-gkr-v1`: body 158 960 bytes (`STMT` 64, GKR 5 760, PCS 1 072, queries 72 × 2 112), `PAD` 4 816.

#### 5.4.8 Transcript schedule and checks

```
T = init("FPN2/transcript/v2"); T.absorb_bytes(header); T.absorb_bytes(STMT)       # root_0 bound before beta
beta = T.squeeze_fp3()
T.absorb_bytes(GKR_ROOT);  check root_num == 0 and root_den != 0                   # else E_GKR_ROOT
for k in 0..s:
    alpha = T.squeeze_fp3();  claim_k = c_num + alpha c_den
    if k >= 1: T.absorb_bytes(G_sum);  lambda = T.squeeze_fp3();  claim_k += lambda G_sum
    for each round: T.absorb_bytes(round); check g(0)+g(1) == claim_k; r = T.squeeze_fp3(); claim_k = g(r)
    T.absorb_bytes(children [‖ g_r]);  check claim_k == eq(pt, r)(n + alpha d) [+ lambda g_r]   # else E_GKR
    branch = T.squeeze_fp3();  pt = (branch, r);  (c_num, c_den) = mix(children, branch)
T.absorb_bytes(z_blind);  mu = T.squeeze_fp3();  y = Y (§5.4.4)
for t in 0..T-1:
    for the two rounds: T.absorb_bytes(round); check h(0)+h(1) == y; r = T.squeeze_fp3(); y = h(r)   # E_SUMCHECK
    if t < T-1: T.absorb_bytes(root_{t+1})
T.absorb_bytes(FINAL);  check y == FINAL · w~(r)                                    # else E_FINAL
for q in 0..Q-1:
    pos = int_le(T.squeeze_bytes(8)) mod (L/4)
    for t in 0..T-1:
        j = pos mod (L_t/4); slot = pos div (L_t/4)
        check MerklePath(root_t, leaf(values), j, siblings)                         # else E_MERKLE
        if t > 0: check values[slot] == prev                                        # else E_FOLD
        prev = fold4(values, x^-1 = ω_t^(−j), r_2t, r_2t+1);  pos = j
    check prev == FINAL                                                             # else E_FOLD
```

#### 5.4.9 Size calculator

`folded_pcs.byte_budget(cells, cohorts = 2, queries = 256)` is a size *model* (exact for its stated layout,
which differs from the relation-3 wire: base-field multi-cohort layer-0 leaves, GKR over `2^(n+1)` leaves) of the
fused folded PCS + masked GKR when `cells` base-field cells are split over `cohorts` multilinear commitments: for
`300 000 000` cells, `n = 28`, `L = 2^31`, 14 folds, **2 457 144 bytes (2.34 MiB) ≤ 8 MiB**.


### 5.5 Body for `relation_id = 4` (`FOLDED_GKR_AIR`)

A Plonkish arithmetic circuit over `F_p` with copy constraints and a fixed `[0, 128)` limb range lookup, proven
with the relation-3 machinery: ONE ZK-masked FracLogUp-GKR tree (wiring + range), ONE Libra-masked zerocheck
(gates), and ONE batched folded ZK-Basefold opening that discharges every resulting linear claim on a private
witness vector `F` and a public circuit vector `C`. Reference prover: `prover/fpn_prover/backend/folded_air.py`
(lowering from the gadget IR: `prover/fpn_prover/circuit/lower.py::compile_air`); verifiers:
`core/src/folded_air.rs`, `verifier/python/fpn_verify_ref/folded_air.py`.

#### 5.5.1 Parameters, statement and the pinned circuit root

From the profile: `n = n_vars` (even, `12 ≤ n ≤ 28`), `sc_degree = 3` (gate degree / GKR round degree),
`Q = n_queries` (`1 ≤ Q ≤ 4096`); otherwise `E_BACKEND_UNAVAILABLE`. Derived: `g = n − 5`, `K = 2^g` gate rows
**and** `K` variables, `S = g + 3`, `L = 8 · 2^n`, `T = n/2`, `ω = 7^((P−1)/L)`, `ι = 7^((P−1)/4)`. The limb table
is `[0, 128)` for every `n` (`TABLE_BITS = 7`).

**Circuit.** A public table of `K` rows `(q_M, q_L, q_R, q_O, q_C, σ_a, σ_b, σ_c)` (`q_* ∈ F_p`, wire ids
`σ ∈ [1, K]` = variable index + 1), public variable multiplicities `mult_k = #{cells wired to variable k}` and
range flags `flag_k ∈ {0, 1}`. Its vector `C` (§5.5.2) is committed deterministically (no blinding) exactly like
`F` (§5.4.5); the root `root_circ` is a pure function of the circuit.

**Statement.** The prover knows `V ∈ F_p^K` such that for every row `x`, with `w_col(x) = V[σ_col(x) − 1]`:
`q_M w_a w_b + q_L w_a + q_R w_b + q_O w_c + q_C = 0`, and `V_k ∈ [0, 128)` whenever `flag_k = 1`.

**Pinning (fail closed).** The statement is relative to `root_circ`, carried in `STMT`. A relation-4 verifier
MUST be given the expected root (`--circuit-root <hex64>` / `expected_circuit_root`): absent →
`E_CIRCUIT_UNPINNED`, different → `E_CIRCUIT_PINNED`. Changing any selector, wire id, multiplicity or flag changes
`root_circ`; a container whose `STMT` claims the honest root but whose openings use another table fails the
circuit-leaf Merkle paths (`E_MERKLE`) or the fold/final checks.

#### 5.5.2 Committed vectors

Both vectors have `2^n` cells. Data coordinates (`n − 1` bits) map, low to high, to cell bits `0, 2, 3, …, n−1` of
the half `{b_1 = 1}`; blocks are aligned in data coordinates:

* `F`: `[0,K) w_a` · `[K,2K) w_b` · `[2K,3K) w_c` · `[3K,4K) V` · `[4K,5K) m` (range multiplicities,
  slot `t` counts table value `t mod 128`) · `[5K,6K) N` · `[6K,7K) E` (dummy fractions, `E ≠ 0`) · `[7K,8K)` spare
  (random) · `[8K,12K) A` · `[12K,16K) D` (additive GKR masks, `D ≠ 0`). The half `{b_1 = 0}` holds the GKR Libra
  coefficients (slot `2k(k−1) + 4i + e`, `k = 1..S`), then the zerocheck Libra coefficients (slot
  `2S(S+1) + 5i + e`, `i < g`, `e ≤ 4`), deposited into bits `0, 2, 3, …`; every other cell there is uniform
  blinding.
* `C`: blocks `0..9` = `q_M, q_L, q_R, q_O, q_C, σ_a, σ_b, σ_c, mult, flag` (each `K` cells, `F_p` values
  embedded in `E`); every other cell is `0`.

#### 5.5.3 GKR leaves (`2^(S+1)`, lowest variable = real/mask bit)

Real leaf `j = blk · K + x` (`blk < 8`, `x < K`) sits at `2j`:

| `blk` | numerator | denominator |
|---|---|---|
| 0, 1, 2 | `1` | `β − w_col(x) − γ σ_col(x)` (`col = a, b, c`) |
| 3 | `−mult_x` | `β − V_x − γ (x + 1)` |
| 4 | `flag_x` | `β − V_x` |
| 5 | `−m_x` | `β − (x mod 128)` |
| 6 / 7 | `N_x` / `−N_x` | `E_x` |

Mask leaf `v < 2^S` sits at `2v + 1` with `(A_v, D_v)` / `(−A_{v'}, D_{v'})` exactly as §5.4.3. The sum is
`Σ_cells 1/(β − w − γσ) − Σ_k mult_k/(β − V_k − γ(k+1))` (zero iff every gate cell equals its variable: wire ids are
`≥ 1`, so these poles depend on `γ`) `+ Σ_k flag_k/(β − V_k) − Σ_t m_t/(β − t mod 128)` (zero iff every flagged
variable is in the table), plus cancelling dummy and mask pairs. `β`, `γ` are squeezed after `STMT`; the root must
be `(0, root_den ≠ 0)`. The masked layer sumchecks are byte-for-byte §5.4.4 (layers `k = 0..S`).

#### 5.5.4 Zerocheck (gate degree 3, round degree 4)

After the GKR: `z_i = squeeze_fp3()` for `i < g`. The prover proves `Σ_{x ∈ {0,1}^g} eq(z, x)·Gate(x) = 0` with a
Libra mask `G_Z(x) = Σ_i h_i(x_i)` (`deg h_i = 4`, coefficients in `F`): it sends `Z_sum = 2^(g−1) Σ_i (h_i(0) + h_i(1))`,
then `λ_Z = squeeze`, the target is `λ_Z · Z_sum`; round `i` sends `s_i(0..4)` (`eq · q_M · w_a · w_b` has degree 4);
finally it sends `e_a, e_b, e_c, e_M, e_L, e_R, e_O, e_C, g_r` (the claimed `w̃_a(r), …, q̃_C(r), G_Z(r)`) and the
verifier checks `target = eq(z, r)(e_M e_a e_b + e_L e_a + e_R e_b + e_O e_c + e_C) + λ_Z g_r` (`E_ZEROCHECK`).

#### 5.5.5 Batched linear claim

With `ρ = (ρ_0, ρ_1..ρ_S)` from the last GKR layer, `low = ρ_1..ρ_g`, `top = ρ_{g+1..g+3}`, `eqb(k) = eq(top, bits(k))`,
`real = 1 − ρ_0`, `X̃ = Σ_{i<g} 2^i low_i`, `T̃ = Σ_{i<7} 2^i low_i`:

```
num_pub = real (eqb0 + eqb1 + eqb2)
den_pub = real (β (eqb0 + … + eqb5) − γ eqb3 (1 + X̃) − eqb5 T̃)
num − num_pub = real(−eqb5 m̃ + (eqb6 − eqb7) Ñ)(low) + ρ_0 (1 − 2ρ_S) Ã(ρ_1..ρ_{S−1})
                + real(−eqb3 mult̃ + eqb4 flag̃)(low)                                          [C]
den − den_pub = −real(eqb0 w̃_a + eqb1 w̃_b + eqb2 w̃_c + (eqb3 + eqb4) Ṽ − (eqb6 + eqb7) Ẽ)(low) + ρ_0 D̃(…)
                − real γ (eqb0 σ̃_a + eqb1 σ̃_b + eqb2 σ̃_c)(low)                                 [C]
Y = (num − num_pub) + μ (den − den_pub) + Σ_{k=1..S} μ^(k+1) g_r(k)
    + Σ_{j<9} μ^(S+2+j) (e_a, e_b, e_c, e_M, e_L, e_R, e_O, e_C, g_r)_j + μ^(S+11) z_blind
```

`z_blind = Σ_{b_1 = 0} F(b)(eq(2, b) + eq(3, b))`. Hence `Y = Σ_b F(b) a(b) + Σ_b C(b) c(b)`: `e_a, e_b, e_c` weight the
`w` blocks of `F` at `r`, `e_M..e_C` the selector blocks of `C` at `r`, `g_r` the zerocheck Libra cells
(`coef · r_i^e`). Both `w̃` evaluations cost `O(n (S² + g))`.

#### 5.5.6 One batched folded opening of `(F, C)`

Let `G(x_0, x_1, y) = F(y) + x_0 C(y)` and `W(0,0,y) = a − c`, `W(1,0,y) = c`, `W(·,1,·) = 0`, so
`Σ G W = Σ F a + Σ C c = Y`. The Basefold sumcheck runs over the `n + 2` variables of `G`, lowest first
(`h(0), h(1), h(2)` per round). After the two batching rounds (`r_0 = θ`, `r_1`) the claim is on
`H = F + θ C`; its layer-0 codeword is **not** committed: per query the verifier opens the coset leaf `j` of `F`
under `root_0` and of `C` under `root_circ` and combines `u_k = v^F_k + θ v^C_k`. Rounds `2 + 2t, 3 + 2t` fold
layer `t` exactly as §5.4.5; layers `1..T−1` of `H` are committed, `FINAL = H̃(r_2..r_{n+1})`, and the final check is
`y = FINAL · (1 − r_1)((1 − θ) ã(r') + (2θ − 1) c̃(r'))`. `θ` is squeezed after both roots are absorbed, so the
cross terms are carried by the sumcheck (no separate per-vector evaluations are ever accepted).

#### 5.5.7 Wire layout

| Section | Size | Contents |
|---|---|---|
| `STMT` | 96 | `root_0` ‖ `root_circ` ‖ `N_s` |
| `GKR` | `48 + 96 + Σ_{k=1..S} (24 + 96k + 120)` | as §5.4.7 |
| `ZEROCHECK` | `24 + 120 g + 216` | `Z_sum` ‖ `g` rounds × `s(0..4)` ‖ 9 finals |
| `PCS` | `24 + 72(n+2) + 32(T−1) + 24` | `z_blind` ‖ batching rounds 0, 1 ‖ for `t < T`: rounds `2+2t`, `3+2t` (‖ `root_{t+1}` if `t < T−1`) ‖ `FINAL` |
| `QUERIES` | `Q · ((96 + 32(n+1)) + Σ_{t<T} (96 + 32(n+1−2t)))` | per query: `F` leaf + siblings, `C` leaf + siblings, then layers `1..T−1` |
| `PAD` | rest | zero |

`fpn2-folded-air-v1`: body 247 176 bytes (`STMT` 96, GKR 9 360, zerocheck 1 320, PCS 1 392, queries 72 × 3 264),
`PAD` 14 904. `fpn2-folded-air-n12`: body 14 504 bytes. `fpn2-folded-air-n20-v1`: body 424 104 bytes (`STMT` 96,
GKR 19 152, zerocheck 2 040, PCS 1 920, queries 72 × 5 568), `PAD` 100 120. Size model `folded_air.byte_budget(n, Q)`: at `n = 28`,
`Q = 72` the body is 729 896 bytes (0.70 MiB ≤ 8 MiB) for `K = 2^23` rows / variables.

#### 5.5.8 Transcript schedule and checks

```
T.absorb(header); T.absorb(STMT); beta = sq; gamma = sq
T.absorb(GKR_ROOT); check root_num == 0, root_den != 0                              # E_GKR_ROOT
GKR layers k = 0..S as §5.4.8                                                       # E_GKR
z_0..z_{g-1} = sq; T.absorb(Z_sum); lambda_Z = sq; target = lambda_Z Z_sum
per round: T.absorb(s(0..4)); check s(0)+s(1) == target; r = sq; target = s(r)        # E_ZEROCHECK
T.absorb(finals); check target == eq(z, r) Gate(finals) + lambda_Z g_r                  # E_ZEROCHECK
T.absorb(z_blind); mu = sq; y = Y (§5.5.5)
for i in 0..n+1: T.absorb(round_i); check h(0)+h(1) == y; r_i = sq; y = h(r_i);
                 if a fold root follows round i: T.absorb(root)                     # E_SUMCHECK
T.absorb(FINAL); check y == FINAL · W~(theta, r_1, r')                                # E_FINAL
per query: pos = int_le(sq_bytes(8)) mod (L/4); layer 0: F leaf (root_0), C leaf (root_circ)   # E_MERKLE
           then layers 1..T-1 as §5.4.8; terminal == FINAL                          # E_MERKLE / E_FOLD
```

#### 5.5.9 Lowering, soundness and zero-knowledge scope (honest statement)

* **What the verifiers check.** Every gate row (zerocheck), every copy constraint and every flagged variable's
  membership in `[0, 128)` (one fractional sum, two independent challenges), all bound to `root_0` and the pinned
  `root_circ`. `circuit/lower.py::compile_air` lowers every `Program` equality (including microprogram selector
  booleanity `s(s − 1) = 0`, the exhaustive one-hot `Σ s = 1`, `rmsnorm` span and the `gemm` centre dot product
  and box difference), every wire recomposition `u + 2^(B−1) = Σ 128^i limb_i` and every tightening identity to
  Plonkish rows (with accumulator variables for gates with more than three terms) and range-flags every limb and
  tightening cell, so an in-range witness violating any of them is rejected by both verifiers (the `wbforge`
  suite `tests/unit/test_relation4_air_and_wbforge.py`).
* **What they do NOT check.** What remains inside `LookupSpec` reference predicates is evaluated only by the
  prover-side reference interpreter; it is not lowered and a relation-4 proof says nothing about it. Algebraic branch
  and tile conditions are lowered as gates (`gemm` `r ≥ 0`, `|w| ≤ wr`, the absolute-value operand when it
  keeps the `2^62` lift bound, the radius product when the declared operand width allows it; microprogram unused
  lanes, `dense_gqa` normalize booleanity, the `moe_router_expert` argmax / maximum / weight / denominator /
  index-list / top-k conditions, the `gdn_lrec` and `swa_sink` range conditions). The residue is: RMSNorm in
  full (BF16 decoding, `rsqrt` endpoints and enclosure); the `gemm` box tolerance always, and the operand /
  radius product where they exceed the lift bound (the radius product at the default 62-bit operand); the `exp`
  enclosures and every nearest-rounded division of the five branch relations. The exact two-sided list is
  `AIR_GATE_FAMILIES` / `AIR_PREDICATE_RESIDUE` in `circuit/lower.py`, checked against the emitted circuit and
  mutation-audited on the lowered table by `tests/audit/test_lowered_air_mutation_audit.py`.
* **Field semantics.** Gates hold over `F_p`; `Program.check_bounds` (`|difference| < 2^62`) makes that equivalent to
  integer equality, and recomposition from `[0, 128)` limbs pins each wire to its signed range.
* **Capacity.** One container holds `K = 2^(n−5)` rows and `K` variables (`2^23` at `n = 28`, the Goldilocks
  2-adicity limit of the rate-1/8 code); larger circuits need several containers (not specified here).
* **Zero knowledge (linear view certified per transcript; full ZK not claimed).** The masking structure mirrors
  §5.4.6 (Libra masks on every GKR layer and on the zerocheck, additive mask leaves, dummy fractions, blinding half
  of `F`, `z_blind`). In addition:
  * *`ZK_ROWS` blinding rows.* The zerocheck finals `e_a, e_b, e_c = w̃_col(r)` reveal one evaluation of
    each gate column at a random point. Padding rows wired `(1, 1, 1)` to a single pad variable would make all three
    columns share one `F_p` mask so `e_a − e_b` would be a witness-only linear functional.
    `circuit/lower.py::compile_air` therefore appends `FA.ZK_ROWS = 3` zero-selector rows
    (`q = 0`, so they constrain nothing and soundness is unchanged) wired to `3 · ZK_ROWS` fresh `#zk` variables that
    the honest prover fills with uniform `F_p` values (never range-checked). Each column then carries `ZK_ROWS`
    independent masks — an `F_p`-basis of `F_p^3` at a random point — so `(e_a, e_b, e_c)` are jointly masked.
    The rows are part of the public table, hence of `root_circ`.
  * *Joint-rank certificate `folded_air.require_joint_mask_air`.* For one honest transcript it builds every
    plaintext value that is `F_p3`-linear in the committed cells (GKR Libra sums / `g_r`, last-layer leaf children,
    zerocheck Libra sum, `e_a, e_b, e_c`, `g_r`, `z_blind`, the `3(n+2)` batched sumcheck samples and `H̃(r)`, and per
    query the layer-0 `F` coset values and every folded-layer coset value) as `F_p` rows over a coordinate model
    of `F` (one `F_p` coordinate per variable summing its `V` cell and every gate cell wired to it, `m_t` as `F_p`
    coordinates, every mask / Libra / blinding cell as a uniform `F_p3` coordinate), splits the columns into
    witness coordinates `A` and mask coordinates `B` (mask variables incl. `#zk`, Libra / zerocheck slots, and a
    random sample of the GKR masks, spare block and reserved half — subsampling `B` can only make the check fail),
    and checks `rank(B) == rank([B | A])` over `F_p` with the native `kernels/fp_rank.cpp` eliminator. With the
    blinding rows the check holds on the lowered `fpn2-folded-air-v1` circuit on real `Qwen3-0.6B` witnesses; with
    `ZK_ROWS = 0` it FAILS (`rank(B) < rank([B | A])`), and the regression test
    `tests/unit/test_e2e_real_model_and_zk.py::test_joint_mask_rank_fails_without_zk_rows` pins that.
  * *Remaining scope (not claimed).* The certificate covers the LINEAR view of ONE transcript (fixed challenges)
    with a sampled `B`. It does not cover the non-linear messages — GKR inner-layer messages and the GKR /
    zerocheck round polynomials — which rest on the mask-leaf (`A`, `D`, `N`, `E`) and Libra arguments and consume
    some of the same mask coordinates; `require_pcs_mask_air` (opening-only, stricter) exists but is not part of
    this claim. There is no simulator proof for relation 4 and no Lean statement of zero knowledge. Measured
    re-proof hygiene: two proofs of the same witness share 0 64-byte blocks inside the private sections
    (`root_0`, GKR / zerocheck / PCS rounds, `F` and folded-layer openings); the header, `root_circ`, `N_s`, the
    `C` openings (public circuit codeword + Merkle siblings) and the zero padding are public and may coincide.
    Container length is `C_p` for every witness and architecture by construction. Soundness does not depend on
    any of this.
* **Security bits & Lean coverage (provisional).** As §5.4.6: `security_bits = 100` at `Q = 72`, rate 1/8, not a full
  round-by-round analysis. `lean/FlashProveNext/Fpn2Rel4.lean` verifies canonical byte-to-typed refinement and
  circuit-root pinning (`verifyBytesRel4`, `verifyBytesAny`, 50/50 vector parity); cryptographic soundness of
  `Rel4TypedVerify` under a random oracle remains for later formalization stages.

### 5.6 Body for `relation_id = 5` (`FOLDED_VEC_MATMUL`)

Full-width GEMM tiles with **public** integer-unit weights: ONE Libra-masked inner-product sumcheck reduces every
output of `B` tiles of `32 × K` multiply-adds to linear claims on a private witness vector `F` and on the public
weight vector `C`, discharged by ONE batched folded ZK-Basefold opening that is byte-for-byte relation 4's (§5.5.6).
Reference prover: `prover/fpn_prover/backend/folded_vec.py` (GPU path: `engine = CudaEngine()`, bit-exact
`HostEngine` / pure-Python CPU paths); verifiers: `core/src/folded_vec.rs`, `verifier/python/fpn_verify_ref/folded_vec.py`.

#### 5.6.1 Parameters, statement and the pinned weight root

From the profile: `n = n_vars` (even, `12 ≤ n ≤ 28`), `sc_degree = 3`, `Q = n_queries` (`1 ≤ Q ≤ 4096`),
`K = max_hidden` (a power of two, `K ≥ 32`); otherwise `E_BACKEND_UNAVAILABLE`. Derived: `R = 32` rows per tile
(fixed), `k = log2 K`, `β = n − 1 − 5 − k ≥ 0`, `B = 2^β` tiles, `d = k + β` sumcheck variables, `L = 8 · 2^n`,
`T = n/2`, `ω = 7^((P−1)/L)`, `ι = 7^((P−1)/4)`. The `B · R · K` weight cells fill the data half of `C` exactly
(`fpn2-folded-vec-n16-v1`: `B = 1`; `fpn2-folded-vec-n22-v1`: `B = 64`).

**Statement.** For public tiles `W[b] ∈ F_p^(R × K)` committed under `weight_root`, the prover knows `X[b] ∈ F_p^K` and
`Y[b] ∈ F_p^R` (`b < B`) with `Y[b][i] = Σ_j W[b][i][j] · X[b][j]` **over `F_p`** for every `b, i`.

**Pinning (fail closed).** `STMT` is `root_0 ‖ weight_root ‖ N_s` (as relation 4). The caller MUST pin the expected
weight root through the same input as relation 4's circuit root (`--circuit-root <hex64>` / `expected_circuit_root`):
absent → `E_CIRCUIT_UNPINNED`, different → `E_CIRCUIT_PINNED`. `weight_root` is a pure function of `(n, K, W)`.

#### 5.6.2 Committed vectors

Both vectors have `2^n` cells; data coordinates (`n − 1` bits) map, low to high, to cell bits `0, 2, 3, …, n−1` of the
half `{b_1 = 1}` (as §5.5.2):

* `F`: `[0, B·K)` `X` (index `b·K + j`) · `[B·K, B·K + B·R)` `Y` (index `B·K + b·R + i`) · every other data cell
  uniform spare. The half `{b_1 = 0}` holds the Libra coefficients `g_{l,e}` (slot `4l + e`, `l < d`, `e < 4`,
  deposited into bits `0, 2, 3, …`); every other cell there is uniform blinding. `X`, `Y` are `F_p` values embedded in `E`.
* `C`: `[0, B·R·K)` `W` (index `(b·R + i)·K + j`, `F_p` values embedded in `E`) — the whole data half; the half
  `{b_1 = 0}` is `0`. `C` is committed deterministically (no blinding) exactly like `F`.

#### 5.6.3 Inner-product sumcheck (zero-claim form, round degree 3)

After `STMT`: `r_i = (squeeze_fp3())^5`, then `r_b = (squeeze_fp3())^β` (the row point `r_row = (r_i, r_b)`). Over
`v = (j, b) ∈ {0,1}^k × {0,1}^β` (column bits first) the prover proves

```
Σ_v P(v) = λ · G_sum,     P(v) = eq(r_b, b) · ( W̃(b, r_i, j) · X(b, j) − eq(0, j_{5..k−1}) · eq(r_i, j_{0..4}) · Y(b, j_{0..4}) ) + λ G(v)
```

whose unmasked part equals `Σ_{b,i} eq(r_row, (b, i)) · ((W X)[b][i] − Y[b][i])`, the multilinear extension of the
output residual at a random point (zero for all but `≤ (5 + β)/|E|` of `r_row` unless every residual is zero).
`G(v) = Σ_{l<d} g_l(v_l)` (`deg g_l = 3`, coefficients in `F`): the prover sends `G_sum = 2^(d−1) Σ_l (g_l(0) + g_l(1))`,
then `λ = squeeze`; round `l` sends `s_l(0..3)` (the tile bits carry `eq · W · X`, degree 3), checks
`s_l(0) + s_l(1) = target`, `s = squeeze`, `target = s_l(s)` (`E_ZEROCHECK`). Finally the prover sends ONLY
`e_W = W̃(s_b, r_i, s_j)` — a function of the public weights — and NOT `X̃(s)`, `Ỹ(s)` or `G(s)`: the verifier never
learns `Ỹ(r_row)` (zero-claim form) nor any other plaintext evaluation of `X` or `Y`.

#### 5.6.4 Batched linear claim

With `s = (s_j, s_b)`, `eqrb = eq(r_b, s_b)` and `E_s = eq(r_i, s_{j,0..4}) · Π_{t ≥ 5} (1 − s_{j,t})`, the final round
value satisfies `target = eqrb (e_W X̃(s) − E_s Ỹ(s_b, s_{j,0..4})) + λ G(s)` for an honest prover; the verifier
replaces the sumcheck's final check by the claim (after `T.absorb(z_blind)`, `μ = squeeze`)

```
y = target + μ e_W + μ² z_blind = Σ_b F(b) a(b) + Σ_b C(b) c(b)
a = eqrb e_W · eq(s, ·) on the X block − eqrb E_s · eq((s_{j,0..4}, s_b), ·) on the Y block + Σ_{l,e} λ s_l^e [slot 4l+e]
    + μ² (eq(2, ·) + eq(3, ·)) on {b_1 = 0}
c = μ · eq((s_j, r_i, s_b), ·) on the W block
```

with `z_blind = Σ_{b_1 = 0} F(b)(eq(2, b) + eq(3, b))` (§5.5.5). A wrong `e_W` breaks the `C` part of the claim; a wrong
`X̃(s)` / `Ỹ(s)` / `G(s)` is impossible to express (they are never sent). `ã(r')` and `c̃(r')` cost `O(n · d)`.

#### 5.6.5 One batched folded opening of `(F, C)`

Byte-for-byte §5.5.6 with `C` = the weight vector and `root_circ` = `weight_root`: two batching rounds (`θ = r_0`,
`r_1`), the fused coset-4 fold chain of `H = F + θ C` (layer 0 read from both Merkle trees per query), `FINAL = H̃(r')`,
final check `y = FINAL · (1 − r_1)((1 − θ) ã(r') + (2θ − 1) c̃(r'))` (`E_FINAL`).

#### 5.6.6 Wire layout

| Section | Size | Contents |
|---|---|---|
| `STMT` | 96 | `root_0` ‖ `weight_root` ‖ `N_s` |
| `INNER_PRODUCT` | `24 + 96 d + 24` | `G_sum` ‖ `d` rounds × `s(0..3)` ‖ `e_W` |
| `PCS` | `24 + 72(n+2) + 32(T−1) + 24` | `z_blind` ‖ batching rounds 0, 1 ‖ for `t < T`: rounds `2+2t`, `3+2t` (‖ `root_{t+1}` if `t < T−1`) ‖ `FINAL` |
| `QUERIES` | `Q · ((96 + 32(n+1)) + Σ_{t<T} (96 + 32(n+1−2t)))` | per query: `F` leaf + siblings, `C` leaf + siblings, then layers `1..T−1` |
| `PAD` | rest | zero |

`fpn2-folded-vec-n16-v1`: body 288 368 bytes (`STMT` 96, inner product 1 008, PCS 1 616, queries 72 × 3 968),
`PAD` 104 784. `fpn2-folded-vec-n22-v1`: body 469 184 bytes (`STMT` 96, inner product 1 584, PCS 2 192,
queries 72 × 6 464), `PAD` 55 040.

#### 5.6.7 Transcript schedule and checks

```
T.absorb(header); T.absorb(STMT); r_i = sq^5; r_b = sq^beta
T.absorb(G_sum); lambda = sq; target = lambda G_sum
per round l < d: T.absorb(s(0..3)); check s(0)+s(1) == target; s_l = sq; target = s(s_l)      # E_ZEROCHECK
T.absorb(e_W); T.absorb(z_blind); mu = sq; y = target + mu e_W + mu^2 z_blind
for i in 0..n+1: T.absorb(round_i); check h(0)+h(1) == y; r_i = sq; y = h(r_i);
                 if a fold root follows round i: T.absorb(root)                                # E_SUMCHECK
T.absorb(FINAL); check y == FINAL · W~(theta, r_1, r')                                         # E_FINAL
queries exactly as §5.5.8 with root_circ := weight_root                                        # E_MERKLE / E_FOLD
```

#### 5.6.8 Soundness, integer semantics and zero-knowledge scope (honest statement)

* **What the verifiers check.** Every output of every tile, `Y = W X` over `F_p`, bound to `root_0` and the pinned
  `weight_root`: a false `X`, `Y` or a weight table other than the pinned one is rejected (golden vectors
  `vec_n16_tampered_x / _y / _weights*`, `tests/unit/test_relation5_vec_matmul.py`); a false `Y` whose every inner-product round is shifted so that each round sum passes dies at the folded-in final claim (`E_SUMCHECK`, `vec_n16_adaptive_ip_rounds`). Soundness error: the row-point
  Schwartz–Zippel term `(5 + β)/|E|`, the sumcheck `3d/|E|`, and the batched opening as §5.4.6 (`security_bits = 100`
  provisional at `Q = 72`, rate 1/8, not a full round-by-round analysis).
* **What they do NOT check.** There is **no range check** on `X` or `Y` inside relation 5. The statement is a congruence mod `p`; it equals the integer statement
  whenever `Σ_j |W[b][i][j]| |X[b][j]| < p/2`, which the prover-side caller (not the verifier) checks and reports for
  real BF16 tiles (exact integer encoding `value = int · 2^−shift`, one shift per tile and operand:
  `folded_vec.bf16_words_to_ints`). `Y` is the exact product of the BF16 operands, not a rounded BF16 GEMM output, and
  nothing here binds `X` to a capture trace or `Y` to a later layer: that is the job of the surrounding protocol.
* **Capacity.** `B · 32 · K = 2^(n−1)` multiply-adds per container (`2^27` at `n = 28`); the single committed `F` is
  mostly blinding / spare (`B (K + 32)` data cells), the price of sharing one codeword length with `C`.
* **Zero knowledge (linear view certified at `n = 16`; general simulator proof open).** No plaintext evaluation of the
  witness is sent: `e_W` is a function of the public `W` and the challenges, the zero-claim form hides `Ỹ(r_row)`,
  every round polynomial carries the Libra mask `λ G` (coefficients committed in `F` before any challenge), and the
  opening is masked by `z_blind` and the blinding half exactly as relation 4. `folded_vec.require_joint_mask_vec`
  verifies `rank(B) = rank([B | A]) = 3495` over the 1,885-observation (5,655 `F_p`-row) joint linear view (including
  all 40 inner-product round messages, `z_blind`, PCS sumcheck rounds and all 72 query cosets) on a concrete
  `fpn2-folded-vec-n16-v1` transcript, and re-proving shares
  zero 64-byte private blocks. A general transcript-wide simulator proof and `n = 22` rank run remain open. Container
  length is `C_p` for every witness by construction.
* **Lean.** `lean/FlashProveNext/Fpn2Rel5.lean` formalizes canonical parsing, required `weight_root` pinning, and
  byte-to-typed refinement (`verifyBytesRel5_refines_typed`, `_complete`, `_downgrade_resistant`,
  `_injective_on_accepted`, `_requires_weight_pin`, `verifyBytesRel5_foldedVecN22` registered `n=22` / `beta=6`
  profile dispatch equivalence, `verifyBytesAny` 4-relation dispatcher; 33/33 `fpn2_folded_vec.json`
  golden vectors + 26/26 multi-tile `beta=2` fixture cases including `E_FOLD` + 9/9 profile pins verified under `lean/check.sh`).
  `lean/FlashProveNext/Fpn2Soundness.lean` transports accepted bytes to `Rel5FieldEquations` over `Fp3Field`
  (`verifyBytesRel5_accept_field`, `Rel5Canonical` covering `QUERIES`, `foldField_cubic` with layer bound `2t + 2 <= n`
  discharged from accepted runs, algebraic error budget `≤ 2^-183` and lifetime soundness at
  `k=105/51/100`). Random oracle, FRI/Basefold proximity, `n=22` Lean execution, and full extraction remain open.

### 5.7 Body for `relation_id = 6` (`FOLDED_VEC_BOUND`)

Relation 5 proves `Y = W X` for an `X` that lives inside the private witness `F` (`root_0`). Nothing ties that `X` to anything
the verifier — or the pre-challenge commitment `C_pre` — has seen: a prover can open honest trace rows and an honest
`N_s` while proving a privately substituted `(X', Y' = W X')`, and every relation-5 verifier accepts (**case (c)**; the
accepting relation-5 proof of such a substitution is exercised in `tests/unit/test_relation6_vec_bound_zk.py`). Relation 6
closes the gap: the activations are committed **before any challenge** under a ZK-blinded root `x_root` that the caller pins
(and that `C_pre` binds), and the same inner-product sumcheck is discharged against that root.
Reference prover: `prover/fpn_prover/backend/folded_vec_bound.py`; verifiers: `core/src/folded_vec_bound.rs`,
`verifier/python/fpn_verify_ref/folded_vec_bound.py`.

#### 5.7.1 Parameters, statement and the two pinned roots

Parameters, derived values and geometry are §5.6.1's (`n`, `Q`, `K`, `R = 32`, `β`, `B`, `d`, `L`, `T`, `ω`, `ι`).

**Statement.** For public tiles `W[b]` committed under `weight_root` and activations `X[b] ∈ F_p^K` committed under
`x_root`, the prover knows `Y[b] ∈ F_p^R` (committed under `root_0`) with `Y[b][i] = Σ_j W[b][i][j] · X[b][j]` over `F_p`.
`STMT` is `root_0 ‖ weight_root ‖ x_root ‖ N_s` (128 bytes).

**Pinning (fail closed).** The caller MUST pin BOTH roots: `weight_root` through relation 4/5's input
(`--weight-root <hex64>`, alias `--circuit-root`; `expected_circuit_root`) and `x_root` through `--x-root <hex64>`
(`expected_x_root`). Order: weight pin absent → `E_CIRCUIT_UNPINNED`; `x_root` pin absent → `E_X_ROOT_UNPINNED`;
`weight_root` differs → `E_CIRCUIT_PINNED`; `x_root` differs → `E_X_ROOT_PINNED`. `weight_root` is a pure function of
`(n, K, W)`; its relation to `SetupCertificate.weight_commitment_root` (the model-wide tile Merkle root) is established by the
surrounding protocol (tile openings), not by this container. `x_root` is **not** deterministic: it commits `X` plus fresh
uniform blinding, is computed by `commit_x` BEFORE the external seed, and `precommit_x_root` binds it into `C_pre` as
`C_pre = B256_{FPN2-CPRE-v1}(setup_id ‖ trace_slot ‖ |resp| ‖ resp ‖ |policy'| ‖ policy')` with
`policy' = "FPN2-XROOT-v1" ‖ x_root ‖ policy` (domain-tagged so it never aliases a `trace_payload_root`;
with no trace payload `trace_slot = B256_{FPN2-XROOT-TPR}(x_root)`). The only activation-derived byte string that
leaves the prover before the challenge is the 32-byte hiding root: **0 bytes of plaintext activations**.

#### 5.7.2 Committed vectors

All three vectors have `2^n` cells, the same data-coordinate mapping as §5.6.2 and the same rate-1/8 Basefold codeword
length `L = 8 · 2^n` (so one fold chain serves all three):

* `F` (`root_0`): `[0, B·K)` **uniform spare** (relation 5's `X` block; it MUST stay uniform — the batched opening exposes
  `⟨F + X_com, xw⟩` on that block) · `[B·K, B·K + B·R)` `Y` (index `B·K + b·R + i`) · every other data cell uniform spare;
  the half `{b_1 = 0}` holds the Libra coefficients (slot `4l + e`) and uniform blinding.
* `X_com` (`x_root`): `[0, B·K)` `X` (index `b·K + j`, `F_p` values embedded in `E`) · every other data cell uniform
  spare; the whole half `{b_1 = 0}` is uniform `F_p^3` blinding (independent of `F`'s).
* `C` (`weight_root`): as §5.6.2.

#### 5.7.3 Inner-product sumcheck

Byte-for-byte §5.6.3, with the transcript seeded by the 128-byte `STMT`
(`T.absorb(header); T.absorb(STMT)` — `root_0`, `weight_root` and `x_root` are all absorbed before `r_i, r_b, λ, s, μ`).
The prover sends `G_sum`, `d` rounds and `e_W = W̃(s_b, r_i, s_j)` only. `e_X = X̃(s)`, `e_Y` and `e_G` are **never sent**:
they stay inside the batched claim (a plaintext `X̃(s)` is a mask-free functional of `X`). "`e_X` comes from `x_root`" means
the block `eqrb · e_W · eq(s, ·)` that relation 5 charged to `F` is charged to `X_com`.

#### 5.7.4 Batched linear claim on three vectors

With `target` the final round value and `z_blind` as §5.6.4 (`μ = squeeze` after `T.absorb(z_blind)`):

```
y = target + μ e_W + μ² z_blind = Σ F(b) a(b) + Σ C(b) c(b) + Σ X_com(b) xw(b)
a  = − eqrb E_s · eq((s_{j,0..4}, s_b), ·) on the Y block (offset B·K) + Σ_{l,e} λ s_l^e [slot 4l+e] + μ² (eq(2,·) + eq(3,·)) on {b_1 = 0}
xw = eqrb e_W · eq(s, ·) on the X block [0, B·K) of X_com
c  = μ · eq((s_j, r_i, s_b), ·) on the W block
```

#### 5.7.5 One batched folded opening of `(F, C, X_com)` with zero extra fold layers

The two batching rounds of §5.5.6 fold the 4-slot table `[F, F + C, F + X_com, F + C + X_com]` with weights
`[a − c − xw, c, xw, 0]` by `(θ, r_1) = (r_0, r_1)`, so the vector the coset-4 fold chain runs on is

```
H = F + θ C + r_1 X_com        (not F + μ C + μ² X_com: the combination challenges are the two batching-round challenges)
FINAL = H̃(r'),   final check  y = FINAL · [ (1−θ)(1−r_1) ã(r') + (1−r_1)(2θ−1) c̃(r') + (1−θ)(2r_1−1) x̃w(r') ]   (E_FINAL)
```

(with `xw = 0` this is exactly relation 5's `(1 − r_1)((1 − θ) ã + (2θ − 1) c̃)`). Per query, layer 0 reads **three** cosets:
the `F` leaf against `root_0`, the `C` leaf against `weight_root`, the `X_com` leaf against `x_root`; the checked coset is
`v = v_F + θ v_C + r_1 v_X` (`E_MERKLE` on any of the three paths); layers `1..T−1` are as §5.6.5 (`E_FOLD`).

#### 5.7.6 Wire layout

| Section | Size | Contents |
|---|---|---|
| `STMT` | 128 | `root_0` ‖ `weight_root` ‖ `x_root` ‖ `N_s` |
| `INNER_PRODUCT` | `24 + 96 d + 24` | as §5.6.6 |
| `PCS` | `24 + 72(n+2) + 32(T−1) + 24` | as §5.6.6 |
| `QUERIES` | `Q · (2 (96 + 32(n+1)) + Σ_{t<T} (96 + 32(n+1−2t)))` | per query: `F` leaf + siblings, `C` leaf + siblings, `X_com` leaf + siblings, then layers `1..T−1` |
| `PAD` | rest | zero |

`fpn2-folded-vec-bound-n16-v1`: body 334 480 bytes (`STMT` 128, inner product 1 008, PCS 1 568, queries 72 × 4 608),
container 393 216 (`PAD` 58 672). `fpn2-folded-vec-bound-n22-v1`: body 529 120 bytes (`STMT` 128, inner product 1 584,
PCS 2 096, queries 72 × 7 296), container 1 048 576 (`PAD` 519 392). The `X_com` openings cost 640 B
(n = 16) / 832 B (n = 22) per query.

#### 5.7.7 Transcript schedule and checks

§5.6.7 with `STMT` of 128 bytes, `weight_root := STMT[32..64]`, `x_root := STMT[64..96]`, check 13 extended to both pins
(order above), and the queries of §5.7.5. Checks 12–17 of relation 6:

| # | Check (relation 6) | Code |
|---|---|---|
| 12 | every typed field canonical, in wire order | `E_NONCANONICAL` |
| 13 | `expected_circuit_root` given / equal to `STMT.weight_root`; then `expected_x_root` given / equal to `STMT.x_root` (order: weight unpinned, x unpinned, weight differs, x differs). Cross-relation dispatch rule (checked in order: `expected_circuit_root` before `expected_x_root`): supplying `expected_circuit_root` (`--circuit-root`/`--weight-root`) on relation 1 or 3 rejects with `E_CIRCUIT_PINNED` (including when both pins are supplied), and supplying `expected_x_root` (`--x-root`) on any non-relation-6 container (relations 1, 3, 4, 5) rejects with `E_X_ROOT_PINNED`, preventing downgrade from a pinned relation to an unpinned one. CLI hex pins require exact 64 ASCII hex characters (no whitespace) in both Rust and Python. | `E_CIRCUIT_UNPINNED` / `E_X_ROOT_UNPINNED` / `E_CIRCUIT_PINNED` / `E_X_ROOT_PINNED` |
| 14 | inner-product sumcheck round sums | `E_ZEROCHECK` |
| 15 | batched Basefold sumcheck round sums | `E_SUMCHECK` |
| 16 | `y == FINAL · W̃(θ, r_1, r')` (three-vector weight above) | `E_FINAL` |
| 17 | per query: `F`, `C` and `X_com` layer-0 Merkle paths, folded-layer paths, fold consistency | `E_MERKLE` / `E_FOLD` |

#### 5.7.8 Soundness, case (c), and zero-knowledge scope (honest statement)

* **What closes case (c).** The proof is about the `X` inside the pinned `x_root`: every query opens `X_com` against that root
  and the final claim couples the three vectors. The substitutions of `(X', Y' = W X')` are rejected — claiming the honest
  root with `X'` openings (`E_MERKLE`), claiming the cheater's own `x_root'` (`E_X_ROOT_PINNED`), or running the sumcheck on `X'`
  while opening the honest `X_com` (a forged `X̃(s)`; `E_SUMCHECK`) — golden vectors `vec_bound_n16_case_c_*`.
  Soundness error: relation 5's terms plus the extra batching challenge `r_1` (`1/|E|`) and the proximity of `X_com`.
* **What it does NOT prove.** `x_root` is bound to `C_pre` BEFORE the challenge, but nothing here proves that the committed
  `X` equals the captured trace rows the challenge later draws (that equality would need a hash-to-codeword link between
  `trace_payload_root` and `x_root`, or a plaintext opening, which breaks zero knowledge): a prover that commits a
  substituted `X*` to `x_root` before the challenge is held to `X*` consistently, no more. `Y` is likewise bound to `root_0`
  only. The draws that select WHICH tiles are proven are derived from `C_pre` (`request_seed`), i.e. AFTER `x_root` is
  fixed, and `X_com` holds only `B · K` activations (at most `2^(n−6)` rows of width `K`): binding the tiles that are
  later drawn therefore requires committing every candidate row before the seed exists (one `X_com` per `B`-row group of
  the trace); relation 6 does not do that selection itself. It binds the proof to a pinned `x_root`, which is a necessary
  step for case (c), not by itself a closure of it. No range check (as §5.6.8). `weight_root` ↔ `SetupCertificate.weight_commitment_root` is established by the
  surrounding protocol.
* **Zero knowledge (linear view certified at `n = 16`; general simulator proof open).** No plaintext evaluation of `X`,
  `Y` or `G` is sent. `folded_vec_bound.require_joint_mask_vec_bound` verifies `rank(B) = rank([B | A])` over the joint
  linear view of the **two** private vectors `(F, X_com)` — all inner-product round messages, `z_blind`, the batching rounds,
  `FINAL` and all 72 query cosets of `F`, of `X_com` and of every folded layer of `H` — after validating every observation row
  against the wire bytes; negative controls (no Libra mask, no `X_com` blinding, a non-random `F` X-block) fail. Re-proofs
  with a fresh `commit_x` share 0 private 64-byte blocks. Limits: (1) re-proving under the SAME `x_commitment` re-opens the
  same `X_com` tree (repeated upper Merkle siblings, and fresh opened cosets spend `X_com`'s blinding rank cumulatively —
  not covered by the per-proof rank check); (2) a general simulator proof and an `n = 22` rank run remain open; (3) the
  `n = 22` prover needs the engine path (`HostEngine` / `CudaEngine`). Measured at `n = 16`: two proofs of the same
  statement under the SAME `x_commitment` share 7–16 aligned private 64-byte blocks (varies with the query positions), all
  of them Merkle siblings inside the `X_com` opening (upper tree levels near `x_root`); no coset value, no `F` byte, no
  round message repeats. With a fresh `commit_x` the root differs and 0 private blocks / windows are shared.

---

## 6. Normative Verification Order and Reject Codes

Given `bytes`, optional pinned `expected_profile_id`, and the `allow_reduced` policy flag:

| # | Check | Code |
|---|---|---|
| 1 | `len(bytes) ≥ 64` | `E_LENGTH` |
| 2 | `magic == "FPN2"` | `E_MAGIC` |
| 3 | `version == 2` | `E_VERSION` |
| 4 | `flags == 0` | `E_FLAGS` |
| 5 | `reserved == 0^24` | `E_RESERVED` |
| 6 | `profile_id` is registered | `E_PROFILE_ID` |
| 7 | `expected_profile_id` absent or equal | `E_PROFILE_PINNED` |
| 8 | tier is `STANDARD` or `allow_reduced` | `E_REDUCED_SECURITY` |
| 9 | `len(bytes) == C_p` | `E_LENGTH` |
| 10 | relation has a wired backend | `E_BACKEND_UNAVAILABLE` |
| 11 | `PAD` all zero | `E_PADDING` |
| 12 | every typed field canonical, in wire order | `E_NONCANONICAL` |
| 13 | sumcheck round sums (transcript replay) | `E_SUMCHECK` |
| 14 | final evaluation | `E_FINAL` |
| 15 | Merkle / Poseidon2 anchors | `E_MERKLE` |

For `relation_id = 3` checks 11–12 use the §5.4.7 layout and checks 13–17 replace 13–15:

| # | Check (relation 3) | Code |
|---|---|---|
| 13 | `root_num == 0` and `root_den ≠ 0` | `E_GKR_ROOT` |
| 14 | masked GKR layers (round sums, layer equations) | `E_GKR` |
| 15 | Basefold sumcheck round sums | `E_SUMCHECK` |
| 16 | `y == FINAL · w̃(r)` | `E_FINAL` |
| 17 | per query, per layer: coset Merkle path, then fold consistency (incl. terminal `== FINAL`) | `E_MERKLE` / `E_FOLD` |

For `relation_id = 4` checks 11–12 use the §5.5.7 layout and checks 13–19 replace 13–15:

| # | Check (relation 4) | Code |
|---|---|---|
| 13 | `expected_circuit_root` given / equal to `STMT.root_circ` | `E_CIRCUIT_UNPINNED` / `E_CIRCUIT_PINNED` |
| 14 | `root_num == 0` and `root_den ≠ 0` | `E_GKR_ROOT` |
| 15 | masked GKR layers | `E_GKR` |
| 16 | zerocheck round sums and final gate equation | `E_ZEROCHECK` |
| 17 | batched Basefold sumcheck round sums | `E_SUMCHECK` |
| 18 | `y == FINAL · W̃(θ, r_1, r')` | `E_FINAL` |
| 19 | per query: `F` and `C` layer-0 Merkle paths, folded-layer paths, fold consistency | `E_MERKLE` / `E_FOLD` |

For `relation_id = 5` checks 11–12 use the §5.6.6 layout and checks 13–17 replace 13–15:

| # | Check (relation 5) | Code |
|---|---|---|
| 13 | `expected_circuit_root` given / equal to `STMT.weight_root` | `E_CIRCUIT_UNPINNED` / `E_CIRCUIT_PINNED` |
| 14 | inner-product sumcheck round sums | `E_ZEROCHECK` |
| 15 | batched Basefold sumcheck round sums | `E_SUMCHECK` |
| 16 | `y == FINAL · W̃(θ, r_1, r')` | `E_FINAL` |
| 17 | per query: `F` and `C` layer-0 Merkle paths, folded-layer paths, fold consistency | `E_MERKLE` / `E_FOLD` |

For `relation_id = 6` checks 11–12 use the §5.7.6 layout and checks 13–17 are §5.7.7's:

| # | Check (relation 6) | Code |
|---|---|---|
| 13 | `expected_circuit_root` / `expected_x_root` given, then equal to `STMT.weight_root` / `STMT.x_root` | `E_CIRCUIT_UNPINNED`, `E_X_ROOT_UNPINNED`, `E_CIRCUIT_PINNED`, `E_X_ROOT_PINNED` (in this order) |
| 14 | inner-product sumcheck round sums | `E_ZEROCHECK` |
| 15 | batched Basefold sumcheck round sums | `E_SUMCHECK` |
| 16 | `y == FINAL · W̃(θ, r_1, r')` | `E_FINAL` |
| 17 | per query: `F`, `C` and `X_com` layer-0 Merkle paths, folded-layer paths, fold consistency | `E_MERKLE` / `E_FOLD` |

For `relation_id = 2` (`LLM_INFERENCE_V2`, verified via `fpn-verify --circuit-desc <desc.bin> --circuit-root <hex64> --stmt-digest <hex64>` or `--stmt-file <path>`), the body layout is determined by the canonical public `CircuitDesc` binary (`magic = "FPN2DESC"`, `version = 1`, `circuit_desc_root = BLAKE2b-256(person="FPN2-CDESC-v2", raw_desc)`), with container body `circuit_root (32 B) ‖ stmt_digest (32 B) ‖ group_roots (32·G B) ‖ GKR_batches ‖ Sumchecks ‖ multi_pcs_proof` followed by all-zero padding to exact `C_p = 8,388,608 B` (`fpn2-llm-std-v1`). Pins are checked before decoding `raw_desc`, all descriptor indices/alignments/non-overlap bounds are validated before evaluation, only linear GKR numerator tags `0`/`1` are accepted, `STANDARD` tier requires `queries >= 72`, and each GKR batch uses `m_mask >= 2` slots (`m_mask - 1` cancelling pairs plus one independent zero-numerator denominator-blinder pair `(0, d_L)` / `(0, d_R)`). Checks 13–17 replace 13–15:

| # | Check (relation 2, with `--circuit-desc`) | Code |
|---|---|---|
| 13 | Pin-before-decode: `expected_circuit_root` == `circuit_desc_root(desc)` == `BODY.circuit_root`, `expected_stmt_digest` == `BODY.stmt_digest`, `--x-root` rejected; descriptor validated and `queries >= 72` under `STANDARD` | `E_CIRCUIT_UNPINNED`, `E_CIRCUIT_PINNED`, `E_STMT_UNPINNED`, `E_STMT_PINNED`, `E_X_ROOT_PINNED`, `E_REDUCED_SECURITY` |
| 14 | Every batched `multi_gkr` tree: `root_num == 0 && root_den ≠ 0`, masked layer rounds, and leaf MLE equations against committed views, cancelling mask pairs, and independent `(0, d_L)` / `(0, d_R)` denominator blinders | `E_GKR_ROOT` / `E_GKR` |
| 15 | Every Libra-masked sumcheck cube: round sums and final symbolic `CircuitDesc` AST evaluation | `E_SUMCHECK` |
| 16 | Batched multi-size `multi_pcs`: Step-A weighted inner-product sumcheck, Libra `L` opening, Step-B single-chain layer-injection Basefold sumcheck and `FINAL` check | `E_SUMCHECK` / `E_FINAL` |
| 17 | Per query, per layer: `B` and injected group Merkle paths (at `inject_layer(g) = (n_max - n_g)/2`), folded-layer Merkle paths, and coset-4 fold consistency to `FINAL` | `E_MERKLE` / `E_FOLD` |

Without `--circuit-desc`, `verify_bytes` preserves backward-compatible rejection with `E_BACKEND_UNAVAILABLE` (and rejects `--stmt-digest` on relations 1, 3, 4, 5, 6 with `E_STMT_PINNED`).

On success: `VERIFIED`. The CLI `fpn-verify <file> [--allow-reduced-security] [--profile <name>] [--circuit-root|--weight-root <hex64>] [--x-root <hex64>] [--circuit-desc <path>] [--stmt-digest <hex64> | --stmt-file <path>]` prints `VERIFIED` and exits `0`, or prints `REJECTED: <code>` and exits `1`. Both CLIs (Rust and detached Python) enforce mutual exclusion and once-only parsing for `--stmt-digest` and `--stmt-file` (repeating either flag or passing both in either order exits `2`). Usage errors and unreadable files exit `2`. No input may cause a panic or crash.

For full interactive 12-round `reply_zk` proofs using the folded backend (`fpn2-folded-v1`), `reply_bundle` provides a transitional legacy-style fixed-size uncompressed envelope (`FPNZK2\n` + le64 `container_bytes` + le64 `body_len` + JSON/base64 payload + all-zero tail padding to `8,388,608 B`). Unlike `FPN2`'s strict Rule-1/Rule-3 binary containers, `FPNZK2` carries an inner `body_len` field and JSON encoding solely to frame interactive `reply_zk` proofs without gzip length-oracle side channels until those 12 rounds are lowered into a self-contained `FPN2` `CircuitDesc`. The verifier enforces that the received envelope size matches the trusted registered profile's fixed container size (`8,388,608 B` by default, rejecting mismatched self-consistent sizes with `E_LENGTH`), rejects any non-zero padding byte (`E_PADDING`), enforces `queries >= 72` (`E_REDUCED_SECURITY`), and rejects unpinned or mismatched external-seed statements (`E_CPRE_UNPINNED` / `E_CPRE_MISMATCH` / `E_CEXT_MISMATCH`).

---

## 7. Security Tiers and Downgrade Rejection

A verifier deployed for production pins `--profile` and never passes `--allow-reduced-security`. A container naming a `REDUCED` profile is then rejected at step 8 before any parsing of its body. Because `profile_id` is a hash of the full canonical profile (including `security_tier`, `security_bits`, `n_queries`, `C_p`), a prover cannot relabel a reduced-parameter proof as standard without changing `profile_id`, which fails step 6 or step 7.

---

## 8. Golden Vectors

- `specs/vectors/fpn2_primitives.json`: `Fp`/`Fp3` operations (edges `0, 1, P−1, 2^32`) and decode cases; Poseidon2 permutation, compression, sponges and Merkle roots/paths (expected values computed by `flashprove_next`); BLAKE2b RFC 7693 and boundary-length KATs (computed by `hashlib`); profile canonical bytes and ids; transcript step-by-step checkpoints.
- `specs/vectors/fpn2_containers.json`: valid containers plus single-mutation negatives with the exact expected code. Containers are stored as `prefix_hex` + `total_len` (the remaining bytes are zero).
- `specs/vectors/fpn2_folded_gkr.json`: relation-3 containers produced by the backend prover with a deterministic DRBG: honest proofs for both relation-3 profiles, adversarial-prover proofs (adaptive Libra-mask tamper, forged children, inconsistent folded layer, query outside the table, wrong multiplicity, zero leaf denominator, forged roots) and single-mutation negatives of the standard-profile proof (tampered coset Merkle path / value, PCS round / fold root / final, GKR mask sum / round / children / `g_r`, `root_den = 0`, `root_num ≠ 0`, non-canonical limbs, statement swap, padding, length). A vector is either `prefix_hex` + `total_len` or `base` + `patches` (`[offset, hex]`) + `truncate` + `append_hex`.
- `specs/vectors/fpn2_folded_air.json`: relation-4 containers (each vector carries its `expected_circuit_root`): honest proofs of a tiny circuit (`n12`) and of the lowered `rmsnorm + gemm + microprogram` demo circuit (`v1`); adversarial provers (bad gate, bad selector gate, bad recomposition, copy-constraint violation, out-of-table limb, wrong multiplicity with and without a forged root, tampered circuit under its own / the honest root (with tampered or honest circuit-leaf openings), adaptive zerocheck claim, inconsistent folded layer); single mutations (circuit pin missing / wrong, profile pin, downgrade without `--allow-reduced-security`, padding, length, non-canonical limb, statement roots, GKR root / round / mask, zerocheck sum / round / selector evaluation, `z_blind`, batching and fold rounds, fold root, final, witness / circuit leaf values and paths).
- `specs/vectors/fpn2_folded_vec.json`: relation-5 containers of `fpn2-folded-vec-n16-v1` (each vector carries its `expected_circuit_root` = weight root): the honest proof of one `32 × 1024` tile of 16-bit signed integers; pins (profile, weight root missing / wrong); adversarial provers (tampered `X`, tampered `Y`, valid proof of tampered weights under its own root, tampered weights smuggled under the honest root, adaptive `e_W`, inconsistent folded layer — bytes the verifier never reads before rejecting are stored as zero and say so in `note`); single mutations (padding, length, non-canonical limbs, statement roots / `N_s`, `G_sum`, inner-product rounds, `e_W`, `z_blind`, batching / fold rounds, fold root, final, witness / weight leaves and paths).
- `specs/vectors/fpn2_folded_vec_bound.json`: relation-6 containers (each vector carries its `expected_circuit_root` = weight root and `expected_x_root`): honest proofs (one tile of `fpn2-folded-vec-bound-n16-v1`, a second statement over other activations, and 64 tiles of `-n22-v1`); pins (profile, weight root / `x_root` missing, wrong, both, and their precedence); **case-(c) forgeries** (a valid-in-itself proof of `(X', Y' = W X')` claiming the honest `x_root`, under the cheater's own `x_root'`, and with a forged `X̃(s)`); adversarial provers (tampered `X`, tampered `Y`, tampered weights under their own / the honest root, adaptive `e_W` / inner-product rounds, inconsistent folded layer); single mutations (padding, length, relabelled profile, non-canonical limbs including an `X_com` leaf, statement roots / `N_s`, `G_sum`, inner-product rounds, `e_W`, `z_blind`, batching / fold rounds, fold root, final, witness / weight / `X_com` leaves and paths).
- Regenerate with `python3.14 specs/vectors/generate.py`; `--check` verifies the committed files are byte-identical to a fresh generation.
