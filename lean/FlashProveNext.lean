/-
FlashProve-Next: machine-checked core of the protocol (Lean 4.29.1 + Mathlib).

The 28 modules fall into eight layers. Each layer depends only on the ones above it and on Mathlib;
`THEOREM.md` maps every paper-level claim to its Lean declaration and lists what stays outside Lean.

1. Algebraic & combinatorial soundness primitives. Low-degree agreement and spot-check miss bounds.
   * `SpotCheck`, `Grinding`      : cost-weighted miss bounds and adaptive random-oracle grinding.
   * `Sumcheck`, `SumcheckSound`  : univariate agreement and full `n`-round Sumcheck soundness `n d / |F|`.
   * `LogUp`                      : logarithmic-derivative lookups, merged zero-sum form, `(n_v + n_t) / |F|`.

2. Zero knowledge. Exact uniformity of masked transcripts.
   * `Mask`, `MaskJoint`          : surjective blinding, Reed–Solomon and `eq(z, ·)` joint masks,
                                    causal multi-round masking, Libra outer mask.

3. Reference semantics of the computation. Uniqueness of what an honest engine must produce.
   * `StdAlg`, `StdAlgOps`        : deterministic traces, rounded division, lookups, Top-K.
   * `Gadgets`, `BoundedLift`     : saturation, certified MoE routing, modular lift without wrap-around.
   * `BucketSampling`             : integer bucketed keyed sampling and its witness-free nucleus envelopes.
   * `Speculative`, `LinearAttn`,
     `ArchGeneral`                : speculative decoding, chunked recurrent attention, MLA, PP/EP sharding.

4. Lifetime soundness composition & constrained-mask binding. Concrete parameter and protocol bounds.
   * `LifetimeSoundness`          : multi-pool sampling, seed-map bias, exact sample-count table.
   * `RoundByRound`               : round-by-round lifetime soundness and the `F_{p^3}` Goldilocks budget.
   * `ConstrainedMask`            : one-hot automaton-state selection and multilinear Schwartz–Zippel
                                    binding of full-vocabulary decoding masks over `F_{p^3}`, `≤ 2^{-186}`.
   * `ProtocolBridge`             : protocol specification to compiled constraint system,
                                    `Pr[BadAccept] ≤ 2^{-40}` over `2^64` adaptive queries.

5. Executable bytes verifier (`specs/11-wire-format.md`, M4 Stages 1–3 and 7). Lean core only, no Mathlib import.
   * `Fpn2Wire`                   : strict `FPN2` parser (64-byte header, exact length `C_p`, canonical
                                    `< P` limbs, zero padding) and uniqueness of the accepted encoding.
   * `Fpn2Transcript`             : raw-byte Fiat–Shamir state machine over an abstract hash oracle,
                                    injective framing and domain separation.
   * `VerifyBytes`                : `verifyBytes` in the normative 15-step order and its refinement to
                                    the typed checks `TypedVerify`, with downgrade resistance.
   * `Fpn2Rel3`                   : `verifyBytesRel3` for relation 3 (`FOLDED_GKR_LOOKUP`: masked
                                    FracLogUp-GKR + fused coset-4 folded Basefold) and its refinement
                                    to `Rel3TypedVerify` over abstract BLAKE2b oracles.
   * `Fpn2Rel4`                   : `verifyBytesRel4` for relation 4 (`FOLDED_GKR_AIR`: Plonkish zerocheck +
                                    wiring/range FracLogUp-GKR + one batched folded opening of `F + θ C`,
                                    mandatory circuit pin) and its refinement to `Rel4TypedVerify`.
   * `Fpn2Rel5`                   : `verifyBytesRel5` for relation 5 (`FOLDED_VEC_MATMUL`: Libra-masked
                                    degree-3 inner-product sumcheck for public `32 × K` GEMM tiles + the
                                    relation-4 batched folded opening, mandatory weight pin), its
                                    refinement to `Rel5TypedVerify`, and the relation dispatcher
                                    `verifyBytesAny` over relations 1, 3, 4 and 5.

6. Concrete field bridge (M4 Stages 4–6, 8 and 9). Imports both Mathlib and the bytes verifier.
   * `Fpn2Soundness`              : `p` is prime, `X^3 − 2` is irreducible, `E = F_p[X]/(X^3 − 2)` has
                                    `p^3` elements; the executable `Fp3` arithmetic (`add`, `sub`, `mul`,
                                    `fpInv`, `lagrangeEval`) is the arithmetic of `E`; Sumcheck, LogUp and
                                    multilinear Schwartz–Zippel instantiated at `E`; the relation-4
                                    algebraic error budget `≤ 2^{-160}`; `ProtocolBridge` instantiated
                                    over `E` (`≤ 2^{-40}`, `≤ 2^{-20}`; the sampling escape bound
                                    `(1 − σ)^k` derived from `SpotCheck` for every `k`, `σ`); acceptance
                                    by `verifyBytesRel4` as equations of `E`, including `root_den ≠ 0`
                                    and the query phase (`fold4` is the coset-4 fold of `E`). Stage 8:
                                    the same for relation 5 (`verifyBytesRel5_accept_field`,
                                    `rel5_alg_error_lt`, `rel5_lifetime_soundness_k105` / `_k51` /
                                    `_k100`), as a model and a budget, not a soundness theorem for
                                    `verifyBytesRel5`. Stage 9: every fold step of an accepted
                                    query is at a layer `t < T` (`foldRun_field`), and every decoded
                                    coset limb is canonical (`AirQueryCanonical`).

7. IR operator translation validation (M4-2). Imports Mathlib, `Gadgets` and `Fpn2Soundness`.
   * `Fpn2Gadgets`                : `constraint satisfied → integer relation holds` for the core IR
                                    operators over `ZMod goldilocks`: signed lift under the `2^62` bound
                                    of `Program.check_bounds`; the 7-bit limb plan (`AIR_LIMB_BITS = 7`,
                                    table `[0, 128)`, recompose + tighten gates) as a signed range; the
                                    BF16 `(s, e, m)` decode against the IEEE-754 value; the MoE
                                    `TopK_τ` bitmap / margin / `rmx` gates; the RMSNorm floor and the
                                    `rsqrt` enclosure; the Gated DeltaNet / LRec state step and its box
                                    propagation. The BF16 decode, `norm_rsqrt` and the in-range state
                                    update are reference predicates in the production lowering, so those
                                    theorems are about the predicate, not about verifier-checked gates.

8. End-to-end instantiation at the executable verifiers (M4-3). Imports `MaskJoint`, `Fpn2Soundness`
   and `Fpn2Gadgets`.
   * `Fpn2EndToEnd`               : the bad-accept event of `verifyBytesRel4` / `verifyBytesRel5`
                                    (accepts a claim whose statement is false) and
                                    `ProtocolBridge.ProtocolRBRSpec` built for it from the explicit
                                    record `CryptoAssumptions` (random oracle, extraction and
                                    proximity, IR correspondence, and the unformalized
                                    round-by-round analysis); `≤ 2^-40` (`k = 105`) and `≤ 2^-20`
                                    (`k = 51`, `k = 100`) conditional on that record. A
                                    `ProtocolRBRSpec` with proved obligations for the plain
                                    sumcheck. `CompiledCS` at a Plonkish system and the lift of a
                                    satisfying witness to the integer relations of `Fpn2Gadgets`.
                                    Witness-independence of the linear view under the joint mask
                                    (Libra, `ZK_ROWS = 3`, `z_blind`), one transcript, fixed
                                    challenges. Not an unconditional soundness or zero-knowledge
                                    theorem for the deployed verifier.
-/
-- Layer 1: algebraic & combinatorial soundness primitives
import FlashProveNext.SpotCheck
import FlashProveNext.Grinding
import FlashProveNext.Sumcheck
import FlashProveNext.SumcheckSound
import FlashProveNext.LogUp
-- Layer 2: zero knowledge
import FlashProveNext.Mask
import FlashProveNext.MaskJoint
-- Layer 3: reference semantics of the computation
import FlashProveNext.StdAlg
import FlashProveNext.StdAlgOps
import FlashProveNext.Gadgets
import FlashProveNext.BoundedLift
import FlashProveNext.BucketSampling
import FlashProveNext.Speculative
import FlashProveNext.LinearAttn
import FlashProveNext.ArchGeneral
-- Layer 4: lifetime soundness composition & constrained-mask binding
import FlashProveNext.LifetimeSoundness
import FlashProveNext.RoundByRound
import FlashProveNext.ConstrainedMask
import FlashProveNext.ProtocolBridge
-- Layer 5: executable bytes verifier
import FlashProveNext.Fpn2Wire
import FlashProveNext.Fpn2Transcript
import FlashProveNext.VerifyBytes
import FlashProveNext.Fpn2Rel3
import FlashProveNext.Fpn2Rel4
import FlashProveNext.Fpn2Rel5
-- Layer 6: concrete Goldilocks / F_{p^3} soundness bridge
import FlashProveNext.Fpn2Soundness
-- Layer 7: IR operator translation validation (M4-2)
import FlashProveNext.Fpn2Gadgets
-- Layer 8: end-to-end instantiation at the executable verifiers (M4-3)
import FlashProveNext.Fpn2EndToEnd
