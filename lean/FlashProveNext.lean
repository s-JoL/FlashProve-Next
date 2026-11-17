/-
FlashProve-Next: machine-checked core of the protocol (Lean 4.29.1 + Mathlib).

The 19 modules fall into four layers. Each layer depends only on the ones above it and on Mathlib;
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
