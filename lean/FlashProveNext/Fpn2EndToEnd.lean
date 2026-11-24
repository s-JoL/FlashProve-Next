/-
M4-3: end-to-end instantiation of `ProtocolBridge.ProtocolRBRSpec` and `RoundByRound` at the
executable bytes verifiers `verifyBytesRel4` (`fpn2-folded-air-v1`, `fpn2-folded-air-n20-v1`) and
`verifyBytesRel5` (`fpn2-folded-vec-n16-v1`, `fpn2-folded-vec-n22-v1`).

What this module is. `Fpn2Soundness` instantiated `ProtocolRBRSpec` at a model whose bad event is
*defined* as the escape event (`rel4BadAcc`), so its `hInit` / `hFinal` / `hStep` hold by
construction. Here the bad event is defined from the executable verifier:

    badAccept = { a | ∃ claim, verifyBytesRel4 (oracles a) … (proof a) = .ok claim ∧ ¬ Statement claim }

and `ProtocolRBRSpec` is built for *that* event from an explicit record of assumptions,
`CryptoAssumptions` (relation 4) / `CryptoAssumptions5` (relation 5). The lifetime theorems are
conditional on that record. They are not unconditional soundness theorems for the verifier.

1. Bytes to field, per registered profile. `rel4_accept_e2e`, `rel4_accept_prodAirV1`,
   `rel4_accept_prodAirN20`, `rel5_accept_e2e`, `rel5_accept_prodVecN16`, `rel5_accept_prodVecN22`,
   `rel4_layout_fold_step_cubic`: acceptance gives `Rel4FieldEquations` / `Rel5FieldEquations` on the
   pinned layout, the algebraic budget (`≤ 2^-160`, relation 5 `≤ 2^-183`) and the fold of a cubic
   on the evaluation domain (`foldField_cubic`). Unconditional.
2. `Rel4Game`, `Rel4Game.badAccept`, `CryptoAssumptions`, `rel4Spec`, `rel4_e2e_lifetime_general`,
   `rel4_e2e_lifetime_k105` / `_k51` / `_k100`; `Rel5Game`, `CryptoAssumptions5`, `rel5Spec`,
   `rel5_e2e_lifetime_*`; the four production games `Rel4Game.prodAirV1`, `Rel4Game.prodAirN20`,
   `Rel5Game.prodVecN16`, `Rel5Game.prodVecN22` with `badAccept_prod*` (the event written out with
   the literal verifier call). `CryptoAssumptions.ofFailUniv` : the record is inhabited for every
   game at `Δ = 1`, so it is consistent and all of its strength is in the size of `Δ`.
3. `runClaim`, `accepts_runClaim`, `sumcheck_step_escapes`, `sumcheckAcc`, `sumcheckRbrSpec`,
   `sumcheck_rbr_lifetime`: a `ProtocolRBRSpec` for the `n`-round sumcheck of `SumcheckSound` whose
   three obligations are *proved* (state function `claim_i = psum P i r`), with no assumption
   record. It shows what discharging `rbr_round_analysis` looks like for one sub-protocol.
4. `AirRow`, `AirSystem`, `AirSystem.compiled`, `air_rows_lift_sound`, `range_checked_wire_lift`,
   `rel4_good_accept_integer_rows`, `LiftEq`, `MoeFieldGates`, `moeGates_of_field`,
   `moe_routing_of_field_gates`, `gdn_of_field_gates`, `gadget_relations_of_witness`:
   a satisfying witness of a Plonkish system (rows + `[0, 128)` limb cells) under the `2^62` lift
   bounds satisfies the integer relations of `Fpn2Gadgets`.
5. `masked_view_uniform`, `blockHom`, `blockHom_surjective`, `zkRowsHom`, `zkRowsHom_surjective`,
   `rel4JointMask`, `rel4_joint_view_independent`, `rel4_joint_view_uniform`,
   `rel5JointMask`, `rel5_joint_view_independent`, `rel5_joint_view_uniform`,
   `rel4_joint_view_fp3`, `constrained_mask_binding_fp3`: the linear view of one transcript
   at fixed challenges (Libra masks, the `ZK_ROWS = 3` blinding rows, `z_blind`) is distributed
   independently of the witness.

Scope (what is NOT proved here).
(a) Every lifetime theorem takes a `CryptoAssumptions` record. Its fields are listed, one by one,
    in `THEOREM.md` §2.4. The field `rbr_round_analysis` is not a cryptographic assumption: it is
    the unformalized round-by-round analysis of the GKR / zerocheck / Libra / Basefold-sumcheck
    composition together with the identification of each Fiat–Shamir challenge with a
    random-oracle answer. Lean proves the escape-set densities (`rel4Round_esc_card_le`) and the
    counting (`RoundByRound`), not that analysis. Item 3 proves it for the plain sumcheck only.
(b) The sample space `Fin Q → Rel4Ans k` and its counting measure are a model of a lazily sampled
    random oracle. The maps `oracles`, `pub`, `proof` from a sample to the hash functions and the
    adversary's bytes are data of the game, not derived from BLAKE2b; the bias of `squeeze_fp3`
    and of the deployed sampler is not modelled (`Fpn2Soundness`, Scope (a), (e)).
(c) Item 4 is about a system given as Lean data (`AirSystem`). That the Python lowering emits that
    system and that `root_circ` commits to it is the field `ir_correspondence` plus the choice of
    `G.cs`; the lift bounds `LiftBounds` are the output of `Program.check_bounds`, taken as a
    hypothesis. `bf16_decode` is a reference predicate the verifier does not check: in
    `gadget_relations_of_witness` it is a separate hypothesis, not a consequence of
    acceptance. `MoeFieldGates` and `gdn_of_field_gates` are about free integers: only
    `rel4_good_accept_integer_rows` and `range_checked_wire_lift` are tied to the witness.
(d) Item 5 is a statement about a linear map at fixed challenges, for one transcript. It does not
    cover the non-linear messages (GKR inner layers, round polynomials as functions of earlier
    challenges), the Merkle openings, the simulation of the random oracle or several proofs. The
    nondegeneracy conditions (`2 ≠ 0`, `ρ ≠ 0`, `β ≠ 0`, the blinding weights span `E` over `F_p`)
    are hypotheses. `ConstrainedMask` is a binding (soundness) lemma for decoding masks, not a
    masking lemma; `constrained_mask_binding_fp3` restates it at `E` and plays no role in the
    view-independence proof. `BucketSampling` is about the sampler of the decoding relation and is
    not used by the relation-4 / relation-5 verifiers; the sampling term of the lifetime bound comes
    from `SpotCheck.frac_bad_bound` through `Rel4Round.spot`.
-/
import Mathlib
import FlashProveNext.MaskJoint
import FlashProveNext.Fpn2Soundness
import FlashProveNext.Fpn2Gadgets

set_option autoImplicit false

namespace FlashProveNext
namespace Fpn2EndToEnd

open Finset

set_option linter.unusedSectionVars false
set_option exponentiation.threshold 2048

/-! ## 1. Bytes to field, with the budget, per registered profile -/

/-- **Relation 4, bytes to field with the budget.** Acceptance by `verifyBytesRel4` gives the parsed
container, the pinned circuit root, the field equations on the layout of the profile and the
algebraic budget `rel4AlgNumer / p^3 ≤ 2^-160`. -/
theorem rel4_accept_e2e {O : Rel4Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {circuitRoot : Option Bytes} {raw : ByteArray} {claim : Rel4Claim}
    (h : verifyBytesRel4 O pub prof allowReduced circuitRoot raw = .ok claim) :
    ∃ (c : Rel4Container) (lay : Rel4Layout) (g : GkrState) (zf : RoundState) (pf : PcsState),
      parseContainerRel4 prof raw = .ok c ∧ circuitRoot = some c.body.stmt.rootCirc ∧
      claim = c.acceptedClaim ∧ rel4Layout prof = some lay ∧
      Rel4FieldEquations O pub lay c g zf pf ∧
      (rel4AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ := by
  obtain ⟨c, lay, g, zf, pf, hparse, _, hpin, hc, hlay, hn, hF⟩ := verifyBytesRel4_accept_field h
  exact ⟨c, lay, g, zf, pf, hparse, hpin, hc, hlay, hF, (rel4_alg_error_lt lay hn.2).1⟩

/-- `fpn2-folded-air-v1` (`n = 14`, `Q = 72`). -/
theorem rel4_accept_prodAirV1 {O : Rel4Oracles} {pub : PublicInput} {allowReduced : Bool}
    {circuitRoot : Option Bytes} {raw : ByteArray} {claim : Rel4Claim}
    (h : verifyBytesRel4 O pub foldedAirV1 allowReduced circuitRoot raw = .ok claim) :
    ∃ (c : Rel4Container) (g : GkrState) (zf : RoundState) (pf : PcsState),
      parseContainerRel4 foldedAirV1 raw = .ok c ∧ circuitRoot = some c.body.stmt.rootCirc ∧
      claim = c.acceptedClaim ∧ Rel4FieldEquations O pub ⟨14, 72⟩ c g zf pf ∧
      (rel4AlgNumer ⟨14, 72⟩ : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ := by
  obtain ⟨c, lay, g, zf, pf, hparse, hpin, hc, hlay, hF, hb⟩ := rel4_accept_e2e h
  obtain rfl : lay = ⟨14, 72⟩ := Option.some.inj (hlay.symm.trans rel4Layout_foldedAirV1.1)
  exact ⟨c, g, zf, pf, hparse, hpin, hc, hF, hb⟩

/-- `fpn2-folded-air-n20-v1` (`n = 20`, `Q = 72`). -/
theorem rel4_accept_prodAirN20 {O : Rel4Oracles} {pub : PublicInput} {allowReduced : Bool}
    {circuitRoot : Option Bytes} {raw : ByteArray} {claim : Rel4Claim}
    (h : verifyBytesRel4 O pub foldedAirN20V1 allowReduced circuitRoot raw = .ok claim) :
    ∃ (c : Rel4Container) (g : GkrState) (zf : RoundState) (pf : PcsState),
      parseContainerRel4 foldedAirN20V1 raw = .ok c ∧ circuitRoot = some c.body.stmt.rootCirc ∧
      claim = c.acceptedClaim ∧ Rel4FieldEquations O pub ⟨20, 72⟩ c g zf pf ∧
      (rel4AlgNumer ⟨20, 72⟩ : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ := by
  obtain ⟨c, lay, g, zf, pf, hparse, hpin, hc, hlay, hF, hb⟩ := rel4_accept_e2e h
  obtain rfl : lay = ⟨20, 72⟩ := Option.some.inj (hlay.symm.trans rel4Layout_foldedAirN20.1)
  exact ⟨c, g, zf, pf, hparse, hpin, hc, hF, hb⟩

/-- **Relation 5, bytes to field with the budget.** Acceptance by `verifyBytesRel5` gives the parsed
container, the pinned weight root, canonical limbs, the field equations on the layout of the
profile and the algebraic budget `rel5AlgNumer / p^3 ≤ 2^-183`. -/
theorem rel5_accept_e2e {O : Rel5Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {weightPin : Option Bytes} {raw : ByteArray} {claim : Rel5Claim}
    (h : verifyBytesRel5 O pub prof allowReduced weightPin raw = .ok claim) :
    ∃ (c : Rel5Container) (lay : Rel5Layout) (sf : RoundState) (pf : PcsState),
      parseContainerRel5 prof raw = .ok c ∧ weightPin = some c.body.stmt.weightRoot ∧
      claim = c.acceptedClaim ∧ rel5Layout prof = some lay ∧ Rel5Canonical lay c ∧
      Rel5FieldEquations O pub lay c sf pf ∧
      (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 183)⁻¹ := by
  obtain ⟨c, lay, sf, pf, hparse, _, hpin, hc, hlay, hadm, hcan, hF⟩ :=
    verifyBytesRel5_accept_field h
  exact ⟨c, lay, sf, pf, hparse, hpin, hc, hlay, hcan, hF,
    (rel5_alg_error_lt lay hadm.2.2.1 hadm.2.2.2).1⟩

/-- `fpn2-folded-vec-n16-v1` (`n = 16`, one tile). -/
theorem rel5_accept_prodVecN16 {O : Rel5Oracles} {pub : PublicInput} {allowReduced : Bool}
    {weightPin : Option Bytes} {raw : ByteArray} {claim : Rel5Claim}
    (h : verifyBytesRel5 O pub foldedVecN16V1 allowReduced weightPin raw = .ok claim) :
    ∃ (c : Rel5Container) (sf : RoundState) (pf : PcsState),
      parseContainerRel5 foldedVecN16V1 raw = .ok c ∧ weightPin = some c.body.stmt.weightRoot ∧
      claim = c.acceptedClaim ∧ Rel5FieldEquations O pub ⟨16, 72, 10⟩ c sf pf ∧
      (rel5AlgNumer ⟨16, 72, 10⟩ : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 183)⁻¹ := by
  obtain ⟨c, lay, sf, pf, hparse, hpin, hc, hlay, _, hF, hb⟩ := rel5_accept_e2e h
  obtain rfl : lay = ⟨16, 72, 10⟩ := Option.some.inj (hlay.symm.trans rel5Layout_foldedVecN16.1)
  exact ⟨c, sf, pf, hparse, hpin, hc, hF, hb⟩

/-- `fpn2-folded-vec-n22-v1` (`n = 22`, 64 tiles). -/
theorem rel5_accept_prodVecN22 {O : Rel5Oracles} {pub : PublicInput} {allowReduced : Bool}
    {weightPin : Option Bytes} {raw : ByteArray} {claim : Rel5Claim}
    (h : verifyBytesRel5 O pub foldedVecN22V1 allowReduced weightPin raw = .ok claim) :
    ∃ (c : Rel5Container) (sf : RoundState) (pf : PcsState),
      parseContainerRel5 foldedVecN22V1 raw = .ok c ∧ weightPin = some c.body.stmt.weightRoot ∧
      claim = c.acceptedClaim ∧ Rel5FieldEquations O pub ⟨22, 72, 10⟩ c sf pf ∧
      (rel5AlgNumer ⟨22, 72, 10⟩ : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 183)⁻¹ := by
  obtain ⟨c, lay, sf, pf, hparse, hpin, hc, hlay, _, hF, hb⟩ := rel5_accept_e2e h
  obtain rfl : lay = ⟨22, 72, 10⟩ := Option.some.inj (hlay.symm.trans rel5Layout_foldedVecN22.1)
  exact ⟨c, sf, pf, hparse, hpin, hc, hF, hb⟩

/-- **A committed fold step on an admitted relation-4 layout folds a cubic.** The bound `n ≤ 28`
that `foldField_cubic` needs holds for every layout `rel4Layout` returns, so for every layout an
accepted proof can have (`rel4_accept_e2e`); the `FoldField` steps of an accepted proof are the
ones inside `Rel4FieldEquations.query`. -/
theorem rel4_layout_fold_step_cubic {O : Rel4Oracles} {prof : Fpn2Profile}
    {lay : Rel4Layout} (hlay : rel4Layout prof = some lay)
    {roots : List Bytes} {rs : List Fp3} {iotaInv : UInt64} {s : FoldState} {o : CosetOpening}
    {s' : FoldState} (hstep : FoldField O lay.pcs roots rs iotaInv s o s')
    (hi : rel3IotaInv = some iotaInv) (c0 c1 c2 c3 : Fp3Field)
    (h0 : o.values.1.toField = cubicAt lay.pcs s.t c0 c1 c2 c3 (s.pos % lay.pcs.quarterAt s.t))
    (h1 : o.values.2.1.toField
      = cubicAt lay.pcs s.t c0 c1 c2 c3 (s.pos % lay.pcs.quarterAt s.t + 1 * lay.pcs.quarterAt s.t))
    (h2 : o.values.2.2.1.toField
      = cubicAt lay.pcs s.t c0 c1 c2 c3 (s.pos % lay.pcs.quarterAt s.t + 2 * lay.pcs.quarterAt s.t))
    (h3 : o.values.2.2.2.toField
      = cubicAt lay.pcs s.t c0 c1 c2 c3
          (s.pos % lay.pcs.quarterAt s.t + 3 * lay.pcs.quarterAt s.t)) :
    s'.prev.toField
      = c0 + (rs.getD (2 * s.t) Fp3.zero).toField * c1
        + (rs.getD (2 * s.t + 1) Fp3.zero).toField * c2
        + (rs.getD (2 * s.t) Fp3.zero).toField * (rs.getD (2 * s.t + 1) Fp3.zero).toField * c3 := by
  have hn : lay.pcs.nVars ≤ 28 := (rel4Layout_nVars_le hlay).2
  exact foldField_cubic hstep hi hn c0 c1 c2 c3 h0 h1 h2 h3

/-! ## 2. The bad-accept event of the executable verifier and the assumption record -/

/-- The sample space: `Q` lazily sampled random-oracle answers, each one algebraic challenge in
`E` and `k` sampling seeds (`Rel4Ans k`). -/
abbrev Sample (k Q : ℕ) : Type := Fin Q → Rel4Ans k

/-- **A relation-4 soundness game.** A sample `a` of the random oracle determines the two hash
functions the verifier calls (`oracles a`), the out-of-band public input and the bytes the
adversary outputs (`proof a`). The statement is relative to a compiled constraint system `cs` over
witnesses `W` and a commitment relation `Commits claim w` ("the accepted `root_0` opens to `w`").
All of this is data: nothing here is assumed about it. -/
structure Rel4Game (k Q : ℕ) (W : Type) where
  prof : Fpn2Profile
  lay : Rel4Layout
  hlay : rel4Layout prof = some lay
  allowReduced : Bool
  circuitRoot : Bytes
  oracles : Sample k Q → Rel4Oracles
  pub : Sample k Q → PublicInput
  proof : Sample k Q → ByteArray
  cs : CompiledCS W
  Commits : Rel4Claim → W → Prop

namespace Rel4Game

variable {k Q : ℕ} {W : Type}

/-- The executable verifier accepts the adversary's bytes with claim `claim`. -/
def Accepts (G : Rel4Game k Q W) (a : Sample k Q) (claim : Rel4Claim) : Prop :=
  verifyBytesRel4 (G.oracles a) (G.pub a) G.prof G.allowReduced (some G.circuitRoot) (G.proof a)
    = .ok claim

/-- The statement of an accepted claim: some witness under the accepted commitment satisfies the
compiled constraint system. -/
def Statement (G : Rel4Game k Q W) (claim : Rel4Claim) : Prop :=
  ∃ w, G.Commits claim w ∧ G.cs.Satisfies w

open Classical in
/-- **The bad-accept event, defined from the executable verifier**: the samples on which
`verifyBytesRel4` accepts a claim whose statement is false. -/
noncomputable def badAccept (G : Rel4Game k Q W) : Finset (Sample k Q) :=
  univ.filter fun a => ∃ claim, G.Accepts a claim ∧ ¬ G.Statement claim

theorem mem_badAccept (G : Rel4Game k Q W) (a : Sample k Q) :
    a ∈ G.badAccept ↔ ∃ claim, G.Accepts a claim ∧ ¬ G.Statement claim := by
  simp [badAccept]

/-- Acceptance in the game gives the field equations on the layout of the game. -/
theorem accept_field (G : Rel4Game k Q W) {a : Sample k Q} {claim : Rel4Claim}
    (h : G.Accepts a claim) :
    ∃ (c : Rel4Container) (g : GkrState) (zf : RoundState) (pf : PcsState),
      parseContainerRel4 G.prof (G.proof a) = .ok c ∧
      Rel4FieldEquations (G.oracles a) (G.pub a) G.lay c g zf pf := by
  obtain ⟨c, lay, g, zf, pf, hparse, _, _, hlay, hF, _⟩ := rel4_accept_e2e h
  obtain rfl : lay = G.lay := Option.some.inj (hlay.symm.trans G.hlay)
  exact ⟨c, g, zf, pf, hparse, hF⟩

/-- Outside the bad event, an accepted claim has a witness. -/
theorem statement_of_good (G : Rel4Game k Q W) {a : Sample k Q} (hgood : a ∉ G.badAccept)
    {claim : Rel4Claim} (hacc : G.Accepts a claim) : G.Statement claim := by
  by_contra hns
  exact hgood ((G.mem_badAccept a).mpr ⟨claim, hacc, hns⟩)

end Rel4Game

/-- **The assumptions the relation-4 lifetime theorems rest on**, for a game `G`, a sampling
threshold `σ = num / den` and a failure mass `Δ`. Data fields say what is being assumed about;
`Prop` fields are the assumptions. No field mentions `badAccept`, and no field is the path premise
of `RoundByRound`: that premise is derived (`rel4Spec`). Caveat: `Fail` is free data and nothing
stops it from containing `badAccept`, in which case `fail_bound` alone is the conclusion. `Δ` is
where all the unproved probability mass sits.

Random oracle / Fiat–Shamir:
* `st`, `ro_lazy` : the round the adversary has prepared when it makes its `j`-th oracle query is
  a function of the other answers only (lazy sampling, state restoration).
* `Fail`, `fail_bound` : an event of mass at most `Δ` outside which binding and extraction work
  (hash collisions, Fiat–Shamir transcript collisions, extractor failure).

Basefold / FRI proximity and polynomial extraction:
* `extract`, `extract_binds` : outside `Fail`, the witness read off the oracle queries is one the
  accepted claim commits to.
* `pcs_proximity_final` : outside `Fail`, if the field equations of an accepted run hold (query
  phase included) then the last claim of the protocol is true of the extracted polynomials.

Python IR / Rust verifier correspondence:
* `ir_correspondence` : outside `Fail`, if the first claim of the protocol is true of the extracted
  witness then that witness satisfies the compiled constraint system `G.cs`.

Round-by-round analysis (not a cryptographic assumption, see the module header, Scope (a)):
* `R`, `claimTrue`, `qIdx` : the rounds of the verifier, the truth of the running claim after each
  round, and the oracle query that answers each round.
* `rbr_round_analysis` : outside `Fail`, a claim that is false before a round and true after it
  means the answer of that round's query fell in the escape set of the prepared round. -/
structure CryptoAssumptions {k Q : ℕ} {W : Type} (G : Rel4Game k Q W) (num den : ℕ) (Δ : ℝ) where
  st : Fin Q → Sample k Q → Rel4Round G.lay k num den
  ro_lazy : ∀ j a x, st j (Function.update a j x) = st j a
  Fail : Finset (Sample k Q)
  fail_bound : (Fail.card : ℝ) ≤ Δ * (Fintype.card (Rel4Ans k) : ℝ) ^ Q
  extract : Sample k Q → W
  extract_binds : ∀ a, a ∉ Fail → ∀ claim, G.Accepts a claim → G.Commits claim (extract a)
  R : Sample k Q → ℕ
  claimTrue : ∀ a : Sample k Q, Fin (R a + 1) → Prop
  qIdx : ∀ a : Sample k Q, Fin (R a) → Fin Q
  ir_correspondence : ∀ a, a ∉ Fail → claimTrue a 0 → G.cs.Satisfies (extract a)
  pcs_proximity_final : ∀ a, a ∉ Fail → ∀ c g zf pf,
    parseContainerRel4 G.prof (G.proof a) = .ok c →
    Rel4FieldEquations (G.oracles a) (G.pub a) G.lay c g zf pf →
    claimTrue a (Fin.last (R a))
  rbr_round_analysis : ∀ a, a ∉ Fail → ∀ c g zf pf,
    parseContainerRel4 G.prof (G.proof a) = .ok c →
    Rel4FieldEquations (G.oracles a) (G.pub a) G.lay c g zf pf →
    ∀ r : Fin (R a), ¬ claimTrue a r.castSucc → claimTrue a r.succ →
      a (qIdx a r) ∈ (st (qIdx a r) a).esc

/-- **Consistency of the record.** With `Fail = univ` (so `Δ = 1`) every assumption is vacuous and
the record is inhabited for every game. Hence the record cannot be contradictory, and everything it
asserts is carried by how small `Δ` is. -/
noncomputable def CryptoAssumptions.ofFailUniv {k Q : ℕ} {W : Type} (G : Rel4Game k Q W)
    (num den : ℕ) (w0 : W) : CryptoAssumptions G num den 1 where
  st := fun _ _ => .linear 0 1 (Or.inr one_ne_zero)
  ro_lazy := fun _ _ _ => rfl
  Fail := univ
  fail_bound := by simp
  extract := fun _ => w0
  extract_binds := fun a ha => absurd (mem_univ a) ha
  R := fun _ => 0
  claimTrue := fun _ _ => True
  qIdx := fun _ i => i.elim0
  ir_correspondence := fun a ha => absurd (mem_univ a) ha
  pcs_proximity_final := fun a ha => absurd (mem_univ a) ha
  rbr_round_analysis := fun a ha => absurd (mem_univ a) ha

/-- **`ProtocolRBRSpec` for the bad-accept event of `verifyBytesRel4`.** `hInit`: a false statement
and `extract_binds` make the extracted witness violate `G.cs`, so by `ir_correspondence` the first
claim is false. `hFinal`: acceptance gives the field equations (`verifyBytesRel4_accept_field`), so
by `pcs_proximity_final` the last claim is true. `hStep`: `rbr_round_analysis` on the same field
equations. -/
noncomputable def rel4Spec {k Q : ℕ} {W : Type} (G : Rel4Game k Q W) {num den : ℕ} {Δ : ℝ}
    (A : CryptoAssumptions G num den Δ) :
    ProtocolRBRSpec A.st (fun _ => True) Rel4Round.esc G.badAccept A.Fail where
  R := A.R
  roundHolds := A.claimTrue
  qIdx := A.qIdx
  hInit := fun a ha hf h0 => by
    obtain ⟨claim, hacc, hns⟩ := (G.mem_badAccept a).mp ha
    exact hns ⟨A.extract a, A.extract_binds a hf claim hacc, A.ir_correspondence a hf h0⟩
  hFinal := fun a ha hf => by
    obtain ⟨claim, hacc, _⟩ := (G.mem_badAccept a).mp ha
    obtain ⟨c, g, zf, pf, hparse, hF⟩ := G.accept_field hacc
    exact A.pcs_proximity_final a hf c g zf pf hparse hF
  hStep := fun a ha hf r h0 h1 => by
    obtain ⟨claim, hacc, _⟩ := (G.mem_badAccept a).mp ha
    obtain ⟨c, g, zf, pf, hparse, hF⟩ := G.accept_field hacc
    exact ⟨trivial, A.rbr_round_analysis a hf c g zf pf hparse hF r h0 h1⟩

/-- **General lifetime bound for `verifyBytesRel4`** under `CryptoAssumptions`:
`Pr[badAccept] ≤ Δ + Q · ((1 − σ)^k + 2^-106)`. The per-round densities are those proved in
`rel4Round_esc_card_le`; the path premise is `hpath_of_protocol_spec (rel4Spec G A)`. -/
theorem rel4_e2e_lifetime_general {k Q : ℕ} {W : Type} (G : Rel4Game k Q W) {num den : ℕ} {Δ : ℝ}
    (A : CryptoAssumptions G num den Δ) (hden : 0 < den) :
    ((G.badAccept).card : ℝ)
      ≤ (Δ + Q * (rel4Miss k num den + (2 ^ 106)⁻¹)) * (Fintype.card (Rel4Ans k) : ℝ) ^ Q := by
  have hm0 := rel4Miss_nonneg k num den
  exact rbr_lifetime_soundness A.st A.ro_lazy (fun _ => True) Rel4Round.esc Rel4Round.isSpot
    (rel4Eps (rel4Miss k num den)) (rel4Miss k num den + (2 ^ 106)⁻¹)
    (add_nonneg hm0 (by positivity))
    (fun b => by cases b <;> simp [rel4Eps, hm0])
    (fun s _ => rel4Round_esc_card_le G.lay (rel4Layout_nVars_le G.hlay).2 k num den hden s)
    G.badAccept A.Fail Δ A.fail_bound (hpath_of_protocol_spec (rel4Spec G A))

/-- **Non-interactive lifetime soundness of `verifyBytesRel4` (`k = 105`, `σ = 1/2`, `Q ≤ 2^64`,
`Δ ≤ 2^-42`):** `Pr[badAccept] ≤ 2^-40`, through `fpn_protocol_lifetime_soundness`. -/
theorem rel4_e2e_lifetime_k105 {Q : ℕ} {W : Type} (G : Rel4Game 105 Q W) {Δ : ℝ}
    (A : CryptoAssumptions G 1 2 Δ) (hQ : Q ≤ 2 ^ 64) (hΔ : Δ ≤ (2 ^ 42)⁻¹) :
    ((G.badAccept).card : ℝ) ≤ (2 ^ 40)⁻¹ * (Fintype.card (Rel4Ans 105) : ℝ) ^ Q :=
  fpn_protocol_lifetime_soundness hQ A.st A.ro_lazy (fun _ => True) Rel4Round.esc Rel4Round.isSpot
    (rel4Eps (rel4Miss 105 1 2)) (rel4Miss 105 1 2) (2 ^ 106)⁻¹ (2 ^ 106)⁻¹ Δ
    (rel4Miss_nonneg _ _ _) (rel4Miss_half 105).le
    (by positivity) le_rfl le_rfl hΔ
    (fun b => by cases b <;> simp [rel4Eps])
    (fun s _ => rel4Round_esc_card_le G.lay (rel4Layout_nVars_le G.hlay).2 105 1 2 (by norm_num) s)
    G.badAccept A.Fail A.fail_bound (rel4Spec G A)

/-- **External-seed lifetime soundness of `verifyBytesRel4` at `σ = 1/2` (`k = 51`, `Q ≤ 2^30`,
`Δ ≤ 2^-22`):** `Pr[badAccept] ≤ 2^-20`. -/
theorem rel4_e2e_lifetime_k51 {Q : ℕ} {W : Type} (G : Rel4Game 51 Q W) {Δ : ℝ}
    (A : CryptoAssumptions G 1 2 Δ) (hQ : Q ≤ 2 ^ 30) (hΔ : Δ ≤ (2 ^ 22)⁻¹) :
    ((G.badAccept).card : ℝ) ≤ (2 ^ 20)⁻¹ * (Fintype.card (Rel4Ans 51) : ℝ) ^ Q := by
  refine (rel4_e2e_lifetime_general G A (by norm_num)).trans
    (mul_le_mul_of_nonneg_right ?_ (by positivity))
  have hQr : (Q : ℝ) ≤ 2 ^ 30 := by exact_mod_cast hQ
  rw [rel4Miss_half]
  have h1 : (Q : ℝ) * ((1 / 2) ^ 51 + (2 ^ 106)⁻¹) ≤ 2 ^ 30 * ((1 / 2) ^ 51 + (2 ^ 106)⁻¹) :=
    mul_le_mul_of_nonneg_right hQr (by positivity)
  have h2 : (2 ^ 22 : ℝ)⁻¹ + 2 ^ 30 * ((1 / 2) ^ 51 + (2 ^ 106)⁻¹) ≤ (2 ^ 20)⁻¹ := by norm_num
  linarith

/-- **External-seed lifetime soundness of `verifyBytesRel4` at `σ = 3/10` (`k = 100`, `Q ≤ 2^30`,
`Δ ≤ 2^-22`):** `Pr[badAccept] ≤ 2^-20`. -/
theorem rel4_e2e_lifetime_k100 {Q : ℕ} {W : Type} (G : Rel4Game 100 Q W) {Δ : ℝ}
    (A : CryptoAssumptions G 3 10 Δ) (hQ : Q ≤ 2 ^ 30) (hΔ : Δ ≤ (2 ^ 22)⁻¹) :
    ((G.badAccept).card : ℝ) ≤ (2 ^ 20)⁻¹ * (Fintype.card (Rel4Ans 100) : ℝ) ^ Q := by
  refine (rel4_e2e_lifetime_general G A (by norm_num)).trans
    (mul_le_mul_of_nonneg_right ?_ (by positivity))
  have hQr : (Q : ℝ) ≤ 2 ^ 30 := by exact_mod_cast hQ
  rw [rel4Miss_three_tenths]
  have h1 : (Q : ℝ) * ((7 / 10) ^ 100 + (2 ^ 106)⁻¹) ≤ 2 ^ 30 * ((7 / 10) ^ 100 + (2 ^ 106)⁻¹) :=
    mul_le_mul_of_nonneg_right hQr (by positivity)
  have h2 : (2 ^ 22 : ℝ)⁻¹ + 2 ^ 30 * ((7 / 10) ^ 100 + (2 ^ 106)⁻¹) ≤ (2 ^ 20)⁻¹ := by norm_num
  linarith

/-- The production game of `fpn2-folded-air-v1`: reduced profiles are not allowed. -/
def Rel4Game.prodAirV1 {k Q : ℕ} {W : Type} (oracles : Sample k Q → Rel4Oracles)
    (pub : Sample k Q → PublicInput) (proof : Sample k Q → ByteArray) (circuitRoot : Bytes)
    (cs : CompiledCS W) (Commits : Rel4Claim → W → Prop) : Rel4Game k Q W where
  prof := foldedAirV1
  lay := ⟨14, 72⟩
  hlay := rel4Layout_foldedAirV1.1
  allowReduced := false
  circuitRoot := circuitRoot
  oracles := oracles
  pub := pub
  proof := proof
  cs := cs
  Commits := Commits

/-- The production game of `fpn2-folded-air-n20-v1`. -/
def Rel4Game.prodAirN20 {k Q : ℕ} {W : Type} (oracles : Sample k Q → Rel4Oracles)
    (pub : Sample k Q → PublicInput) (proof : Sample k Q → ByteArray) (circuitRoot : Bytes)
    (cs : CompiledCS W) (Commits : Rel4Claim → W → Prop) : Rel4Game k Q W where
  prof := foldedAirN20V1
  lay := ⟨20, 72⟩
  hlay := rel4Layout_foldedAirN20.1
  allowReduced := false
  circuitRoot := circuitRoot
  oracles := oracles
  pub := pub
  proof := proof
  cs := cs
  Commits := Commits

open Classical in
/-- The bad-accept event of the `fpn2-folded-air-v1` game, with the verifier call written out. -/
theorem badAccept_prodAirV1 {k Q : ℕ} {W : Type} (oracles : Sample k Q → Rel4Oracles)
    (pub : Sample k Q → PublicInput) (proof : Sample k Q → ByteArray) (circuitRoot : Bytes)
    (cs : CompiledCS W) (Commits : Rel4Claim → W → Prop) :
    (Rel4Game.prodAirV1 oracles pub proof circuitRoot cs Commits).badAccept
      = univ.filter fun a => ∃ claim,
          verifyBytesRel4 (oracles a) (pub a) foldedAirV1 false (some circuitRoot) (proof a)
            = .ok claim ∧ ¬ ∃ w, Commits claim w ∧ cs.Satisfies w := rfl

open Classical in
/-- The bad-accept event of the `fpn2-folded-air-n20-v1` game, with the verifier call written out. -/
theorem badAccept_prodAirN20 {k Q : ℕ} {W : Type} (oracles : Sample k Q → Rel4Oracles)
    (pub : Sample k Q → PublicInput) (proof : Sample k Q → ByteArray) (circuitRoot : Bytes)
    (cs : CompiledCS W) (Commits : Rel4Claim → W → Prop) :
    (Rel4Game.prodAirN20 oracles pub proof circuitRoot cs Commits).badAccept
      = univ.filter fun a => ∃ claim,
          verifyBytesRel4 (oracles a) (pub a) foldedAirN20V1 false (some circuitRoot) (proof a)
            = .ok claim ∧ ¬ ∃ w, Commits claim w ∧ cs.Satisfies w := rfl

/-! ### Relation 5 -/

/-- **A relation-5 soundness game**, as `Rel4Game` with the weight pin in place of the circuit
pin. -/
structure Rel5Game (k Q : ℕ) (W : Type) where
  prof : Fpn2Profile
  lay : Rel5Layout
  hlay : rel5Layout prof = some lay
  allowReduced : Bool
  weightRoot : Bytes
  oracles : Sample k Q → Rel5Oracles
  pub : Sample k Q → PublicInput
  proof : Sample k Q → ByteArray
  cs : CompiledCS W
  Commits : Rel5Claim → W → Prop

namespace Rel5Game

variable {k Q : ℕ} {W : Type}

/-- The executable verifier accepts the adversary's bytes with claim `claim`. -/
def Accepts (G : Rel5Game k Q W) (a : Sample k Q) (claim : Rel5Claim) : Prop :=
  verifyBytesRel5 (G.oracles a) (G.pub a) G.prof G.allowReduced (some G.weightRoot) (G.proof a)
    = .ok claim

/-- The statement of an accepted claim (for relation 5: `Y = W X` on the committed tiles, given
here as an abstract `CompiledCS`; Lean has no definition of that relation). -/
def Statement (G : Rel5Game k Q W) (claim : Rel5Claim) : Prop :=
  ∃ w, G.Commits claim w ∧ G.cs.Satisfies w

open Classical in
/-- **The bad-accept event of `verifyBytesRel5`.** -/
noncomputable def badAccept (G : Rel5Game k Q W) : Finset (Sample k Q) :=
  univ.filter fun a => ∃ claim, G.Accepts a claim ∧ ¬ G.Statement claim

theorem mem_badAccept (G : Rel5Game k Q W) (a : Sample k Q) :
    a ∈ G.badAccept ↔ ∃ claim, G.Accepts a claim ∧ ¬ G.Statement claim := by
  simp [badAccept]

/-- Acceptance in the game gives the field equations on the layout of the game. -/
theorem accept_field (G : Rel5Game k Q W) {a : Sample k Q} {claim : Rel5Claim}
    (h : G.Accepts a claim) :
    ∃ (c : Rel5Container) (sf : RoundState) (pf : PcsState),
      parseContainerRel5 G.prof (G.proof a) = .ok c ∧
      Rel5FieldEquations (G.oracles a) (G.pub a) G.lay c sf pf := by
  obtain ⟨c, lay, sf, pf, hparse, _, _, hlay, _, hF, _⟩ := rel5_accept_e2e h
  obtain rfl : lay = G.lay := Option.some.inj (hlay.symm.trans G.hlay)
  exact ⟨c, sf, pf, hparse, hF⟩

/-- Outside the bad event, an accepted claim has a witness. -/
theorem statement_of_good (G : Rel5Game k Q W) {a : Sample k Q} (hgood : a ∉ G.badAccept)
    {claim : Rel5Claim} (hacc : G.Accepts a claim) : G.Statement claim := by
  by_contra hns
  exact hgood ((G.mem_badAccept a).mpr ⟨claim, hacc, hns⟩)

end Rel5Game

/-- **The assumptions the relation-5 lifetime theorems rest on.** Field for field the record
`CryptoAssumptions`, with `Rel5Round` (sumcheck of degree `≤ 3`, a nonzero polynomial of degree
`≤ 2`, sampling) and `Rel5FieldEquations`. -/
structure CryptoAssumptions5 {k Q : ℕ} {W : Type} (G : Rel5Game k Q W) (num den : ℕ) (Δ : ℝ) where
  st : Fin Q → Sample k Q → Rel5Round k num den
  ro_lazy : ∀ j a x, st j (Function.update a j x) = st j a
  Fail : Finset (Sample k Q)
  fail_bound : (Fail.card : ℝ) ≤ Δ * (Fintype.card (Rel4Ans k) : ℝ) ^ Q
  extract : Sample k Q → W
  extract_binds : ∀ a, a ∉ Fail → ∀ claim, G.Accepts a claim → G.Commits claim (extract a)
  R : Sample k Q → ℕ
  claimTrue : ∀ a : Sample k Q, Fin (R a + 1) → Prop
  qIdx : ∀ a : Sample k Q, Fin (R a) → Fin Q
  ir_correspondence : ∀ a, a ∉ Fail → claimTrue a 0 → G.cs.Satisfies (extract a)
  pcs_proximity_final : ∀ a, a ∉ Fail → ∀ c sf pf,
    parseContainerRel5 G.prof (G.proof a) = .ok c →
    Rel5FieldEquations (G.oracles a) (G.pub a) G.lay c sf pf →
    claimTrue a (Fin.last (R a))
  rbr_round_analysis : ∀ a, a ∉ Fail → ∀ c sf pf,
    parseContainerRel5 G.prof (G.proof a) = .ok c →
    Rel5FieldEquations (G.oracles a) (G.pub a) G.lay c sf pf →
    ∀ r : Fin (R a), ¬ claimTrue a r.castSucc → claimTrue a r.succ →
      a (qIdx a r) ∈ (st (qIdx a r) a).esc

/-- Consistency of the relation-5 record: inhabited for every game at `Δ = 1`. -/
noncomputable def CryptoAssumptions5.ofFailUniv {k Q : ℕ} {W : Type} (G : Rel5Game k Q W)
    (num den : ℕ) (w0 : W) : CryptoAssumptions5 G num den 1 where
  st := fun _ _ => .poly (Polynomial.X ^ 2) (pow_ne_zero 2 Polynomial.X_ne_zero) (by simp)
  ro_lazy := fun _ _ _ => rfl
  Fail := univ
  fail_bound := by simp
  extract := fun _ => w0
  extract_binds := fun a ha => absurd (mem_univ a) ha
  R := fun _ => 0
  claimTrue := fun _ _ => True
  qIdx := fun _ i => i.elim0
  ir_correspondence := fun a ha => absurd (mem_univ a) ha
  pcs_proximity_final := fun a ha => absurd (mem_univ a) ha
  rbr_round_analysis := fun a ha => absurd (mem_univ a) ha

/-- **`ProtocolRBRSpec` for the bad-accept event of `verifyBytesRel5`.** -/
noncomputable def rel5Spec {k Q : ℕ} {W : Type} (G : Rel5Game k Q W) {num den : ℕ} {Δ : ℝ}
    (A : CryptoAssumptions5 G num den Δ) :
    ProtocolRBRSpec A.st (fun _ => True) Rel5Round.esc G.badAccept A.Fail where
  R := A.R
  roundHolds := A.claimTrue
  qIdx := A.qIdx
  hInit := fun a ha hf h0 => by
    obtain ⟨claim, hacc, hns⟩ := (G.mem_badAccept a).mp ha
    exact hns ⟨A.extract a, A.extract_binds a hf claim hacc, A.ir_correspondence a hf h0⟩
  hFinal := fun a ha hf => by
    obtain ⟨claim, hacc, _⟩ := (G.mem_badAccept a).mp ha
    obtain ⟨c, sf, pf, hparse, hF⟩ := G.accept_field hacc
    exact A.pcs_proximity_final a hf c sf pf hparse hF
  hStep := fun a ha hf r h0 h1 => by
    obtain ⟨claim, hacc, _⟩ := (G.mem_badAccept a).mp ha
    obtain ⟨c, sf, pf, hparse, hF⟩ := G.accept_field hacc
    exact ⟨trivial, A.rbr_round_analysis a hf c sf pf hparse hF r h0 h1⟩

/-- **General lifetime bound for `verifyBytesRel5`** under `CryptoAssumptions5`. -/
theorem rel5_e2e_lifetime_general {k Q : ℕ} {W : Type} (G : Rel5Game k Q W) {num den : ℕ} {Δ : ℝ}
    (A : CryptoAssumptions5 G num den Δ) (hden : 0 < den) :
    ((G.badAccept).card : ℝ)
      ≤ (Δ + Q * (rel4Miss k num den + (2 ^ 106)⁻¹)) * (Fintype.card (Rel4Ans k) : ℝ) ^ Q := by
  have hm0 := rel4Miss_nonneg k num den
  exact rbr_lifetime_soundness A.st A.ro_lazy (fun _ => True) Rel5Round.esc Rel5Round.isSpot
    (rel4Eps (rel4Miss k num den)) (rel4Miss k num den + (2 ^ 106)⁻¹)
    (add_nonneg hm0 (by positivity))
    (fun b => by cases b <;> simp [rel4Eps, hm0])
    (fun s _ => rel5Round_esc_card_le k num den hden s)
    G.badAccept A.Fail Δ A.fail_bound (hpath_of_protocol_spec (rel5Spec G A))

/-- **Lifetime soundness of `verifyBytesRel5` (`k = 105`, `σ = 1/2`, `Q ≤ 2^64`, `Δ ≤ 2^-42`):**
`Pr[badAccept] ≤ 2^-40`. -/
theorem rel5_e2e_lifetime_k105 {Q : ℕ} {W : Type} (G : Rel5Game 105 Q W) {Δ : ℝ}
    (A : CryptoAssumptions5 G 1 2 Δ) (hQ : Q ≤ 2 ^ 64) (hΔ : Δ ≤ (2 ^ 42)⁻¹) :
    ((G.badAccept).card : ℝ) ≤ (2 ^ 40)⁻¹ * (Fintype.card (Rel4Ans 105) : ℝ) ^ Q :=
  fpn_protocol_lifetime_soundness hQ A.st A.ro_lazy (fun _ => True) Rel5Round.esc Rel5Round.isSpot
    (rel4Eps (rel4Miss 105 1 2)) (rel4Miss 105 1 2) (2 ^ 106)⁻¹ (2 ^ 106)⁻¹ Δ
    (rel4Miss_nonneg _ _ _) (rel4Miss_half 105).le
    (by positivity) le_rfl le_rfl hΔ
    (fun b => by cases b <;> simp [rel4Eps])
    (fun s _ => rel5Round_esc_card_le 105 1 2 (by norm_num) s)
    G.badAccept A.Fail A.fail_bound (rel5Spec G A)

/-- **External-seed lifetime soundness of `verifyBytesRel5` at `σ = 1/2` (`k = 51`, `Q ≤ 2^30`,
`Δ ≤ 2^-22`):** `Pr[badAccept] ≤ 2^-20`. -/
theorem rel5_e2e_lifetime_k51 {Q : ℕ} {W : Type} (G : Rel5Game 51 Q W) {Δ : ℝ}
    (A : CryptoAssumptions5 G 1 2 Δ) (hQ : Q ≤ 2 ^ 30) (hΔ : Δ ≤ (2 ^ 22)⁻¹) :
    ((G.badAccept).card : ℝ) ≤ (2 ^ 20)⁻¹ * (Fintype.card (Rel4Ans 51) : ℝ) ^ Q := by
  refine (rel5_e2e_lifetime_general G A (by norm_num)).trans
    (mul_le_mul_of_nonneg_right ?_ (by positivity))
  have hQr : (Q : ℝ) ≤ 2 ^ 30 := by exact_mod_cast hQ
  rw [rel4Miss_half]
  have h1 : (Q : ℝ) * ((1 / 2) ^ 51 + (2 ^ 106)⁻¹) ≤ 2 ^ 30 * ((1 / 2) ^ 51 + (2 ^ 106)⁻¹) :=
    mul_le_mul_of_nonneg_right hQr (by positivity)
  have h2 : (2 ^ 22 : ℝ)⁻¹ + 2 ^ 30 * ((1 / 2) ^ 51 + (2 ^ 106)⁻¹) ≤ (2 ^ 20)⁻¹ := by norm_num
  linarith

/-- **External-seed lifetime soundness of `verifyBytesRel5` at `σ = 3/10` (`k = 100`, `Q ≤ 2^30`,
`Δ ≤ 2^-22`):** `Pr[badAccept] ≤ 2^-20`. -/
theorem rel5_e2e_lifetime_k100 {Q : ℕ} {W : Type} (G : Rel5Game 100 Q W) {Δ : ℝ}
    (A : CryptoAssumptions5 G 3 10 Δ) (hQ : Q ≤ 2 ^ 30) (hΔ : Δ ≤ (2 ^ 22)⁻¹) :
    ((G.badAccept).card : ℝ) ≤ (2 ^ 20)⁻¹ * (Fintype.card (Rel4Ans 100) : ℝ) ^ Q := by
  refine (rel5_e2e_lifetime_general G A (by norm_num)).trans
    (mul_le_mul_of_nonneg_right ?_ (by positivity))
  have hQr : (Q : ℝ) ≤ 2 ^ 30 := by exact_mod_cast hQ
  rw [rel4Miss_three_tenths]
  have h1 : (Q : ℝ) * ((7 / 10) ^ 100 + (2 ^ 106)⁻¹) ≤ 2 ^ 30 * ((7 / 10) ^ 100 + (2 ^ 106)⁻¹) :=
    mul_le_mul_of_nonneg_right hQr (by positivity)
  have h2 : (2 ^ 22 : ℝ)⁻¹ + 2 ^ 30 * ((7 / 10) ^ 100 + (2 ^ 106)⁻¹) ≤ (2 ^ 20)⁻¹ := by norm_num
  linarith

/-- The production game of `fpn2-folded-vec-n16-v1`. -/
def Rel5Game.prodVecN16 {k Q : ℕ} {W : Type} (oracles : Sample k Q → Rel5Oracles)
    (pub : Sample k Q → PublicInput) (proof : Sample k Q → ByteArray) (weightRoot : Bytes)
    (cs : CompiledCS W) (Commits : Rel5Claim → W → Prop) : Rel5Game k Q W where
  prof := foldedVecN16V1
  lay := ⟨16, 72, 10⟩
  hlay := rel5Layout_foldedVecN16.1
  allowReduced := false
  weightRoot := weightRoot
  oracles := oracles
  pub := pub
  proof := proof
  cs := cs
  Commits := Commits

/-- The production game of `fpn2-folded-vec-n22-v1`. -/
def Rel5Game.prodVecN22 {k Q : ℕ} {W : Type} (oracles : Sample k Q → Rel5Oracles)
    (pub : Sample k Q → PublicInput) (proof : Sample k Q → ByteArray) (weightRoot : Bytes)
    (cs : CompiledCS W) (Commits : Rel5Claim → W → Prop) : Rel5Game k Q W where
  prof := foldedVecN22V1
  lay := ⟨22, 72, 10⟩
  hlay := rel5Layout_foldedVecN22.1
  allowReduced := false
  weightRoot := weightRoot
  oracles := oracles
  pub := pub
  proof := proof
  cs := cs
  Commits := Commits

open Classical in
/-- The bad-accept event of the `fpn2-folded-vec-n16-v1` game, with the verifier call written out. -/
theorem badAccept_prodVecN16 {k Q : ℕ} {W : Type} (oracles : Sample k Q → Rel5Oracles)
    (pub : Sample k Q → PublicInput) (proof : Sample k Q → ByteArray) (weightRoot : Bytes)
    (cs : CompiledCS W) (Commits : Rel5Claim → W → Prop) :
    (Rel5Game.prodVecN16 oracles pub proof weightRoot cs Commits).badAccept
      = univ.filter fun a => ∃ claim,
          verifyBytesRel5 (oracles a) (pub a) foldedVecN16V1 false (some weightRoot) (proof a)
            = .ok claim ∧ ¬ ∃ w, Commits claim w ∧ cs.Satisfies w := rfl

open Classical in
/-- The bad-accept event of the `fpn2-folded-vec-n22-v1` game, with the verifier call written out. -/
theorem badAccept_prodVecN22 {k Q : ℕ} {W : Type} (oracles : Sample k Q → Rel5Oracles)
    (pub : Sample k Q → PublicInput) (proof : Sample k Q → ByteArray) (weightRoot : Bytes)
    (cs : CompiledCS W) (Commits : Rel5Claim → W → Prop) :
    (Rel5Game.prodVecN22 oracles pub proof weightRoot cs Commits).badAccept
      = univ.filter fun a => ∃ claim,
          verifyBytesRel5 (oracles a) (pub a) foldedVecN22V1 false (some weightRoot) (proof a)
            = .ok claim ∧ ¬ ∃ w, Commits claim w ∧ cs.Satisfies w := rfl

/-! ## 3. A `ProtocolRBRSpec` with proved obligations: the sumcheck of `SumcheckSound` -/

section SumcheckRBR

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F] {n : ℕ}

/-- The verifier's running claim: `c_0 = S`, `c_{i+1} = g_i(r_i)`. -/
noncomputable def runClaim (S : F) (g : Fin n → (Fin n → F) → Polynomial F) (r : Fin n → F) :
    ℕ → F
  | 0 => S
  | i + 1 => if h : i < n then (g ⟨i, h⟩ r).eval (r ⟨i, h⟩) else runClaim S g r i

theorem runClaim_succ (S : F) (g : Fin n → (Fin n → F) → Polynomial F) (r : Fin n → F)
    (i : Fin n) : runClaim S g r ((i : ℕ) + 1) = (g i r).eval (r i) := by
  simp [runClaim]

/-- On an accepting run the round checks are about `runClaim`, and the final check is
`runClaim n = P r`. -/
theorem accepts_runClaim {P : (Fin n → F) → F} {S : F}
    {g : Fin n → (Fin n → F) → Polynomial F} {r : Fin n → F} (h : Accepts P S g r) :
    (∀ i : Fin n, (g i r).eval 0 + (g i r).eval 1 = runClaim S g r i) ∧
      runClaim S g r n = P r := by
  obtain ⟨c, h0, hstep, hend⟩ := h
  have key : ∀ m : ℕ, m ≤ n → c m = runClaim S g r m := by
    intro m
    induction m with
    | zero => intro _; rw [h0]; rfl
    | succ m ih =>
      intro hm
      have hlt : m < n := hm
      rw [(hstep ⟨m, hlt⟩).2]
      exact (runClaim_succ S g r ⟨m, hlt⟩).symm
  exact ⟨fun i => by rw [← key i i.isLt.le]; exact (hstep i).1, by rw [← key n le_rfl]; exact hend⟩

/-- **The round-by-round step of the sumcheck, proved.** If the running claim is false before round
`i` and true after it, the challenge `r_i` is in the bad set of round `i`. -/
theorem sumcheck_step_escapes {P : (Fin n → F) → F} {S : F}
    {g : Fin n → (Fin n → F) → Polynomial F} {r : Fin n → F} (hacc : Accepts P S g r) (i : Fin n)
    (h0 : ¬ runClaim S g r i = psum P i r)
    (h1 : runClaim S g r ((i : ℕ) + 1) = psum P ((i : ℕ) + 1) r) : r i ∈ badSet P g i r := by
  classical
  obtain ⟨hchk, _⟩ := accepts_runClaim hacc
  have hall : ¬ ∀ x, (g i r).eval x = psum P (i + 1) (Function.update r i x) := by
    intro hall
    apply h0
    rw [← hchk i, hall 0, hall 1]
    exact (psum_split P i r).symm
  unfold badSet
  rw [if_neg hall]
  simp only [mem_filter, mem_univ, true_and]
  rw [Function.update_eq_self, ← h1, runClaim_succ]

open Classical in
/-- The accepting challenge vectors of a sumcheck with claimed sum `S`. -/
noncomputable def sumcheckAcc (P : (Fin n → F) → F) (S : F)
    (g : Fin n → (Fin n → F) → Polynomial F) : Finset (Fin n → F) :=
  univ.filter fun r => Accepts P S g r

theorem mem_sumcheckAcc (P : (Fin n → F) → F) (S : F)
    (g : Fin n → (Fin n → F) → Polynomial F) (r : Fin n → F) :
    r ∈ sumcheckAcc P S g ↔ Accepts P S g r := by
  simp [sumcheckAcc]

/-- **`ProtocolRBRSpec` for the sumcheck, with every obligation proved.** The state at query `j` is
the bad set of round `j`, a round is doomed when its bad set has at most `d` elements, and the
state function is `runClaim i = psum P i r`: false at round 0 because `S ≠ cubeSum P`, true at the
end because the verifier checks `runClaim n = P r`, and a false-to-true step puts `r_i` in the bad
set (`sumcheck_step_escapes`). No assumption record. -/
noncomputable def sumcheckRbrSpec (P : (Fin n → F) → F) (d : ℕ) (hP : RoundDeg d P) (S : F)
    (hS : S ≠ cubeSum P) (g : Fin n → (Fin n → F) → Polynomial F)
    (hg : ∀ i r, (g i r).natDegree ≤ d) :
    ProtocolRBRSpec (Ω := F) (S := Finset F) (Q := n) (fun j r => badSet P g j r)
      (fun s => s.card ≤ d) (fun s => s) (sumcheckAcc P S g) ∅ where
  R := fun _ => n
  roundHolds := fun r i => runClaim S g r (i : ℕ) = psum P (i : ℕ) r
  qIdx := fun _ i => i
  hInit := fun r _ _ h => hS (h.trans (psum_zero P r))
  hFinal := fun r hr _ => by
    have hacc : Accepts P S g r := (mem_sumcheckAcc P S g r).mp hr
    exact (accepts_runClaim hacc).2.trans (psum_last P r).symm
  hStep := fun r hr _ i h0 h1 => by
    have hacc : Accepts P S g r := (mem_sumcheckAcc P S g r).mp hr
    exact ⟨badSet_card_le P d hP g hg i r, sumcheck_step_escapes hacc i h0 h1⟩

/-- **Sumcheck soundness through `ProtocolRBRSpec` and `RoundByRound`:** a false claimed sum is
accepted on at most `n · d / |F|` of the challenge vectors. Same bound as
`SumcheckSound.sumcheck_sound`, obtained from `hpath_of_protocol_spec` and
`rbr_lifetime_soundness` with no failure event. -/
theorem sumcheck_rbr_lifetime (P : (Fin n → F) → F) (d : ℕ) (hP : RoundDeg d P) (S : F)
    (hS : S ≠ cubeSum P) (g : Fin n → (Fin n → F) → Polynomial F)
    (hind : ∀ i r x, g i (Function.update r i x) = g i r) (hg : ∀ i r, (g i r).natDegree ≤ d) :
    ((sumcheckAcc P S g).card : ℝ)
      ≤ (n * ((d : ℝ) / Fintype.card F)) * (Fintype.card F : ℝ) ^ n := by
  have hpos : (0 : ℝ) < Fintype.card F := Nat.cast_pos.mpr Fintype.card_pos
  have h := rbr_lifetime_soundness (Ω := F) (S := Finset F) (ι := Unit) (Q := n)
    (fun j r => badSet P g j r) (fun j a x => badSet_ind P g hind j a x)
    (fun s => s.card ≤ d) (fun s => s) (fun _ => ()) (fun _ => (d : ℝ) / Fintype.card F)
    ((d : ℝ) / Fintype.card F) (by positivity) (fun _ => le_rfl)
    (fun s hs => by
      rw [div_mul_cancel₀ _ (ne_of_gt hpos)]
      exact_mod_cast hs)
    (sumcheckAcc P S g) ∅ 0 (by simp)
    (hpath_of_protocol_spec (sumcheckRbrSpec P d hP S hS g hg))
  rw [zero_add] at h
  exact h

end SumcheckRBR

/-! ## 4. From a satisfying Plonkish witness to the integer relations of `Fpn2Gadgets` -/

section Gadgets

open Fpn2Gadgets

/-- One Plonkish row `q_M a b + q_L a + q_R b + q_O c + q_C = 0` on three wires. -/
structure AirRow (ι : Type) where
  qM : ℤ
  qL : ℤ
  qR : ℤ
  qO : ℤ
  qC : ℤ
  a : ι
  b : ι
  c : ι

/-- The row as an equation of `F_p` on a field assignment. -/
def AirRow.holds {ι : Type} (r : AirRow ι) (w : ι → ZMod goldilocks) : Prop :=
  (r.qM : ZMod goldilocks) * w r.a * w r.b + r.qL * w r.a + r.qR * w r.b + r.qO * w r.c + r.qC = 0

/-- The integer value of the row on an integer assignment. -/
def AirRow.intVal {ι : Type} (r : AirRow ι) (z : ι → ℤ) : ℤ :=
  r.qM * z r.a * z r.b + r.qL * z r.a + r.qR * z r.b + r.qO * z r.c + r.qC

/-- A Plonkish system: gate rows and the cells looked up in the limb table `[0, 128)`. -/
structure AirSystem (ι : Type) where
  rows : List (AirRow ι)
  limbCells : List ι

/-- Satisfaction: every row holds in `F_p` and every limb cell is in `[0, 128)`. -/
def AirSystem.Sat {ι : Type} (S : AirSystem ι) (w : ι → ZMod goldilocks) : Prop :=
  (∀ r ∈ S.rows, r.holds w) ∧ ∀ i ∈ S.limbCells, ∃ n : ℕ, n < 128 ∧ w i = (n : ZMod goldilocks)

/-- **`ProtocolBridge.CompiledCS` instantiated** at a Plonkish system over `F_p`. -/
def AirSystem.compiled {ι : Type} (S : AirSystem ι) : CompiledCS (ι → ZMod goldilocks) :=
  ⟨S.Sat⟩

/-- The lift bounds of `Program.check_bounds`: every row value is below `2^62` in absolute value. -/
def AirSystem.LiftBounds {ι : Type} (S : AirSystem ι) (z : ι → ℤ) : Prop :=
  ∀ r ∈ S.rows, |r.intVal z| < 2 ^ 62

/-- **Rows lift to `ℤ`.** If the image of an integer assignment satisfies the system and the lift
bounds hold, every row is an integer equation (`signed_lift_mul_add_sound`). -/
theorem air_rows_lift_sound {ι : Type} (S : AirSystem ι) (z : ι → ℤ)
    (hsat : S.Sat fun i => (z i : ZMod goldilocks)) (hb : S.LiftBounds z) :
    ∀ r ∈ S.rows, r.intVal z = 0 := by
  intro r hr
  have h1 : |r.qM * z r.a * z r.b + r.qL * z r.a + r.qR * z r.b + r.qO * z r.c + r.qC| < 2 ^ 62 :=
    hb r hr
  have h2 : (r.qM : ZMod goldilocks) * (z r.a : ZMod goldilocks) * (z r.b : ZMod goldilocks)
      + r.qL * (z r.a : ZMod goldilocks) + r.qR * (z r.b : ZMod goldilocks)
      + r.qO * (z r.c : ZMod goldilocks) + r.qC = 0 := hsat.1 r hr
  exact signed_lift_mul_add_sound h1 h2

/-- **A range-checked wire has a unique signed integer value.** For a wire `x` of `B = 7k + t ≤ 62`
signed bits whose `k + 1` limbs and tightening cell are limb cells of a satisfied system, the
`#recompose` and `#tighten` equations give exactly one integer of `[-2^(B-1), 2^(B-1))` with image
`w x` (`limb7_field_range_sound`). The two equations are hypotheses: `#recompose` is a chain of
rows (`air_row_chain_sound`), not one row of `S`. -/
theorem range_checked_wire_lift {ι : Type} (S : AirSystem ι) (w : ι → ZMod goldilocks)
    (hsat : S.Sat w) {k t : ℕ} (ht1 : 1 ≤ t) (ht7 : t ≤ 7) (hB : 7 * k + t ≤ 62) (x : ι)
    (limb : Fin (k + 1) → ι) (tight : ι) (hlimb : ∀ i, limb i ∈ S.limbCells)
    (htight : tight ∈ S.limbCells)
    (hrec : w x + 2 ^ (7 * k + t - 1) - ∑ i, w (limb i) * 128 ^ (i : ℕ) = 0)
    (htg : w tight - w (limb (Fin.last k)) - (128 - 2 ^ t) = 0) :
    ∃! v : ℤ, (-2 ^ (7 * k + t - 1) ≤ v ∧ v < 2 ^ (7 * k + t - 1)) ∧
      (v : ZMod goldilocks) = w x := by
  choose l hl hle using fun i => hsat.2 (limb i) (hlimb i)
  obtain ⟨s, hs, hse⟩ := hsat.2 tight htight
  refine limb7_field_range_sound ht1 ht7 hB (w x) l s hl hs ?_ ?_
  · simp only [hle] at hrec
    exact hrec
  · rw [← hse, ← hle]
    exact htg

/-- **From acceptance to integer rows.** In a relation-4 game whose compiled constraint system is
the Plonkish system `S`, on every sample outside the bad-accept event an accepted claim has a
committed witness that satisfies `S`, and every integer assignment with that image that meets the
lift bounds satisfies every row over `ℤ`. The bad-accept event is the one bounded in §2. -/
theorem rel4_good_accept_integer_rows {k Q : ℕ} {ι : Type} (S : AirSystem ι)
    (G : Rel4Game k Q (ι → ZMod goldilocks)) (hcs : G.cs = S.compiled) {a : Sample k Q}
    (hgood : a ∉ G.badAccept) {claim : Rel4Claim} (hacc : G.Accepts a claim) :
    ∃ w, G.Commits claim w ∧ S.Sat w ∧
      ∀ z : ι → ℤ, (∀ i, (z i : ZMod goldilocks) = w i) → S.LiftBounds z →
        ∀ r ∈ S.rows, r.intVal z = 0 := by
  obtain ⟨w, hc, hs⟩ := G.statement_of_good hgood hacc
  rw [hcs] at hs
  have hs' : S.Sat w := hs
  refine ⟨w, hc, hs', fun z hz hb => air_rows_lift_sound S z ?_ hb⟩
  have e : (fun i => (z i : ZMod goldilocks)) = w := funext hz
  rw [e]
  exact hs'

/-- A gate equation between two integer expressions: it holds in `F_p` on the images, and the
difference meets the lift bound. -/
def LiftEq (a b : ℤ) : Prop :=
  |a - b| < 2 ^ 62 ∧ (a : ZMod goldilocks) = (b : ZMod goldilocks)

theorem LiftEq.eq {a b : ℤ} (h : LiftEq a b) : a = b :=
  signed_lift_eq_of_sub_bound h.1 h.2

/-- A `nonneg` gate: the expression equals, as a lifted gate, a cell `v` that the limb plan put in
a non-negative range (`nonneg_gate_sound`, `limb7_field_range_sound`). -/
def LiftNonneg (a : ℤ) : Prop := ∃ v : ℤ, 0 ≤ v ∧ LiftEq a v

theorem LiftNonneg.nonneg {a : ℤ} (h : LiftNonneg a) : 0 ≤ a := by
  obtain ⟨v, hv, he⟩ := h
  rw [he.eq]
  exact hv

/-- The `moe_router_expert` gates of `Fpn2Gadgets.MoeGates` as field equations on the images of an
integer witness, each with its lift bound. -/
structure MoeFieldGates (E k : ℕ) (m : ℤ) (score active argmax on best : Fin E → ℤ)
    (h : Fin k → Fin E → ℤ) (index : Fin k → ℤ) (maximum : ℤ) : Prop where
  active_gate : ∀ i, LiftEq (on i) (m * active i)
  argmax_gate : ∀ i, LiftEq (best i) (m * argmax i)
  argmax_boolean : ∀ i, LiftEq (best i * (argmax i - 1)) 0
  argmax_one : LiftEq (∑ i, best i) m
  argmax_attains : ∀ i, LiftEq (best i * (score i - maximum)) 0
  maximum_bound : ∀ i, LiftNonneg (on i * (maximum - score i))
  index_onehot_boolean : ∀ r i, LiftEq (h r i * (h r i - 1)) 0
  index_row : ∀ r, LiftEq (∑ i, h r i) m
  index_column : ∀ i, LiftEq (∑ r, h r i) (on i)
  index_value : ∀ r, LiftEq (m * index r) (∑ i : Fin E, ((i : ℕ) : ℤ) * h r i)
  index_order : ∀ r r' : Fin k, (r' : ℕ) = (r : ℕ) + 1 →
    LiftNonneg (m * (index r' - index r - 1))

/-- The field gates give the integer gates. -/
theorem moeGates_of_field {E k : ℕ} {m : ℤ} {score active argmax on best : Fin E → ℤ}
    {h : Fin k → Fin E → ℤ} {index : Fin k → ℤ} {maximum : ℤ}
    (g : MoeFieldGates E k m score active argmax on best h index maximum) :
    MoeGates E k m score active argmax on best h index maximum where
  active_gate := fun i => (g.active_gate i).eq
  argmax_gate := fun i => (g.argmax_gate i).eq
  argmax_boolean := fun i => (g.argmax_boolean i).eq
  argmax_one := g.argmax_one.eq
  argmax_attains := fun i => (g.argmax_attains i).eq
  maximum_bound := fun i => (g.maximum_bound i).nonneg
  index_onehot_boolean := fun r i => (g.index_onehot_boolean r i).eq
  index_row := fun r => (g.index_row r).eq
  index_column := fun i => (g.index_column i).eq
  index_value := fun r => (g.index_value r).eq
  index_order := fun r r' hr => (g.index_order r r' hr).nonneg

/-- **MoE routing from field gates.** On the selected branch the bitmap selects exactly `top_k`
experts, they form a `2τ`-approximate Top-K set of the integer scores, and (with the
`argmax_selected` gate) `maximum` is attained by a selected expert and bounds every selected
score. -/
theorem moe_routing_of_field_gates {E k : ℕ} {m : ℤ} {score active argmax on best : Fin E → ℤ}
    {h : Fin k → Fin E → ℤ} {index : Fin k → ℤ} {maximum threshold tau : ℤ}
    (g : MoeFieldGates E k m score active argmax on best h index maximum) (hm : m = 1)
    (hmargin : ∀ i, LiftNonneg ((2 * on i - m) * (score i - threshold) + m * tau))
    (hsel : ∀ i, LiftEq (best i * (active i - 1)) 0) :
    (Finset.univ.filter fun i => active i = 1).card = k ∧
      IsApproxTopK score (2 * tau) k (Finset.univ.filter fun i => active i = 1) ∧
      (∃ i, active i = 1 ∧ argmax i = 1 ∧ score i = maximum) ∧
      ∀ i, active i = 1 → score i ≤ maximum := by
  have gz := moeGates_of_field g
  have hr := moe_rmx_sound gz hm fun i => (hsel i).eq
  exact ⟨(moe_bitmap_sound gz hm).2, moe_topk_margin_sound gz hm fun i => (hmargin i).nonneg,
    hr.1, hr.2⟩

/-- **Gated DeltaNet / LRec gates from field gates** (`gdn_gates_sound` after the lift). -/
theorem gdn_of_field_gates {inRange inside outside position extent state previous convolution
    decay scale : ℤ}
    (hin : LiftEq inside (1 * inRange)) (hout : LiftEq outside (1 - inside))
    (hbool : LiftEq (inside * (inRange - 1)) 0)
    (hcmp : LiftNonneg (inside * (extent - 1 - position) + outside * (position - extent)))
    (hconv : LiftEq (outside * convolution) 0) (hstate : LiftEq (outside * (state - previous)) 0)
    (hdlo : LiftNonneg (inside * decay)) (hdhi : LiftNonneg (inside * (scale - decay))) :
    (inRange = 0 ∨ inRange = 1) ∧ (inRange = 1 ↔ position < extent) ∧
      (inRange = 0 → state = previous ∧ convolution = 0) ∧
      (inRange = 1 → 0 ≤ decay ∧ decay ≤ scale) :=
  gdn_gates_sound hin.eq hout.eq hbool.eq hcmp.nonneg hconv.eq hstate.eq hdlo.nonneg hdhi.nonneg

/-- **The gadget relations of a lifted witness, in one statement.** Verifier-checked gates [G]:
the range-checked wire `x` has the signed integer value `v`, and the MoE gates select a
`2τ`-approximate Top-K set of size `k`. Reference predicate [P]: `hbf` is the `.finite` predicate
of `install_bf16`, which the relation-4 verifier does not check; it is a hypothesis of its own and
only under it is the decoded dyadic the IEEE-754 value. `hsat` is used for the range lift only: the
integers of `MoeFieldGates` and of `hbf` are free variables, not stated to be lifts of cells of `w`,
and their `LiftEq` equations are not stated to be rows of `S`. The statement tied to an accepted
proof is `rel4_good_accept_integer_rows`. -/
theorem gadget_relations_of_witness {ι : Type} (S : AirSystem ι)
    (w : ι → ZMod goldilocks) (hsat : S.compiled.Satisfies w)
    {kl t : ℕ} (ht1 : 1 ≤ t) (ht7 : t ≤ 7) (hB : 7 * kl + t ≤ 62) (x : ι)
    (limb : Fin (kl + 1) → ι) (tight : ι) (hlimb : ∀ i, limb i ∈ S.limbCells)
    (htight : tight ∈ S.limbCells)
    (hrec : w x + 2 ^ (7 * kl + t - 1) - ∑ i, w (limb i) * 128 ^ (i : ℕ) = 0)
    (htg : w tight - w (limb (Fin.last kl)) - (128 - 2 ^ t) = 0)
    {E k : ℕ} {m : ℤ} {score active argmax on best : Fin E → ℤ}
    {h : Fin k → Fin E → ℤ} {index : Fin k → ℤ} {maximum threshold tau : ℤ}
    (g : MoeFieldGates E k m score active argmax on best h index maximum) (hm : m = 1)
    (hmargin : ∀ i, LiftNonneg ((2 * on i - m) * (score i - threshold) + m * tau))
    {s e mant : ℕ} {significand power : ℤ} (hbf : Bf16Finite s e mant significand power) :
    (∃! v : ℤ, (-2 ^ (7 * kl + t - 1) ≤ v ∧ v < 2 ^ (7 * kl + t - 1)) ∧
        (v : ZMod goldilocks) = w x) ∧
      IsApproxTopK score (2 * tau) k (Finset.univ.filter fun i => active i = 1) ∧
      bf16Value s significand power = bf16Ieee s e mant :=
  ⟨range_checked_wire_lift S w hsat ht1 ht7 hB x limb tight hlimb htight hrec htg,
    moe_topk_margin_sound (moeGates_of_field g) hm fun i => (hmargin i).nonneg,
    bf16_decode_sound hbf⟩

end Gadgets

/-! ## 5. Zero knowledge of the linear view (one transcript, fixed challenges) -/

section ZK

/-- **A masked linear view is uniform on its fibers.** With `view = S w + L mask` and `L`
surjective, the number of masks producing a given view is the size of the kernel of `L`, whatever
the witness and the view. A simulator that outputs a uniform element of the view space therefore
has the distribution of the real view. -/
theorem masked_view_uniform {W G H : Type*} [AddCommGroup G] [AddCommGroup H] [Fintype G]
    [DecidableEq H] (S : W → H) (L : G →+ H) (hs : Function.Surjective L) (w : W) (v : H) :
    (Finset.univ.filter fun r => S w + L r = v).card
      = (Finset.univ.filter fun r : G => L r = 0).card := by
  have h := masked_view_independent (fun y : H => y) L hs (S w) v v
  refine h.trans ?_
  congr 1
  ext r
  simp

/-- Block-triangular composition of two masking maps with a cross term. -/
def blockHom {G1 G2 H1 H2 : Type*} [AddCommGroup G1] [AddCommGroup G2] [AddCommGroup H1]
    [AddCommGroup H2] (L1 : G1 →+ H1) (L2 : G2 →+ H2) (X : G1 →+ H2) : G1 × G2 →+ H1 × H2 :=
  (L1.comp (AddMonoidHom.fst G1 G2)).prod
    (X.comp (AddMonoidHom.fst G1 G2) + L2.comp (AddMonoidHom.snd G1 G2))

theorem blockHom_apply {G1 G2 H1 H2 : Type*} [AddCommGroup G1] [AddCommGroup G2]
    [AddCommGroup H1] [AddCommGroup H2] (L1 : G1 →+ H1) (L2 : G2 →+ H2) (X : G1 →+ H2)
    (p : G1 × G2) : blockHom L1 L2 X p = (L1 p.1, X p.1 + L2 p.2) := rfl

/-- The composition is surjective when both diagonal blocks are (`Mask.block_surjective`). -/
theorem blockHom_surjective {G1 G2 H1 H2 : Type*} [AddCommGroup G1] [AddCommGroup G2]
    [AddCommGroup H1] [AddCommGroup H2] (L1 : G1 →+ H1) (L2 : G2 →+ H2) (X : G1 →+ H2)
    (h1 : Function.Surjective L1) (h2 : Function.Surjective L2) :
    Function.Surjective (blockHom L1 L2 X) := by
  intro y
  obtain ⟨p, hp⟩ := block_surjective L1 L2 X h1 h2 y
  exact ⟨p, (blockHom_apply L1 L2 X p).trans hp⟩

/-- **The `ZK_ROWS` blinding rows as a linear map.** Each of `c` gate columns carries `z` blinding
cells in `K = F_p`; the evaluation of the column at the zerocheck point receives
`Σ_j m_{col,j} · λ_j`, where `λ_j ∈ E` is the `eq` weight of blinding row `j` at that point. -/
def zkRowsHom {K E : Type*} [Field K] [AddCommGroup E] [Module K E] (c z : ℕ) (lam : Fin z → E) :
    (Fin c → Fin z → K) →+ (Fin c → E) where
  toFun m := fun col => ∑ j, m col j • lam j
  map_zero' := by
    funext col
    simp
  map_add' x y := by
    funext col
    simp [add_smul, Finset.sum_add_distrib]

/-- The blinding rows mask every column when the weights `λ_j` span `E` over `K`. This is the
nondegeneracy of the zerocheck point (an `F_p`-basis of `E = F_p^3` when `z = 3`). It is a
hypothesis; that it fails for `z < 3` is a dimension count, not proved here. -/
theorem zkRowsHom_surjective {K E : Type*} [Field K] [AddCommGroup E] [Module K E] (c z : ℕ)
    (lam : Fin z → E)
    (hspan : Function.Surjective fun ρ : Fin z → K => ∑ j, ρ j • lam j) :
    Function.Surjective (zkRowsHom (K := K) c z lam) := by
  intro y
  choose ρ hρ using fun col => hspan (y col)
  exact ⟨ρ, funext fun col => hρ col⟩

section Joint

variable {K E : Type*} [Field K] [Fintype K] [Field E] [Fintype E] [DecidableEq E] [Algebra K E]
  {n d : ℕ}

/-- **The joint mask of the relation-4 linear view.** Masks: the Libra mask `(a_0, a)` of a masked
sumcheck, the `3 × 3` blinding cells of the `ZK_ROWS = 3` rows (in `K = F_p`) and the scalar behind
`z_blind`. View: the Libra-masked round data, the three gate-column evaluations
`(e_a, e_b, e_c)` and the blinded batched opening. `X1` and `X2` are the (arbitrary) linear
contributions of earlier masks to later view components. -/
noncomputable def rel4JointMask (ρ β : E) (lam : Fin 3 → E)
    (X1 : (E × (Fin n → Fin d → E)) →+ (Fin 3 → E))
    (X2 : ((E × (Fin n → Fin d → E)) × (Fin 3 → Fin 3 → K)) →+ E) :
    (((E × (Fin n → Fin d → E)) × (Fin 3 → Fin 3 → K)) × E) →+
      (((E × (Fin n → Fin d → E)) × (Fin 3 → E)) × E) :=
  blockHom (blockHom (libraPhi n d ρ) (zkRowsHom (K := K) 3 3 lam) X1) (AddMonoidHom.mulLeft β) X2

theorem rel4JointMask_surjective {ρ β : E} {lam : Fin 3 → E}
    (X1 : (E × (Fin n → Fin d → E)) →+ (Fin 3 → E))
    (X2 : ((E × (Fin n → Fin d → E)) × (Fin 3 → Fin 3 → K)) →+ E)
    (h2 : (2 : E) ≠ 0) (hρ : ρ ≠ 0) (hβ : β ≠ 0)
    (hspan : Function.Surjective fun r : Fin 3 → K => ∑ j, r j • lam j) :
    Function.Surjective (rel4JointMask (K := K) (n := n) (d := d) ρ β lam X1 X2) := by
  refine blockHom_surjective _ _ _
    (blockHom_surjective _ _ _ (libra_mask_surjective h2 hρ) (zkRowsHom_surjective 3 3 lam hspan))
    fun y => ⟨β⁻¹ * y, ?_⟩
  simp [hβ]

/-- **Relation-4 joint-mask composition, linear view.** For every witness-dependent shift `S`, the
number of joint masks (Libra, `ZK_ROWS = 3` blinding cells, `z_blind`) that produce a given linear
view is the same for any two witnesses. Hypotheses: `2 ≠ 0`, the Libra challenge `ρ ≠ 0`, the
coefficient `β ≠ 0` of the blinding scalar, and the blinding weights span `E` over `K`. -/
theorem rel4_joint_view_independent {W : Type*} {ρ β : E} {lam : Fin 3 → E}
    (X1 : (E × (Fin n → Fin d → E)) →+ (Fin 3 → E))
    (X2 : ((E × (Fin n → Fin d → E)) × (Fin 3 → Fin 3 → K)) →+ E)
    (h2 : (2 : E) ≠ 0) (hρ : ρ ≠ 0) (hβ : β ≠ 0)
    (hspan : Function.Surjective fun r : Fin 3 → K => ∑ j, r j • lam j)
    (S : W → ((E × (Fin n → Fin d → E)) × (Fin 3 → E)) × E) (w w' : W)
    (v : ((E × (Fin n → Fin d → E)) × (Fin 3 → E)) × E) :
    (Finset.univ.filter fun m => S w + rel4JointMask (K := K) ρ β lam X1 X2 m = v).card
      = (Finset.univ.filter fun m => S w' + rel4JointMask (K := K) ρ β lam X1 X2 m = v).card :=
  masked_view_independent S _ (rel4JointMask_surjective X1 X2 h2 hρ hβ hspan) w w' v

/-- **The simulator statement for the relation-4 linear view.** Every view has the same number of
preimages, the size of the kernel of the joint mask, for every witness: the linear view is uniform
on its space and a simulator samples it without the witness. This is all the theorem says; see the
module header, Scope (d), for what it does not cover. -/
theorem rel4_joint_view_uniform {W : Type*} {ρ β : E} {lam : Fin 3 → E}
    (X1 : (E × (Fin n → Fin d → E)) →+ (Fin 3 → E))
    (X2 : ((E × (Fin n → Fin d → E)) × (Fin 3 → Fin 3 → K)) →+ E)
    (h2 : (2 : E) ≠ 0) (hρ : ρ ≠ 0) (hβ : β ≠ 0)
    (hspan : Function.Surjective fun r : Fin 3 → K => ∑ j, r j • lam j)
    (S : W → ((E × (Fin n → Fin d → E)) × (Fin 3 → E)) × E) (w : W)
    (v : ((E × (Fin n → Fin d → E)) × (Fin 3 → E)) × E) :
    (Finset.univ.filter fun m => S w + rel4JointMask (K := K) ρ β lam X1 X2 m = v).card
      = (Finset.univ.filter fun m => rel4JointMask (K := K) (n := n) (d := d) ρ β lam X1 X2 m
          = 0).card :=
  masked_view_uniform S _ (rel4JointMask_surjective X1 X2 h2 hρ hβ hspan) w v

/-- **The joint mask of the relation-5 linear view**: the Libra mask of the inner-product sumcheck
and the scalar behind `z_blind` in `Y = target + μ e_W + μ² z_blind`. -/
noncomputable def rel5JointMask (ρ β : E) (X : (E × (Fin n → Fin d → E)) →+ E) :
    ((E × (Fin n → Fin d → E)) × E) →+ ((E × (Fin n → Fin d → E)) × E) :=
  blockHom (libraPhi n d ρ) (AddMonoidHom.mulLeft β) X

theorem rel5JointMask_surjective {ρ β : E} (X : (E × (Fin n → Fin d → E)) →+ E)
    (h2 : (2 : E) ≠ 0) (hρ : ρ ≠ 0) (hβ : β ≠ 0) :
    Function.Surjective (rel5JointMask (n := n) (d := d) ρ β X) := by
  refine blockHom_surjective _ _ _ (libra_mask_surjective h2 hρ) fun y => ⟨β⁻¹ * y, ?_⟩
  simp [hβ]

/-- **Relation-5 joint-mask composition, linear view** (`Libra + z_blind`). -/
theorem rel5_joint_view_independent {W : Type*} {ρ β : E}
    (X : (E × (Fin n → Fin d → E)) →+ E) (h2 : (2 : E) ≠ 0) (hρ : ρ ≠ 0) (hβ : β ≠ 0)
    (S : W → (E × (Fin n → Fin d → E)) × E) (w w' : W) (v : (E × (Fin n → Fin d → E)) × E) :
    (Finset.univ.filter fun m => S w + rel5JointMask ρ β X m = v).card
      = (Finset.univ.filter fun m => S w' + rel5JointMask ρ β X m = v).card :=
  masked_view_independent S _ (rel5JointMask_surjective X h2 hρ hβ) w w' v

/-- The simulator statement for the relation-5 linear view. -/
theorem rel5_joint_view_uniform {W : Type*} {ρ β : E}
    (X : (E × (Fin n → Fin d → E)) →+ E) (h2 : (2 : E) ≠ 0) (hρ : ρ ≠ 0) (hβ : β ≠ 0)
    (S : W → (E × (Fin n → Fin d → E)) × E) (w : W) (v : (E × (Fin n → Fin d → E)) × E) :
    (Finset.univ.filter fun m => S w + rel5JointMask ρ β X m = v).card
      = (Finset.univ.filter fun m => rel5JointMask (n := n) (d := d) ρ β X m = 0).card :=
  masked_view_uniform S _ (rel5JointMask_surjective X h2 hρ hβ) w v

end Joint

theorem fp3Field_two_ne_zero : (2 : Fp3Field) ≠ 0 := by
  have h : ((2 : ℕ) : Fp3Field) ≠ 0 := by
    rw [Ne, CharP.cast_eq_zero_iff Fp3Field goldilocks, goldilocks_eq]; norm_num
  exact_mod_cast h

/-- **The relation-4 linear view over the deployed fields** (`K = F_p`, `E = F_p[X]/(X^3 − 2)`):
`2 ≠ 0` is discharged, the three remaining nondegeneracy conditions stay hypotheses. -/
theorem rel4_joint_view_fp3 {W : Type*} {n d : ℕ} {ρ β : Fp3Field} {lam : Fin 3 → Fp3Field}
    (X1 : (Fp3Field × (Fin n → Fin d → Fp3Field)) →+ (Fin 3 → Fp3Field))
    (X2 : ((Fp3Field × (Fin n → Fin d → Fp3Field)) × (Fin 3 → Fin 3 → GoldilocksField))
      →+ Fp3Field)
    (hρ : ρ ≠ 0) (hβ : β ≠ 0)
    (hspan : Function.Surjective fun r : Fin 3 → GoldilocksField => ∑ j, r j • lam j)
    (S : W → ((Fp3Field × (Fin n → Fin d → Fp3Field)) × (Fin 3 → Fp3Field)) × Fp3Field)
    (w w' : W) (v : ((Fp3Field × (Fin n → Fin d → Fp3Field)) × (Fin 3 → Fp3Field)) × Fp3Field) :
    (Finset.univ.filter fun m =>
        S w + rel4JointMask (K := GoldilocksField) ρ β lam X1 X2 m = v).card
      = (Finset.univ.filter fun m =>
        S w' + rel4JointMask (K := GoldilocksField) ρ β lam X1 X2 m = v).card :=
  rel4_joint_view_independent X1 X2 fp3Field_two_ne_zero hρ hβ hspan S w w' v

/-- **Constrained-mask binding at `E`** (`ConstrainedMask.constrained_mask_goldilocks` with
`|E| = p^3`). A committed decoding-mask row that differs from the honest one passes the multilinear
check on at most `2^-186` of the points, for vocabularies up to `2^32`. This is a soundness
(binding) statement; it is not part of the view-independence argument above. -/
theorem constrained_mask_binding_fp3 {vb : ℕ} (A M : (Fin vb → Bool) → Fp3Field) (hne : A ≠ M)
    (hvb : vb ≤ 32) :
    ((univ.filter fun r : Fin vb → Fp3Field => mleEval A r = mleEval M r).card : ℝ)
      ≤ (2 ^ 186)⁻¹ * (Fintype.card (Fin vb → Fp3Field) : ℝ) :=
  (constrained_mask_goldilocks A M hne fp3Field_card hvb).1

end ZK

end Fpn2EndToEnd
end FlashProveNext

#print axioms FlashProveNext.Fpn2EndToEnd.rel4_accept_e2e
#print axioms FlashProveNext.Fpn2EndToEnd.rel4_accept_prodAirV1
#print axioms FlashProveNext.Fpn2EndToEnd.rel4_accept_prodAirN20
#print axioms FlashProveNext.Fpn2EndToEnd.rel5_accept_e2e
#print axioms FlashProveNext.Fpn2EndToEnd.rel5_accept_prodVecN16
#print axioms FlashProveNext.Fpn2EndToEnd.rel5_accept_prodVecN22
#print axioms FlashProveNext.Fpn2EndToEnd.rel4_layout_fold_step_cubic
#print axioms FlashProveNext.Fpn2EndToEnd.Rel4Game.mem_badAccept
#print axioms FlashProveNext.Fpn2EndToEnd.Rel4Game.accept_field
#print axioms FlashProveNext.Fpn2EndToEnd.Rel4Game.statement_of_good
#print axioms FlashProveNext.Fpn2EndToEnd.CryptoAssumptions.ofFailUniv
#print axioms FlashProveNext.Fpn2EndToEnd.rel4Spec
#print axioms FlashProveNext.Fpn2EndToEnd.rel4_e2e_lifetime_general
#print axioms FlashProveNext.Fpn2EndToEnd.rel4_e2e_lifetime_k105
#print axioms FlashProveNext.Fpn2EndToEnd.rel4_e2e_lifetime_k51
#print axioms FlashProveNext.Fpn2EndToEnd.rel4_e2e_lifetime_k100
#print axioms FlashProveNext.Fpn2EndToEnd.Rel4Game.prodAirV1
#print axioms FlashProveNext.Fpn2EndToEnd.Rel4Game.prodAirN20
#print axioms FlashProveNext.Fpn2EndToEnd.badAccept_prodAirV1
#print axioms FlashProveNext.Fpn2EndToEnd.badAccept_prodAirN20
#print axioms FlashProveNext.Fpn2EndToEnd.Rel5Game.mem_badAccept
#print axioms FlashProveNext.Fpn2EndToEnd.Rel5Game.accept_field
#print axioms FlashProveNext.Fpn2EndToEnd.Rel5Game.statement_of_good
#print axioms FlashProveNext.Fpn2EndToEnd.CryptoAssumptions5.ofFailUniv
#print axioms FlashProveNext.Fpn2EndToEnd.rel5Spec
#print axioms FlashProveNext.Fpn2EndToEnd.rel5_e2e_lifetime_general
#print axioms FlashProveNext.Fpn2EndToEnd.rel5_e2e_lifetime_k105
#print axioms FlashProveNext.Fpn2EndToEnd.rel5_e2e_lifetime_k51
#print axioms FlashProveNext.Fpn2EndToEnd.rel5_e2e_lifetime_k100
#print axioms FlashProveNext.Fpn2EndToEnd.Rel5Game.prodVecN16
#print axioms FlashProveNext.Fpn2EndToEnd.Rel5Game.prodVecN22
#print axioms FlashProveNext.Fpn2EndToEnd.badAccept_prodVecN16
#print axioms FlashProveNext.Fpn2EndToEnd.badAccept_prodVecN22
#print axioms FlashProveNext.Fpn2EndToEnd.runClaim_succ
#print axioms FlashProveNext.Fpn2EndToEnd.accepts_runClaim
#print axioms FlashProveNext.Fpn2EndToEnd.sumcheck_step_escapes
#print axioms FlashProveNext.Fpn2EndToEnd.mem_sumcheckAcc
#print axioms FlashProveNext.Fpn2EndToEnd.sumcheckRbrSpec
#print axioms FlashProveNext.Fpn2EndToEnd.sumcheck_rbr_lifetime
#print axioms FlashProveNext.Fpn2EndToEnd.air_rows_lift_sound
#print axioms FlashProveNext.Fpn2EndToEnd.range_checked_wire_lift
#print axioms FlashProveNext.Fpn2EndToEnd.rel4_good_accept_integer_rows
#print axioms FlashProveNext.Fpn2EndToEnd.LiftEq.eq
#print axioms FlashProveNext.Fpn2EndToEnd.LiftNonneg.nonneg
#print axioms FlashProveNext.Fpn2EndToEnd.moeGates_of_field
#print axioms FlashProveNext.Fpn2EndToEnd.moe_routing_of_field_gates
#print axioms FlashProveNext.Fpn2EndToEnd.gdn_of_field_gates
#print axioms FlashProveNext.Fpn2EndToEnd.gadget_relations_of_witness
#print axioms FlashProveNext.Fpn2EndToEnd.masked_view_uniform
#print axioms FlashProveNext.Fpn2EndToEnd.blockHom_apply
#print axioms FlashProveNext.Fpn2EndToEnd.blockHom_surjective
#print axioms FlashProveNext.Fpn2EndToEnd.zkRowsHom_surjective
#print axioms FlashProveNext.Fpn2EndToEnd.rel4JointMask_surjective
#print axioms FlashProveNext.Fpn2EndToEnd.rel4_joint_view_independent
#print axioms FlashProveNext.Fpn2EndToEnd.rel4_joint_view_uniform
#print axioms FlashProveNext.Fpn2EndToEnd.rel5JointMask_surjective
#print axioms FlashProveNext.Fpn2EndToEnd.rel5_joint_view_independent
#print axioms FlashProveNext.Fpn2EndToEnd.rel5_joint_view_uniform
#print axioms FlashProveNext.Fpn2EndToEnd.fp3Field_two_ne_zero
#print axioms FlashProveNext.Fpn2EndToEnd.rel4_joint_view_fp3
#print axioms FlashProveNext.Fpn2EndToEnd.constrained_mask_binding_fp3
