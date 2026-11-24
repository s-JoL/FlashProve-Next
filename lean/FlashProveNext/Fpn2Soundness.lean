/-
Concrete Goldilocks / `F_{p^3}` algebraic soundness bridge for relation 4 (`FOLDED_GKR_AIR`, M4
Stages 4–6, items 1–8) and relation 5 (`FOLDED_VEC_MATMUL`, M4 Stage 8, item 9); Stage 9 adds the
decoded query shape and the fold-layer bound of both relations (items 7 and 9).
This is the only module that imports both the Mathlib soundness layer and the executable bytes verifier.

1. `goldilocks_prime`, `goldilocksP_prime`, `goldilocks_two_not_cube`, `fp3Poly_irreducible`,
   `fp3Field_card` : `p = 2^64 − 2^32 + 1` is prime (Lucas test, witness 7, kernel-evaluated
   square-and-multiply `powMod`, no `native_decide`), `2` is not a cube mod `p`, `X^3 − 2` is
   irreducible, and `E = Fp3Field = F_p[X]/(X^3 − 2)` is a field with `p^3 > 2^191` elements.
2. `Fp3.toField_add / _sub / _mul / _scale / _zero / _one / _ofFp`, `Fp3.toField_injective`,
   `fpInv_toField`, `lagrangeEval_toField` : the executable `Fp3` arithmetic of `VerifyBytes` is the
   arithmetic of `E` (unconditionally), the map is injective on canonical limbs, `fpInv` is field
   inversion and `lagrangeEval` is evaluation of the Lagrange interpolant (`roundPoly`).
3. `fp3_sumcheck_sound`, `fp3_mle_zero_sound`, `fp3_logup_sound`, `fp3_linear_root_card_le`,
   `fp3_zerocheck_sound` : `SumcheckSound`, `ConstrainedMask` and `LogUp` instantiated at `E`, and
   one composed interactive statement (zerocheck point + `g`-round sumcheck, `(g + g d) / p^3`).
4. `rel4AlgNumer`, `rel4_alg_error_lt`, `rel4_alg_union_bound` : an explicit union-bound numerator
   over every relation-4 challenge and the numeric bound `numerator / p^3 ≤ 2^-160 < 2^-150` for
   every admissible layout (`12 ≤ n ≤ 28`; `n = 14` is the registered `fpn2-folded-air-v1`).
5. `Rel4Ans`, `Rel4Round`, `spot_miss_card_le`, `rel4Round_esc_card_le`, `Rel4Round.ofSpotCheck`,
   `seedShare_of_fiber_ge`, `rel4RbrSpec`, `rel4_lifetime_soundness_k105` / `_k51` / `_k100` :
   `ProtocolBridge.fpn_protocol_lifetime_soundness` with `Ω = Rel4Ans k = E × E^k` (one algebraic
   challenge and `k` sampling seeds per oracle answer) and every argument instantiated. `hEsc` is
   proved for all four kinds of round: a sampling round carries only the threshold premise "bad
   seeds have density `≥ σ = num/den`", and its escape bound `(1 − σ)^k` is derived from
   `SpotCheck.frac_bad_bound` for every `k` and `σ` (no bound is a field of the constructor).
   `ProtocolRBRSpec` is inhabited by construction, since `BadAcc` is defined as the escape event.
   `< 2^-40` for `k = 105`, `σ = 1/2`, `Q ≤ 2^64`; `≤ 2^-20` for `k = 51`, `σ = 1/2`, `Q ≤ 2^30`;
   `< 2^-20` for `k = 100`, `σ = 3/10`, `Q ≤ 2^30` (`rel4_lifetime_soundness_strict`).
6. `Rel4FieldEquations`, `rel4TypedVerify_field`, `verifyBytesRel4_accept_field`,
   `zcRound_accepts_shape`, `gkrRound_accepts_shape`, `pcsRound_accepts_shape` : an accepting
   `verifyBytesRel4` run yields the relation-4 acceptance equations as equations of `E`, and each
   sumcheck round has the shape of `SumcheckSound.Accepts` (degree bound, `p(0) + p(1) = target`,
   next target `= p(challenge)`).
7. `parseContainerRel4_root_canonical`, `Fp3.toField_ne_zero` : the parser returns canonical
   `GKR_ROOT` limbs, so `root_den ≠ 0` holds in `E` (field `root_den` of `Rel4FieldEquations`).
   `fold4Field`, `fold4_toField`, `rel3IotaInv_sq`, `fold4Field_interpolates`, `QuerySpec4Field`,
   `querySpec4_field` : the executable `fold4` is the coset-4 fold of `E` (`ι^(−2) = −1` proved
   for the verifier's constant), that fold maps the four values of a cubic on a coset `{x ι^k}`
   to `c_0 + r_a c_1 + r_b c_2 + r_a r_b c_3`, and every query of `QuerySpec4` (the combined
   layer-0 coset `v^F + θ v^C`, previous-slot consistency, the fold chain, the last fold `= FINAL`)
   holds as equations of `E` (field `query` of `Rel4FieldEquations`). Stage 9:
   `AirQueryCanonical`, `airQuery_dec_canonical`, `parseContainerRel4_queries_canonical` (a decoded
   query has canonical coset limbs and exactly `T − 1` committed layers),
   `Rel3Layout.layer_of_lt_folds`, `foldRun_field`, `foldField_layer_bound` : every fold step of
   an accepted query is at a layer `t < T`, so `2 t + 2 ≤ n`; `FoldField` carries that bound.
8. `rel3Omega_toField`, `rel3Omega_pow_half`, `rel3Omega_orderOf`, `foldDomainPoint`,
   `foldXInv_mul_domainPoint`, `foldXInv_mul_domainPoint_fp3`, `foldXInv_toField`,
   `foldXInv_toField_fp3`, `foldDomainPoint_quarter`,
   `foldDomainPoint_add_quarter`, `foldDomainPoint_succ`, `foldDomainPoint_injOn`,
   `rel3IotaInv_toField`, `fold4_foldXInv_interpolates` (Stage 6) : the verifier's `ω = 7^((p−1)/L)`
   has exact order `L = 8 · 2^n` in `F_p` (`n ≤ 28`), the executable `foldXInv lay t j` is the
   multiplicative inverse, in `F_p` and in `E`, of the layer-`t` domain point `ω_t^j`,
   `ω_t = ω^(4^t)`, the four slots of
   coset leaf `j` are the points `ω_t^j · ι^k` with `ι = ω_t^(L_t/4) = 7^((p−1)/4)`, the `L_t`
   points of a layer are distinct and their fourth powers are the points of the next layer.
   Hence `fold4Field`, fed with the verifier's own `foldXInv` and `rel3IotaInv`, folds a cubic
   given by its values on the actual Reed–Solomon evaluation domain, with no hypothesis left on
   `x^(−1)` or `ι`.
9. Relation 5 (Stages 8 and 9). `Rel5Canonical`, `parseContainerRel5_canonical` : every decoded
   limb of a parsed relation-5 container is canonical (`IP`, `PCS`, and the coset values of
   `QUERIES`), with exactly `d` inner-product rounds, `T − 1` fold roots and `Q` queries of `T − 1`
   committed layers. `IpRoundField`, `ipRound_accepts_shape`, `batchedClaim5_toField`,
   `finalWeight5_toField`, `QuerySpec5Field`, `Rel5FieldEquations`, `rel5TypedVerify_field`,
   `verifyBytesRel5_accept_field` : an accepting `verifyBytesRel5` run yields the relation-5
   acceptance equations as equations of `E` (degree-3 inner-product rounds from `λ · G_sum`,
   degree-2 Basefold rounds from `Y = target + μ e_W + μ² z_blind`, `y = FINAL · W̃`, every query
   with the weight leaf under the pinned `weight_root`). `foldField_cubic`,
   `rel5_layer0_fold_cubic` : `fold4_foldXInv_interpolates` applied to the fold equations of an
   accepted query (a committed layer, with no layer-bound hypothesis; the first fold of
   `F + θ C`). `rel5AlgNumer`,
   `rel5_alg_error_lt`, `rel5_alg_error_foldedVecN16V1` / `N22V1`, `rel5_alg_union_bound` : the
   budget `6 n − k − 12 ≤ 156` and `numerator / p^3 ≤ 2^-160 < 2^-150` for every admissible layout
   (`74` at `n = 16`, `110` at `n = 22`). `Rel5Round`, `rel5Round_esc_card_le`, `rel5RbrSpec`,
   `Rel5Round.ofSpotCheck`, `rel5_lifetime_soundness_k105` / `_k51` / `_k100` / `_strict` : the lifetime model of item 5 for
   the relation-5 round kinds (sumcheck of degree `≤ 3`, a nonzero polynomial of degree `≤ 2` in
   the challenge, sampling; no lookup round), with the same three bounds.

Scope (what is NOT proved here). There is no theorem "`verifyBytesRel4` accepts a false statement
with probability `< ε`", and none for `verifyBytesRel5`. Everything below applies to relation 5 as
well: (a) and (b) verbatim for `Rel5Round` / `rel5BadAcc`; (c) with `rel5Terms` / `weightsEval` in
place of the Plonkish statement (no Lean definition of `Y = W X` on tiles, no proof that the
batched claim `Y` is the stated combination of multilinear evaluations of `F` and `C`, no Libra
mask composition; `rel5AlgNumer` is a budget); (d) and (e) unchanged. The sibling counts of the
openings are not stated, and no converse (field equations imply typed acceptance) is proved. Lean has no statement / witness semantics for relation 4, and the missing
links are exactly:
(a) the Fiat–Shamir hash is an abstract function; modelling it as a lazily sampled random oracle
    whose answers are uniform in `Rel4Ans k` (one challenge and `k` independent sampling seeds, all
    uniform in `E`) is an assumption of item 5 (`squeeze_fp3` reduces 128-bit words mod `p`, whose
    statistical bias is not modelled, and the deployed sampler derives each draw from a 192-bit
    BLAKE2b word `U` through `⌊U C / 2^192⌋`, a seed space of size `2^192`, not `p^3`);
(b) extraction: no proof that an accepting transcript of a false statement makes some oracle answer
    fall in the escape set of a `Rel4Round`; in item 5 the prover's state function `st` is arbitrary,
    not derived from `verifyBytesRel4`, and `BadAcc` is that model's event. In particular the
    premise of a sampling round (bad seeds have density `≥ σ`) is the definition of the threshold
    class, not a consequence of the statement being false;
(c) the semantic identification of the verified equations with the Plonkish statement
    (`rel4Terms` / `weightsEval` as multilinear extensions, the gate table, the wiring / range
    lookup) and the composition of all phases with the Libra masks; item 4 is a budget;
(d) Basefold / FRI proximity (Reed–Solomon proximity gaps), query soundness and Merkle binding
    (collision resistance): not in Lean. Item 7 transports the query equations and proves that
    `fold4` folds a cubic correctly; it does not prove that passing queries implies proximity of
    the committed words, that the opened values are the committed ones, or that the squeezed
    positions are uniform. Item 8 identifies `foldXInv` with the inverse of the domain point and
    the leaf slots with the coset `ω_t^j ι^k`; it does not show that the prover's encoder
    (`core/src/folded_air.rs`) stores the codeword in this order, which stays with the golden
    vectors;
(e) seed map: `Rel4Round.ofSpotCheck` takes the block pool through an arbitrary map `u : E → B`
    and needs the seed share of `Bad`; `seedShare_of_fiber_ge` reduces it to the block share for a
    map whose fibers have size `≥ c`. No concrete seed map is constructed or proved near-balanced,
    and the with-replacement, per-draw-uniform sampling model is not linked to
    `protocol.py::sample_request` (stratified, cost-weighted, lineage-padded).
-/
import Mathlib
import FlashProveNext.ProtocolBridge
import FlashProveNext.ConstrainedMask
import FlashProveNext.LogUp
import FlashProveNext.Fpn2Rel4
import FlashProveNext.Fpn2Rel5

namespace FlashProveNext

open Polynomial

set_option linter.unusedSectionVars false
set_option exponentiation.threshold 4096

/-! ## 1. The Goldilocks prime -/

theorem goldilocksP_toNat_eq_goldilocks : goldilocksP.toNat = goldilocks := by
  rw [goldilocksP_toNat_eq, goldilocks_eq]

theorem goldilocks_sub_one_factor :
    goldilocks - 1 = 2 ^ 32 * 3 * 5 * 17 * 257 * 65537 := by
  rw [goldilocks_eq]; norm_num

/-- Modular exponentiation by squaring on `ℕ` (structural in the fuel), evaluated by the kernel. -/
def powModAux (m : ℕ) : ℕ → ℕ → ℕ → ℕ → ℕ
  | 0, _, _, acc => acc
  | fuel + 1, b, e, acc =>
    powModAux m fuel (b * b % m) (e / 2) (if e % 2 = 1 then acc * b % m else acc)

theorem powModAux_spec (m : ℕ) :
    ∀ (fuel b e acc : ℕ), e < 2 ^ fuel → powModAux m fuel b e acc % m = acc * b ^ e % m
  | 0, b, e, acc, he => by
    have : e = 0 := by simpa using he
    subst this
    simp [powModAux]
  | fuel + 1, b, e, acc, he => by
    have he2 : e / 2 < 2 ^ fuel := by
      rw [Nat.div_lt_iff_lt_mul (by norm_num)]; rw [pow_succ] at he; exact he
    rw [powModAux, powModAux_spec m fuel _ _ _ he2]
    have hdecomp : b ^ e = (b * b) ^ (e / 2) * b ^ (e % 2) := by
      rw [← pow_two, ← pow_mul, ← pow_add, Nat.div_add_mod]
    rcases Nat.mod_two_eq_zero_or_one e with h0 | h1
    · have hne : ¬ e % 2 = 1 := by omega
      rw [if_neg hne, hdecomp, h0, pow_zero, mul_one]
      exact (Nat.ModEq.refl acc).mul ((Nat.mod_modEq (b * b) m).pow (e / 2))
    · rw [if_pos h1, hdecomp, h1, pow_one]
      have h := (Nat.mod_modEq (acc * b) m).mul ((Nat.mod_modEq (b * b) m).pow (e / 2))
      have e' : acc * b * (b * b) ^ (e / 2) = acc * ((b * b) ^ (e / 2) * b) := by ring
      rw [e'] at h
      exact h

/-- `b ^ e mod m` for `e < 2^64`. -/
def powMod (b e m : ℕ) : ℕ := powModAux m 64 (b % m) e 1 % m

theorem powMod_eq (b e m : ℕ) (he : e < 2 ^ 64) : powMod b e m = b ^ e % m := by
  rw [powMod, powModAux_spec m 64 _ _ _ he, one_mul, ← Nat.pow_mod]

theorem zmod_pow_eq_iff (p a e r : ℕ) (he : e < 2 ^ 64) :
    (a : ZMod p) ^ e = (r : ZMod p) ↔ powMod a e p = r % p := by
  rw [powMod_eq a e p he, ← ZMod.natCast_eq_natCast_iff', Nat.cast_pow]

theorem seven_pow_goldilocks :
    powMod 7 (goldilocks - 1) goldilocks = 1 ∧
    powMod 7 ((goldilocks - 1) / 2) goldilocks ≠ 1 ∧
    powMod 7 ((goldilocks - 1) / 3) goldilocks ≠ 1 ∧
    powMod 7 ((goldilocks - 1) / 5) goldilocks ≠ 1 ∧
    powMod 7 ((goldilocks - 1) / 17) goldilocks ≠ 1 ∧
    powMod 7 ((goldilocks - 1) / 257) goldilocks ≠ 1 ∧
    powMod 7 ((goldilocks - 1) / 65537) goldilocks ≠ 1 := by
  decide +kernel

theorem two_pow_goldilocks_third : powMod 2 ((goldilocks - 1) / 3) goldilocks ≠ 1 := by
  decide +kernel

theorem prime_dvd_goldilocks_sub_one {q : ℕ} (hq : q.Prime) (h : q ∣ goldilocks - 1) :
    q = 2 ∨ q = 3 ∨ q = 5 ∨ q = 17 ∨ q = 257 ∨ q = 65537 := by
  rw [goldilocks_sub_one_factor] at h
  simp only [Nat.Prime.dvd_mul hq] at h
  rcases h with ((((h | h) | h) | h) | h) | h
  · exact .inl ((Nat.prime_dvd_prime_iff_eq hq Nat.prime_two).mp (hq.dvd_of_dvd_pow h))
  · exact .inr (.inl ((Nat.prime_dvd_prime_iff_eq hq (by norm_num)).mp h))
  · exact .inr (.inr (.inl ((Nat.prime_dvd_prime_iff_eq hq (by norm_num)).mp h)))
  · exact .inr (.inr (.inr (.inl ((Nat.prime_dvd_prime_iff_eq hq (by norm_num)).mp h))))
  · exact .inr (.inr (.inr (.inr (.inl ((Nat.prime_dvd_prime_iff_eq hq (by norm_num)).mp h)))))
  · exact .inr (.inr (.inr (.inr (.inr ((Nat.prime_dvd_prime_iff_eq hq (by norm_num)).mp h)))))

theorem goldilocks_lt : goldilocks < 2 ^ 64 := by rw [goldilocks_eq]; norm_num

/-- **The Goldilocks modulus `p = 2^64 − 2^32 + 1` is prime** (Lucas test with witness 7). -/
theorem goldilocks_prime : Nat.Prime goldilocks := by
  obtain ⟨h1, h2, h3, h5, h17, h257, h65537⟩ := seven_pow_goldilocks
  have hlt : ∀ q, (goldilocks - 1) / q < 2 ^ 64 := fun q =>
    lt_of_le_of_lt (Nat.div_le_self _ _) (lt_of_le_of_lt (Nat.sub_le _ _) goldilocks_lt)
  have hne : ∀ q, powMod 7 ((goldilocks - 1) / q) goldilocks ≠ 1 →
      ((7 : ℕ) : ZMod goldilocks) ^ ((goldilocks - 1) / q) ≠ 1 := by
    intro q hq h
    apply hq
    have h' : ((7 : ℕ) : ZMod goldilocks) ^ ((goldilocks - 1) / q) = ((1 : ℕ) : ZMod goldilocks) := by
      rw [h, Nat.cast_one]
    rw [(zmod_pow_eq_iff _ _ _ _ (hlt q)).mp h']
    rw [goldilocks_eq]
  refine lucas_primality goldilocks ((7 : ℕ) : ZMod goldilocks) ?_ ?_
  · have := (zmod_pow_eq_iff goldilocks 7 (goldilocks - 1) 1
      (lt_of_le_of_lt (Nat.sub_le _ _) goldilocks_lt)).mpr (by rw [h1, goldilocks_eq])
    rw [this, Nat.cast_one]
  · intro q hq hdvd
    rcases prime_dvd_goldilocks_sub_one hq hdvd with rfl | rfl | rfl | rfl | rfl | rfl
    · exact hne 2 h2
    · exact hne 3 h3
    · exact hne 5 h5
    · exact hne 17 h17
    · exact hne 257 h257
    · exact hne 65537 h65537

theorem goldilocksP_prime : Nat.Prime goldilocksP.toNat := by
  rw [goldilocksP_toNat_eq_goldilocks]; exact goldilocks_prime

instance goldilocks_fact : Fact (Nat.Prime goldilocks) := ⟨goldilocks_prime⟩

/-! ## 2. The cubic extension `F_p[X]/(X^3 − 2)` -/

/-- The base field `F_p`. -/
abbrev GoldilocksField : Type := ZMod goldilocks

/-- `2` is not a cube in `F_p`: `2^((p−1)/3) ≠ 1`. -/
theorem goldilocks_two_not_cube (x : GoldilocksField) : x ^ 3 ≠ 2 := by
  intro h
  have h2 : (2 : GoldilocksField) ≠ 0 := by
    have : ((2 : ℕ) : ZMod goldilocks) ≠ 0 := by
      rw [Ne, ZMod.natCast_eq_zero_iff, goldilocks_eq]; norm_num
    simpa using this
  have hx : x ≠ 0 := by
    rintro rfl
    apply h2; rw [← h]; simp
  have h1 : x ^ (goldilocks - 1) = 1 := ZMod.pow_card_sub_one_eq_one hx
  have hdiv : 3 * ((goldilocks - 1) / 3) = goldilocks - 1 :=
    Nat.mul_div_cancel' (by rw [goldilocks_sub_one_factor]; norm_num)
  have hpow : ((2 : ℕ) : ZMod goldilocks) ^ ((goldilocks - 1) / 3) = ((1 : ℕ) : ZMod goldilocks) := by
    rw [Nat.cast_ofNat, Nat.cast_one, ← h, ← pow_mul, hdiv, h1]
  have := (zmod_pow_eq_iff _ _ _ _ (lt_of_le_of_lt (Nat.div_le_self _ _)
    (lt_of_le_of_lt (Nat.sub_le _ _) goldilocks_lt))).mp hpow
  apply two_pow_goldilocks_third
  rw [this, goldilocks_eq]

/-- The defining polynomial `X^3 − 2`. -/
noncomputable def fp3Poly : GoldilocksField[X] := X ^ 3 - C 2

theorem fp3Poly_monic : fp3Poly.Monic := monic_X_pow_sub_C 2 (by norm_num)

theorem fp3Poly_natDegree : fp3Poly.natDegree = 3 := natDegree_X_pow_sub_C

theorem fp3Poly_ne_zero : fp3Poly ≠ 0 := fp3Poly_monic.ne_zero

/-- **`X^3 − 2` is irreducible over `F_p`** (a cubic without roots). -/
theorem fp3Poly_irreducible : Irreducible fp3Poly := by
  rw [fp3Poly_monic.irreducible_iff_roots_eq_zero_of_degree_le_three
    (by rw [fp3Poly_natDegree]; norm_num) (by rw [fp3Poly_natDegree])]
  refine Multiset.eq_zero_of_forall_notMem fun x hx => ?_
  have := (mem_roots fp3Poly_ne_zero).mp hx
  rw [IsRoot, fp3Poly, eval_sub, eval_pow, eval_X, eval_C, sub_eq_zero] at this
  exact goldilocks_two_not_cube x this

instance fp3Poly_fact : Fact (Irreducible fp3Poly) := ⟨fp3Poly_irreducible⟩

/-- The challenge field `E = F_p[X]/(X^3 − 2)`. -/
abbrev Fp3Field : Type := AdjoinRoot fp3Poly

noncomputable instance : Field Fp3Field := AdjoinRoot.instField

noncomputable instance : DecidableEq Fp3Field := Classical.decEq _

instance : Module.Finite GoldilocksField Fp3Field := (AdjoinRoot.powerBasis fp3Poly_ne_zero).finite

instance : Finite Fp3Field := Module.finite_of_finite GoldilocksField

noncomputable instance : Fintype Fp3Field := Fintype.ofFinite _

/-- **`|E| = p^3`.** -/
theorem fp3Field_card : Fintype.card Fp3Field = goldilocks ^ 3 := by
  rw [← Nat.card_eq_fintype_card, Module.natCard_eq_pow_finrank (K := GoldilocksField),
    (AdjoinRoot.powerBasis fp3Poly_ne_zero).finrank, AdjoinRoot.powerBasis_dim, fp3Poly_natDegree,
    Nat.card_eq_fintype_card, ZMod.card]

theorem fp3Field_card_gt : 2 ^ 191 < Fintype.card Fp3Field := by
  rw [fp3Field_card]; exact goldilocks_p3_bounds.1


/-! ## 3. The executable `Fp3` arithmetic is the arithmetic of `E` -/

/-- A 64-bit limb read as an element of `F_p`. -/
def fpToField (x : UInt64) : GoldilocksField := ((x.toNat : ℕ) : ZMod goldilocks)

theorem fpOfNat_toNat (n : ℕ) : (fpOfNat n).toNat = n % goldilocks := by
  have hP' : goldilocksP.toNat < 2 ^ 64 := goldilocksP.toNat_lt
  have hmod := Nat.mod_lt n (show 0 < goldilocksP.toNat by decide)
  rw [fpOfNat, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega), goldilocksP_toNat_eq_goldilocks]

theorem fpToField_ofNat (n : ℕ) : fpToField (fpOfNat n) = (n : GoldilocksField) := by
  unfold fpToField
  rw [fpOfNat_toNat, ZMod.natCast_mod]

theorem fpToField_zero : fpToField 0 = 0 := by simp [fpToField]

theorem fpToField_one : fpToField 1 = 1 := by simp [fpToField]

theorem fpToField_two : fpToField 2 = 2 := by
  have h : (2 : UInt64).toNat = 2 := by decide
  simp [fpToField, h]

theorem fpToField_add (a b : UInt64) : fpToField (fpAdd a b) = fpToField a + fpToField b := by
  rw [fpAdd, fpToField_ofNat, Nat.cast_add]; rfl

theorem fpToField_mul (a b : UInt64) : fpToField (fpMul a b) = fpToField a * fpToField b := by
  rw [fpMul, fpToField_ofNat, Nat.cast_mul]; rfl

theorem fpToField_sub (a b : UInt64) : fpToField (fpSub a b) = fpToField a - fpToField b := by
  have hlt : b.toNat % goldilocks < goldilocks := Nat.mod_lt _ goldilocks_prime.pos
  rw [fpSub, fpToField_ofNat, goldilocksP_toNat_eq_goldilocks, Nat.cast_sub (by omega),
    Nat.cast_add, ZMod.natCast_self, add_zero, ZMod.natCast_mod]; rfl

/-- Canonical limbs embed injectively. -/
theorem fpToField_injective {x y : UInt64} (hx : x < goldilocksP) (hy : y < goldilocksP)
    (h : fpToField x = fpToField y) : x = y := by
  unfold fpToField at h
  rw [ZMod.natCast_eq_natCast_iff'] at h
  have hx' := UInt64.lt_iff_toNat_lt.mp hx
  have hy' := UInt64.lt_iff_toNat_lt.mp hy
  rw [goldilocksP_toNat_eq_goldilocks] at hx' hy'
  rw [Nat.mod_eq_of_lt hx', Nat.mod_eq_of_lt hy'] at h
  exact UInt64.toNat_inj.mp h

/-- The adjoined root `u`, `u^3 = 2`. -/
noncomputable def fp3U : Fp3Field := AdjoinRoot.root fp3Poly

/-- The embedding `F_p → E`. -/
noncomputable def fpEmbed : GoldilocksField →+* Fp3Field := AdjoinRoot.of fp3Poly

theorem fp3U_cube : fp3U ^ 3 = 2 := by
  have h : AdjoinRoot.mk fp3Poly (X ^ 3 - C 2) = 0 := AdjoinRoot.mk_self
  rw [map_sub, map_pow, AdjoinRoot.mk_X, AdjoinRoot.mk_C, sub_eq_zero] at h
  rw [fp3U, h, map_ofNat]

/-- `1, u, u^2` are linearly independent over `F_p`. -/
theorem fp3_coords_eq_zero (c0 c1 c2 : GoldilocksField)
    (h : fpEmbed c0 + fpEmbed c1 * fp3U + fpEmbed c2 * fp3U ^ 2 = 0) :
    c0 = 0 ∧ c1 = 0 ∧ c2 = 0 := by
  have hq : AdjoinRoot.mk fp3Poly (C c2 * X ^ 2 + C c1 * X + C c0) = 0 := by
    rw [map_add, map_add, map_mul, map_mul, map_pow, AdjoinRoot.mk_X, AdjoinRoot.mk_C,
      AdjoinRoot.mk_C, AdjoinRoot.mk_C, ← h]
    simp only [fpEmbed, fp3U]; ring
  have hdvd := AdjoinRoot.mk_eq_zero.mp hq
  have hz := eq_zero_of_dvd_of_natDegree_lt hdvd
    (lt_of_le_of_lt natDegree_quadratic_le (by rw [fp3Poly_natDegree]; norm_num))
  have e0 := congrArg (fun p => p.coeff 0) hz
  have e1 := congrArg (fun p => p.coeff 1) hz
  have e2 := congrArg (fun p => p.coeff 2) hz
  simp [coeff_X, coeff_C, coeff_X_pow] at e0 e1 e2
  exact ⟨e0, e1, e2⟩

namespace Fp3

/-- **The bridge map** `a0 + a1 u + a2 u^2 ↦ E`. -/
noncomputable def toField (a : Fp3) : Fp3Field :=
  fpEmbed (fpToField a.a0) + fpEmbed (fpToField a.a1) * fp3U + fpEmbed (fpToField a.a2) * fp3U ^ 2

/-- All three limbs are `< P` (the wire format's canonical form). -/
def Canonical (a : Fp3) : Prop := a.a0 < goldilocksP ∧ a.a1 < goldilocksP ∧ a.a2 < goldilocksP

theorem toField_zero : Fp3.zero.toField = 0 := by
  simp [toField, Fp3.zero, fpToField_zero]

theorem toField_one : Fp3.one.toField = 1 := by
  simp [toField, Fp3.one, fpToField_zero, fpToField_one]

theorem toField_ofFp (x : UInt64) : (Fp3.ofFp x).toField = fpEmbed (fpToField x) := by
  simp [toField, Fp3.ofFp, fpToField_zero]

theorem toField_add (a b : Fp3) : (a.add b).toField = a.toField + b.toField := by
  simp only [toField, Fp3.add, fpToField_add, map_add]; ring

theorem toField_sub (a b : Fp3) : (a.sub b).toField = a.toField - b.toField := by
  simp only [toField, Fp3.sub, fpToField_sub, map_sub]; ring

theorem toField_scale (a : Fp3) (c : UInt64) :
    (a.scale c).toField = a.toField * fpEmbed (fpToField c) := by
  simp only [toField, Fp3.scale, fpToField_mul, map_mul]; ring

/-- The schoolbook product with `u^3 = 2` is the product of `E`. -/
theorem toField_mul (a b : Fp3) : (a.mul b).toField = a.toField * b.toField := by
  simp only [toField, Fp3.mul, fpToField_add, fpToField_mul, fpToField_two, map_add, map_mul,
    map_ofNat]
  linear_combination
    (-(fpEmbed (fpToField a.a1) * fpEmbed (fpToField b.a2)
        + fpEmbed (fpToField a.a2) * fpEmbed (fpToField b.a1))
      - fpEmbed (fpToField a.a2) * fpEmbed (fpToField b.a2) * fp3U) * fp3U_cube

theorem zero_canonical : Fp3.zero.Canonical := ⟨by decide, by decide, by decide⟩

theorem one_canonical : Fp3.one.Canonical := ⟨by decide, by decide, by decide⟩

theorem add_canonical (a b : Fp3) : (a.add b).Canonical :=
  ⟨fpOfNat_lt _, fpOfNat_lt _, fpOfNat_lt _⟩

theorem sub_canonical (a b : Fp3) : (a.sub b).Canonical :=
  ⟨fpOfNat_lt _, fpOfNat_lt _, fpOfNat_lt _⟩

theorem mul_canonical (a b : Fp3) : (a.mul b).Canonical :=
  ⟨fpOfNat_lt _, fpOfNat_lt _, fpOfNat_lt _⟩

/-- **Faithfulness.** On canonical elements the bridge map is injective, so an equation between
canonical `Fp3` values holds iff it holds in `E`. -/
theorem toField_injective {a b : Fp3} (ha : a.Canonical) (hb : b.Canonical)
    (h : a.toField = b.toField) : a = b := by
  have h0 : fpEmbed (fpToField a.a0 - fpToField b.a0)
      + fpEmbed (fpToField a.a1 - fpToField b.a1) * fp3U
      + fpEmbed (fpToField a.a2 - fpToField b.a2) * fp3U ^ 2 = 0 := by
    simp only [map_sub]; unfold toField at h; linear_combination h
  obtain ⟨e0, e1, e2⟩ := fp3_coords_eq_zero _ _ _ h0
  have k0 := fpToField_injective ha.1 hb.1 (sub_eq_zero.mp e0)
  have k1 := fpToField_injective ha.2.1 hb.2.1 (sub_eq_zero.mp e1)
  have k2 := fpToField_injective ha.2.2 hb.2.2 (sub_eq_zero.mp e2)
  cases a; cases b
  simp only [Fp3.mk.injEq]
  exact ⟨k0, k1, k2⟩

theorem toField_eq_zero_iff {a : Fp3} (ha : a.Canonical) : a.toField = 0 ↔ a = Fp3.zero :=
  ⟨fun h => toField_injective ha zero_canonical (h.trans toField_zero.symm),
    fun h => h ▸ toField_zero⟩

end Fp3

instance : CharP Fp3Field goldilocks :=
  charP_of_injective_algebraMap (algebraMap GoldilocksField Fp3Field).injective goldilocks

/-! ### `fpInv` and `lagrangeEval` are inversion and polynomial evaluation in `E` -/

theorem fpPowAux_toField : ∀ (fuel : ℕ) (b : UInt64) (e : ℕ) (acc : UInt64), e < 2 ^ fuel →
    fpToField (fpPowAux fuel b e acc) = fpToField acc * fpToField b ^ e
  | 0, b, e, acc, he => by
    have : e = 0 := by simpa using he
    subst this
    simp [fpPowAux]
  | fuel + 1, b, e, acc, he => by
    have he2 : e / 2 < 2 ^ fuel := by
      rw [Nat.div_lt_iff_lt_mul (by norm_num)]; rw [pow_succ] at he; exact he
    rw [fpPowAux]
    split
    · rename_i h0
      subst h0
      simp
    · rw [fpPowAux_toField fuel _ _ _ he2, fpToField_mul]
      have hdecomp : fpToField b ^ e
          = (fpToField b * fpToField b) ^ (e / 2) * fpToField b ^ (e % 2) := by
        rw [← pow_two, ← pow_mul, ← pow_add, Nat.div_add_mod]
      rcases Nat.mod_two_eq_zero_or_one e with h0 | h1
      · rw [if_neg (by omega), hdecomp, h0, pow_zero, mul_one]
      · rw [if_pos h1, fpToField_mul, hdecomp, h1, pow_one]; ring

theorem fpPow_toField (b : UInt64) (e : ℕ) (he : e < 2 ^ 64) :
    fpToField (fpPow b e) = fpToField b ^ e := by
  rw [fpPow, fpPowAux_toField 64 b e 1 he, fpToField_one, one_mul]

/-- **`fpInv` is field inversion** (Fermat, `a^(p−2)`), with `0 ↦ 0` for the rejected input. -/
theorem fpInv_toField (a : UInt64) : fpToField ((fpInv a).getD 0) = (fpToField a)⁻¹ := by
  unfold fpInv
  split
  · rename_i h
    rw [goldilocksP_toNat_eq_goldilocks] at h
    have h0 : fpToField a = 0 := by
      unfold fpToField
      rw [ZMod.natCast_eq_zero_iff]
      exact Nat.dvd_of_mod_eq_zero h
    simp [h0, fpToField_zero]
  · rename_i h
    rw [goldilocksP_toNat_eq_goldilocks] at h
    have hne : fpToField a ≠ 0 := by
      unfold fpToField
      rw [Ne, ZMod.natCast_eq_zero_iff]
      exact fun hd => h (Nat.mod_eq_zero_of_dvd hd)
    have hlt := goldilocks_lt
    have h2le := goldilocks_prime.two_le
    rw [Option.getD_some, fpPow_toField _ _ (by rw [goldilocksP_toNat_eq_goldilocks]; omega),
      goldilocksP_toNat_eq_goldilocks]
    have h1 : fpToField a ^ (goldilocks - 1) = 1 := ZMod.pow_card_sub_one_eq_one hne
    have h2 : fpToField a ^ (goldilocks - 2) * fpToField a = 1 := by
      rw [← pow_succ, show goldilocks - 2 + 1 = goldilocks - 1 by omega, h1]
    exact eq_inv_of_mul_eq_one_left h2

theorem lagrangeBasis_toField (d : ℕ) (r : Fp3) (i : ℕ) :
    (lagrangeBasis d r i).1.toField
        = ∏ j ∈ (Finset.range d).erase i, (r.toField - (j : Fp3Field)) ∧
      fpToField (lagrangeBasis d r i).2
        = ∏ j ∈ (Finset.range d).erase i, ((i : GoldilocksField) - (j : GoldilocksField)) := by
  unfold lagrangeBasis
  induction d with
  | zero => simp [Fp3.toField_one, fpToField_one]
  | succ d ih =>
    simp only [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    by_cases hdi : d = i
    · rw [if_pos hdi]
      have he : (Finset.range (d + 1)).erase i = (Finset.range d).erase i := by
        subst hdi
        rw [Finset.range_add_one, Finset.erase_insert (by simp),
          Finset.erase_eq_of_notMem (by simp)]
      rw [he]
      exact ih
    · rw [if_neg hdi]
      have he : (Finset.range (d + 1)).erase i = insert d ((Finset.range d).erase i) := by
        rw [Finset.range_add_one, Finset.erase_insert_of_ne hdi]
      rw [he, Finset.prod_insert (by simp), Finset.prod_insert (by simp)]
      refine ⟨?_, ?_⟩
      · rw [Fp3.toField_mul, Fp3.toField_sub, Fp3.toField_ofFp, fpToField_ofNat, map_natCast,
          ih.1]
        ring
      · rw [fpToField_mul, fpToField_sub, fpToField_ofNat, fpToField_ofNat, ih.2]
        ring

/-- The polynomial of degree `< len` over `E` through the wire values `g(0), g(1), …`. -/
noncomputable def roundPoly (vals : List Fp3) : Polynomial Fp3Field :=
  Lagrange.interpolate (Finset.range vals.length) (fun j : ℕ => (j : Fp3Field))
    (fun i => (vals.getD i Fp3.zero).toField)

/-- **`lagrangeEval` is evaluation of the interpolating polynomial in `E`.** -/
theorem lagrangeEval_toField (vals : List Fp3) (r : Fp3) :
    (lagrangeEval vals r).toField = (roundPoly vals).eval r.toField := by
  have hsum : ∀ (n : ℕ) (F : ℕ → Fp3),
      ((List.range n).foldl (fun out i => out.add (F i)) Fp3.zero).toField
        = ∑ i ∈ Finset.range n, (F i).toField := by
    intro n F
    induction n with
    | zero => simp [Fp3.toField_zero]
    | succ n ih =>
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Fp3.toField_add,
        ih, Finset.sum_range_succ]
  unfold lagrangeEval roundPoly
  refine (hsum vals.length (fun i => ((vals.getD i Fp3.zero).mul
    (lagrangeBasis vals.length r i).1).scale ((fpInv (lagrangeBasis vals.length r i).2).getD 0))).trans ?_
  rw [Lagrange.interpolate_apply, eval_finset_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [Fp3.toField_scale, Fp3.toField_mul, fpInv_toField, (lagrangeBasis_toField _ r i).1,
    (lagrangeBasis_toField _ r i).2, mul_assoc, eval_mul, eval_C, Lagrange.basis, eval_prod]
  congr 1
  rw [map_inv₀, map_prod, ← Finset.prod_inv_distrib, ← Finset.prod_mul_distrib]
  refine Finset.prod_congr rfl fun j _ => ?_
  simp [Lagrange.basisDivisor, mul_comm]

theorem roundNodes_injOn (d : ℕ) (hd : d ≤ goldilocks) :
    Set.InjOn (fun j : ℕ => (j : Fp3Field)) (Finset.range d : Set ℕ) := by
  intro a ha b hb hab
  have ha' : a < goldilocks := lt_of_lt_of_le (Finset.mem_range.mp ha) hd
  have hb' : b < goldilocks := lt_of_lt_of_le (Finset.mem_range.mp hb) hd
  have h := (CharP.natCast_eq_natCast Fp3Field goldilocks).mp hab
  exact Nat.ModEq.eq_of_lt_of_lt h ha' hb'

theorem roundPoly_natDegree_le (vals : List Fp3) (d : ℕ) (hlen : vals.length = d + 1)
    (hd : d + 1 ≤ goldilocks) : (roundPoly vals).natDegree ≤ d := by
  by_cases hz : roundPoly vals = 0
  · simp [hz]
  · have h : (roundPoly vals).degree < ((Finset.range vals.length).card : WithBot ℕ) :=
      Lagrange.degree_interpolate_lt (fun i => (vals.getD i Fp3.zero).toField)
        (roundNodes_injOn vals.length (hlen ▸ hd))
    rw [Finset.card_range, hlen] at h
    exact Nat.lt_succ_iff.mp ((natDegree_lt_iff_degree_lt hz).mpr h)

theorem roundPoly_eval_node (vals : List Fp3) (hd : vals.length ≤ goldilocks) (i : ℕ)
    (hi : i < vals.length) :
    (roundPoly vals).eval (i : Fp3Field) = (vals.getD i Fp3.zero).toField :=
  Lagrange.eval_interpolate_at_node (fun i => (vals.getD i Fp3.zero).toField)
    (roundNodes_injOn vals.length hd) (Finset.mem_range.mpr hi)

/-! ## 4. Soundness lemmas instantiated at `E`, `|E| = p^3` -/

section Instantiation

open Finset

theorem fp3Field_card_pos : (0 : ℝ) < ((goldilocks ^ 3 : ℕ) : ℝ) := by
  rw [← fp3Field_card]; exact Nat.cast_pos.mpr Fintype.card_pos

/-- A set of at most `2^85` challenges has density at most `2^-106` in `E`. -/
theorem fp3_small_card_le (c : ℕ) (hc : c ≤ 2 ^ 85) :
    (c : ℝ) ≤ (2 ^ 106)⁻¹ * (Fintype.card Fp3Field : ℝ) := by
  have h := field_p3_bits c hc
  rw [fp3Field_card]
  exact (div_le_iff₀ fp3Field_card_pos).mp h

open Classical in
/-- **Sumcheck over `E`.** `n` rounds of degree `≤ d`: a false claim survives with probability at
most `n d / p^3` (`sumcheck_sound_real` at `E`). Relation 4 uses `d = 4` with `n = g` (zerocheck),
`d = 3` with `n = k` (GKR layer `k`), `d = 2` with `n = n_vars + 2` (Basefold). -/
theorem fp3_sumcheck_sound {n : ℕ} (P : (Fin n → Fp3Field) → Fp3Field) (d : ℕ) (hP : RoundDeg d P)
    (S : Fp3Field) (hS : S ≠ cubeSum P) (g : Fin n → (Fin n → Fp3Field) → Polynomial Fp3Field)
    (hind : ∀ i r x, g i (Function.update r i x) = g i r) (hg : ∀ i r, (g i r).natDegree ≤ d) :
    ((univ.filter fun r : Fin n → Fp3Field => Accepts P S g r).card : ℝ)
      ≤ ((n : ℝ) * d / (goldilocks ^ 3 : ℕ)) * (Fintype.card (Fin n → Fp3Field) : ℝ) := by
  have h := sumcheck_sound_real P d hP S hS g hind hg
  rw [fp3Field_card] at h
  exact h

/-- **Zerocheck point over `E`.** A nonzero table on `{0,1}^g` has a multilinear extension that
vanishes at a uniform `z ∈ E^g` with probability at most `g / p^3`. -/
theorem fp3_mle_zero_sound {g : ℕ} (G : (Fin g → Bool) → Fp3Field) (hG : G ≠ 0) :
    ((univ.filter fun z : Fin g → Fp3Field => mleEval G z = 0).card : ℝ)
      ≤ ((g : ℝ) / (goldilocks ^ 3 : ℕ)) * (Fintype.card (Fin g → Fp3Field) : ℝ) := by
  have h := constrained_mask_bad_prob G 0 hG
  rw [fp3Field_card] at h
  have e : ∀ z : Fin g → Fp3Field, mleEval (0 : (Fin g → Bool) → Fp3Field) z = 0 :=
    fun z => by simp [mleEval]
  simpa only [e] using h

open Classical in
/-- **FracLogUp over `E`.** An invalid lookup with `nv < p` queries passes (or hits a pole) on at
most `2 (nv + nt)` challenges out of `p^3`. -/
theorem fp3_logup_sound {nv nt : ℕ} (a : Fin nv → Fp3Field) (t : Fin nt → Fp3Field)
    (m : Fin nt → Fp3Field) (ht : Function.Injective t) (hnv : nv < goldilocks)
    (hinv : ¬ LookupValid a t m) :
    ((univ.filter fun l : Fp3Field => ¬ NonPole a t l ∨ Harmonic a t m l).card : ℝ)
      ≤ (((2 * (nv + nt) : ℕ) : ℝ) / (goldilocks ^ 3 : ℕ)) * (Fintype.card Fp3Field : ℝ) := by
  have h := pole_or_harmonic_card_le a t m ht (natCast_ne_zero_of_lt goldilocks nv hnv) hinv
  rw [fp3Field_card, div_mul_cancel₀ _ fp3Field_card_pos.ne']
  exact_mod_cast h

open Classical in
/-- **Wiring FracLogUp over `E`** (two-column fingerprint `w + γ σ`, then `β`): an invalid tuple
lookup passes on at most `(2 nt + nv) · p^3` of the `p^6` challenge pairs `(γ, β)`
(`logup_multicol_card_le` at width 2). Together with the poles (`≤ nv + nt` values of `β` for each
`γ`) this is the `3 · 2^(S+1)` term of `rel4AlgNumer`. -/
theorem fp3_wiring_logup_card_le {nv nt : ℕ} (a : Fin nv → Fin 2 → Fp3Field)
    (t : Fin nt → Fin 2 → Fp3Field) (m : Fin nt → Fp3Field) (ht : Function.Injective t)
    (hnv : nv < goldilocks) (hinv : ¬ TupleValid a t m) :
    (univ.filter fun p : Fp3Field × Fp3Field =>
        NonPole (fpa p.1 a) (fpt p.1 t) p.2 ∧ Harmonic (fpa p.1 a) (fpt p.1 t) m p.2).card
      ≤ (2 * nt + nv) * goldilocks ^ 3 := by
  have h := logup_multicol_card_le a t m ht (natCast_ne_zero_of_lt goldilocks nv hnv) hinv
  rw [fp3Field_card] at h
  exact h.trans (Nat.mul_le_mul_right _ (by omega))

/-- **Random linear batching over `E`** (`α`, `λ`, `λ_Z`, `μ`, the branch challenge, one coordinate
of a multilinear point): a nonzero affine form `a + x b` vanishes on at most one challenge. -/
theorem fp3_linear_root_card_le (a b : Fp3Field) (h : a ≠ 0 ∨ b ≠ 0) :
    (univ.filter fun x : Fp3Field => a + x * b = 0).card ≤ 1 := by
  by_cases hb : b = 0
  · have ha : a ≠ 0 := h.resolve_right (not_not.mpr hb)
    have : (univ.filter fun x : Fp3Field => a + x * b = 0) = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro x _
      simp [hb, ha]
    rw [this]; simp
  · refine (card_le_card fun x hx => ?_).trans (cmRoot_card_le a b)
    exact cm_mem_root a b x hb (by simpa using hx)

open Classical in
/-- **Zerocheck over `E`, composed.** Let `G ≠ 0` be the table of gate values on `{0,1}^g` (some
row violates the gate), and let `P z` be any family of round-degree `≤ d` polynomials whose
hypercube sum is `G̃(z)` (for relation 4, `P z x = eq(z, x) · Gate(x)` and `d = 4`). Against any
adaptive prover strategy, the verifier that draws `z`, runs the `g`-round sumcheck from the claim
`0` and checks the final evaluation accepts with probability at most `(g + g d) / p^3`. -/
theorem fp3_zerocheck_sound {g : ℕ} (G : (Fin g → Bool) → Fp3Field) (hG : G ≠ 0) (d : ℕ)
    (P : (Fin g → Fp3Field) → (Fin g → Fp3Field) → Fp3Field)
    (hP : ∀ z, RoundDeg d (P z)) (hsum : ∀ z, cubeSum (P z) = mleEval G z)
    (strat : (Fin g → Fp3Field) → Fin g → (Fin g → Fp3Field) → Polynomial Fp3Field)
    (hind : ∀ z i r x, strat z i (Function.update r i x) = strat z i r)
    (hdeg : ∀ z i r, (strat z i r).natDegree ≤ d) :
    ((univ.filter fun zr : (Fin g → Fp3Field) × (Fin g → Fp3Field) =>
        Accepts (P zr.1) 0 (strat zr.1) zr.2).card : ℝ)
      ≤ (((g + g * d : ℕ) : ℝ) / (goldilocks ^ 3 : ℕ))
        * (Fintype.card ((Fin g → Fp3Field) × (Fin g → Fp3Field)) : ℝ) := by
  set N : ℝ := (Fintype.card (Fin g → Fp3Field) : ℝ) with hN
  have hN0 : 0 ≤ N := Nat.cast_nonneg _
  have hcount : (univ.filter fun zr : (Fin g → Fp3Field) × (Fin g → Fp3Field) =>
        Accepts (P zr.1) 0 (strat zr.1) zr.2).card
      = ∑ z : Fin g → Fp3Field,
          (univ.filter fun r : Fin g → Fp3Field => Accepts (P z) 0 (strat z) r).card := by
    rw [Finset.card_filter, Fintype.sum_prod_type]
    exact Finset.sum_congr rfl fun z _ => (Finset.card_filter _ _).symm
  have hc0 : 0 ≤ ((g : ℝ) * d / (goldilocks ^ 3 : ℕ)) := by positivity
  have hz : ∀ z : Fin g → Fp3Field,
      ((univ.filter fun r : Fin g → Fp3Field => Accepts (P z) 0 (strat z) r).card : ℝ)
        ≤ (if mleEval G z = 0 then 1 else 0) * N + ((g : ℝ) * d / (goldilocks ^ 3 : ℕ)) * N := by
    intro z
    by_cases h0 : mleEval G z = 0
    · rw [if_pos h0, one_mul]
      have h1 : ((univ.filter fun r : Fin g → Fp3Field => Accepts (P z) 0 (strat z) r).card : ℝ)
          ≤ N := by
        have hle : (univ.filter fun r : Fin g → Fp3Field => Accepts (P z) 0 (strat z) r).card
            ≤ Fintype.card (Fin g → Fp3Field) := Finset.card_le_univ _
        rw [hN]; exact_mod_cast hle
      have h2 := mul_nonneg hc0 hN0
      linarith
    · rw [if_neg h0, zero_mul, zero_add]
      exact fp3_sumcheck_sound (P z) d (hP z) 0 (by rw [hsum z]; exact fun h => h0 h.symm)
        (strat z) (hind z) (hdeg z)
  have hbad := fp3_mle_zero_sound G hG
  have hsplit : (((g + g * d : ℕ) : ℝ) / (goldilocks ^ 3 : ℕ))
      = (g : ℝ) / (goldilocks ^ 3 : ℕ) + (g : ℝ) * d / (goldilocks ^ 3 : ℕ) := by
    push_cast; ring
  rw [hcount, Nat.cast_sum, Fintype.card_prod, Nat.cast_mul, hsplit]
  calc ∑ z : Fin g → Fp3Field,
        ((univ.filter fun r : Fin g → Fp3Field => Accepts (P z) 0 (strat z) r).card : ℝ)
      ≤ ∑ z : Fin g → Fp3Field,
          ((if mleEval G z = 0 then 1 else 0) * N + ((g : ℝ) * d / (goldilocks ^ 3 : ℕ)) * N) :=
        Finset.sum_le_sum fun z _ => hz z
    _ = ((univ.filter fun z : Fin g → Fp3Field => mleEval G z = 0).card : ℝ) * N
          + ((g : ℝ) * d / (goldilocks ^ 3 : ℕ)) * N * N := by
        rw [Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_boole, Finset.sum_const,
          Finset.card_univ, nsmul_eq_mul]
        ring
    _ ≤ _ := by
        have := mul_le_mul_of_nonneg_right hbad hN0
        linarith

end Instantiation

/-! ## 5. The relation-4 algebraic error budget -/

/-- An explicit numerator for the union bound over every verifier challenge of relation 4 on a
layout with `g = n − 5`, `S = g + 3` (errors are `numerator / p^3`):

* `β`, `γ`: `3 · 2^(S+1)`. With `nv + nt ≤ 2^(S+1)` leaves of the fraction tree, a width-2
  fingerprinted lookup fails on `≤ (2 nt + nv) / p^3` of the pairs `(γ, β)`
  (`fp3_wiring_logup_card_le`) and hits a pole on `≤ (nv + nt) / p^3` (`pole_card_le`);
* layer 0: `α` and the branch challenge, `2`;
* layers `k = 1..S`: `k` degree-3 rounds plus `α`, `λ`, branch, `Σ_k (3k + 3)`;
* zerocheck: the point `z` (`g`), `λ_Z` (`1`), `g` degree-4 rounds (`4g`);
* `μ`: batching of at most `S + 9` claimed evaluations and three more terms, `S + 16`;
* Basefold: `n + 2` degree-2 rounds (`θ`, `r_1`, `r'`), `2 (n + 2)`.

The `μ` term is an allowance (one root per batched claim), not an instantiated lemma. The whole
count is a budget: it is not derived from the executable verifier. -/
def rel4AlgNumer (lay : Rel4Layout) : ℕ :=
  3 * 2 ^ (lay.s + 1) + 2 + (3 * (lay.s * (lay.s + 1) / 2) + 3 * lay.s)
    + (lay.g + 1 + 4 * lay.g) + (lay.s + 16) + 2 * (lay.nVars + 2)

theorem rel4AlgNumer_lt (lay : Rel4Layout) (h : lay.nVars ≤ 28) : rel4AlgNumer lay < 2 ^ 30 := by
  have hg : lay.g ≤ 23 := by unfold Rel4Layout.g; omega
  have hs : lay.s ≤ 26 := by unfold Rel4Layout.s; omega
  have hp : 2 ^ (lay.s + 1) ≤ 2 ^ 27 := Nat.pow_le_pow_right (by norm_num) (by omega)
  have hq : lay.s * (lay.s + 1) ≤ 26 * 27 := Nat.mul_le_mul hs (by omega)
  have hq' : lay.s * (lay.s + 1) / 2 ≤ 351 := by omega
  unfold rel4AlgNumer
  generalize 2 ^ (lay.s + 1) = t at hp ⊢
  generalize lay.s * (lay.s + 1) / 2 = u at hq' ⊢
  norm_num at hp ⊢
  omega

/-- Every layout the bytes verifier admits has `12 ≤ n ≤ 28`. -/
theorem rel4Layout_nVars_le {p : Fpn2Profile} {lay : Rel4Layout} (h : rel4Layout p = some lay) :
    12 ≤ lay.nVars ∧ lay.nVars ≤ 28 := by
  unfold rel4Layout at h
  split at h
  · unfold Rel4Layout.ofProfile at h
    split at h
    · rename_i hc
      cases h
      exact ⟨hc.2.1, hc.2.2.2.1⟩
    · cases h
  · cases h

/-- **Concrete algebraic error of relation 4 over `E`:** below `2^-150` (indeed `≤ 2^-160`) for
every admissible layout, in particular `n = 14` (`fpn2-folded-air-v1`), `n = 24`, `n = 28`. -/
theorem rel4_alg_error_lt (lay : Rel4Layout) (h : lay.nVars ≤ 28) :
    (rel4AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ ∧
      (rel4AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) < (2 ^ 150)⁻¹ := by
  have h160 : (rel4AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ := by
    rw [frac_le_inv_pow2_iff _ _ _ (by rw [goldilocks_eq]; norm_num)]
    calc rel4AlgNumer lay * 2 ^ 160 ≤ 2 ^ 30 * 2 ^ 160 :=
          Nat.mul_le_mul_right _ (rel4AlgNumer_lt lay h).le
      _ ≤ goldilocks ^ 3 := by rw [goldilocks_eq]; norm_num
  exact ⟨h160, lt_of_le_of_lt h160 (by norm_num)⟩

theorem rel4_alg_error_n14_n24_n28 :
    (rel4AlgNumer ⟨14, 72⟩ : ℝ) / (goldilocks ^ 3 : ℕ) < (2 ^ 150)⁻¹ ∧
      (∀ q, (rel4AlgNumer ⟨24, q⟩ : ℝ) / (goldilocks ^ 3 : ℕ) < (2 ^ 150)⁻¹) ∧
      (∀ q, (rel4AlgNumer ⟨28, q⟩ : ℝ) / (goldilocks ^ 3 : ℕ) < (2 ^ 150)⁻¹) :=
  ⟨(rel4_alg_error_lt _ (by decide)).2, fun _ => (rel4_alg_error_lt _ (by simp)).2,
    fun _ => (rel4_alg_error_lt _ (by simp)).2⟩

/-- The registered production profile `fpn2-folded-air-v1`. -/
theorem rel4_alg_error_foldedAirV1 :
    ∃ lay, rel4Layout foldedAirV1 = some lay ∧
      (rel4AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) < (2 ^ 150)⁻¹ :=
  ⟨_, rel4Layout_foldedAirV1.1, (rel4_alg_error_lt _ (by decide)).2⟩

/-- The registered production-scale profile `fpn2-folded-air-n20-v1` (`n = 20`, `K = 32768`). -/
theorem rel4_alg_error_foldedAirN20V1 :
    ∃ lay, rel4Layout foldedAirN20V1 = some lay ∧
      (rel4AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) < (2 ^ 150)⁻¹ :=
  ⟨_, rel4Layout_foldedAirN20.1, (rel4_alg_error_lt _ (by decide)).2⟩

open Finset in
/-- **Union bound at the relation-4 budget.** Any finite family of bad-challenge events whose
densities are `d i / p^3` with `Σ d i ≤ rel4AlgNumer lay` has a union of density `≤ 2^-160`. -/
theorem rel4_alg_union_bound {Ω ι : Type*} [Fintype Ω] [DecidableEq Ω] [Fintype ι]
    (lay : Rel4Layout) (hlay : lay.nVars ≤ 28) (Bad : ι → Finset Ω) (d : ι → ℕ)
    (hBad : ∀ i, ((Bad i).card : ℝ) ≤ ((d i : ℝ) / (goldilocks ^ 3 : ℕ)) * Fintype.card Ω)
    (hd : ∑ i, d i ≤ rel4AlgNumer lay) :
    ((univ.biUnion Bad).card : ℝ) ≤ (2 ^ 160)⁻¹ * Fintype.card Ω := by
  have hΩ : (0 : ℝ) ≤ Fintype.card Ω := Nat.cast_nonneg _
  have h1 : ((univ.biUnion Bad).card : ℝ) ≤ ∑ i, ((Bad i).card : ℝ) := by
    exact_mod_cast Finset.card_biUnion_le
  have h2 : ∑ i, ((Bad i).card : ℝ)
      ≤ ((∑ i, d i : ℕ) : ℝ) / (goldilocks ^ 3 : ℕ) * Fintype.card Ω := by
    rw [Nat.cast_sum, Finset.sum_div, Finset.sum_mul]
    exact Finset.sum_le_sum fun i _ => hBad i
  have h3 : ((∑ i, d i : ℕ) : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ :=
    le_trans (div_le_div_of_nonneg_right (by exact_mod_cast hd) fp3Field_card_pos.le)
      (rel4_alg_error_lt lay hlay).1
  exact h1.trans (h2.trans (mul_le_mul_of_nonneg_right h3 hΩ))

/-! ## 6. A concrete round-by-round instance of `ProtocolBridge` over `E` -/

section Lifetime

open Finset

/-- **The answer space of one Fiat–Shamir round**: one algebraic challenge in `E` and `k` sampling
seeds in `E`, one per spot-check draw. A random-oracle answer is modelled as a uniform element. -/
abbrev Rel4Ans (k : ℕ) : Type := Fp3Field × (Fin k → Fp3Field)

/-- The spot-check miss probability `(1 − σ)^k` at the threshold `σ = num / den`. -/
noncomputable def rel4Miss (k num den : ℕ) : ℝ := (((den - num : ℕ) : ℝ) / (den : ℝ)) ^ k

theorem rel4Miss_nonneg (k num den : ℕ) : 0 ≤ rel4Miss k num den := by
  unfold rel4Miss; positivity

/-- `σ = 1/2`: the miss probability is `2^-k`. -/
theorem rel4Miss_half (k : ℕ) : rel4Miss k 1 2 = (1 / 2) ^ k := by
  unfold rel4Miss; norm_num

/-- `σ = 3/10`: the miss probability is `(7/10)^k`. -/
theorem rel4Miss_three_tenths (k : ℕ) : rel4Miss k 3 10 = (7 / 10) ^ k := by
  unfold rel4Miss; norm_num

/-- **One Fiat–Shamir round of the relation-4 argument, as an adaptive prover prepares it.** The
random-oracle answer of the round is a uniform element of `Rel4Ans k`; the constructor fixes which
answers let a false claim survive:

* `sumcheck`: a degree-`≤ 4` round polynomial that differs from the true partial sum;
* `logup`: an invalid lookup of at most `2^(S+1)` fractions, before `β`;
* `linear`: a nonzero affine form (`α`, `λ`, `λ_Z`, `μ`, branch, one coordinate of `z`);
* `spot`: a sampling round. `badSeed` is the set of seeds whose draw opens a bad block, and the
  only premise is the threshold itself: bad seeds have density at least `σ = num / den`. No escape
  bound is carried: `rel4Round_esc_card_le` derives `(1 − σ)^k` from `SpotCheck.frac_bad_bound`.

The three algebraic rounds read the first component of the answer, the sampling round the `k`
seeds. -/
inductive Rel4Round (lay : Rel4Layout) (k num den : ℕ) : Type
  | sumcheck (n d : ℕ) (hd : d ≤ 4) (P : (Fin n → Fp3Field) → Fp3Field) (hP : RoundDeg d P)
      (g : Fin n → (Fin n → Fp3Field) → Polynomial Fp3Field) (hg : ∀ i r, (g i r).natDegree ≤ d)
      (i : Fin n) (r : Fin n → Fp3Field)
  | logup (nv nt : ℕ) (hsz : nv + nt ≤ 2 ^ (lay.s + 1)) (a : Fin nv → Fp3Field)
      (t : Fin nt → Fp3Field) (mult : Fin nt → Fp3Field) (ht : Function.Injective t)
      (hinv : ¬ LookupValid a t mult)
  | linear (a b : Fp3Field) (h : a ≠ 0 ∨ b ≠ 0)
  | spot (badSeed : Finset Fp3Field)
      (hbad : num * Fintype.card Fp3Field ≤ den * badSeed.card)

open Classical in
/-- The escape set of a round: the oracle answers on which the false claim survives. For a
sampling round these are the answers all of whose `k` seeds avoid `badSeed`
(`SpotCheck.missing`). -/
noncomputable def Rel4Round.esc {lay : Rel4Layout} {k num den : ℕ} :
    Rel4Round lay k num den → Finset (Rel4Ans k)
  | .sumcheck _ _ _ P _ g _ i r => badSet P g i r ×ˢ univ
  | .logup _ _ _ a t mult _ _ =>
    (univ.filter fun l => ¬ NonPole a t l ∨ Harmonic a t mult l) ×ˢ univ
  | .linear a b _ => (univ.filter fun x => a + x * b = 0) ×ˢ univ
  | .spot badSeed _ => univ ×ˢ missing badSeed k

/-- `true` for sampling rounds, `false` for algebraic rounds. -/
def Rel4Round.isSpot {lay : Rel4Layout} {k num den : ℕ} : Rel4Round lay k num den → Bool
  | .spot _ _ => true
  | _ => false

/-- Per-round escape density: `m + 2^-106` (sampling), `2^-106` (algebraic). The sampling rounds
meet the tighter `m`; the `2^-106` term is the unused seed-map allowance of `ProtocolBridge`. -/
noncomputable def rel4Eps (m : ℝ) (b : Bool) : ℝ := if b then m + (2 ^ 106)⁻¹ else (2 ^ 106)⁻¹

/-- A set of challenges of density `≤ ε` in `E` has density `≤ ε` in the answer space. -/
theorem rel4_alg_lift (k : ℕ) (bad : Finset Fp3Field) (ε : ℝ)
    (h : (bad.card : ℝ) ≤ ε * Fintype.card Fp3Field) :
    (((bad ×ˢ (univ : Finset (Fin k → Fp3Field))).card : ℕ) : ℝ)
      ≤ ε * Fintype.card (Rel4Ans k) := by
  rw [card_product, card_univ, Fintype.card_prod, Nat.cast_mul, Nat.cast_mul, ← mul_assoc]
  exact mul_le_mul_of_nonneg_right h (Nat.cast_nonneg _)

/-- **The sampling escape bound, from `SpotCheck.frac_bad_bound`.** If the bad seeds have density
at least `num / den` in `E`, the `k`-tuples of seeds that all avoid them have density at most
`((den − num) / den)^k`, for every `k` and every threshold. -/
theorem spot_miss_card_le (k num den : ℕ) (hden : 0 < den) (badSeed : Finset Fp3Field)
    (hbad : num * Fintype.card Fp3Field ≤ den * badSeed.card) :
    ((missing badSeed k).card : ℝ)
      ≤ rel4Miss k num den * Fintype.card (Fin k → Fp3Field) := by
  have h := frac_bad_bound badSeed k num den hbad
  have h' : ((missing badSeed k).card : ℝ) * (den : ℝ) ^ k
      ≤ (((den - num : ℕ) : ℝ) * (Fintype.card Fp3Field : ℝ)) ^ k := by exact_mod_cast h
  have hd0 : (0 : ℝ) < (den : ℝ) := by exact_mod_cast hden
  have hd : (0 : ℝ) < (den : ℝ) ^ k := pow_pos hd0 k
  rw [rel4Miss, Fintype.card_fun, Fintype.card_fin, Nat.cast_pow, div_pow, div_mul_eq_mul_div,
    le_div_iff₀ hd, ← mul_pow]
  exact h'

/-- The same bound on the answer space `Rel4Ans k`. -/
theorem spot_esc_card_le (k num den : ℕ) (hden : 0 < den) (badSeed : Finset Fp3Field)
    (hbad : num * Fintype.card Fp3Field ≤ den * badSeed.card) :
    ((((univ : Finset Fp3Field) ×ˢ missing badSeed k).card : ℕ) : ℝ)
      ≤ rel4Miss k num den * Fintype.card (Rel4Ans k) := by
  have h := spot_miss_card_le k num den hden badSeed hbad
  rw [card_product, card_univ, Fintype.card_prod, Nat.cast_mul, Nat.cast_mul, mul_left_comm]
  exact mul_le_mul_of_nonneg_left h (Nat.cast_nonneg _)

open Classical in
/-- **Every escape set meets its bound**: this is `hEsc` of `ProtocolBridge`, proved for all four
kinds of round. In particular no sampling bound is assumed, for any `k` and any `σ = num / den`. -/
theorem rel4Round_esc_card_le (lay : Rel4Layout) (hlay : lay.nVars ≤ 28) (k num den : ℕ)
    (hden : 0 < den) (s : Rel4Round lay k num den) :
    ((s.esc).card : ℝ) ≤ rel4Eps (rel4Miss k num den) s.isSpot * Fintype.card (Rel4Ans k) := by
  cases s with
  | sumcheck n d hd P hP g hg i r =>
    have h := badSet_card_le P d hP g hg i r
    have h' : ((badSet P g i r).card : ℝ) ≤ ((4 : ℕ) : ℝ) := by exact_mod_cast h.trans hd
    exact rel4_alg_lift k _ _ (h'.trans (fp3_small_card_le 4 (by norm_num)))
  | logup nv nt hsz a t mult ht hinv =>
    have hs : lay.s ≤ 26 := by unfold Rel4Layout.s Rel4Layout.g; omega
    have hp : 2 ^ (lay.s + 1) ≤ 2 ^ 27 := Nat.pow_le_pow_right (by norm_num) (by omega)
    have hnv : nv < goldilocks := by rw [goldilocks_eq]; norm_num at hp; omega
    have h := pole_or_harmonic_card_le a t mult ht (natCast_ne_zero_of_lt goldilocks nv hnv) hinv
    have h' : (((univ.filter fun l : Fp3Field => ¬ NonPole a t l ∨ Harmonic a t mult l).card : ℕ) : ℝ)
        ≤ ((2 ^ 28 : ℕ) : ℝ) := by
      refine Nat.cast_le.mpr (h.trans ?_)
      norm_num at hp ⊢; omega
    exact rel4_alg_lift k _ _ (h'.trans (fp3_small_card_le _ (by norm_num)))
  | linear a b h =>
    have h' : (((univ.filter fun x : Fp3Field => a + x * b = 0).card : ℕ) : ℝ) ≤ ((1 : ℕ) : ℝ) :=
      Nat.cast_le.mpr (fp3_linear_root_card_le a b h)
    exact rel4_alg_lift k _ _ (h'.trans (fp3_small_card_le 1 (by norm_num)))
  | spot badSeed hbad =>
    refine (spot_esc_card_le k num den hden badSeed hbad).trans
      (mul_le_mul_of_nonneg_right ?_ (Nat.cast_nonneg _))
    show rel4Miss k num den ≤ rel4Miss k num den + (2 ^ 106)⁻¹
    linarith [show (0 : ℝ) ≤ (2 ^ 106)⁻¹ by positivity]

/-- **A sampling round built from a block pool** (`SpotCheck.cost_weighted_bound` with the seeds as
cost units). Each of the `k` seeds `e ∈ E` opens the block `u e` of a pool `B`; `Bad` is the set of
blocks violating the block relation. The single premise `hshare` says that the sampling
distribution induced by `u` puts mass at least `σ = num / den` on `Bad`. Any `k`, any `σ`:
`σ = 1/2` with `k = 51` or `105`, and `σ = 3/10` with `k = 100`, are instances. -/
noncomputable def Rel4Round.ofSpotCheck (lay : Rel4Layout) (k num den : ℕ) {B : Type*}
    [DecidableEq B] (u : Fp3Field → B) (Bad : Finset B)
    (hshare : num * Fintype.card Fp3Field ≤ den * (univ.filter fun e => u e ∈ Bad).card) :
    Rel4Round lay k num den :=
  .spot (univ.filter fun e => u e ∈ Bad) hshare

/-- The escape answers of `ofSpotCheck` are exactly those whose `k` sampled blocks all miss `Bad`. -/
theorem Rel4Round.mem_ofSpotCheck_esc (lay : Rel4Layout) (k num den : ℕ) {B : Type*}
    [DecidableEq B] (u : Fp3Field → B) (Bad : Finset B)
    (hshare : num * Fintype.card Fp3Field ≤ den * (univ.filter fun e => u e ∈ Bad).card)
    (a : Rel4Ans k) :
    a ∈ (Rel4Round.ofSpotCheck lay k num den u Bad hshare).esc ↔ ∀ j, u (a.2 j) ∉ Bad := by
  simp [Rel4Round.ofSpotCheck, Rel4Round.esc, missing]

/-- **Seed share from block share.** If every bad block is opened by at least `c` seeds (a seed
map whose fibers have size `≥ c`, e.g. `c = ⌊p^3 / |B|⌋` for a floor map) and
`num · p^3 ≤ den · c · |Bad|`, then the premise of `ofSpotCheck` holds. With an exactly balanced
map (`c · |B| = p^3`) this is the block share `num · |B| ≤ den · |Bad|`; otherwise the gap between
`c · |B|` and `p^3` is the seed-map bias, here an explicit loss of threshold. -/
theorem seedShare_of_fiber_ge {B : Type*} [DecidableEq B] (u : Fp3Field → B) (Bad : Finset B)
    (c num den : ℕ) (hfib : ∀ b ∈ Bad, c ≤ (univ.filter fun e => u e = b).card)
    (h : num * Fintype.card Fp3Field ≤ den * (c * Bad.card)) :
    num * Fintype.card Fp3Field ≤ den * (univ.filter fun e => u e ∈ Bad).card := by
  refine h.trans (Nat.mul_le_mul_left _ ?_)
  rw [card_eq_sum_card_fiberwise (f := u) (s := univ.filter fun e => u e ∈ Bad) (t := Bad)
    (by intro e he; simpa using he)]
  calc c * Bad.card = ∑ _b ∈ Bad, c := by rw [sum_const, smul_eq_mul, mul_comm]
    _ ≤ _ := by
        refine sum_le_sum fun b hb => (hfib b hb).trans (card_le_card fun e he => ?_)
        simp only [mem_filter, mem_univ, true_and] at he ⊢
        exact ⟨by rw [he]; exact hb, he⟩

/-- The bad-accept event of the model: some oracle answer lands in the escape set of the round
the adaptive prover prepared for it. -/
noncomputable def rel4BadAcc {lay : Rel4Layout} {k num den Q : ℕ}
    (st : Fin Q → (Fin Q → Rel4Ans k) → Rel4Round lay k num den) :
    Finset (Fin Q → Rel4Ans k) := by
  classical exact univ.filter fun a => ∃ j, a j ∈ (st j a).esc

/-- **`ProtocolRBRSpec` inhabited at a concrete model** (`BadAcc = rel4BadAcc st`, `Fail = ∅`,
`Doomed = True`, `Esc = Rel4Round.esc`): the first-escape path has one step, located at a query
whose answer escapes. Since `BadAcc` is *defined* as the escape event, `hInit`, `hFinal` and `hStep`
hold by construction: this shows the structure is satisfiable at a non-empty event, it is not an
extraction argument. The proved content of the instance is `rel4Round_esc_card_le` plus the
counting of `RoundByRound`. -/
noncomputable def rel4RbrSpec {lay : Rel4Layout} {k num den Q : ℕ} (hQ : 0 < Q)
    (st : Fin Q → (Fin Q → Rel4Ans k) → Rel4Round lay k num den) :
    ProtocolRBRSpec st (fun _ => True) Rel4Round.esc (rel4BadAcc st) ∅ := by
  classical exact
  { R := fun _ => 1
    roundHolds := fun _ i => (i : ℕ) = 1
    qIdx := fun a _ => if h : ∃ j, a j ∈ (st j a).esc then Classical.choose h else ⟨0, hQ⟩
    hInit := fun _ _ _ h => absurd h (by decide)
    hFinal := fun _ _ _ => rfl
    hStep := fun a ha _ _ _ _ => by
      have h : ∃ j, a j ∈ (st j a).esc := by
        unfold rel4BadAcc at ha
        exact (Finset.mem_filter.mp ha).2
      rw [dif_pos h]
      exact ⟨trivial, Classical.choose_spec h⟩ }

/-- **General lifetime bound of the model**, through `hpath_of_protocol_spec`: over `Q` adaptive
random-oracle queries, `Pr[BadAcc] ≤ Q · ((1 − σ)^k + 2^-106)`, for every `k` and `σ = num / den`. -/
theorem rel4_rbr_lifetime_general (lay : Rel4Layout) (hlay : lay.nVars ≤ 28) (k num den : ℕ)
    (hden : 0 < den) {Q : ℕ} (hQ0 : 0 < Q)
    (st : Fin Q → (Fin Q → Rel4Ans k) → Rel4Round lay k num den)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a) :
    ((rel4BadAcc st).card : ℝ)
      ≤ (Q * (rel4Miss k num den + (2 ^ 106)⁻¹)) * (Fintype.card (Rel4Ans k) : ℝ) ^ Q := by
  have hm0 := rel4Miss_nonneg k num den
  have h := rbr_lifetime_soundness st hst (fun _ => True) Rel4Round.esc Rel4Round.isSpot
    (rel4Eps (rel4Miss k num den)) (rel4Miss k num den + (2 ^ 106)⁻¹)
    (add_nonneg hm0 (by positivity))
    (fun b => by cases b <;> simp [rel4Eps, hm0])
    (fun s _ => rel4Round_esc_card_le lay hlay k num den hden s) (rel4BadAcc st) ∅ 0 (by simp)
    (hpath_of_protocol_spec (rel4RbrSpec hQ0 st))
  simpa using h

/-- The general bound against a query budget `Q ≤ Qmax` and a target `ε`. -/
theorem rel4_lifetime_of_budget (lay : Rel4Layout) (hlay : lay.nVars ≤ 28) (k num den : ℕ)
    (hden : 0 < den) {Q : ℕ} (hQ0 : 0 < Q)
    (st : Fin Q → (Fin Q → Rel4Ans k) → Rel4Round lay k num den)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a) (Qmax ε : ℝ) (hQ : (Q : ℝ) ≤ Qmax)
    (hε : Qmax * (rel4Miss k num den + (2 ^ 106)⁻¹) ≤ ε) :
    ((rel4BadAcc st).card : ℝ) ≤ ε * (Fintype.card (Rel4Ans k) : ℝ) ^ Q := by
  have h := rel4_rbr_lifetime_general lay hlay k num den hden hQ0 st hst
  refine h.trans (mul_le_mul_of_nonneg_right ?_ (by positivity))
  exact (mul_le_mul_of_nonneg_right hQ
    (add_nonneg (rel4Miss_nonneg k num den) (by positivity))).trans hε

/-- **Non-interactive lifetime soundness of the model (`k = 105`, `σ = 1/2`, `Q ≤ 2^64`):**
`Pr[BadAcc] ≤ 2^-40`. This is `fpn_protocol_lifetime_soundness` with every argument instantiated:
`Ω = Rel4Ans 105`, `m = 2^-105`, `amap = εalg = 2^-106`, `Δ = 0`, and the spec `rel4RbrSpec`. -/
theorem rel4_lifetime_soundness_k105 (lay : Rel4Layout) (hlay : lay.nVars ≤ 28) {Q : ℕ}
    (hQ0 : 0 < Q) (hQ : Q ≤ 2 ^ 64)
    (st : Fin Q → (Fin Q → Rel4Ans 105) → Rel4Round lay 105 1 2)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a) :
    ((rel4BadAcc st).card : ℝ) ≤ (2 ^ 40)⁻¹ * (Fintype.card (Rel4Ans 105) : ℝ) ^ Q :=
  fpn_protocol_lifetime_soundness hQ st hst (fun _ => True) Rel4Round.esc Rel4Round.isSpot
    (rel4Eps (rel4Miss 105 1 2)) (rel4Miss 105 1 2) (2 ^ 106)⁻¹ (2 ^ 106)⁻¹ 0
    (rel4Miss_nonneg _ _ _) (rel4Miss_half 105).le
    (by positivity) le_rfl le_rfl (by positivity)
    (fun b => by cases b <;> simp [rel4Eps])
    (fun s _ => rel4Round_esc_card_le lay hlay 105 1 2 (by norm_num) s) (rel4BadAcc st) ∅
    (by simp) (rel4RbrSpec hQ0 st)

/-- **External-seed lifetime soundness at `σ = 1/2` (`k = 51`, `Q ≤ 2^30`):** `Pr[BadAcc] ≤ 2^-20`
(`LifetimeSoundness.table_Q30_e20`). -/
theorem rel4_lifetime_soundness_k51 (lay : Rel4Layout) (hlay : lay.nVars ≤ 28) {Q : ℕ}
    (hQ0 : 0 < Q) (hQ : Q ≤ 2 ^ 30)
    (st : Fin Q → (Fin Q → Rel4Ans 51) → Rel4Round lay 51 1 2)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a) :
    ((rel4BadAcc st).card : ℝ) ≤ (2 ^ 20)⁻¹ * (Fintype.card (Rel4Ans 51) : ℝ) ^ Q :=
  rel4_lifetime_of_budget lay hlay 51 1 2 (by norm_num) hQ0 st hst (2 ^ 30) _
    (by exact_mod_cast hQ) (by rw [rel4Miss_half]; norm_num)

/-- **External-seed lifetime soundness at `σ = 3/10` (`k = 100`, `Q ≤ 2^30`):**
`Pr[BadAcc] ≤ 2^-20` (`LifetimeSoundness.table_Q30_e20`: `k = 100` is the requirement at
`σ = 3/10`). The sampling term `(7/10)^100` is derived (`spot_miss_card_le`), not assumed. -/
theorem rel4_lifetime_soundness_k100 (lay : Rel4Layout) (hlay : lay.nVars ≤ 28) {Q : ℕ}
    (hQ0 : 0 < Q) (hQ : Q ≤ 2 ^ 30)
    (st : Fin Q → (Fin Q → Rel4Ans 100) → Rel4Round lay 100 3 10)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a) :
    ((rel4BadAcc st).card : ℝ) ≤ (2 ^ 20)⁻¹ * (Fintype.card (Rel4Ans 100) : ℝ) ^ Q :=
  rel4_lifetime_of_budget lay hlay 100 3 10 (by norm_num) hQ0 st hst (2 ^ 30) _
    (by exact_mod_cast hQ) (by rw [rel4Miss_three_tenths]; norm_num)

/-- Strict forms: `Pr[BadAcc] < 2^-40` (`k = 105`, `σ = 1/2`, `Q ≤ 2^64`) and `< 2^-20`
(`k = 100`, `σ = 3/10`, `Q ≤ 2^30`). -/
theorem rel4_lifetime_soundness_strict (lay : Rel4Layout) (hlay : lay.nVars ≤ 28) {Q : ℕ}
    (hQ0 : 0 < Q) :
    (Q ≤ 2 ^ 64 → ∀ st : Fin Q → (Fin Q → Rel4Ans 105) → Rel4Round lay 105 1 2,
      (∀ j a x, st j (Function.update a j x) = st j a) →
      ((rel4BadAcc st).card : ℝ) < (2 ^ 40)⁻¹ * (Fintype.card (Rel4Ans 105) : ℝ) ^ Q) ∧
    (Q ≤ 2 ^ 30 → ∀ st : Fin Q → (Fin Q → Rel4Ans 100) → Rel4Round lay 100 3 10,
      (∀ j a x, st j (Function.update a j x) = st j a) →
      ((rel4BadAcc st).card : ℝ) < (2 ^ 20)⁻¹ * (Fintype.card (Rel4Ans 100) : ℝ) ^ Q) := by
  have hpos : ∀ k : ℕ, (0 : ℝ) < (Fintype.card (Rel4Ans k) : ℝ) ^ Q := fun k =>
    pow_pos (Nat.cast_pos.mpr Fintype.card_pos) Q
  constructor
  · intro hQ st hst
    have h := rel4_rbr_lifetime_general lay hlay 105 1 2 (by norm_num) hQ0 st hst
    refine lt_of_le_of_lt h (mul_lt_mul_of_pos_right ?_ (hpos 105))
    have hQr : (Q : ℝ) ≤ 2 ^ 64 := by exact_mod_cast hQ
    rw [rel4Miss_half]
    calc (Q : ℝ) * ((1 / 2) ^ 105 + (2 ^ 106)⁻¹)
        ≤ 2 ^ 64 * ((1 / 2) ^ 105 + (2 ^ 106)⁻¹) :=
          mul_le_mul_of_nonneg_right hQr (by positivity)
      _ < (2 ^ 40)⁻¹ := by norm_num
  · intro hQ st hst
    have h := rel4_rbr_lifetime_general lay hlay 100 3 10 (by norm_num) hQ0 st hst
    refine lt_of_le_of_lt h (mul_lt_mul_of_pos_right ?_ (hpos 100))
    have hQr : (Q : ℝ) ≤ 2 ^ 30 := by exact_mod_cast hQ
    rw [rel4Miss_three_tenths]
    calc (Q : ℝ) * ((7 / 10) ^ 100 + (2 ^ 106)⁻¹)
        ≤ 2 ^ 30 * ((7 / 10) ^ 100 + (2 ^ 106)⁻¹) :=
          mul_le_mul_of_nonneg_right hQr (by positivity)
      _ < (2 ^ 20)⁻¹ := by norm_num

/-- The model is not vacuous: already with one query there is a non-anticipating prover state
whose bad-accept event is nonempty. -/
theorem rel4_rbr_model_nontrivial (lay : Rel4Layout) (k num den : ℕ) :
    ∃ st : Fin 1 → (Fin 1 → Rel4Ans k) → Rel4Round lay k num den,
      (∀ j a x, st j (Function.update a j x) = st j a) ∧ (rel4BadAcc st).Nonempty := by
  refine ⟨fun _ _ => .linear 0 1 (Or.inr one_ne_zero), fun _ _ _ => rfl,
    fun _ => (0, fun _ => 0), ?_⟩
  unfold rel4BadAcc
  simp [Rel4Round.esc]

/-- The sampling premise is satisfiable at `σ = 3/10` with a nonempty escape event: a sampling
state whose bad seeds are all of `E` but one. -/
theorem rel4_spot_model_nontrivial (lay : Rel4Layout) (k : ℕ) :
    ∃ st : Fin 1 → (Fin 1 → Rel4Ans k) → Rel4Round lay k 3 10,
      (∀ j a x, st j (Function.update a j x) = st j a) ∧ (∀ j a, (st j a).isSpot = true) ∧
        (rel4BadAcc st).Nonempty := by
  have hc : 1 < Fintype.card Fp3Field := Fintype.one_lt_card
  refine ⟨fun _ _ => .spot (univ.erase 0) ?_, fun _ _ _ => rfl, fun _ _ => rfl,
    fun _ => (0, fun _ => 0), ?_⟩
  · rw [card_erase_of_mem (mem_univ _), card_univ]; omega
  · unfold rel4BadAcc
    simp [Rel4Round.esc, missing]

end Lifetime

/-! ## 7. What an accepting `verifyBytesRel4` run says in `E` -/

/-- `eq(a, b) = Π_i (a_i b_i + (1 − a_i)(1 − b_i))` in `E`. -/
noncomputable def eqField (a b : List Fp3) : Fp3Field :=
  ((a.zip b).map fun p =>
    p.1.toField * p.2.toField + (1 - p.1.toField) * (1 - p.2.toField)).prod

theorem eq1_toField (a b : Fp3) :
    (eq1 a b).toField = a.toField * b.toField + (1 - a.toField) * (1 - b.toField) := by
  simp only [eq1, Fp3.toField_add, Fp3.toField_mul, Fp3.toField_sub, Fp3.toField_one]

theorem eqFold_toField (l : List (Fp3 × Fp3)) (acc : Fp3) :
    (l.foldl (fun acc p => acc.mul (eq1 p.1 p.2)) acc).toField =
      acc.toField * (l.map fun p =>
        p.1.toField * p.2.toField + (1 - p.1.toField) * (1 - p.2.toField)).prod := by
  induction l generalizing acc with
  | nil => simp
  | cons p l ih =>
    rw [List.foldl_cons, ih, Fp3.toField_mul, eq1_toField, List.map_cons, List.prod_cons, mul_assoc]

theorem eqEval_toField (a b : List Fp3) : (eqEval a b).toField = eqField a b := by
  rw [eqEval, eqFold_toField, Fp3.toField_one, one_mul]; rfl

/-- The Plonkish gate `q_M a b + q_L a + q_R b + q_O c + q_C` in `E`. -/
noncomputable def gateField (f : ZcFinals) : Fp3Field :=
  f.qm.toField * (f.ea.toField * f.eb.toField) + f.ql.toField * f.ea.toField
    + (f.qr.toField * f.eb.toField + f.qo.toField * f.ec.toField) + f.qc.toField

theorem gateEval_toField (f : ZcFinals) : (gateEval f).toField = gateField f := by
  simp only [gateEval, gateField, Fp3.toField_add, Fp3.toField_mul]

/-- `(n_l d_r + n_r d_l) + α d_l d_r` in `E`. -/
noncomputable def childrenField (ch : F3x4) (alpha : Fp3) : Fp3Field :=
  ch.1.toField * ch.2.2.2.toField + ch.2.2.1.toField * ch.2.1.toField
    + alpha.toField * (ch.2.1.toField * ch.2.2.2.toField)

theorem childrenClaim_toField (ch : F3x4) (alpha : Fp3) :
    (childrenClaim ch alpha).toField = childrenField ch alpha := by
  simp only [childrenClaim, childrenField, Fp3.toField_add, Fp3.toField_mul]

theorem RunList.imp {σ α : Type} {R R' : σ → α → σ → Prop}
    (h : ∀ s a s', R s a s' → R' s a s') {s : σ} {l : List α} {sF : σ}
    (hr : RunList R s l sF) : RunList R' s l sF := by
  induction hr with
  | nil s => exact .nil s
  | cons s a s' as sF h1 _ ih => exact .cons s a s' as sF (h _ _ _ h1) ih

/-- `RunList.imp` with the step relation weakened only on the inputs of the list. -/
theorem RunList.imp_mem {σ α : Type} {R R' : σ → α → σ → Prop} {s : σ} {l : List α} {sF : σ}
    (hr : RunList R s l sF) :
    (∀ s a s', a ∈ l → R s a s' → R' s a s') → RunList R' s l sF := by
  induction hr with
  | nil s => intro _; exact .nil s
  | cons s a s' as sF h1 _ ih =>
    intro h
    exact .cons s a s' as sF (h _ _ _ (by simp) h1)
      (ih fun s a s' ha => h s a s' (List.mem_cons_of_mem _ ha))

/-- A sumcheck round of a GKR layer in `E`: `g(0) + g(1) = target`. -/
def GkrRoundField (O : Rel4Oracles) (s : RoundState) (g : F3x4) (s' : RoundState) : Prop :=
  g.1.toField + g.2.1.toField = s.target.toField ∧ s' = gkrRoundNext O s g

/-- A masked GKR layer in `E`: the round sums from `num + α den + λ G_sum`, and
`target = eq(pt, r) · children + λ g_r`. -/
def GkrLayerField (O : Rel4Oracles) (s : GkrState) (L : GkrLayer) (s' : GkrState) : Prop :=
  ∃ rf, RunList (GkrRoundField O) (layerRoundsInit O s L) L.rounds rf ∧
    (layerRoundsInit O s L).target.toField =
      s.num.toField + (layerAlpha O s).toField * s.den.toField
        + (layerLamTr O s L).1.toField * L.gSum.toField ∧
    rf.target.toField =
      eqField s.point rf.rs * childrenField L.children (layerAlpha O s)
        + (layerLamTr O s L).1.toField * L.gR.toField ∧
    s' = layerNext O s L rf

/-- A zerocheck round in `E`: `s(0) + s(1) = target`. -/
def ZcRoundField (O : Rel4Oracles) (s : RoundState) (sv : F3x5) (s' : RoundState) : Prop :=
  sv.1.toField + sv.2.1.toField = s.target.toField ∧ s' = zcRoundNext O s sv

/-- A Basefold step in `E`: `h(0) + h(1) = y`, or a fold-root absorption. -/
def PcsOpField (O : Rel4Oracles) (s : PcsState) : PcsOp → PcsState → Prop
  | .round h, s' => h.1.toField + h.2.1.toField = s.y.toField ∧ s' = pcsRoundNext O s h
  | .root r, s' => s' = pcsRootNext O s r

/-- **A sumcheck round of the executable verifier in the shape of `SumcheckSound.Accepts`.** With
`p` the polynomial of degree `≤ d` through the `d + 1` wire values: `p(0) + p(1)` is the sum the
verifier checks, and the next target it computes with `lagrangeEval` is `p` at the challenge. -/
theorem roundPoly_round_shape (vals : List Fp3) (d : ℕ) (hlen : vals.length = d + 1) (hd : 1 ≤ d)
    (hp : d + 1 ≤ goldilocks) (r : Fp3) :
    (roundPoly vals).natDegree ≤ d ∧
      (roundPoly vals).eval 0 + (roundPoly vals).eval 1
        = ((vals.getD 0 Fp3.zero).add (vals.getD 1 Fp3.zero)).toField ∧
      (lagrangeEval vals r).toField = (roundPoly vals).eval r.toField := by
  have hsmall : vals.length ≤ goldilocks := hlen ▸ hp
  have h0 := roundPoly_eval_node vals hsmall 0 (by omega)
  have h1 := roundPoly_eval_node vals hsmall 1 (by omega)
  rw [Nat.cast_zero] at h0
  rw [Nat.cast_one] at h1
  exact ⟨roundPoly_natDegree_le vals d hlen hp, by rw [h0, h1, Fp3.toField_add],
    lagrangeEval_toField vals r⟩

/-- Zerocheck round (degree 4): the round polynomial `p` has `p(0) + p(1) = target` and the next
target is `p(challenge)`, in `E`. -/
theorem zcRound_accepts_shape (O : Rel4Oracles) (s : RoundState) (sv : F3x5) (s' : RoundState)
    (h : ZcRoundSpec O s sv s') :
    (roundPoly (list5 sv)).natDegree ≤ 4 ∧
      (roundPoly (list5 sv)).eval 0 + (roundPoly (list5 sv)).eval 1 = s.target.toField ∧
      s'.target.toField = (roundPoly (list5 sv)).eval (zcRoundChal O s sv).toField := by
  obtain ⟨hd, hs, he⟩ := roundPoly_round_shape (list5 sv) 4 rfl (by norm_num)
    (by rw [goldilocks_eq]; norm_num) (zcRoundChal O s sv)
  refine ⟨hd, ?_, ?_⟩
  · rw [hs, ← h.1]; rfl
  · rw [h.2]; exact he

/-- GKR layer round (degree 3), same shape. -/
theorem gkrRound_accepts_shape (O : Rel4Oracles) (s : RoundState) (g : F3x4) (s' : RoundState)
    (h : GkrRoundSpec O s g s') :
    (roundPoly (list4 g)).natDegree ≤ 3 ∧
      (roundPoly (list4 g)).eval 0 + (roundPoly (list4 g)).eval 1 = s.target.toField ∧
      s'.target.toField = (roundPoly (list4 g)).eval (gkrRoundChal O s g).toField := by
  obtain ⟨hd, hs, he⟩ := roundPoly_round_shape (list4 g) 3 rfl (by norm_num)
    (by rw [goldilocks_eq]; norm_num) (gkrRoundChal O s g)
  refine ⟨hd, ?_, ?_⟩
  · rw [hs, ← h.1]; rfl
  · rw [h.2]; exact he

/-- Basefold round (degree 2), same shape. -/
theorem pcsRound_accepts_shape (O : Rel4Oracles) (s : PcsState) (hh : F3x3) (s' : PcsState)
    (h : PcsOpSpec O s (.round hh) s') :
    (roundPoly (list3 hh)).natDegree ≤ 2 ∧
      (roundPoly (list3 hh)).eval 0 + (roundPoly (list3 hh)).eval 1 = s.y.toField ∧
      s'.y.toField = (roundPoly (list3 hh)).eval (pcsRoundChal O s hh).toField := by
  obtain ⟨hd, hs, he⟩ := roundPoly_round_shape (list3 hh) 2 rfl (by norm_num)
    (by rw [goldilocks_eq]; norm_num) (pcsRoundChal O s hh)
  refine ⟨hd, ?_, ?_⟩
  · rw [hs, ← h.1]; rfl
  · rw [h.2]; exact he

/-! ### Canonical limbs of a parsed container: `root_den ≠ 0` in `E` -/

theorem decMap_ok {α β : Type} {c : Codec α} {f : α → β} {l : Bytes} {b : β} {rest : Bytes}
    (h : decMap c f l = .ok (b, rest)) : ∃ a, c.dec l = .ok (a, rest) ∧ b = f a := by
  unfold decMap at h
  cases h1 : c.dec l with
  | error e => simp [h1] at h
  | ok x =>
    obtain ⟨a, r⟩ := x
    simp only [h1] at h
    injection h with h
    injection h with hb hr
    subst hb hr
    exact ⟨a, rfl, rfl⟩

theorem decPair_ok {α β : Type} {c₁ : Codec α} {c₂ : Codec β} {l : Bytes} {p : α × β}
    {rest : Bytes} (h : decPair c₁ c₂ l = .ok (p, rest)) :
    ∃ r, c₁.dec l = .ok (p.1, r) ∧ c₂.dec r = .ok (p.2, rest) := by
  unfold decPair at h
  cases h1 : c₁.dec l with
  | error e => simp [h1] at h
  | ok x =>
    obtain ⟨a, r⟩ := x
    simp only [h1] at h
    cases h2 : c₂.dec r with
    | error e => simp [h2] at h
    | ok y =>
      obtain ⟨b, r'⟩ := y
      simp only [h2] at h
      injection h with h
      injection h with hp hr
      subst hp hr
      exact ⟨r, rfl, h2⟩

/-- A decoded limb is canonical (`< P`): the parser rejects, never reduces. -/
theorem decFp_lt {l : Bytes} {x : UInt64} {rest : Bytes} (h : decFp l = .ok (x, rest)) :
    x < goldilocksP := by
  unfold decFp at h
  cases hd : decodeFpBytes (l.take 8) with
  | error e => simp [hd] at h
  | ok y =>
    simp only [hd] at h
    injection h with h
    injection h with hx hr
    subst hx
    exact (encodeFpBytes_of_decodeFpBytes hd).2

/-- Every decoded `Fp3` is canonical. -/
theorem fp3_dec_canonical {l : Bytes} {a : Fp3} {rest : Bytes}
    (h : Codec.fp3.dec l = .ok (a, rest)) : a.Canonical := by
  obtain ⟨t, ht, rfl⟩ := decMap_ok (c := Codec.fp.pair (Codec.fp.pair Codec.fp)) h
  obtain ⟨r1, h0, h12⟩ := decPair_ok (c₁ := Codec.fp) (c₂ := Codec.fp.pair Codec.fp) ht
  obtain ⟨r2, h1, h2⟩ := decPair_ok (c₁ := Codec.fp) (c₂ := Codec.fp) h12
  exact ⟨decFp_lt h0, decFp_lt h1, decFp_lt h2⟩

theorem airGkr_dec_canonical {lay : Rel4Layout} {l : Bytes} {G : AirGkr} {rest : Bytes}
    (h : (Codec.airGkr lay).dec l = .ok (G, rest)) :
    G.rootNum.Canonical ∧ G.rootDen.Canonical := by
  obtain ⟨t, ht, rfl⟩ := decMap_ok h
  obtain ⟨r1, h0, h1⟩ := decPair_ok ht
  obtain ⟨r2, h2, _⟩ := decPair_ok h1
  exact ⟨fp3_dec_canonical h0, fp3_dec_canonical h2⟩

theorem airBody_dec_canonical {lay : Rel4Layout} {l : Bytes} {b : AirBody} {rest : Bytes}
    (h : (Codec.airBody lay).dec l = .ok (b, rest)) :
    b.gkr.rootNum.Canonical ∧ b.gkr.rootDen.Canonical := by
  obtain ⟨t, ht, rfl⟩ := decMap_ok h
  obtain ⟨r1, _, h1⟩ := decPair_ok ht
  obtain ⟨r2, h2, _⟩ := decPair_ok h1
  exact airGkr_dec_canonical h2

/-- **The `GKR_ROOT` limbs of an accepted container are canonical.** -/
theorem parseContainerRel4_root_canonical {prof : Fpn2Profile} {raw : ByteArray}
    {c : Rel4Container} (hp : parseContainerRel4 prof raw = .ok c) :
    c.body.gkr.rootNum.Canonical ∧ c.body.gkr.rootDen.Canonical := by
  have hb := (parseContainerRel4_spec hp).2.2
  unfold parseBodyRel4 at hb
  split at hb
  · cases hb
  · split at hb
    · cases hb
    · split at hb
      · cases hb
      · split at hb
        · cases hb
        · split at hb
          · cases hb
          · rename_i b' rest hdec
            split at hb
            · injection hb with hb
              rw [← hb]
              exact airBody_dec_canonical hdec
            · cases hb

/-- Three canonical `Fp3` values (a degree-2 round `h(0), h(1), h(2)`). -/
def f3x3Canonical (v : F3x3) : Prop := v.1.Canonical ∧ v.2.1.Canonical ∧ v.2.2.Canonical

/-- Four canonical `Fp3` values (a degree-3 round `s(0..3)`, a coset leaf). -/
def f3x4Canonical (v : F3x4) : Prop := v.1.Canonical ∧ f3x3Canonical v.2

theorem fp3x3_dec_canonical {l : Bytes} {v : F3x3} {rest : Bytes}
    (h : Codec.fp3x3.dec l = .ok (v, rest)) : f3x3Canonical v := by
  obtain ⟨r1, h0, h12⟩ := decPair_ok (c₁ := Codec.fp3) (c₂ := Codec.fp3.pair Codec.fp3) h
  obtain ⟨r2, h1, h2⟩ := decPair_ok (c₁ := Codec.fp3) (c₂ := Codec.fp3) h12
  exact ⟨fp3_dec_canonical h0, fp3_dec_canonical h1, fp3_dec_canonical h2⟩

theorem fp3x4_dec_canonical {l : Bytes} {v : F3x4} {rest : Bytes}
    (h : Codec.fp3x4.dec l = .ok (v, rest)) : f3x4Canonical v := by
  obtain ⟨r1, h0, h1⟩ := decPair_ok (c₁ := Codec.fp3) (c₂ := Codec.fp3x3) h
  exact ⟨fp3_dec_canonical h0, fp3x3_dec_canonical h1⟩

/-- A decoded family has exactly the requested number of entries, and every entry was produced by
one of the entry decoders. -/
theorem decFam_forall {α : Type} {c : Nat → Codec α} {Pr : α → Prop}
    (hc : ∀ i l a rest, (c i).dec l = .ok (a, rest) → Pr a) (i n : Nat) (l : Bytes)
    (as : List α) (rest : Bytes) (h : decFam c i n l = .ok (as, rest)) :
    as.length = n ∧ ∀ a ∈ as, Pr a := by
  induction n generalizing i l as rest with
  | zero =>
    simp only [decFam] at h
    injection h with h
    injection h with ha hr
    subst ha
    exact ⟨rfl, fun a ha => absurd ha (by simp)⟩
  | succ n ih =>
    simp only [decFam] at h
    cases h1 : (c i).dec l with
    | error e => simp [h1] at h
    | ok x =>
      obtain ⟨a, r⟩ := x
      simp only [h1] at h
      cases h2 : decFam c (i + 1) n r with
      | error e => simp [h2] at h
      | ok y =>
        obtain ⟨as', r'⟩ := y
        simp only [h2] at h
        injection h with h
        injection h with ha hr
        subst ha
        obtain ⟨hlen, hall⟩ := ih (i + 1) r as' r' h2
        refine ⟨by rw [List.length_cons, hlen], fun b hb => ?_⟩
        rcases List.mem_cons.mp hb with rfl | hb
        · exact hc _ _ _ _ h1
        · exact hall b hb

/-- The decoded shape of one query (relations 4 and 5): canonical limbs in both layer-0 leaves and
in every committed coset, and exactly `T − 1` committed layers `1 .. T − 1`. -/
def AirQueryCanonical (lay : Rel4Layout) (q : AirQuery) : Prop :=
  f3x4Canonical q.fLeaf.values ∧ f3x4Canonical q.cLeaf.values ∧
    q.rest.length = lay.folds - 1 ∧ ∀ o ∈ q.rest, f3x4Canonical o.values

/-- The four values of a decoded coset opening are canonical, at every Merkle depth. -/
theorem coset_dec_canonical {d : Nat} {l : Bytes} {o : CosetOpening} {rest : Bytes}
    (h : (Codec.coset d).dec l = .ok (o, rest)) : f3x4Canonical o.values := by
  obtain ⟨t, ht, rfl⟩ := decMap_ok h
  obtain ⟨r, hv, _⟩ := decPair_ok ht
  exact fp3x4_dec_canonical hv

/-- A decoded query has canonical coset values and exactly `T − 1` committed layers. -/
theorem airQuery_dec_canonical {lay : Rel4Layout} {l : Bytes} {q : AirQuery} {rest : Bytes}
    (h : (Codec.airQuery lay).dec l = .ok (q, rest)) : AirQueryCanonical lay q := by
  obtain ⟨t, ht, rfl⟩ := decMap_ok h
  obtain ⟨r1, hf, h1⟩ := decPair_ok ht
  obtain ⟨r2, hc, hr⟩ := decPair_ok h1
  have hR := decFam_forall (c := fun t => Codec.coset (lay.depth t))
    (Pr := fun o : CosetOpening => f3x4Canonical o.values)
    (fun _ _ _ _ h => coset_dec_canonical h) _ _ _ _ _ hr
  exact ⟨coset_dec_canonical hf, coset_dec_canonical hc, hR.1, hR.2⟩

/-- A decoded `QUERIES` section has exactly `n` queries, each of the shape `AirQueryCanonical`. -/
theorem airQueries_dec_canonical {lay : Rel4Layout} {n : Nat} {l : Bytes} {qs : List AirQuery}
    {rest : Bytes} (h : decFam (fun _ => Codec.airQuery lay) 0 n l = .ok (qs, rest)) :
    qs.length = n ∧ ∀ q ∈ qs, AirQueryCanonical lay q :=
  decFam_forall (c := fun _ => Codec.airQuery lay) (Pr := AirQueryCanonical lay)
    (fun _ _ _ _ h => airQuery_dec_canonical h) _ _ _ _ _ h

/-- A selected slot of a canonical coset is canonical. -/
theorem sel4_canonical {v : F3x4} (h : f3x4Canonical v) (i : Nat) : (sel4 v i).Canonical := by
  unfold sel4
  split
  · exact h.1
  · exact h.2.1
  · exact h.2.2.1
  · exact h.2.2.2

theorem airBody_dec_queries {lay : Rel4Layout} {l : Bytes} {b : AirBody} {rest : Bytes}
    (h : (Codec.airBody lay).dec l = .ok (b, rest)) :
    b.queries.length = lay.nQueries ∧ ∀ q ∈ b.queries, AirQueryCanonical lay q := by
  obtain ⟨t, ht, rfl⟩ := decMap_ok h
  obtain ⟨r1, _, h1⟩ := decPair_ok ht
  obtain ⟨r2, _, h2⟩ := decPair_ok h1
  obtain ⟨r3, _, h3⟩ := decPair_ok h2
  obtain ⟨r4, _, h4⟩ := decPair_ok h3
  exact airQueries_dec_canonical h4

/-- **The `QUERIES` section of an accepted relation-4 container**: exactly `Q` queries, each with
canonical coset limbs and exactly `T − 1` committed layers, for the layout of the profile. -/
theorem parseContainerRel4_queries_canonical {prof : Fpn2Profile} {raw : ByteArray}
    {c : Rel4Container} {lay : Rel4Layout} (hlay : rel4Layout prof = some lay)
    (hp : parseContainerRel4 prof raw = .ok c) :
    c.body.queries.length = lay.nQueries ∧ ∀ q ∈ c.body.queries, AirQueryCanonical lay q := by
  have hb := (parseContainerRel4_spec hp).2.2
  unfold parseBodyRel4 at hb
  split at hb
  · cases hb
  · split at hb
    · cases hb
    · rename_i lay' hlay'
      rw [hlay] at hlay'
      injection hlay' with hlay'
      subst hlay'
      split at hb
      · cases hb
      · split at hb
        · cases hb
        · split at hb
          · cases hb
          · rename_i b' rest hdec
            split at hb
            · injection hb with hb
              rw [← hb]
              exact airBody_dec_queries hdec
            · cases hb

/-- On a canonical value, `≠ Fp3.zero` is `≠ 0` in `E`. -/
theorem Fp3.toField_ne_zero {a : Fp3} (ha : a.Canonical) (h : a ≠ Fp3.zero) : a.toField ≠ 0 :=
  fun h0 => h ((Fp3.toField_eq_zero_iff ha).mp h0)

/-! ### The coset-4 fold and the query phase in `E` -/

theorem fpToField_four : fpToField 4 = 4 := by
  have h : (4 : UInt64).toNat = 4 := by decide
  simp [fpToField, h]

theorem fpToField_seven : fpToField 7 = ((7 : ℕ) : GoldilocksField) := by
  have h : (7 : UInt64).toNat = 7 := by decide
  simp [fpToField, h]

/-- `7^((p−1)/2) = −1` in `F_p` (`7` is a quadratic non-residue; from the Lucas certificate). -/
theorem seven_pow_half : ((7 : ℕ) : GoldilocksField) ^ ((goldilocks - 1) / 2) = -1 := by
  obtain ⟨h1, h2, _⟩ := seven_pow_goldilocks
  have hlt1 : goldilocks - 1 < 2 ^ 64 := lt_of_le_of_lt (Nat.sub_le _ _) goldilocks_lt
  have hlt : (goldilocks - 1) / 2 < 2 ^ 64 := lt_of_le_of_lt (Nat.div_le_self _ _) hlt1
  obtain ⟨t, ht⟩ : ∃ t : GoldilocksField, t = ((7 : ℕ) : GoldilocksField) ^ ((goldilocks - 1) / 2) :=
    ⟨_, rfl⟩
  rw [← ht]
  have hfull : ((7 : ℕ) : ZMod goldilocks) ^ (goldilocks - 1) = ((1 : ℕ) : ZMod goldilocks) :=
    (zmod_pow_eq_iff goldilocks 7 (goldilocks - 1) 1 hlt1).mpr (by rw [h1, goldilocks_eq])
  have hdiv : (goldilocks - 1) / 2 * 2 = goldilocks - 1 :=
    Nat.div_mul_cancel (by rw [goldilocks_sub_one_factor]; norm_num)
  have hsq : t ^ 2 = 1 := by
    rw [ht, ← pow_mul, hdiv, hfull, Nat.cast_one]
  have hne : t ≠ 1 := by
    intro h
    apply h2
    have h' : ((7 : ℕ) : ZMod goldilocks) ^ ((goldilocks - 1) / 2) = ((1 : ℕ) : ZMod goldilocks) := by
      rw [← ht, h, Nat.cast_one]
    rw [(zmod_pow_eq_iff _ _ _ _ hlt).mp h']
    rw [goldilocks_eq]
  have hprod : (t - 1) * (t + 1) = 0 := by linear_combination hsq
  rcases mul_eq_zero.mp hprod with h | h
  · exact absurd (sub_eq_zero.mp h) hne
  · exact eq_neg_of_add_eq_zero_left h

/-- **`ι^(−1)` is a primitive fourth root of unity**: the constant `rel3IotaInv` of the bytes
verifier squares to `−1` in `F_p`. -/
theorem rel3IotaInv_sq {iotaInv : UInt64} (h : rel3IotaInv = some iotaInv) :
    fpToField iotaInv ^ 2 = -1 := by
  have hP : goldilocksP.toNat = goldilocks := goldilocksP_toNat_eq_goldilocks
  have hlt : (goldilocksP.toNat - 1) / 4 < 2 ^ 64 :=
    lt_of_le_of_lt (Nat.div_le_self _ _) (lt_of_le_of_lt (Nat.sub_le _ _) goldilocksP.toNat_lt)
  have hι : fpToField (fpPow 7 ((goldilocksP.toNat - 1) / 4)) ^ 2 = -1 := by
    rw [fpPow_toField _ _ hlt, fpToField_seven, hP, ← pow_mul,
      show (goldilocks - 1) / 4 * 2 = (goldilocks - 1) / 2 by rw [goldilocks_eq]]
    exact seven_pow_half
  have hinv := fpInv_toField (fpPow 7 ((goldilocksP.toNat - 1) / 4))
  unfold rel3IotaInv at h
  rw [h, Option.getD_some] at hinv
  rw [hinv, inv_pow, hι, inv_neg_one]

/-- **The coset-4 fold over `E`.** For the four values `v_k` on a coset `{x ι^k}` (`ι^4 = 1`),
`f_e = x^(−e)/4 · Σ_k ι^(−ek) v_k` are the coefficients of the cubic through them and the fold is
`f_0 + r_a f_1 + r_b f_2 + r_a r_b f_3`. `xi = x^(−1)` and `j = ι^(−1)` lie in `F_p`. -/
noncomputable def fold4Field (v0 v1 v2 v3 : Fp3Field) (xi j : GoldilocksField)
    (ra rb : Fp3Field) : Fp3Field :=
  (4 : Fp3Field)⁻¹ * (v0 + v1 + v2 + v3)
    + ra * ((4 : Fp3Field)⁻¹ * fpEmbed xi
        * (v0 + fpEmbed j * v1 + fpEmbed j ^ 2 * v2 + fpEmbed j ^ 3 * v3))
    + rb * ((4 : Fp3Field)⁻¹ * fpEmbed xi ^ 2
        * (v0 + fpEmbed j ^ 2 * v1 + fpEmbed j ^ 4 * v2 + fpEmbed j ^ 6 * v3))
    + ra * rb * ((4 : Fp3Field)⁻¹ * fpEmbed xi ^ 3
        * (v0 + fpEmbed j ^ 3 * v1 + fpEmbed j ^ 6 * v2 + fpEmbed j ^ 9 * v3))

/-- **The executable `fold4` is the coset-4 fold of `E`**, for any `ι^(−1)` with `ι^(−2) = −1`. -/
theorem fold4_toField (v : F3x4) (xInv iotaInv : UInt64) (ra rb : Fp3)
    (hι : fpToField iotaInv ^ 2 = -1) :
    (fold4 v xInv iotaInv ra rb).toField
      = fold4Field v.1.toField v.2.1.toField v.2.2.1.toField v.2.2.2.toField
          (fpToField xInv) (fpToField iotaInv) ra.toField rb.toField := by
  have hJ : fpEmbed (fpToField iotaInv) ^ 2 = -1 := by
    rw [← map_pow, hι, map_neg, map_one]
  have h4 : fpEmbed (fpToField ((fpInv 4).getD 0)) = (4 : Fp3Field)⁻¹ := by
    rw [fpInv_toField, fpToField_four, map_inv₀, map_ofNat]
  simp only [fold4, fold4Field, Fp3.toField_add, Fp3.toField_sub, Fp3.toField_mul,
    Fp3.toField_scale, fpToField_mul, map_mul, h4]
  linear_combination
    (-(ra.toField * (4 : Fp3Field)⁻¹ * fpEmbed (fpToField xInv)
          * (v.2.2.1.toField + fpEmbed (fpToField iotaInv) * v.2.2.2.toField)
        + rb.toField * (4 : Fp3Field)⁻¹ * fpEmbed (fpToField xInv) ^ 2
          * (v.2.1.toField + (fpEmbed (fpToField iotaInv) ^ 2 - 1) * v.2.2.1.toField
            + (fpEmbed (fpToField iotaInv) ^ 4 - fpEmbed (fpToField iotaInv) ^ 2 + 1)
              * v.2.2.2.toField)
        + ra.toField * rb.toField * (4 : Fp3Field)⁻¹ * fpEmbed (fpToField xInv) ^ 3
          * (fpEmbed (fpToField iotaInv) * v.2.1.toField
            + (fpEmbed (fpToField iotaInv) ^ 4 - fpEmbed (fpToField iotaInv) ^ 2 + 1)
              * v.2.2.1.toField
            + fpEmbed (fpToField iotaInv) * (fpEmbed (fpToField iotaInv) ^ 2 - 1)
              * (fpEmbed (fpToField iotaInv) ^ 4 + 1) * v.2.2.2.toField))) * hJ

/-! The four character sums of a coset `{x ι^k}` with `ι = −J`, `J^2 = −1`: for a cubic `P`,
`Σ_k J^(ek) P(x ι^k) = 4 c_e x^e`. The cofactors of `J^2 + 1` were computed by polynomial division. -/

theorem coset4_sum0 {R : Type*} [CommRing R] (J x c0 c1 c2 c3 : R) (hj : J ^ 2 = -1) :
    (c0 + c1 * x + c2 * x ^ 2 + c3 * x ^ 3)
        + (c0 + c1 * (x * -J) + c2 * (x * -J) ^ 2 + c3 * (x * -J) ^ 3)
        + (c0 + c1 * (x * (-J) ^ 2) + c2 * (x * (-J) ^ 2) ^ 2 + c3 * (x * (-J) ^ 2) ^ 3)
        + (c0 + c1 * (x * (-J) ^ 3) + c2 * (x * (-J) ^ 3) ^ 2 + c3 * (x * (-J) ^ 3) ^ 3)
      = 4 * c0 := by
  linear_combination
    (x * c1 + x ^ 2 * c2 + x ^ 3 * c3 - J * x * c1 - J ^ 2 * x ^ 3 * c3 - J ^ 3 * x ^ 3 * c3 +
      J ^ 4 * x ^ 2 * c2 + J ^ 4 * x ^ 3 * c3 + J ^ 5 * x ^ 3 * c3 - J ^ 7 * x ^ 3 * c3) * hj

theorem coset4_sum1 {R : Type*} [CommRing R] (J x c0 c1 c2 c3 : R) (hj : J ^ 2 = -1) :
    (c0 + c1 * x + c2 * x ^ 2 + c3 * x ^ 3)
        + J * (c0 + c1 * (x * -J) + c2 * (x * -J) ^ 2 + c3 * (x * -J) ^ 3)
        + J ^ 2 * (c0 + c1 * (x * (-J) ^ 2) + c2 * (x * (-J) ^ 2) ^ 2 + c3 * (x * (-J) ^ 2) ^ 3)
        + J ^ 3 * (c0 + c1 * (x * (-J) ^ 3) + c2 * (x * (-J) ^ 3) ^ 2 + c3 * (x * (-J) ^ 3) ^ 3)
      = 4 * c1 * x := by
  linear_combination
    (c0 - 3 * x * c1 + x ^ 2 * c2 + x ^ 3 * c3 + J * c0 + 2 * J ^ 2 * x * c1 - J ^ 2 * x ^ 2 *
      c2 - J ^ 2 * x ^ 3 * c3 + J ^ 3 * x ^ 2 * c2 - J ^ 4 * x * c1 + J ^ 4 * x ^ 2 * c2 - J ^ 5
      * x ^ 2 * c2 + J ^ 7 * x ^ 2 * c2 + J ^ 8 * x ^ 3 * c3 - J ^ 10 * x ^ 3 * c3) * hj

theorem coset4_sum2 {R : Type*} [CommRing R] (J x c0 c1 c2 c3 : R) (hj : J ^ 2 = -1) :
    (c0 + c1 * x + c2 * x ^ 2 + c3 * x ^ 3)
        + J ^ 2 * (c0 + c1 * (x * -J) + c2 * (x * -J) ^ 2 + c3 * (x * -J) ^ 3)
        + J ^ 4 * (c0 + c1 * (x * (-J) ^ 2) + c2 * (x * (-J) ^ 2) ^ 2 + c3 * (x * (-J) ^ 2) ^ 3)
        + J ^ 6 * (c0 + c1 * (x * (-J) ^ 3) + c2 * (x * (-J) ^ 3) ^ 2 + c3 * (x * (-J) ^ 3) ^ 3)
      = 4 * c2 * x ^ 2 := by
  linear_combination
    (c0 + x * c1 - 3 * x ^ 2 * c2 + x ^ 3 * c3 - J ^ 2 * x * c1 + 3 * J ^ 2 * x ^ 2 * c2 - J ^
      2 * x ^ 3 * c3 - J ^ 3 * x * c1 + J ^ 4 * c0 + J ^ 4 * x * c1 - 2 * J ^ 4 * x ^ 2 * c2 + J
      ^ 4 * x ^ 3 * c3 + J ^ 5 * x * c1 - J ^ 5 * x ^ 3 * c3 + 2 * J ^ 6 * x ^ 2 * c2 - J ^ 6 *
      x ^ 3 * c3 - J ^ 7 * x * c1 + J ^ 7 * x ^ 3 * c3 - J ^ 8 * x ^ 2 * c2 + J ^ 8 * x ^ 3 * c3
      - J ^ 9 * x ^ 3 * c3 + J ^ 10 * x ^ 2 * c2 + J ^ 11 * x ^ 3 * c3 - J ^ 13 * x ^ 3 * c3) * hj

theorem coset4_sum3 {R : Type*} [CommRing R] (J x c0 c1 c2 c3 : R) (hj : J ^ 2 = -1) :
    (c0 + c1 * x + c2 * x ^ 2 + c3 * x ^ 3)
        + J ^ 3 * (c0 + c1 * (x * -J) + c2 * (x * -J) ^ 2 + c3 * (x * -J) ^ 3)
        + J ^ 6 * (c0 + c1 * (x * (-J) ^ 2) + c2 * (x * (-J) ^ 2) ^ 2 + c3 * (x * (-J) ^ 2) ^ 3)
        + J ^ 9 * (c0 + c1 * (x * (-J) ^ 3) + c2 * (x * (-J) ^ 3) ^ 2 + c3 * (x * (-J) ^ 3) ^ 3)
      = 4 * c3 * x ^ 3 := by
  linear_combination
    (c0 + x * c1 + x ^ 2 * c2 - 3 * x ^ 3 * c3 - J ^ 2 * c0 - J ^ 2 * x * c1 - J ^ 2 * x ^ 2 *
      c2 + 3 * J ^ 2 * x ^ 3 * c3 + J ^ 3 * c0 + J ^ 4 * c0 + J ^ 4 * x ^ 2 * c2 - 3 * J ^ 4 * x
      ^ 3 * c3 - J ^ 5 * c0 + J ^ 5 * x ^ 2 * c2 - J ^ 6 * x ^ 2 * c2 + 2 * J ^ 6 * x ^ 3 * c3 +
      J ^ 7 * c0 - J ^ 7 * x ^ 2 * c2 + J ^ 8 * x * c1 + J ^ 8 * x ^ 2 * c2 - 2 * J ^ 8 * x ^ 3
      * c3 + J ^ 9 * x ^ 2 * c2 - J ^ 10 * x * c1 + 2 * J ^ 10 * x ^ 3 * c3 - J ^ 11 * x ^ 2 *
      c2 - J ^ 12 * x ^ 3 * c3 + J ^ 13 * x ^ 2 * c2 + J ^ 14 * x ^ 3 * c3 - J ^ 16 * x ^ 3 * c3) * hj

/-- **`fold4Field` folds the cubic through the coset.** If `v_k = P(x ι^k)` for
`P = c_0 + c_1 X + c_2 X^2 + c_3 X^3` over `E`, with `x · xi = 1`, `ι · j = 1` and `j^2 = −1`,
then the fold is `c_0 + r_a c_1 + r_b c_2 + r_a r_b c_3`: the coset-4 (two fused binary) folding
relation of Basefold / FRI. -/
theorem fold4Field_interpolates (c0 c1 c2 c3 x ι ra rb : Fp3Field) (xi j : GoldilocksField)
    (hx : x * fpEmbed xi = 1) (hι : ι * fpEmbed j = 1) (hj : fpEmbed j ^ 2 = -1)
    (h4 : (4 : Fp3Field) ≠ 0) :
    fold4Field (c0 + c1 * x + c2 * x ^ 2 + c3 * x ^ 3)
        (c0 + c1 * (x * ι) + c2 * (x * ι) ^ 2 + c3 * (x * ι) ^ 3)
        (c0 + c1 * (x * ι ^ 2) + c2 * (x * ι ^ 2) ^ 2 + c3 * (x * ι ^ 2) ^ 3)
        (c0 + c1 * (x * ι ^ 3) + c2 * (x * ι ^ 3) ^ 2 + c3 * (x * ι ^ 3) ^ 3) xi j ra rb
      = c0 + ra * c1 + rb * c2 + ra * rb * c3 := by
  have hιj : ι = -fpEmbed j := by
    have h : ι * (fpEmbed j * fpEmbed j) = fpEmbed j := by rw [← mul_assoc, hι, one_mul]
    rw [← pow_two, hj] at h
    linear_combination -h
  subst hιj
  have h4' : (4 : Fp3Field)⁻¹ * 4 = 1 := inv_mul_cancel₀ h4
  simp only [fold4Field]
  rw [coset4_sum0 _ _ _ _ _ _ hj, coset4_sum1 _ _ _ _ _ _ hj, coset4_sum2 _ _ _ _ _ _ hj,
    coset4_sum3 _ _ _ _ _ _ hj]
  linear_combination
    (c0 + ra * c1 * (x * fpEmbed xi) + rb * c2 * (x * fpEmbed xi) ^ 2
        + ra * rb * c3 * (x * fpEmbed xi) ^ 3) * h4'
      + (ra * c1 + rb * c2 * (x * fpEmbed xi + 1)
        + ra * rb * c3 * ((x * fpEmbed xi) ^ 2 + x * fpEmbed xi + 1)) * hx

/-! ### The evaluation domain of the folded opening: `foldXInv` inverts the domain point -/

theorem goldilocks_seven_ne_zero : ((7 : ℕ) : GoldilocksField) ≠ 0 := by
  rw [Ne, ZMod.natCast_eq_zero_iff, goldilocks_eq]; norm_num

theorem goldilocks_neg_one_ne_one : (-1 : GoldilocksField) ≠ 1 := by
  intro h
  have h2 : ((2 : ℕ) : GoldilocksField) = 0 := by
    push_cast; linear_combination -h
  rw [ZMod.natCast_eq_zero_iff, goldilocks_eq] at h2
  norm_num at h2

/-- `L = 8 · 2^n = 2^(n+3)`. -/
theorem Rel3Layout.codeLen_eq (lay : Rel3Layout) : lay.codeLen = 2 ^ (lay.nVars + 3) := by
  unfold Rel3Layout.codeLen; ring

/-- The codeword length divides `p − 1` (2-adicity 32) for every admissible `n ≤ 28`. -/
theorem Rel3Layout.codeLen_dvd (lay : Rel3Layout) (hn : lay.nVars ≤ 28) :
    lay.codeLen ∣ goldilocks - 1 := by
  rw [goldilocks_sub_one_factor, lay.codeLen_eq]
  have h : 2 ^ (lay.nVars + 3) ∣ 2 ^ 32 := pow_dvd_pow 2 (by omega)
  exact ((((h.mul_right 3).mul_right 5).mul_right 17).mul_right 257).mul_right 65537

/-- `ω = 7^((p−1)/L)` in `F_p`. -/
theorem rel3Omega_toField (lay : Rel3Layout) :
    fpToField (rel3Omega lay) = ((7 : ℕ) : GoldilocksField) ^ ((goldilocks - 1) / lay.codeLen) := by
  unfold rel3Omega
  rw [fpPow_toField _ _ (lt_of_le_of_lt (Nat.div_le_self _ _)
    (lt_of_le_of_lt (Nat.sub_le _ _) goldilocksP.toNat_lt)), fpToField_seven,
    goldilocksP_toNat_eq_goldilocks]

/-- `ω^(L/2) = −1`: `ω` has exact order `L`. -/
theorem rel3Omega_pow_half (lay : Rel3Layout) (hn : lay.nVars ≤ 28) :
    fpToField (rel3Omega lay) ^ (lay.codeLen / 2) = -1 := by
  have hdvd := lay.codeLen_dvd hn
  have heven : 2 ∣ lay.codeLen := by
    rw [lay.codeLen_eq, pow_succ]; exact dvd_mul_left 2 _
  have he : (goldilocks - 1) / lay.codeLen * (lay.codeLen / 2) = (goldilocks - 1) / 2 := by
    have h : (goldilocks - 1) / lay.codeLen * (lay.codeLen / 2) * 2 = goldilocks - 1 := by
      rw [mul_assoc, Nat.div_mul_cancel heven, Nat.div_mul_cancel hdvd]
    exact (Nat.div_eq_of_eq_mul_left (by norm_num) h.symm).symm
  rw [rel3Omega_toField, ← pow_mul, he]
  exact seven_pow_half

theorem rel3Omega_pow_codeLen (lay : Rel3Layout) (hn : lay.nVars ≤ 28) :
    fpToField (rel3Omega lay) ^ lay.codeLen = 1 := by
  have heven : 2 ∣ lay.codeLen := by
    rw [lay.codeLen_eq, pow_succ]; exact dvd_mul_left 2 _
  have h := rel3Omega_pow_half lay hn
  rw [← Nat.div_mul_cancel heven, pow_mul, h]; norm_num

/-- **`ω` is a primitive `L`-th root of unity** in `F_p`. -/
theorem rel3Omega_orderOf (lay : Rel3Layout) (hn : lay.nVars ≤ 28) :
    orderOf (fpToField (rel3Omega lay)) = lay.codeLen := by
  have h1 := rel3Omega_pow_half lay hn
  have h2 := rel3Omega_pow_codeLen lay hn
  rw [lay.codeLen_eq] at h1 h2 ⊢
  rw [show 2 ^ (lay.nVars + 3) / 2 = 2 ^ (lay.nVars + 2) by
    rw [pow_succ 2 (lay.nVars + 2)]; exact Nat.mul_div_cancel _ (by norm_num)] at h1
  exact orderOf_eq_prime_pow (by rw [h1]; exact goldilocks_neg_one_ne_one) h2

/-- The point `ω_t^pos` of the layer-`t` evaluation domain, `ω_t = ω^(4^t)`: the domain of the
Reed–Solomon word folded at layer `t` (`L_t = L / 4^t` points). -/
noncomputable def foldDomainPoint (lay : Rel3Layout) (t pos : ℕ) : GoldilocksField :=
  (fpToField (rel3Omega lay) ^ 4 ^ t) ^ pos

theorem foldDomainPoint_ne_zero (lay : Rel3Layout) (t pos : ℕ) : foldDomainPoint lay t pos ≠ 0 := by
  unfold foldDomainPoint
  rw [rel3Omega_toField]
  exact pow_ne_zero _ (pow_ne_zero _ (pow_ne_zero _ goldilocks_seven_ne_zero))

/-- **`foldXInv` is the inverse of the domain point.** The executable `x^(−1)` of
`Fpn2Rel3.foldXInv` times the domain point `ω_t^j` is `1` in `F_p`. -/
theorem foldXInv_mul_domainPoint (lay : Rel3Layout) (t j : ℕ) (ht : 2 * t < 64)
    (hj : j < goldilocks) :
    foldDomainPoint lay t j * fpToField (foldXInv lay t j) = 1 := by
  have hP : goldilocksP.toNat = goldilocks := goldilocksP_toNat_eq_goldilocks
  have hpos : 0 < goldilocks - 1 := by rw [goldilocks_eq]; norm_num
  have hlt1 : goldilocks - 1 < 2 ^ 64 := lt_of_le_of_lt (Nat.sub_le _ _) goldilocks_lt
  have hne := foldDomainPoint_ne_zero lay t 1
  unfold foldDomainPoint at hne ⊢
  rw [pow_one] at hne
  have hfermat : (fpToField (rel3Omega lay) ^ 4 ^ t) ^ (goldilocks - 1) = 1 :=
    ZMod.pow_card_sub_one_eq_one hne
  unfold foldXInv
  rw [hP, fpPow_toField _ _ (lt_of_lt_of_le (Nat.mod_lt _ hpos) hlt1.le),
    fpPow_toField _ _ (Nat.pow_lt_pow_right (by norm_num) ht),
    show (2 : ℕ) ^ (2 * t) = 4 ^ t by rw [pow_mul]; norm_num, ← pow_add]
  rcases Nat.eq_zero_or_pos j with h0 | hjpos
  · subst h0
    rw [Nat.sub_zero, Nat.mod_self, pow_zero]
  · rw [Nat.mod_eq_of_lt (by omega), show j + (goldilocks - 1 - j) = goldilocks - 1 by omega,
      hfermat]

theorem foldXInv_toField (lay : Rel3Layout) (t j : ℕ) (ht : 2 * t < 64) (hj : j < goldilocks) :
    fpToField (foldXInv lay t j) = (foldDomainPoint lay t j)⁻¹ :=
  eq_inv_of_mul_eq_one_right (foldXInv_mul_domainPoint lay t j ht hj)

/-- **`foldXInv` is the exact multiplicative inverse in `E = Fp3Field`** of the embedded domain
point `ω_t^j = ω^(4^t · j)`: the form in which `fold4Field` consumes it. -/
theorem foldXInv_mul_domainPoint_fp3 (lay : Rel3Layout) (t j : ℕ) (ht : 2 * t < 64)
    (hj : j < goldilocks) :
    fpEmbed (foldDomainPoint lay t j) * fpEmbed (fpToField (foldXInv lay t j)) = (1 : Fp3Field) := by
  rw [← map_mul, foldXInv_mul_domainPoint lay t j ht hj, map_one]

theorem foldXInv_toField_fp3 (lay : Rel3Layout) (t j : ℕ) (ht : 2 * t < 64) (hj : j < goldilocks) :
    fpEmbed (fpToField (foldXInv lay t j)) = (fpEmbed (foldDomainPoint lay t j) : Fp3Field)⁻¹ :=
  eq_inv_of_mul_eq_one_right (foldXInv_mul_domainPoint_fp3 lay t j ht hj)

/-- `4^t · (L_t / 4) · 4 = L` for every layer that exists (`2t + 2 ≤ n + 3`). -/
theorem Rel3Layout.quarterAt_mul (lay : Rel3Layout) (t : ℕ) (ht : 2 * t + 2 ≤ lay.nVars + 3) :
    4 ^ t * lay.quarterAt t * 4 = lay.codeLen := by
  unfold Rel3Layout.quarterAt
  rw [lay.codeLen_eq, Nat.pow_div ht (by norm_num), show (4 : ℕ) = 2 ^ 2 by norm_num, ← pow_mul,
    ← pow_add, ← pow_add]
  congr 1; omega

/-- **The coset generator.** `ω_t^(L_t/4) = ι = 7^((p−1)/4)` for every layer: the four values of
coset leaf `j` sit at `ω_t^j · ι^k`. -/
theorem foldDomainPoint_quarter (lay : Rel3Layout) (t : ℕ) (hn : lay.nVars ≤ 28)
    (ht : 2 * t + 2 ≤ lay.nVars + 3) :
    foldDomainPoint lay t (lay.quarterAt t) = ((7 : ℕ) : GoldilocksField) ^ ((goldilocks - 1) / 4) := by
  have hdvd := lay.codeLen_dvd hn
  have he : (goldilocks - 1) / lay.codeLen * 4 ^ t * lay.quarterAt t = (goldilocks - 1) / 4 := by
    have h : (goldilocks - 1) / lay.codeLen * 4 ^ t * lay.quarterAt t * 4 = goldilocks - 1 := by
      rw [mul_assoc, mul_assoc, ← mul_assoc (4 ^ t), lay.quarterAt_mul t ht,
        Nat.div_mul_cancel hdvd]
    exact (Nat.div_eq_of_eq_mul_left (by norm_num) h.symm).symm
  unfold foldDomainPoint
  rw [rel3Omega_toField, ← pow_mul, ← pow_mul, ← mul_assoc, he]

theorem foldDomainPoint_add_quarter (lay : Rel3Layout) (t j k : ℕ) (hn : lay.nVars ≤ 28)
    (ht : 2 * t + 2 ≤ lay.nVars + 3) :
    foldDomainPoint lay t (j + k * lay.quarterAt t)
      = foldDomainPoint lay t j * (((7 : ℕ) : GoldilocksField) ^ ((goldilocks - 1) / 4)) ^ k := by
  rw [← foldDomainPoint_quarter lay t hn ht]
  unfold foldDomainPoint
  rw [pow_add, mul_comm k, pow_mul]

/-- The fold maps layer `t` onto layer `t + 1`: `(ω_t^pos)^4 = ω_{t+1}^pos`. -/
theorem foldDomainPoint_succ (lay : Rel3Layout) (t pos : ℕ) :
    foldDomainPoint lay (t + 1) pos = foldDomainPoint lay t pos ^ 4 := by
  unfold foldDomainPoint
  rw [← pow_mul, ← pow_mul, ← pow_mul]
  congr 1; ring

/-- The layer-`t` domain has `L_t = L / 4^t` distinct points. -/
theorem foldDomainPoint_injOn (lay : Rel3Layout) (t : ℕ) (hn : lay.nVars ≤ 28)
    (ht : 2 * t + 2 ≤ lay.nVars + 3) :
    Set.InjOn (foldDomainPoint lay t) (Set.Iio (4 * lay.quarterAt t)) := by
  have ho : orderOf (fpToField (rel3Omega lay) ^ 4 ^ t) = 4 * lay.quarterAt t := by
    have hL := lay.quarterAt_mul t ht
    rw [orderOf_pow_of_dvd (pow_ne_zero _ (by norm_num))
      (by rw [rel3Omega_orderOf lay hn, ← hL, mul_assoc]; exact dvd_mul_right _ _),
      rel3Omega_orderOf lay hn, ← hL, mul_assoc, Nat.mul_div_cancel_left _ (by positivity),
      mul_comm]
  intro a ha b hb hab
  rw [← ho] at ha hb
  exact pow_injOn_Iio_orderOf ha hb hab

theorem rel3IotaInv_toField {iotaInv : UInt64} (h : rel3IotaInv = some iotaInv) :
    ((7 : ℕ) : GoldilocksField) ^ ((goldilocks - 1) / 4) * fpToField iotaInv = 1 := by
  have hlt : (goldilocksP.toNat - 1) / 4 < 2 ^ 64 :=
    lt_of_le_of_lt (Nat.div_le_self _ _) (lt_of_le_of_lt (Nat.sub_le _ _) goldilocksP.toNat_lt)
  have hinv := fpInv_toField (fpPow 7 ((goldilocksP.toNat - 1) / 4))
  unfold rel3IotaInv at h
  rw [h, Option.getD_some, fpPow_toField _ _ hlt, fpToField_seven,
    goldilocksP_toNat_eq_goldilocks] at hinv
  rw [hinv]
  exact mul_inv_cancel₀ (pow_ne_zero _ goldilocks_seven_ne_zero)

theorem fp3Field_four_ne_zero : (4 : Fp3Field) ≠ 0 := by
  have h : ((4 : ℕ) : Fp3Field) ≠ 0 := by
    rw [Ne, CharP.cast_eq_zero_iff Fp3Field goldilocks, goldilocks_eq]; norm_num
  exact_mod_cast h

/-- A cubic over `E` evaluated at the layer-`t` domain point `ω_t^pos`. -/
noncomputable def cubicAt (lay : Rel3Layout) (t : ℕ) (c0 c1 c2 c3 : Fp3Field) (pos : ℕ) : Fp3Field :=
  c0 + c1 * fpEmbed (foldDomainPoint lay t pos) + c2 * fpEmbed (foldDomainPoint lay t pos) ^ 2
    + c3 * fpEmbed (foldDomainPoint lay t pos) ^ 3

/-- **The executable fold on the actual evaluation domain.** If the four values of coset leaf `j`
of layer `t` are the evaluations of a cubic `c_0 + c_1 X + c_2 X^2 + c_3 X^3` at the domain points
`ω_t^(j + k L_t/4)`, `k < 4`, then `fold4Field` with the verifier's own `foldXInv lay t j` and
`rel3IotaInv` returns `c_0 + r_a c_1 + r_b c_2 + r_a r_b c_3`. No hypothesis on `x^(−1)` or `ι`
remains. -/
theorem fold4_foldXInv_interpolates (lay : Rel3Layout) (t j : ℕ) (hn : lay.nVars ≤ 28)
    (ht : 2 * t + 2 ≤ lay.nVars + 3) (hj : j < goldilocks) {iotaInv : UInt64}
    (hi : rel3IotaInv = some iotaInv) (c0 c1 c2 c3 ra rb : Fp3Field) :
    fold4Field (cubicAt lay t c0 c1 c2 c3 j) (cubicAt lay t c0 c1 c2 c3 (j + 1 * lay.quarterAt t))
        (cubicAt lay t c0 c1 c2 c3 (j + 2 * lay.quarterAt t))
        (cubicAt lay t c0 c1 c2 c3 (j + 3 * lay.quarterAt t))
        (fpToField (foldXInv lay t j)) (fpToField iotaInv) ra rb
      = c0 + ra * c1 + rb * c2 + ra * rb * c3 := by
  have hq := fun k => foldDomainPoint_add_quarter lay t j k hn ht
  have hι0 := rel3IotaInv_toField hi
  generalize ((7 : ℕ) : GoldilocksField) ^ ((goldilocks - 1) / 4) = ι at hq hι0
  have hx := foldXInv_mul_domainPoint_fp3 lay t j (by omega) hj
  have hι : fpEmbed ι * fpEmbed (fpToField iotaInv) = 1 := by rw [← map_mul, hι0, map_one]
  have hj2 : fpEmbed (fpToField iotaInv) ^ 2 = -1 := by
    rw [← map_pow, rel3IotaInv_sq hi, map_neg, map_one]
  have h := fold4Field_interpolates c0 c1 c2 c3 _ _ ra rb _ _ hx hι hj2 fp3Field_four_ne_zero
  unfold cubicAt
  rw [hq 1, hq 2, hq 3, pow_one]
  simp only [map_mul, map_pow]
  exact h

/-- A layer below the fold count exists: `t < T = n / 2` gives `2 t + 2 ≤ n`, hence the bound
`2 t + 2 ≤ n + 3` the domain lemmas (`foldDomainPoint_quarter`, `fold4_foldXInv_interpolates`) ask
for. No parity assumption on `n` is needed. -/
theorem Rel3Layout.layer_of_lt_folds (lay : Rel3Layout) {t : ℕ} (ht : t < lay.folds) :
    2 * t + 2 ≤ lay.nVars ∧ 2 * t + 2 ≤ lay.nVars + 3 := by
  unfold Rel3Layout.folds at ht
  omega

/-- A committed fold layer of one query in `E`: the layer exists (`t < T`), the Merkle path (a
hash statement, unchanged), the previous-slot consistency and the fold, as equations of `E`. The
layer bound is part of the relation and is proved for every step of an accepted run
(`foldRun_field`), from the number of committed layers the decoder returns. -/
def FoldField (O : Rel4Oracles) (lay : Rel3Layout) (roots : List Bytes) (rs : List Fp3)
    (iotaInv : UInt64) (s : FoldState) (o : CosetOpening) (s' : FoldState) : Prop :=
  s.t < lay.folds ∧
  cosetRoot O (merkleLeaf O (Codec.fp3x4.enc o.values)) (s.pos % lay.quarterAt s.t) o.siblings =
      roots.getD s.t [] ∧
    (s.t = 0 ∨ (sel4 o.values (s.pos / lay.quarterAt s.t)).toField = s.prev.toField) ∧
    s'.prev.toField =
      fold4Field o.values.1.toField o.values.2.1.toField o.values.2.2.1.toField
        o.values.2.2.2.toField (fpToField (foldXInv lay s.t (s.pos % lay.quarterAt s.t)))
        (fpToField iotaInv) (rs.getD (2 * s.t) Fp3.zero).toField
        (rs.getD (2 * s.t + 1) Fp3.zero).toField ∧
    s' = foldNext lay rs iotaInv s o

/-- One query in `E`: both layer-0 leaves anchored (hash statements), the first fold applied to
the combined coset `v^F + θ v^C`, every committed layer, and the last fold equal to `FINAL`. -/
def QueryStepField4 (O : Rel4Oracles) (lay : Rel4Layout) (roots : List Bytes) (rootCirc : Bytes)
    (rs : List Fp3) (theta final : Fp3) (iotaInv : UInt64) (tr : Fpn2Transcript) (q : AirQuery)
    (tr' : Fpn2Transcript) : Prop :=
  cosetRoot O (merkleLeaf O (Codec.fp3x4.enc q.fLeaf.values)) (airLeafIdx O lay tr)
      q.fLeaf.siblings = roots.getD 0 [] ∧
    cosetRoot O (merkleLeaf O (Codec.fp3x4.enc q.cLeaf.values)) (airLeafIdx O lay tr)
      q.cLeaf.siblings = rootCirc ∧
    (airLayer0 O lay rs theta iotaInv tr q).prev.toField =
      fold4Field
        (q.fLeaf.values.1.toField + theta.toField * q.cLeaf.values.1.toField)
        (q.fLeaf.values.2.1.toField + theta.toField * q.cLeaf.values.2.1.toField)
        (q.fLeaf.values.2.2.1.toField + theta.toField * q.cLeaf.values.2.2.1.toField)
        (q.fLeaf.values.2.2.2.toField + theta.toField * q.cLeaf.values.2.2.2.toField)
        (fpToField (foldXInv lay.pcs 0 (airLeafIdx O lay tr))) (fpToField iotaInv)
        (rs.getD 0 Fp3.zero).toField (rs.getD 1 Fp3.zero).toField ∧
    ∃ ff, RunList (FoldField O lay.pcs roots rs iotaInv)
        (airLayer0 O lay rs theta iotaInv tr q) q.rest ff ∧
      ff.prev.toField = final.toField ∧ tr' = (tr.squeezeBytes O.hFs 8).2

/-- **The query phase of relation 4 in `E`**: `ι^(−2) = −1` in `F_p`, and every query satisfies
`QueryStepField4` along the transcript, with `θ = r_0` and fold challenges `r_2..r_{n+1}`. -/
def QuerySpec4Field (O : Rel4Oracles) (lay : Rel4Layout) (c : Rel4Container) (pf : PcsState) :
    Prop :=
  ∃ iotaInv trF, rel3IotaInv = some iotaInv ∧ fpToField iotaInv ^ 2 = -1 ∧
    RunList
      (QueryStepField4 O lay (airRoots c.body) c.body.stmt.rootCirc (pf.rs.drop 2)
        (pf.rs.getD 0 Fp3.zero) c.body.pcs.final iotaInv)
      (queryStart4 O c pf) c.body.queries trF

theorem airLayer0_toField (O : Rel4Oracles) (lay : Rel4Layout) (rs : List Fp3) (theta : Fp3)
    (iotaInv : UInt64) (hι : fpToField iotaInv ^ 2 = -1) (tr : Fpn2Transcript) (q : AirQuery) :
    (airLayer0 O lay rs theta iotaInv tr q).prev.toField =
      fold4Field
        (q.fLeaf.values.1.toField + theta.toField * q.cLeaf.values.1.toField)
        (q.fLeaf.values.2.1.toField + theta.toField * q.cLeaf.values.2.1.toField)
        (q.fLeaf.values.2.2.1.toField + theta.toField * q.cLeaf.values.2.2.1.toField)
        (q.fLeaf.values.2.2.2.toField + theta.toField * q.cLeaf.values.2.2.2.toField)
        (fpToField (foldXInv lay.pcs 0 (airLeafIdx O lay tr))) (fpToField iotaInv)
        (rs.getD 0 Fp3.zero).toField (rs.getD 1 Fp3.zero).toField := by
  simp only [airLayer0, fold4_toField _ _ _ _ _ hι, combine4, Fp3.toField_add, Fp3.toField_mul]

theorem foldSpec_field (O : Rel4Oracles) (lay : Rel3Layout) (roots : List Bytes) (rs : List Fp3)
    (iotaInv : UInt64) (hι : fpToField iotaInv ^ 2 = -1) (s : FoldState) (o : CosetOpening)
    (s' : FoldState) (ht : s.t < lay.folds) (h : FoldSpec O lay roots rs iotaInv s o s') :
    FoldField O lay roots rs iotaInv s o s' := by
  obtain ⟨hm, hp, hs⟩ := h
  refine ⟨ht, hm, hp.imp id (congrArg Fp3.toField), ?_, hs⟩
  rw [hs]
  exact fold4_toField _ _ _ _ _ hι

/-- **Every fold step of a run is at a layer that exists.** A run of the executable fold steps
from state `s` over `os` openings visits the layers `s.t, s.t + 1, …` (`foldNext` increments `t`),
so if at most `T − s.t` openings follow, every step has `t < T` and the run holds in the field
form `FoldField`, layer bound included. -/
theorem foldRun_field {O : Rel4Oracles} {lay : Rel3Layout} {roots : List Bytes} {rs : List Fp3}
    {iotaInv : UInt64} (hι : fpToField iotaInv ^ 2 = -1) {s : FoldState} {os : List CosetOpening}
    {ff : FoldState} (hr : RunList (FoldSpec O lay roots rs iotaInv) s os ff) :
    os.length ≤ lay.folds - s.t → RunList (FoldField O lay roots rs iotaInv) s os ff := by
  induction hr with
  | nil s => intro _; exact .nil s
  | cons s a s' as sF h1 _ ih =>
    intro hlen
    have ht' : s'.t = s.t + 1 := by rw [h1.2.2]; rfl
    simp only [List.length_cons] at hlen
    exact .cons s a s' as sF (foldSpec_field O lay roots rs iotaInv hι s a s' (by omega) h1)
      (ih (by omega))

/-- The layer of a field fold step exists: `t < T`, `2 t + 2 ≤ n`. -/
theorem foldField_layer_bound {O : Rel4Oracles} {lay : Rel3Layout} {roots : List Bytes}
    {rs : List Fp3} {iotaInv : UInt64} {s : FoldState} {o : CosetOpening} {s' : FoldState}
    (h : FoldField O lay roots rs iotaInv s o s') : s.t < lay.folds ∧ 2 * s.t + 2 ≤ lay.nVars :=
  ⟨h.1, (lay.layer_of_lt_folds h.1).1⟩

/-- One typed query step (relation 4 or 5) implies its field form. `hshape` is the number of
committed layers the decoder returns (`AirQueryCanonical`): the run starts at layer 1 with `T − 1`
openings, so every fold step is at a layer `t < T`. -/
theorem queryStepSpec4_field {O : Rel4Oracles} {lay : Rel4Layout} {roots : List Bytes}
    {rootCirc : Bytes} {rs : List Fp3} {theta final : Fp3} {iotaInv : UInt64}
    (hι : fpToField iotaInv ^ 2 = -1) {tr : Fpn2Transcript} {q : AirQuery} {tr' : Fpn2Transcript}
    (hshape : q.rest.length = lay.folds - 1)
    (hq : QueryStepSpec4 O lay roots rootCirc rs theta final iotaInv tr q tr') :
    QueryStepField4 O lay roots rootCirc rs theta final iotaInv tr q tr' := by
  obtain ⟨hF, hC, ff, hrun, hfin, htr⟩ := hq
  have hlen : q.rest.length ≤ lay.pcs.folds - (airLayer0 O lay rs theta iotaInv tr q).t := by
    show q.rest.length ≤ lay.folds - 1
    exact le_of_eq hshape
  exact ⟨hF, hC, airLayer0_toField O lay _ _ iotaInv hι tr q, ff,
    foldRun_field hι hrun hlen, congrArg Fp3.toField hfin, htr⟩

/-- **Typed query acceptance implies the field query equations.** -/
theorem querySpec4_field {O : Rel4Oracles} {lay : Rel4Layout} {c : Rel4Container} {pf : PcsState}
    (hshape : ∀ q ∈ c.body.queries, q.rest.length = lay.folds - 1)
    (h : QuerySpec4 O lay c pf) : QuerySpec4Field O lay c pf := by
  obtain ⟨iotaInv, trF, hi, hr⟩ := h
  have hι := rel3IotaInv_sq hi
  exact ⟨iotaInv, trF, hi, hι,
    hr.imp_mem fun _ q _ hq h => queryStepSpec4_field hι (hshape q hq) h⟩

/-- **The acceptance equations of relation 4 as equations of the field `E`.** -/
structure Rel4FieldEquations (O : Rel4Oracles) (pub : PublicInput) (lay : Rel4Layout)
    (c : Rel4Container) (g : GkrState) (zf : RoundState) (pf : PcsState) : Prop where
  /-- The fractional sum is zero: `root_num = 0` in `E`. -/
  root_num : c.body.gkr.rootNum.toField = 0
  /-- The fraction is well defined: `root_den ≠ 0` in `E` (uses the canonical limbs of the parse). -/
  root_den : c.body.gkr.rootDen.toField ≠ 0
  /-- Layer 0: `num + α den = eq(pt, []) · children`, then every masked layer. -/
  gkr : ∃ s0,
    (gkrInit4 O pub c).num.toField
        + (layer0Alpha O (gkrInit4 O pub c)).toField * (gkrInit4 O pub c).den.toField
      = eqField (gkrInit4 O pub c).point []
        * childrenField c.body.gkr.layer0 (layer0Alpha O (gkrInit4 O pub c)) ∧
    s0 = layer0Next O (gkrInit4 O pub c) c.body.gkr.layer0 ∧
    RunList (GkrLayerField O) s0 c.body.gkr.layers g
  /-- The zerocheck starts from `λ_Z · Z_sum`. -/
  zc_init : (zcInit O lay c g).target.toField
    = (zcLamTr O lay c g).1.toField * c.body.zc.zSum.toField
  /-- Every zerocheck round sum. -/
  zc_rounds : RunList (ZcRoundField O) (zcInit O lay c g) c.body.zc.rounds zf
  /-- `target = eq(z, r) · Gate(finals) + λ_Z · g_r`. -/
  zc_final : zf.target.toField
    = eqField (zcPointTr O lay g).1 zf.rs * gateField c.body.zc.finals
      + (zcLamTr O lay c g).1.toField * c.body.zc.finals.gR.toField
  /-- Every Basefold round sum. -/
  pcs_rounds : RunList (PcsOpField O) (pcsInit4 O pub lay c g zf) (pcsOps4 c.body.pcs) pf
  /-- `y = FINAL · W̃(θ, r_1, r')`. -/
  pcs_final : pf.y.toField
    = c.body.pcs.final.toField * (finalWeight4 lay (pcsTerms4 O pub lay c g zf) pf.rs).toField
  /-- Every queried coset opening and fold-chain equation (`QuerySpec4`), in `E`. -/
  query : QuerySpec4Field O lay c pf

theorem gkrLayerSpec_field (O : Rel4Oracles) (s : GkrState) (L : GkrLayer) (s' : GkrState)
    (h : GkrLayerSpec O s L s') : GkrLayerField O s L s' := by
  obtain ⟨rf, hr, ht, hs⟩ := h
  refine ⟨rf, hr.imp fun s g s' h => ⟨by rw [← Fp3.toField_add, h.1], h.2⟩, ?_, ?_, hs⟩
  · simp only [layerRoundsInit, Fp3.toField_add, Fp3.toField_mul]
  · rw [ht, layerExpect, Fp3.toField_add, Fp3.toField_mul, Fp3.toField_mul, eqEval_toField,
      childrenClaim_toField]

/-- **Typed acceptance implies the field equations**, including `root_den ≠ 0` and the whole
query phase with its layer bounds. `hcan` and `hqs` are the facts the parser provides
(`parseContainerRel4_root_canonical`, `parseContainerRel4_queries_canonical`): without `hcan`,
`root_den ≠ Fp3.zero` does not give `≠ 0` in `E`; without the number of committed layers per query
in `hqs`, a fold step is not known to be at a layer `t < T`. -/
theorem rel4TypedVerify_field {O : Rel4Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {c : Rel4Container} (hcan : c.body.gkr.rootDen.Canonical)
    (hqs : ∀ lay, rel4Layout prof = some lay → ∀ q ∈ c.body.queries, AirQueryCanonical lay q)
    (h : Rel4TypedVerify O pub prof c) :
    ∃ lay g zf pf, rel4Layout prof = some lay ∧ Rel4FieldEquations O pub lay c g zf pf := by
  obtain ⟨lay, g, zf, pf, hlay, ⟨⟨hnum, hden⟩, s0, ⟨h0, hs0⟩, hlayers⟩, ⟨hzr, hzf⟩, ⟨hpr, hpf⟩,
    hq⟩ := h
  refine ⟨lay, g, zf, pf, hlay, ?_, Fp3.toField_ne_zero hcan hden, ?_, ?_, ?_, ?_, ?_, ?_,
    querySpec4_field (fun q hm => (hqs lay hlay q hm).2.2.1) hq⟩
  · rw [hnum, Fp3.toField_zero]
  · refine ⟨s0, ?_, hs0, hlayers.imp (gkrLayerSpec_field O)⟩
    have := congrArg Fp3.toField h0
    rwa [Fp3.toField_add, Fp3.toField_mul, Fp3.toField_mul, eqEval_toField,
      childrenClaim_toField] at this
  · simp only [zcInit, Fp3.toField_mul]
  · exact hzr.imp fun s sv s' h => ⟨by rw [← Fp3.toField_add, h.1], h.2⟩
  · rw [hzf, zcExpect, Fp3.toField_add, Fp3.toField_mul, Fp3.toField_mul, eqEval_toField,
      gateEval_toField]
  · refine hpr.imp fun s op s' h => ?_
    cases op with
    | round hh => exact ⟨by rw [← Fp3.toField_add, h.1], h.2⟩
    | root r => exact h
  · rw [hpf, Fp3.toField_mul]

/-- **Bytes to field.** If `verifyBytesRel4` accepts `raw` against a pinned profile and a pinned
circuit root, then `raw` is the canonical encoding of a container `c` whose `root_circ` is the pin,
the layout has `12 ≤ n ≤ 28`, and the relation-4 acceptance equations hold in the field
`E = F_p[X]/(X^3 − 2)`: `root_num = 0`, `root_den ≠ 0`, every GKR / zerocheck / Basefold equation,
and every query's coset openings and `fold4` chain (`QuerySpec4Field`), each fold step at a layer
`t < T` (`FoldField`). No hypothesis beyond acceptance. -/
theorem verifyBytesRel4_accept_field {O : Rel4Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {circuitRoot : Option Bytes} {raw : ByteArray} {claim : Rel4Claim}
    (h : verifyBytesRel4 O pub prof allowReduced circuitRoot raw = .ok claim) :
    ∃ (c : Rel4Container) (lay : Rel4Layout) (g : GkrState) (zf : RoundState) (pf : PcsState),
      parseContainerRel4 prof raw = .ok c ∧ toBytes raw = encodeContainerRel4 prof c ∧
      circuitRoot = some c.body.stmt.rootCirc ∧ claim = c.acceptedClaim ∧
      rel4Layout prof = some lay ∧ (12 ≤ lay.nVars ∧ lay.nVars ≤ 28) ∧
      Rel4FieldEquations O pub lay c g zf pf := by
  obtain ⟨c, ⟨hparse, _, _, henc, _⟩, _, hpin, ht, hc⟩ := verifyBytesRel4_refines_typed h
  obtain ⟨lay, g, zf, pf, hlay, hF⟩ :=
    rel4TypedVerify_field (parseContainerRel4_root_canonical hparse).2
      (fun _ hl => (parseContainerRel4_queries_canonical hl hparse).2) ht
  exact ⟨c, lay, g, zf, pf, hparse, henc, hpin, hc, hlay, rel4Layout_nVars_le hlay, hF⟩

/-! ## 9. Relation 5 (`FOLDED_VEC_MATMUL`): canonical limbs, field transport, budget, lifetime -/

/-! ### 9.1 Canonical limbs of a parsed relation-5 container -/

/-- **Canonical limbs of every decoded `Fp3` of a relation-5 container** (`IP`, `PCS` and the
coset values of `QUERIES`), with the list lengths the layout fixes: `d` inner-product rounds,
`T − 1` fold roots, `Q` queries, `T − 1` committed layers per query. `STMT` holds no field element. -/
structure Rel5Canonical (lay : Rel5Layout) (c : Rel5Container) : Prop where
  gSum : c.body.ip.gSum.Canonical
  ip_len : c.body.ip.rounds.length = lay.d
  ip_rounds : ∀ sv ∈ c.body.ip.rounds, f3x4Canonical sv
  eW : c.body.ip.eW.Canonical
  zBlind : c.body.pcs.zBlind.Canonical
  batch : f3x3Canonical c.body.pcs.batch.1 ∧ f3x3Canonical c.body.pcs.batch.2
  head_len : c.body.pcs.head.length = lay.folds - 1
  head : ∀ f ∈ c.body.pcs.head, f3x3Canonical f.1 ∧ f3x3Canonical f.2.1
  last : f3x3Canonical c.body.pcs.last.1 ∧ f3x3Canonical c.body.pcs.last.2
  final : c.body.pcs.final.Canonical
  queries_len : c.body.queries.length = lay.nQueries
  /-- Every query: canonical `F` leaf, weight leaf and committed cosets, `T − 1` committed layers. -/
  queries : ∀ q ∈ c.body.queries, AirQueryCanonical lay.air q

theorem vecBody_dec_canonical {lay : Rel5Layout} {l : Bytes} {b : VecBody} {rest : Bytes}
    {h : Fpn2Header} (hd : (Codec.vecBody lay).dec l = .ok (b, rest)) :
    Rel5Canonical lay ⟨h, b⟩ := by
  obtain ⟨t, ht, rfl⟩ := decMap_ok hd
  obtain ⟨r1, _, h1⟩ := decPair_ok ht
  obtain ⟨r2, hip, h2⟩ := decPair_ok h1
  obtain ⟨r3, hpcs, hqs⟩ := decPair_ok h2
  have hQ := airQueries_dec_canonical (lay := lay.air) hqs
  obtain ⟨ti, hti, hI⟩ := decMap_ok hip
  obtain ⟨ri1, hg, hi1⟩ := decPair_ok hti
  obtain ⟨ri2, hrounds, hew⟩ := decPair_ok hi1
  obtain ⟨tp, htp, hP⟩ := decMap_ok hpcs
  obtain ⟨rp1, hzb, hp1⟩ := decPair_ok htp
  obtain ⟨rp2, hbatch, hp2⟩ := decPair_ok hp1
  obtain ⟨rp3, hhead, hp3⟩ := decPair_ok hp2
  obtain ⟨rp4, hlast, hfinal⟩ := decPair_ok hp3
  obtain ⟨rb, hb1, hb2⟩ := decPair_ok hbatch
  obtain ⟨rl, hl1, hl2⟩ := decPair_ok hlast
  have hR := decFam_forall (Pr := f3x4Canonical) (fun _ _ _ _ h => fp3x4_dec_canonical h) _ _ _ _ _
    hrounds
  have hH := decFam_forall (c := fun _ => Codec.pcsFold)
    (Pr := fun f : F3x3 × F3x3 × Bytes => f3x3Canonical f.1 ∧ f3x3Canonical f.2.1)
    (fun _ _ _ _ h => by
      obtain ⟨q1, k1, k2⟩ := decPair_ok h
      obtain ⟨q2, k3, _⟩ := decPair_ok k2
      exact ⟨fp3x3_dec_canonical k1, fp3x3_dec_canonical k3⟩) _ _ _ _ _ hhead
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hQ.1, hQ.2⟩
  · show t.2.1.gSum.Canonical
    rw [hI]; exact fp3_dec_canonical hg
  · show t.2.1.rounds.length = lay.d
    rw [hI]; exact hR.1
  · show ∀ sv ∈ t.2.1.rounds, f3x4Canonical sv
    rw [hI]; exact hR.2
  · show t.2.1.eW.Canonical
    rw [hI]; exact fp3_dec_canonical hew
  · show t.2.2.1.zBlind.Canonical
    rw [hP]; exact fp3_dec_canonical hzb
  · show f3x3Canonical t.2.2.1.batch.1 ∧ f3x3Canonical t.2.2.1.batch.2
    rw [hP]; exact ⟨fp3x3_dec_canonical hb1, fp3x3_dec_canonical hb2⟩
  · show t.2.2.1.head.length = lay.folds - 1
    rw [hP]; exact hH.1
  · show ∀ f ∈ t.2.2.1.head, f3x3Canonical f.1 ∧ f3x3Canonical f.2.1
    rw [hP]; exact hH.2
  · show f3x3Canonical t.2.2.1.last.1 ∧ f3x3Canonical t.2.2.1.last.2
    rw [hP]; exact ⟨fp3x3_dec_canonical hl1, fp3x3_dec_canonical hl2⟩
  · show t.2.2.1.final.Canonical
    rw [hP]; exact fp3_dec_canonical hfinal

/-- **Every decoded `Fp3` of an accepted relation-5 container is canonical**, the coset values of
`QUERIES` included, and it carries exactly `d` inner-product rounds, `T − 1` fold roots and `Q`
queries of `T − 1` committed layers each, for the layout of the profile. -/
theorem parseContainerRel5_canonical {prof : Fpn2Profile} {raw : ByteArray} {c : Rel5Container}
    {lay : Rel5Layout} (hlay : rel5Layout prof = some lay)
    (hp : parseContainerRel5 prof raw = .ok c) : Rel5Canonical lay c := by
  have hb := (parseContainerRel5_spec hp).2.2
  unfold parseBodyRel5 at hb
  split at hb
  · cases hb
  · split at hb
    · cases hb
    · rename_i lay' hlay'
      rw [hlay] at hlay'
      injection hlay' with hlay'
      subst hlay'
      split at hb
      · cases hb
      · split at hb
        · cases hb
        · split at hb
          · cases hb
          · rename_i b' rest hdec
            split at hb
            · injection hb with hb
              have := vecBody_dec_canonical (h := c.header) hdec
              rw [hb] at this
              exact this
            · cases hb

/-! ### 9.2 What an accepting `verifyBytesRel5` run says in `E` -/

/-- An inner-product sumcheck round in `E`: `s(0) + s(1) = target`. -/
def IpRoundField (O : Rel5Oracles) (s : RoundState) (sv : F3x4) (s' : RoundState) : Prop :=
  sv.1.toField + sv.2.1.toField = s.target.toField ∧ s' = gkrRoundNext O s sv

/-- Inner-product round (degree 3) in the shape of `SumcheckSound.Accepts`: the cubic `p` through
the four wire values has `p(0) + p(1) = target`, and the next target is `p(challenge)`, in `E`. -/
theorem ipRound_accepts_shape (O : Rel5Oracles) (s : RoundState) (sv : F3x4) (s' : RoundState)
    (h : IpRoundSpec O s sv s') :
    (roundPoly (list4 sv)).natDegree ≤ 3 ∧
      (roundPoly (list4 sv)).eval 0 + (roundPoly (list4 sv)).eval 1 = s.target.toField ∧
      s'.target.toField = (roundPoly (list4 sv)).eval (gkrRoundChal O s sv).toField :=
  gkrRound_accepts_shape O s sv s' h

/-- `Y = target + μ e_W + μ² z_blind` in `E`. -/
theorem batchedClaim5_toField (target mu eW zBlind : Fp3) :
    (batchedClaim5 target mu eW zBlind).toField =
      target.toField + mu.toField * eW.toField + mu.toField ^ 2 * zBlind.toField := by
  simp only [batchedClaim5, Fp3.toField_add, Fp3.toField_mul]; ring

/-- `W̃(θ, r_1, r') = (1 − r_1)((1 − θ) ã(r') + (2θ − 1) c̃(r'))` in `E`, with `ã(r')`, `c̃(r')`
the executable `weightsEval` of the `F` and `C` terms. -/
theorem finalWeight5_toField (lay : Rel5Layout) (tm : Rel5Terms) (rs : List Fp3) :
    (finalWeight5 lay tm rs).toField =
      (1 - (rs.getD 1 Fp3.zero).toField) *
        ((1 - (rs.getD 0 Fp3.zero).toField) *
            (weightsEval lay.air.pcs ⟨Fp3.zero, Fp3.zero, tm.fBlocks, tm.fSparse, Fp3.zero⟩
              (rs.drop 2)).toField
          + (2 * (rs.getD 0 Fp3.zero).toField - 1) *
            (weightsEval lay.air.pcs ⟨Fp3.zero, Fp3.zero, tm.cBlocks, [], Fp3.zero⟩
              (rs.drop 2)).toField) := by
  simp only [finalWeight5, Fp3.toField_add, Fp3.toField_sub, Fp3.toField_mul, Fp3.toField_one]
  ring

/-- **The query phase of relation 5 in `E`**: `ι^(−2) = −1` in `F_p`, and every query satisfies
`QueryStepField4` at the relation-4 geometry `lay.air`, with the `F` leaf under `root_0`, the
weight leaf under the pinned `STMT.weight_root`, `θ = r_0` and fold challenges `r_2..r_{n+1}`. -/
def QuerySpec5Field (O : Rel5Oracles) (lay : Rel5Layout) (c : Rel5Container) (pf : PcsState) :
    Prop :=
  ∃ iotaInv trF, rel3IotaInv = some iotaInv ∧ fpToField iotaInv ^ 2 = -1 ∧
    RunList
      (QueryStepField4 O lay.air (vecRoots c.body) c.body.stmt.weightRoot (pf.rs.drop 2)
        (pf.rs.getD 0 Fp3.zero) c.body.pcs.final iotaInv)
      (queryStart5 O c pf) c.body.queries trF

theorem querySpec5_field {O : Rel5Oracles} {lay : Rel5Layout} {c : Rel5Container} {pf : PcsState}
    (hshape : ∀ q ∈ c.body.queries, q.rest.length = lay.air.folds - 1)
    (h : QuerySpec5 O lay c pf) : QuerySpec5Field O lay c pf := by
  obtain ⟨iotaInv, trF, hi, hr⟩ := h
  have hι := rel3IotaInv_sq hi
  exact ⟨iotaInv, trF, hi, hι,
    hr.imp_mem fun _ q _ hq h => queryStepSpec4_field hι (hshape q hq) h⟩

/-- **The acceptance equations of relation 5 as equations of the field `E`.** -/
structure Rel5FieldEquations (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout)
    (c : Rel5Container) (sf : RoundState) (pf : PcsState) : Prop where
  /-- The inner-product sumcheck starts from `λ · G_sum`. -/
  ip_init : (ipInit O pub lay c).target.toField
    = (ipLamTr O pub lay c).1.toField * c.body.ip.gSum.toField
  /-- Every inner-product round sum (degree 3). -/
  ip_rounds : RunList (IpRoundField O) (ipInit O pub lay c) c.body.ip.rounds sf
  /-- The Basefold sumcheck starts from `Y = target + μ e_W + μ² z_blind`. -/
  pcs_init : (pcsInit5 O c sf).y.toField
    = sf.target.toField + (pcsMuTr5 O c sf).1.toField * c.body.ip.eW.toField
      + (pcsMuTr5 O c sf).1.toField ^ 2 * c.body.pcs.zBlind.toField
  /-- Every Basefold round sum (degree 2). -/
  pcs_rounds : RunList (PcsOpField O) (pcsInit5 O c sf) (pcsOps4 c.body.pcs) pf
  /-- `y = FINAL · W̃(θ, r_1, r')` (expanded by `finalWeight5_toField`). -/
  pcs_final : pf.y.toField
    = c.body.pcs.final.toField * (finalWeight5 lay (pcsTerms5 O pub lay c sf) pf.rs).toField
  /-- Every queried coset opening and fold-chain equation (`QuerySpec5`), in `E`; every fold step
  is at a layer `t < T` (`FoldField`). -/
  query : QuerySpec5Field O lay c pf
  /-- The coset values the query equations are about are canonical (`toField` is injective on
  them), and every query has exactly `T − 1` committed layers. -/
  query_canonical : ∀ q ∈ c.body.queries, AirQueryCanonical lay.air q

/-- **Typed acceptance of relation 5 implies the field equations**, on the layout of the profile.
`hcan` is what the parser provides (`parseContainerRel5_canonical`); its `queries` field gives the
number of committed layers per query, hence the layer bound of every fold step. -/
theorem rel5TypedVerify_field {O : Rel5Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {c : Rel5Container} {lay : Rel5Layout} (hlay : rel5Layout prof = some lay)
    (hcan : Rel5Canonical lay c) (h : Rel5TypedVerify O pub prof c) :
    ∃ sf pf, Rel5FieldEquations O pub lay c sf pf := by
  obtain ⟨lay', sf, pf, hlay', hip, ⟨hpr, hpf⟩, hq⟩ := h
  rw [hlay] at hlay'
  injection hlay' with hlay'
  subst hlay'
  refine ⟨sf, pf, ?_, ?_, ?_, ?_, ?_,
    querySpec5_field (fun q hm => (hcan.queries q hm).2.2.1) hq, hcan.queries⟩
  · simp only [ipInit, Fp3.toField_mul]
  · exact hip.imp fun s sv s' h => ⟨by rw [← Fp3.toField_add, h.1], h.2⟩
  · simp only [pcsInit5, batchedClaim5_toField]
  · refine hpr.imp fun s op s' h => ?_
    cases op with
    | round hh => exact ⟨by rw [← Fp3.toField_add, h.1], h.2⟩
    | root r => exact h
  · rw [hpf, Fp3.toField_mul]

/-- The layouts the bytes verifier admits for relation 5: `12 ≤ n ≤ 28`, `n` even and
`1 + 5 + k ≤ n` (so `beta = n − 6 − k` is not truncated and `d = n − 6`). -/
theorem rel5Layout_admissible {p : Fpn2Profile} {lay : Rel5Layout} (h : rel5Layout p = some lay) :
    12 ≤ lay.nVars ∧ lay.nVars % 2 = 0 ∧ lay.nVars ≤ 28 ∧
      1 + Rel5Layout.rowBits + lay.k ≤ lay.nVars := by
  unfold rel5Layout at h
  split at h
  · unfold Rel5Layout.ofProfile at h
    split at h
    · rename_i hc
      split at h
      · cases h
      · split at h
        · rename_i hfit
          injection h with h
          subst h
          exact ⟨hc.2.1, hc.2.2.1, hc.2.2.2.1, hfit⟩
        · cases h
    · cases h
  · cases h

/-- **Bytes to field (relation 5).** If `verifyBytesRel5` accepts `raw` against a pinned profile
and a pinned weight commitment, then `raw` is the canonical encoding of a container `c` whose
`STMT.weight_root` is the pin, the layout is admissible, every decoded limb (`IP`, `PCS`, the
`QUERIES` coset values) is canonical with exactly `d` inner-product rounds and `T − 1` committed
layers per query, and the relation-5 acceptance equations hold in the field
`E = F_p[X]/(X^3 − 2)`: every degree-3 inner-product round from `λ · G_sum`, every degree-2
Basefold round from `Y = target + μ e_W + μ² z_blind`, the final evaluation
`y = FINAL · W̃(θ, r_1, r')`, and every query's dual layer-0 leaves and `fold4` chain, each fold
step at a layer `t < T`. No hypothesis beyond acceptance. -/
theorem verifyBytesRel5_accept_field {O : Rel5Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {weightPin : Option Bytes} {raw : ByteArray} {claim : Rel5Claim}
    (h : verifyBytesRel5 O pub prof allowReduced weightPin raw = .ok claim) :
    ∃ (c : Rel5Container) (lay : Rel5Layout) (sf : RoundState) (pf : PcsState),
      parseContainerRel5 prof raw = .ok c ∧ toBytes raw = encodeContainerRel5 prof c ∧
      weightPin = some c.body.stmt.weightRoot ∧ claim = c.acceptedClaim ∧
      rel5Layout prof = some lay ∧
      (12 ≤ lay.nVars ∧ lay.nVars % 2 = 0 ∧ lay.nVars ≤ 28 ∧
        1 + Rel5Layout.rowBits + lay.k ≤ lay.nVars) ∧
      Rel5Canonical lay c ∧ Rel5FieldEquations O pub lay c sf pf := by
  obtain ⟨c, ⟨hparse, _, _, henc, lay, hlay, _⟩, _, hpin, ht, hc⟩ :=
    verifyBytesRel5_refines_typed h
  have hcan := parseContainerRel5_canonical hlay hparse
  obtain ⟨sf, pf, hF⟩ := rel5TypedVerify_field hlay hcan ht
  exact ⟨c, lay, sf, pf, hparse, henc, hpin, hc, hlay, rel5Layout_admissible hlay, hcan, hF⟩

/-! ### 9.3 The folds of an accepted query on the actual evaluation domain -/

theorem Rel3Layout.quarterAt_pos (lay : Rel3Layout) (t : ℕ) (ht : 2 * t + 2 ≤ lay.nVars + 3) :
    0 < lay.quarterAt t := by
  unfold Rel3Layout.quarterAt
  rw [lay.codeLen_eq]
  exact Nat.div_pos (Nat.pow_le_pow_right (by norm_num) ht) (by positivity)

theorem Rel3Layout.quarterAt_lt_goldilocks (lay : Rel3Layout) (t : ℕ) (hn : lay.nVars ≤ 28) :
    lay.quarterAt t < goldilocks := by
  unfold Rel3Layout.quarterAt
  have h1 : lay.codeLen / 2 ^ (2 * t + 2) ≤ lay.codeLen := Nat.div_le_self _ _
  have h2 : lay.codeLen ≤ 2 ^ 31 := by
    rw [lay.codeLen_eq]; exact Nat.pow_le_pow_right (by norm_num) (by omega)
  have h3 : (2 : ℕ) ^ 31 < goldilocks := by rw [goldilocks_eq]; norm_num
  omega

/-- A cubic is linear in its coefficients: the combined word `F + θ C` of two cubics is the cubic
with coefficients `a_e + θ b_e`. -/
theorem cubicAt_combine (lay : Rel3Layout) (t : ℕ) (a0 a1 a2 a3 b0 b1 b2 b3 θ : Fp3Field) (pos : ℕ) :
    cubicAt lay t a0 a1 a2 a3 pos + θ * cubicAt lay t b0 b1 b2 b3 pos
      = cubicAt lay t (a0 + θ * b0) (a1 + θ * b1) (a2 + θ * b2) (a3 + θ * b3) pos := by
  unfold cubicAt; ring

/-- **A committed fold step of an accepted query, on the evaluation domain**
(`fold4_foldXInv_interpolates` applied to `FoldField`). No layer bound is assumed: `FoldField`
carries `t < T`, proved for every fold step of an accepted run (`foldRun_field`), and
`2 t + 2 ≤ n < n + 3` follows (`Rel3Layout.layer_of_lt_folds`). If the four opened values of the coset
leaf `j = pos mod L_t/4` of layer `t` are the values of a cubic `c_0 + c_1 X + c_2 X^2 + c_3 X^3`
at the domain points `ω_t^(j + k L_t/4)`, the value the verifier carries to the next layer is
`c_0 + r_{2t} c_1 + r_{2t+1} c_2 + r_{2t} r_{2t+1} c_3`. -/
theorem foldField_cubic {O : Rel4Oracles} {lay : Rel3Layout} {roots : List Bytes} {rs : List Fp3}
    {iotaInv : UInt64} {s : FoldState} {o : CosetOpening} {s' : FoldState}
    (h : FoldField O lay roots rs iotaInv s o s') (hi : rel3IotaInv = some iotaInv)
    (hn : lay.nVars ≤ 28) (c0 c1 c2 c3 : Fp3Field)
    (h0 : o.values.1.toField = cubicAt lay s.t c0 c1 c2 c3 (s.pos % lay.quarterAt s.t))
    (h1 : o.values.2.1.toField
      = cubicAt lay s.t c0 c1 c2 c3 (s.pos % lay.quarterAt s.t + 1 * lay.quarterAt s.t))
    (h2 : o.values.2.2.1.toField
      = cubicAt lay s.t c0 c1 c2 c3 (s.pos % lay.quarterAt s.t + 2 * lay.quarterAt s.t))
    (h3 : o.values.2.2.2.toField
      = cubicAt lay s.t c0 c1 c2 c3 (s.pos % lay.quarterAt s.t + 3 * lay.quarterAt s.t)) :
    s'.prev.toField
      = c0 + (rs.getD (2 * s.t) Fp3.zero).toField * c1
        + (rs.getD (2 * s.t + 1) Fp3.zero).toField * c2
        + (rs.getD (2 * s.t) Fp3.zero).toField * (rs.getD (2 * s.t + 1) Fp3.zero).toField * c3 := by
  have ht : 2 * s.t + 2 ≤ lay.nVars + 3 := (lay.layer_of_lt_folds h.1).2
  have hj : s.pos % lay.quarterAt s.t < goldilocks :=
    lt_trans (Nat.mod_lt _ (lay.quarterAt_pos s.t ht)) (lay.quarterAt_lt_goldilocks s.t hn)
  rw [h.2.2.2.1, h0, h1, h2, h3]
  exact fold4_foldXInv_interpolates lay s.t _ hn ht hj hi c0 c1 c2 c3 _ _

/-- **The first fold of an accepted relation-5 query, on the evaluation domain.** If the opened
`F` leaf (under `root_0`) holds the values of a cubic `a` and the opened weight leaf (under the
pinned `weight_root`) the values of a cubic `b` at the four points `ω^(j + k L/4)` of the queried
coset, then the value carried to layer 1 is the fold of the cubic `a + θ b` of the combined word
`H = F + θ C` at the first two fold challenges. -/
theorem rel5_layer0_fold_cubic {O : Rel5Oracles} {lay : Rel5Layout} {roots : List Bytes}
    {wroot : Bytes} {rs : List Fp3} {theta final : Fp3} {iotaInv : UInt64} {tr : Fpn2Transcript}
    {q : AirQuery} {tr' : Fpn2Transcript}
    (h : QueryStepField4 O lay.air roots wroot rs theta final iotaInv tr q tr')
    (hi : rel3IotaInv = some iotaInv) (hn : lay.nVars ≤ 28) (a0 a1 a2 a3 b0 b1 b2 b3 : Fp3Field)
    (hF : ∀ k : Fin 4, ((list4 q.fLeaf.values).getD k Fp3.zero).toField
      = cubicAt lay.air.pcs 0 a0 a1 a2 a3 (airLeafIdx O lay.air tr + k * lay.air.pcs.quarterAt 0))
    (hC : ∀ k : Fin 4, ((list4 q.cLeaf.values).getD k Fp3.zero).toField
      = cubicAt lay.air.pcs 0 b0 b1 b2 b3 (airLeafIdx O lay.air tr + k * lay.air.pcs.quarterAt 0)) :
    (airLayer0 O lay.air rs theta iotaInv tr q).prev.toField
      = (a0 + theta.toField * b0) + (rs.getD 0 Fp3.zero).toField * (a1 + theta.toField * b1)
        + (rs.getD 1 Fp3.zero).toField * (a2 + theta.toField * b2)
        + (rs.getD 0 Fp3.zero).toField * (rs.getD 1 Fp3.zero).toField
          * (a3 + theta.toField * b3) := by
  have hn' : lay.air.pcs.nVars ≤ 28 := hn
  have ht : 2 * 0 + 2 ≤ lay.air.pcs.nVars + 3 := by omega
  have hj : airLeafIdx O lay.air tr < goldilocks :=
    lt_trans (Nat.mod_lt _ (lay.air.pcs.quarterAt_pos 0 ht))
      (lay.air.pcs.quarterAt_lt_goldilocks 0 hn')
  have f0 := hF 0
  have f1 := hF 1
  have f2 := hF 2
  have f3 := hF 3
  have g0 := hC 0
  have g1 := hC 1
  have g2 := hC 2
  have g3 := hC 3
  simp only [list4, Fin.val_zero, Fin.val_one, Fin.val_two, List.getD_cons_zero,
    List.getD_cons_succ, zero_mul, add_zero] at f0 f1 f2 g0 g1 g2
  have e3 : ((3 : Fin 4) : ℕ) = 3 := rfl
  rw [e3] at f3 g3
  simp only [list4, List.getD_cons_zero, List.getD_cons_succ] at f3 g3
  rw [h.2.2.1, f0, f1, f2, f3, g0, g1, g2, g3]
  simp only [cubicAt_combine]
  exact fold4_foldXInv_interpolates lay.air.pcs 0 _ hn' ht hj hi _ _ _ _ _ _

/-! ### 9.4 The relation-5 algebraic error budget -/

/-- An explicit numerator for the union bound over every verifier challenge of relation 5 (errors
are `numerator / p^3`), on a layout with `d = k + beta` inner-product variables:

* the row point `r_i` and the tile point `r_b`: `5 + beta` (a nonzero multilinear table
  `Y − W X` over the `32 · B` output rows vanishes at a uniform point, `fp3_mle_zero_sound`);
* the Libra challenge `λ`: `1` (`fp3_linear_root_card_le`);
* the inner-product sumcheck: `d` rounds of degree 3, `3 d` (`fp3_sumcheck_sound`);
* `μ`: three batched claims, a nonzero polynomial of degree `≤ 2` in `μ`, `2`
  (`fp3_poly_root_card_le`);
* Basefold: `n + 2` rounds of degree 2 (`θ`, `r_1`, `r'`), `2 (n + 2)` (`fp3_sumcheck_sound`).

Every term is the numerator of a lemma instantiated at `E` in this file. The count itself is a
budget: it is not derived from the executable verifier, and it does not contain the proximity /
query error of the opening. -/
def rel5AlgNumer (lay : Rel5Layout) : ℕ :=
  (Rel5Layout.rowBits + lay.beta) + 1 + 3 * lay.d + 2 + 2 * (lay.nVars + 2)

/-- On an admissible layout (`n ≤ 28`, `1 + 5 + k ≤ n`) the numerator is `6 n − k − 12 ≤ 156`. -/
theorem rel5AlgNumer_le (lay : Rel5Layout) (hn : lay.nVars ≤ 28)
    (hk : 1 + Rel5Layout.rowBits + lay.k ≤ lay.nVars) :
    rel5AlgNumer lay = 6 * lay.nVars - lay.k - 12 ∧ rel5AlgNumer lay ≤ 156 := by
  unfold rel5AlgNumer Rel5Layout.d Rel5Layout.beta Rel5Layout.rowBits at *
  omega

open Finset in
/-- A nonzero polynomial over `E` has at most `deg` roots (the `μ` batching: degree `≤ 2`). -/
theorem fp3_poly_root_card_le (q : Polynomial Fp3Field) (hq : q ≠ 0) :
    (univ.filter fun x : Fp3Field => q.eval x = 0).card ≤ q.natDegree := by
  calc (univ.filter fun x : Fp3Field => q.eval x = 0).card
      ≤ q.roots.toFinset.card := Finset.card_le_card fun x hx => by
        rw [Multiset.mem_toFinset, mem_roots hq]
        exact (Finset.mem_filter.mp hx).2
    _ ≤ Multiset.card q.roots := Multiset.toFinset_card_le _
    _ ≤ q.natDegree := card_roots' q

/-- **Concrete algebraic error of relation 5 over `E`:** at most `2^-160` (indeed `2^-183`),
hence below `2^-150`, for every admissible layout. -/
theorem rel5_alg_error_lt (lay : Rel5Layout) (hn : lay.nVars ≤ 28)
    (hk : 1 + Rel5Layout.rowBits + lay.k ≤ lay.nVars) :
    (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 183)⁻¹ ∧
      (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ ∧
      (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) < (2 ^ 150)⁻¹ := by
  have h183 : (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 183)⁻¹ := by
    rw [frac_le_inv_pow2_iff _ _ _ (by rw [goldilocks_eq]; norm_num)]
    calc rel5AlgNumer lay * 2 ^ 183 ≤ 2 ^ 8 * 2 ^ 183 :=
          Nat.mul_le_mul_right _ ((rel5AlgNumer_le lay hn hk).2.trans (by norm_num))
      _ ≤ goldilocks ^ 3 := by rw [goldilocks_eq]; norm_num
  have h160 : (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ :=
    h183.trans (by norm_num)
  exact ⟨h183, h160, lt_of_le_of_lt h160 (by norm_num)⟩

/-- Every layout the bytes verifier admits for relation 5 meets the bound. -/
theorem rel5_alg_error_of_layout {p : Fpn2Profile} {lay : Rel5Layout}
    (h : rel5Layout p = some lay) :
    (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ ∧
      (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) < (2 ^ 150)⁻¹ :=
  (rel5_alg_error_lt lay (rel5Layout_admissible h).2.2.1 (rel5Layout_admissible h).2.2.2).2

/-- The registered profile `fpn2-folded-vec-n16-v1` (`n = 16`, one tile): numerator `74`. -/
theorem rel5_alg_error_foldedVecN16V1 :
    ∃ lay, rel5Layout foldedVecN16V1 = some lay ∧ rel5AlgNumer lay = 74 ∧
      (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ ∧
      (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) < (2 ^ 150)⁻¹ :=
  ⟨_, rel5Layout_foldedVecN16.1, by decide, rel5_alg_error_of_layout rel5Layout_foldedVecN16.1⟩

/-- The registered profile `fpn2-folded-vec-n22-v1` (`n = 22`, 64 tiles, `beta = 6`): numerator
`110`. -/
theorem rel5_alg_error_foldedVecN22V1 :
    ∃ lay, rel5Layout foldedVecN22V1 = some lay ∧ rel5AlgNumer lay = 110 ∧
      (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ ∧
      (rel5AlgNumer lay : ℝ) / (goldilocks ^ 3 : ℕ) < (2 ^ 150)⁻¹ :=
  ⟨_, rel5Layout_foldedVecN22.1, by decide, rel5_alg_error_of_layout rel5Layout_foldedVecN22.1⟩

open Finset in
/-- **Union bound at the relation-5 budget.** Any finite family of bad-challenge events whose
densities are `d i / p^3` with `Σ d i ≤ rel5AlgNumer lay` has a union of density `≤ 2^-160`. -/
theorem rel5_alg_union_bound {Ω ι : Type*} [Fintype Ω] [DecidableEq Ω] [Fintype ι]
    (lay : Rel5Layout) (hn : lay.nVars ≤ 28) (hk : 1 + Rel5Layout.rowBits + lay.k ≤ lay.nVars)
    (Bad : ι → Finset Ω) (d : ι → ℕ)
    (hBad : ∀ i, ((Bad i).card : ℝ) ≤ ((d i : ℝ) / (goldilocks ^ 3 : ℕ)) * Fintype.card Ω)
    (hd : ∑ i, d i ≤ rel5AlgNumer lay) :
    ((univ.biUnion Bad).card : ℝ) ≤ (2 ^ 160)⁻¹ * Fintype.card Ω := by
  have hΩ : (0 : ℝ) ≤ Fintype.card Ω := Nat.cast_nonneg _
  have h1 : ((univ.biUnion Bad).card : ℝ) ≤ ∑ i, ((Bad i).card : ℝ) := by
    exact_mod_cast Finset.card_biUnion_le
  have h2 : ∑ i, ((Bad i).card : ℝ)
      ≤ ((∑ i, d i : ℕ) : ℝ) / (goldilocks ^ 3 : ℕ) * Fintype.card Ω := by
    rw [Nat.cast_sum, Finset.sum_div, Finset.sum_mul]
    exact Finset.sum_le_sum fun i _ => hBad i
  have h3 : ((∑ i, d i : ℕ) : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 160)⁻¹ :=
    le_trans (div_le_div_of_nonneg_right (by exact_mod_cast hd) fp3Field_card_pos.le)
      (rel5_alg_error_lt lay hn hk).2.1
  exact h1.trans (h2.trans (mul_le_mul_of_nonneg_right h3 hΩ))

/-! ### 9.5 The round-by-round lifetime model for relation 5 -/

section LifetimeRel5

open Finset

/-- **One Fiat–Shamir round of the relation-5 argument, as an adaptive prover prepares it.** The
answer space is that of relation 4 (`Rel4Ans k`: one challenge in `E`, `k` sampling seeds).
Relation 5 has no lookup, so there is no `logup` round:

* `sumcheck`: a round polynomial of degree `≤ 3` (inner product: 3, Basefold: 2) that differs
  from the true partial sum;
* `poly`: a nonzero polynomial of degree `≤ 2` in the challenge (`λ` and one coordinate of `r_i`
  / `r_b`: degree 1; the batching challenge `μ`: degree 2);
* `spot`: a sampling round whose bad seeds have density at least `σ = num / den`. -/
inductive Rel5Round (k num den : ℕ) : Type
  | sumcheck (n d : ℕ) (hd : d ≤ 3) (P : (Fin n → Fp3Field) → Fp3Field) (hP : RoundDeg d P)
      (g : Fin n → (Fin n → Fp3Field) → Polynomial Fp3Field) (hg : ∀ i r, (g i r).natDegree ≤ d)
      (i : Fin n) (r : Fin n → Fp3Field)
  | poly (q : Polynomial Fp3Field) (hq : q ≠ 0) (hdeg : q.natDegree ≤ 2)
  | spot (badSeed : Finset Fp3Field)
      (hbad : num * Fintype.card Fp3Field ≤ den * badSeed.card)

/-- The escape set of a round: the oracle answers on which the false claim survives. -/
noncomputable def Rel5Round.esc {k num den : ℕ} : Rel5Round k num den → Finset (Rel4Ans k)
  | .sumcheck _ _ _ P _ g _ i r => badSet P g i r ×ˢ univ
  | .poly q _ _ => (univ.filter fun x => q.eval x = 0) ×ˢ univ
  | .spot badSeed _ => univ ×ˢ missing badSeed k

/-- `true` for sampling rounds, `false` for algebraic rounds. -/
def Rel5Round.isSpot {k num den : ℕ} : Rel5Round k num den → Bool
  | .spot _ _ => true
  | _ => false

/-- **Every relation-5 escape set meets its bound** (`hEsc` of `ProtocolBridge`): `2^-106` for the
algebraic rounds, `(1 − σ)^k + 2^-106` for a sampling round, the latter derived from
`SpotCheck.frac_bad_bound` for every `k` and `σ`. -/
theorem rel5Round_esc_card_le (k num den : ℕ) (hden : 0 < den) (s : Rel5Round k num den) :
    ((s.esc).card : ℝ) ≤ rel4Eps (rel4Miss k num den) s.isSpot * Fintype.card (Rel4Ans k) := by
  cases s with
  | sumcheck n d hd P hP g hg i r =>
    have h := badSet_card_le P d hP g hg i r
    have h' : ((badSet P g i r).card : ℝ) ≤ ((3 : ℕ) : ℝ) := by exact_mod_cast h.trans hd
    exact rel4_alg_lift k _ _ (h'.trans (fp3_small_card_le 3 (by norm_num)))
  | poly q hq hdeg =>
    have h' : (((univ.filter fun x : Fp3Field => q.eval x = 0).card : ℕ) : ℝ) ≤ ((2 : ℕ) : ℝ) :=
      Nat.cast_le.mpr ((fp3_poly_root_card_le q hq).trans hdeg)
    exact rel4_alg_lift k _ _ (h'.trans (fp3_small_card_le 2 (by norm_num)))
  | spot badSeed hbad =>
    refine (spot_esc_card_le k num den hden badSeed hbad).trans
      (mul_le_mul_of_nonneg_right ?_ (Nat.cast_nonneg _))
    show rel4Miss k num den ≤ rel4Miss k num den + (2 ^ 106)⁻¹
    linarith [show (0 : ℝ) ≤ (2 ^ 106)⁻¹ by positivity]

/-- **A relation-5 sampling round built from a block pool**, as `Rel4Round.ofSpotCheck`: each of
the `k` seeds `e ∈ E` opens the block `u e` of a pool `B`, `Bad` is the set of bad blocks, and the
single premise is that the seeds opening a bad block have density at least `σ = num / den`
(`seedShare_of_fiber_ge` reduces it to the block share for a map with fibers of size `≥ c`). -/
noncomputable def Rel5Round.ofSpotCheck (k num den : ℕ) {B : Type*} [DecidableEq B]
    (u : Fp3Field → B) (Bad : Finset B)
    (hshare : num * Fintype.card Fp3Field ≤ den * (univ.filter fun e => u e ∈ Bad).card) :
    Rel5Round k num den :=
  .spot (univ.filter fun e => u e ∈ Bad) hshare

/-- The escape answers of `ofSpotCheck` are exactly those whose `k` sampled blocks all miss `Bad`. -/
theorem Rel5Round.mem_ofSpotCheck_esc (k num den : ℕ) {B : Type*} [DecidableEq B]
    (u : Fp3Field → B) (Bad : Finset B)
    (hshare : num * Fintype.card Fp3Field ≤ den * (univ.filter fun e => u e ∈ Bad).card)
    (a : Rel4Ans k) :
    a ∈ (Rel5Round.ofSpotCheck k num den u Bad hshare).esc ↔ ∀ j, u (a.2 j) ∉ Bad := by
  simp [Rel5Round.ofSpotCheck, Rel5Round.esc, missing]

/-- The bad-accept event of the model: some oracle answer lands in the escape set of the round
the adaptive prover prepared for it. -/
noncomputable def rel5BadAcc {k num den Q : ℕ}
    (st : Fin Q → (Fin Q → Rel4Ans k) → Rel5Round k num den) : Finset (Fin Q → Rel4Ans k) := by
  classical exact univ.filter fun a => ∃ j, a j ∈ (st j a).esc

/-- **`ProtocolRBRSpec` inhabited at the relation-5 model** (`BadAcc = rel5BadAcc st`, `Fail = ∅`,
`Doomed = True`, `Esc = Rel5Round.esc`). As for relation 4, `BadAcc` is *defined* as the escape
event, so `hInit`, `hFinal` and `hStep` hold by construction: this is not an extraction argument. -/
noncomputable def rel5RbrSpec {k num den Q : ℕ} (hQ : 0 < Q)
    (st : Fin Q → (Fin Q → Rel4Ans k) → Rel5Round k num den) :
    ProtocolRBRSpec st (fun _ => True) Rel5Round.esc (rel5BadAcc st) ∅ := by
  classical exact
  { R := fun _ => 1
    roundHolds := fun _ i => (i : ℕ) = 1
    qIdx := fun a _ => if h : ∃ j, a j ∈ (st j a).esc then Classical.choose h else ⟨0, hQ⟩
    hInit := fun _ _ _ h => absurd h (by decide)
    hFinal := fun _ _ _ => rfl
    hStep := fun a ha _ _ _ _ => by
      have h : ∃ j, a j ∈ (st j a).esc := by
        unfold rel5BadAcc at ha
        exact (Finset.mem_filter.mp ha).2
      rw [dif_pos h]
      exact ⟨trivial, Classical.choose_spec h⟩ }

/-- **General lifetime bound of the relation-5 model**: over `Q` adaptive random-oracle queries,
`Pr[BadAcc] ≤ Q · ((1 − σ)^k + 2^-106)`, for every `k` and `σ = num / den`. -/
theorem rel5_rbr_lifetime_general (k num den : ℕ) (hden : 0 < den) {Q : ℕ} (hQ0 : 0 < Q)
    (st : Fin Q → (Fin Q → Rel4Ans k) → Rel5Round k num den)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a) :
    ((rel5BadAcc st).card : ℝ)
      ≤ (Q * (rel4Miss k num den + (2 ^ 106)⁻¹)) * (Fintype.card (Rel4Ans k) : ℝ) ^ Q := by
  have hm0 := rel4Miss_nonneg k num den
  have h := rbr_lifetime_soundness st hst (fun _ => True) Rel5Round.esc Rel5Round.isSpot
    (rel4Eps (rel4Miss k num den)) (rel4Miss k num den + (2 ^ 106)⁻¹)
    (add_nonneg hm0 (by positivity))
    (fun b => by cases b <;> simp [rel4Eps, hm0])
    (fun s _ => rel5Round_esc_card_le k num den hden s) (rel5BadAcc st) ∅ 0 (by simp)
    (hpath_of_protocol_spec (rel5RbrSpec hQ0 st))
  simpa using h

/-- The general bound against a query budget `Q ≤ Qmax` and a target `ε`. -/
theorem rel5_lifetime_of_budget (k num den : ℕ) (hden : 0 < den) {Q : ℕ} (hQ0 : 0 < Q)
    (st : Fin Q → (Fin Q → Rel4Ans k) → Rel5Round k num den)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a) (Qmax ε : ℝ) (hQ : (Q : ℝ) ≤ Qmax)
    (hε : Qmax * (rel4Miss k num den + (2 ^ 106)⁻¹) ≤ ε) :
    ((rel5BadAcc st).card : ℝ) ≤ ε * (Fintype.card (Rel4Ans k) : ℝ) ^ Q := by
  have h := rel5_rbr_lifetime_general k num den hden hQ0 st hst
  refine h.trans (mul_le_mul_of_nonneg_right ?_ (by positivity))
  exact (mul_le_mul_of_nonneg_right hQ
    (add_nonneg (rel4Miss_nonneg k num den) (by positivity))).trans hε

/-- **Lifetime soundness of the relation-5 model (`k = 105`, `σ = 1/2`, `Q ≤ 2^64`):**
`Pr[BadAcc] ≤ 2^-40`, through `fpn_protocol_lifetime_soundness` with every argument instantiated. -/
theorem rel5_lifetime_soundness_k105 {Q : ℕ} (hQ0 : 0 < Q) (hQ : Q ≤ 2 ^ 64)
    (st : Fin Q → (Fin Q → Rel4Ans 105) → Rel5Round 105 1 2)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a) :
    ((rel5BadAcc st).card : ℝ) ≤ (2 ^ 40)⁻¹ * (Fintype.card (Rel4Ans 105) : ℝ) ^ Q :=
  fpn_protocol_lifetime_soundness hQ st hst (fun _ => True) Rel5Round.esc Rel5Round.isSpot
    (rel4Eps (rel4Miss 105 1 2)) (rel4Miss 105 1 2) (2 ^ 106)⁻¹ (2 ^ 106)⁻¹ 0
    (rel4Miss_nonneg _ _ _) (rel4Miss_half 105).le
    (by positivity) le_rfl le_rfl (by positivity)
    (fun b => by cases b <;> simp [rel4Eps])
    (fun s _ => rel5Round_esc_card_le 105 1 2 (by norm_num) s) (rel5BadAcc st) ∅
    (by simp) (rel5RbrSpec hQ0 st)

/-- **External-seed lifetime soundness at `σ = 1/2` (`k = 51`, `Q ≤ 2^30`):** `Pr[BadAcc] ≤ 2^-20`. -/
theorem rel5_lifetime_soundness_k51 {Q : ℕ} (hQ0 : 0 < Q) (hQ : Q ≤ 2 ^ 30)
    (st : Fin Q → (Fin Q → Rel4Ans 51) → Rel5Round 51 1 2)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a) :
    ((rel5BadAcc st).card : ℝ) ≤ (2 ^ 20)⁻¹ * (Fintype.card (Rel4Ans 51) : ℝ) ^ Q :=
  rel5_lifetime_of_budget 51 1 2 (by norm_num) hQ0 st hst (2 ^ 30) _
    (by exact_mod_cast hQ) (by rw [rel4Miss_half]; norm_num)

/-- **External-seed lifetime soundness at `σ = 3/10` (`k = 100`, `Q ≤ 2^30`):**
`Pr[BadAcc] ≤ 2^-20`. -/
theorem rel5_lifetime_soundness_k100 {Q : ℕ} (hQ0 : 0 < Q) (hQ : Q ≤ 2 ^ 30)
    (st : Fin Q → (Fin Q → Rel4Ans 100) → Rel5Round 100 3 10)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a) :
    ((rel5BadAcc st).card : ℝ) ≤ (2 ^ 20)⁻¹ * (Fintype.card (Rel4Ans 100) : ℝ) ^ Q :=
  rel5_lifetime_of_budget 100 3 10 (by norm_num) hQ0 st hst (2 ^ 30) _
    (by exact_mod_cast hQ) (by rw [rel4Miss_three_tenths]; norm_num)

/-- Strict forms: `Pr[BadAcc] < 2^-40` (`k = 105`, `σ = 1/2`, `Q ≤ 2^64`) and `< 2^-20`
(`k = 100`, `σ = 3/10`, `Q ≤ 2^30`). -/
theorem rel5_lifetime_soundness_strict {Q : ℕ} (hQ0 : 0 < Q) :
    (Q ≤ 2 ^ 64 → ∀ st : Fin Q → (Fin Q → Rel4Ans 105) → Rel5Round 105 1 2,
      (∀ j a x, st j (Function.update a j x) = st j a) →
      ((rel5BadAcc st).card : ℝ) < (2 ^ 40)⁻¹ * (Fintype.card (Rel4Ans 105) : ℝ) ^ Q) ∧
    (Q ≤ 2 ^ 30 → ∀ st : Fin Q → (Fin Q → Rel4Ans 100) → Rel5Round 100 3 10,
      (∀ j a x, st j (Function.update a j x) = st j a) →
      ((rel5BadAcc st).card : ℝ) < (2 ^ 20)⁻¹ * (Fintype.card (Rel4Ans 100) : ℝ) ^ Q) := by
  have hpos : ∀ k : ℕ, (0 : ℝ) < (Fintype.card (Rel4Ans k) : ℝ) ^ Q := fun k =>
    pow_pos (Nat.cast_pos.mpr Fintype.card_pos) Q
  constructor
  · intro hQ st hst
    have h := rel5_rbr_lifetime_general 105 1 2 (by norm_num) hQ0 st hst
    refine lt_of_le_of_lt h (mul_lt_mul_of_pos_right ?_ (hpos 105))
    have hQr : (Q : ℝ) ≤ 2 ^ 64 := by exact_mod_cast hQ
    rw [rel4Miss_half]
    calc (Q : ℝ) * ((1 / 2) ^ 105 + (2 ^ 106)⁻¹)
        ≤ 2 ^ 64 * ((1 / 2) ^ 105 + (2 ^ 106)⁻¹) :=
          mul_le_mul_of_nonneg_right hQr (by positivity)
      _ < (2 ^ 40)⁻¹ := by norm_num
  · intro hQ st hst
    have h := rel5_rbr_lifetime_general 100 3 10 (by norm_num) hQ0 st hst
    refine lt_of_le_of_lt h (mul_lt_mul_of_pos_right ?_ (hpos 100))
    have hQr : (Q : ℝ) ≤ 2 ^ 30 := by exact_mod_cast hQ
    rw [rel4Miss_three_tenths]
    calc (Q : ℝ) * ((7 / 10) ^ 100 + (2 ^ 106)⁻¹)
        ≤ 2 ^ 30 * ((7 / 10) ^ 100 + (2 ^ 106)⁻¹) :=
          mul_le_mul_of_nonneg_right hQr (by positivity)
      _ < (2 ^ 20)⁻¹ := by norm_num

/-- The relation-5 model is not vacuous: with one query, the batching round `μ ↦ μ^2` (a nonzero
polynomial of degree 2) has a nonempty bad-accept event. -/
theorem rel5_rbr_model_nontrivial (k num den : ℕ) :
    ∃ st : Fin 1 → (Fin 1 → Rel4Ans k) → Rel5Round k num den,
      (∀ j a x, st j (Function.update a j x) = st j a) ∧ (rel5BadAcc st).Nonempty := by
  refine ⟨fun _ _ => .poly (X ^ 2) (pow_ne_zero 2 X_ne_zero) (by simp), fun _ _ _ => rfl,
    fun _ => (0, fun _ => 0), ?_⟩
  unfold rel5BadAcc
  simp [Rel5Round.esc]

/-- The sampling premise is satisfiable at `σ = 3/10` with a nonempty escape event. -/
theorem rel5_spot_model_nontrivial (k : ℕ) :
    ∃ st : Fin 1 → (Fin 1 → Rel4Ans k) → Rel5Round k 3 10,
      (∀ j a x, st j (Function.update a j x) = st j a) ∧ (∀ j a, (st j a).isSpot = true) ∧
        (rel5BadAcc st).Nonempty := by
  have hc : 1 < Fintype.card Fp3Field := Fintype.one_lt_card
  refine ⟨fun _ _ => .spot (univ.erase 0) ?_, fun _ _ _ => rfl, fun _ _ => rfl,
    fun _ => (0, fun _ => 0), ?_⟩
  · rw [card_erase_of_mem (mem_univ _), card_univ]; omega
  · unfold rel5BadAcc
    simp [Rel5Round.esc, missing]

end LifetimeRel5

end FlashProveNext

#print axioms FlashProveNext.goldilocksP_toNat_eq_goldilocks
#print axioms FlashProveNext.goldilocks_sub_one_factor
#print axioms FlashProveNext.powModAux_spec
#print axioms FlashProveNext.powMod_eq
#print axioms FlashProveNext.zmod_pow_eq_iff
#print axioms FlashProveNext.seven_pow_goldilocks
#print axioms FlashProveNext.two_pow_goldilocks_third
#print axioms FlashProveNext.prime_dvd_goldilocks_sub_one
#print axioms FlashProveNext.goldilocks_lt
#print axioms FlashProveNext.goldilocks_prime
#print axioms FlashProveNext.goldilocksP_prime
#print axioms FlashProveNext.goldilocks_two_not_cube
#print axioms FlashProveNext.fp3Poly_monic
#print axioms FlashProveNext.fp3Poly_natDegree
#print axioms FlashProveNext.fp3Poly_ne_zero
#print axioms FlashProveNext.fp3Poly_irreducible
#print axioms FlashProveNext.fp3Field_card
#print axioms FlashProveNext.fp3Field_card_gt
#print axioms FlashProveNext.fpOfNat_toNat
#print axioms FlashProveNext.fpToField_ofNat
#print axioms FlashProveNext.fpToField_zero
#print axioms FlashProveNext.fpToField_one
#print axioms FlashProveNext.fpToField_two
#print axioms FlashProveNext.fpToField_add
#print axioms FlashProveNext.fpToField_mul
#print axioms FlashProveNext.fpToField_sub
#print axioms FlashProveNext.fpToField_injective
#print axioms FlashProveNext.fp3U_cube
#print axioms FlashProveNext.fp3_coords_eq_zero
#print axioms FlashProveNext.Fp3.toField_zero
#print axioms FlashProveNext.Fp3.toField_one
#print axioms FlashProveNext.Fp3.toField_ofFp
#print axioms FlashProveNext.Fp3.toField_add
#print axioms FlashProveNext.Fp3.toField_sub
#print axioms FlashProveNext.Fp3.toField_scale
#print axioms FlashProveNext.Fp3.toField_mul
#print axioms FlashProveNext.Fp3.zero_canonical
#print axioms FlashProveNext.Fp3.one_canonical
#print axioms FlashProveNext.Fp3.add_canonical
#print axioms FlashProveNext.Fp3.sub_canonical
#print axioms FlashProveNext.Fp3.mul_canonical
#print axioms FlashProveNext.Fp3.toField_injective
#print axioms FlashProveNext.Fp3.toField_eq_zero_iff
#print axioms FlashProveNext.fpPowAux_toField
#print axioms FlashProveNext.fpPow_toField
#print axioms FlashProveNext.fpInv_toField
#print axioms FlashProveNext.lagrangeBasis_toField
#print axioms FlashProveNext.lagrangeEval_toField
#print axioms FlashProveNext.roundNodes_injOn
#print axioms FlashProveNext.roundPoly_natDegree_le
#print axioms FlashProveNext.roundPoly_eval_node
#print axioms FlashProveNext.fp3Field_card_pos
#print axioms FlashProveNext.fp3_small_card_le
#print axioms FlashProveNext.fp3_sumcheck_sound
#print axioms FlashProveNext.fp3_mle_zero_sound
#print axioms FlashProveNext.fp3_logup_sound
#print axioms FlashProveNext.fp3_wiring_logup_card_le
#print axioms FlashProveNext.fp3_linear_root_card_le
#print axioms FlashProveNext.fp3_zerocheck_sound
#print axioms FlashProveNext.rel4AlgNumer_lt
#print axioms FlashProveNext.rel4Layout_nVars_le
#print axioms FlashProveNext.rel4_alg_error_lt
#print axioms FlashProveNext.rel4_alg_error_n14_n24_n28
#print axioms FlashProveNext.rel4_alg_error_foldedAirV1
#print axioms FlashProveNext.rel4_alg_error_foldedAirN20V1
#print axioms FlashProveNext.rel4_alg_union_bound
#print axioms FlashProveNext.rel4Miss_nonneg
#print axioms FlashProveNext.rel4Miss_half
#print axioms FlashProveNext.rel4Miss_three_tenths
#print axioms FlashProveNext.rel4_alg_lift
#print axioms FlashProveNext.spot_miss_card_le
#print axioms FlashProveNext.rel4Round_esc_card_le
#print axioms FlashProveNext.spot_esc_card_le
#print axioms FlashProveNext.Rel4Round.ofSpotCheck
#print axioms FlashProveNext.Rel4Round.mem_ofSpotCheck_esc
#print axioms FlashProveNext.seedShare_of_fiber_ge
#print axioms FlashProveNext.rel4RbrSpec
#print axioms FlashProveNext.rel4_rbr_lifetime_general
#print axioms FlashProveNext.rel4_lifetime_of_budget
#print axioms FlashProveNext.rel4_lifetime_soundness_k105
#print axioms FlashProveNext.rel4_lifetime_soundness_k51
#print axioms FlashProveNext.rel4_lifetime_soundness_k100
#print axioms FlashProveNext.rel4_lifetime_soundness_strict
#print axioms FlashProveNext.rel4_rbr_model_nontrivial
#print axioms FlashProveNext.rel4_spot_model_nontrivial
#print axioms FlashProveNext.eq1_toField
#print axioms FlashProveNext.eqFold_toField
#print axioms FlashProveNext.eqEval_toField
#print axioms FlashProveNext.gateEval_toField
#print axioms FlashProveNext.childrenClaim_toField
#print axioms FlashProveNext.RunList.imp
#print axioms FlashProveNext.RunList.imp_mem
#print axioms FlashProveNext.roundPoly_round_shape
#print axioms FlashProveNext.zcRound_accepts_shape
#print axioms FlashProveNext.gkrRound_accepts_shape
#print axioms FlashProveNext.pcsRound_accepts_shape
#print axioms FlashProveNext.decMap_ok
#print axioms FlashProveNext.decPair_ok
#print axioms FlashProveNext.decFp_lt
#print axioms FlashProveNext.fp3_dec_canonical
#print axioms FlashProveNext.airGkr_dec_canonical
#print axioms FlashProveNext.airBody_dec_canonical
#print axioms FlashProveNext.parseContainerRel4_root_canonical
#print axioms FlashProveNext.Fp3.toField_ne_zero
#print axioms FlashProveNext.fpToField_four
#print axioms FlashProveNext.fpToField_seven
#print axioms FlashProveNext.seven_pow_half
#print axioms FlashProveNext.rel3IotaInv_sq
#print axioms FlashProveNext.fold4_toField
#print axioms FlashProveNext.coset4_sum0
#print axioms FlashProveNext.coset4_sum1
#print axioms FlashProveNext.coset4_sum2
#print axioms FlashProveNext.coset4_sum3
#print axioms FlashProveNext.fold4Field_interpolates
#print axioms FlashProveNext.goldilocks_seven_ne_zero
#print axioms FlashProveNext.goldilocks_neg_one_ne_one
#print axioms FlashProveNext.Rel3Layout.codeLen_eq
#print axioms FlashProveNext.Rel3Layout.codeLen_dvd
#print axioms FlashProveNext.rel3Omega_toField
#print axioms FlashProveNext.rel3Omega_pow_half
#print axioms FlashProveNext.rel3Omega_pow_codeLen
#print axioms FlashProveNext.rel3Omega_orderOf
#print axioms FlashProveNext.foldDomainPoint_ne_zero
#print axioms FlashProveNext.foldXInv_mul_domainPoint
#print axioms FlashProveNext.foldXInv_toField
#print axioms FlashProveNext.foldXInv_mul_domainPoint_fp3
#print axioms FlashProveNext.foldXInv_toField_fp3
#print axioms FlashProveNext.Rel3Layout.quarterAt_mul
#print axioms FlashProveNext.foldDomainPoint_quarter
#print axioms FlashProveNext.foldDomainPoint_add_quarter
#print axioms FlashProveNext.foldDomainPoint_succ
#print axioms FlashProveNext.foldDomainPoint_injOn
#print axioms FlashProveNext.rel3IotaInv_toField
#print axioms FlashProveNext.fp3Field_four_ne_zero
#print axioms FlashProveNext.fold4_foldXInv_interpolates
#print axioms FlashProveNext.foldSpec_field
#print axioms FlashProveNext.airLayer0_toField
#print axioms FlashProveNext.querySpec4_field
#print axioms FlashProveNext.gkrLayerSpec_field
#print axioms FlashProveNext.rel4TypedVerify_field
#print axioms FlashProveNext.verifyBytesRel4_accept_field
#print axioms FlashProveNext.fp3x3_dec_canonical
#print axioms FlashProveNext.fp3x4_dec_canonical
#print axioms FlashProveNext.decFam_forall
#print axioms FlashProveNext.coset_dec_canonical
#print axioms FlashProveNext.airQuery_dec_canonical
#print axioms FlashProveNext.airQueries_dec_canonical
#print axioms FlashProveNext.sel4_canonical
#print axioms FlashProveNext.airBody_dec_queries
#print axioms FlashProveNext.parseContainerRel4_queries_canonical
#print axioms FlashProveNext.Rel3Layout.layer_of_lt_folds
#print axioms FlashProveNext.foldRun_field
#print axioms FlashProveNext.foldField_layer_bound
#print axioms FlashProveNext.vecBody_dec_canonical
#print axioms FlashProveNext.parseContainerRel5_canonical
#print axioms FlashProveNext.ipRound_accepts_shape
#print axioms FlashProveNext.batchedClaim5_toField
#print axioms FlashProveNext.finalWeight5_toField
#print axioms FlashProveNext.queryStepSpec4_field
#print axioms FlashProveNext.querySpec5_field
#print axioms FlashProveNext.rel5TypedVerify_field
#print axioms FlashProveNext.rel5Layout_admissible
#print axioms FlashProveNext.verifyBytesRel5_accept_field
#print axioms FlashProveNext.Rel3Layout.quarterAt_pos
#print axioms FlashProveNext.Rel3Layout.quarterAt_lt_goldilocks
#print axioms FlashProveNext.cubicAt_combine
#print axioms FlashProveNext.foldField_cubic
#print axioms FlashProveNext.rel5_layer0_fold_cubic
#print axioms FlashProveNext.rel5AlgNumer_le
#print axioms FlashProveNext.fp3_poly_root_card_le
#print axioms FlashProveNext.rel5_alg_error_lt
#print axioms FlashProveNext.rel5_alg_error_of_layout
#print axioms FlashProveNext.rel5_alg_error_foldedVecN16V1
#print axioms FlashProveNext.rel5_alg_error_foldedVecN22V1
#print axioms FlashProveNext.rel5_alg_union_bound
#print axioms FlashProveNext.rel5Round_esc_card_le
#print axioms FlashProveNext.Rel5Round.ofSpotCheck
#print axioms FlashProveNext.Rel5Round.mem_ofSpotCheck_esc
#print axioms FlashProveNext.rel5RbrSpec
#print axioms FlashProveNext.rel5_rbr_lifetime_general
#print axioms FlashProveNext.rel5_lifetime_of_budget
#print axioms FlashProveNext.rel5_lifetime_soundness_k105
#print axioms FlashProveNext.rel5_lifetime_soundness_k51
#print axioms FlashProveNext.rel5_lifetime_soundness_k100
#print axioms FlashProveNext.rel5_lifetime_soundness_strict
#print axioms FlashProveNext.rel5_rbr_model_nontrivial
#print axioms FlashProveNext.rel5_spot_model_nontrivial
