/-
Combinatorial core of cost-weighted spot-checking and non-adaptive random-oracle grinding.

Given a finite set of computation blocks `B` and the subset `Bad` of blocks that violate the
local block relation, the verifier samples `k` blocks independently with replacement and accepts
only if none of the sampled blocks lies in `Bad`:

* `card_missing`              : exact count `(|B| - |Bad|)^k` of challenge tuples that miss `Bad`;
* `frac_bad_bound`            : if `|Bad| / |B| ≥ num / den`, the miss fraction is at most `((den - num) / den)^k`;
* `half_bad_bound`            : special case `|Bad| ≥ |B| / 2` giving `2^{-k}`;
* `cost_weighted_bound`       : uniform sampling over cost units `I` mapped to blocks via `π : I → B`
                                yields miss probability at most `(1 - σ)^k` whenever bad cost share `≥ σ`;
* `grind_frac_bound`          : non-adaptive grinding bound `|C| · (1 - num/den)^k` across a family `C` of commitments;
* `grind_cost_weighted_bound` : cost-weighted non-adaptive grinding bound.
-/
import Mathlib

namespace FlashProveNext

open Finset

variable {B : Type*} [Fintype B] [DecidableEq B]

/-- Challenge tuples of length `k` that miss every block in `Bad`. -/
def missing (Bad : Finset B) (k : ℕ) : Finset (Fin k → B) :=
  Finset.univ.filter fun q => ∀ i, q i ∉ Bad

theorem card_missing (Bad : Finset B) (k : ℕ) :
    (missing Bad k).card = (Fintype.card B - Bad.card) ^ k := by
  have h : missing Bad k = Fintype.piFinset (fun _ : Fin k => Badᶜ) := by
    ext q
    simp [missing, Fintype.mem_piFinset]
  rw [h, Fintype.card_piFinset]
  simp [Finset.card_compl]

/-- If the bad-block fraction is at least `num / den`, then `|missing Bad k| * den^k ≤ ((den - num) * |B|)^k`. -/
theorem frac_bad_bound (Bad : Finset B) (k num den : ℕ)
    (h : num * Fintype.card B ≤ den * Bad.card) :
    (missing Bad k).card * den ^ k ≤ ((den - num) * Fintype.card B) ^ k := by
  rw [card_missing, ← Nat.mul_pow]
  apply Nat.pow_le_pow_left
  rw [Nat.sub_mul, Nat.sub_mul]
  have h' : num * Fintype.card B ≤ Bad.card * den := by
    rw [Nat.mul_comm Bad.card]; exact h
  have e : Fintype.card B * den = den * Fintype.card B := Nat.mul_comm _ _
  omega

/-- When at least half of all blocks are bad, the miss probability is at most `2^{-k}`. -/
theorem half_bad_bound (Bad : Finset B) (k : ℕ)
    (h : Fintype.card B ≤ 2 * Bad.card) :
    (missing Bad k).card * 2 ^ k ≤ Fintype.card B ^ k := by
  have := frac_bad_bound Bad k 1 2 (by simpa using h)
  simpa using this

/-- Non-adaptive grinding bound over a finite candidate set `C` of size `Q = |C|`. -/
theorem grind_frac_bound {C : Type*} [Fintype C] [DecidableEq C] (Bad : C → Finset B)
    (k num den : ℕ) (h : ∀ c, num * Fintype.card B ≤ den * (Bad c).card) :
    (Finset.univ.filter fun O : C → Fin k → B => ∃ c, O c ∈ missing (Bad c) k).card * den ^ k
      ≤ Fintype.card C * (den - num) ^ k * Fintype.card (Fin k → B) ^ Fintype.card C := by
  classical
  have hMk : Fintype.card (Fin k → B) = Fintype.card B ^ k := by simp
  have key : ∀ c, (Finset.univ.filter fun O : C → Fin k → B => O c ∈ missing (Bad c) k).card * den ^ k
      ≤ (den - num) ^ k * Fintype.card (Fin k → B) ^ Fintype.card C := by
    intro c
    have e : (Finset.univ.filter fun O : C → Fin k → B => O c ∈ missing (Bad c) k)
        = Fintype.piFinset (fun c' => if c' = c then missing (Bad c) k else Finset.univ) := by
      ext O
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fintype.mem_piFinset]
      constructor
      · intro hO c'
        by_cases hc : c' = c
        · subst hc; simpa using hO
        · simp [hc]
      · intro hO
        simpa using hO c
    rw [e, Fintype.card_piFinset, ← Finset.mul_prod_erase Finset.univ _ (Finset.mem_univ c)]
    have e2 : ∏ x ∈ Finset.univ.erase c,
        (if x = c then missing (Bad c) k else Finset.univ).card
        = ∏ _x ∈ Finset.univ.erase c, Fintype.card (Fin k → B) :=
      Finset.prod_congr rfl (fun x hx => by simp [Finset.ne_of_mem_erase hx])
    rw [e2, Finset.prod_const, Finset.card_erase_of_mem (Finset.mem_univ c), Finset.card_univ,
      if_pos rfl]
    have hb : (missing (Bad c) k).card * den ^ k ≤ (den - num) ^ k * Fintype.card (Fin k → B) := by
      calc (missing (Bad c) k).card * den ^ k
          ≤ ((den - num) * Fintype.card B) ^ k := frac_bad_bound (Bad c) k num den (h c)
        _ = (den - num) ^ k * Fintype.card B ^ k := Nat.mul_pow ..
        _ = (den - num) ^ k * Fintype.card (Fin k → B) := by rw [hMk]
    have hpos : 0 < Fintype.card C := Fintype.card_pos_iff.mpr ⟨c⟩
    calc (missing (Bad c) k).card * Fintype.card (Fin k → B) ^ (Fintype.card C - 1) * den ^ k
        = ((missing (Bad c) k).card * den ^ k) * Fintype.card (Fin k → B) ^ (Fintype.card C - 1) := by
          ring
      _ ≤ ((den - num) ^ k * Fintype.card (Fin k → B)) *
            Fintype.card (Fin k → B) ^ (Fintype.card C - 1) :=
          Nat.mul_le_mul_right _ hb
      _ = (den - num) ^ k * (Fintype.card (Fin k → B) *
            Fintype.card (Fin k → B) ^ (Fintype.card C - 1)) := by ring
      _ = (den - num) ^ k * Fintype.card (Fin k → B) ^ Fintype.card C := by
          rw [← pow_succ', Nat.sub_add_cancel hpos]
  have sub : (Finset.univ.filter fun O : C → Fin k → B => ∃ c, O c ∈ missing (Bad c) k)
      ⊆ Finset.univ.biUnion
        (fun c => Finset.univ.filter fun O : C → Fin k → B => O c ∈ missing (Bad c) k) := by
    intro O hO
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hO
    obtain ⟨c, hc⟩ := hO
    simp only [Finset.mem_biUnion, Finset.mem_univ, true_and, Finset.mem_filter]
    exact ⟨c, hc⟩
  calc (Finset.univ.filter fun O : C → Fin k → B => ∃ c, O c ∈ missing (Bad c) k).card * den ^ k
      ≤ (∑ c, (Finset.univ.filter fun O : C → Fin k → B => O c ∈ missing (Bad c) k).card)
          * den ^ k :=
        Nat.mul_le_mul_right _ ((Finset.card_le_card sub).trans Finset.card_biUnion_le)
    _ = ∑ c, (Finset.univ.filter fun O : C → Fin k → B => O c ∈ missing (Bad c) k).card
          * den ^ k := Finset.sum_mul ..
    _ ≤ ∑ _c : C, (den - num) ^ k * Fintype.card (Fin k → B) ^ Fintype.card C :=
        Finset.sum_le_sum fun c _ => key c
    _ = Fintype.card C * (den - num) ^ k * Fintype.card (Fin k → B) ^ Fintype.card C := by
        simp [Finset.card_univ, mul_assoc]

/-- Grinding bound at threshold `1/2`: `|C| · 2^{-k}`. -/
theorem grind_bound {C : Type*} [Fintype C] [DecidableEq C] (Bad : C → Finset B) (k : ℕ)
    (h : ∀ c, Fintype.card B ≤ 2 * (Bad c).card) :
    (Finset.univ.filter fun O : C → Fin k → B => ∃ c, O c ∈ missing (Bad c) k).card * 2 ^ k
      ≤ Fintype.card C * Fintype.card (Fin k → B) ^ Fintype.card C := by
  have := grind_frac_bound Bad k 1 2 (fun c => by simpa using h c)
  simpa using this

omit [Fintype B] in
/-- Cost-weighted spot-checking: `I` is the set of unit-cost atoms and `π i` is the block containing
unit `i`. Uniform draws over `I` miss `Bad` with probability at most `(1 - num/den)^k` regardless of
how cost is distributed across blocks. -/
theorem cost_weighted_bound {I : Type*} [Fintype I] [DecidableEq I] (π : I → B) (Bad : Finset B)
    (k num den : ℕ)
    (h : num * Fintype.card I ≤ den * (Finset.univ.filter fun i => π i ∈ Bad).card) :
    (Finset.univ.filter fun q : Fin k → I => ∀ j, π (q j) ∉ Bad).card * den ^ k
      ≤ ((den - num) * Fintype.card I) ^ k := by
  have hb := frac_bad_bound (Finset.univ.filter fun i => π i ∈ Bad) k num den h
  have e : (Finset.univ.filter fun q : Fin k → I => ∀ j, π (q j) ∉ Bad)
      = missing (Finset.univ.filter fun i => π i ∈ Bad) k := by
    ext q; simp [missing]
  rw [e]; exact hb

omit [Fintype B] in
/-- Non-adaptive grinding bound for cost-weighted spot-checking across `|C|` candidate commitments. -/
theorem grind_cost_weighted_bound {I C : Type*} [Fintype I] [DecidableEq I]
    [Fintype C] [DecidableEq C] (π : I → B) (Bad : C → Finset B) (k num den : ℕ)
    (h : ∀ c, num * Fintype.card I ≤ den * (Finset.univ.filter fun i => π i ∈ Bad c).card) :
    (Finset.univ.filter fun O : C → Fin k → I => ∃ c, ∀ j, π (O c j) ∉ Bad c).card * den ^ k
      ≤ Fintype.card C * (den - num) ^ k * Fintype.card (Fin k → I) ^ Fintype.card C := by
  have hb := grind_frac_bound (fun c => Finset.univ.filter fun i => π i ∈ Bad c) k num den h
  have e : (Finset.univ.filter fun O : C → Fin k → I => ∃ c, ∀ j, π (O c j) ∉ Bad c)
      = (Finset.univ.filter fun O : C → Fin k → I =>
          ∃ c, O c ∈ missing (Finset.univ.filter fun i => π i ∈ Bad c) k) := by
    ext O; simp [missing]
  rw [e]; exact hb

end FlashProveNext

#print axioms FlashProveNext.card_missing
#print axioms FlashProveNext.frac_bad_bound
#print axioms FlashProveNext.half_bad_bound
#print axioms FlashProveNext.grind_frac_bound
#print axioms FlashProveNext.grind_bound
#print axioms FlashProveNext.cost_weighted_bound
#print axioms FlashProveNext.grind_cost_weighted_bound
