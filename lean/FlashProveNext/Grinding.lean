/-
Adaptive random-oracle grinding under lazy sampling.

Each distinct first query `j < n` draws an independent uniform oracle answer `a j : Q` while the
adversary's query candidate `cand j a` depends only on answers outside coordinate `j` (`hcand`).

* `adaptive_grind`      : integer cardinality bound `n · E · |Q|^{n-1}` when each accepting set has size `≤ E`;
* `adaptive_grind_real` : real-valued fraction bound `n · ε · |Q|^n` when each accepting set has density `≤ ε`;
* `prod_frac_le`        : product-space density bound `(τ · ε)` on `P × Q'`.
-/
import Mathlib
import FlashProveNext.Sumcheck

namespace FlashProveNext

open Finset

set_option linter.unusedSectionVars false

theorem adaptive_grind {Q C : Type*} [Fintype Q] [DecidableEq Q] {n : ℕ}
    (cand : Fin n → (Fin n → Q) → C)
    (hcand : ∀ j a x, cand j (Function.update a j x) = cand j a)
    (Acc : C → Finset Q) (E : ℕ) (hE : ∀ c, (Acc c).card ≤ E) :
    (Finset.univ.filter fun a : Fin n → Q => ∃ j, a j ∈ Acc (cand j a)).card * Fintype.card Q
      ≤ n * E * Fintype.card Q ^ n :=
  sumcheck_bad_challenges (fun j a => Acc (cand j a)) E
    (fun j a x => by simp only [hcand]) (fun j a => hE _)

theorem adaptive_grind_real {Q C : Type*} [Fintype Q] [DecidableEq Q] [Nonempty Q] {n : ℕ}
    (cand : Fin n → (Fin n → Q) → C)
    (hcand : ∀ j a x, cand j (Function.update a j x) = cand j a)
    (Acc : C → Finset Q) (ε : ℝ) (hε0 : 0 ≤ ε)
    (hε : ∀ c, ((Acc c).card : ℝ) ≤ ε * Fintype.card Q) :
    ((Finset.univ.filter fun a : Fin n → Q => ∃ j, a j ∈ Acc (cand j a)).card : ℝ)
      ≤ n * ε * (Fintype.card Q : ℝ) ^ n := by
  have hE : ∀ c, (Acc c).card ≤ ⌊ε * Fintype.card Q⌋₊ := fun c => Nat.le_floor (hε c)
  have h := adaptive_grind cand hcand Acc _ hE
  have hpos : (0 : ℝ) < Fintype.card Q := Nat.cast_pos.mpr Fintype.card_pos
  have h' : ((Finset.univ.filter fun a : Fin n → Q => ∃ j, a j ∈ Acc (cand j a)).card : ℝ)
      * Fintype.card Q ≤ n * (⌊ε * Fintype.card Q⌋₊ : ℝ) * (Fintype.card Q : ℝ) ^ n := by
    exact_mod_cast h
  have hf : (⌊ε * Fintype.card Q⌋₊ : ℝ) ≤ ε * Fintype.card Q := Nat.floor_le (by positivity)
  have h2 : ((Finset.univ.filter fun a : Fin n → Q => ∃ j, a j ∈ Acc (cand j a)).card : ℝ)
      * Fintype.card Q ≤ (n * ε * (Fintype.card Q : ℝ) ^ n) * Fintype.card Q := by
    refine h'.trans ?_
    have : (0:ℝ) ≤ (n : ℝ) * (Fintype.card Q : ℝ) ^ n := by positivity
    nlinarith [mul_le_mul_of_nonneg_left hf this]
  exact le_of_mul_le_mul_right h2 hpos

theorem prod_frac_le {P Q' : Type*} [Fintype P] [Fintype Q'] (T : Finset P) (A : Finset Q')
    (τ ε : ℝ) (hτ0 : 0 ≤ τ) (hT : (T.card : ℝ) ≤ τ * Fintype.card P)
    (hA : (A.card : ℝ) ≤ ε * Fintype.card Q') :
    ((T ×ˢ A).card : ℝ) ≤ (τ * ε) * Fintype.card (P × Q') := by
  rw [Finset.card_product, Fintype.card_prod]
  push_cast
  have h0 : (0:ℝ) ≤ ε * Fintype.card Q' := le_trans (Nat.cast_nonneg _) hA
  calc (T.card : ℝ) * A.card ≤ (τ * Fintype.card P) * (ε * Fintype.card Q') :=
        mul_le_mul hT hA (Nat.cast_nonneg _) (by positivity)
    _ = _ := by ring

end FlashProveNext

#print axioms FlashProveNext.adaptive_grind
#print axioms FlashProveNext.adaptive_grind_real
#print axioms FlashProveNext.prod_frac_le
