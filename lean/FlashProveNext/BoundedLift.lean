/-
Bounded integer modular lift uniqueness and inner-product / rescaling overflow bounds.

* `signedLift_unique_of_bounded` : if `a ≡ s [ZMOD p]` with `|a| < B`, `|s| < B`, and `2 B ≤ p`, then `a = s` over `ℤ`
                                   (no modular wrap-around in Goldilocks field arithmetic);
* `inner_prod_abs_le`            : `ℓ_1` bound `|∑_{i < D} w_i m_i| ≤ D · (W · M)` on bounded dot products;
* `ffn_per_slot_product_lt`      : per-slot FFN rescaled product bound `|(a · w) · m| < 2^49`.
-/
import Mathlib

open scoped BigOperators

namespace FlashProveNext

/-- Canonical symmetric lift uniqueness modulo `p` when both operands lie in `(-B, B)` and `2B ≤ p`. -/
theorem signedLift_unique_of_bounded
    {a s B p : ℤ}
    (hmod : a ≡ s [ZMOD p])
    (ha : |a| < B)
    (hs : |s| < B)
    (hp : 2 * B ≤ p) :
    a = s := by
  rw [Int.modEq_iff_dvd] at hmod
  obtain ⟨k, hk⟩ := hmod
  have hdiff : |s - a| < p := by
    have h1 : |s - a| ≤ |s| + |a| := abs_sub s a
    linarith
  rw [hk, abs_mul] at hdiff
  have hp_pos : 0 < p := by
    have hB : 0 < B := lt_of_le_of_lt (abs_nonneg a) ha
    linarith
  have habs_p : |p| = p := abs_of_pos hp_pos
  rw [habs_p] at hdiff
  have hk0 : |k| < 1 := by
    by_contra hge
    have hge1 : 1 ≤ |k| := by omega
    have hmul : p * 1 ≤ p * |k| := mul_le_mul_of_nonneg_left hge1 (le_of_lt hp_pos)
    linarith
  have hk_zero : k = 0 := by
    have hkn : 0 ≤ |k| := abs_nonneg k
    have : |k| = 0 := by omega
    exact abs_eq_zero.mp this
  rw [hk_zero, mul_zero] at hk
  linarith

/-- Magnitude bound on a `D`-dimensional integer inner product. -/
theorem inner_prod_abs_le
    {D : ℕ} (w m : Fin D → ℤ) {W M : ℤ}
    (hW : 0 ≤ W)
    (hw : ∀ i, |w i| ≤ W)
    (hm : ∀ i, |m i| ≤ M) :
    |∑ i : Fin D, w i * m i| ≤ (D : ℤ) * (W * M) := by
  calc
    |∑ i : Fin D, w i * m i| ≤ ∑ i : Fin D, |w i * m i| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin D, W * M := by
      apply Finset.sum_le_sum
      intro i _
      rw [abs_mul]
      exact mul_le_mul (hw i) (hm i) (abs_nonneg _) hW
    _ = (D : ℤ) * (W * M) := by
      simp [Finset.card_univ]

/-- Per-slot rescaled product bound `< 2^49` (well below half the Goldilocks modulus `p / 2 > 2^62`). -/
theorem ffn_per_slot_product_lt
    (a w m : ℤ)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1)
    (hw0 : 0 ≤ w) (hw1 : w < (2 : ℤ) ^ 25)
    (hm : |m| < (2 : ℤ) ^ 24) :
    |(a * w) * m| < (2 : ℤ) ^ 49 := by
  have haw0 : 0 ≤ a * w := mul_nonneg ha0 hw0
  have haw1 : a * w < (2 : ℤ) ^ 25 := by
    have hle : a * w ≤ 1 * w := mul_le_mul_of_nonneg_right ha1 hw0
    linarith
  rw [abs_mul, abs_of_nonneg haw0]
  calc
    (a * w) * |m| ≤ ((2 : ℤ) ^ 25 - 1) * ((2 : ℤ) ^ 24 - 1) := by
      have h1 : a * w ≤ (2 : ℤ) ^ 25 - 1 := by omega
      have h2 : |m| ≤ (2 : ℤ) ^ 24 - 1 := by omega
      exact mul_le_mul h1 h2 (abs_nonneg m) (by norm_num)
    _ < (2 : ℤ) ^ 49 := by norm_num

end FlashProveNext

#print axioms FlashProveNext.signedLift_unique_of_bounded
#print axioms FlashProveNext.inner_prod_abs_le
#print axioms FlashProveNext.ffn_per_slot_product_lt
