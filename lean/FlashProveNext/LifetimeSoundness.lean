/-
Multi-pool spot-check miss bounds (with and without replacement), finite seed-mapping bias,
adaptive lifetime union bound, and exact sample-count sufficiency (`specs/02-soundness.md` §4, §7).

Contains zero economic ("work paid") premises:
1. `two_pool_max`, `two_pool_all_good_le` : `max((1 - σ)^k₁, (1 - ρ)^k₂)` miss bound over cost-weighted pools;
2. `choose_mul_pow_le`, `hyper_miss_le`, `two_pool_wor_le` : hypergeometric without-replacement dominance;
3. `bucketCount_bounds`, `bucket_prob_close`, `pool_map_le`, `two_pool_map_le_max` :
   finite seed map `U ↦ ⌊U · C / 2^h⌋` bias `< 2^{-h}` per draw;
4. `lifetime_union`                       : `Δ + Q · ε` adaptive lazy-sampled random-oracle union bound;
5. `kSuff_iff_kReq_le`, `table_Q64_e40`   : closed-form sample count `⌈ln(Q / ε_sp) / (-ln(1 - s))⌉` and exact
                                            verification of all 16 parameter-table cells (including `k - 1` insufficiency).
-/
import Mathlib
import FlashProveNext.SpotCheck
import FlashProveNext.Grinding

namespace FlashProveNext

open Finset

set_option linter.unusedSectionVars false

theorem two_pool_max {α β : Type*} (S1 : Finset α) (S2 : Finset β) (A1 : Finset α) (A2 : Finset β)
    (h1 : A1 ⊆ S1) (h2 : A2 ⊆ S2) (m1 m2 : ℝ)
    (hbad : (A1.card : ℝ) ≤ m1 * S1.card ∨ (A2.card : ℝ) ≤ m2 * S2.card) :
    ((A1 ×ˢ A2).card : ℝ) ≤ max m1 m2 * (S1 ×ˢ S2).card := by
  rw [card_product, card_product]
  push_cast
  have c1 : (A1.card : ℝ) ≤ S1.card := by exact_mod_cast card_le_card h1
  have c2 : (A2.card : ℝ) ≤ S2.card := by exact_mod_cast card_le_card h2
  have p1 : (0:ℝ) ≤ A1.card := Nat.cast_nonneg _
  have p2 : (0:ℝ) ≤ A2.card := Nat.cast_nonneg _
  have q12 : (0:ℝ) ≤ (S1.card : ℝ) * S2.card := by positivity
  rcases hbad with h | h
  · calc (A1.card : ℝ) * A2.card ≤ (m1 * S1.card) * S2.card :=
          mul_le_mul h c2 p2 (le_trans p1 h)
      _ = m1 * (S1.card * S2.card) := by ring
      _ ≤ max m1 m2 * (S1.card * S2.card) := mul_le_mul_of_nonneg_right (le_max_left _ _) q12
      _ = _ := by ring
  · calc (A1.card : ℝ) * A2.card ≤ S1.card * (m2 * S2.card) :=
          mul_le_mul c1 h p2 (Nat.cast_nonneg _)
      _ = m2 * (S1.card * S2.card) := by ring
      _ ≤ max m1 m2 * (S1.card * S2.card) := mul_le_mul_of_nonneg_right (le_max_right _ _) q12
      _ = _ := by ring

theorem miss_card_le_real {U : Type*} [Fintype U] [DecidableEq U] (Bad : Finset U) (k : ℕ) (σ : ℝ)
    (h : σ * Fintype.card U ≤ Bad.card) :
    ((missing Bad k).card : ℝ) ≤ (1 - σ) ^ k * (Fintype.card U : ℝ) ^ k := by
  rw [card_missing, Nat.cast_pow, Nat.cast_sub (Finset.card_le_univ Bad), ← mul_pow]
  have hb : (Bad.card : ℝ) ≤ Fintype.card U := by exact_mod_cast Finset.card_le_univ Bad
  exact pow_le_pow_left₀ (by linarith) (by linarith) _

theorem two_pool_miss_le {IC IG : Type*} [Fintype IC] [DecidableEq IC] [Fintype IG] [DecidableEq IG]
    (BadC : Finset IC) (BadG : Finset IG) (k1 k2 : ℕ) (σ ρ : ℝ)
    (hbad : σ * Fintype.card IC ≤ BadC.card ∨ ρ * Fintype.card IG ≤ BadG.card) :
    ((missing BadC k1 ×ˢ missing BadG k2).card : ℝ)
      ≤ max ((1 - σ) ^ k1) ((1 - ρ) ^ k2) * Fintype.card ((Fin k1 → IC) × (Fin k2 → IG)) := by
  have e : Fintype.card ((Fin k1 → IC) × (Fin k2 → IG))
      = ((univ : Finset (Fin k1 → IC)) ×ˢ (univ : Finset (Fin k2 → IG))).card := by
    rw [card_product, card_univ, card_univ, Fintype.card_prod]
  rw [e]
  refine two_pool_max _ _ _ _ (subset_univ _) (subset_univ _) _ _ ?_
  have u1 : ((univ : Finset (Fin k1 → IC)).card : ℝ) = (Fintype.card IC : ℝ) ^ k1 := by simp
  have u2 : ((univ : Finset (Fin k2 → IG)).card : ℝ) = (Fintype.card IG : ℝ) ^ k2 := by simp
  rw [u1, u2]
  rcases hbad with h | h
  · exact Or.inl (miss_card_le_real BadC k1 σ h)
  · exact Or.inr (miss_card_le_real BadG k2 ρ h)

theorem two_pool_all_good_le {IC BC IG : Type*} [Fintype IC] [DecidableEq IC] [DecidableEq BC]
    [Fintype IG] [DecidableEq IG]
    (π : IC → BC) (BadBlk : Finset BC) (BadG : Finset IG) (k1 k2 : ℕ) (σ ρ : ℝ)
    (hbad : σ * Fintype.card IC ≤ (univ.filter fun i => π i ∈ BadBlk).card ∨
      ρ * Fintype.card IG ≤ BadG.card) :
    ((univ.filter fun x : (Fin k1 → IC) × (Fin k2 → IG) =>
        (∀ j, π (x.1 j) ∉ BadBlk) ∧ ∀ j, x.2 j ∉ BadG).card : ℝ)
      ≤ max ((1 - σ) ^ k1) ((1 - ρ) ^ k2) * Fintype.card ((Fin k1 → IC) × (Fin k2 → IG)) := by
  have e : (univ.filter fun x : (Fin k1 → IC) × (Fin k2 → IG) =>
        (∀ j, π (x.1 j) ∉ BadBlk) ∧ ∀ j, x.2 j ∉ BadG)
      = missing (univ.filter fun i => π i ∈ BadBlk) k1 ×ˢ missing BadG k2 := by
    ext x
    simp [missing, mem_product]
  rw [e]
  exact two_pool_miss_le _ _ k1 k2 σ ρ hbad

theorem choose_mul_pow_le (m n k : ℕ) (hmn : m ≤ n) :
    m.choose k * n ^ k ≤ n.choose k * m ^ k := by
  induction k with
  | zero => simp
  | succ k ih =>
    have key : (m - k) * n ≤ (n - k) * m := by
      rcases le_or_gt m k with h | h
      · simp [Nat.sub_eq_zero_of_le h]
      · obtain ⟨a, rfl⟩ := Nat.exists_eq_add_of_le h.le
        obtain ⟨b, rfl⟩ := Nat.exists_eq_add_of_le hmn
        have e1 : k + a - k = a := by omega
        have e2 : k + a + b - k = a + b := by omega
        rw [e1, e2]
        nlinarith [Nat.zero_le (b * k)]
    have hm := Nat.choose_succ_right_eq m k
    have hn := Nat.choose_succ_right_eq n k
    have lhs : m.choose (k + 1) * n ^ (k + 1) * (k + 1) = m.choose k * n ^ k * ((m - k) * n) := by
      calc m.choose (k + 1) * n ^ (k + 1) * (k + 1)
          = (m.choose (k + 1) * (k + 1)) * (n ^ k * n) := by ring
        _ = (m.choose k * (m - k)) * (n ^ k * n) := by rw [hm]
        _ = _ := by ring
    have rhs : n.choose (k + 1) * m ^ (k + 1) * (k + 1) = n.choose k * m ^ k * ((n - k) * m) := by
      calc n.choose (k + 1) * m ^ (k + 1) * (k + 1)
          = (n.choose (k + 1) * (k + 1)) * (m ^ k * m) := by ring
        _ = (n.choose k * (n - k)) * (m ^ k * m) := by rw [hn]
        _ = _ := by ring
    have h := Nat.mul_le_mul ih key
    rw [← lhs, ← rhs] at h
    exact Nat.le_of_mul_le_mul_right h (Nat.succ_pos k)

theorem hyper_miss_card {U : Type*} [Fintype U] [DecidableEq U] (Bad : Finset U) (k : ℕ) :
    (((univ : Finset U).powersetCard k).filter fun S => Disjoint S Bad).card
      = (Fintype.card U - Bad.card).choose k := by
  have e : ((univ : Finset U).powersetCard k).filter (fun S => Disjoint S Bad)
      = Badᶜ.powersetCard k := by
    ext S
    simp only [mem_filter, mem_powersetCard, subset_univ, true_and,
      Finset.subset_compl_iff_disjoint_right]
    tauto
  rw [e, card_powersetCard, card_compl]

theorem hyper_miss_le {U : Type*} [Fintype U] [DecidableEq U] [Nonempty U]
    (Bad : Finset U) (k : ℕ) (σ : ℝ) (h : σ * Fintype.card U ≤ Bad.card) :
    ((((univ : Finset U).powersetCard k).filter fun S => Disjoint S Bad).card : ℝ)
      ≤ (1 - σ) ^ k * ((univ : Finset U).powersetCard k).card := by
  rw [hyper_miss_card, card_powersetCard, card_univ]
  have hle : Bad.card ≤ Fintype.card U := Finset.card_le_univ Bad
  have hC : (0:ℝ) < Fintype.card U := Nat.cast_pos.mpr Fintype.card_pos
  have hn := choose_mul_pow_le (Fintype.card U - Bad.card) (Fintype.card U) k (Nat.sub_le _ _)
  have hn' : (((Fintype.card U - Bad.card).choose k : ℕ) : ℝ) * (Fintype.card U : ℝ) ^ k
      ≤ ((Fintype.card U).choose k : ℝ) * (((Fintype.card U - Bad.card : ℕ) : ℝ)) ^ k := by
    exact_mod_cast hn
  have hb : (Bad.card : ℝ) ≤ Fintype.card U := by exact_mod_cast hle
  have hgood : (((Fintype.card U - Bad.card : ℕ) : ℝ)) ^ k ≤ ((1 - σ) * Fintype.card U) ^ k := by
    rw [Nat.cast_sub hle]
    exact pow_le_pow_left₀ (by linarith) (by linarith) _
  have hCk : (0:ℝ) < (Fintype.card U : ℝ) ^ k := by positivity
  have hch : (0:ℝ) ≤ ((Fintype.card U).choose k : ℝ) := Nat.cast_nonneg _
  refine le_of_mul_le_mul_right ?_ hCk
  calc (((Fintype.card U - Bad.card).choose k : ℕ) : ℝ) * (Fintype.card U : ℝ) ^ k
      ≤ ((Fintype.card U).choose k : ℝ) * (((Fintype.card U - Bad.card : ℕ) : ℝ)) ^ k := hn'
    _ ≤ ((Fintype.card U).choose k : ℝ) * ((1 - σ) * Fintype.card U) ^ k :=
        mul_le_mul_of_nonneg_left hgood hch
    _ = (1 - σ) ^ k * ((Fintype.card U).choose k : ℝ) * (Fintype.card U : ℝ) ^ k := by
        rw [mul_pow]; ring

theorem two_pool_wor_le {IC IG : Type*} [Fintype IC] [DecidableEq IC] [Nonempty IC]
    [Fintype IG] [DecidableEq IG] [Nonempty IG]
    (BadC : Finset IC) (BadG : Finset IG) (k1 k2 : ℕ) (σ ρ : ℝ)
    (hbad : σ * Fintype.card IC ≤ BadC.card ∨ ρ * Fintype.card IG ≤ BadG.card) :
    (((((univ : Finset IC).powersetCard k1).filter fun S => Disjoint S BadC) ×ˢ
        (((univ : Finset IG).powersetCard k2).filter fun S => Disjoint S BadG)).card : ℝ)
      ≤ max ((1 - σ) ^ k1) ((1 - ρ) ^ k2) *
          (((univ : Finset IC).powersetCard k1) ×ˢ ((univ : Finset IG).powersetCard k2)).card := by
  refine two_pool_max _ _ _ _ (filter_subset _ _) (filter_subset _ _) _ _ ?_
  rcases hbad with h | h
  · exact Or.inl (hyper_miss_le BadC k1 σ h)
  · exact Or.inr (hyper_miss_le BadG k2 ρ h)

theorem card_filter_mul_lt (M C y : ℕ) (hC : 0 < C) :
    ((range M).filter fun U => U * C < y).card = min M ((y + C - 1) / C) := by
  have e : (range M).filter (fun U => U * C < y) = range (min M ((y + C - 1) / C)) := by
    ext U
    simp only [mem_filter, mem_range, lt_min_iff]
    have h3 : U + 1 ≤ (y + C - 1) / C ↔ (U + 1) * C ≤ y + C - 1 := Nat.le_div_iff_mul_le hC
    rw [Nat.add_mul, one_mul] at h3
    have key : U < (y + C - 1) / C ↔ U * C < y := by omega
    rw [key]
  rw [e, card_range]

theorem bucket_filter_eq (M C j : ℕ) (hM : 0 < M) :
    (range M).filter (fun U => U * C / M = j)
      = (range M).filter (fun U => U * C < (j + 1) * M) \
          (range M).filter (fun U => U * C < j * M) := by
  ext U
  simp only [mem_sdiff, mem_filter, mem_range]
  have h1 : j ≤ U * C / M ↔ j * M ≤ U * C := Nat.le_div_iff_mul_le hM
  have h2 : U * C / M < j + 1 ↔ U * C < (j + 1) * M := Nat.div_lt_iff_lt_mul hM
  omega

theorem ceil_diff_bounds (a M C : ℕ) (hC : 0 < C) :
    ((a + M) / C - a / C) * C < M + C ∧ M < ((a + M) / C - a / C) * C + C := by
  have h0 := Nat.div_add_mod a C
  have h1 := Nat.div_add_mod (a + M) C
  have r0 := Nat.mod_lt a hC
  have r1 := Nat.mod_lt (a + M) hC
  rw [Nat.sub_mul, Nat.mul_comm ((a + M) / C), Nat.mul_comm (a / C)]
  omega

def bucketCount (M C j : ℕ) : ℕ := ((range M).filter fun U => U * C / M = j).card

theorem bucketCount_bounds (M C j : ℕ) (hM : 0 < M) (hC : 0 < C) (hj : j < C) :
    bucketCount M C j * C < M + C ∧ M < bucketCount M C j * C + C := by
  have hsub : (range M).filter (fun U => U * C < j * M)
      ⊆ (range M).filter (fun U => U * C < (j + 1) * M) := by
    intro U hU
    simp only [mem_filter] at hU ⊢
    exact ⟨hU.1, lt_of_lt_of_le hU.2 (Nat.mul_le_mul_right _ (Nat.le_succ j))⟩
  have hcard := card_sdiff_add_card_eq_card hsub
  rw [← bucket_filter_eq M C j hM, card_filter_mul_lt M C _ hC, card_filter_mul_lt M C _ hC] at hcard
  have hle : ∀ y, y ≤ C * M → (y + C - 1) / C ≤ M := by
    intro y hy
    have : (y + C - 1) / C < M + 1 := by
      rw [Nat.div_lt_iff_lt_mul hC, Nat.add_mul, one_mul]
      have : C * M = M * C := Nat.mul_comm _ _
      omega
    omega
  have hjM : (j + 1) * M ≤ C * M := Nat.mul_le_mul_right _ hj
  have hjM' : j * M ≤ C * M := Nat.mul_le_mul_right _ hj.le
  rw [min_eq_right (hle _ hjM), min_eq_right (hle _ hjM')] at hcard
  have hsplit : (j + 1) * M + C - 1 = (j * M + C - 1) + M := by
    rw [Nat.add_mul, one_mul]; omega
  rw [hsplit] at hcard
  have hX : bucketCount M C j = (j * M + C - 1 + M) / C - (j * M + C - 1) / C := by
    unfold bucketCount; omega
  rw [hX]
  exact ceil_diff_bounds (j * M + C - 1) M C hC

theorem bucket_prob_close (M C j : ℕ) (hM : 0 < M) (hC : 0 < C) (hj : j < C) :
    |(bucketCount M C j : ℝ) / M - 1 / C| < 1 / M := by
  have h := bucketCount_bounds M C j hM hC hj
  have hMr : (0:ℝ) < M := by exact_mod_cast hM
  have hCr : (0:ℝ) < C := by exact_mod_cast hC
  have u : (bucketCount M C j : ℝ) * C < M + C := by exact_mod_cast h.1
  have l : (M : ℝ) < bucketCount M C j * C + C := by exact_mod_cast h.2
  rw [abs_sub_lt_iff]
  constructor
  · have e : 1 / (M:ℝ) - ((bucketCount M C j : ℝ) / M - 1 / C)
        = (M + C - bucketCount M C j * C) / (M * C) := by field_simp; ring
    have : 0 < ((M:ℝ) + C - bucketCount M C j * C) / (M * C) :=
      div_pos (by linarith) (by positivity)
    linarith
  · have e : 1 / (M:ℝ) - (1 / C - (bucketCount M C j : ℝ) / M)
        = (bucketCount M C j * C + C - M) / (M * C) := by field_simp; ring
    have : 0 < ((bucketCount M C j : ℝ) * C + C - M) / (M * C) :=
      div_pos (by linarith) (by positivity)
    linarith

theorem good_preimage_card (M C : ℕ) (hM : 0 < M) (hC : 0 < C) (Good : Finset ℕ)
    (hG : Good ⊆ range C) :
    ((range M).filter fun U => U * C / M ∈ Good).card * C ≤ Good.card * (M + C) := by
  have hsub : (range M).filter (fun U => U * C / M ∈ Good)
      ⊆ Good.biUnion fun j => (range M).filter fun U => U * C / M = j := by
    intro U hU
    simp only [mem_filter] at hU
    simp only [mem_biUnion, mem_filter]
    exact ⟨_, hU.2, hU.1, rfl⟩
  calc ((range M).filter fun U => U * C / M ∈ Good).card * C
      ≤ (∑ j ∈ Good, bucketCount M C j) * C :=
        Nat.mul_le_mul_right _ ((card_le_card hsub).trans card_biUnion_le)
    _ = ∑ j ∈ Good, bucketCount M C j * C := sum_mul ..
    _ ≤ ∑ _j ∈ Good, (M + C) :=
        sum_le_sum fun j hj => (bucketCount_bounds M C j hM hC (mem_range.mp (hG hj))).1.le
    _ = Good.card * (M + C) := by rw [sum_const, smul_eq_mul]

theorem fin_filter_card (M C : ℕ) (Good : Finset ℕ) :
    (univ.filter fun U : Fin M => (U : ℕ) * C / M ∈ Good).card
      = ((range M).filter fun U => U * C / M ∈ Good).card := by
  rw [card_filter, card_filter]
  exact Fin.sum_univ_eq_sum_range (fun i => if i * C / M ∈ Good then 1 else 0) M

theorem pow_le_pow_add_mul (p x y : ℝ) (k : ℕ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hx0 : 0 ≤ x)
    (hx1 : x ≤ 1) (hy : 0 ≤ y) (hpxy : p ≤ x + y) : p ^ k ≤ x ^ k + k * y := by
  induction k with
  | zero => simp
  | succ k ih =>
    have hxk : x ^ k ≤ 1 := pow_le_one₀ hx0 hx1
    have hxk0 : 0 ≤ x ^ k := pow_nonneg hx0 k
    have hky : 0 ≤ (k : ℝ) * y := by positivity
    calc p ^ (k + 1) = p * p ^ k := by ring
      _ ≤ p * (x ^ k + k * y) := mul_le_mul_of_nonneg_left ih hp0
      _ = p * x ^ k + p * (k * y) := by ring
      _ ≤ (x + y) * x ^ k + 1 * (k * y) :=
          add_le_add (mul_le_mul_of_nonneg_right hpxy hxk0) (mul_le_mul_of_nonneg_right hp1 hky)
      _ = x ^ (k + 1) + y * x ^ k + k * y := by ring
      _ ≤ x ^ (k + 1) + y + k * y := by linarith [mul_le_mul_of_nonneg_left hxk hy]
      _ = x ^ (k + 1) + ((k + 1 : ℕ) : ℝ) * y := by push_cast; ring

theorem pool_map_le (M C k : ℕ) (hM : 0 < M) (hC : 0 < C) (Good : Finset ℕ)
    (hG : Good ⊆ range C) (σ : ℝ) (hσ0 : 0 ≤ σ) (hσ1 : σ ≤ 1)
    (h : (Good.card : ℝ) ≤ (1 - σ) * C) :
    ((univ.filter fun U : Fin M => (U : ℕ) * C / M ∈ Good).card : ℝ) ^ k
      ≤ ((1 - σ) ^ k + k * ((C : ℝ) / M)) * (M : ℝ) ^ k := by
  have h1 := good_preimage_card M C hM hC Good hG
  rw [← fin_filter_card] at h1
  have hgM : (univ.filter fun U : Fin M => (U : ℕ) * C / M ∈ Good).card ≤ M :=
    (card_filter_le _ _).trans (le_of_eq (by simp))
  generalize (univ.filter fun U : Fin M => (U : ℕ) * C / M ∈ Good).card = g at h1 hgM ⊢
  have hMr : (0:ℝ) < M := by exact_mod_cast hM
  have hCr : (0:ℝ) < C := by exact_mod_cast hC
  have h1r : (g : ℝ) * C ≤ Good.card * (M + C) := by exact_mod_cast h1
  have hgMr : (g : ℝ) ≤ M := by exact_mod_cast hgM
  have h2 : (g : ℝ) * C ≤ ((1 - σ) * (M + C)) * C := by
    have := mul_le_mul_of_nonneg_right h (by positivity : (0:ℝ) ≤ M + C)
    nlinarith
  have h3 : (g : ℝ) ≤ (1 - σ) * (M + C) := le_of_mul_le_mul_right h2 hCr
  have hp : (g : ℝ) / M ≤ (1 - σ) + C / M := by
    rw [div_le_iff₀ hMr, add_mul, div_mul_cancel₀ _ hMr.ne']
    nlinarith [mul_nonneg hσ0 hCr.le]
  have hp1 : (g : ℝ) / M ≤ 1 := by rw [div_le_one hMr]; exact hgMr
  have key := pow_le_pow_add_mul ((g : ℝ) / M) (1 - σ) ((C : ℝ) / M) k (by positivity) hp1
    (by linarith) (by linarith) (by positivity) hp
  calc (g : ℝ) ^ k = ((g : ℝ) / M) ^ k * (M : ℝ) ^ k := by
        rw [div_pow, div_mul_cancel₀ _ (by positivity)]
    _ ≤ _ := mul_le_mul_of_nonneg_right key (by positivity)

theorem two_pool_map_le (M C N k1 k2 : ℕ) (hM : 0 < M) (hC : 0 < C) (hN : 0 < N)
    (GoodC GoodG : Finset ℕ) (hGC : GoodC ⊆ range C) (hGG : GoodG ⊆ range N)
    (σ ρ : ℝ) (hσ0 : 0 ≤ σ) (hσ1 : σ ≤ 1) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (hbad : (GoodC.card : ℝ) ≤ (1 - σ) * C ∨ (GoodG.card : ℝ) ≤ (1 - ρ) * N) :
    ((univ.filter fun x : (Fin k1 → Fin M) × (Fin k2 → Fin M) =>
        (∀ i, (x.1 i : ℕ) * C / M ∈ GoodC) ∧ ∀ i, (x.2 i : ℕ) * N / M ∈ GoodG).card : ℝ)
      ≤ (max ((1 - σ) ^ k1) ((1 - ρ) ^ k2) + (k1 * ((C : ℝ) / M) + k2 * ((N : ℝ) / M)))
          * Fintype.card ((Fin k1 → Fin M) × (Fin k2 → Fin M)) := by
  have e : (univ.filter fun x : (Fin k1 → Fin M) × (Fin k2 → Fin M) =>
        (∀ i, (x.1 i : ℕ) * C / M ∈ GoodC) ∧ ∀ i, (x.2 i : ℕ) * N / M ∈ GoodG)
      = Fintype.piFinset (fun _ : Fin k1 => univ.filter fun U : Fin M => (U : ℕ) * C / M ∈ GoodC)
        ×ˢ Fintype.piFinset (fun _ : Fin k2 => univ.filter fun U : Fin M => (U : ℕ) * N / M ∈ GoodG) := by
    ext x
    simp [Fintype.mem_piFinset, mem_product]
  have eU : Fintype.card ((Fin k1 → Fin M) × (Fin k2 → Fin M))
      = ((univ : Finset (Fin k1 → Fin M)) ×ˢ (univ : Finset (Fin k2 → Fin M))).card := by
    rw [card_product, card_univ, card_univ, Fintype.card_prod]
  rw [e, eU]
  have step := two_pool_max (univ : Finset (Fin k1 → Fin M)) (univ : Finset (Fin k2 → Fin M))
    (Fintype.piFinset (fun _ : Fin k1 => univ.filter fun U : Fin M => (U : ℕ) * C / M ∈ GoodC))
    (Fintype.piFinset (fun _ : Fin k2 => univ.filter fun U : Fin M => (U : ℕ) * N / M ∈ GoodG))
    (subset_univ _) (subset_univ _)
    ((1 - σ) ^ k1 + k1 * ((C : ℝ) / M)) ((1 - ρ) ^ k2 + k2 * ((N : ℝ) / M)) ?_
  · refine step.trans (mul_le_mul_of_nonneg_right ?_ (Nat.cast_nonneg _))
    have e1 : 0 ≤ (k1 : ℝ) * ((C : ℝ) / M) := by positivity
    have e2 : 0 ≤ (k2 : ℝ) * ((N : ℝ) / M) := by positivity
    apply max_le
    · linarith [le_max_left ((1 - σ) ^ k1) ((1 - ρ) ^ k2)]
    · linarith [le_max_right ((1 - σ) ^ k1) ((1 - ρ) ^ k2)]
  · have u1 : ((univ : Finset (Fin k1 → Fin M)).card : ℝ) = (M : ℝ) ^ k1 := by simp
    have u2 : ((univ : Finset (Fin k2 → Fin M)).card : ℝ) = (M : ℝ) ^ k2 := by simp
    rw [u1, u2, Fintype.card_piFinset, Fintype.card_piFinset, prod_const, prod_const,
      card_univ, card_univ, Fintype.card_fin, Fintype.card_fin]
    push_cast
    rcases hbad with h | h
    · exact Or.inl (pool_map_le M C k1 hM hC GoodC hGC σ hσ0 hσ1 h)
    · exact Or.inr (pool_map_le M N k2 hM hN GoodG hGG ρ hρ0 hρ1 h)

theorem two_pool_map_le_max (h C N Cmax Nmax k1 k2 : ℕ) (hC : 0 < C) (hN : 0 < N)
    (hCm : C ≤ Cmax) (hNm : N ≤ Nmax)
    (GoodC GoodG : Finset ℕ) (hGC : GoodC ⊆ range C) (hGG : GoodG ⊆ range N)
    (σ ρ : ℝ) (hσ0 : 0 ≤ σ) (hσ1 : σ ≤ 1) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1)
    (hbad : (GoodC.card : ℝ) ≤ (1 - σ) * C ∨ (GoodG.card : ℝ) ≤ (1 - ρ) * N) :
    ((univ.filter fun x : (Fin k1 → Fin (2 ^ h)) × (Fin k2 → Fin (2 ^ h)) =>
        (∀ i, (x.1 i : ℕ) * C / 2 ^ h ∈ GoodC) ∧ ∀ i, (x.2 i : ℕ) * N / 2 ^ h ∈ GoodG).card : ℝ)
      ≤ (max ((1 - σ) ^ k1) ((1 - ρ) ^ k2) + (k1 * (Cmax : ℝ) / 2 ^ h + k2 * (Nmax : ℝ) / 2 ^ h))
          * Fintype.card ((Fin k1 → Fin (2 ^ h)) × (Fin k2 → Fin (2 ^ h))) := by
  have base := two_pool_map_le (2 ^ h) C N k1 k2 (by positivity) hC hN GoodC GoodG hGC hGG
    σ ρ hσ0 hσ1 hρ0 hρ1 hbad
  refine base.trans (mul_le_mul_of_nonneg_right ?_ (Nat.cast_nonneg _))
  have hM : (0:ℝ) < 2 ^ h := by positivity
  have c1 : (C : ℝ) ≤ Cmax := by exact_mod_cast hCm
  have c2 : (N : ℝ) ≤ Nmax := by exact_mod_cast hNm
  have k1n : (0:ℝ) ≤ k1 := Nat.cast_nonneg _
  have k2n : (0:ℝ) ≤ k2 := Nat.cast_nonneg _
  have t1 : (k1 : ℝ) * ((C : ℝ) / ((2 ^ h : ℕ) : ℝ)) ≤ k1 * (Cmax : ℝ) / 2 ^ h := by
    rw [mul_div_assoc]; push_cast
    exact mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right c1 hM.le) k1n
  have t2 : (k2 : ℝ) * ((N : ℝ) / ((2 ^ h : ℕ) : ℝ)) ≤ k2 * (Nmax : ℝ) / 2 ^ h := by
    rw [mul_div_assoc]; push_cast
    exact mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right c2 hM.le) k2n
  linarith

open Classical in
/-- Adaptive lifetime union bound under lazy-sampled random oracle queries (`Δ + Q · ε`). -/
theorem lifetime_union {Ω R : Type*} [Fintype Ω] [DecidableEq Ω] [Nonempty Ω] {Q : ℕ}
    (cand : Fin Q → (Fin Q → Ω) → R)
    (hcand : ∀ j a x, cand j (Function.update a j x) = cand j a)
    (BadCand : R → Prop) (Acc : R → Finset Ω) (ε Δ : ℝ) (hε0 : 0 ≤ ε)
    (hAcc : ∀ r, BadCand r → ((Acc r).card : ℝ) ≤ ε * Fintype.card Ω)
    (Fail : Finset (Fin Q → Ω)) (hFail : (Fail.card : ℝ) ≤ Δ * (Fintype.card Ω : ℝ) ^ Q) :
    ((Fail ∪ univ.filter fun a : Fin Q → Ω =>
        ∃ j, BadCand (cand j a) ∧ a j ∈ Acc (cand j a)).card : ℝ)
      ≤ (Δ + Q * ε) * (Fintype.card Ω : ℝ) ^ Q := by
  let Acc' : R → Finset Ω := fun r => if BadCand r then Acc r else ∅
  have hAcc' : ∀ r, ((Acc' r).card : ℝ) ≤ ε * Fintype.card Ω := by
    intro r
    by_cases hr : BadCand r
    · simp only [Acc', if_pos hr]; exact hAcc r hr
    · simp only [Acc', if_neg hr, card_empty, Nat.cast_zero]; positivity
  have hg := adaptive_grind_real cand hcand Acc' ε hε0 hAcc'
  have hsub : (univ.filter fun a : Fin Q → Ω => ∃ j, BadCand (cand j a) ∧ a j ∈ Acc (cand j a))
      ⊆ univ.filter fun a : Fin Q → Ω => ∃ j, a j ∈ Acc' (cand j a) := by
    intro a ha
    simp only [mem_filter, mem_univ, true_and] at ha ⊢
    obtain ⟨j, hb, hj⟩ := ha
    exact ⟨j, by simp only [Acc', if_pos hb]; exact hj⟩
  calc ((Fail ∪ univ.filter fun a : Fin Q → Ω =>
          ∃ j, BadCand (cand j a) ∧ a j ∈ Acc (cand j a)).card : ℝ)
      ≤ (Fail.card : ℝ) + (univ.filter fun a : Fin Q → Ω =>
          ∃ j, BadCand (cand j a) ∧ a j ∈ Acc (cand j a)).card := by
        exact_mod_cast card_union_le _ _
    _ ≤ Δ * (Fintype.card Ω : ℝ) ^ Q + Q * ε * (Fintype.card Ω : ℝ) ^ Q :=
        add_le_add hFail ((Nat.cast_le.mpr (card_le_card hsub)).trans hg)
    _ = (Δ + Q * ε) * (Fintype.card Ω : ℝ) ^ Q := by ring

set_option exponentiation.threshold 1024

def KSuff (Q εsp s : ℝ) (k : ℕ) : Prop := Q * (1 - s) ^ k ≤ εsp

noncomputable def kReq (Q εsp s : ℝ) : ℕ := ⌈Real.log (Q / εsp) / -Real.log (1 - s)⌉₊

theorem kSuff_iff_kReq_le (Q εsp s : ℝ) (k : ℕ) (hQ : 0 < Q) (hε : 0 < εsp) (hs0 : 0 < s)
    (hs1 : s < 1) : KSuff Q εsp s k ↔ kReq Q εsp s ≤ k := by
  have h1s : 0 < 1 - s := by linarith
  have hL : 0 < -Real.log (1 - s) := by
    have := Real.log_neg h1s (by linarith); linarith
  unfold KSuff kReq
  rw [Nat.ceil_le, div_le_iff₀ hL,
    ← Real.log_le_log_iff (x := Q * (1 - s) ^ k) (y := εsp) (by positivity) hε,
    Real.log_mul hQ.ne' (by positivity), Real.log_pow, Real.log_div hQ.ne' hε.ne']
  constructor <;> intro h <;> linarith

theorem kReq_eq (Q εsp s : ℝ) (n : ℕ) (hQ : 0 < Q) (hε : 0 < εsp) (hs0 : 0 < s) (hs1 : s < 1)
    (h1 : KSuff Q εsp s (n + 1)) (h0 : ¬ KSuff Q εsp s n) : kReq Q εsp s = n + 1 := by
  have a := (kSuff_iff_kReq_le Q εsp s (n + 1) hQ hε hs0 hs1).mp h1
  have b : ¬ kReq Q εsp s ≤ n := fun hb => h0 ((kSuff_iff_kReq_le Q εsp s n hQ hε hs0 hs1).mpr hb)
  omega

theorem kSuff_pow2_iff (q t a b n : ℕ) (s : ℝ) (hb : 0 < b) (hs : 1 - s = (a : ℝ) / b) :
    KSuff (2 ^ q) ((2 ^ t)⁻¹ / 2) s n ↔ 2 ^ (q + t + 1) * a ^ n ≤ b ^ n := by
  unfold KSuff
  have hb' : (0:ℝ) < b := by exact_mod_cast hb
  rw [hs, div_pow, ← mul_div_assoc,
    show ((2:ℝ) ^ t)⁻¹ / 2 = 1 / (2 ^ t * 2) by rw [one_div, mul_inv, div_eq_mul_inv],
    div_le_div_iff₀ (by positivity) (by positivity), one_mul, ← Nat.cast_le (α := ℝ)]
  push_cast
  have e : (2:ℝ) ^ (q + t + 1) * (a : ℝ) ^ n = 2 ^ q * (a : ℝ) ^ n * (2 ^ t * 2) := by ring
  rw [e]

theorem kReq_eq_pow2 (q t a b n : ℕ) (s : ℝ) (hb : 0 < b) (hs0 : 0 < s) (hs1 : s < 1)
    (hs : 1 - s = (a : ℝ) / b)
    (h1 : 2 ^ (q + t + 1) * a ^ (n + 1) ≤ b ^ (n + 1)) (h0 : ¬ 2 ^ (q + t + 1) * a ^ n ≤ b ^ n) :
    kReq (2 ^ q) ((2 ^ t)⁻¹ / 2) s = n + 1 :=
  kReq_eq _ _ _ n (by positivity) (by positivity) hs0 hs1
    ((kSuff_pow2_iff q t a b (n + 1) s hb hs).mpr h1)
    (fun h => h0 ((kSuff_pow2_iff q t a b n s hb hs).mp h))

theorem k105_sufficient : (2:ℝ) ^ 64 * (1 / 2) ^ 105 ≤ (2 ^ 41)⁻¹ := by norm_num

theorem k104_insufficient : ¬ (2:ℝ) ^ 64 * (1 / 2) ^ 104 ≤ (2 ^ 41)⁻¹ := by norm_num

theorem kReq_ideal_104 : kReq (2 ^ 64) (2 ^ 40)⁻¹ (1 / 2) = 104 :=
  kReq_eq _ _ _ 103 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [KSuff]) (by norm_num [KSuff])

theorem table_Q30_e20 :
    kReq (2 ^ 30) ((2 ^ 20)⁻¹ / 2) (1 / 2) = 51 ∧ kReq (2 ^ 30) ((2 ^ 20)⁻¹ / 2) (3 / 10) = 100 ∧
    kReq (2 ^ 30) ((2 ^ 20)⁻¹ / 2) (1 / 5) = 159 ∧ kReq (2 ^ 30) ((2 ^ 20)⁻¹ / 2) (1 / 10) = 336 := by
  exact ⟨kReq_eq_pow2 30 20 1 2 50 (1 / 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 30 20 7 10 99 (3 / 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 30 20 4 5 158 (1 / 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 30 20 9 10 335 (1 / 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)⟩

theorem table_Q30_e40 :
    kReq (2 ^ 30) ((2 ^ 40)⁻¹ / 2) (1 / 2) = 71 ∧ kReq (2 ^ 30) ((2 ^ 40)⁻¹ / 2) (3 / 10) = 138 ∧
    kReq (2 ^ 30) ((2 ^ 40)⁻¹ / 2) (1 / 5) = 221 ∧ kReq (2 ^ 30) ((2 ^ 40)⁻¹ / 2) (1 / 10) = 468 := by
  exact ⟨kReq_eq_pow2 30 40 1 2 70 (1 / 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 30 40 7 10 137 (3 / 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 30 40 4 5 220 (1 / 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 30 40 9 10 467 (1 / 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)⟩

theorem table_Q64_e20 :
    kReq (2 ^ 64) ((2 ^ 20)⁻¹ / 2) (1 / 2) = 85 ∧ kReq (2 ^ 64) ((2 ^ 20)⁻¹ / 2) (3 / 10) = 166 ∧
    kReq (2 ^ 64) ((2 ^ 20)⁻¹ / 2) (1 / 5) = 265 ∧ kReq (2 ^ 64) ((2 ^ 20)⁻¹ / 2) (1 / 10) = 560 := by
  exact ⟨kReq_eq_pow2 64 20 1 2 84 (1 / 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 64 20 7 10 165 (3 / 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 64 20 4 5 264 (1 / 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 64 20 9 10 559 (1 / 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)⟩

theorem table_Q64_e40 :
    kReq (2 ^ 64) ((2 ^ 40)⁻¹ / 2) (1 / 2) = 105 ∧ kReq (2 ^ 64) ((2 ^ 40)⁻¹ / 2) (3 / 10) = 205 ∧
    kReq (2 ^ 64) ((2 ^ 40)⁻¹ / 2) (1 / 5) = 327 ∧ kReq (2 ^ 64) ((2 ^ 40)⁻¹ / 2) (1 / 10) = 691 := by
  exact ⟨kReq_eq_pow2 64 40 1 2 104 (1 / 2) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 64 40 7 10 204 (3 / 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 64 40 4 5 326 (1 / 5) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num),
    kReq_eq_pow2 64 40 9 10 690 (1 / 10) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (by norm_num) (by norm_num)⟩

theorem total_le_of_kSuff (Q ε εsp σ ρ a Δ : ℝ) (k1 k2 : ℕ) (hQ : 0 ≤ Q)
    (h1 : KSuff Q εsp σ k1) (h2 : KSuff Q εsp ρ k2) (haux : Δ + Q * a ≤ ε - εsp) :
    Δ + Q * (max ((1 - σ) ^ k1) ((1 - ρ) ^ k2) + a) ≤ ε := by
  have : Q * max ((1 - σ) ^ k1) ((1 - ρ) ^ k2) ≤ εsp := by
    rw [mul_max_of_nonneg _ _ hQ]; exact max_le h1 h2
  linarith [mul_add Q (max ((1 - σ) ^ k1) ((1 - ρ) ^ k2)) a]

theorem default_budget (σ ρ a Δ : ℝ) (hσ0 : 1 / 2 ≤ σ) (hσ1 : σ ≤ 1) (hρ0 : 1 / 2 ≤ ρ)
    (hρ1 : ρ ≤ 1) (hΔ : Δ ≤ (2 ^ 42)⁻¹) (ha : a ≤ (2 ^ 106)⁻¹) :
    Δ + 2 ^ 64 * (max ((1 - σ) ^ 105) ((1 - ρ) ^ 105) + a) ≤ (2 ^ 40)⁻¹ := by
  have h1 : (1 - σ) ^ 105 ≤ (1 / 2 : ℝ) ^ 105 := pow_le_pow_left₀ (by linarith) (by linarith) _
  have h2 : (1 - ρ) ^ 105 ≤ (1 / 2 : ℝ) ^ 105 := pow_le_pow_left₀ (by linarith) (by linarith) _
  have hm := max_le h1 h2
  have hb := mul_le_mul_of_nonneg_left (add_le_add hm ha) (by norm_num : (0:ℝ) ≤ 2 ^ 64)
  have e : (2:ℝ) ^ 64 * ((1 / 2) ^ 105 + (2 ^ 106)⁻¹) + (2 ^ 42)⁻¹ = (2 ^ 40)⁻¹ := by norm_num
  linarith

end FlashProveNext

#print axioms FlashProveNext.two_pool_max
#print axioms FlashProveNext.miss_card_le_real
#print axioms FlashProveNext.two_pool_miss_le
#print axioms FlashProveNext.two_pool_all_good_le
#print axioms FlashProveNext.choose_mul_pow_le
#print axioms FlashProveNext.hyper_miss_card
#print axioms FlashProveNext.hyper_miss_le
#print axioms FlashProveNext.two_pool_wor_le
#print axioms FlashProveNext.card_filter_mul_lt
#print axioms FlashProveNext.bucket_filter_eq
#print axioms FlashProveNext.ceil_diff_bounds
#print axioms FlashProveNext.bucketCount_bounds
#print axioms FlashProveNext.bucket_prob_close
#print axioms FlashProveNext.good_preimage_card
#print axioms FlashProveNext.fin_filter_card
#print axioms FlashProveNext.pow_le_pow_add_mul
#print axioms FlashProveNext.pool_map_le
#print axioms FlashProveNext.two_pool_map_le
#print axioms FlashProveNext.two_pool_map_le_max
#print axioms FlashProveNext.lifetime_union
#print axioms FlashProveNext.kSuff_iff_kReq_le
#print axioms FlashProveNext.kReq_eq
#print axioms FlashProveNext.kSuff_pow2_iff
#print axioms FlashProveNext.kReq_eq_pow2
#print axioms FlashProveNext.k105_sufficient
#print axioms FlashProveNext.k104_insufficient
#print axioms FlashProveNext.kReq_ideal_104
#print axioms FlashProveNext.table_Q30_e20
#print axioms FlashProveNext.table_Q30_e40
#print axioms FlashProveNext.table_Q64_e20
#print axioms FlashProveNext.table_Q64_e40
#print axioms FlashProveNext.total_le_of_kSuff
#print axioms FlashProveNext.default_budget
