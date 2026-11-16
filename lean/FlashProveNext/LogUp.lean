/-
Algebraic soundness and completeness of LogUp (logarithmic-derivative lookups) and merged-domain
zero-sum ZK-LogUp.

* `logup_complete`                 : honest lookup multiplicities satisfy the harmonic identity for every `λ`;
* `logup_bad_card_le`              : any false lookup claim (out-of-table value or wrong multiplicity, `nv < char F`)
                                     satisfies the harmonic identity on at most `nv + nt` non-pole points `λ ∈ F`;
* `logup_sound_prob`               : probability bound `(nv + nt) / |F|`;
* `pole_card_le`                   : at most `nv + nt` poles exist;
* `mergedZeroSum_iff`              : merged-domain zero-sum form (public target `0`, no leaked partial sum) is
                                     algebraically equivalent to harmonic equality;
* `fingerprint_collision_card_le`  : pairwise Schwartz–Zippel bound for `w`-column polynomial fingerprints
                                     (full table-wide multi-column LogUp soundness is proved in `RoundByRound.lean`).
-/
import Mathlib

namespace FlashProveNext

open Finset Polynomial

set_option linter.unusedSectionVars false

section logup

variable {F : Type*} [Field F] [DecidableEq F] {nv nt : ℕ}

/-- Multiplicity of `x` in `a`. -/
def cnt (a : Fin nv → F) (x : F) : ℕ := (Finset.univ.filter fun j => a j = x).card

/-- Honest lookup claim: every query value appears in `t` and multiplicities match in `F`. -/
def LookupValid (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F) : Prop :=
  (∀ j, ∃ i, a j = t i) ∧ ∀ i, (cnt a (t i) : F) = m i

/-- Harmonic-sum identity `∑_j (λ + a_j)⁻¹ = ∑_i m_i (λ + t_i)⁻¹`. -/
def Harmonic (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F) (l : F) : Prop :=
  ∑ j, (l + a j)⁻¹ = ∑ i, m i * (l + t i)⁻¹

/-- `λ` avoids all denominators (`λ + a_j ≠ 0` and `λ + t_i ≠ 0`). -/
def NonPole (a : Fin nv → F) (t : Fin nt → F) (l : F) : Prop :=
  (∀ j, l + a j ≠ 0) ∧ ∀ i, l + t i ≠ 0

/-- Distinct values occurring across `a` and `t`. -/
def vals (a : Fin nv → F) (t : Fin nt → F) : Finset F :=
  Finset.univ.image a ∪ Finset.univ.image t

/-- Grouped net multiplicity `c_x = true count − claimed multiplicity`. -/
def coef (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F) (x : F) : F :=
  (cnt a x : F) - ∑ i ∈ Finset.univ.filter (fun i => t i = x), m i

theorem vals_card_le (a : Fin nv → F) (t : Fin nt → F) : (vals a t).card ≤ nv + nt := by
  unfold vals
  refine (Finset.card_union_le _ _).trans (add_le_add ?_ ?_)
  · simpa using (Finset.card_image_le : (Finset.univ.image a).card ≤ (Finset.univ : Finset (Fin nv)).card)
  · simpa using (Finset.card_image_le : (Finset.univ.image t).card ≤ (Finset.univ : Finset (Fin nt)).card)

theorem mem_vals_a (a : Fin nv → F) (t : Fin nt → F) (j : Fin nv) : a j ∈ vals a t := by
  simp [vals]

theorem mem_vals_t (a : Fin nv → F) (t : Fin nt → F) (i : Fin nt) : t i ∈ vals a t := by
  simp [vals]

theorem harmonic_diff_eq (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F) (l : F) :
    (∑ j, (l + a j)⁻¹) - ∑ i, m i * (l + t i)⁻¹
      = ∑ x ∈ vals a t, coef a t m x * (l + x)⁻¹ := by
  have h1 : ∑ j, (l + a j)⁻¹ = ∑ x ∈ vals a t, (cnt a x : F) * (l + x)⁻¹ := by
    rw [← Finset.sum_fiberwise_of_maps_to (s := Finset.univ) (t := vals a t) (g := a)
      (fun j _ => mem_vals_a a t j)]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [Finset.sum_congr rfl (g := fun _ => (l + x)⁻¹) (fun j hj => by
      simp only [Finset.mem_filter] at hj; rw [hj.2])]
    simp [cnt, Finset.sum_const, nsmul_eq_mul]
  have h2 : ∑ i, m i * (l + t i)⁻¹
      = ∑ x ∈ vals a t, (∑ i ∈ Finset.univ.filter (fun i => t i = x), m i) * (l + x)⁻¹ := by
    rw [← Finset.sum_fiberwise_of_maps_to (s := Finset.univ) (t := vals a t) (g := t)
      (fun i _ => mem_vals_t a t i)]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun i hi => ?_
    simp only [Finset.mem_filter] at hi
    rw [hi.2]
  rw [h1, h2, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun x _ => ?_
  simp only [coef, sub_mul]

/-- Cleared-denominator polynomial `N(X) = ∑_x c_x ∏_{y ≠ x} (X + y)`. -/
noncomputable def numer (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F) : F[X] :=
  ∑ x ∈ vals a t, C (coef a t m x) * ∏ y ∈ (vals a t).erase x, (X + C y)

theorem numer_natDegree_le (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F) :
    (numer a t m).natDegree ≤ (vals a t).card := by
  unfold numer
  refine Polynomial.natDegree_sum_le_of_forall_le _ _ fun x _ => ?_
  calc (C (coef a t m x) * ∏ y ∈ (vals a t).erase x, (X + C y)).natDegree
      ≤ (∏ y ∈ (vals a t).erase x, (X + C y)).natDegree := Polynomial.natDegree_C_mul_le _ _
    _ ≤ ∑ y ∈ (vals a t).erase x, (X + C y).natDegree := Polynomial.natDegree_prod_le _ _
    _ = ((vals a t).erase x).card := by simp
    _ ≤ (vals a t).card := Finset.card_erase_le

theorem numer_eval (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F) (l : F) :
    (numer a t m).eval l = ∑ x ∈ vals a t, coef a t m x * ∏ y ∈ (vals a t).erase x, (l + y) := by
  simp [numer, Polynomial.eval_finset_sum, Polynomial.eval_prod]

theorem numer_eval_nonpole (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F) (l : F)
    (hl : NonPole a t l) :
    (numer a t m).eval l
      = (∏ y ∈ vals a t, (l + y)) * ∑ x ∈ vals a t, coef a t m x * (l + x)⁻¹ := by
  rw [numer_eval, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x hx => ?_
  have hx0 : l + x ≠ 0 := by
    unfold vals at hx
    simp only [Finset.mem_union, Finset.mem_image, Finset.mem_univ, true_and] at hx
    rcases hx with ⟨j, rfl⟩ | ⟨i, rfl⟩
    · exact hl.1 j
    · exact hl.2 i
  have := Finset.mul_prod_erase (vals a t) (fun y => l + y) hx
  rw [← this]
  field_simp

theorem numer_ne_zero (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F) {x0 : F}
    (hx0 : x0 ∈ vals a t) (hc : coef a t m x0 ≠ 0) : numer a t m ≠ 0 := by
  intro h0
  have h := congrArg (Polynomial.eval (-x0)) h0
  rw [numer_eval, Polynomial.eval_zero] at h
  rw [Finset.sum_eq_single x0] at h
  · refine mul_ne_zero hc ?_ h
    rw [Finset.prod_ne_zero_iff]
    intro y hy
    have hy' := (Finset.mem_erase.mp hy).1
    intro hz
    exact hy' (neg_add_eq_zero.mp hz).symm
  · intro x _ hxne
    have : x0 ∈ (vals a t).erase x := Finset.mem_erase.mpr ⟨fun e => hxne e.symm, hx0⟩
    rw [Finset.prod_eq_zero this (by simp), mul_zero]
  · intro hn; exact absurd hx0 hn

theorem logup_complete (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F)
    (ht : Function.Injective t) (hv : LookupValid a t m) (l : F) : Harmonic a t m l := by
  unfold Harmonic
  rw [← sub_eq_zero, harmonic_diff_eq]
  refine Finset.sum_eq_zero fun x _ => ?_
  have : coef a t m x = 0 := by
    unfold coef
    by_cases hx : ∃ i, t i = x
    · obtain ⟨i, rfl⟩ := hx
      have e : Finset.univ.filter (fun i' => t i' = t i) = {i} := by
        ext i'; simp [ht.eq_iff]
      rw [e, Finset.sum_singleton, hv.2 i, sub_self]
    · push Not at hx
      have e : Finset.univ.filter (fun i' => t i' = x) = ∅ := by
        ext i'; simp [hx i']
      have e0 : cnt a x = 0 := by
        unfold cnt
        rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
        intro j _ hj
        obtain ⟨i, hi⟩ := hv.1 j
        exact hx i (hi ▸ hj)
      rw [e, e0]; simp
  rw [this, zero_mul]

theorem exists_coef_ne_zero (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F)
    (ht : Function.Injective t) (hnv : ∀ c : ℕ, 0 < c → c ≤ nv → (c : F) ≠ 0)
    (hinv : ¬ LookupValid a t m) : ∃ x0 ∈ vals a t, coef a t m x0 ≠ 0 := by
  unfold LookupValid at hinv
  rw [not_and_or] at hinv
  rcases hinv with h | h
  · push Not at h
    obtain ⟨j, hj⟩ := h
    refine ⟨a j, mem_vals_a a t j, ?_⟩
    have e : Finset.univ.filter (fun i' => t i' = a j) = ∅ := by
      ext i'; simp [(hj i').symm]
    unfold coef
    rw [e, Finset.sum_empty, sub_zero]
    apply hnv
    · exact Finset.card_pos.mpr ⟨j, by simp⟩
    · exact (Finset.card_le_univ _).trans (by simp)
  · push Not at h
    obtain ⟨i, hi⟩ := h
    refine ⟨t i, mem_vals_t a t i, ?_⟩
    have e : Finset.univ.filter (fun i' => t i' = t i) = {i} := by
      ext i'; simp [ht.eq_iff]
    unfold coef
    rw [e, Finset.sum_singleton]
    exact sub_ne_zero.mpr hi

variable [Fintype F]

open Classical in
/-- LogUp soundness: any invalid lookup claim passes at most `nv + nt` non-pole challenges `λ ∈ F`. -/
theorem logup_bad_card_le (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F)
    (ht : Function.Injective t) (hnv : ∀ c : ℕ, 0 < c → c ≤ nv → (c : F) ≠ 0)
    (hinv : ¬ LookupValid a t m) :
    (Finset.univ.filter fun l : F => NonPole a t l ∧ Harmonic a t m l).card ≤ nv + nt := by
  obtain ⟨x0, hx0, hc⟩ := exists_coef_ne_zero a t m ht hnv hinv
  have hN := numer_ne_zero a t m hx0 hc
  have hsub : (Finset.univ.filter fun l : F => NonPole a t l ∧ Harmonic a t m l)
      ⊆ (numer a t m).roots.toFinset := by
    intro l hl
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hl
    obtain ⟨hnp, hh⟩ := hl
    rw [Multiset.mem_toFinset, Polynomial.mem_roots hN, Polynomial.IsRoot,
      numer_eval_nonpole a t m l hnp]
    have : (∑ x ∈ vals a t, coef a t m x * (l + x)⁻¹) = 0 := by
      rw [← harmonic_diff_eq]; exact sub_eq_zero.mpr hh
    rw [this, mul_zero]
  calc (Finset.univ.filter fun l : F => NonPole a t l ∧ Harmonic a t m l).card
      ≤ (numer a t m).roots.toFinset.card := Finset.card_le_card hsub
    _ ≤ Multiset.card (numer a t m).roots := Multiset.toFinset_card_le _
    _ ≤ (numer a t m).natDegree := Polynomial.card_roots' _
    _ ≤ (vals a t).card := numer_natDegree_le a t m
    _ ≤ nv + nt := vals_card_le a t

open Classical in
theorem logup_sound_prob (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F)
    (ht : Function.Injective t) (hnv : ∀ c : ℕ, 0 < c → c ≤ nv → (c : F) ≠ 0)
    (hinv : ¬ LookupValid a t m) :
    ((Finset.univ.filter fun l : F => NonPole a t l ∧ Harmonic a t m l).card : ℝ) / Fintype.card F
      ≤ ((nv : ℝ) + nt) / Fintype.card F := by
  have h := logup_bad_card_le a t m ht hnv hinv
  have h' : ((Finset.univ.filter fun l : F => NonPole a t l ∧ Harmonic a t m l).card : ℝ)
      ≤ (nv : ℝ) + nt := by exact_mod_cast h
  exact div_le_div_of_nonneg_right h' (Nat.cast_nonneg _)

open Classical in
theorem pole_card_le (a : Fin nv → F) (t : Fin nt → F) :
    (Finset.univ.filter fun l : F => ¬ NonPole a t l).card ≤ nv + nt := by
  have hsub : (Finset.univ.filter fun l : F => ¬ NonPole a t l)
      ⊆ (Finset.univ.image fun j => -a j) ∪ (Finset.univ.image fun i => -t i) := by
    intro l hl
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, NonPole, not_and_or, not_forall,
      not_not] at hl
    simp only [Finset.mem_union, Finset.mem_image, Finset.mem_univ, true_and]
    rcases hl with ⟨j, hj⟩ | ⟨i, hi⟩
    · exact Or.inl ⟨j, by rw [← eq_neg_of_add_eq_zero_left hj]⟩
    · exact Or.inr ⟨i, by rw [← eq_neg_of_add_eq_zero_left hi]⟩
  refine (Finset.card_le_card hsub).trans ((Finset.card_union_le _ _).trans (add_le_add ?_ ?_))
  · simpa using (Finset.card_image_le : (Finset.univ.image fun j : Fin nv => -a j).card
      ≤ (Finset.univ : Finset (Fin nv)).card)
  · simpa using (Finset.card_image_le : (Finset.univ.image fun i : Fin nt => -t i).card
      ≤ (Finset.univ : Finset (Fin nt)).card)

open Classical in
theorem pole_or_harmonic_card_le (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F)
    (ht : Function.Injective t) (hnv : ∀ c : ℕ, 0 < c → c ≤ nv → (c : F) ≠ 0)
    (hinv : ¬ LookupValid a t m) :
    (Finset.univ.filter fun l : F => ¬ NonPole a t l ∨ Harmonic a t m l).card ≤ 2 * (nv + nt) := by
  have hsub : (Finset.univ.filter fun l : F => ¬ NonPole a t l ∨ Harmonic a t m l)
      ⊆ (Finset.univ.filter fun l : F => ¬ NonPole a t l) ∪
        (Finset.univ.filter fun l : F => NonPole a t l ∧ Harmonic a t m l) := by
    intro l hl
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union] at hl ⊢
    by_cases hp : NonPole a t l
    · rcases hl with h | h
      · exact absurd hp h
      · exact Or.inr ⟨hp, h⟩
    · exact Or.inl hp
  have := (Finset.card_le_card hsub).trans (Finset.card_union_le _ _)
  have h1 := pole_card_le a t
  have h2 := logup_bad_card_le a t m ht hnv hinv
  omega

end logup

section merged

variable {F : Type*} [Field F] {nv nt : ℕ}

def MergedZeroSum (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F) (l : F) : Prop :=
  ∑ s : Fin nv ⊕ Fin nt, Sum.elim (fun j => (l + a j)⁻¹) (fun i => -(m i * (l + t i)⁻¹)) s = 0

theorem mergedZeroSum_iff (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F) (l : F) :
    MergedZeroSum a t m l ↔ ∑ j, (l + a j)⁻¹ = ∑ i, m i * (l + t i)⁻¹ := by
  unfold MergedZeroSum
  rw [Fintype.sum_sum_type]
  simp only [Sum.elim_inl, Sum.elim_inr, Finset.sum_neg_distrib]
  exact add_neg_eq_zero

end merged

section fingerprint

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F] {w : ℕ}

theorem fingerprint_collision_card_le (u v : Fin w → F) (h : u ≠ v) :
    (Finset.univ.filter fun g : F => ∑ k, u k * g ^ (k : ℕ) = ∑ k, v k * g ^ (k : ℕ)).card ≤ w := by
  set p : F[X] := ∑ k : Fin w, C (u k - v k) * X ^ (k : ℕ) with hp
  have hcoef : ∀ k : Fin w, p.coeff k = u k - v k := by
    intro k
    simp only [hp, Polynomial.finset_sum_coeff, Polynomial.coeff_C_mul, Polynomial.coeff_X_pow]
    rw [Finset.sum_eq_single k]
    · simp
    · intro j _ hjk
      have : (k : ℕ) ≠ (j : ℕ) := fun e => hjk (Fin.ext e).symm
      simp [this]
    · simp
  have hp0 : p ≠ 0 := by
    intro h0
    obtain ⟨k, hk⟩ := Function.ne_iff.mp h
    have := hcoef k
    rw [h0, Polynomial.coeff_zero] at this
    exact hk (sub_eq_zero.mp this.symm)
  have hdeg : p.natDegree ≤ w := by
    refine Polynomial.natDegree_sum_le_of_forall_le _ _ fun k _ => ?_
    refine (Polynomial.natDegree_C_mul_le _ _).trans ?_
    rw [Polynomial.natDegree_X_pow]
    exact k.isLt.le
  have hev : ∀ g : F, p.eval g = ∑ k, u k * g ^ (k : ℕ) - ∑ k, v k * g ^ (k : ℕ) := by
    intro g
    simp [hp, Polynomial.eval_finset_sum, sub_mul, Finset.sum_sub_distrib]
  have hsub : (Finset.univ.filter fun g : F => ∑ k, u k * g ^ (k : ℕ) = ∑ k, v k * g ^ (k : ℕ))
      ⊆ p.roots.toFinset := by
    intro g hg
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hg
    rw [Multiset.mem_toFinset, Polynomial.mem_roots hp0, Polynomial.IsRoot, hev, hg, sub_self]
  exact (Finset.card_le_card hsub).trans
    ((Multiset.toFinset_card_le _).trans ((Polynomial.card_roots' _).trans hdeg))

end fingerprint

theorem natCast_ne_zero_of_lt {F : Type*} [Field F] (p : ℕ) [CharP F p] (nv : ℕ) (hp : nv < p) :
    ∀ c : ℕ, 0 < c → c ≤ nv → (c : F) ≠ 0 := by
  intro c hc hcn h
  have := (CharP.cast_eq_zero_iff F p c).mp h
  exact absurd (Nat.le_of_dvd hc this) (by omega)

end FlashProveNext

#print axioms FlashProveNext.harmonic_diff_eq
#print axioms FlashProveNext.numer_natDegree_le
#print axioms FlashProveNext.numer_ne_zero
#print axioms FlashProveNext.logup_complete
#print axioms FlashProveNext.logup_bad_card_le
#print axioms FlashProveNext.logup_sound_prob
#print axioms FlashProveNext.pole_card_le
#print axioms FlashProveNext.pole_or_harmonic_card_le
#print axioms FlashProveNext.mergedZeroSum_iff
#print axioms FlashProveNext.fingerprint_collision_card_le
#print axioms FlashProveNext.natCast_ne_zero_of_lt
