/-
Symmetric saturation uniqueness and certified MoE Top-K routing bounds (`specs/06-tolerance-and-cheats.md` §3.3).

1. `sat_iff`, `sat_existsUnique`          : symmetric clamping `SatRel A y` admits a unique witness `(ys, ep, en)`;
2. `approxTopK_sound`, `approxTopK_of_float` : uniform `δ`-band Top-K completeness and total-score proximity;
3. `approxTopK_pairwise`, `pairwiseApproxTopK_tighter`, `pairwiseApproxTopK_sum` :
   componentwise certified routing `s_e + r_e ≥ s_{e'} - r_{e'}` when engine scores satisfy `|s_e - s_hat_e| ≤ r_e`.
-/
import Mathlib
import FlashProveNext.StdAlgOps

namespace FlashProveNext

open Finset

set_option linter.unusedSectionVars false

def SatRel (A y : ℤ) (t : ℤ × ℤ × ℤ) : Prop :=
  -A ≤ t.1 ∧ t.1 < A ∧ 0 ≤ t.2.1 ∧ 0 ≤ t.2.2 ∧ y = t.1 + t.2.1 - t.2.2 ∧
    t.2.1 * t.2.2 = 0 ∧ t.2.1 * (A - 1 - t.1) = 0 ∧ t.2.2 * (t.1 + A) = 0

def satEval (A y : ℤ) : ℤ × ℤ × ℤ :=
  (max (-A) (min (A - 1) y), max 0 (y - (A - 1)), max 0 (-A - y))

theorem sat_iff {A : ℤ} (hA : 0 < A) (y : ℤ) (t : ℤ × ℤ × ℤ) : SatRel A y t ↔ t = satEval A y := by
  obtain ⟨ys, ep, en⟩ := t
  unfold SatRel satEval
  simp only [Prod.mk.injEq]
  constructor
  · rintro ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩
    have c1 := mul_eq_zero.mp h6
    have c2 := mul_eq_zero.mp h7
    have c3 := mul_eq_zero.mp h8
    refine ⟨?_, ?_, ?_⟩ <;> rcases c1 with c1 | c1 <;> rcases c2 with c2 | c2 <;>
      rcases c3 with c3 | c3 <;> omega
  · rintro ⟨rfl, rfl, rfl⟩
    refine ⟨by omega, by omega, by omega, by omega, by omega, ?_, ?_, ?_⟩
    · rcases (by omega : max 0 (y - (A - 1)) = 0 ∨ max 0 (-A - y) = 0) with h | h <;> simp [h]
    · rcases (by omega : max 0 (y - (A - 1)) = 0 ∨ A - 1 - max (-A) (min (A - 1) y) = 0) with h | h <;>
        simp [h]
    · rcases (by omega : max 0 (-A - y) = 0 ∨ max (-A) (min (A - 1) y) + A = 0) with h | h <;>
        simp [h]

theorem sat_existsUnique {A : ℤ} (hA : 0 < A) (y : ℤ) : ∃! t : ℤ × ℤ × ℤ, SatRel A y t :=
  ⟨satEval A y, (sat_iff hA y _).2 rfl, fun t ht => (sat_iff hA y t).1 ht⟩

section approx

variable {E : ℕ}

def IsApproxTopK (s : Fin E → ℤ) (δ : ℤ) (K : ℕ) (S : Finset (Fin E)) : Prop :=
  S.card = K ∧ ∀ i ∈ S, ∀ j ∉ S, s j - δ ≤ s i

theorem approxTopK_of_topK {s : Fin E → ℤ} {K : ℕ} {S : Finset (Fin E)} (h : IsTopK s K S)
    {δ : ℤ} (hδ : 0 ≤ δ) : IsApproxTopK s δ K S := by
  refine ⟨h.1, fun i hi j hj => ?_⟩
  have := h.2 i hi j hj
  unfold before at this
  omega

theorem approxTopK_of_weak {s : Fin E → ℤ} {K : ℕ} {S : Finset (Fin E)} (hc : S.card = K)
    (h : ∀ i ∈ S, ∀ j ∉ S, s j ≤ s i) {δ : ℤ} (hδ : 0 ≤ δ) : IsApproxTopK s δ K S :=
  ⟨hc, fun i hi j hj => by have := h i hi j hj; omega⟩

theorem sum_le_add_card_mul {ι : Type*} (T1 T2 : Finset ι) (s : ι → ℤ) (δ : ℤ)
    (hc : T1.card = T2.card) (h : ∀ i ∈ T1, ∀ j ∈ T2, s i ≤ s j + δ) :
    ∑ i ∈ T1, s i ≤ ∑ j ∈ T2, s j + (T1.card : ℤ) * δ := by
  rcases T2.eq_empty_or_nonempty with h2 | hne
  · subst h2
    have : T1 = ∅ := Finset.card_eq_zero.mp (by simpa using hc)
    subst this
    simp
  · obtain ⟨j0, hj0, hmin⟩ := Finset.exists_min_image T2 s hne
    have h1 : ∑ i ∈ T1, s i ≤ ∑ _i ∈ T1, (s j0 + δ) := Finset.sum_le_sum fun i hi => h i hi j0 hj0
    have h2 : ∑ _j ∈ T2, s j0 ≤ ∑ j ∈ T2, s j := Finset.sum_le_sum fun j hj => hmin j hj
    simp only [Finset.sum_const, nsmul_eq_mul] at h1 h2
    rw [hc] at h1 ⊢
    have : (T2.card : ℤ) * (s j0 + δ) = T2.card * s j0 + T2.card * δ := by ring
    linarith

theorem approxTopK_card_sum {s : Fin E → ℤ} {δ : ℤ} (hδ : 0 ≤ δ) {K : ℕ} {S : Finset (Fin E)}
    (hS : IsApproxTopK s δ K S) {S' : Finset (Fin E)} (hS' : IsTopK s K S') :
    ∑ i ∈ S', s i - (K : ℤ) * δ ≤ ∑ i ∈ S, s i ∧ ∑ i ∈ S, s i ≤ ∑ i ∈ S', s i := by
  classical
  have hc : (S' \ S).card = (S \ S').card := Finset.card_sdiff_comm (by rw [hS.1, hS'.1])
  have e1 : ∑ i ∈ S', s i = ∑ i ∈ S' ∩ S, s i + ∑ i ∈ S' \ S, s i :=
    (Finset.sum_inter_add_sum_diff S' S s).symm
  have e2 : ∑ i ∈ S, s i = ∑ i ∈ S' ∩ S, s i + ∑ i ∈ S \ S', s i := by
    rw [Finset.inter_comm]; exact (Finset.sum_inter_add_sum_diff S S' s).symm
  constructor
  · have hk : ∀ i ∈ S' \ S, ∀ j ∈ S \ S', s i ≤ s j + δ := by
      intro i hi j hj
      have := hS.2 j (Finset.mem_sdiff.mp hj).1 i (Finset.mem_sdiff.mp hi).2
      omega
    have h := sum_le_add_card_mul (S' \ S) (S \ S') s δ hc hk
    have hle : ((S' \ S).card : ℤ) ≤ K := by
      have : (S' \ S).card ≤ S'.card := Finset.card_le_card Finset.sdiff_subset
      rw [hS'.1] at this; exact_mod_cast this
    have : ((S' \ S).card : ℤ) * δ ≤ K * δ := mul_le_mul_of_nonneg_right hle hδ
    linarith
  · have hk : ∀ i ∈ S \ S', ∀ j ∈ S' \ S, s i ≤ s j + 0 := by
      intro i hi j hj
      have := hS'.2 j (Finset.mem_sdiff.mp hj).1 i (Finset.mem_sdiff.mp hi).2
      unfold before at this
      omega
    have h := sum_le_add_card_mul (S \ S') (S' \ S) s 0 hc.symm hk
    simp only [mul_zero, add_zero] at h
    linarith

theorem approxTopK_sound (s : Fin E → ℤ) {δ : ℤ} (hδ : 0 ≤ δ) {K : ℕ} (hK : K ≤ E) :
    (∃ S', IsTopK s K S' ∧ IsApproxTopK s δ K S') ∧
    (∀ S, IsApproxTopK s δ K S → S.card = K ∧ ∀ S', IsTopK s K S' →
      ∑ i ∈ S', s i - (K : ℤ) * δ ≤ ∑ i ∈ S, s i ∧ ∑ i ∈ S, s i ≤ ∑ i ∈ S', s i) := by
  refine ⟨?_, fun S hS => ⟨hS.1, fun S' hS' => approxTopK_card_sum hδ hS hS'⟩⟩
  obtain ⟨S', hS'⟩ := topK_exists s K hK
  exact ⟨S', hS', approxTopK_of_topK hS' hδ⟩

theorem approxTopK_of_float (s sh : Fin E → ℤ) (δ : ℤ) {K : ℕ} {S : Finset (Fin E)}
    (hc : S.card = K) (hS : ∀ i ∈ S, ∀ j ∉ S, s j ≤ s i) (hclose : ∀ e, 2 * |s e - sh e| ≤ δ) :
    IsApproxTopK sh δ K S := by
  refine ⟨hc, fun i hi j hj => ?_⟩
  have h1 := hclose i
  have h2 := hclose j
  have h3 := hS i hi j hj
  have a1 := le_abs_self (s i - sh i)
  have a2 := neg_abs_le (s i - sh i)
  have a3 := le_abs_self (s j - sh j)
  have a4 := neg_abs_le (s j - sh j)
  linarith

/-- Componentwise pairwise-band MoE routing relation (`specs/06` §3.3). -/
def IsPairwiseApproxTopK (s r : Fin E → ℤ) (K : ℕ) (S : Finset (Fin E)) : Prop :=
  S.card = K ∧ ∀ e ∈ S, ∀ e' ∉ S, s e + r e ≥ s e' - r e'

theorem approxTopK_pairwise (s s_hat r : Fin E → ℤ) {K : ℕ} {S : Finset (Fin E)}
    (hc : S.card = K) (hS : ∀ e ∈ S, ∀ e' ∉ S, s_hat e' ≤ s_hat e)
    (hclose : ∀ e, |s e - s_hat e| ≤ r e) :
    IsPairwiseApproxTopK s r K S := by
  refine ⟨hc, fun e he e' he' => ?_⟩
  have h1 := hclose e
  have h2 := hclose e'
  have h3 := hS e he e' he'
  have a1 := le_abs_self (s e - s_hat e)
  have a2 := neg_abs_le (s e - s_hat e)
  have a3 := le_abs_self (s e' - s_hat e')
  have a4 := neg_abs_le (s e' - s_hat e')
  linarith

theorem approxTopK_pairwise_of_topK (s s_hat r : Fin E → ℤ) {K : ℕ} {S : Finset (Fin E)}
    (hS : IsTopK s_hat K S) (hclose : ∀ e, |s e - s_hat e| ≤ r e) :
    IsPairwiseApproxTopK s r K S := by
  refine approxTopK_pairwise s s_hat r hS.1 (fun e he e' he' => ?_) hclose
  have := hS.2 e he e' he'
  unfold before at this
  omega

theorem pairwiseApproxTopK_tighter {s r : Fin E → ℤ} {rmax : ℤ} {K : ℕ} {S : Finset (Fin E)}
    (h : IsPairwiseApproxTopK s r K S) (hrmax : ∀ e, r e ≤ rmax) :
    IsApproxTopK s (2 * rmax) K S := by
  refine ⟨h.1, fun e he e' he' => ?_⟩
  have hpair := h.2 e he e' he'
  have hr1 := hrmax e
  have hr2 := hrmax e'
  linarith

theorem pairwiseApproxTopK_sum {s r : Fin E → ℤ} (hr : ∀ e, 0 ≤ r e) {K : ℕ}
    {S S' : Finset (Fin E)} (hS : IsPairwiseApproxTopK s r K S) (hS' : IsTopK s K S') :
    ∑ e ∈ S', (s e - r e) ≤ ∑ e ∈ S, (s e + r e) ∧ ∑ e ∈ S, s e ≤ ∑ e ∈ S', s e := by
  classical
  have hc : (S' \ S).card = (S \ S').card := Finset.card_sdiff_comm (by rw [hS.1, hS'.1])
  constructor
  · have e1 : ∑ e ∈ S', (s e - r e) = ∑ e ∈ S' ∩ S, (s e - r e) + ∑ e ∈ S' \ S, (s e - r e) :=
      (Finset.sum_inter_add_sum_diff S' S (fun e => s e - r e)).symm
    have e2 : ∑ e ∈ S, (s e + r e) = ∑ e ∈ S' ∩ S, (s e + r e) + ∑ e ∈ S \ S', (s e + r e) := by
      rw [Finset.inter_comm]
      exact (Finset.sum_inter_add_sum_diff S S' (fun e => s e + r e)).symm
    have hinter : ∑ e ∈ S' ∩ S, (s e - r e) ≤ ∑ e ∈ S' ∩ S, (s e + r e) :=
      Finset.sum_le_sum fun e _ => by linarith [hr e]
    have hdiff : ∑ e ∈ S' \ S, (s e - r e) ≤ ∑ e ∈ S \ S', (s e + r e) := by
      rcases (S \ S').eq_empty_or_nonempty with hemp | hne
      · have hemp' : S' \ S = ∅ := Finset.card_eq_zero.mp (by rw [hc, hemp]; simp)
        simp [hemp, hemp']
      · obtain ⟨j0, hj0, hmin⟩ := Finset.exists_min_image (S \ S') (fun e => s e + r e) hne
        have h1 : ∑ i ∈ S' \ S, (s i - r i) ≤ ∑ _i ∈ S' \ S, (s j0 + r j0) :=
          Finset.sum_le_sum fun i hi =>
            hS.2 j0 (Finset.mem_sdiff.mp hj0).1 i (Finset.mem_sdiff.mp hi).2
        have h2 : ∑ _j ∈ S \ S', (s j0 + r j0) ≤ ∑ j ∈ S \ S', (s j + r j) :=
          Finset.sum_le_sum fun j hj => hmin j hj
        simp only [Finset.sum_const, nsmul_eq_mul] at h1 h2
        rw [hc] at h1
        exact h1.trans h2
    linarith
  · have e1 : ∑ i ∈ S', s i = ∑ i ∈ S' ∩ S, s i + ∑ i ∈ S' \ S, s i :=
      (Finset.sum_inter_add_sum_diff S' S s).symm
    have e2 : ∑ i ∈ S, s i = ∑ i ∈ S' ∩ S, s i + ∑ i ∈ S \ S', s i := by
      rw [Finset.inter_comm]; exact (Finset.sum_inter_add_sum_diff S S' s).symm
    have hk : ∀ i ∈ S \ S', ∀ j ∈ S' \ S, s i ≤ s j + 0 := by
      intro i hi j hj
      have := hS'.2 j (Finset.mem_sdiff.mp hj).1 i (Finset.mem_sdiff.mp hi).2
      unfold before at this
      omega
    have h := sum_le_add_card_mul (S \ S') (S' \ S) s 0 hc.symm hk
    simp only [mul_zero, add_zero] at h
    linarith

end approx

end FlashProveNext

#print axioms FlashProveNext.sat_iff
#print axioms FlashProveNext.sat_existsUnique
#print axioms FlashProveNext.approxTopK_of_topK
#print axioms FlashProveNext.approxTopK_of_weak
#print axioms FlashProveNext.sum_le_add_card_mul
#print axioms FlashProveNext.approxTopK_card_sum
#print axioms FlashProveNext.approxTopK_sound
#print axioms FlashProveNext.approxTopK_of_float
#print axioms FlashProveNext.approxTopK_pairwise
#print axioms FlashProveNext.approxTopK_pairwise_of_topK
#print axioms FlashProveNext.pairwiseApproxTopK_tighter
#print axioms FlashProveNext.pairwiseApproxTopK_sum
