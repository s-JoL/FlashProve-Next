/-
Operator-level uniqueness and existence lemmas for integer reference arithmetic:

1. `divmod_unique` / `round_div_unique` : Euclidean division and half-up rounded integer division `⌊(2a+d)/(2d)⌋`;
2. `pwRel_existsUnique`                 : piecewise table lookup (SiLU / GELU / activation curves);
3. `sqrt_ineq_iff` / `l2_rel_int`       : square-root-free polynomial equivalence for `L_2` ball checks;
4. `intCast_inj_small`                  : lossless embedding of bounded integers `|x - y| < p` into `ZMod p`;
5. `topK_unique` / `topK_exists`        : deterministic Top-K / argmax selection with index tie-breaking;
6. `trace_unique_of_graph`              : composition with `StdAlg.trace_unique`.
-/
import Mathlib
import FlashProveNext.StdAlg

namespace FlashProveNext

open Finset

set_option linter.unusedSectionVars false

theorem divmod_unique {a m : ℤ} (hm : 0 < m) {q q' r r' : ℤ}
    (h : a = m * q + r) (h' : a = m * q' + r')
    (hr : 0 ≤ r ∧ r < m) (hr' : 0 ≤ r' ∧ r' < m) : q = q' ∧ r = r' := by
  have e1 := (Int.ediv_emod_unique hm (a := a) (q := q) (r := r)).2 ⟨by linarith, hr⟩
  have e2 := (Int.ediv_emod_unique hm (a := a) (q := q') (r := r')).2 ⟨by linarith, hr'⟩
  exact ⟨e1.1.symm.trans e2.1, e1.2.symm.trans e2.2⟩

theorem divmod_exists {a m : ℤ} (hm : 0 < m) : ∃ q r : ℤ, a = m * q + r ∧ 0 ≤ r ∧ r < m :=
  ⟨a / m, a % m, (Int.mul_ediv_add_emod a m).symm, Int.emod_nonneg _ hm.ne', Int.emod_lt_of_pos _ hm⟩

theorem round_div_unique (a d : ℤ) (hd : 0 < d) :
    ∃! q : ℤ, 2 * d * q ≤ 2 * a + d ∧ 2 * a + d < 2 * d * q + 2 * d := by
  refine ⟨(2 * a + d) / (2 * d), ?_, ?_⟩
  · have h2 : 0 < 2 * d := by linarith
    have := Int.mul_ediv_add_emod (2 * a + d) (2 * d)
    have h0 := Int.emod_nonneg (2 * a + d) h2.ne'
    have h1 := Int.emod_lt_of_pos (2 * a + d) h2
    constructor <;> nlinarith
  · rintro q ⟨h1, h2⟩
    have hm : 0 < 2 * d := by linarith
    have := (divmod_unique hm (a := 2 * a + d) (q := q) (q' := (2 * a + d) / (2 * d))
      (r := 2 * a + d - 2 * d * q) (r' := (2 * a + d) % (2 * d)) (by ring)
      (Int.mul_ediv_add_emod (2 * a + d) (2 * d)).symm ⟨by linarith, by linarith⟩
      ⟨Int.emod_nonneg _ hm.ne', Int.emod_lt_of_pos _ hm⟩).1
    exact this

example : ∃ a d q q' r r' : ℤ, 0 < d ∧ a = d * q + r ∧ a = d * q' + r' ∧ 0 ≤ r ∧ r < 2 * d ∧
    0 ≤ r' ∧ r' < 2 * d ∧ q ≠ q' := ⟨5, 3, 0, 1, 5, 2, by norm_num⟩

section pw
variable {X Y : Type*} [LinearOrder X]

def PwRel (lo hi : X) (fl fm fh : X → Y) (x : X) (y : Y) : Prop :=
  (x ≤ lo → y = fl x) ∧ (hi ≤ x → y = fh x) ∧ (lo < x → x < hi → y = fm x)

noncomputable def pwEval (lo hi : X) (fl fm fh : X → Y) (x : X) : Y := by
  classical exact if x ≤ lo then fl x else if hi ≤ x then fh x else fm x

theorem pwRel_iff (lo hi : X) (hlt : lo < hi) (fl fm fh : X → Y) (x : X) (y : Y) :
    PwRel lo hi fl fm fh x y ↔ y = pwEval lo hi fl fm fh x := by
  classical
  unfold PwRel pwEval
  by_cases h1 : x ≤ lo
  · have h2 : ¬ hi ≤ x := not_le.mpr (lt_of_le_of_lt h1 hlt)
    rw [if_pos h1]
    constructor
    · exact fun h => h.1 h1
    · intro h; exact ⟨fun _ => h, fun h' => absurd h' h2, fun h' => absurd h1 (not_le.mpr h')⟩
  · by_cases h2 : hi ≤ x
    · rw [if_neg h1, if_pos h2]
      constructor
      · exact fun h => h.2.1 h2
      · intro h; exact ⟨fun h' => absurd h' h1, fun _ => h, fun _ h' => absurd h2 (not_le.mpr h')⟩
    · rw [if_neg h1, if_neg h2]
      constructor
      · exact fun h => h.2.2 (not_le.mp h1) (not_le.mp h2)
      · intro h; exact ⟨fun h' => absurd h' h1, fun h' => absurd h' h2, fun _ _ => h⟩

theorem pwRel_existsUnique (lo hi : X) (hlt : lo < hi) (fl fm fh : X → Y) (x : X) :
    ∃! y, PwRel lo hi fl fm fh x y :=
  ⟨pwEval lo hi fl fm fh x, (pwRel_iff lo hi hlt fl fm fh x _).2 rfl,
    fun y hy => (pwRel_iff lo hi hlt fl fm fh x y).1 hy⟩
end pw

theorem sqrt_ineq_iff {A B C a b : ℝ} (hB : 0 ≤ B) (hC : 0 ≤ C) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    Real.sqrt A ≤ a * Real.sqrt B + b * Real.sqrt C ↔
      (A - a ^ 2 * B - b ^ 2 * C ≤ 0 ∨ (A - a ^ 2 * B - b ^ 2 * C) ^ 2 ≤ 4 * a ^ 2 * b ^ 2 * (B * C)) := by
  have hs : 0 ≤ a * Real.sqrt B + b * Real.sqrt C := by positivity
  rw [Real.sqrt_le_iff]
  have hsq : (a * Real.sqrt B + b * Real.sqrt C) ^ 2
      = a ^ 2 * B + b ^ 2 * C + 2 * (a * b * (Real.sqrt B * Real.sqrt C)) := by
    have e1 := Real.sq_sqrt hB
    have e2 := Real.sq_sqrt hC
    nlinarith [e1, e2]
  have hm : Real.sqrt B * Real.sqrt C = Real.sqrt (B * C) := (Real.sqrt_mul hB C).symm
  set m := Real.sqrt (B * C) with hmdef
  have hm0 : 0 ≤ m := Real.sqrt_nonneg _
  have hm2 : m ^ 2 = B * C := Real.sq_sqrt (mul_nonneg hB hC)
  constructor
  · rintro ⟨_, h⟩
    rw [hsq, hm] at h
    by_cases hD : A - a ^ 2 * B - b ^ 2 * C ≤ 0
    · exact Or.inl hD
    · right
      have h1 : A - a ^ 2 * B - b ^ 2 * C ≤ 2 * (a * b * m) := by linarith
      have h0 : 0 ≤ A - a ^ 2 * B - b ^ 2 * C := le_of_lt (not_le.mp hD)
      have := pow_le_pow_left₀ h0 h1 2
      calc _ ≤ (2 * (a * b * m)) ^ 2 := this
        _ = 4 * a ^ 2 * b ^ 2 * (B * C) := by rw [← hm2]; ring
  · intro h
    refine ⟨hs, ?_⟩
    rw [hsq, hm]
    rcases h with hD | hD
    · have : 0 ≤ 2 * (a * b * m) := by positivity
      linarith
    · have h2 : 0 ≤ 2 * (a * b * m) := by positivity
      have : (A - a ^ 2 * B - b ^ 2 * C) ^ 2 ≤ (2 * (a * b * m)) ^ 2 := by
        calc _ ≤ 4 * a ^ 2 * b ^ 2 * (B * C) := hD
          _ = (2 * (a * b * m)) ^ 2 := by rw [← hm2]; ring
      have := abs_le_of_sq_le_sq' this h2
      linarith [this.2]

theorem l2_rel_int {A B C : ℝ} (hB : 0 ≤ B) (hC : 0 ≤ C) :
    Real.sqrt A ≤ (1 / 16) * Real.sqrt B + (1 / 32) * Real.sqrt C ↔
      (1024 * A ≤ 4 * B + C ∨ (1024 * A - 4 * B - C) ^ 2 ≤ 16 * (B * C)) := by
  rw [sqrt_ineq_iff hB hC (by norm_num) (by norm_num)]
  have e : (1024 * A - 4 * B - C) ^ 2 = 1024 ^ 2 * (A - (1 / 16) ^ 2 * B - (1 / 32) ^ 2 * C) ^ 2 := by ring
  constructor
  · rintro (h | h)
    · left; nlinarith
    · right; rw [e]; nlinarith
  · rintro (h | h)
    · left; nlinarith
    · right; rw [e] at h; nlinarith

theorem intCast_inj_small (p : ℕ) {x y : ℤ} (h : |x - y| < p) (hxy : (x : ZMod p) = (y : ZMod p)) : x = y := by
  have := (ZMod.intCast_eq_intCast_iff_dvd_sub y x p).1 hxy.symm
  have h0 := Int.eq_zero_of_abs_lt_dvd this h
  linarith

section topk
variable {E : ℕ}

def before (s : Fin E → ℤ) (i j : Fin E) : Prop := s j < s i ∨ (s j = s i ∧ i < j)

def IsTopK (s : Fin E → ℤ) (K : ℕ) (S : Finset (Fin E)) : Prop :=
  S.card = K ∧ ∀ i ∈ S, ∀ j ∉ S, before s i j

theorem before_asymm (s : Fin E → ℤ) {i j : Fin E} (h : before s i j) : ¬ before s j i := by
  unfold before at *
  omega

theorem topK_unique (s : Fin E → ℤ) (K : ℕ) {S S' : Finset (Fin E)} (h : IsTopK s K S)
    (h' : IsTopK s K S') : S = S' := by
  by_contra hne
  have h1 : ¬ S ⊆ S' := fun hs => hne (Finset.eq_of_subset_of_card_le hs (by rw [h.1, h'.1]))
  have h2 : ¬ S' ⊆ S := fun hs => hne (Finset.eq_of_subset_of_card_le hs (by rw [h.1, h'.1])).symm
  obtain ⟨i, hi, hi'⟩ := Finset.not_subset.mp h1
  obtain ⟨j, hj, hj'⟩ := Finset.not_subset.mp h2
  exact before_asymm s (h.2 i hi j hj') (h'.2 j hj i hi')

theorem topK_exists (s : Fin E → ℤ) (K : ℕ) (hK : K ≤ E) : ∃ S, IsTopK s K S := by
  induction K with
  | zero => exact ⟨∅, by simp, by simp⟩
  | succ K ih =>
    obtain ⟨S, hS, hbef⟩ := ih (Nat.le_of_succ_le hK)
    have hne : Sᶜ.Nonempty := by
      rw [Finset.nonempty_iff_ne_empty]; intro e
      have := Finset.card_compl S
      rw [e] at this; simp at this; omega
    obtain ⟨m, hm, hmax⟩ := Finset.exists_max_image Sᶜ
      (fun i : Fin E => s i * (E : ℤ) + ((E : ℤ) - 1 - (i : ℕ))) hne
    refine ⟨insert m S, ?_, ?_⟩
    · rw [Finset.card_insert_of_notMem (by simpa using hm), hS]
    · intro i hi j hj
      rw [Finset.mem_insert, not_or] at hj
      rcases Finset.mem_insert.mp hi with rfl | hiS
      · have hjc : j ∈ Sᶜ := by simpa using hj.2
        have hle := hmax j hjc
        have hne : j ≠ i := hj.1
        have hv : (j : ℕ) ≠ (i : ℕ) := fun e => hne (Fin.ext e)
        have hEj : (j : ℕ) < E := j.isLt
        have hEi : (i : ℕ) < E := i.isLt
        unfold before
        rcases lt_trichotomy (s j) (s i) with h | h | h
        · exact Or.inl h
        · right; refine ⟨h, ?_⟩
          rw [h] at hle
          have : (i : ℕ) < (j : ℕ) := by
            have : ((E : ℤ) - 1 - (j : ℕ)) ≤ ((E : ℤ) - 1 - (i : ℕ)) := by linarith
            omega
          exact this
        · exfalso
          have h1 : s i + 1 ≤ s j := h
          have : (s i + 1) * (E : ℤ) ≤ s j * (E : ℤ) :=
            mul_le_mul_of_nonneg_right h1 (by positivity)
          have hE0 : (0 : ℤ) ≤ (E : ℤ) - 1 - (j : ℕ) := by omega
          nlinarith
      · exact hbef i hiS j hj.2
end topk

theorem hfun_of_graph {S : Type*} (f : ℕ → S → S) (step : ℕ → S → S → Prop)
    (h : ∀ i s a, step i s a ↔ a = f i s) : ∀ i s a b, step i s a → step i s b → a = b :=
  fun i s a b ha hb => ((h i s a).1 ha).trans ((h i s b).1 hb).symm

theorem trace_unique_of_graph {S : Type*} (f : ℕ → S → S) (step : ℕ → S → S → Prop)
    (h : ∀ i s a, step i s a ↔ a = f i s) (n : ℕ) (τ τ' : ℕ → S) (h0 : τ 0 = τ' 0)
    (hτ : ∀ i < n, step i (τ i) (τ (i + 1))) (hτ' : ∀ i < n, step i (τ' i) (τ' (i + 1))) :
    ∀ i ≤ n, τ i = τ' i :=
  trace_unique step (hfun_of_graph f step h) n τ τ' h0 hτ hτ'

end FlashProveNext

#print axioms FlashProveNext.divmod_unique
#print axioms FlashProveNext.round_div_unique
#print axioms FlashProveNext.pwRel_existsUnique
#print axioms FlashProveNext.l2_rel_int
#print axioms FlashProveNext.intCast_inj_small
#print axioms FlashProveNext.topK_unique
#print axioms FlashProveNext.topK_exists
#print axioms FlashProveNext.trace_unique_of_graph
