/-
Full `n`-variable Sumcheck protocol soundness against adaptive provers.

* `psum P i r`             : partial hypercube sum fixing the first `i` coordinates to `r` and summing over `{0,1}^{n-i}`;
* `roundDeg_of_indivDeg`   : individual degree `≤ d` implies each honest round polynomial has degree `≤ d`;
* `exists_bad_of_accepts`  : if `S ≠ ∑_{x ∈ {0,1}^n} P(x)` and the verifier accepts on challenges `r`,
                             some round `i` hit `badSet P g i r`;
* `sumcheck_sound`         : any adaptive prover strategy fools all round checks and final evaluation with
                             probability at most `n · d / |F|` (`sumcheck_sound_real` in real probability form).
-/
import Mathlib
import FlashProveNext.Sumcheck

namespace FlashProveNext

open Finset

set_option linter.unusedSectionVars false

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F] {n : ℕ}

def hybrid (i : ℕ) (r : Fin n → F) : Finset (Fin n → F) :=
  Fintype.piFinset fun j => if (j : ℕ) < i then {r j} else ({0, 1} : Finset F)

def psum (P : (Fin n → F) → F) (i : ℕ) (r : Fin n → F) : F := ∑ x ∈ hybrid i r, P x

def cubeSum (P : (Fin n → F) → F) : F :=
  ∑ x ∈ Fintype.piFinset (fun _ : Fin n => ({0, 1} : Finset F)), P x

theorem mem_hybrid (i : ℕ) (r x : Fin n → F) :
    x ∈ hybrid i r ↔ ∀ j : Fin n, ((j : ℕ) < i → x j = r j) ∧ (i ≤ (j : ℕ) → x j = 0 ∨ x j = 1) := by
  unfold hybrid
  rw [Fintype.mem_piFinset]
  refine forall_congr' fun j => ?_
  by_cases h : (j : ℕ) < i
  · simp [h, not_le.mpr h]
  · simp [h, not_lt.mp h]

theorem hybrid_succ (i : Fin n) (r : Fin n → F) (t : F) (ht : t = 0 ∨ t = 1) (x : Fin n → F) :
    x ∈ hybrid (i + 1) (Function.update r i t) ↔ x ∈ hybrid i r ∧ x i = t := by
  simp only [mem_hybrid]
  constructor
  · intro h
    have hit := (h i).1 (by omega)
    rw [Function.update_self] at hit
    refine ⟨fun j => ?_, hit⟩
    have hj := h j
    rcases lt_trichotomy (j : ℕ) i with hl | he | hg
    · have hji : j ≠ i := fun e => by simp [e] at hl
      refine ⟨fun _ => ?_, fun _ => by omega⟩
      have := hj.1 (by omega); rwa [Function.update_of_ne hji] at this
    · have : j = i := Fin.ext he
      subst this
      exact ⟨fun h' => by omega, fun _ => by rw [hit]; exact ht⟩
    · exact ⟨fun h' => by omega, fun _ => hj.2 (by omega)⟩
  · rintro ⟨h, hit⟩ j
    have hj := h j
    rcases lt_trichotomy (j : ℕ) i with hl | he | hg
    · have hji : j ≠ i := fun e => by simp [e] at hl
      refine ⟨fun _ => ?_, fun _ => by omega⟩
      rw [Function.update_of_ne hji]; exact hj.1 hl
    · have : j = i := Fin.ext he
      subst this
      exact ⟨fun _ => by rw [Function.update_self]; exact hit, fun h' => by omega⟩
    · exact ⟨fun h' => by omega, fun _ => hj.2 (by omega)⟩

theorem psum_split (P : (Fin n → F) → F) (i : Fin n) (r : Fin n → F) :
    psum P i r = psum P (i + 1) (Function.update r i 0) + psum P (i + 1) (Function.update r i 1) := by
  have e : ∀ t : F, (t = 0 ∨ t = 1) → hybrid (i + 1) (Function.update r i t)
      = (hybrid i r).filter (fun x => x i = t) := by
    intro t ht; ext x
    rw [hybrid_succ i r t ht, Finset.mem_filter]
  unfold psum
  rw [e 0 (Or.inl rfl), e 1 (Or.inr rfl), ← Finset.sum_union]
  · apply Finset.sum_congr _ (fun _ _ => rfl)
    ext x
    simp only [Finset.mem_union, Finset.mem_filter]
    constructor
    · intro h
      have := ((mem_hybrid _ _ _).1 h i).2 le_rfl
      rcases this with h0 | h1
      · exact Or.inl ⟨h, h0⟩
      · exact Or.inr ⟨h, h1⟩
    · rintro (⟨h, _⟩ | ⟨h, _⟩) <;> exact h
  · rw [Finset.disjoint_left]
    intro x hx hx'
    simp only [Finset.mem_filter] at hx hx'
    exact zero_ne_one (hx.2.symm.trans hx'.2)

theorem psum_zero (P : (Fin n → F) → F) (r : Fin n → F) : psum P 0 r = cubeSum P := by
  simp [psum, hybrid, cubeSum]

theorem psum_last (P : (Fin n → F) → F) (r : Fin n → F) : psum P n r = P r := by
  have h : hybrid n r = {r} := by
    unfold hybrid
    have : (fun j : Fin n => if (j : ℕ) < n then ({r j} : Finset F) else ({0, 1} : Finset F))
        = fun j => {r j} := by funext j; simp [j.isLt]
    rw [this, Fintype.piFinset_singleton]
  simp [psum, h]

/-- Each honest round message has univariate degree at most `d`. -/
def RoundDeg (d : ℕ) (P : (Fin n → F) → F) : Prop :=
  ∀ (i : Fin n) (r : Fin n → F), ∃ p : Polynomial F, p.natDegree ≤ d ∧
    ∀ x, p.eval x = psum P (i + 1) (Function.update r i x)

/-- Acceptance predicate: round-sum consistency at `{0,1}`, claim folding, and final evaluation `c_n = P(r)`. -/
def Accepts (P : (Fin n → F) → F) (S : F) (g : Fin n → (Fin n → F) → Polynomial F)
    (r : Fin n → F) : Prop :=
  ∃ c : ℕ → F, c 0 = S ∧
    (∀ i : Fin n, (g i r).eval 0 + (g i r).eval 1 = c i ∧ c (i + 1) = (g i r).eval (r i)) ∧
    c n = P r

/-- Bad challenge set at round `i`: evaluation points where a dishonest round polynomial agrees with the true partial sum. -/
noncomputable def badSet (P : (Fin n → F) → F) (g : Fin n → (Fin n → F) → Polynomial F)
    (i : Fin n) (r : Fin n → F) : Finset F := by
  classical exact
    if ∀ x, (g i r).eval x = psum P (i + 1) (Function.update r i x) then ∅
    else Finset.univ.filter fun x => (g i r).eval x = psum P (i + 1) (Function.update r i x)

theorem exists_bad_of_accepts (P : (Fin n → F) → F) (S : F) (hS : S ≠ cubeSum P)
    (g : Fin n → (Fin n → F) → Polynomial F) (r : Fin n → F) (h : Accepts P S g r) :
    ∃ i, r i ∈ badSet P g i r := by
  classical
  obtain ⟨c, h0, hstep, hend⟩ := h
  by_contra hcon
  push Not at hcon
  have key : ∀ m : ℕ, m ≤ n → c m ≠ psum P m r := by
    intro m
    induction m with
    | zero => intro _; rw [h0, psum_zero]; exact hS
    | succ m ih =>
      intro hm
      have hmn : m < n := hm
      have ihm := ih (Nat.le_of_lt hmn)
      intro hcm
      set i : Fin n := ⟨m, hmn⟩ with hi
      obtain ⟨hchk, hrec⟩ := hstep i
      have hrec' : c (m + 1) = (g i r).eval (r i) := hrec
      have hbad := hcon i
      unfold badSet at hbad
      have hupd : Function.update r i (r i) = r := Function.update_eq_self i r
      have hall : ∀ x, (g i r).eval x = psum P (i + 1) (Function.update r i x) := by
        by_contra hna
        rw [if_neg hna] at hbad
        apply hbad
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        have : psum P ((i : ℕ) + 1) (Function.update r i (r i)) = psum P (m + 1) r := by rw [hupd]
        rw [this, ← hcm, hrec']
      apply ihm
      have hsplit := psum_split P i r
      rw [← hchk, hall 0, hall 1]
      exact hsplit.symm
  have := key n le_rfl
  rw [hend, psum_last] at this
  exact this rfl

theorem badSet_card_le (P : (Fin n → F) → F) (d : ℕ) (hP : RoundDeg d P)
    (g : Fin n → (Fin n → F) → Polynomial F) (hg : ∀ i r, (g i r).natDegree ≤ d)
    (i : Fin n) (r : Fin n → F) : (badSet P g i r).card ≤ d := by
  classical
  unfold badSet
  split_ifs with hall
  · simp
  · obtain ⟨p, hpd, hp⟩ := hP i r
    have hne : g i r ≠ p := by
      intro e; apply hall; intro x; rw [e, hp]
    have := agree_card_le (g i r) p hne d (hg i r) hpd
    convert this using 2
    ext x; simp [hp]

theorem badSet_ind (P : (Fin n → F) → F) (g : Fin n → (Fin n → F) → Polynomial F)
    (hind : ∀ i r x, g i (Function.update r i x) = g i r) (i : Fin n) (r : Fin n → F) (x : F) :
    badSet P g i (Function.update r i x) = badSet P g i r := by
  unfold badSet
  rw [hind]
  simp [Function.update_idem]

theorem sumcheck_sound (P : (Fin n → F) → F) (d : ℕ) (hP : RoundDeg d P) (S : F)
    (hS : S ≠ cubeSum P) (g : Fin n → (Fin n → F) → Polynomial F)
    [DecidablePred fun r : Fin n → F => Accepts P S g r]
    (hind : ∀ i r x, g i (Function.update r i x) = g i r) (hg : ∀ i r, (g i r).natDegree ≤ d) :
    (Finset.univ.filter fun r : Fin n → F => Accepts P S g r).card * Fintype.card F
      ≤ n * d * Fintype.card F ^ n := by
  classical
  refine le_trans (Nat.mul_le_mul_right _ (Finset.card_le_card ?_))
    (sumcheck_bad_challenges (badSet P g) d (badSet_ind P g hind) (badSet_card_le P d hP g hg))
  intro r hr
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hr ⊢
  exact exists_bad_of_accepts P S hS g r hr

/-- Individual degree at most `d` in each variable. -/
def IndivDeg (d : ℕ) (P : (Fin n → F) → F) : Prop :=
  ∀ (i : Fin n) (r : Fin n → F), ∃ p : Polynomial F, p.natDegree ≤ d ∧
    ∀ x, p.eval x = P (Function.update r i x)

theorem roundDeg_of_indivDeg (d : ℕ) (P : (Fin n → F) → F) (hP : IndivDeg d P) : RoundDeg d P := by
  classical
  intro i r
  set Z := hybrid (i + 1) (Function.update r i 0) with hZ
  choose pz hpzd hpz using fun z : Fin n → F => hP i z
  refine ⟨∑ z ∈ Z, pz z, ?_, ?_⟩
  · exact Polynomial.natDegree_sum_le_of_forall_le _ _ (fun z _ => hpzd z)
  · intro x
    rw [Polynomial.eval_finset_sum]
    simp only [hpz]
    unfold psum
    refine Finset.sum_nbij' (fun z => Function.update z i x) (fun y => Function.update y i 0)
      ?_ ?_ ?_ ?_ (fun _ _ => rfl)
    · intro z hz
      rw [hZ, mem_hybrid] at hz
      rw [mem_hybrid]
      intro j
      have hj := hz j
      rcases lt_trichotomy (j : ℕ) i with hl | he | hg
      · have hji : j ≠ i := fun e => by simp [e] at hl
        refine ⟨fun _ => ?_, fun _ => by omega⟩
        have := hj.1 (by omega)
        simpa [Function.update_of_ne hji] using this
      · have : j = i := Fin.ext he
        subst this
        exact ⟨fun _ => by simp, fun h' => by omega⟩
      · have hji : j ≠ i := fun e => by simp [e] at hg
        refine ⟨fun h' => by omega, fun _ => ?_⟩
        simpa [Function.update_of_ne hji] using hj.2 (by omega)
    · intro y hy
      rw [mem_hybrid] at hy
      rw [hZ, mem_hybrid]
      intro j
      have hj := hy j
      rcases lt_trichotomy (j : ℕ) i with hl | he | hg
      · have hji : j ≠ i := fun e => by simp [e] at hl
        refine ⟨fun _ => ?_, fun _ => by omega⟩
        have := hj.1 (by omega)
        simpa [Function.update_of_ne hji] using this
      · have : j = i := Fin.ext he
        subst this
        exact ⟨fun _ => by simp, fun h' => by omega⟩
      · have hji : j ≠ i := fun e => by simp [e] at hg
        refine ⟨fun h' => by omega, fun _ => ?_⟩
        simpa [Function.update_of_ne hji] using hj.2 (by omega)
    · intro z hz
      rw [hZ, mem_hybrid] at hz
      have := (hz i).1 (by omega)
      simp [this]
    · intro y hy
      rw [mem_hybrid] at hy
      have := (hy i).1 (by omega)
      simp [this]

theorem sumcheck_sound_real (P : (Fin n → F) → F) (d : ℕ) (hP : RoundDeg d P) (S : F)
    (hS : S ≠ cubeSum P) (g : Fin n → (Fin n → F) → Polynomial F)
    [DecidablePred fun r : Fin n → F => Accepts P S g r]
    (hind : ∀ i r x, g i (Function.update r i x) = g i r) (hg : ∀ i r, (g i r).natDegree ≤ d) :
    ((Finset.univ.filter fun r : Fin n → F => Accepts P S g r).card : ℝ)
      ≤ ((n : ℝ) * d / Fintype.card F) * (Fintype.card (Fin n → F) : ℝ) := by
  have h := sumcheck_sound P d hP S hS g hind hg
  have hpos : (0 : ℝ) < Fintype.card F := Nat.cast_pos.mpr Fintype.card_pos
  have h' : ((Finset.univ.filter fun r : Fin n → F => Accepts P S g r).card : ℝ) * Fintype.card F
      ≤ n * d * (Fintype.card F : ℝ) ^ n := by exact_mod_cast h
  rw [Fintype.card_fun, Fintype.card_fin, Nat.cast_pow, div_mul_eq_mul_div, le_div_iff₀ hpos]
  exact h'

end FlashProveNext

#print axioms FlashProveNext.psum_split
#print axioms FlashProveNext.roundDeg_of_indivDeg
#print axioms FlashProveNext.exists_bad_of_accepts
#print axioms FlashProveNext.sumcheck_sound
#print axioms FlashProveNext.sumcheck_sound_real
