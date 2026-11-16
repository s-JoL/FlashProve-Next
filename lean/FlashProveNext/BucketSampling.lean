/-
Integer bucketed keyed sampling (`fpn-samp-bk-v3`): witness-free nucleus sandwich, `j`-order
prefix monotonicity, zero-slack boundary-pair circuit compilation, and relaxation completeness
(`specs/06-tolerance-and-cheats.md` §3.1–§3.2).

1. `nucleus_iff_per_token`, `argmax_mem_nucleus` (Lemma 6.7a):
   threshold top-`p` nucleus equals per-token check `0 < w v ∧ (2^16 - P16) · A(v) < P16 · R(v)`
   for all `1 ≤ P16 ≤ 2^16` (including `P16 = 2^16`) and always contains the positive-weight argmax;
2. `sure`, `poss`, `nucleus_sandwich` (Lemma 6.7'):
   witness-free tight envelopes satisfy `N^lo ⊆ N_hat ⊆ N^hi` for every engine weight vector `w^lo ≤ w_hat ≤ w^hi`;
3. `sure_mono_j`, `poss_mono_j` (Lemma 6.7''), `prefix_boundary_exists`, `boundary_pair_sound_complete`:
   downward closure in table index `j_v` when `E` is antitone, existence of a boundary witness, and
   exact if-and-only-if circuit characterization (zero prover slack);
4. `bucket_sampling_complete` (Theorem 6.8):
   honest bucketed Gumbel-max bucket selection + 48-bit inverse CDF always passes all four verifier checks.
-/
import Mathlib

namespace FlashProveNext

open Finset

set_option linter.unusedSectionVars false

section Nucleus

variable {V : Type*} [Fintype V] [DecidableEq V]

abbrev PScale : ℕ := 65536

def totalWeight (w : V → ℕ) : ℕ := ∑ v, w v

def nucTarget (S P16 : ℕ) : ℕ := (S * P16 + (PScale - 1)) / PScale

def massGe (w : V → ℕ) (θ : ℕ) : ℕ := ∑ u ∈ univ.filter (fun u => θ ≤ w u), w u

def strictAbove (w : V → ℕ) (v : V) : ℕ := ∑ u ∈ univ.filter (fun u => w v < w u), w u

def restAtOrBelow (w : V → ℕ) (v : V) : ℕ := ∑ u ∈ univ.filter (fun u => w u ≤ w v), w u

def inThresholdNucleus (w : V → ℕ) (P16 : ℕ) (v : V) : Prop :=
  0 < w v ∧ ∀ θ : ℕ, w v < θ → massGe w θ < nucTarget (totalWeight w) P16

def inPerTokenNucleus (w : V → ℕ) (P16 : ℕ) (v : V) : Prop :=
  0 < w v ∧ (PScale - P16) * strictAbove w v < P16 * restAtOrBelow w v

theorem lt_tgt_iff (a S P16 : ℕ) : a < nucTarget S P16 ↔ PScale * a < S * P16 := by
  unfold nucTarget PScale
  omega

theorem strictAbove_add_rest (w : V → ℕ) (v : V) :
    strictAbove w v + restAtOrBelow w v = totalWeight w := by
  unfold strictAbove restAtOrBelow totalWeight
  have hdisj : Disjoint (univ.filter fun u => w v < w u) (univ.filter fun u => w u ≤ w v) := by
    rw [disjoint_left]
    intro u h1 h2
    simp only [mem_filter, mem_univ, true_and] at h1 h2
    omega
  rw [← sum_union hdisj]
  congr 1
  ext u
  simp only [mem_union, mem_filter, mem_univ, true_and, iff_true]
  exact lt_or_ge (w v) (w u)

theorem massGe_anti (w : V → ℕ) {θ₁ θ₂ : ℕ} (h : θ₁ ≤ θ₂) : massGe w θ₂ ≤ massGe w θ₁ := by
  unfold massGe
  refine sum_le_sum_of_subset fun u hu => ?_
  simp only [mem_filter, mem_univ, true_and] at hu ⊢
  omega

theorem massGe_succ_eq_strictAbove (w : V → ℕ) (v : V) :
    massGe w (w v + 1) = strictAbove w v := rfl

theorem nucleus_iff_per_token (w : V → ℕ) (P16 : ℕ) (hP : P16 ≤ PScale) (v : V) :
    inThresholdNucleus w P16 v ↔ inPerTokenNucleus w P16 v := by
  unfold inThresholdNucleus inPerTokenNucleus
  refine and_congr_right fun _ => ?_
  have hsum := strictAbove_add_rest w v
  constructor
  · intro hall
    have h1 := hall (w v + 1) (Nat.lt_succ_self _)
    rw [massGe_succ_eq_strictAbove, lt_tgt_iff, ← hsum] at h1
    have hmul : (strictAbove w v + restAtOrBelow w v) * P16
        = P16 * strictAbove w v + P16 * restAtOrBelow w v := by ring
    have hsub : (PScale - P16) * strictAbove w v + P16 * strictAbove w v
        = PScale * strictAbove w v := by
      rw [← Nat.add_mul, Nat.sub_add_cancel hP]
    omega
  · intro hineq θ hθ
    have hle : massGe w θ ≤ strictAbove w v := by
      rw [← massGe_succ_eq_strictAbove]
      exact massGe_anti w hθ
    rw [lt_tgt_iff, ← hsum]
    have hmul : (strictAbove w v + restAtOrBelow w v) * P16
        = P16 * strictAbove w v + P16 * restAtOrBelow w v := by ring
    have hsub : (PScale - P16) * strictAbove w v + P16 * strictAbove w v
        = PScale * strictAbove w v := by
      rw [← Nat.add_mul, Nat.sub_add_cancel hP]
    have hA : PScale * strictAbove w v < (strictAbove w v + restAtOrBelow w v) * P16 := by omega
    exact Nat.lt_of_le_of_lt (Nat.mul_le_mul_left PScale hle) hA

theorem argmax_mem_nucleus (w : V → ℕ) (P16 : ℕ) (hP0 : 1 ≤ P16) (v : V)
    (hpos : 0 < w v) (hmax : ∀ u, w u ≤ w v) :
    inPerTokenNucleus w P16 v := by
  refine ⟨hpos, ?_⟩
  have hA : strictAbove w v = 0 := by
    unfold strictAbove
    refine sum_eq_zero fun u hu => ?_
    simp only [mem_filter, mem_univ, true_and] at hu
    exact absurd (hmax u) (not_le_of_gt hu)
  have hR : w v ≤ restAtOrBelow w v := by
    unfold restAtOrBelow
    exact single_le_sum (fun _ _ => Nat.zero_le _) (mem_filter.mpr ⟨mem_univ v, le_rfl⟩)
  rw [hA, Nat.mul_zero]
  exact Nat.mul_pos hP0 (lt_of_lt_of_le hpos hR)

def sureAbove (wlo whi : V → ℕ) (v : V) : ℕ :=
  ∑ u ∈ univ.filter (fun u => u ≠ v ∧ wlo v < whi u), whi u

def sureRest (wlo whi : V → ℕ) (v : V) : ℕ :=
  wlo v + ∑ u ∈ univ.filter (fun u => u ≠ v ∧ whi u ≤ wlo v), wlo u

def sure (wlo whi : V → ℕ) (P16 : ℕ) (v : V) : Prop :=
  0 < wlo v ∧ (PScale - P16) * sureAbove wlo whi v < P16 * sureRest wlo whi v

def possAbove (wlo whi : V → ℕ) (v : V) : ℕ :=
  ∑ u ∈ univ.filter (fun u => whi v < wlo u), wlo u

def possRest (wlo whi : V → ℕ) (v : V) : ℕ :=
  ∑ u ∈ univ.filter (fun u => wlo u ≤ whi v), whi u

def poss (wlo whi : V → ℕ) (P16 : ℕ) (v : V) : Prop :=
  0 < whi v ∧ (PScale - P16) * possAbove wlo whi v < P16 * possRest wlo whi v

theorem sure_iff_pos_of_full (wlo whi : V → ℕ) (v : V) :
    sure wlo whi PScale v ↔ 0 < wlo v := by
  unfold sure sureRest
  constructor
  · exact And.left
  · intro h
    refine ⟨h, ?_⟩
    simp only [Nat.sub_self, Nat.zero_mul]
    have : 0 < PScale := by decide
    exact Nat.mul_pos this (by omega)

theorem poss_iff_pos_of_full (wlo whi : V → ℕ) (v : V) (hlh : wlo v ≤ whi v) :
    poss wlo whi PScale v ↔ 0 < whi v := by
  unfold poss
  constructor
  · exact And.left
  · intro h
    refine ⟨h, ?_⟩
    simp only [Nat.sub_self, Nat.zero_mul]
    have hv : whi v ≤ possRest wlo whi v :=
      single_le_sum (fun _ _ => Nat.zero_le _) (mem_filter.mpr ⟨mem_univ v, hlh⟩)
    have : 0 < PScale := by decide
    exact Nat.mul_pos this (lt_of_lt_of_le h hv)

theorem sure_imp_inNucleus (wlo what whi : V → ℕ) (P16 : ℕ)
    (hlo : ∀ u, wlo u ≤ what u) (hhi : ∀ u, what u ≤ whi u)
    {v : V} (hs : sure wlo whi P16 v) :
    inPerTokenNucleus what P16 v := by
  rcases hs with ⟨hwpos, hineq⟩
  refine ⟨lt_of_lt_of_le hwpos (hlo v), ?_⟩
  have hA : strictAbove what v ≤ sureAbove wlo whi v := by
    unfold strictAbove sureAbove
    calc ∑ u ∈ univ.filter (fun u => what v < what u), what u
        ≤ ∑ u ∈ univ.filter (fun u => what v < what u), whi u :=
          sum_le_sum fun u _ => hhi u
      _ ≤ ∑ u ∈ univ.filter (fun u => u ≠ v ∧ wlo v < whi u), whi u := by
          refine sum_le_sum_of_subset fun u hu => ?_
          simp only [mem_filter, mem_univ, true_and] at hu ⊢
          refine ⟨fun heq => by subst heq; omega, lt_of_le_of_lt (hlo v) (lt_of_lt_of_le hu (hhi u))⟩
  have hR : sureRest wlo whi v ≤ restAtOrBelow what v := by
    unfold sureRest restAtOrBelow
    have hv_mem : v ∈ univ.filter (fun u => what u ≤ what v) := mem_filter.mpr ⟨mem_univ v, le_rfl⟩
    rw [← add_sum_erase _ _ hv_mem]
    refine Nat.add_le_add (hlo v) ?_
    calc ∑ u ∈ univ.filter (fun u => u ≠ v ∧ whi u ≤ wlo v), wlo u
        ≤ ∑ u ∈ univ.filter (fun u => u ≠ v ∧ whi u ≤ wlo v), what u :=
          sum_le_sum fun u _ => hlo u
      _ ≤ ∑ u ∈ (univ.filter (fun u => what u ≤ what v)).erase v, what u := by
          refine sum_le_sum_of_subset fun u hu => ?_
          simp only [mem_filter, mem_univ, true_and, mem_erase] at hu ⊢
          exact ⟨hu.1, (hhi u).trans (hu.2.trans (hlo v))⟩
  calc (PScale - P16) * strictAbove what v
      ≤ (PScale - P16) * sureAbove wlo whi v := Nat.mul_le_mul_left _ hA
    _ < P16 * sureRest wlo whi v := hineq
    _ ≤ P16 * restAtOrBelow what v := Nat.mul_le_mul_left _ hR

theorem inNucleus_imp_poss (wlo what whi : V → ℕ) (P16 : ℕ)
    (hlo : ∀ u, wlo u ≤ what u) (hhi : ∀ u, what u ≤ whi u)
    {v : V} (hn : inPerTokenNucleus what P16 v) :
    poss wlo whi P16 v := by
  rcases hn with ⟨hwpos, hineq⟩
  refine ⟨lt_of_lt_of_le hwpos (hhi v), ?_⟩
  have hA : possAbove wlo whi v ≤ strictAbove what v := by
    unfold possAbove strictAbove
    calc ∑ u ∈ univ.filter (fun u => whi v < wlo u), wlo u
        ≤ ∑ u ∈ univ.filter (fun u => whi v < wlo u), what u :=
          sum_le_sum fun u _ => hlo u
      _ ≤ ∑ u ∈ univ.filter (fun u => what v < what u), what u := by
          refine sum_le_sum_of_subset fun u hu => ?_
          simp only [mem_filter, mem_univ, true_and] at hu ⊢
          exact lt_of_le_of_lt (hhi v) (lt_of_lt_of_le hu (hlo u))
  have hR : restAtOrBelow what v ≤ possRest wlo whi v := by
    unfold restAtOrBelow possRest
    calc ∑ u ∈ univ.filter (fun u => what u ≤ what v), what u
        ≤ ∑ u ∈ univ.filter (fun u => what u ≤ what v), whi u :=
          sum_le_sum fun u _ => hhi u
      _ ≤ ∑ u ∈ univ.filter (fun u => wlo u ≤ whi v), whi u := by
          refine sum_le_sum_of_subset fun u hu => ?_
          simp only [mem_filter, mem_univ, true_and] at hu ⊢
          exact (hlo u).trans (hu.trans (hhi v))
  calc (PScale - P16) * possAbove wlo whi v
      ≤ (PScale - P16) * strictAbove what v := Nat.mul_le_mul_left _ hA
    _ < P16 * restAtOrBelow what v := hineq
    _ ≤ P16 * possRest wlo whi v := Nat.mul_le_mul_left _ hR

theorem nucleus_sandwich (wlo what whi : V → ℕ) (P16 : ℕ) (hP : P16 ≤ PScale)
    (hlo : ∀ u, wlo u ≤ what u) (hhi : ∀ u, what u ≤ whi u) (v : V) :
    (sure wlo whi P16 v → inThresholdNucleus what P16 v) ∧
    (inThresholdNucleus what P16 v → poss wlo whi P16 v) := by
  rw [nucleus_iff_per_token what P16 hP v]
  exact ⟨sure_imp_inNucleus wlo what whi P16 hlo hhi,
    inNucleus_imp_poss wlo what whi P16 hlo hhi⟩

theorem sure_imp_poss (wlo whi : V → ℕ) (P16 : ℕ) (hlh : ∀ u, wlo u ≤ whi u)
    {v : V} (hs : sure wlo whi P16 v) : poss wlo whi P16 v :=
  inNucleus_imp_poss wlo wlo whi P16 (fun _ => le_rfl) hlh
    (sure_imp_inNucleus wlo wlo whi P16 (fun _ => le_rfl) hlh hs)

theorem poss_mono_weights (wlo whi : V → ℕ) (P16 : ℕ) {v v' : V}
    (hhi : whi v' ≤ whi v) (hp : poss wlo whi P16 v') : poss wlo whi P16 v := by
  rcases hp with ⟨hpos, hineq⟩
  refine ⟨lt_of_lt_of_le hpos hhi, ?_⟩
  have hA : possAbove wlo whi v ≤ possAbove wlo whi v' := by
    unfold possAbove
    refine sum_le_sum_of_subset fun u hu => ?_
    simp only [mem_filter, mem_univ, true_and] at hu ⊢
    omega
  have hR : possRest wlo whi v' ≤ possRest wlo whi v := by
    unfold possRest
    refine sum_le_sum_of_subset fun u hu => ?_
    simp only [mem_filter, mem_univ, true_and] at hu ⊢
    omega
  calc (PScale - P16) * possAbove wlo whi v
      ≤ (PScale - P16) * possAbove wlo whi v' := Nat.mul_le_mul_left _ hA
    _ < P16 * possRest wlo whi v' := hineq
    _ ≤ P16 * possRest wlo whi v := Nat.mul_le_mul_left _ hR

theorem sure_mono_weights (wlo whi : V → ℕ) (P16 : ℕ) (hlh : ∀ u, wlo u ≤ whi u)
    {v v' : V} (hlo : wlo v' ≤ wlo v) (hhi : whi v' ≤ whi v)
    (hs : sure wlo whi P16 v') : sure wlo whi P16 v := by
  by_cases hvv : v = v'
  · subst hvv; exact hs
  rcases hs with ⟨hpos, hineq⟩
  refine ⟨lt_of_lt_of_le hpos hlo, ?_⟩
  let restU : Finset V := univ.filter (fun u => u ≠ v ∧ u ≠ v')
  have hA_split_v : sureAbove wlo whi v =
      (if wlo v < whi v' then whi v' else 0) +
      ∑ u ∈ restU.filter (fun u => wlo v < whi u), whi u := by
    unfold sureAbove
    have hpart : univ.filter (fun u => u ≠ v ∧ wlo v < whi u) =
        (if wlo v < whi v' then {v'} else ∅) ∪ restU.filter (fun u => wlo v < whi u) := by
      ext u
      simp only [restU, mem_filter, mem_univ, true_and, mem_union]
      by_cases huv' : u = v'
      · subst huv'
        split_ifs with hlt <;> simp [Ne.symm hvv, hlt]
      · split_ifs with hlt <;> simp [huv']
    have hdisj : Disjoint (if wlo v < whi v' then ({v'} : Finset V) else ∅)
        (restU.filter (fun u => wlo v < whi u)) := by
      rw [disjoint_left]
      intro u hu1 hu2
      split_ifs at hu1
      · rw [mem_singleton.mp hu1] at hu2; simp [restU] at hu2
      · simp at hu1
    rw [hpart, sum_union hdisj]
    split_ifs <;> simp
  have hA_split_v' : sureAbove wlo whi v' =
      (if wlo v' < whi v then whi v else 0) +
      ∑ u ∈ restU.filter (fun u => wlo v' < whi u), whi u := by
    unfold sureAbove
    have hpart : univ.filter (fun u => u ≠ v' ∧ wlo v' < whi u) =
        (if wlo v' < whi v then {v} else ∅) ∪ restU.filter (fun u => wlo v' < whi u) := by
      ext u
      simp only [restU, mem_filter, mem_univ, true_and, mem_union]
      by_cases huv : u = v
      · subst huv
        split_ifs with hlt <;> simp [hvv, hlt]
      · split_ifs <;> simp [huv]
    have hdisj : Disjoint (if wlo v' < whi v then ({v} : Finset V) else ∅)
        (restU.filter (fun u => wlo v' < whi u)) := by
      rw [disjoint_left]
      intro u hu1 hu2
      split_ifs at hu1
      · rw [mem_singleton.mp hu1] at hu2; simp [restU] at hu2
      · simp at hu1
    rw [hpart, sum_union hdisj]
    split_ifs <;> simp
  have hR_split_v : sureRest wlo whi v =
      wlo v + (if whi v' ≤ wlo v then wlo v' else 0) +
      ∑ u ∈ restU.filter (fun u => whi u ≤ wlo v), wlo u := by
    unfold sureRest
    have hpart : univ.filter (fun u => u ≠ v ∧ whi u ≤ wlo v) =
        (if whi v' ≤ wlo v then {v'} else ∅) ∪ restU.filter (fun u => whi u ≤ wlo v) := by
      ext u
      simp only [restU, mem_filter, mem_univ, true_and, mem_union]
      by_cases huv' : u = v'
      · subst huv'
        split_ifs with hle <;> simp [Ne.symm hvv, hle]
      · split_ifs <;> simp [huv']
    have hdisj : Disjoint (if whi v' ≤ wlo v then ({v'} : Finset V) else ∅)
        (restU.filter (fun u => whi u ≤ wlo v)) := by
      rw [disjoint_left]
      intro u hu1 hu2
      split_ifs at hu1
      · rw [mem_singleton.mp hu1] at hu2; simp [restU] at hu2
      · simp at hu1
    rw [hpart, sum_union hdisj, Nat.add_assoc]
    split_ifs <;> simp
  have hR_split_v' : sureRest wlo whi v' =
      wlo v' + (if whi v ≤ wlo v' then wlo v else 0) +
      ∑ u ∈ restU.filter (fun u => whi u ≤ wlo v'), wlo u := by
    unfold sureRest
    have hpart : univ.filter (fun u => u ≠ v' ∧ whi u ≤ wlo v') =
        (if whi v ≤ wlo v' then {v} else ∅) ∪ restU.filter (fun u => whi u ≤ wlo v') := by
      ext u
      simp only [restU, mem_filter, mem_univ, true_and, mem_union]
      by_cases huv : u = v
      · subst huv
        split_ifs with hle <;> simp [hvv, hle]
      · split_ifs <;> simp [huv]
    have hdisj : Disjoint (if whi v ≤ wlo v' then ({v} : Finset V) else ∅)
        (restU.filter (fun u => whi u ≤ wlo v')) := by
      rw [disjoint_left]
      intro u hu1 hu2
      split_ifs at hu1
      · rw [mem_singleton.mp hu1] at hu2; simp [restU] at hu2
      · simp at hu1
    rw [hpart, sum_union hdisj, Nat.add_assoc]
    split_ifs <;> simp
  have hA_rest : ∑ u ∈ restU.filter (fun u => wlo v < whi u), whi u
      ≤ ∑ u ∈ restU.filter (fun u => wlo v' < whi u), whi u :=
    sum_le_sum_of_subset fun u hu => by
      simp only [mem_filter] at hu ⊢
      exact ⟨hu.1, lt_of_le_of_lt hlo hu.2⟩
  have hR_rest : ∑ u ∈ restU.filter (fun u => whi u ≤ wlo v'), wlo u
      ≤ ∑ u ∈ restU.filter (fun u => whi u ≤ wlo v), wlo u :=
    sum_le_sum_of_subset fun u hu => by
      simp only [mem_filter] at hu ⊢
      exact ⟨hu.1, hu.2.trans hlo⟩
  have hA_pair : (if wlo v < whi v' then whi v' else 0) ≤ (if wlo v' < whi v then whi v else 0) := by
    have hv1 := hlh v
    have hv2 := hlh v'
    split_ifs <;> omega
  have hR_pair : wlo v' + (if whi v ≤ wlo v' then wlo v else 0)
      ≤ wlo v + (if whi v' ≤ wlo v then wlo v' else 0) := by
    have hv1 := hlh v
    have hv2 := hlh v'
    split_ifs <;> omega
  have hA : sureAbove wlo whi v ≤ sureAbove wlo whi v' := by
    rw [hA_split_v, hA_split_v']
    exact Nat.add_le_add hA_pair hA_rest
  have hR : sureRest wlo whi v' ≤ sureRest wlo whi v := by
    rw [hR_split_v, hR_split_v']
    exact Nat.add_le_add hR_pair hR_rest
  calc (PScale - P16) * sureAbove wlo whi v
      ≤ (PScale - P16) * sureAbove wlo whi v' := Nat.mul_le_mul_left _ hA
    _ < P16 * sureRest wlo whi v' := hineq
    _ ≤ P16 * sureRest wlo whi v := Nat.mul_le_mul_left _ hR

def bandLo (E : ℤ → ℕ) (a : V → Bool) (j : V → ℤ) (Δ : ℤ) (u : V) : ℕ :=
  if a u then E (j u + Δ) else 0

def bandHi (E : ℤ → ℕ) (a : V → Bool) (j : V → ℤ) (Δ : ℤ) (u : V) : ℕ :=
  if a u then E (j u - Δ) else 0

theorem bandLo_le_bandHi (E : ℤ → ℕ) (hE : Antitone E) (a : V → Bool) (j : V → ℤ)
    {Δ : ℤ} (hΔ : 0 ≤ Δ) (u : V) :
    bandLo E a j Δ u ≤ bandHi E a j Δ u := by
  unfold bandLo bandHi
  split_ifs with ha
  · exact hE (by omega)
  · exact le_rfl

theorem sure_mono_j (E : ℤ → ℕ) (hE : Antitone E) (a : V → Bool) (j : V → ℤ)
    {Δ : ℤ} (hΔ : 0 ≤ Δ) (P16 : ℕ) {v v' : V}
    (hav : a v = true) (hav' : a v' = true) (hj : j v ≤ j v')
    (hs : sure (bandLo E a j Δ) (bandHi E a j Δ) P16 v') :
    sure (bandLo E a j Δ) (bandHi E a j Δ) P16 v := by
  refine sure_mono_weights (bandLo E a j Δ) (bandHi E a j Δ) P16
    (bandLo_le_bandHi E hE a j hΔ) ?_ ?_ hs
  · simp only [bandLo, hav, hav', ↓reduceIte]; exact hE (by omega)
  · simp only [bandHi, hav, hav', ↓reduceIte]; exact hE (by omega)

theorem poss_mono_j (E : ℤ → ℕ) (hE : Antitone E) (a : V → Bool) (j : V → ℤ)
    (Δ : ℤ) (P16 : ℕ) {v v' : V}
    (hav : a v = true) (hav' : a v' = true) (hj : j v ≤ j v')
    (hp : poss (bandLo E a j Δ) (bandHi E a j Δ) P16 v') :
    poss (bandLo E a j Δ) (bandHi E a j Δ) P16 v := by
  refine poss_mono_weights (bandLo E a j Δ) (bandHi E a j Δ) P16 ?_ hp
  simp only [bandHi, hav, hav', ↓reduceIte]; exact hE (by omega)

inductive BoundaryWitness (V : Type*) where
  | empty : BoundaryWitness V
  | full (v_in : V) : BoundaryWitness V
  | split (v_in v_out : V) : BoundaryWitness V

def ValidBoundaryCheck (a : V → Bool) (j : V → ℤ) (P : V → Prop)
    (w : BoundaryWitness V) (s : V → Bool) : Prop :=
  (∀ u, a u = false → s u = false) ∧
  match w with
  | .empty =>
      (∀ u, a u = true → ¬ P u) ∧ (∀ u, s u = false)
  | .full v_in =>
      a v_in = true ∧ P v_in ∧
      (∀ u, a u = true → j u ≤ j v_in) ∧
      (∀ u, a u = true → s u = true)
  | .split v_in v_out =>
      a v_in = true ∧ P v_in ∧
      a v_out = true ∧ ¬ P v_out ∧
      j v_in < j v_out ∧
      (∀ u, a u = true → (s u = true → j u ≤ j v_in) ∧ (s u = false → j v_out ≤ j u))

theorem prefix_boundary_exists (a : V → Bool) (j : V → ℤ) (P : V → Prop) [DecidablePred P]
    (haP : ∀ u, P u → a u = true)
    (hmono : ∀ u u', a u = true → a u' = true → j u ≤ j u' → P u' → P u) :
    ∃ w : BoundaryWitness V, ValidBoundaryCheck a j P w (fun u => decide (P u)) := by
  let S_in := univ.filter (fun u => P u)
  let S_out := univ.filter (fun u => a u = true ∧ ¬ P u)
  rcases S_in.eq_empty_or_nonempty with hemp_in | hne_in
  · refine ⟨.empty, ?_⟩
    refine ⟨fun u hu => by simp only [decide_eq_false_iff_not]; intro hp; rw [haP u hp] at hu; simp at hu,
      fun u _ hp => ?_, fun u => ?_⟩
    · have : u ∈ S_in := mem_filter.mpr ⟨mem_univ u, hp⟩
      simp [hemp_in] at this
    · simp only [decide_eq_false_iff_not]
      intro hp
      have : u ∈ S_in := mem_filter.mpr ⟨mem_univ u, hp⟩
      simp [hemp_in] at this
  · obtain ⟨v_in, hvin_mem, hvin_max⟩ := exists_max_image S_in j hne_in
    have hvinP : P v_in := (mem_filter.mp hvin_mem).2
    have hvinA : a v_in = true := haP v_in hvinP
    rcases S_out.eq_empty_or_nonempty with hemp_out | hne_out
    · refine ⟨.full v_in, ?_⟩
      refine ⟨fun u hu => by simp only [decide_eq_false_iff_not]; intro hp; rw [haP u hp] at hu; simp at hu,
        hvinA, hvinP, fun u hu => ?_, fun u hu => ?_⟩
      · have huP : P u := by
          by_contra hnp
          have : u ∈ S_out := mem_filter.mpr ⟨mem_univ u, hu, hnp⟩
          simp [hemp_out] at this
        exact hvin_max u (mem_filter.mpr ⟨mem_univ u, huP⟩)
      · simp only [decide_eq_true_eq]
        by_contra hnp
        have : u ∈ S_out := mem_filter.mpr ⟨mem_univ u, hu, hnp⟩
        simp [hemp_out] at this
    · obtain ⟨v_out, hvout_mem, hvout_min⟩ := exists_min_image S_out j hne_out
      rcases (mem_filter.mp hvout_mem).2 with ⟨hvoutA, hvoutNP⟩
      have hlt_sep : j v_in < j v_out := by
        by_contra hge
        exact hvoutNP (hmono v_out v_in hvoutA hvinA (le_of_not_gt hge) hvinP)
      refine ⟨.split v_in v_out, ?_⟩
      refine ⟨fun u hu => by simp only [decide_eq_false_iff_not]; intro hp; rw [haP u hp] at hu; simp at hu,
        hvinA, hvinP, hvoutA, hvoutNP, hlt_sep, fun u hu => ⟨?_, ?_⟩⟩
      · simp only [decide_eq_true_eq]
        intro huP
        exact hvin_max u (mem_filter.mpr ⟨mem_univ u, huP⟩)
      · simp only [decide_eq_false_iff_not]
        intro huNP
        exact hvout_min u (mem_filter.mpr ⟨mem_univ u, hu, huNP⟩)

theorem boundary_pair_sound_complete (a : V → Bool) (j : V → ℤ) (P : V → Prop) [DecidablePred P]
    (haP : ∀ u, P u → a u = true)
    (hmono : ∀ u u', a u = true → a u' = true → j u ≤ j u' → P u' → P u)
    (s : V → Bool) :
    (∃ w : BoundaryWitness V, ValidBoundaryCheck a j P w s) ↔ (∀ u, s u = true ↔ P u) := by
  constructor
  · rintro ⟨w, hmask, hw⟩ u
    cases hv : a u with
    | false =>
      rw [hmask u hv]
      simp only [Bool.false_eq_true, false_iff]
      intro hp; rw [haP u hp] at hv; simp at hv
    | true =>
      cases w with
      | empty =>
        rcases hw with ⟨hno, hs0⟩
        rw [hs0 u]
        simp [hno u hv]
      | full v_in =>
        rcases hw with ⟨hvinA, hvinP, hmax, hs1⟩
        rw [hs1 u hv]
        simp only [true_iff]
        exact hmono u v_in hv hvinA (hmax u hv) hvinP
      | split v_in v_out =>
        rcases hw with ⟨hvinA, hvinP, hvoutA, hvoutNP, _, hcls⟩
        rcases hcls u hv with ⟨h1, h0⟩
        cases hsu : s u with
        | true =>
          simp only [true_iff]
          exact hmono u v_in hv hvinA (h1 hsu) hvinP
        | false =>
          simp only [Bool.false_eq_true, false_iff]
          intro huP
          exact hvoutNP (hmono v_out u hvoutA hv (h0 hsu) huP)
  · intro hiff
    obtain ⟨w, hw⟩ := prefix_boundary_exists a j P haP hmono
    refine ⟨w, ?_⟩
    have hfun : s = fun u => decide (P u) := by
      funext u
      cases hsu : s u with
      | true =>
        symm
        rw [decide_eq_true_eq]
        exact (hiff u).mp hsu
      | false =>
        symm
        rw [decide_eq_false_iff_not]
        intro hp
        have := (hiff u).mpr hp
        simp [hsu] at this
    rwa [hfun]

end Nucleus

section BucketComplete

abbrev U48Scale : ℕ := 2 ^ 48

def bucketPrefix {β : ℕ} (X : Fin β → ℕ) (k : Fin β) : ℕ :=
  ∑ j ∈ univ.filter (fun j : Fin β => j ≤ k), X j

def bucketStrictPrefix {β : ℕ} (X : Fin β → ℕ) (k : Fin β) : ℕ :=
  ∑ j ∈ univ.filter (fun j : Fin β => j < k), X j

def bucketMass {β : ℕ} (X : Fin β → ℕ) : ℕ := ∑ j, X j

theorem bucketStrictPrefix_add_self {β : ℕ} (X : Fin β → ℕ) (k : Fin β) :
    bucketStrictPrefix X k + X k = bucketPrefix X k := by
  unfold bucketStrictPrefix bucketPrefix
  have hk : k ∈ univ.filter (fun j : Fin β => j ≤ k) := mem_filter.mpr ⟨mem_univ k, le_rfl⟩
  rw [← add_sum_erase _ _ hk, Nat.add_comm]
  refine congrArg (X k + ·) ?_
  congr 1
  ext j
  simp only [mem_filter, mem_univ, true_and, mem_erase]
  exact ⟨fun h => ⟨ne_of_lt h, le_of_lt h⟩, fun ⟨hne, hle⟩ => lt_of_le_of_ne hle hne⟩

theorem invCDF_exists {β : ℕ} (X : Fin β → ℕ) (tau : ℕ) (htau : tau < bucketMass X) :
    ∃ k : Fin β, 0 < X k ∧ bucketStrictPrefix X k ≤ tau ∧ tau < bucketPrefix X k := by
  have hβ : 0 < β := by
    by_contra h0
    have hz : β = 0 := by omega
    subst hz
    simp [bucketMass] at htau
  let S_hit := univ.filter (fun k : Fin β => tau < bucketPrefix X k)
  have hne : S_hit.Nonempty := by
    let last : Fin β := ⟨β - 1, by omega⟩
    have hlast : bucketPrefix X last = bucketMass X := by
      unfold bucketPrefix bucketMass
      congr 1
      ext j
      have hj : (j : ℕ) < β := j.isLt
      simp only [mem_filter, mem_univ, true_and, iff_true, Fin.le_iff_val_le_val, last]
      omega
    exact ⟨last, mem_filter.mpr ⟨mem_univ last, by rwa [hlast]⟩⟩
  obtain ⟨k, hk_mem, hk_min⟩ := exists_min_image S_hit id hne
  have hgt : tau < bucketPrefix X k := (mem_filter.mp hk_mem).2
  have hle : bucketStrictPrefix X k ≤ tau := by
    rcases eq_or_ne (k : ℕ) 0 with hk0 | hkne
    · have hemp : univ.filter (fun j : Fin β => j < k) = ∅ := by
        ext j; simp [Fin.lt_def, hk0]
      simp [bucketStrictPrefix, hemp]
    · let prev : Fin β := ⟨(k : ℕ) - 1, by omega⟩
      have hprev_eq : bucketStrictPrefix X k = bucketPrefix X prev := by
        unfold bucketStrictPrefix bucketPrefix
        congr 1
        ext j
        simp only [mem_filter, mem_univ, true_and, Fin.lt_def, Fin.le_iff_val_le_val, prev]
        omega
      rw [hprev_eq]
      by_contra hnotle
      have hprev_hit : prev ∈ S_hit := mem_filter.mpr ⟨mem_univ prev, lt_of_not_ge hnotle⟩
      have := hk_min prev hprev_hit
      simp only [id_eq, Fin.le_iff_val_le_val, prev] at this
      omega
  have hpos : 0 < X k := by
    have hadd := bucketStrictPrefix_add_self X k
    omega
  exact ⟨k, hpos, hle, hgt⟩

theorem bucket_sampling_complete {B β : ℕ}
    (wlo what whi : Fin B × Fin β → ℕ) (P16 : ℕ)
    (hlo : ∀ v, wlo v ≤ what v) (hhi : ∀ v, what v ≤ whi v)
    (LN : ℕ → ℤ) (hLN : Monotone LN) (gNoise : Fin B → ℤ)
    (u48 : ℕ) (hu48 : u48 < U48Scale) :
    let Xlo (b : Fin B) (k : Fin β) : ℕ :=
      haveI := Classical.decPred (sure wlo whi P16)
      if sure wlo whi P16 (b, k) then wlo (b, k) else 0
    let Xhat (b : Fin B) (k : Fin β) : ℕ :=
      haveI := Classical.decPred (inPerTokenNucleus what P16)
      if inPerTokenNucleus what P16 (b, k) then what (b, k) else 0
    let Xhi (b : Fin B) (k : Fin β) : ℕ :=
      haveI := Classical.decPred (poss wlo whi P16)
      if poss wlo whi P16 (b, k) then whi (b, k) else 0
    let Wlo (b : Fin B) := bucketMass (Xlo b)
    let What (b : Fin B) := bucketMass (Xhat b)
    let Whi (b : Fin B) := bucketMass (Xhi b)
    ∀ (b_hat : Fin B),
      0 < What b_hat →
      (∀ b : Fin B, 0 < What b → LN (What b) + gNoise b ≤ LN (What b_hat) + gNoise b_hat) →
      let tau := (u48 * What b_hat) / U48Scale
      ∃ k_hat : Fin β,
        0 < Xhat b_hat k_hat ∧
        bucketStrictPrefix (Xhat b_hat) k_hat ≤ tau ∧
        tau < bucketPrefix (Xhat b_hat) k_hat ∧
        poss wlo whi P16 (b_hat, k_hat) ∧
        (∀ b : Fin B, 0 < Wlo b → LN (Wlo b) + gNoise b ≤ LN (Whi b_hat) + gNoise b_hat) ∧
        bucketStrictPrefix (Xlo b_hat) k_hat * U48Scale ≤ u48 * Whi b_hat ∧
        u48 * Wlo b_hat < bucketPrefix (Xhi b_hat) k_hat * U48Scale := by
  classical
  intro Xlo Xhat Xhi Wlo What Whi b_hat hWhat_pos hb_max tau
  have hX_lo : ∀ b k, Xlo b k ≤ Xhat b k := by
    intro b k
    dsimp only [Xlo, Xhat]
    by_cases hs : sure wlo whi P16 (b, k)
    · have hn := sure_imp_inNucleus wlo what whi P16 hlo hhi hs
      simp [hs, hn, hlo (b, k)]
    · simp [hs]
  have hX_hi : ∀ b k, Xhat b k ≤ Xhi b k := by
    intro b k
    dsimp only [Xhat, Xhi]
    by_cases hn : inPerTokenNucleus what P16 (b, k)
    · have hp := inNucleus_imp_poss wlo what whi P16 hlo hhi hn
      simp [hn, hp, hhi (b, k)]
    · simp [hn]
  have hW_lo : ∀ b, Wlo b ≤ What b := fun b => sum_le_sum fun k _ => hX_lo b k
  have hW_hi : ∀ b, What b ≤ Whi b := fun b => sum_le_sum fun k _ => hX_hi b k
  have htau_lt : tau < What b_hat := by
    dsimp only [tau]
    have hmul : u48 * What b_hat < U48Scale * What b_hat :=
      Nat.mul_lt_mul_of_pos_right hu48 hWhat_pos
    exact Nat.div_lt_of_lt_mul hmul
  obtain ⟨k_hat, hXk_pos, hpref_lo, hpref_hi⟩ := invCDF_exists (Xhat b_hat) tau htau_lt
  refine ⟨k_hat, hXk_pos, hpref_lo, hpref_hi, ?_, ?_, ?_, ?_⟩
  · have hn : inPerTokenNucleus what P16 (b_hat, k_hat) := by
      dsimp only [Xhat] at hXk_pos
      by_contra hnot
      simp [hnot] at hXk_pos
    exact inNucleus_imp_poss wlo what whi P16 hlo hhi hn
  · intro b hWlo_pos
    have hWhat_b : 0 < What b := lt_of_lt_of_le hWlo_pos (hW_lo b)
    have h1 : LN (Wlo b) ≤ LN (What b) := hLN (hW_lo b)
    have h2 : LN (What b) + gNoise b ≤ LN (What b_hat) + gNoise b_hat := hb_max b hWhat_b
    have h3 : LN (What b_hat) ≤ LN (Whi b_hat) := hLN (hW_hi b_hat)
    linarith
  · have hS_lo : bucketStrictPrefix (Xlo b_hat) k_hat ≤ bucketStrictPrefix (Xhat b_hat) k_hat :=
      sum_le_sum fun j _ => hX_lo b_hat j
    have hdiv_mul : tau * U48Scale ≤ u48 * What b_hat := Nat.div_mul_le_self _ _
    calc bucketStrictPrefix (Xlo b_hat) k_hat * U48Scale
        ≤ bucketStrictPrefix (Xhat b_hat) k_hat * U48Scale := Nat.mul_le_mul_right _ hS_lo
      _ ≤ tau * U48Scale := Nat.mul_le_mul_right _ hpref_lo
      _ ≤ u48 * What b_hat := hdiv_mul
      _ ≤ u48 * Whi b_hat := Nat.mul_le_mul_left _ (hW_hi b_hat)
  · have hS_hi : bucketPrefix (Xhat b_hat) k_hat ≤ bucketPrefix (Xhi b_hat) k_hat :=
      sum_le_sum fun j _ => hX_hi b_hat j
    have hlt_mul : u48 * What b_hat < (tau + 1) * U48Scale := by
      dsimp only [tau]
      rw [Nat.add_mul, Nat.one_mul]
      have hmod := Nat.div_add_mod (u48 * What b_hat) U48Scale
      have hlt_rem : (u48 * What b_hat) % U48Scale < U48Scale :=
        Nat.mod_lt _ (by decide)
      have hcomm : U48Scale * ((u48 * What b_hat) / U48Scale)
          = ((u48 * What b_hat) / U48Scale) * U48Scale := Nat.mul_comm _ _
      omega
    calc u48 * Wlo b_hat
        ≤ u48 * What b_hat := Nat.mul_le_mul_left _ (hW_lo b_hat)
      _ < (tau + 1) * U48Scale := hlt_mul
      _ ≤ bucketPrefix (Xhat b_hat) k_hat * U48Scale := Nat.mul_le_mul_right _ hpref_hi
      _ ≤ bucketPrefix (Xhi b_hat) k_hat * U48Scale := Nat.mul_le_mul_right _ hS_hi

end BucketComplete

end FlashProveNext

#print axioms FlashProveNext.lt_tgt_iff
#print axioms FlashProveNext.strictAbove_add_rest
#print axioms FlashProveNext.nucleus_iff_per_token
#print axioms FlashProveNext.argmax_mem_nucleus
#print axioms FlashProveNext.sure_iff_pos_of_full
#print axioms FlashProveNext.poss_iff_pos_of_full
#print axioms FlashProveNext.sure_imp_inNucleus
#print axioms FlashProveNext.inNucleus_imp_poss
#print axioms FlashProveNext.nucleus_sandwich
#print axioms FlashProveNext.sure_imp_poss
#print axioms FlashProveNext.poss_mono_weights
#print axioms FlashProveNext.sure_mono_weights
#print axioms FlashProveNext.sure_mono_j
#print axioms FlashProveNext.poss_mono_j
#print axioms FlashProveNext.prefix_boundary_exists
#print axioms FlashProveNext.boundary_pair_sound_complete
#print axioms FlashProveNext.invCDF_exists
#print axioms FlashProveNext.bucket_sampling_complete
