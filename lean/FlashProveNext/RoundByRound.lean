/-
Round-by-round lifetime knowledge soundness, multi-stratum sampling with empty-stratum bypass,
multi-column LogUp soundness, and Goldilocks `F_{p^3}` budget arithmetic (`specs/02-soundness.md` §4–§8).

1. `rbr_lifetime_soundness`,
   `rbr_lifetime_soundness_sum` : lazy-sampled `Q`-query round-by-round bound `Pr[BadAccept] ≤ Δ + Q · ε̄`
                                  and adaptive expectation form `Δ + E[∑_j ε_{τ(j)}]`;
2. `multi_pool_map_le`,
   `three_pool_default`         : stratified cost-weighted sampling with empty-stratum bypass (`3 × 105 = 315` draws);
3. `multicol_badgamma_card_le`,
   `logup_multicol_card_le`     : tight `w`-column LogUp bound `(nt · (w - 1) + nv + nt) / |F|`;
4. `field_p2_iff`, `field_p3_bits`, `slot_table_numerator`, `default_budget_rbr`, `rbr_default_lifetime` :
   Goldilocks `F_{p^2}` vs. `F_{p^3}` degree thresholds and `Pr[BadAccept] ≤ 2^{-40}` at `Q ≤ 2^64`.
-/
import Mathlib
import FlashProveNext.LifetimeSoundness
import FlashProveNext.LogUp

namespace FlashProveNext

open Finset

set_option linter.unusedSectionVars false
set_option exponentiation.threshold 1024

open Classical in
theorem rbr_lifetime_soundness {Ω S ι : Type*} [Fintype Ω] [DecidableEq Ω] [Nonempty Ω] {Q : ℕ}
    (st : Fin Q → (Fin Q → Ω) → S)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a)
    (Doomed : S → Prop) (Esc : S → Finset Ω) (round : S → ι) (ε : ι → ℝ) (εbar : ℝ)
    (hε0 : 0 ≤ εbar) (hεbar : ∀ r, ε r ≤ εbar)
    (hEsc : ∀ s, Doomed s → ((Esc s).card : ℝ) ≤ ε (round s) * Fintype.card Ω)
    (BadAcc Fail : Finset (Fin Q → Ω)) (Δ : ℝ)
    (hFail : (Fail.card : ℝ) ≤ Δ * (Fintype.card Ω : ℝ) ^ Q)
    (hpath : ∀ a ∈ BadAcc, a ∈ Fail ∨ ∃ j, Doomed (st j a) ∧ a j ∈ Esc (st j a)) :
    (BadAcc.card : ℝ) ≤ (Δ + Q * εbar) * (Fintype.card Ω : ℝ) ^ Q := by
  have h := lifetime_union st hst Doomed Esc εbar Δ hε0
    (fun s hs => (hEsc s hs).trans (mul_le_mul_of_nonneg_right (hεbar _) (Nat.cast_nonneg _)))
    Fail hFail
  refine le_trans ?_ h
  refine Nat.cast_le.mpr (card_le_card fun a ha => ?_)
  rcases hpath a ha with hf | ⟨j, hd, he⟩
  · exact mem_union_left _ hf
  · exact mem_union_right _ (mem_filter.mpr ⟨mem_univ _, j, hd, he⟩)

theorem hit_sum_le {Ω : Type*} [Fintype Ω] [DecidableEq Ω] [Nonempty Ω] {n : ℕ} (j : Fin n)
    (B : (Fin n → Ω) → Finset Ω) (e : (Fin n → Ω) → ℝ)
    (hB : ∀ a x, B (Function.update a j x) = B a)
    (hBe : ∀ a, ((B a).card : ℝ) ≤ e a * Fintype.card Ω) :
    ((univ.filter fun a : Fin n → Ω => a j ∈ B a).card : ℝ) ≤ ∑ a, e a := by
  let Ψ : (Fin n → Ω) × Ω → (Fin n → Ω) × Ω := fun p => (Function.update p.1 j p.2, p.1 j)
  have hΨ : Function.Involutive Ψ := by
    intro p
    simp only [Ψ, Function.update_self, Function.update_idem, Function.update_eq_self]
  have hswap : (∑ p : (Fin n → Ω) × Ω, if p.2 ∈ B p.1 then (1:ℝ) else 0)
      = ∑ p : (Fin n → Ω) × Ω, if p.1 j ∈ B p.1 then (1:ℝ) else 0 := by
    refine Fintype.sum_equiv hΨ.toPerm _ _ fun p => ?_
    simp only [Function.Involutive.coe_toPerm, Ψ, Function.update_self, hB]
  have hL : (∑ p : (Fin n → Ω) × Ω, if p.1 j ∈ B p.1 then (1:ℝ) else 0)
      = ((univ.filter fun a : Fin n → Ω => a j ∈ B a).card : ℝ) * Fintype.card Ω := by
    rw [Fintype.sum_prod_type' (f := fun a (_ : Ω) => if a j ∈ B a then (1:ℝ) else 0)]
    simp only [sum_const, card_univ, nsmul_eq_mul]
    rw [← mul_sum, sum_boole, mul_comm]
  have hR : (∑ p : (Fin n → Ω) × Ω, if p.2 ∈ B p.1 then (1:ℝ) else 0)
      = ∑ a, ((B a).card : ℝ) := by
    rw [Fintype.sum_prod_type' (f := fun a x => if x ∈ B a then (1:ℝ) else 0)]
    refine sum_congr rfl fun a _ => ?_
    rw [sum_boole]
    congr 2
    ext x; simp
  have hpos : (0:ℝ) < Fintype.card Ω := Nat.cast_pos.mpr Fintype.card_pos
  have key : ((univ.filter fun a : Fin n → Ω => a j ∈ B a).card : ℝ) * Fintype.card Ω
      ≤ (∑ a, e a) * Fintype.card Ω := by
    rw [← hL, ← hswap, hR, sum_mul]
    exact sum_le_sum fun a _ => hBe a
  exact le_of_mul_le_mul_right key hpos

open Classical in
theorem rbr_lifetime_soundness_sum {Ω S ι : Type*} [Fintype Ω] [DecidableEq Ω] [Nonempty Ω]
    {Q : ℕ} (st : Fin Q → (Fin Q → Ω) → S)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a)
    (Doomed : S → Prop) (Esc : S → Finset Ω) (round : S → ι) (ε : ι → ℝ)
    (hEsc : ∀ s, Doomed s → ((Esc s).card : ℝ) ≤ ε (round s) * Fintype.card Ω)
    (BadAcc Fail : Finset (Fin Q → Ω)) (Δ : ℝ)
    (hFail : (Fail.card : ℝ) ≤ Δ * (Fintype.card Ω : ℝ) ^ Q)
    (hpath : ∀ a ∈ BadAcc, a ∈ Fail ∨ ∃ j, Doomed (st j a) ∧ a j ∈ Esc (st j a)) :
    (BadAcc.card : ℝ) ≤ Δ * (Fintype.card Ω : ℝ) ^ Q
      + ∑ a : Fin Q → Ω, ∑ j : Fin Q,
          (if Doomed (st j a) then ε (round (st j a)) else 0) := by
  let B : Fin Q → (Fin Q → Ω) → Finset Ω := fun j a => if Doomed (st j a) then Esc (st j a) else ∅
  let e : Fin Q → (Fin Q → Ω) → ℝ := fun j a => if Doomed (st j a) then ε (round (st j a)) else 0
  have hB : ∀ j a x, B j (Function.update a j x) = B j a := by
    intro j a x; simp only [B, hst]
  have hBe : ∀ j a, ((B j a).card : ℝ) ≤ e j a * Fintype.card Ω := by
    intro j a
    by_cases hd : Doomed (st j a)
    · simp only [B, e, if_pos hd]; exact hEsc _ hd
    · simp only [B, e, if_neg hd, card_empty, Nat.cast_zero, zero_mul, le_refl]
  have hsub : BadAcc ⊆ Fail ∪ univ.biUnion fun j => univ.filter fun a : Fin Q → Ω => a j ∈ B j a := by
    intro a ha
    rcases hpath a ha with hf | ⟨j, hd, he⟩
    · exact mem_union_left _ hf
    · refine mem_union_right _ (mem_biUnion.mpr ⟨j, mem_univ _, mem_filter.mpr ⟨mem_univ _, ?_⟩⟩)
      simp only [B, if_pos hd]; exact he
  have hcard : (BadAcc.card : ℝ) ≤ Fail.card
      + ∑ j, ((univ.filter fun a : Fin Q → Ω => a j ∈ B j a).card : ℝ) := by
    have h1 := (card_le_card hsub).trans ((card_union_le _ _).trans
      (Nat.add_le_add_left card_biUnion_le _))
    exact_mod_cast h1
  have hsum : ∑ j, ((univ.filter fun a : Fin Q → Ω => a j ∈ B j a).card : ℝ)
      ≤ ∑ a : Fin Q → Ω, ∑ j : Fin Q, e j a := by
    rw [sum_comm]
    exact sum_le_sum fun j _ => hit_sum_le j (B j) (e j) (hB j) (hBe j)
  calc (BadAcc.card : ℝ) ≤ Fail.card + ∑ j, ((univ.filter fun a : Fin Q → Ω => a j ∈ B j a).card : ℝ) :=
        hcard
    _ ≤ _ := add_le_add hFail hsum

theorem esc_preimage_le {Ω X : Type*} [Fintype Ω] [Fintype X] [DecidableEq X] (g : Ω → X) (c : ℕ)
    (hg : ∀ x, (univ.filter fun ω => g ω = x).card = c) (A : Finset X) (ε : ℝ)
    (hA : (A.card : ℝ) ≤ ε * Fintype.card X) :
    ((univ.filter fun ω => g ω ∈ A).card : ℝ) ≤ ε * Fintype.card Ω := by
  have hΩ : Fintype.card Ω = Fintype.card X * c := by
    rw [← card_univ (α := Ω), card_eq_sum_card_fiberwise (f := g) (t := univ)
      (fun _ _ => mem_univ _)]
    simp [hg]
  have hpre : (univ.filter fun ω => g ω ∈ A).card = A.card * c := by
    rw [card_eq_sum_card_fiberwise (f := g) (s := univ.filter fun ω => g ω ∈ A) (t := A)
      (by intro ω hω; simpa using hω)]
    rw [sum_congr rfl (g := fun _ => c)]
    · simp
    · intro x hx
      rw [← hg x]
      congr 1
      ext ω
      simp only [mem_filter, mem_univ, true_and]
      constructor
      · exact fun h => h.2
      · intro h; exact ⟨h ▸ hx, h⟩
  rw [hpre, hΩ]
  push_cast
  calc (A.card : ℝ) * c ≤ (ε * Fintype.card X) * c :=
        mul_le_mul_of_nonneg_right hA (Nat.cast_nonneg _)
    _ = ε * (Fintype.card X * c) := by ring

theorem multi_pool_max {ι : Type*} [Fintype ι] [DecidableEq ι] {α : ι → Type*}
    (S A : (i : ι) → Finset (α i)) (hAS : ∀ i, A i ⊆ S i) (m : ι → ℝ) (b : ℝ)
    (hb : ∀ i, m i ≤ b) (hbad : ∃ i, ((A i).card : ℝ) ≤ m i * (S i).card) :
    ((Fintype.piFinset A).card : ℝ) ≤ b * (Fintype.piFinset S).card := by
  obtain ⟨i0, h0⟩ := hbad
  rw [Fintype.card_piFinset, Fintype.card_piFinset]
  push_cast
  rw [← mul_prod_erase univ (fun i => ((A i).card : ℝ)) (mem_univ i0),
    ← mul_prod_erase univ (fun i => ((S i).card : ℝ)) (mem_univ i0)]
  have hP : ∏ i ∈ univ.erase i0, ((A i).card : ℝ) ≤ ∏ i ∈ univ.erase i0, ((S i).card : ℝ) :=
    prod_le_prod (fun i _ => Nat.cast_nonneg _) fun i _ => by exact_mod_cast card_le_card (hAS i)
  have hP0 : 0 ≤ ∏ i ∈ univ.erase i0, ((A i).card : ℝ) := prod_nonneg fun i _ => Nat.cast_nonneg _
  have hS0 : 0 ≤ ∏ i ∈ univ.erase i0, ((S i).card : ℝ) := prod_nonneg fun i _ => Nat.cast_nonneg _
  have hA0 : (0:ℝ) ≤ (A i0).card := Nat.cast_nonneg _
  calc ((A i0).card : ℝ) * ∏ i ∈ univ.erase i0, ((A i).card : ℝ)
      ≤ (m i0 * (S i0).card) * ∏ i ∈ univ.erase i0, ((S i).card : ℝ) :=
        mul_le_mul h0 hP hP0 (hA0.trans h0)
    _ ≤ (b * (S i0).card) * ∏ i ∈ univ.erase i0, ((S i).card : ℝ) :=
        mul_le_mul_of_nonneg_right
          (mul_le_mul_of_nonneg_right (hb i0) (Nat.cast_nonneg _)) hS0
    _ = b * (((S i0).card : ℝ) * ∏ i ∈ univ.erase i0, ((S i).card : ℝ)) := by ring

theorem multi_pool_map_le {ι : Type*} [Fintype ι] [DecidableEq ι] (h : ℕ) (k C Cmax : ι → ℕ)
    (Good : ι → Finset ℕ) (hG : ∀ i, Good i ⊆ range (C i)) (hCm : ∀ i, C i ≤ Cmax i)
    (σ : ι → ℝ) (hσ0 : ∀ i, 0 ≤ σ i) (hσ1 : ∀ i, σ i ≤ 1) (b : ℝ)
    (hb : ∀ i, (1 - σ i) ^ k i + k i * (Cmax i : ℝ) / 2 ^ h ≤ b)
    (hbad : ∃ i, 0 < C i ∧ ((Good i).card : ℝ) ≤ (1 - σ i) * C i) :
    ((univ.filter fun x : (i : ι) → Fin (k i) → Fin (2 ^ h) =>
        ∀ i, C i = 0 ∨ ∀ l, (x i l : ℕ) * C i / 2 ^ h ∈ Good i).card : ℝ)
      ≤ b * Fintype.card ((i : ι) → Fin (k i) → Fin (2 ^ h)) := by
  let A : (i : ι) → Finset (Fin (k i) → Fin (2 ^ h)) := fun i =>
    univ.filter fun y => C i = 0 ∨ ∀ l, (y l : ℕ) * C i / 2 ^ h ∈ Good i
  have e : (univ.filter fun x : (i : ι) → Fin (k i) → Fin (2 ^ h) =>
        ∀ i, C i = 0 ∨ ∀ l, (x i l : ℕ) * C i / 2 ^ h ∈ Good i) = Fintype.piFinset A := by
    ext x; simp [A, Fintype.mem_piFinset]
  have eU : Fintype.card ((i : ι) → Fin (k i) → Fin (2 ^ h))
      = (Fintype.piFinset fun i => (univ : Finset (Fin (k i) → Fin (2 ^ h)))).card := by
    rw [Fintype.piFinset_univ, card_univ]
  rw [e, eU]
  refine multi_pool_max _ A (fun i => subset_univ _) (fun _ => b) b (fun _ => le_rfl) ?_
  obtain ⟨i0, hC0, hbad0⟩ := hbad
  refine ⟨i0, ?_⟩
  have eA : A i0 = Fintype.piFinset
      (fun _ : Fin (k i0) => univ.filter fun U : Fin (2 ^ h) => (U : ℕ) * C i0 / 2 ^ h ∈ Good i0) := by
    ext y; simp [A, Fintype.mem_piFinset, hC0.ne']
  rw [eA, Fintype.card_piFinset, prod_const, card_univ, Fintype.card_fin, card_univ,
    Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]
  push_cast
  have hpm := pool_map_le (2 ^ h) (C i0) (k i0) (by positivity) hC0 (Good i0) (hG i0) (σ i0)
    (hσ0 i0) (hσ1 i0) hbad0
  push_cast at hpm
  refine hpm.trans (mul_le_mul_of_nonneg_right ?_ (by positivity))
  refine le_trans ?_ (hb i0)
  have hM : (0:ℝ) < 2 ^ h := by positivity
  have c1 : (C i0 : ℝ) ≤ Cmax i0 := by exact_mod_cast hCm i0
  have : (k i0 : ℝ) * ((C i0 : ℝ) / 2 ^ h) ≤ k i0 * (Cmax i0 : ℝ) / 2 ^ h := by
    rw [mul_div_assoc]
    exact mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right c1 hM.le) (Nat.cast_nonneg _)
  linarith

theorem multi_pool_map_le_max {ι : Type*} [Fintype ι] [DecidableEq ι] [Nonempty ι] (h : ℕ)
    (k C Cmax : ι → ℕ) (Good : ι → Finset ℕ) (hG : ∀ i, Good i ⊆ range (C i))
    (hCm : ∀ i, C i ≤ Cmax i) (σ : ι → ℝ) (hσ0 : ∀ i, 0 ≤ σ i) (hσ1 : ∀ i, σ i ≤ 1)
    (hbad : ∃ i, 0 < C i ∧ ((Good i).card : ℝ) ≤ (1 - σ i) * C i) :
    ((univ.filter fun x : (i : ι) → Fin (k i) → Fin (2 ^ h) =>
        ∀ i, C i = 0 ∨ ∀ l, (x i l : ℕ) * C i / 2 ^ h ∈ Good i).card : ℝ)
      ≤ univ.sup' univ_nonempty (fun i => (1 - σ i) ^ k i + k i * (Cmax i : ℝ) / 2 ^ h)
          * Fintype.card ((i : ι) → Fin (k i) → Fin (2 ^ h)) :=
  multi_pool_map_le h k C Cmax Good hG hCm σ hσ0 hσ1 _
    (fun i => le_sup' (fun i => (1 - σ i) ^ k i + k i * (Cmax i : ℝ) / 2 ^ h) (mem_univ i)) hbad

theorem multi_pool_map_le_sum {ι : Type*} [Fintype ι] [DecidableEq ι] (h : ℕ)
    (k C Cmax : ι → ℕ) (Good : ι → Finset ℕ) (hG : ∀ i, Good i ⊆ range (C i))
    (hCm : ∀ i, C i ≤ Cmax i) (σ : ι → ℝ) (hσ0 : ∀ i, 0 ≤ σ i) (hσ1 : ∀ i, σ i ≤ 1) (m : ℝ)
    (hm : ∀ i, (1 - σ i) ^ k i ≤ m)
    (hbad : ∃ i, 0 < C i ∧ ((Good i).card : ℝ) ≤ (1 - σ i) * C i) :
    ((univ.filter fun x : (i : ι) → Fin (k i) → Fin (2 ^ h) =>
        ∀ i, C i = 0 ∨ ∀ l, (x i l : ℕ) * C i / 2 ^ h ∈ Good i).card : ℝ)
      ≤ (m + ∑ i, k i * (Cmax i : ℝ) / 2 ^ h)
          * Fintype.card ((i : ι) → Fin (k i) → Fin (2 ^ h)) := by
  refine multi_pool_map_le h k C Cmax Good hG hCm σ hσ0 hσ1 _ (fun i => ?_) hbad
  have hs : k i * (Cmax i : ℝ) / 2 ^ h ≤ ∑ j, k j * (Cmax j : ℝ) / 2 ^ h :=
    single_le_sum (f := fun j => k j * (Cmax j : ℝ) / 2 ^ h) (fun j _ => by positivity) (mem_univ i)
  linarith [hm i]

theorem three_pool_default (σ : Fin 3 → ℝ) (hσ0 : ∀ i, 1 / 2 ≤ σ i) (hσ1 : ∀ i, σ i ≤ 1)
    (Cmax : Fin 3 → ℕ) (hC : ∑ i, 105 * Cmax i ≤ 2 ^ 85) :
    (∑ _i : Fin 3, (105 : ℕ)) = 315 ∧
    (∀ _i : Fin 3, kReq (2 ^ 64) ((2 ^ 40)⁻¹ / 2) (1 / 2) = 105) ∧
    KSuff (2 ^ 64) ((2 ^ 40)⁻¹ / 2) (1 / 2) 105 ∧ ¬ KSuff (2 ^ 64) ((2 ^ 40)⁻¹ / 2) (1 / 2) 104 ∧
    ∀ i, (1 - σ i) ^ 105 + (105 : ℕ) * (Cmax i : ℝ) / 2 ^ 192 ≤ (1 / 2) ^ 105 + (2 ^ 107)⁻¹ := by
  refine ⟨by simp, fun _ => table_Q64_e40.1, by norm_num [KSuff], by norm_num [KSuff], fun i => ?_⟩
  have h1 : (1 - σ i) ^ 105 ≤ (1 / 2 : ℝ) ^ 105 :=
    pow_le_pow_left₀ (by linarith [hσ1 i]) (by linarith [hσ0 i]) _
  have hi : 105 * Cmax i ≤ 2 ^ 85 :=
    (single_le_sum (f := fun j => 105 * Cmax j) (fun _ _ => Nat.zero_le _) (mem_univ i)).trans hC
  have hi' : ((105 : ℕ) : ℝ) * (Cmax i : ℝ) ≤ 2 ^ 85 := by exact_mod_cast hi
  have h2 : ((105 : ℕ) : ℝ) * (Cmax i : ℝ) / 2 ^ 192 ≤ (2 ^ 107)⁻¹ := by
    rw [div_le_iff₀ (by positivity)]
    calc ((105 : ℕ) : ℝ) * (Cmax i : ℝ) ≤ 2 ^ 85 := hi'
      _ = (2 ^ 107)⁻¹ * 2 ^ 192 := by norm_num
  linarith

section multicol

variable {F : Type*} [Field F] [Fintype F] [DecidableEq F] {nv nt w : ℕ}

open Classical in
theorem logup_bad_card_le_of_coef (a : Fin nv → F) (t : Fin nt → F) (m : Fin nt → F)
    (h : ∃ x0 ∈ vals a t, coef a t m x0 ≠ 0) :
    (univ.filter fun l : F => NonPole a t l ∧ Harmonic a t m l).card ≤ nv + nt := by
  obtain ⟨x0, hx0, hc⟩ := h
  have hN := numer_ne_zero a t m hx0 hc
  have hsub : (univ.filter fun l : F => NonPole a t l ∧ Harmonic a t m l)
      ⊆ (numer a t m).roots.toFinset := by
    intro l hl
    simp only [mem_filter, mem_univ, true_and] at hl
    obtain ⟨hnp, hh⟩ := hl
    rw [Multiset.mem_toFinset, Polynomial.mem_roots hN, Polynomial.IsRoot,
      numer_eval_nonpole a t m l hnp]
    have : (∑ x ∈ vals a t, coef a t m x * (l + x)⁻¹) = 0 := by
      rw [← harmonic_diff_eq]; exact sub_eq_zero.mpr hh
    rw [this, mul_zero]
  calc (univ.filter fun l : F => NonPole a t l ∧ Harmonic a t m l).card
      ≤ (numer a t m).roots.toFinset.card := card_le_card hsub
    _ ≤ Multiset.card (numer a t m).roots := Multiset.toFinset_card_le _
    _ ≤ (numer a t m).natDegree := Polynomial.card_roots' _
    _ ≤ (vals a t).card := numer_natDegree_le a t m
    _ ≤ nv + nt := vals_card_le a t

def fp (g : F) (x : Fin w → F) : F := ∑ c, x c * g ^ (c : ℕ)

theorem fp_collision_card_le (u v : Fin w → F) (h : u ≠ v) :
    (univ.filter fun g : F => fp g u = fp g v).card ≤ w - 1 := by
  set p : Polynomial F := ∑ k : Fin w, Polynomial.C (u k - v k) * Polynomial.X ^ (k : ℕ) with hp
  have hcoef : ∀ k : Fin w, p.coeff k = u k - v k := by
    intro k
    simp only [hp, Polynomial.finset_sum_coeff, Polynomial.coeff_C_mul, Polynomial.coeff_X_pow]
    rw [sum_eq_single k]
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
  have hdeg : p.natDegree ≤ w - 1 := by
    refine Polynomial.natDegree_sum_le_of_forall_le _ _ fun k _ => ?_
    refine (Polynomial.natDegree_C_mul_le _ _).trans ?_
    rw [Polynomial.natDegree_X_pow]
    have := k.isLt
    omega
  have hev : ∀ g : F, p.eval g = fp g u - fp g v := by
    intro g
    simp [hp, fp, Polynomial.eval_finset_sum, sub_mul, sum_sub_distrib]
  have hsub : (univ.filter fun g : F => fp g u = fp g v) ⊆ p.roots.toFinset := by
    intro g hg
    simp only [mem_filter, mem_univ, true_and] at hg
    rw [Multiset.mem_toFinset, Polynomial.mem_roots hp0, Polynomial.IsRoot, hev, hg, sub_self]
  exact (card_le_card hsub).trans
    ((Multiset.toFinset_card_le _).trans ((Polynomial.card_roots' _).trans hdeg))

def TupleValid (a : Fin nv → Fin w → F) (t : Fin nt → Fin w → F) (m : Fin nt → F) : Prop :=
  (∀ j, ∃ i, a j = t i) ∧ ∀ i, ((univ.filter fun j => a j = t i).card : F) = m i

def fpa (g : F) (a : Fin nv → Fin w → F) : Fin nv → F := fun j => fp g (a j)
def fpt (g : F) (t : Fin nt → Fin w → F) : Fin nt → F := fun i => fp g (t i)

def FpTrue (a : Fin nv → Fin w → F) (t : Fin nt → Fin w → F) (m : Fin nt → F) (g : F) : Prop :=
  ∀ x0 ∈ vals (fpa g a) (fpt g t), coef (fpa g a) (fpt g t) m x0 = 0

open Classical in
theorem multicol_badgamma_card_le (a : Fin nv → Fin w → F) (t : Fin nt → Fin w → F)
    (m : Fin nt → F) (ht : Function.Injective t)
    (hnv : ∀ c : ℕ, 0 < c → c ≤ nv → (c : F) ≠ 0) (hinv : ¬ TupleValid a t m) :
    (univ.filter fun g : F => FpTrue a t m g).card ≤ nt * (w - 1) := by
  by_cases hA : ∀ j, ∃ i, a j = t i
  · obtain ⟨i0, hi0⟩ : ∃ i0, ((univ.filter fun j => a j = t i0).card : F) ≠ m i0 := by
      by_contra hc
      push Not at hc
      exact hinv ⟨hA, hc⟩
    have hsub : (univ.filter fun g : F => FpTrue a t m g)
        ⊆ (univ.erase i0).biUnion fun i => univ.filter fun g : F => fp g (t i0) = fp g (t i) := by
      intro g hg
      simp only [mem_filter, mem_univ, true_and] at hg
      simp only [mem_biUnion, mem_erase, mem_univ, and_true, mem_filter, true_and]
      by_contra hno
      push Not at hno
      have hc := hg (fp g (t i0)) (mem_vals_t _ _ i0)
      apply hi0
      have e1 : (univ.filter fun i => fpt g t i = fp g (t i0)) = {i0} := by
        ext i
        simp only [mem_filter, mem_univ, true_and, mem_singleton, fpt]
        constructor
        · intro he
          by_contra hne
          exact hno i hne he.symm
        · intro he; rw [he]
      have e2 : cnt (fpa g a) (fp g (t i0)) = (univ.filter fun j => a j = t i0).card := by
        unfold cnt
        congr 1
        ext j
        simp only [mem_filter, mem_univ, true_and, fpa]
        constructor
        · intro he
          obtain ⟨i, hi⟩ := hA j
          rw [hi] at he ⊢
          by_contra hne
          have hne' : i ≠ i0 := fun e => hne (e ▸ rfl)
          exact hno i hne' he.symm
        · intro he; rw [he]
      unfold coef at hc
      rw [e1, sum_singleton, e2] at hc
      exact sub_eq_zero.mp hc
    calc (univ.filter fun g : F => FpTrue a t m g).card
        ≤ ∑ i ∈ univ.erase i0, (univ.filter fun g : F => fp g (t i0) = fp g (t i)).card :=
          (card_le_card hsub).trans card_biUnion_le
      _ ≤ ∑ _i ∈ univ.erase i0, (w - 1) := sum_le_sum fun i hi =>
          fp_collision_card_le _ _ fun e => (mem_erase.mp hi).1 (ht e).symm
      _ = (nt - 1) * (w - 1) := by
          rw [sum_const, card_erase_of_mem (mem_univ _), card_univ, Fintype.card_fin, smul_eq_mul]
      _ ≤ nt * (w - 1) := Nat.mul_le_mul_right _ (Nat.sub_le _ _)
  · push Not at hA
    obtain ⟨j0, hj0⟩ := hA
    have hsub : (univ.filter fun g : F => FpTrue a t m g)
        ⊆ univ.biUnion fun i => univ.filter fun g : F => fp g (a j0) = fp g (t i) := by
      intro g hg
      simp only [mem_filter, mem_univ, true_and] at hg
      simp only [mem_biUnion, mem_univ, true_and, mem_filter]
      by_contra hno
      push Not at hno
      have hc := hg (fp g (a j0)) (mem_vals_a _ _ j0)
      have e1 : (univ.filter fun i => fpt g t i = fp g (a j0)) = ∅ := by
        rw [filter_eq_empty_iff]
        intro i _ he
        exact hno i he.symm
      unfold coef at hc
      rw [e1, sum_empty, sub_zero] at hc
      refine hnv _ ?_ ?_ hc
      · exact card_pos.mpr ⟨j0, by simp [fpa]⟩
      · exact (card_le_univ _).trans (by simp)
    calc (univ.filter fun g : F => FpTrue a t m g).card
        ≤ ∑ i, (univ.filter fun g : F => fp g (a j0) = fp g (t i)).card :=
          (card_le_card hsub).trans card_biUnion_le
      _ ≤ ∑ _i : Fin nt, (w - 1) := sum_le_sum fun i _ => fp_collision_card_le _ _ (hj0 i)
      _ = nt * (w - 1) := by simp

open Classical in
theorem logup_multicol_card_le (a : Fin nv → Fin w → F) (t : Fin nt → Fin w → F)
    (m : Fin nt → F) (ht : Function.Injective t)
    (hnv : ∀ c : ℕ, 0 < c → c ≤ nv → (c : F) ≠ 0) (hinv : ¬ TupleValid a t m) :
    (univ.filter fun p : F × F =>
        NonPole (fpa p.1 a) (fpt p.1 t) p.2 ∧ Harmonic (fpa p.1 a) (fpt p.1 t) m p.2).card
      ≤ (nt * (w - 1) + nv + nt) * Fintype.card F := by
  set G := univ.filter fun g : F => FpTrue a t m g with hGdef
  have hG := multicol_badgamma_card_le a t m ht hnv hinv
  rw [← hGdef] at hG
  have hfib : ∀ g : F, (univ.filter fun l : F =>
      NonPole (fpa g a) (fpt g t) l ∧ Harmonic (fpa g a) (fpt g t) m l).card
        ≤ (if g ∈ G then Fintype.card F else 0) + (nv + nt) := by
    intro g
    by_cases hg : g ∈ G
    · rw [if_pos hg]
      exact (card_le_univ _).trans (Nat.le_add_right _ _)
    · rw [if_neg hg, zero_add]
      apply logup_bad_card_le_of_coef
      simp only [hGdef, mem_filter, mem_univ, true_and, FpTrue, not_forall] at hg
      obtain ⟨x0, hx0, hc⟩ := hg
      exact ⟨x0, hx0, hc⟩
  calc (univ.filter fun p : F × F =>
        NonPole (fpa p.1 a) (fpt p.1 t) p.2 ∧ Harmonic (fpa p.1 a) (fpt p.1 t) m p.2).card
      = ∑ g : F, (univ.filter fun l : F =>
          NonPole (fpa g a) (fpt g t) l ∧ Harmonic (fpa g a) (fpt g t) m l).card := by
        rw [card_filter, Fintype.sum_prod_type]
        refine sum_congr rfl fun g _ => ?_
        rw [card_filter]
    _ ≤ ∑ g : F, ((if g ∈ G then Fintype.card F else 0) + (nv + nt)) := sum_le_sum fun g _ => hfib g
    _ = G.card * Fintype.card F + Fintype.card F * (nv + nt) := by
        rw [sum_add_distrib, sum_ite_mem, univ_inter, sum_const, sum_const, card_univ, smul_eq_mul,
          smul_eq_mul]
    _ ≤ nt * (w - 1) * Fintype.card F + Fintype.card F * (nv + nt) :=
        Nat.add_le_add_right (Nat.mul_le_mul_right _ hG) _
    _ = (nt * (w - 1) + nv + nt) * Fintype.card F := by ring

open Classical in
theorem logup_multicol_prob (a : Fin nv → Fin w → F) (t : Fin nt → Fin w → F)
    (m : Fin nt → F) (ht : Function.Injective t)
    (hnv : ∀ c : ℕ, 0 < c → c ≤ nv → (c : F) ≠ 0) (hinv : ¬ TupleValid a t m) :
    ((univ.filter fun p : F × F =>
        NonPole (fpa p.1 a) (fpt p.1 t) p.2 ∧ Harmonic (fpa p.1 a) (fpt p.1 t) m p.2).card : ℝ)
        / Fintype.card (F × F)
      ≤ ((nt * (w - 1) + nv + nt : ℕ) : ℝ) / Fintype.card F := by
  have h := logup_multicol_card_le a t m ht hnv hinv
  have hF : (0:ℝ) < Fintype.card F := Nat.cast_pos.mpr Fintype.card_pos
  rw [Fintype.card_prod, Nat.cast_mul, div_le_div_iff₀ (by positivity) hF]
  have h' : ((univ.filter fun p : F × F =>
        NonPole (fpa p.1 a) (fpt p.1 t) p.2 ∧ Harmonic (fpa p.1 a) (fpt p.1 t) m p.2).card : ℝ)
      ≤ ((nt * (w - 1) + nv + nt : ℕ) : ℝ) * Fintype.card F := by exact_mod_cast h
  nlinarith

end multicol

def goldilocks : ℕ := 2 ^ 64 - 2 ^ 32 + 1

theorem goldilocks_eq : goldilocks = 18446744069414584321 := by norm_num [goldilocks]

theorem goldilocks_p2_bounds : 2 ^ 127 < goldilocks ^ 2 ∧ goldilocks ^ 2 < 2 ^ 128 := by
  rw [goldilocks_eq]; norm_num

theorem goldilocks_p3_bounds : 2 ^ 191 < goldilocks ^ 3 ∧ goldilocks ^ 3 < 2 ^ 192 := by
  rw [goldilocks_eq]; norm_num

theorem frac_le_inv_pow2_iff (D N t : ℕ) (hN : 0 < N) :
    (D : ℝ) / N ≤ (2 ^ t)⁻¹ ↔ D * 2 ^ t ≤ N := by
  rw [← one_div, div_le_div_iff₀ (by exact_mod_cast hN) (by positivity), one_mul]
  exact_mod_cast Iff.rfl

theorem field_p2_iff (D : ℕ) : (D : ℝ) / (goldilocks ^ 2 : ℕ) ≤ (2 ^ 106)⁻¹ ↔ D ≤ 2 ^ 22 - 1 := by
  rw [frac_le_inv_pow2_iff _ _ _ (by rw [goldilocks_eq]; norm_num), goldilocks_eq]
  norm_num
  omega

theorem field_p3_iff (D : ℕ) :
    (D : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 106)⁻¹ ↔ D ≤ goldilocks ^ 3 / 2 ^ 106 := by
  rw [frac_le_inv_pow2_iff _ _ _ (by rw [goldilocks_eq]; norm_num)]
  exact (Nat.le_div_iff_mul_le (by positivity)).symm

theorem field_p3_bits (D : ℕ) (hD : D ≤ 2 ^ 85) :
    (D : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 106)⁻¹ := by
  rw [frac_le_inv_pow2_iff _ _ _ (by rw [goldilocks_eq]; norm_num)]
  calc D * 2 ^ 106 ≤ 2 ^ 85 * 2 ^ 106 := Nat.mul_le_mul_right _ hD
    _ ≤ goldilocks ^ 3 := by rw [goldilocks_eq]; norm_num

theorem slot_table_numerator (nv : ℕ) (hnv : nv ≤ 2 ^ 84) :
    ¬ ((2 ^ 29 * (8 - 1) : ℕ) : ℝ) / (goldilocks ^ 2 : ℕ) ≤ (2 ^ 106)⁻¹ ∧
    ((2 ^ 29 * (8 - 1) + nv + 2 ^ 29 : ℕ) : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 106)⁻¹ := by
  refine ⟨?_, field_p3_bits _ (by omega)⟩
  rw [field_p2_iff]; norm_num

theorem codeword_p2_insufficient :
    ¬ ((2 ^ 32 : ℕ) : ℝ) / (goldilocks ^ 2 : ℕ) ≤ (2 ^ 106)⁻¹ := by
  rw [field_p2_iff]; norm_num

theorem fold_p3_bits (n : ℕ) (hn : n ≤ 2 ^ 42) :
    (n : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 149)⁻¹ ∧
    ¬ ((2 ^ 42 : ℕ) : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 150)⁻¹ := by
  have hp : 0 < goldilocks ^ 3 := by rw [goldilocks_eq]; norm_num
  rw [frac_le_inv_pow2_iff _ _ _ hp, frac_le_inv_pow2_iff _ _ _ hp]
  constructor
  · calc n * 2 ^ 149 ≤ 2 ^ 42 * 2 ^ 149 := Nat.mul_le_mul_right _ hn
      _ ≤ goldilocks ^ 3 := by rw [goldilocks_eq]; norm_num
  · rw [goldilocks_eq]; norm_num

theorem amap_default (D : ℕ) (hD : D ≤ 2 ^ 85) : (D : ℝ) / 2 ^ 192 ≤ (2 ^ 107)⁻¹ := by
  have := (frac_le_inv_pow2_iff D (2 ^ 192) 107 (by positivity)).mpr
    (by calc D * 2 ^ 107 ≤ 2 ^ 85 * 2 ^ 107 := Nat.mul_le_mul_right _ hD
          _ = 2 ^ 192 := by norm_num)
  simpa only [Nat.cast_pow, Nat.cast_ofNat] using this

theorem total_le_rbr (Q ε εsp m amap εalg Δ : ℝ) (hm : 0 ≤ m) (hQ : 0 ≤ Q)
    (hk : Q * m ≤ εsp) (haux : Δ + Q * max amap εalg ≤ ε - εsp) :
    Δ + Q * max (m + amap) εalg ≤ ε := by
  have h1 : max (m + amap) εalg ≤ m + max amap εalg :=
    max_le (by linarith [le_max_left amap εalg]) (by linarith [le_max_right amap εalg])
  have h2 := mul_le_mul_of_nonneg_left h1 hQ
  have e : Q * (m + max amap εalg) = Q * m + Q * max amap εalg := by ring
  linarith

theorem total_le_rbr_pools {ι : Type*} [Fintype ι] [Nonempty ι] (Q ε εsp amap εalg Δ : ℝ)
    (σ : ι → ℝ) (k : ι → ℕ) (hσ1 : ∀ i, σ i ≤ 1) (hQ : 0 ≤ Q)
    (hk : ∀ i, KSuff Q εsp (σ i) (k i)) (haux : Δ + Q * max amap εalg ≤ ε - εsp) :
    Δ + Q * max (univ.sup' univ_nonempty (fun i => (1 - σ i) ^ k i) + amap) εalg ≤ ε := by
  obtain ⟨i0, _, hi0⟩ := exists_mem_eq_sup' (univ_nonempty (α := ι)) (fun i => (1 - σ i) ^ k i)
  refine total_le_rbr Q ε εsp _ amap εalg Δ ?_ hQ ?_ haux
  · rw [hi0]; exact pow_nonneg (by linarith [hσ1 i0]) _
  · rw [hi0]; exact hk i0

theorem default_budget_rbr (m amap εalg Δ : ℝ) (hm0 : 0 ≤ m) (hm : m ≤ (1 / 2) ^ 105)
    (ha : amap ≤ (2 ^ 106)⁻¹) (he : εalg ≤ (2 ^ 106)⁻¹) (hΔ : Δ ≤ (2 ^ 42)⁻¹) :
    Δ + 2 ^ 64 * max (m + amap) εalg ≤ (2 ^ 40)⁻¹ := by
  refine total_le_rbr (2 ^ 64) (2 ^ 40)⁻¹ ((2 ^ 40)⁻¹ / 2) m amap εalg Δ hm0 (by positivity) ?_ ?_
  · calc (2:ℝ) ^ 64 * m ≤ 2 ^ 64 * (1 / 2) ^ 105 := mul_le_mul_of_nonneg_left hm (by positivity)
      _ ≤ (2 ^ 40)⁻¹ / 2 := by norm_num
  · have hmx : max amap εalg ≤ (2 ^ 106)⁻¹ := max_le ha he
    have := mul_le_mul_of_nonneg_left hmx (by positivity : (0:ℝ) ≤ 2 ^ 64)
    have e : (2:ℝ) ^ 64 * (2 ^ 106)⁻¹ + (2 ^ 42)⁻¹ = (2 ^ 40)⁻¹ - (2 ^ 40)⁻¹ / 2 := by norm_num
    linarith

theorem k104_fails_rbr (Δ : ℝ) (hΔ : 0 < Δ) : ¬ (Δ + 2 ^ 64 * (1 / 2 : ℝ) ^ 104 ≤ (2 ^ 40)⁻¹) := by
  have e : (2:ℝ) ^ 64 * (1 / 2) ^ 104 = (2 ^ 40)⁻¹ := by norm_num
  rw [e]; linarith

theorem rbr_default_lifetime {Ω S ι : Type*} [Fintype Ω] [DecidableEq Ω] [Nonempty Ω] {Q : ℕ}
    (hQ : Q ≤ 2 ^ 64) (st : Fin Q → (Fin Q → Ω) → S)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a)
    (Doomed : S → Prop) (Esc : S → Finset Ω) (round : S → ι) (ε : ι → ℝ)
    (m amap εalg Δ : ℝ) (hm0 : 0 ≤ m) (hm : m ≤ (1 / 2) ^ 105) (ha0 : 0 ≤ amap)
    (ha : amap ≤ (2 ^ 106)⁻¹) (he : εalg ≤ (2 ^ 106)⁻¹) (hΔ : Δ ≤ (2 ^ 42)⁻¹)
    (hεr : ∀ r, ε r ≤ max (m + amap) εalg)
    (hEsc : ∀ s, Doomed s → ((Esc s).card : ℝ) ≤ ε (round s) * Fintype.card Ω)
    (BadAcc Fail : Finset (Fin Q → Ω))
    (hFail : (Fail.card : ℝ) ≤ Δ * (Fintype.card Ω : ℝ) ^ Q)
    (hpath : ∀ a ∈ BadAcc, a ∈ Fail ∨ ∃ j, Doomed (st j a) ∧ a j ∈ Esc (st j a)) :
    (BadAcc.card : ℝ) ≤ (2 ^ 40)⁻¹ * (Fintype.card Ω : ℝ) ^ Q := by
  have hε0 : 0 ≤ max (m + amap) εalg := le_max_of_le_left (by linarith)
  have h := rbr_lifetime_soundness st hst Doomed Esc round ε _ hε0 hεr hEsc BadAcc Fail Δ hFail hpath
  refine h.trans (mul_le_mul_of_nonneg_right ?_ (by positivity))
  have hQr : (Q : ℝ) ≤ 2 ^ 64 := by exact_mod_cast hQ
  have := mul_le_mul_of_nonneg_right hQr hε0
  linarith [default_budget_rbr m amap εalg Δ hm0 hm ha he hΔ]

end FlashProveNext

#print axioms FlashProveNext.rbr_lifetime_soundness
#print axioms FlashProveNext.hit_sum_le
#print axioms FlashProveNext.rbr_lifetime_soundness_sum
#print axioms FlashProveNext.esc_preimage_le
#print axioms FlashProveNext.multi_pool_max
#print axioms FlashProveNext.multi_pool_map_le
#print axioms FlashProveNext.multi_pool_map_le_max
#print axioms FlashProveNext.multi_pool_map_le_sum
#print axioms FlashProveNext.three_pool_default
#print axioms FlashProveNext.logup_bad_card_le_of_coef
#print axioms FlashProveNext.fp_collision_card_le
#print axioms FlashProveNext.multicol_badgamma_card_le
#print axioms FlashProveNext.logup_multicol_card_le
#print axioms FlashProveNext.logup_multicol_prob
#print axioms FlashProveNext.goldilocks_eq
#print axioms FlashProveNext.goldilocks_p2_bounds
#print axioms FlashProveNext.goldilocks_p3_bounds
#print axioms FlashProveNext.frac_le_inv_pow2_iff
#print axioms FlashProveNext.field_p2_iff
#print axioms FlashProveNext.field_p3_iff
#print axioms FlashProveNext.field_p3_bits
#print axioms FlashProveNext.slot_table_numerator
#print axioms FlashProveNext.codeword_p2_insufficient
#print axioms FlashProveNext.fold_p3_bits
#print axioms FlashProveNext.amap_default
#print axioms FlashProveNext.total_le_rbr
#print axioms FlashProveNext.total_le_rbr_pools
#print axioms FlashProveNext.default_budget_rbr
#print axioms FlashProveNext.k104_fails_rbr
#print axioms FlashProveNext.rbr_default_lifetime
