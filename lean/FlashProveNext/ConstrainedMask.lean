/-
In-circuit binding of constrained-decoding vocabulary masks.

At every sampled step `s` the prover commits a Boolean mask row `A_s : {0,1}^vb → {0,1}` over the
padded vocabulary, a gate bit `act_s` (constrained or unconstrained step) and a one-hot vector
`oh_s` over the automaton states. After these commitments the verifier draws one point
`r ∈ E^vb`, evaluates the public allowed-token table on it,
`m_q = Ã_q(r)` for every state `q` and `m_unc = 1̃_{v < V}(r)`, and checks the single equation

  `Ã_s(r) = (1 - act_s) · m_unc + act_s · ∑_q oh_s(q) · m_q`,

where `f̃` is the multilinear extension `f̃(r) = ∑_{x ∈ {0,1}^vb} eq(r, x) f(x)`. On an
unconstrained step the one-hot vector sits on state `0` and the gate discards it. Beyond the mask
itself, the check adds `O(Q)` committed cells per step and none per vocabulary entry.

1. `onehot_eq_indicator`, `onehot_select`, `gate_select`, `gated_onehot_select` :
   a Boolean vector with `∑_q oh(q) = 1` and `∑_q q · oh(q) = q⋆` is the indicator of `q⋆`, so the
   selector returns row `q⋆` of any table, and the gate returns that row or the unconstrained row;
2. `eqKer_bool`, `mleEval_bool`, `mleEval_eq_iff` : the multilinear extension interpolates on the
   Boolean cube, so two masks have the same extension exactly when they are equal;
3. `mle_diff_zero_card_le`, `constrained_mask_bad_card_le`, `constrained_mask_bad_card_le_of_pos`,
   `constrained_mask_bad_prob` : Schwartz–Zippel for multilinear polynomials, proved by induction on
   `vb`. Writing `r = (r₀, r')`, the extension is `a(r') + r₀ · b(r')`; for each `r'` either `b(r')`
   vanishes (a smaller instance) or exactly one `r₀` is bad. Two distinct masks therefore agree on
   at most a `vb / |E|` fraction of points `r ∈ E^vb`;
4. `constrained_mask_check_sound`, `constrained_mask_check_complete` : the verifier equation above,
   with the honest row `if act = 1 then allowed_{q⋆} else 1_{v < V}`, is passed by the honest mask at
   every `r` and by any other mask on at most a `vb / |E|` fraction of `r`;
5. `constrained_mask_goldilocks` : over `E = F_{p^3}` (`p = 2^64 - 2^32 + 1`) with `vb ≤ 32`, the
   escape probability is at most `2^{-186}`, and hence within the per-round budget `2^{-106}`.
-/
import Mathlib
import FlashProveNext.RoundByRound

namespace FlashProveNext

open Finset

set_option linter.unusedSectionVars false

/-! ### One-hot state selector and activity gate -/

section Selector

/-- A Boolean vector over `Fin Q` with total weight `1` and index-weighted sum `q⋆` is the indicator of `q⋆`. -/
theorem onehot_eq_indicator {Q : ℕ} (oh : Fin Q → ℕ) (q_star : Fin Q)
    (hbool : ∀ q, oh q = 0 ∨ oh q = 1) (hsum : ∑ q, oh q = 1)
    (hval : ∑ q, q.val * oh q = q_star.val) :
    ∀ q, oh q = if q = q_star then 1 else 0 := by
  have hex : ∃ q0, oh q0 = 1 := by
    by_contra hno
    have hzero : ∀ q, oh q = 0 := fun q => (hbool q).resolve_right fun h1 => hno ⟨q, h1⟩
    have h0 : ∑ q, oh q = 0 := Finset.sum_eq_zero fun q _ => hzero q
    omega
  obtain ⟨q0, hq0⟩ := hex
  have hrest : ∑ q ∈ univ.erase q0, oh q = 0 := by
    have h := Finset.add_sum_erase univ oh (Finset.mem_univ q0)
    rw [hq0, hsum] at h
    omega
  have hz : ∀ q, q ≠ q0 → oh q = 0 := by
    intro q hq
    have hle : oh q ≤ ∑ x ∈ univ.erase q0, oh x :=
      Finset.single_le_sum (f := fun x => oh x) (fun _ _ => Nat.zero_le _)
        (Finset.mem_erase.mpr ⟨hq, Finset.mem_univ q⟩)
    omega
  have hv : ∑ q, q.val * oh q = q0.val := by
    rw [Finset.sum_eq_single q0]
    · rw [hq0, Nat.mul_one]
    · intro q _ hq
      rw [hz q hq, Nat.mul_zero]
    · intro h
      exact (h (Finset.mem_univ q0)).elim
  have hq : q0 = q_star := Fin.ext (hv.symm.trans hval)
  intro q
  rw [← hq]
  split_ifs with h
  · rw [h]
    exact hq0
  · exact hz q h

/-- The one-hot selector returns row `q⋆` of any table. -/
theorem onehot_select {E : Type*} [CommSemiring E] {Q : ℕ} (oh : Fin Q → ℕ) (q_star : Fin Q)
    (hbool : ∀ q, oh q = 0 ∨ oh q = 1) (hsum : ∑ q, oh q = 1)
    (hval : ∑ q, q.val * oh q = q_star.val) (M : Fin Q → E) :
    ∑ q, (oh q : E) * M q = M q_star := by
  rw [Finset.sum_eq_single q_star]
  · rw [onehot_eq_indicator oh q_star hbool hsum hval q_star, if_pos rfl, Nat.cast_one, one_mul]
  · intro q _ hq
    rw [onehot_eq_indicator oh q_star hbool hsum hval q, if_neg hq, Nat.cast_zero, zero_mul]
  · intro h
    exact (h (Finset.mem_univ q_star)).elim

/-- The activity gate `(1 - act) · m_unc + act · m_sel` with `act ∈ {0, 1}` is a two-way multiplexer. -/
theorem gate_select {E : Type*} [CommRing E] (act : ℕ) (hact : act = 0 ∨ act = 1) (m_unc m_sel : E) :
    (1 - (act : E)) * m_unc + (act : E) * m_sel = if act = 1 then m_sel else m_unc := by
  rcases hact with h | h
  · rw [h, if_neg Nat.zero_ne_one, Nat.cast_zero, sub_zero, one_mul, zero_mul, add_zero]
  · rw [h, if_pos rfl, Nat.cast_one, sub_self, zero_mul, one_mul, zero_add]

/-- Gate and selector combined: the right-hand side of the mask check is row `q⋆` on a constrained
step and the unconstrained row otherwise. -/
theorem gated_onehot_select {E : Type*} [CommRing E] {Q : ℕ} (oh : Fin Q → ℕ) (q_star : Fin Q)
    (hbool : ∀ q, oh q = 0 ∨ oh q = 1) (hsum : ∑ q, oh q = 1)
    (hval : ∑ q, q.val * oh q = q_star.val) (act : ℕ) (hact : act = 0 ∨ act = 1)
    (M : Fin Q → E) (m_unc : E) :
    (1 - (act : E)) * m_unc + (act : E) * (∑ q, (oh q : E) * M q)
      = if act = 1 then M q_star else m_unc := by
  rw [onehot_select oh q_star hbool hsum hval M, gate_select act hact]

end Selector

/-! ### Multilinear extension on the Boolean cube -/

section Multilinear

variable {E : Type*} [CommRing E]

/-- The multilinear equality kernel `eq(r, x) = ∏_i (x_i ? r_i : 1 - r_i)`. -/
def eqKer {vb : ℕ} (r : Fin vb → E) (x : Fin vb → Bool) : E :=
  ∏ i : Fin vb, if x i then r i else (1 - r i)

/-- Multilinear extension `f̃(r) = ∑_{x ∈ {0,1}^vb} eq(r, x) · f(x)`. -/
def mleEval {vb : ℕ} (f : (Fin vb → Bool) → E) (r : Fin vb → E) : E :=
  ∑ x : Fin vb → Bool, eqKer r x * f x

/-- Prepend a first coordinate (built on `Fin.cases`). -/
def cmCons {β : Type*} {n : ℕ} (b : β) (y : Fin n → β) : Fin (n + 1) → β :=
  fun i => @Fin.cases n (fun _ => β) b y i

/-- Drop the first coordinate. -/
def cmTail {β : Type*} {n : ℕ} (x : Fin (n + 1) → β) : Fin n → β :=
  fun i => x i.succ

theorem cmCons_self {β : Type*} {n : ℕ} (x : Fin (n + 1) → β) : cmCons (x 0) (cmTail x) = x := by
  funext i
  cases i using Fin.cases with
  | zero => rfl
  | succ j => rfl

theorem cmTail_update_zero {β : Type*} {n : ℕ} (x : Fin (n + 1) → β) (z : β) :
    cmTail (Function.update x 0 z) = cmTail x := by
  funext j
  exact Function.update_of_ne (Fin.succ_ne_zero j) z x

/-- `{0,1} × {0,1}^n ≃ {0,1}^{n+1}` by prepending the first coordinate. -/
def cmConsEquiv (n : ℕ) : Bool × (Fin n → Bool) ≃ (Fin (n + 1) → Bool) where
  toFun p := cmCons p.1 p.2
  invFun x := (x 0, cmTail x)
  left_inv _ := rfl
  right_inv x := cmCons_self x

theorem cm_sum_cons {M : Type*} [AddCommMonoid M] (n : ℕ) (G : (Fin (n + 1) → Bool) → M) :
    ∑ x, G x = ∑ y : Fin n → Bool, (G (cmCons false y) + G (cmCons true y)) := by
  rw [← Fintype.sum_equiv (cmConsEquiv n) (fun p => G (cmCons p.1 p.2)) G (fun _ => rfl),
    Fintype.sum_prod_type, Fintype.sum_bool, Finset.sum_add_distrib]
  exact add_comm _ _

theorem eqKer_cons {n : ℕ} (r : Fin (n + 1) → E) (b : Bool) (y : Fin n → Bool) :
    eqKer r (cmCons b y) = (if b then r 0 else 1 - r 0) * eqKer (cmTail r) y := by
  unfold eqKer
  rw [Fin.prod_univ_succ]
  rfl

/-- First-coordinate decomposition: `f̃(r) = (1 - r₀) · f̃₀(r') + r₀ · f̃₁(r')`. -/
theorem mleEval_cons {n : ℕ} (D : (Fin (n + 1) → Bool) → E) (r : Fin (n + 1) → E) :
    mleEval D r = (1 - r 0) * mleEval (fun y => D (cmCons false y)) (cmTail r)
      + r 0 * mleEval (fun y => D (cmCons true y)) (cmTail r) := by
  unfold mleEval
  refine (cm_sum_cons n (fun x => eqKer r x * D x)).trans ?_
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [eqKer_cons, eqKer_cons, if_neg (show ¬ (false = true) by decide),
    if_pos (show true = true from rfl)]
  ring

theorem mleEval_sub {vb : ℕ} (A M : (Fin vb → Bool) → E) (r : Fin vb → E) :
    mleEval (A - M) r = mleEval A r - mleEval M r := by
  simp only [mleEval, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]

/-- On a Boolean point `b` the kernel is the indicator of `b`. -/
theorem eqKer_bool {vb : ℕ} (b x : Fin vb → Bool) :
    eqKer (fun i => if b i then (1 : E) else 0) x = if x = b then 1 else 0 := by
  unfold eqKer
  split_ifs with h
  · rw [h]
    refine Finset.prod_eq_one fun i _ => ?_
    cases hbi : b i <;> simp [hbi]
  · obtain ⟨i, hi⟩ := Function.ne_iff.mp h
    refine Finset.prod_eq_zero (Finset.mem_univ i) ?_
    revert hi
    cases hx : x i <;> cases hb : b i <;> simp [hb]

/-- Boolean-cube interpolation: `f̃(b) = f(b)` for every `b ∈ {0,1}^vb`. -/
theorem mleEval_bool {vb : ℕ} (f : (Fin vb → Bool) → E) (b : Fin vb → Bool) :
    mleEval f (fun i => if b i then 1 else 0) = f b := by
  unfold mleEval
  rw [Finset.sum_eq_single b]
  · rw [eqKer_bool, if_pos rfl, one_mul]
  · intro x _ hx
    rw [eqKer_bool, if_neg hx, zero_mul]
  · intro h
    exact (h (Finset.mem_univ b)).elim

/-- Two masks have the same multilinear extension exactly when they are equal. -/
theorem mleEval_eq_iff {vb : ℕ} (A M : (Fin vb → Bool) → E) : mleEval A = mleEval M ↔ A = M := by
  constructor
  · intro h
    funext b
    have hb := congrFun h (fun i => if b i then 1 else 0)
    rwa [mleEval_bool, mleEval_bool] at hb
  · rintro rfl
    rfl

end Multilinear

/-! ### Schwartz–Zippel for multilinear extensions -/

section SchwartzZippel

variable {E : Type*} [Field E] [Fintype E] [DecidableEq E]

/-- Root set of `t ↦ a + t · b`: the single point `-a / b` when `b ≠ 0`, empty otherwise. -/
def cmRoot (a b : E) : Finset E := if b = 0 then ∅ else {-a / b}

theorem cmRoot_card_le (a b : E) : (cmRoot a b).card ≤ 1 := by
  unfold cmRoot
  split_ifs <;> simp

theorem cm_mem_root (a b t : E) (hb : b ≠ 0) (h : a + t * b = 0) : t ∈ cmRoot a b := by
  unfold cmRoot
  rw [if_neg hb, Finset.mem_singleton, eq_div_iff hb]
  linear_combination h

/-- A predicate on the last `n` coordinates holds on at most `|E|` times as many points of `E^{n+1}`. -/
theorem cm_card_filter_tail_le {n : ℕ} (P : (Fin n → E) → Prop) [DecidablePred P] :
    (univ.filter fun r : Fin (n + 1) → E => P (cmTail r)).card
      ≤ (univ.filter P).card * Fintype.card E := by
  calc (univ.filter fun r : Fin (n + 1) → E => P (cmTail r)).card
      ≤ ((univ.filter P) ×ˢ (univ : Finset E)).card := by
        apply Finset.card_le_card_of_injOn (fun r => (cmTail r, r 0))
        · intro r hr
          simp only [Finset.mem_coe, Finset.mem_filter, Finset.mem_product, Finset.mem_univ,
            true_and, and_true] at hr ⊢
          exact hr
        · intro r _ r' _ h
          simp only [Prod.mk.injEq] at h
          obtain ⟨h1, h2⟩ := h
          funext i
          cases i using Fin.cases with
          | zero => exact h2
          | succ j => exact congrFun h1 j
    _ = (univ.filter P).card * Fintype.card E := by
        rw [Finset.card_product, Finset.card_univ]

theorem cm_arith {z a b g c n : ℕ} (hz : z ≤ a + b) (ha : a ≤ g * c) (hg : g * c ≤ n * c ^ n)
    (hb : b * c ≤ 1 * c ^ (n + 1)) : z * c ≤ (n + 1) * c ^ (n + 1) := by
  have h1 : a * c ≤ n * c ^ n * c :=
    (Nat.mul_le_mul_right c ha).trans (Nat.mul_le_mul_right c hg)
  calc z * c ≤ (a + b) * c := Nat.mul_le_mul_right c hz
    _ = a * c + b * c := Nat.add_mul a b c
    _ ≤ n * c ^ n * c + 1 * c ^ (n + 1) := Nat.add_le_add h1 hb
    _ = (n + 1) * c ^ (n + 1) := by ring

/-- Inductive step: if every point of `Z` either has a prefix in the zero set of a nonzero
multilinear `G` or has its first challenge at the root of `t ↦ a + t · b`, then `|Z| · |E| ≤ (n+1) |E|^{n+1}`. -/
theorem cm_card_step {n : ℕ} (Z : Finset (Fin (n + 1) → E)) (G : (Fin n → Bool) → E)
    (a b : (Fin n → E) → E)
    (hG : (univ.filter fun rt : Fin n → E => mleEval G rt = 0).card * Fintype.card E
      ≤ n * Fintype.card E ^ n)
    (hZ : ∀ r ∈ Z, mleEval G (cmTail r) = 0 ∨ r 0 ∈ cmRoot (a (cmTail r)) (b (cmTail r))) :
    Z.card * Fintype.card E ≤ (n + 1) * Fintype.card E ^ (n + 1) := by
  have hsub : Z ⊆ (univ.filter fun r : Fin (n + 1) → E => mleEval G (cmTail r) = 0) ∪
      (univ.filter fun r : Fin (n + 1) → E => r 0 ∈ cmRoot (a (cmTail r)) (b (cmTail r))) := by
    intro r hr
    rw [Finset.mem_union, Finset.mem_filter, Finset.mem_filter]
    rcases hZ r hr with h | h
    · exact Or.inl ⟨Finset.mem_univ _, h⟩
    · exact Or.inr ⟨Finset.mem_univ _, h⟩
  have hZc := (Finset.card_le_card hsub).trans (Finset.card_union_le _ _)
  have hA := cm_card_filter_tail_le (fun rt : Fin n → E => mleEval G rt = 0)
  have hB := hit_card_le (0 : Fin (n + 1))
    (fun r : Fin (n + 1) → E => cmRoot (a (cmTail r)) (b (cmTail r))) 1
    (fun r x => by simp only [cmTail_update_zero]) (fun r => cmRoot_card_le _ _)
  exact cm_arith hZc hA hG hB

/-- **Schwartz–Zippel for multilinear extensions.** A nonzero `D : {0,1}^vb → E` has a
multilinear extension vanishing on at most a `vb / |E|` fraction of `E^vb`. -/
theorem mle_diff_zero_card_le (vb : ℕ) :
    ∀ D : (Fin vb → Bool) → E, D ≠ 0 →
      (univ.filter fun r : Fin vb → E => mleEval D r = 0).card * Fintype.card E
        ≤ vb * Fintype.card E ^ vb := by
  induction vb with
  | zero =>
    intro D hD
    have hD0 : D (fun i => i.elim0) ≠ 0 := by
      intro h
      apply hD
      funext x
      have hx : x = fun i => i.elim0 := funext fun i => i.elim0
      exact (congrArg D hx).trans h
    have hev : ∀ r : Fin 0 → E, mleEval D r = D (fun i => i.elim0) := by
      intro r
      unfold mleEval
      rw [Finset.sum_eq_single (fun i : Fin 0 => (Fin.elim0 i : Bool))]
      · simp [eqKer]
      · intro x _ hx
        exact (hx (funext fun i => i.elim0)).elim
      · intro h
        exact (h (Finset.mem_univ _)).elim
    have hcard : (univ.filter fun r : Fin 0 → E => mleEval D r = 0).card = 0 := by
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      intro r _ hr
      exact hD0 ((hev r).symm.trans hr)
    rw [hcard]
    simp
  | succ n ih =>
    intro D hD
    have hsplit : ∀ r : Fin (n + 1) → E, mleEval D r =
        mleEval (fun y => D (cmCons false y)) (cmTail r) +
          r 0 * (mleEval (fun y => D (cmCons true y)) (cmTail r) -
            mleEval (fun y => D (cmCons false y)) (cmTail r)) := by
      intro r
      rw [mleEval_cons]
      ring
    by_cases hEq : (fun y : Fin n → Bool => D (cmCons false y)) = (fun y => D (cmCons true y))
    · -- Both halves agree: `D̃(r) = D̃₀(r')` and `D₀ ≠ 0`.
      have hG : (fun y : Fin n → Bool => D (cmCons false y)) ≠ 0 := by
        intro h0
        apply hD
        funext x
        show D x = 0
        rw [← cmCons_self x]
        cases hx : x 0
        · exact congrFun h0 (cmTail x)
        · exact (congrFun hEq (cmTail x)).symm.trans (congrFun h0 (cmTail x))
      refine cm_card_step _ (fun y => D (cmCons false y)) (mleEval fun y => D (cmCons false y))
        (fun rt => mleEval (fun y => D (cmCons true y)) rt - mleEval (fun y => D (cmCons false y)) rt)
        (ih _ hG) ?_
      intro r hr
      left
      have h := hsplit r
      have hb : mleEval (fun y => D (cmCons false y)) (cmTail r) =
          mleEval (fun y => D (cmCons true y)) (cmTail r) :=
        congrArg (fun f => mleEval f (cmTail r)) hEq
      rw [← hb, sub_self, mul_zero, add_zero] at h
      rw [← h]
      exact (Finset.mem_filter.mp hr).2
    · -- The halves differ: `D̃(r) = a(r') + r₀ · b(r')` with `b = (D₁ - D₀)~` not identically zero.
      have hG : (fun y : Fin n → Bool => D (cmCons true y)) - (fun y => D (cmCons false y)) ≠ 0 :=
        sub_ne_zero.mpr (Ne.symm hEq)
      refine cm_card_step _
        ((fun y : Fin n → Bool => D (cmCons true y)) - (fun y => D (cmCons false y)))
        (mleEval fun y => D (cmCons false y))
        (fun rt => mleEval (fun y => D (cmCons true y)) rt - mleEval (fun y => D (cmCons false y)) rt)
        (ih _ hG) ?_
      intro r hr
      have hr0 : mleEval D r = 0 := (Finset.mem_filter.mp hr).2
      by_cases hb : mleEval (fun y => D (cmCons true y)) (cmTail r) -
          mleEval (fun y => D (cmCons false y)) (cmTail r) = 0
      · left
        rw [mleEval_sub]
        exact hb
      · right
        exact cm_mem_root _ _ _ hb ((hsplit r).symm.trans hr0)

/-- Two distinct masks on `{0,1}^vb` have equal multilinear extensions on at most `vb · |E|^{vb-1}` points. -/
theorem constrained_mask_bad_card_le {vb : ℕ} (A M : (Fin vb → Bool) → E) (hne : A ≠ M) :
    (univ.filter fun r : Fin vb → E => mleEval A r = mleEval M r).card * Fintype.card E
      ≤ vb * Fintype.card E ^ vb := by
  have heq : (univ.filter fun r : Fin vb → E => mleEval A r = mleEval M r)
      = univ.filter fun r : Fin vb → E => mleEval (A - M) r = 0 := by
    ext r
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, mleEval_sub, sub_eq_zero]
  rw [heq]
  exact mle_diff_zero_card_le vb (A - M) (sub_ne_zero.mpr hne)

theorem constrained_mask_bad_card_le_of_pos {vb : ℕ} (A M : (Fin vb → Bool) → E) (hne : A ≠ M)
    (hvb : 1 ≤ vb) :
    (univ.filter fun r : Fin vb → E => mleEval A r = mleEval M r).card
      ≤ vb * Fintype.card E ^ (vb - 1) := by
  have h := constrained_mask_bad_card_le A M hne
  have hpos : 0 < Fintype.card E := Fintype.card_pos
  have e : vb * Fintype.card E ^ vb = vb * Fintype.card E ^ (vb - 1) * Fintype.card E := by
    rw [mul_assoc, ← pow_succ, Nat.sub_add_cancel hvb]
  rw [e] at h
  exact Nat.le_of_mul_le_mul_right h hpos

/-- Probability form: `Pr_r[Ã(r) = M̃(r)] ≤ vb / |E|` for `A ≠ M`. -/
theorem constrained_mask_bad_prob {vb : ℕ} (A M : (Fin vb → Bool) → E) (hne : A ≠ M) :
    ((univ.filter fun r : Fin vb → E => mleEval A r = mleEval M r).card : ℝ)
      ≤ ((vb : ℝ) / Fintype.card E) * (Fintype.card (Fin vb → E) : ℝ) := by
  have h := constrained_mask_bad_card_le A M hne
  have hpos : (0 : ℝ) < Fintype.card E := Nat.cast_pos.mpr Fintype.card_pos
  have h' : ((univ.filter fun r : Fin vb → E => mleEval A r = mleEval M r).card : ℝ)
      * Fintype.card E ≤ vb * (Fintype.card E : ℝ) ^ vb := by exact_mod_cast h
  rw [Fintype.card_fun, Fintype.card_fin, Nat.cast_pow, div_mul_eq_mul_div, le_div_iff₀ hpos]
  exact h'

/-- **Soundness of the mask check.** If the committed row `A` differs from the honest row
(`allowed_{q⋆}` on a constrained step, `Unc` otherwise), the verifier equation holds on at most
`vb · |E|^{vb-1}` challenge points. -/
theorem constrained_mask_check_sound {vb Q : ℕ} (A Unc : (Fin vb → Bool) → E)
    (Mtab : Fin Q → (Fin vb → Bool) → E) (oh : Fin Q → ℕ) (q_star : Fin Q) (act : ℕ)
    (hact : act = 0 ∨ act = 1) (hbool : ∀ q, oh q = 0 ∨ oh q = 1) (hsum : ∑ q, oh q = 1)
    (hval : ∑ q, q.val * oh q = q_star.val)
    (hne : A ≠ if act = 1 then Mtab q_star else Unc) :
    (univ.filter fun r : Fin vb → E =>
        mleEval A r = (1 - (act : E)) * mleEval Unc r
          + (act : E) * ∑ q, (oh q : E) * mleEval (Mtab q) r).card * Fintype.card E
      ≤ vb * Fintype.card E ^ vb := by
  have heq : (univ.filter fun r : Fin vb → E =>
        mleEval A r = (1 - (act : E)) * mleEval Unc r
          + (act : E) * ∑ q, (oh q : E) * mleEval (Mtab q) r)
      = univ.filter fun r : Fin vb → E =>
        mleEval A r = mleEval (if act = 1 then Mtab q_star else Unc) r := by
    ext r
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    rw [gated_onehot_select oh q_star hbool hsum hval act hact (fun q => mleEval (Mtab q) r),
      apply_ite (fun f => mleEval f r)]
  rw [heq]
  exact constrained_mask_bad_card_le A _ hne

/-- **Completeness of the mask check.** The honest row passes the verifier equation at every point. -/
theorem constrained_mask_check_complete {vb Q : ℕ} (Unc : (Fin vb → Bool) → E)
    (Mtab : Fin Q → (Fin vb → Bool) → E) (oh : Fin Q → ℕ) (q_star : Fin Q) (act : ℕ)
    (hact : act = 0 ∨ act = 1) (hbool : ∀ q, oh q = 0 ∨ oh q = 1) (hsum : ∑ q, oh q = 1)
    (hval : ∑ q, q.val * oh q = q_star.val) (r : Fin vb → E) :
    mleEval (if act = 1 then Mtab q_star else Unc) r
      = (1 - (act : E)) * mleEval Unc r + (act : E) * ∑ q, (oh q : E) * mleEval (Mtab q) r := by
  rw [gated_onehot_select oh q_star hbool hsum hval act hact (fun q => mleEval (Mtab q) r),
    apply_ite (fun f => mleEval f r)]

/-- Concrete escape bound over `F_{p^3}`, `p = 2^64 - 2^32 + 1`, for vocabularies up to `2^32`. -/
theorem constrained_mask_goldilocks {vb : ℕ} (A M : (Fin vb → Bool) → E) (hne : A ≠ M)
    (hE : Fintype.card E = goldilocks ^ 3) (hvb : vb ≤ 32) :
    ((univ.filter fun r : Fin vb → E => mleEval A r = mleEval M r).card : ℝ)
        ≤ (2 ^ 186)⁻¹ * (Fintype.card (Fin vb → E) : ℝ) ∧
      ((univ.filter fun r : Fin vb → E => mleEval A r = mleEval M r).card : ℝ)
        ≤ (2 ^ 106)⁻¹ * (Fintype.card (Fin vb → E) : ℝ) := by
  have h := constrained_mask_bad_prob A M hne
  rw [hE] at h
  have hp : 0 < goldilocks ^ 3 := by rw [goldilocks_eq]; norm_num
  have hN : (0 : ℝ) ≤ (Fintype.card (Fin vb → E) : ℝ) := Nat.cast_nonneg _
  have b186 : (vb : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 186)⁻¹ := by
    rw [frac_le_inv_pow2_iff _ _ _ hp]
    calc vb * 2 ^ 186 ≤ 32 * 2 ^ 186 := Nat.mul_le_mul_right _ hvb
      _ ≤ goldilocks ^ 3 := by rw [goldilocks_eq]; norm_num
  have b106 : (vb : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 106)⁻¹ := by
    rw [frac_le_inv_pow2_iff _ _ _ hp]
    calc vb * 2 ^ 106 ≤ 32 * 2 ^ 106 := Nat.mul_le_mul_right _ hvb
      _ ≤ goldilocks ^ 3 := by rw [goldilocks_eq]; norm_num
  exact ⟨h.trans (mul_le_mul_of_nonneg_right b186 hN),
    h.trans (mul_le_mul_of_nonneg_right b106 hN)⟩

end SchwartzZippel

end FlashProveNext

#print axioms FlashProveNext.onehot_eq_indicator
#print axioms FlashProveNext.onehot_select
#print axioms FlashProveNext.gate_select
#print axioms FlashProveNext.gated_onehot_select
#print axioms FlashProveNext.cmCons_self
#print axioms FlashProveNext.cmTail_update_zero
#print axioms FlashProveNext.cm_sum_cons
#print axioms FlashProveNext.eqKer_cons
#print axioms FlashProveNext.mleEval_cons
#print axioms FlashProveNext.mleEval_sub
#print axioms FlashProveNext.eqKer_bool
#print axioms FlashProveNext.mleEval_bool
#print axioms FlashProveNext.mleEval_eq_iff
#print axioms FlashProveNext.cmRoot_card_le
#print axioms FlashProveNext.cm_mem_root
#print axioms FlashProveNext.cm_card_filter_tail_le
#print axioms FlashProveNext.cm_arith
#print axioms FlashProveNext.cm_card_step
#print axioms FlashProveNext.mle_diff_zero_card_le
#print axioms FlashProveNext.constrained_mask_bad_card_le
#print axioms FlashProveNext.constrained_mask_bad_card_le_of_pos
#print axioms FlashProveNext.constrained_mask_bad_prob
#print axioms FlashProveNext.constrained_mask_check_sound
#print axioms FlashProveNext.constrained_mask_check_complete
#print axioms FlashProveNext.constrained_mask_goldilocks
