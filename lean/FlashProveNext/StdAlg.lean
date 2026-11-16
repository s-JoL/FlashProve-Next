/-
Stepwise determinism and uniqueness of local reference traces.

* `trace_unique`       : functional step relations determine a unique intermediate trace from any fixed input;
* `trace_exists`       : total step relations admit a valid trace from any initial state;
* `round_unique`       : half-up integer rounding `y ≤ x + 1/2 < y + 1` has a unique integer solution with `|x - y| ≤ 1/2`;
* `verdict_determined` : any block-level verdict on the final state is invariant across valid witness traces.
-/
import Mathlib

namespace FlashProveNext

theorem trace_unique {S : Type*} (step : ℕ → S → S → Prop)
    (hfun : ∀ i s a b, step i s a → step i s b → a = b)
    (n : ℕ) (τ τ' : ℕ → S) (h0 : τ 0 = τ' 0)
    (h : ∀ i < n, step i (τ i) (τ (i + 1))) (h' : ∀ i < n, step i (τ' i) (τ' (i + 1))) :
    ∀ i ≤ n, τ i = τ' i := by
  intro i
  induction i with
  | zero => intro _; exact h0
  | succ j ih =>
    intro hj
    have hj' : j < n := hj
    have e := ih (Nat.le_of_lt hj')
    have a := h j hj'
    have b := h' j hj'
    rw [e] at a
    exact hfun j _ _ _ a b

theorem trace_exists {S : Type*} (step : ℕ → S → S → Prop)
    (htot : ∀ i s, ∃ a, step i s a) (s0 : S) :
    ∃ τ : ℕ → S, τ 0 = s0 ∧ ∀ i, step i (τ i) (τ (i + 1)) := by
  classical
  choose f hf using htot
  exact ⟨fun i => Nat.rec s0 (fun j s => f j s) i, rfl, fun i => hf i _⟩

theorem round_unique (x : ℝ) : ∃! y : ℤ, (y : ℝ) ≤ x + 1 / 2 ∧ x + 1 / 2 < y + 1 := by
  refine ⟨⌊x + 1 / 2⌋, ⟨Int.floor_le _, Int.lt_floor_add_one _⟩, ?_⟩
  rintro y ⟨h1, h2⟩
  exact (Int.floor_eq_iff.mpr ⟨h1, h2⟩).symm

theorem round_error (x : ℝ) (y : ℤ) (h1 : (y : ℝ) ≤ x + 1 / 2) (h2 : x + 1 / 2 < y + 1) :
    |x - y| ≤ 1 / 2 := by
  rw [abs_le]; constructor <;> linarith

theorem verdict_determined {S : Type*} (step : ℕ → S → S → Prop)
    (hfun : ∀ i s a b, step i s a → step i s b → a = b)
    (n : ℕ) (τ τ' : ℕ → S) (h0 : τ 0 = τ' 0)
    (h : ∀ i < n, step i (τ i) (τ (i + 1))) (h' : ∀ i < n, step i (τ' i) (τ' (i + 1)))
    (rel : S → Prop) : rel (τ n) ↔ rel (τ' n) := by
  rw [trace_unique step hfun n τ τ' h0 h h' n le_rfl]

end FlashProveNext

#print axioms FlashProveNext.trace_unique
#print axioms FlashProveNext.trace_exists
#print axioms FlashProveNext.round_unique
#print axioms FlashProveNext.verdict_determined
