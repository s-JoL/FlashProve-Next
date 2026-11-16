/-
Chunkwise linear and gated recurrent attention (`LREC`: Gated DeltaNet, Mamba-2; `specs/05-computation-model.md` §2.4):

1. `chunkRun_eq_recRun`          : folding `step : S → X → S × Y` across arbitrary chunks equals token-by-token
                                   sequential execution on `chunks.flatten`;
2. `lrec_state_unique`,
   `lrec_exists_bad_of_mismatch` : committing recurrent state only at chunk boundaries uniquely fixes all boundary
                                   states and outputs, forcing any state-truncation or zeroing cheat to leave a bad chunk;
3. `lrec_state_bytes_bound`,
   `lrec_spotcheck_bound`        : `≥ B_c ×` state-commitment bandwidth reduction and `(1 - σ)^k` spot-check miss bound.
-/
import Mathlib
import FlashProveNext.SpotCheck

open scoped BigOperators

namespace FlashProveNext

section LREC

def recRun {S X Y : Type*} (step : S → X → S × Y) : S → List X → S × List Y
  | s, [] => (s, [])
  | s, x :: xs =>
      let p := step s x
      let q := recRun step p.1 xs
      (q.1, p.2 :: q.2)

@[simp]
theorem recRun_nil {S X Y : Type*} (step : S → X → S × Y) (s : S) :
    recRun step s [] = (s, []) := rfl

@[simp]
theorem recRun_length {S X Y : Type*} (step : S → X → S × Y) (s : S) (xs : List X) :
    (recRun step s xs).2.length = xs.length := by
  induction xs generalizing s with
  | nil => rfl
  | cons _ _ ih => simp [recRun, ih]

theorem recRun_append {S X Y : Type*} (step : S → X → S × Y) (s : S) (xs zs : List X) :
    recRun step s (xs ++ zs) =
      let p := recRun step s xs
      let q := recRun step p.1 zs
      (q.1, p.2 ++ q.2) := by
  induction xs generalizing s with
  | nil => simp [recRun]
  | cons x xs' ih =>
    simp only [List.cons_append, recRun]
    rw [ih (step s x).1]

def chunkRun {S X Y : Type*} (step : S → X → S × Y) : S → List (List X) → S × List Y
  | s, [] => (s, [])
  | s, c :: cs =>
      let p := recRun step s c
      let q := chunkRun step p.1 cs
      (q.1, p.2 ++ q.2)

theorem chunkRun_eq_recRun {S X Y : Type*} (step : S → X → S × Y) (s : S) (chunks : List (List X)) :
    chunkRun step s chunks = recRun step s chunks.flatten := by
  induction chunks generalizing s with
  | nil => rfl
  | cons c cs ih =>
    simp only [chunkRun, List.flatten_cons]
    rw [recRun_append, ih]

def prefixInputs {X : Type*} (chunk : ℕ → List X) : ℕ → List X
  | 0 => []
  | j + 1 => prefixInputs chunk j ++ chunk j

def prefixOutputs {Y : Type*} (outs : ℕ → List Y) : ℕ → List Y
  | 0 => []
  | j + 1 => prefixOutputs outs j ++ outs j

def goodChunkNat {S X Y : Type*} (step : S → X → S × Y) (chunk : ℕ → List X)
    (st : ℕ → S) (outs : ℕ → List Y) (m : ℕ) : Prop :=
  (st (m + 1), outs m) = recRun step (st m) (chunk m)

theorem lrec_state_unique {S X Y : Type*} (step : S → X → S × Y)
    (chunk : ℕ → List X) (s₀ : S) (st : ℕ → S) (outs : ℕ → List Y) (M : ℕ)
    (h0 : st 0 = s₀) (hgood : ∀ m < M, goodChunkNat step chunk st outs m) :
    ∀ j ≤ M, (st j, prefixOutputs outs j) = recRun step s₀ (prefixInputs chunk j) := by
  intro j
  induction j with
  | zero =>
    intro _
    simp [prefixInputs, prefixOutputs, h0]
  | succ i ih =>
    intro hj
    have hi : i ≤ M := Nat.le_of_succ_le hj
    have hi_lt : i < M := Nat.lt_of_succ_le hj
    have hprev := ih hi
    have hm := hgood i hi_lt
    unfold goodChunkNat at hm
    simp only [prefixInputs, prefixOutputs, recRun_append]
    have hst : (recRun step s₀ (prefixInputs chunk i)).1 = st i := (Prod.ext_iff.mp hprev.symm).1
    have hout : (recRun step s₀ (prefixInputs chunk i)).2 = prefixOutputs outs i := (Prod.ext_iff.mp hprev.symm).2
    rw [hst, hout]
    have hm1 : (recRun step (st i) (chunk i)).1 = st (i + 1) := (Prod.ext_iff.mp hm.symm).1
    have hm2 : (recRun step (st i) (chunk i)).2 = outs i := (Prod.ext_iff.mp hm.symm).2
    rw [hm1, hm2]

theorem lrec_exists_bad_of_mismatch {S X Y : Type*} (step : S → X → S × Y)
    (chunk : ℕ → List X) (s₀ : S) (st : ℕ → S) (outs : ℕ → List Y) (M : ℕ)
    (h0 : st 0 = s₀)
    (hmis : (st M, prefixOutputs outs M) ≠ recRun step s₀ (prefixInputs chunk M)) :
    ∃ m < M, ¬ goodChunkNat step chunk st outs m := by
  by_contra hall
  exact hmis (lrec_state_unique step chunk s₀ st outs M h0 (fun m hm => Classical.not_not.mp fun hng => hall ⟨m, hm, hng⟩) M (le_refl M))

theorem lrec_state_bytes_bound (n Bc stateBytes : ℕ) :
    Bc * ((n / Bc) * stateBytes) ≤ n * stateBytes := by
  calc
    Bc * ((n / Bc) * stateBytes) = (Bc * (n / Bc)) * stateBytes := by ring
    _ ≤ n * stateBytes := Nat.mul_le_mul_right stateBytes (Nat.mul_div_le n Bc)

theorem lrec_spotcheck_bound {I C : Type*} [Fintype I] [DecidableEq I] [DecidableEq C]
    (π : I → C) (Bad : Finset C) (k num den : ℕ)
    (hbad : num * Fintype.card I ≤ den * (Finset.univ.filter fun i => π i ∈ Bad).card) :
    (Finset.univ.filter fun q : Fin k → I => ∀ j, π (q j) ∉ Bad).card * den ^ k
      ≤ ((den - num) * Fintype.card I) ^ k :=
  cost_weighted_bound π Bad k num den hbad

end LREC

end FlashProveNext

#print axioms FlashProveNext.recRun_length
#print axioms FlashProveNext.recRun_append
#print axioms FlashProveNext.chunkRun_eq_recRun
#print axioms FlashProveNext.lrec_state_unique
#print axioms FlashProveNext.lrec_exists_bad_of_mismatch
#print axioms FlashProveNext.lrec_state_bytes_bound
#print axioms FlashProveNext.lrec_spotcheck_bound
