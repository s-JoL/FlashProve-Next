/-
Coupled keyed speculative decoding equivalence (`specs/05-computation-model.md` §3.3).

When the committed keyed sampling rule determines a deterministic step transition `tgt : List α → α`
for fixed `(K_prf, nonce)` and logical step position `ctx.length`, multi-round speculative verification
against any untrusted, uncommitted draft proposer `draft : List α → List α` produces the exact same
token sequence as sequential target decoding (`specLoop_eq_seqGen`), requiring zero draft-model
commitment or extra circuit state.
-/
import Mathlib

namespace FlashProveNext

def seqGen {α : Type*} (tgt : List α → α) : ℕ → List α → List α
  | 0, _ => []
  | n + 1, ctx =>
      let y := tgt ctx
      y :: seqGen tgt n (ctx ++ [y])

@[simp]
theorem seqGen_length {α : Type*} (tgt : List α → α) (n : ℕ) (ctx : List α) :
    (seqGen tgt n ctx).length = n := by
  induction n generalizing ctx with
  | zero => rfl
  | succ m ih => simp [seqGen, ih]

theorem seqGen_add {α : Type*} (tgt : List α → α) (m k : ℕ) (ctx : List α) :
    seqGen tgt (m + k) ctx = seqGen tgt m ctx ++ seqGen tgt k (ctx ++ seqGen tgt m ctx) := by
  induction m generalizing ctx with
  | zero => simp [seqGen]
  | succ m' ih =>
    simp only [Nat.succ_add, seqGen, List.cons_append]
    rw [ih (ctx ++ [tgt ctx]), List.append_assoc]
    simp

theorem seqGen_take {α : Type*} (tgt : List α → α) (m n : ℕ) (h : n ≤ m) (ctx : List α) :
    (seqGen tgt m ctx).take n = seqGen tgt n ctx := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [seqGen_add, List.take_left' (seqGen_length tgt n ctx)]

/-- Single-round coupled speculative verification: accept longest matching prefix and append target token at mismatch. -/
def specVerify {α : Type*} [DecidableEq α] (tgt : List α → α) : List α → List α → List α
  | ctx, [] => [tgt ctx]
  | ctx, d :: ds =>
      if d = tgt ctx then
        d :: specVerify tgt (ctx ++ [d]) ds
      else
        [tgt ctx]

theorem specVerify_length_bounds {α : Type*} [DecidableEq α] (tgt : List α → α) (ctx ds : List α) :
    1 ≤ (specVerify tgt ctx ds).length ∧ (specVerify tgt ctx ds).length ≤ ds.length + 1 := by
  induction ds generalizing ctx with
  | nil => simp [specVerify]
  | cons d ds' ih =>
    simp only [specVerify]
    split_ifs with hd
    · obtain ⟨hlo, hhi⟩ := ih (ctx ++ [d])
      simp only [List.length_cons]
      omega
    · simp

theorem specVerify_eq_seqGen {α : Type*} [DecidableEq α] (tgt : List α → α) (ctx ds : List α) :
    specVerify tgt ctx ds = seqGen tgt (specVerify tgt ctx ds).length ctx := by
  induction ds generalizing ctx with
  | nil => simp [specVerify, seqGen]
  | cons d ds' ih =>
    simp only [specVerify]
    split_ifs with hd
    · subst hd
      simp only [List.length_cons, seqGen]
      congr 1
      exact ih (ctx ++ [tgt ctx])
    · simp [seqGen]

def specRounds {α : Type*} [DecidableEq α] (tgt : List α → α) (draft : List α → List α) :
    ℕ → ℕ → List α → List α
  | _, 0, _ => []
  | 0, rem + 1, ctx => seqGen tgt (rem + 1) ctx
  | fuel + 1, rem + 1, ctx =>
      let chunk := specVerify tgt ctx (draft ctx)
      if chunk.length ≤ rem + 1 then
        chunk ++ specRounds tgt draft fuel (rem + 1 - chunk.length) (ctx ++ chunk)
      else
        chunk.take (rem + 1)

def specLoop {α : Type*} [DecidableEq α] (tgt : List α → α) (draft : List α → List α)
    (n : ℕ) (ctx : List α) : List α :=
  specRounds tgt draft n n ctx

theorem specRounds_eq_seqGen {α : Type*} [DecidableEq α] (tgt : List α → α) (draft : List α → List α)
    (fuel rem : ℕ) (ctx : List α) :
    specRounds tgt draft fuel rem ctx = seqGen tgt rem ctx := by
  induction fuel generalizing rem ctx with
  | zero =>
    cases rem <;> simp [specRounds, seqGen]
  | succ f ih =>
    cases rem with
    | zero => simp [specRounds, seqGen]
    | succ m =>
      simp only [specRounds]
      set chunk := specVerify tgt ctx (draft ctx)
      have hchunk : chunk = seqGen tgt chunk.length ctx := specVerify_eq_seqGen tgt ctx (draft ctx)
      split_ifs with hle
      · rw [ih (m + 1 - chunk.length) (ctx ++ chunk)]
        have hsum : chunk.length + (m + 1 - chunk.length) = m + 1 := Nat.add_sub_of_le hle
        have hrhs : seqGen tgt (m + 1) ctx = chunk ++ seqGen tgt (m + 1 - chunk.length) (ctx ++ chunk) := by
          calc
            seqGen tgt (m + 1) ctx
                = seqGen tgt (chunk.length + (m + 1 - chunk.length)) ctx := by rw [hsum]
            _ = seqGen tgt chunk.length ctx ++ seqGen tgt (m + 1 - chunk.length) (ctx ++ seqGen tgt chunk.length ctx) :=
                seqGen_add tgt chunk.length (m + 1 - chunk.length) ctx
            _ = chunk ++ seqGen tgt (m + 1 - chunk.length) (ctx ++ chunk) := by rw [← hchunk]
        exact hrhs.symm
      · have hle' : m + 1 ≤ chunk.length := by omega
        rw [hchunk, seqGen_take tgt chunk.length (m + 1) hle' ctx]

/-- Multi-round coupled speculative decoding equals sequential target decoding for every draft proposer `draft`. -/
theorem specLoop_eq_seqGen {α : Type*} [DecidableEq α] (tgt : List α → α) (draft : List α → List α)
    (n : ℕ) (ctx : List α) :
    specLoop tgt draft n ctx = seqGen tgt n ctx :=
  specRounds_eq_seqGen tgt draft n n ctx

end FlashProveNext

#print axioms FlashProveNext.seqGen_length
#print axioms FlashProveNext.seqGen_add
#print axioms FlashProveNext.seqGen_take
#print axioms FlashProveNext.specVerify_length_bounds
#print axioms FlashProveNext.specVerify_eq_seqGen
#print axioms FlashProveNext.specRounds_eq_seqGen
#print axioms FlashProveNext.specLoop_eq_seqGen
