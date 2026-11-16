/-
Joint blinding, causal multi-round zero knowledge, and Libra sumcheck masking (`specs/03-zero-knowledge.md`).

* `fiber_card_mul`                  : exact fiber cardinality `|G| / |H|` for surjective homomorphisms;
* `joint_tensor_surjective`         : joint surjectivity of `m` Reed–Solomon codeword queries and a multilinear
                                      evaluation `eq(z, ·)` on shared blinding rows;
* `fold_mask_independent`           : ZK-Basefold folded-codeword masking `f + β • r` (`β ≠ 0`);
* `postprocess_independent`         : closure of secret-independence under arbitrary deterministic post-processing;
* `causal_transcript_existsUnique`,
  `multiround_independent_causal`   : causal multi-round masking (`specs/03` §4.3): when round-`i` shift `A i w t`
                                      depends strictly on prefix `j < i`, every `(w, r)` induces a unique transcript
                                      `evalTranscript A L w r`, and uniform masks induce the exact uniform distribution
                                      on transcripts independently of `w`;
* `libraG_eq_cubeSum`, `libra_mask_bijective`, `libra_transcript_independent`, `libra_mask_bijOn_hyperplane` :
                                      Libra outer sumcheck mask bijection on the round-sum consistency hyperplane.
-/
import Mathlib
import FlashProveNext.Mask

namespace FlashProveNext

open Finset

set_option linter.unusedSectionVars false

theorem fiber_card_mul {G H : Type*} [AddCommGroup G] [AddCommGroup H] [Fintype G] [Fintype H]
    [DecidableEq H] (L : G →+ H) (hs : Function.Surjective L) (y : H) :
    (Finset.univ.filter fun r => L r = y).card * Fintype.card H = Fintype.card G := by
  have h1 : ∑ z : H, (Finset.univ.filter fun r => L r = z).card = Fintype.card G := by
    rw [← Finset.card_eq_sum_card_fiberwise (f := L) (s := Finset.univ) (t := Finset.univ)
      (fun _ _ => Finset.mem_univ _)]
    simp
  have h2 : ∑ z : H, (Finset.univ.filter fun r => L r = z).card
      = Fintype.card H * (Finset.univ.filter fun r => L r = y).card := by
    rw [Finset.sum_congr rfl (fun z _ => fiber_card_eq L hs z y)]
    simp
  rw [← h1, h2, mul_comm]

theorem multiround_independent {ι : Type*} [Fintype ι] [DecidableEq ι] {W : Type*}
    {G H : ι → Type*} [∀ i, AddCommGroup (G i)] [∀ i, AddCommGroup (H i)] [∀ i, Fintype (G i)]
    [∀ i, DecidableEq (H i)] (L : ∀ i, G i →+ H i) (hL : ∀ i, Function.Surjective (L i))
    (A : ∀ i, W → (∀ j, H j) → H i) (w w' : W) (t : ∀ j, H j) :
    (Finset.univ.filter fun r : ∀ i, G i => ∀ i, A i w t + L i (r i) = t i).card
      = (Finset.univ.filter fun r : ∀ i, G i => ∀ i, A i w' t + L i (r i) = t i).card := by
  have e : ∀ x : W, (Finset.univ.filter fun r : ∀ i, G i => ∀ i, A i x t + L i (r i) = t i)
      = Fintype.piFinset (fun i => Finset.univ.filter fun g : G i => L i g = t i - A i x t) := by
    intro x; ext r
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fintype.mem_piFinset]
    refine forall_congr' fun i => ?_
    constructor
    · intro h; rw [← h]; abel
    · intro h; rw [h]; abel
  rw [e w, e w', Fintype.card_piFinset, Fintype.card_piFinset]
  exact Finset.prod_congr rfl (fun i _ => fiber_card_eq (L i) (hL i) _ _)

/-- Causality (prefix-measurability): round-`i` witness shift depends strictly on earlier transcript coordinates `j < i`. -/
def IsCausal {m : ℕ} {W : Type*} {H : Fin m → Type*}
    (A : (i : Fin m) → W → ((j : Fin m) → H j) → H i) : Prop :=
  ∀ (i : Fin m) (w : W) (t t' : (j : Fin m) → H j),
    (∀ j : Fin m, j < i → t j = t' j) → A i w t = A i w t'

def partialTranscript {m : ℕ} {W : Type*} {G H : Fin m → Type*}
    [∀ i, AddCommGroup (G i)] [∀ i, AddCommGroup (H i)]
    (A : (i : Fin m) → W → ((j : Fin m) → H j) → H i)
    (L : ∀ i, G i →+ H i) (w : W) (r : ∀ i, G i) : ℕ → (∀ j : Fin m, H j)
  | 0 => 0
  | k + 1 => fun j =>
      if (j : ℕ) = k then A j w (partialTranscript A L w r k) + L j (r j)
      else partialTranscript A L w r k j

def evalTranscript {m : ℕ} {W : Type*} {G H : Fin m → Type*}
    [∀ i, AddCommGroup (G i)] [∀ i, AddCommGroup (H i)]
    (A : (i : Fin m) → W → ((j : Fin m) → H j) → H i)
    (L : ∀ i, G i →+ H i) (w : W) (r : ∀ i, G i) : ∀ j : Fin m, H j :=
  partialTranscript A L w r m

theorem partialTranscript_stable {m : ℕ} {W : Type*} {G H : Fin m → Type*}
    [∀ i, AddCommGroup (G i)] [∀ i, AddCommGroup (H i)]
    (A : (i : Fin m) → W → ((j : Fin m) → H j) → H i)
    (L : ∀ i, G i →+ H i) (w : W) (r : ∀ i, G i) {k k' : ℕ} (hk : k ≤ k')
    (j : Fin m) (hj : (j : ℕ) < k) :
    partialTranscript A L w r k' j = partialTranscript A L w r k j := by
  induction k' with
  | zero => omega
  | succ n ih =>
    rcases Nat.of_le_succ hk with hle | rfl
    · have hne : (j : ℕ) ≠ n := by omega
      simp [partialTranscript, hne, ih hle]
    · rfl

theorem evalTranscript_spec {m : ℕ} {W : Type*} {G H : Fin m → Type*}
    [∀ i, AddCommGroup (G i)] [∀ i, AddCommGroup (H i)]
    (A : (i : Fin m) → W → ((j : Fin m) → H j) → H i)
    (L : ∀ i, G i →+ H i) (hA : IsCausal A) (w : W) (r : ∀ i, G i) (i : Fin m) :
    A i w (evalTranscript A L w r) + L i (r i) = evalTranscript A L w r i := by
  have hi : evalTranscript A L w r i = A i w (partialTranscript A L w r (i : ℕ)) + L i (r i) := by
    unfold evalTranscript
    rw [partialTranscript_stable A L w r (Nat.succ_le_of_lt i.isLt) i (Nat.lt_succ_self _)]
    simp [partialTranscript]
  have hpref : ∀ j : Fin m, j < i →
      evalTranscript A L w r j = partialTranscript A L w r (i : ℕ) j := by
    intro j hj
    exact partialTranscript_stable A L w r (Nat.le_of_lt i.isLt) j hj
  rw [hi, hA i w _ _ hpref]

theorem causal_transcript_unique {m : ℕ} {W : Type*} {G H : Fin m → Type*}
    [∀ i, AddCommGroup (G i)] [∀ i, AddCommGroup (H i)]
    (A : (i : Fin m) → W → ((j : Fin m) → H j) → H i)
    (L : ∀ i, G i →+ H i) (hA : IsCausal A) (w : W) (r : ∀ i, G i)
    {t₁ t₂ : ∀ j : Fin m, H j}
    (ht₁ : ∀ i, A i w t₁ + L i (r i) = t₁ i) (ht₂ : ∀ i, A i w t₂ + L i (r i) = t₂ i) :
    t₁ = t₂ := by
  have key : ∀ (n : ℕ) (i : Fin m), (i : ℕ) < n → t₁ i = t₂ i := by
    intro n
    induction n with
    | zero => intro i hi; omega
    | succ n ih =>
      intro i hi
      rcases Nat.lt_succ_iff_lt_or_eq.mp hi with hlt | heq
      · exact ih i hlt
      · have hpref : ∀ j : Fin m, j < i → t₁ j = t₂ j := fun j hj => ih j (by omega)
        rw [← ht₁ i, ← ht₂ i, hA i w t₁ t₂ hpref]
  funext i
  exact key m i i.isLt

theorem causal_transcript_existsUnique {m : ℕ} {W : Type*} {G H : Fin m → Type*}
    [∀ i, AddCommGroup (G i)] [∀ i, AddCommGroup (H i)]
    (A : (i : Fin m) → W → ((j : Fin m) → H j) → H i)
    (L : ∀ i, G i →+ H i) (hA : IsCausal A) (w : W) (r : ∀ i, G i) :
    ∃! t : ∀ j : Fin m, H j, ∀ i, A i w t + L i (r i) = t i :=
  ⟨evalTranscript A L w r, evalTranscript_spec A L hA w r,
    fun _ ht => causal_transcript_unique A L hA w r ht (evalTranscript_spec A L hA w r)⟩

theorem evalTranscript_eq_iff {m : ℕ} {W : Type*} {G H : Fin m → Type*}
    [∀ i, AddCommGroup (G i)] [∀ i, AddCommGroup (H i)]
    (A : (i : Fin m) → W → ((j : Fin m) → H j) → H i)
    (L : ∀ i, G i →+ H i) (hA : IsCausal A) (w : W) (r : ∀ i, G i) (t : ∀ j : Fin m, H j) :
    evalTranscript A L w r = t ↔ ∀ i, A i w t + L i (r i) = t i := by
  constructor
  · rintro rfl; exact evalTranscript_spec A L hA w r
  · intro ht; exact causal_transcript_unique A L hA w r (evalTranscript_spec A L hA w r) ht

theorem multiround_independent_causal {m : ℕ} {W : Type*} {G H : Fin m → Type*}
    [∀ i, AddCommGroup (G i)] [∀ i, AddCommGroup (H i)]
    [∀ i, Fintype (G i)] [∀ i, Fintype (H i)] [∀ i, DecidableEq (H i)]
    (L : ∀ i, G i →+ H i) (hL : ∀ i, Function.Surjective (L i))
    (A : (i : Fin m) → W → ((j : Fin m) → H j) → H i) (hA : IsCausal A)
    (w w' : W) (t : ∀ j : Fin m, H j) :
    (Finset.univ.filter fun r : ∀ i, G i => evalTranscript A L w r = t).card
      = (Finset.univ.filter fun r : ∀ i, G i => evalTranscript A L w' r = t).card ∧
    (Finset.univ.filter fun r : ∀ i, G i => evalTranscript A L w r = t).card
      = ∏ i : Fin m, (Fintype.card (G i) / Fintype.card (H i)) ∧
    (Finset.univ.filter fun r : ∀ i, G i => evalTranscript A L w r = t).card
      * Fintype.card (∀ i, H i) = Fintype.card (∀ i, G i) := by
  have heq : ∀ x : W,
      (Finset.univ.filter fun r : ∀ i, G i => evalTranscript A L x r = t)
        = (Finset.univ.filter fun r : ∀ i, G i => ∀ i, A i x t + L i (r i) = t i) := by
    intro x; ext r
    simp [evalTranscript_eq_iff A L hA x r t]
  have hpi : ∀ x : W,
      (Finset.univ.filter fun r : ∀ i, G i => ∀ i, A i x t + L i (r i) = t i)
        = Fintype.piFinset (fun i => Finset.univ.filter fun g : G i => L i g = t i - A i x t) := by
    intro x; ext r
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Fintype.mem_piFinset]
    refine forall_congr' fun i => ?_
    constructor
    · intro h; rw [← h]; abel
    · intro h; rw [h]; abel
  have hfib_mul : ∀ (x : W) (i : Fin m),
      (Finset.univ.filter fun g : G i => L i g = t i - A i x t).card * Fintype.card (H i)
        = Fintype.card (G i) :=
    fun x i => fiber_card_mul (L i) (hL i) (t i - A i x t)
  have hfib_div : ∀ (x : W) (i : Fin m),
      (Finset.univ.filter fun g : G i => L i g = t i - A i x t).card
        = Fintype.card (G i) / Fintype.card (H i) := by
    intro x i
    have hpos : 0 < Fintype.card (H i) := Fintype.card_pos
    rw [← hfib_mul x i, Nat.mul_div_cancel _ hpos]
  have hcard : ∀ x : W,
      (Finset.univ.filter fun r : ∀ i, G i => evalTranscript A L x r = t).card
        = ∏ i : Fin m, (Fintype.card (G i) / Fintype.card (H i)) := by
    intro x
    rw [heq x, hpi x, Fintype.card_piFinset]
    exact Finset.prod_congr rfl fun i _ => hfib_div x i
  refine ⟨(hcard w).trans (hcard w').symm, hcard w, ?_⟩
  rw [heq w, hpi w, Fintype.card_piFinset, Fintype.card_pi, Fintype.card_pi,
    ← Finset.prod_mul_distrib]
  exact Finset.prod_congr rfl fun i _ => hfib_mul w i

theorem fold_mask_independent {F C : Type*} [Field F] [AddCommGroup C] [Module F C] [Fintype C]
    [DecidableEq C] (β : F) (hβ : β ≠ 0) (f f' g : C) :
    (Finset.univ.filter fun r : C => f + β • r = g).card
      = (Finset.univ.filter fun r : C => f' + β • r = g).card := by
  have hs : Function.Surjective (DistribSMul.toAddMonoidHom C β) :=
    fun y => ⟨β⁻¹ • y, by simp [hβ]⟩
  exact masked_view_independent (fun f : C => f) (DistribSMul.toAddMonoidHom C β) hs f f' g

theorem postprocess_independent {W G H K : Type*} [AddCommGroup G] [AddCommGroup H] [Fintype G]
    [DecidableEq G] [Fintype H] [DecidableEq H] [DecidableEq K] (S : W → H) (L : G →+ H) (hs : Function.Surjective L)
    (φ : H → K) (w w' : W) (k : K) :
    (Finset.univ.filter fun r => φ (S w + L r) = k).card
      = (Finset.univ.filter fun r => φ (S w' + L r) = k).card := by
  have e : ∀ x : W, (Finset.univ.filter fun r => φ (S x + L r) = k)
      = (Finset.univ.filter fun v : H => φ v = k).biUnion
          (fun v => Finset.univ.filter fun r => S x + L r = v) := by
    intro x; ext r; simp
  have hd : ∀ x : W, ((Finset.univ.filter fun v : H => φ v = k).biUnion
          (fun v => Finset.univ.filter fun r => S x + L r = v)).card
      = ∑ v ∈ (Finset.univ.filter fun v : H => φ v = k),
          (Finset.univ.filter fun r => S x + L r = v).card := by
    intro x
    apply Finset.card_biUnion
    intro v _ v' _ hvv'
    show Disjoint _ _
    rw [Finset.disjoint_left]
    intro r hr hr'
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hr hr'
    exact hvv' (hr.symm.trans hr')
  rw [e w, e w', hd w, hd w']
  exact Finset.sum_congr rfl (fun v _ => masked_view_independent S L hs w w' v)

theorem joint_disjoint_surjective {F : Type*} [Field F] {m ℓ ℓ' : ℕ} (x : Fin m → F)
    (hx : Function.Injective x) (hm : m ≤ ℓ) (e : Fin ℓ' → F) (he : e ≠ 0) :
    Function.Surjective (fun p : (Fin ℓ → F) × (Fin ℓ' → F) =>
      ((fun i : Fin m => ∑ j : Fin ℓ, p.1 j * x i ^ (j : ℕ)), ∑ j : Fin ℓ', p.2 j * e j)) := by
  have h1 := vandermonde_surjective x (fun _ => (1 : F)) hx (fun _ => one_ne_zero) hm
  rintro ⟨y, t⟩
  obtain ⟨p, hp⟩ := h1 y
  obtain ⟨j0, hj0⟩ := Function.ne_iff.mp he
  refine ⟨(p, Pi.single j0 (t / e j0)), ?_⟩
  refine Prod.ext ?_ ?_
  · funext i
    have := congrFun hp i
    simpa using this
  · simp [Pi.single_apply, Finset.sum_ite_eq', div_mul_cancel₀ _ (show e j0 ≠ 0 from hj0)]

theorem joint_surjective_of_det {F : Type*} [Field F] {t : ℕ} (M : Matrix (Fin t) (Fin t) F)
    (h : M.det ≠ 0) : Function.Surjective M.mulVec :=
  Matrix.mulVec_surjective_iff_isUnit.mpr ((Matrix.isUnit_iff_isUnit_det M).mpr (isUnit_iff_ne_zero.mpr h))

theorem joint_tensor_surjective {F : Type*} [Field F] {R m ℓ : ℕ} (x : Fin m → F)
    (hx : Function.Injective x) (hm : m ≤ ℓ) (e : Fin ℓ → F) (he : e ≠ 0)
    (u w : Fin R → F) (k0 k1 : Fin R) (hΔ : u k0 * w k1 - u k1 * w k0 ≠ 0) :
    Function.Surjective (fun r : Fin R → Fin ℓ → F =>
      ((fun i : Fin m => ∑ k, u k * ∑ j, r k j * x i ^ (j : ℕ)), ∑ k, w k * ∑ j, r k j * e j)) := by
  classical
  intro yt
  obtain ⟨⟨p, q⟩, hpq⟩ := joint_disjoint_surjective x hx hm e he yt
  set Δ := u k0 * w k1 - u k1 * w k0 with hΔdef
  let α : Fin R → F := Pi.single k0 (w k1 / Δ) + Pi.single k1 (-(w k0) / Δ)
  let β : Fin R → F := Pi.single k0 (-(u k1) / Δ) + Pi.single k1 (u k0 / Δ)
  have hua : ∑ k, u k * α k = 1 := by
    simp only [α, Pi.add_apply, mul_add, Finset.sum_add_distrib, Pi.single_apply, mul_ite, mul_zero,
      Finset.sum_ite_eq', Finset.mem_univ, if_true]
    field_simp; ring
  have hub : ∑ k, u k * β k = 0 := by
    simp only [β, Pi.add_apply, mul_add, Finset.sum_add_distrib, Pi.single_apply, mul_ite, mul_zero,
      Finset.sum_ite_eq', Finset.mem_univ, if_true]
    field_simp; ring
  have hwa : ∑ k, w k * α k = 0 := by
    simp only [α, Pi.add_apply, mul_add, Finset.sum_add_distrib, Pi.single_apply, mul_ite, mul_zero,
      Finset.sum_ite_eq', Finset.mem_univ, if_true]
    field_simp; ring
  have hwb : ∑ k, w k * β k = 1 := by
    simp only [β, Pi.add_apply, mul_add, Finset.sum_add_distrib, Pi.single_apply, mul_ite, mul_zero,
      Finset.sum_ite_eq', Finset.mem_univ, if_true]
    field_simp; ring
  refine ⟨fun k => α k • p + β k • q, ?_⟩
  have key : ∀ (c : Fin ℓ → F) (g : Fin R → F),
      ∑ k, g k * ∑ j, (α k • p + β k • q) j * c j
        = (∑ k, g k * α k) * (∑ j, p j * c j) + (∑ k, g k * β k) * (∑ j, q j * c j) := by
    intro c g
    have : ∀ k, ∑ j, (α k • p + β k • q) j * c j = α k * (∑ j, p j * c j) + β k * (∑ j, q j * c j) := by
      intro k
      simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, add_mul, Finset.sum_add_distrib,
        mul_assoc, ← Finset.mul_sum]
    simp only [this, mul_add, Finset.sum_add_distrib, Finset.sum_mul]
    congr 1 <;> exact Finset.sum_congr rfl (fun k _ => by ring)
  obtain ⟨hp1, hp2⟩ := Prod.ext_iff.mp hpq
  refine Prod.ext ?_ ?_
  · funext i
    have := congrFun hp1 i
    simp only [key, hua, hub, one_mul, zero_mul, add_zero]
    simpa using this
  · simp only [key, hwa, hwb, zero_mul, one_mul, zero_add]
    simpa using hp2

section libra

variable {F : Type*} [Field F] {n d : ℕ} {ρ : F}

def libraW (K : Type*) [Field K] (n : ℕ) (i : Fin n) : K := 2 ^ (n - 1 - (i : ℕ))

def libraG (a0 : F) (a : Fin n → Fin d → F) : F := 2 ^ n * a0 + 2 ^ (n - 1) * ∑ i, ∑ k, a i k

theorem libraG_eq_cubeSum [DecidableEq F] (a0 : F) (a : Fin n → Fin d → F) :
    ∑ x ∈ Fintype.piFinset (fun _ : Fin n => ({0, 1} : Finset F)),
      (a0 + ∑ i, ∑ k : Fin d, a i k * x i ^ ((k : ℕ) + 1)) = libraG a0 a := by
  set cube : Finset (Fin n → F) := Fintype.piFinset (fun _ : Fin n => ({0, 1} : Finset F)) with hcube
  have hcard : cube.card = 2 ^ n := by
    rw [hcube, Fintype.card_piFinset]
    simp [Finset.card_pair (zero_ne_one' F)]
  have hxi : ∀ i : Fin n, ∑ x ∈ cube, x i = 2 ^ (n - 1) := by
    intro i
    have h := Finset.prod_univ_sum (fun _ : Fin n => ({0, 1} : Finset F))
      (fun j y => if j = i then y else (1 : F))
    have hR : ∑ p ∈ cube, ∏ j, (if j = i then p j else (1 : F)) = ∑ p ∈ cube, p i := by
      refine Finset.sum_congr rfl fun p _ => ?_
      simp
    have hL : ∏ j : Fin n, ∑ y ∈ ({0, 1} : Finset F), (if j = i then y else (1 : F))
        = 2 ^ (n - 1) := by
      have : ∀ j : Fin n, ∑ y ∈ ({0, 1} : Finset F), (if j = i then y else (1 : F))
          = if j = i then 1 else 2 := by
        intro j
        by_cases hj : j = i
        · simp [hj]
        · simp [hj, one_add_one_eq_two]
      rw [Finset.prod_congr rfl (fun j _ => this j)]
      simp [Finset.prod_ite, Finset.filter_eq', Finset.filter_ne', Finset.card_erase_of_mem]
    rw [← hR, ← h, hL]
  have hpow : ∀ x ∈ cube, ∀ (i : Fin n) (k : ℕ), x i ^ (k + 1) = x i := by
    intro x hx i k
    have hxi' : x i = 0 ∨ x i = 1 := by
      have := (Fintype.mem_piFinset.mp hx) i
      simpa using this
    rcases hxi' with h | h <;> simp [h]
  rw [Finset.sum_add_distrib]
  simp only [Finset.sum_const, hcard, nsmul_eq_mul]
  have hmid : ∑ x ∈ cube, ∑ i, ∑ k : Fin d, a i k * x i ^ ((k : ℕ) + 1)
      = ∑ i, ∑ k : Fin d, a i k * 2 ^ (n - 1) := by
    calc ∑ x ∈ cube, ∑ i, ∑ k : Fin d, a i k * x i ^ ((k : ℕ) + 1)
        = ∑ x ∈ cube, ∑ i, ∑ k : Fin d, a i k * x i :=
          Finset.sum_congr rfl fun x hx => Finset.sum_congr rfl fun i _ =>
            Finset.sum_congr rfl fun k _ => by rw [hpow x hx i]
      _ = ∑ i, ∑ k : Fin d, a i k * ∑ x ∈ cube, x i := by
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl fun k _ => ?_
          rw [Finset.mul_sum]
      _ = _ := by simp only [hxi]
  have hfin : ∑ i, ∑ k : Fin d, a i k * (2 : F) ^ (n - 1) = 2 ^ (n - 1) * ∑ i, ∑ k, a i k := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [hmid, hfin, libraG]
  push_cast
  ring

def libraPhi (n d : ℕ) (ρ : F) : (F × (Fin n → Fin d → F)) →+ (F × (Fin n → Fin d → F)) where
  toFun m := (libraG m.1 m.2, fun i k => ρ * libraW F n i * m.2 i k)
  map_zero' := by
    refine Prod.ext ?_ ?_
    · simp [libraG]
    · funext i k; simp
  map_add' x y := by
    refine Prod.ext ?_ ?_
    · simp only [libraG, Prod.fst_add, Prod.snd_add, Pi.add_apply, Finset.sum_add_distrib]; ring
    · funext i k; simp only [Prod.snd_add, Pi.add_apply]; ring

theorem libraPhi_apply (ρ : F) (m : F × (Fin n → Fin d → F)) :
    libraPhi n d ρ m = (libraG m.1 m.2, fun i k => ρ * libraW F n i * m.2 i k) := rfl

theorem libraW_ne_zero (h2 : (2 : F) ≠ 0) (i : Fin n) : (libraW F n i) ≠ 0 :=
  pow_ne_zero _ h2

theorem libra_mask_surjective (h2 : (2 : F) ≠ 0) (hρ : ρ ≠ 0) :
    Function.Surjective (fun m : F × (Fin n → Fin d → F) =>
      (libraG m.1 m.2, fun i k => ρ * libraW F n i * m.2 i k)) := by
  rintro ⟨g, e⟩
  have hw : ∀ i : Fin n, ρ * libraW F n i ≠ 0 := fun i => mul_ne_zero hρ (libraW_ne_zero h2 i)
  have hn : (2 : F) ^ n ≠ 0 := pow_ne_zero n h2
  refine ⟨((g - 2 ^ (n - 1) * ∑ i, ∑ k, e i k / (ρ * libraW F n i)) / 2 ^ n,
    fun i k => e i k / (ρ * libraW F n i)), ?_⟩
  refine Prod.ext ?_ ?_
  · simp only [libraG]
    rw [mul_div_cancel₀ _ hn]
    ring
  · funext i k
    simp only
    rw [mul_div_assoc']
    exact mul_div_cancel_left₀ _ (hw i)

theorem libraPhi_injective (h2 : (2 : F) ≠ 0) (hρ : ρ ≠ 0) :
    Function.Injective (libraPhi n d ρ) := by
  rw [injective_iff_map_eq_zero]
  intro m hm
  have hn : (2 : F) ^ n ≠ 0 := pow_ne_zero n h2
  have h1 : libraG m.1 m.2 = 0 := congrArg Prod.fst hm
  have h2' : (fun i k => ρ * libraW F n i * m.2 i k) = (0 : Fin n → Fin d → F) := congrArg Prod.snd hm
  have hz : ∀ i k, m.2 i k = 0 := by
    intro i k
    have := congrFun (congrFun h2' i) k
    exact (mul_eq_zero.mp this).resolve_left (mul_ne_zero hρ (libraW_ne_zero h2 i))
  have h1' : 2 ^ n * m.1 = 0 := by
    have hG : libraG m.1 m.2 = 2 ^ n * m.1 := by simp [libraG, hz]
    rw [← hG]; exact h1
  refine Prod.ext ((mul_eq_zero.mp h1').resolve_left hn) ?_
  funext i k
  exact hz i k

theorem libra_mask_bijective (h2 : (2 : F) ≠ 0) (hρ : ρ ≠ 0) : Function.Bijective (libraPhi n d ρ) :=
  ⟨libraPhi_injective h2 hρ, libra_mask_surjective h2 hρ⟩

theorem libra_view_independent [Fintype F] [DecidableEq F] (h2 : (2 : F) ≠ 0) (hρ : ρ ≠ 0) {W : Type*}
    (S : W → F × (Fin n → Fin d → F)) (w w' : W) (v : F × (Fin n → Fin d → F)) :
    (Finset.univ.filter fun m : F × (Fin n → Fin d → F) => S w + libraPhi n d ρ m = v).card
      = (Finset.univ.filter fun m : F × (Fin n → Fin d → F) => S w' + libraPhi n d ρ m = v).card :=
  masked_view_independent S (libraPhi n d ρ) (libra_mask_surjective h2 hρ) w w' v

theorem libra_transcript_independent [Fintype F] [DecidableEq F] (h2 : (2 : F) ≠ 0) (ρ : F → F)
    (hρ : ∀ g, ρ g ≠ 0) {W : Type*} (A : W → (F × (Fin n → Fin d → F)) → Fin n → Fin d → F)
    (w w' : W) (t : F × (Fin n → Fin d → F)) :
    (Finset.univ.filter fun m : F × (Fin n → Fin d → F) =>
        libraG m.1 m.2 = t.1 ∧ ∀ i k, A w t i k + ρ t.1 * libraW F n i * m.2 i k = t.2 i k).card
      = (Finset.univ.filter fun m : F × (Fin n → Fin d → F) =>
        libraG m.1 m.2 = t.1 ∧ ∀ i k, A w' t i k + ρ t.1 * libraW F n i * m.2 i k = t.2 i k).card := by
  have e : ∀ x : W, (Finset.univ.filter fun m : F × (Fin n → Fin d → F) =>
        libraG m.1 m.2 = t.1 ∧ ∀ i k, A x t i k + ρ t.1 * libraW F n i * m.2 i k = t.2 i k)
      = Finset.univ.filter fun m : F × (Fin n → Fin d → F) =>
        ((0 : F), A x t) + libraPhi n d (ρ t.1) m = t := by
    intro x
    ext m
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, libraPhi_apply, Prod.mk_add_mk,
      zero_add, Prod.ext_iff, funext_iff, Pi.add_apply]
  rw [e w, e w']
  exact libra_view_independent h2 (hρ t.1) (fun x : W => ((0 : F), A x t)) w w' t

theorem libra_const_unique (h2 : (2 : F) ≠ 0) (claim : F) (e : Fin d → F) :
    ∃! c : F, 2 * c + ∑ k, e k = claim :=
  ⟨(claim - ∑ k, e k) / 2, by field_simp; ring,
    fun c hc => (eq_div_iff h2).mpr (by linear_combination hc)⟩

def LibraHyperplane (claims : Fin n → F) : Set (F × (Fin n → F × (Fin d → F))) :=
  {t | ∀ i, 2 * (t.2 i).1 + ∑ k, (t.2 i).2 k = claims i}

noncomputable def libraTuple (n d : ℕ) (ρ : F) (claims : Fin n → F)
    (m : F × (Fin n → Fin d → F)) : F × (Fin n → F × (Fin d → F)) :=
  ((libraPhi n d ρ m).1, fun i => ((claims i - ∑ k, (libraPhi n d ρ m).2 i k) / 2,
    (libraPhi n d ρ m).2 i))

theorem libra_mask_bijOn_hyperplane (h2 : (2 : F) ≠ 0) (hρ : ρ ≠ 0) (claims : Fin n → F) :
    Set.BijOn (libraTuple n d ρ claims) Set.univ (LibraHyperplane claims) := by
  have hΨ : Set.BijOn (fun p : F × (Fin n → Fin d → F) =>
      ((p.1, fun i => ((claims i - ∑ k, p.2 i k) / 2, p.2 i)) : F × (Fin n → F × (Fin d → F))))
      Set.univ (LibraHyperplane claims) := by
    refine ⟨?_, ?_, ?_⟩
    · intro p _ i
      simp only
      field_simp
      ring
    · intro p _ q _ hpq
      obtain ⟨h1, h2'⟩ := Prod.ext_iff.mp hpq
      refine Prod.ext h1 ?_
      funext i k
      have := congrArg (fun f => (f i).2 k) h2'
      simpa using this
    · rintro ⟨g, msgs⟩ hmem
      refine ⟨(g, fun i => (msgs i).2), Set.mem_univ _, ?_⟩
      refine Prod.ext rfl ?_
      funext i
      have hi := hmem i
      have := (libra_const_unique h2 (claims i) (msgs i).2).unique
        (show 2 * (msgs i).1 + ∑ k, (msgs i).2 k = claims i from hi)
        (show 2 * ((claims i - ∑ k, (msgs i).2 k) / 2) + ∑ k, (msgs i).2 k = claims i by
          field_simp; ring)
      exact Prod.ext this.symm rfl
  have hΦ : Set.BijOn (libraPhi n d ρ) Set.univ Set.univ := Set.bijOn_univ.mpr (libra_mask_bijective h2 hρ)
  exact hΨ.comp hΦ

end libra

end FlashProveNext

#print axioms FlashProveNext.fiber_card_mul
#print axioms FlashProveNext.multiround_independent
#print axioms FlashProveNext.evalTranscript_spec
#print axioms FlashProveNext.causal_transcript_unique
#print axioms FlashProveNext.causal_transcript_existsUnique
#print axioms FlashProveNext.evalTranscript_eq_iff
#print axioms FlashProveNext.multiround_independent_causal
#print axioms FlashProveNext.fold_mask_independent
#print axioms FlashProveNext.postprocess_independent
#print axioms FlashProveNext.joint_disjoint_surjective
#print axioms FlashProveNext.joint_surjective_of_det
#print axioms FlashProveNext.joint_tensor_surjective
#print axioms FlashProveNext.libraG_eq_cubeSum
#print axioms FlashProveNext.libra_mask_surjective
#print axioms FlashProveNext.libraPhi_injective
#print axioms FlashProveNext.libra_mask_bijective
#print axioms FlashProveNext.libra_view_independent
#print axioms FlashProveNext.libra_transcript_independent
#print axioms FlashProveNext.libra_const_unique
#print axioms FlashProveNext.libra_mask_bijOn_hyperplane
