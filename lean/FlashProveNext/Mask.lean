/-
Foundational zero-knowledge masking lemmas over finite abelian groups and fields.

* `fiber_card_eq`           : every fiber of a surjective group homomorphism `L : G →+ H` has identical cardinality;
* `masked_view_independent` : `view = S w + L r` is uniformly distributed and independent of `w` when `r` is uniform;
* `vandermonde_surjective`  : evaluation of `ℓ` consecutive blinding coefficients at `m ≤ ℓ` distinct points is surjective;
* `block_surjective`        : block-triangular composition of surjective blinding maps remains surjective.
-/
import Mathlib

namespace FlashProveNext

theorem fiber_card_eq {G H : Type*} [AddCommGroup G] [AddCommGroup H] [Fintype G] [DecidableEq H]
    (L : G →+ H) (hs : Function.Surjective L) (y y' : H) :
    (Finset.univ.filter fun r => L r = y).card = (Finset.univ.filter fun r => L r = y').card := by
  obtain ⟨r0, h0⟩ := hs (y' - y)
  apply Finset.card_bij (fun r _ => r + r0)
  · intro r hr
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hr ⊢
    rw [map_add, hr, h0]; abel
  · intro a _ b _ h
    exact add_right_cancel h
  · intro b hb
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hb
    refine ⟨b - r0, ?_, by abel⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    rw [map_sub, hb, h0]; abel

theorem masked_view_independent {W G H : Type*} [AddCommGroup G] [AddCommGroup H] [Fintype G]
    [DecidableEq H] (S : W → H) (L : G →+ H) (hs : Function.Surjective L) (w w' : W) (v : H) :
    (Finset.univ.filter fun r => S w + L r = v).card
      = (Finset.univ.filter fun r => S w' + L r = v).card := by
  have e : ∀ x : W, (Finset.univ.filter fun r => S x + L r = v)
      = (Finset.univ.filter fun r => L r = v - S x) := by
    intro x
    ext r
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    constructor
    · intro h; rw [← h]; abel
    · intro h; rw [h]; abel
  rw [e w, e w']
  exact fiber_card_eq L hs _ _

theorem vandermonde_surjective {F : Type*} [Field F] {m ℓ : ℕ} (x c : Fin m → F)
    (hx : Function.Injective x) (hc : ∀ i, c i ≠ 0) (h : m ≤ ℓ) :
    Function.Surjective
      (fun (r : Fin ℓ → F) (i : Fin m) => c i * ∑ j : Fin ℓ, r j * x i ^ (j : ℕ)) := by
  classical
  intro y
  set p := Lagrange.interpolate Finset.univ x (fun i => y i / c i) with hp
  have hdeg : p.degree < (m : WithBot ℕ) := by
    have := Lagrange.degree_interpolate_lt (s := Finset.univ) (v := x)
      (r := fun i => y i / c i) hx.injOn
    simpa using this
  refine ⟨fun j => p.coeff j, ?_⟩
  funext i
  have hm : 0 < m := i.pos
  have hlt : p.natDegree < ℓ := by
    by_cases hp0 : p = 0
    · rw [hp0]; simpa using lt_of_lt_of_le hm h
    · exact lt_of_lt_of_le ((Polynomial.natDegree_lt_iff_degree_lt hp0).mpr hdeg) h
  have hev : p.eval (x i) = y i / c i :=
    Lagrange.eval_interpolate_at_node (fun i => y i / c i) hx.injOn (Finset.mem_univ i)
  show c i * ∑ j : Fin ℓ, p.coeff j * x i ^ (j : ℕ) = y i
  rw [Fin.sum_univ_eq_sum_range (fun j => p.coeff j * x i ^ j) ℓ,
    ← Polynomial.eval_eq_sum_range' hlt (x i), hev]
  field_simp [hc i]

theorem block_surjective {G1 G2 H1 H2 : Type*} [AddCommGroup G1] [AddCommGroup G2]
    [AddCommGroup H1] [AddCommGroup H2] (L1 : G1 →+ H1) (L2 : G2 →+ H2) (X : G1 →+ H2)
    (h1 : Function.Surjective L1) (h2 : Function.Surjective L2) :
    Function.Surjective (fun p : G1 × G2 => (L1 p.1, X p.1 + L2 p.2)) := by
  rintro ⟨y1, y2⟩
  obtain ⟨a, ha⟩ := h1 y1
  obtain ⟨b, hb⟩ := h2 (y2 - X a)
  refine ⟨(a, b), ?_⟩
  simp only [ha, hb]
  congr 1
  abel

end FlashProveNext

#print axioms FlashProveNext.fiber_card_eq
#print axioms FlashProveNext.masked_view_independent
#print axioms FlashProveNext.vandermonde_surjective
#print axioms FlashProveNext.block_surjective
