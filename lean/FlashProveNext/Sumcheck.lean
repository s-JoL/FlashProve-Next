/-
Univariate low-degree agreement bound and conditional multi-round bad-challenge union bound.

* `agree_card_le`           : two distinct polynomials of degree `≤ d` over a field `F` agree on at most `d` points;
* `hit_card_le`             : if the round-`i` bad challenge set `Bad r` is independent of `r i` and has size `≤ d`,
                              at most a `d / |F|` fraction of challenge vectors `r : Fin n → F` hit `Bad r` at coordinate `i`;
* `sumcheck_bad_challenges` : across `n` rounds, at most an `n · d / |F|` fraction of challenge vectors hit any bad set.
-/
import Mathlib

namespace FlashProveNext

open Finset

theorem agree_card_le {F : Type*} [Field F] [Fintype F] [DecidableEq F] (p q : Polynomial F)
    (hne : p ≠ q) (d : ℕ) (hp : p.natDegree ≤ d) (hq : q.natDegree ≤ d) :
    (Finset.univ.filter fun r => p.eval r = q.eval r).card ≤ d := by
  have hsub : p - q ≠ 0 := sub_ne_zero.mpr hne
  have hdeg : (p - q).natDegree ≤ d := (Polynomial.natDegree_sub_le p q).trans (max_le hp hq)
  calc (Finset.univ.filter fun r => p.eval r = q.eval r).card
      ≤ (p - q).roots.toFinset.card := by
        apply Finset.card_le_card
        intro r hr
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hr
        rw [Multiset.mem_toFinset, Polynomial.mem_roots hsub, Polynomial.IsRoot, Polynomial.eval_sub,
          hr, sub_self]
    _ ≤ Multiset.card (p - q).roots := Multiset.toFinset_card_le _
    _ ≤ (p - q).natDegree := Polynomial.card_roots' _
    _ ≤ d := hdeg

theorem hit_card_le {F : Type*} [Fintype F] [DecidableEq F] {n : ℕ} (i : Fin n)
    (Bad : (Fin n → F) → Finset F) (d : ℕ)
    (hind : ∀ r x, Bad (Function.update r i x) = Bad r) (hd : ∀ r, (Bad r).card ≤ d) :
    (Finset.univ.filter fun r : Fin n → F => r i ∈ Bad r).card * Fintype.card F
      ≤ d * Fintype.card F ^ n := by
  classical
  have hinj : ((Finset.univ.filter fun r : Fin n → F => r i ∈ Bad r) ×ˢ (Finset.univ : Finset F)).card
      ≤ ((Finset.univ : Finset (Fin n → F)).sigma fun s => Bad s).card := by
    apply Finset.card_le_card_of_injOn (fun p => (⟨Function.update p.1 i p.2, p.1 i⟩ : Σ _ : Fin n → F, F))
    · intro p hp
      simp only [Finset.coe_product, Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_prod,
        Set.mem_setOf_eq, Finset.coe_univ, Set.mem_univ, and_true] at hp
      simp only [Finset.coe_sigma, Finset.coe_univ, Set.mem_sigma_iff, Set.mem_univ, true_and,
        Finset.mem_coe]
      rw [hind]; exact hp
    · intro p _ p' _ h
      simp only [Sigma.mk.injEq, heq_eq_eq] at h
      obtain ⟨h1, h2⟩ := h
      have hx : p.2 = p'.2 := by
        have := congrFun h1 i
        simpa using this
      have hr : p.1 = p'.1 := by
        funext j
        by_cases hj : j = i
        · subst hj; exact h2
        · have := congrFun h1 j
          simpa [Function.update_of_ne hj] using this
      exact Prod.ext hr hx
  rw [Finset.card_product, Finset.card_univ] at hinj
  calc (Finset.univ.filter fun r : Fin n → F => r i ∈ Bad r).card * Fintype.card F
      ≤ ((Finset.univ : Finset (Fin n → F)).sigma fun s => Bad s).card := hinj
    _ = ∑ s : Fin n → F, (Bad s).card := Finset.card_sigma _ _
    _ ≤ ∑ _s : Fin n → F, d := Finset.sum_le_sum fun s _ => hd s
    _ = d * Fintype.card F ^ n := by simp [Finset.card_univ, mul_comm]

theorem sumcheck_bad_challenges {F : Type*} [Fintype F] [DecidableEq F] {n : ℕ}
    (Bad : Fin n → (Fin n → F) → Finset F) (d : ℕ)
    (hind : ∀ i r x, Bad i (Function.update r i x) = Bad i r) (hd : ∀ i r, (Bad i r).card ≤ d) :
    (Finset.univ.filter fun r : Fin n → F => ∃ i, r i ∈ Bad i r).card * Fintype.card F
      ≤ n * d * Fintype.card F ^ n := by
  classical
  have hsub : (Finset.univ.filter fun r : Fin n → F => ∃ i, r i ∈ Bad i r)
      ⊆ Finset.univ.biUnion (fun i => Finset.univ.filter fun r : Fin n → F => r i ∈ Bad i r) := by
    intro r hr
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hr
    obtain ⟨i, hi⟩ := hr
    simp only [Finset.mem_biUnion, Finset.mem_univ, true_and, Finset.mem_filter]
    exact ⟨i, hi⟩
  calc (Finset.univ.filter fun r : Fin n → F => ∃ i, r i ∈ Bad i r).card * Fintype.card F
      ≤ (∑ i : Fin n, (Finset.univ.filter fun r : Fin n → F => r i ∈ Bad i r).card)
          * Fintype.card F :=
        Nat.mul_le_mul_right _ ((Finset.card_le_card hsub).trans Finset.card_biUnion_le)
    _ = ∑ i : Fin n, (Finset.univ.filter fun r : Fin n → F => r i ∈ Bad i r).card
          * Fintype.card F := Finset.sum_mul ..
    _ ≤ ∑ _i : Fin n, d * Fintype.card F ^ n :=
        Finset.sum_le_sum fun i _ => hit_card_le i (Bad i) d (hind i) (hd i)
    _ = n * d * Fintype.card F ^ n := by simp [Finset.card_univ, mul_assoc]

end FlashProveNext

#print axioms FlashProveNext.agree_card_le
#print axioms FlashProveNext.hit_card_le
#print axioms FlashProveNext.sumcheck_bad_challenges
