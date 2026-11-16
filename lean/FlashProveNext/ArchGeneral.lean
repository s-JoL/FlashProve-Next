/-
Exact algebraic equivalences for modern LLM architectures and distributed execution (`specs/05-computation-model.md`):

1. `sbfp_shift_unique`      : canonical signed-exponent block floating-point (`e = eb - E_BIAS`) shift uniqueness;
2. `mla_score_absorbed_eq`,
   `mla_out_absorbed_eq`    : Multi-head Latent Attention (MLA) matrix absorption equals full `(K, V)` expansion over any commutative ring;
3. `pp_rangeLayers_add`,
   `ep_allreduce_eq_moe`    : pipeline-parallel stage composition and expert-parallel disjoint sharding + `all_reduce(SUM)` invariance.
-/
import Mathlib

open scoped BigOperators

namespace FlashProveNext

section SignedBFP

def ValidBFP {n : ℕ} (a : Fin n → ℤ) (M eb : ℕ) : Prop :=
  ∀ i, -((2 : ℤ) ^ (M + eb)) ≤ a i ∧ a i < (2 : ℤ) ^ (M + eb)

def CanonicalBFP {n : ℕ} (a : Fin n → ℤ) (M eb : ℕ) : Prop :=
  ValidBFP a M eb ∧ (eb = 0 ∨ ¬ ValidBFP a M (eb - 1))

theorem validBFP_mono {n : ℕ} (a : Fin n → ℤ) (M : ℕ) {e₁ e₂ : ℕ} (hle : e₁ ≤ e₂)
    (h : ValidBFP a M e₁) : ValidBFP a M e₂ := by
  intro i
  obtain ⟨hlo, hhi⟩ := h i
  have hpow : (2 : ℤ) ^ (M + e₁) ≤ (2 : ℤ) ^ (M + e₂) := by
    exact pow_le_pow_right₀ (by norm_num) (Nat.add_le_add_left hle M)
  exact ⟨by linarith, by linarith⟩

theorem sbfp_shift_unique {n : ℕ} (a : Fin n → ℤ) (M eb₁ eb₂ : ℕ)
    (h₁ : CanonicalBFP a M eb₁) (h₂ : CanonicalBFP a M eb₂) :
    eb₁ = eb₂ := by
  rcases Nat.lt_trichotomy eb₁ eb₂ with hlt | rfl | hgt
  · have hle : eb₁ ≤ eb₂ - 1 := Nat.le_pred_of_lt hlt
    have hv : ValidBFP a M (eb₂ - 1) := validBFP_mono a M hle h₁.1
    rcases h₂.2 with hz | hnv
    · omega
    · exact False.elim (hnv hv)
  · rfl
  · have hle : eb₂ ≤ eb₁ - 1 := Nat.le_pred_of_lt hgt
    have hv : ValidBFP a M (eb₁ - 1) := validBFP_mono a M hle h₂.1
    rcases h₁.2 with hz | hnv
    · omega
    · exact False.elim (hnv hv)

end SignedBFP

section MLA

variable {R : Type*} [CommRing R]
variable {H D C DR T Out : Type*} [Fintype D] [Fintype C] [Fintype DR] [Fintype T] [Fintype H]

def mlaScoreExpanded (W_UK : H → D → C → R) (qC : H → D → R) (qR : H → DR → R)
    (cKV : T → C → R) (kR : T → DR → R) (h : H) (τ : T) : R :=
  (∑ d, qC h d * (∑ c, W_UK h d c * cKV τ c)) + (∑ r, qR h r * kR τ r)

def mlaScoreAbsorbed (W_UK : H → D → C → R) (qC : H → D → R) (qR : H → DR → R)
    (cKV : T → C → R) (kR : T → DR → R) (h : H) (τ : T) : R :=
  (∑ c, (∑ d, qC h d * W_UK h d c) * cKV τ c) + (∑ r, qR h r * kR τ r)

omit [Fintype T] [Fintype H] in
theorem mla_score_absorbed_eq (W_UK : H → D → C → R) (qC : H → D → R) (qR : H → DR → R)
    (cKV : T → C → R) (kR : T → DR → R) (h : H) (τ : T) :
    mlaScoreAbsorbed W_UK qC qR cKV kR h τ = mlaScoreExpanded W_UK qC qR cKV kR h τ := by
  unfold mlaScoreAbsorbed mlaScoreExpanded
  congr 1
  simp_rw [Finset.sum_mul, Finset.mul_sum, mul_assoc]
  exact Finset.sum_comm

def mlaOutExpanded (W_UV : H → D → C → R) (W_O : Out → H → D → R)
    (prob : H → T → R) (cKV : T → C → R) (o : Out) : R :=
  ∑ h, ∑ d, W_O o h d * (∑ τ, prob h τ * (∑ c, W_UV h d c * cKV τ c))

def mlaOutAbsorbed (W_UV : H → D → C → R) (W_O : Out → H → D → R)
    (prob : H → T → R) (cKV : T → C → R) (o : Out) : R :=
  ∑ h, ∑ c, (∑ d, W_O o h d * W_UV h d c) * (∑ τ, prob h τ * cKV τ c)

theorem mla_out_absorbed_eq (W_UV : H → D → C → R) (W_O : Out → H → D → R)
    (prob : H → T → R) (cKV : T → C → R) (o : Out) :
    mlaOutAbsorbed W_UV W_O prob cKV o = mlaOutExpanded W_UV W_O prob cKV o := by
  unfold mlaOutAbsorbed mlaOutExpanded
  apply Finset.sum_congr rfl
  intro h _
  have h1 : ∀ c : C, (∑ d, W_O o h d * W_UV h d c) * (∑ τ, prob h τ * cKV τ c) =
      ∑ d, ∑ τ, W_O o h d * (prob h τ * (W_UV h d c * cKV τ c)) := by
    intro c
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro d _
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro τ _
    ring
  simp_rw [h1]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro d _
  rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro τ _
  rw [Finset.mul_sum, Finset.mul_sum]

end MLA

section DistributedPPEP

def rangeLayers {A : Type*} (layer : ℕ → A → A) : ℕ → ℕ → A → A
  | _, 0, x => x
  | lo, k + 1, x => rangeLayers layer (lo + 1) k (layer lo x)

theorem pp_rangeLayers_add {A : Type*} (layer : ℕ → A → A) (lo m n : ℕ) (x : A) :
    rangeLayers layer lo (m + n) x = rangeLayers layer (lo + m) n (rangeLayers layer lo m x) := by
  induction m generalizing lo x with
  | zero => simp [rangeLayers]
  | succ k ih =>
    simp only [Nat.succ_add, rangeLayers]
    rw [ih (lo + 1) (layer lo x)]
    have hidx : lo + 1 + k = lo + (k + 1) := by omega
    rw [hidx]

theorem ep_allreduce_eq_moe {G E M : Type*} [Fintype G] [Fintype E] [DecidableEq E]
    [AddCommMonoid M]
    (shard : G → Finset E)
    (hdisj : Pairwise fun g₁ g₂ => Disjoint (shard g₁) (shard g₂))
    (hcov : (Finset.univ : Finset G).biUnion shard = Finset.univ)
    (S : Finset E) (term : E → M) :
    (∑ g : G, ∑ e ∈ S ∩ shard g, term e) = ∑ e ∈ S, term e := by
  have hdisjS : (↑(Finset.univ : Finset G) : Set G).PairwiseDisjoint (fun g => S ∩ shard g) := by
    intro g₁ _ g₂ _ hne
    exact Disjoint.mono Finset.inter_subset_right Finset.inter_subset_right (hdisj hne)
  rw [← Finset.sum_biUnion hdisjS]
  congr 1
  ext e
  simp only [Finset.mem_biUnion, Finset.mem_univ, true_and, Finset.mem_inter]
  constructor
  · rintro ⟨_, heS, _⟩; exact heS
  · intro heS
    have heU : e ∈ (Finset.univ : Finset G).biUnion shard := by
      rw [hcov]; exact Finset.mem_univ e
    simp only [Finset.mem_biUnion, Finset.mem_univ, true_and] at heU
    obtain ⟨g, heg⟩ := heU
    exact ⟨g, heS, heg⟩

end DistributedPPEP

end FlashProveNext

#print axioms FlashProveNext.validBFP_mono
#print axioms FlashProveNext.sbfp_shift_unique
#print axioms FlashProveNext.mla_score_absorbed_eq
#print axioms FlashProveNext.mla_out_absorbed_eq
#print axioms FlashProveNext.pp_rangeLayers_add
#print axioms FlashProveNext.ep_allreduce_eq_moe
