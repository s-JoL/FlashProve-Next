/-
Formal bridge from protocol specification and compiled constraint-system semantics to
round-by-round lifetime knowledge soundness (`specs/01-relation.md` §4.5, `specs/02-soundness.md` §2.2, §6–§7):

1. `goldilocks_p4_bounds`, `delta_SL_p4_default`:
   exact `p^4`-denominator straight-line extraction failure bound `((α + 2) Q_tot^2) / p^4 ≤ 2^{-86}`
   over invertible Goldilocks width-12 Poseidon2 permutation queries (capacity = 4 elements);
2. `sumcheck_round_esc_le`, `hco_col_esc_le`, `setup_col_esc_le`, `tc_col_esc_le`, `basefold_qry_esc_le`:
   concrete per-round escape bounds (`≤ 2^{-106}`) for Sumcheck over `F_{p^3}`, HCO 123-column opening,
   `π_setup` 206-column joint check, trace-commitment 268-column check, and Basefold 168-query check;
3. `AtomBad`, `atomBad_of_mem_lineage`, `compiled_cs_forces_all_good`, `compiled_cs_doomed_of_bad_sample`:
   lineage atom locality (`P-loc`) and compiled constraint-system bridge (`P-compile`) forcing `¬ Satisfies(CS, w)`
   whenever an extracted bad trace fails to escape at sampling;
4. `first_escape_of_doomed_to_clear`, `hpath_of_protocol_spec`, `fpn_protocol_lifetime_soundness`:
   derives `RoundByRound`'s `hpath` premise from `ProtocolRBRSpec` (`P-ext`, `P-compile`, `P-loc`, `P-final`)
   and establishes `Pr[BadAccept] ≤ 2^{-40}` across `Q ≤ 2^64` adaptive lifetime queries.
-/
import Mathlib
import FlashProveNext.RoundByRound
import FlashProveNext.SumcheckSound

namespace FlashProveNext

open Finset

set_option linter.unusedSectionVars false
set_option exponentiation.threshold 2048

theorem goldilocks_p4_bounds : 2 ^ 255 < goldilocks ^ 4 ∧ goldilocks ^ 4 < 2 ^ 256 := by
  rw [goldilocks_eq]
  norm_num

theorem delta_SL_p4_default (α Qt : ℕ) (hα : α < 2 ^ 9) (hQ : Qt ≤ 2 ^ 80) :
    ((α + 2 : ℕ) : ℝ) * (Qt : ℝ) ^ 2 / (goldilocks ^ 4 : ℕ) ≤ (2 ^ 86)⁻¹ ∧
    (α ≤ 510 → ((α + 2 : ℕ) : ℝ) * (Qt : ℝ) ^ 2 / (goldilocks ^ 4 : ℕ) ≤ (101 / 100) * (2 ^ 87)⁻¹) := by
  have hp4 : 0 < goldilocks ^ 4 := by rw [goldilocks_eq]; norm_num
  constructor
  · have hn : (α + 2) * Qt ^ 2 ≤ (2 ^ 9 + 1) * (2 ^ 80) ^ 2 :=
      Nat.mul_le_mul (by omega) (Nat.pow_le_pow_left hQ 2)
    have h := (frac_le_inv_pow2_iff ((α + 2) * Qt ^ 2) (goldilocks ^ 4) 86 hp4).mpr
      (by calc (α + 2) * Qt ^ 2 * 2 ^ 86
            ≤ (2 ^ 9 + 1) * (2 ^ 80) ^ 2 * 2 ^ 86 := Nat.mul_le_mul_right _ hn
          _ ≤ goldilocks ^ 4 := by rw [goldilocks_eq]; norm_num)
    simp only [Nat.cast_mul, Nat.cast_add, Nat.cast_pow, Nat.cast_ofNat] at h ⊢
    exact h
  · intro hα510
    have hp4r : (0 : ℝ) < (goldilocks ^ 4 : ℕ) := Nat.cast_pos.mpr hp4
    rw [div_le_iff₀ hp4r]
    have hn : ((α + 2 : ℕ) : ℝ) * (Qt : ℝ) ^ 2 ≤ 512 * ((2 : ℝ) ^ 80) ^ 2 := by
      have h1 : ((α + 2 : ℕ) : ℝ) ≤ 512 := by exact_mod_cast (show α + 2 ≤ 512 by omega)
      have h2 : (Qt : ℝ) ^ 2 ≤ ((2 : ℝ) ^ 80) ^ 2 := by
        exact pow_le_pow_left₀ (Nat.cast_nonneg _) (by exact_mod_cast hQ) 2
      exact mul_le_mul h1 h2 (by positivity) (by positivity)
    calc ((α + 2 : ℕ) : ℝ) * (Qt : ℝ) ^ 2
        ≤ 512 * ((2 : ℝ) ^ 80) ^ 2 := hn
      _ ≤ (101 / 100) * (2 ^ 87)⁻¹ * (goldilocks ^ 4 : ℕ) := by
          rw [goldilocks_eq]; push_cast; norm_num

theorem sumcheck_round_esc_le {E : Type*} [Field E] [Fintype E] [DecidableEq E]
    {n d : ℕ} (P : (Fin n → E) → E) (hd : RoundDeg d P)
    (g : Fin n → (Fin n → E) → Polynomial E) (hg : ∀ i r, (g i r).natDegree ≤ d)
    (i : Fin n) (r : Fin n → E) :
    ((badSet P g i r).card : ℝ) ≤ ((d : ℝ) / Fintype.card E) * Fintype.card E ∧
    (Fintype.card E = goldilocks ^ 3 → d ≤ 2 ^ 85 →
      ((badSet P g i r).card : ℝ) ≤ (2 ^ 106)⁻¹ * Fintype.card E) := by
  have hcard : (badSet P g i r).card ≤ d := by
    unfold badSet
    split_ifs with h
    · simp
    · obtain ⟨x0, hx0⟩ := not_forall.mp h
      obtain ⟨p, hpdeg, hpeval⟩ := hd i r
      have hne : g i r ≠ p := fun heq => hx0 (by rw [heq, hpeval])
      have hsub : (univ.filter fun x => (g i r).eval x = psum P (i + 1) (Function.update r i x))
          ⊆ univ.filter (fun x => (g i r).eval x = p.eval x) := by
        intro x hx
        simp only [mem_filter, mem_univ, true_and] at hx ⊢
        rw [hx, hpeval]
      exact (card_le_card hsub).trans (agree_card_le (g i r) p hne d (hg i r) hpdeg)
  have hpos : (0 : ℝ) < Fintype.card E := Nat.cast_pos.mpr Fintype.card_pos
  constructor
  · rw [div_mul_cancel₀ _ (ne_of_gt hpos)]
    exact_mod_cast hcard
  · intro hE hd85
    have hfrac : (d : ℝ) / (goldilocks ^ 3 : ℕ) ≤ (2 ^ 106)⁻¹ := field_p3_bits d hd85
    have h1 : (d : ℝ) ≤ (2 ^ 106)⁻¹ * Fintype.card E := by
      rw [hE]
      exact (div_le_iff₀ (by rw [← hE]; exact hpos)).mp hfrac
    exact le_trans (by exact_mod_cast hcard) h1

theorem hco_col_esc_le : ((55 : ℝ) / 100) ^ 123 ≤ (2 ^ 106)⁻¹ := by
  norm_num

theorem setup_col_esc_le : ((7 : ℝ) / 10) ^ 206 ≤ (2 ^ 106)⁻¹ := by
  norm_num

theorem tc_col_esc_le :
    ((74 : ℝ) / 100) ^ 268 ≤ (2 ^ 106)⁻¹ ∧ ((76 : ℝ) / 100) ^ 268 ≤ (2 ^ 106)⁻¹ := by
  norm_num

theorem basefold_qry_esc_le : ((645 : ℝ) / 1000) ^ 168 ≤ (2 ^ 106)⁻¹ := by
  norm_num

def AtomBad {Atom UnitU : Type*} (Lin : Atom → Finset UnitU) (unitBad : UnitU → Prop) (a : Atom) : Prop :=
  ∃ u ∈ Lin a, unitBad u

theorem atomBad_of_mem_lineage {Atom UnitU : Type*} (Lin : Atom → Finset UnitU)
    (unitBad : UnitU → Prop) {a : Atom} {u : UnitU} (hu : u ∈ Lin a) (hbad : unitBad u) :
    AtomBad Lin unitBad a :=
  ⟨u, hu, hbad⟩

def ExtractedBad {ι : Type*} (GlobalOK ChainOK : Prop)
    (C badCount : ι → ℕ) (σ : ι → ℝ) : Prop :=
  ¬ GlobalOK ∨ ¬ ChainOK ∨ ∃ i : ι, 0 < C i ∧ σ i * (C i : ℝ) ≤ (badCount i : ℝ)

structure CompiledCS (WCol : Type*) where
  Satisfies : WCol → Prop

theorem compiled_cs_forces_all_good {ι WCol : Type*}
    (CS : CompiledCS WCol) (w : WCol)
    (GlobalOK ChainOK : Prop)
    (C : ι → ℕ) (σ : ι → ℝ) (k : ι → ℕ)
    (PoolBad : ∀ i, Fin (C i) → Prop) [∀ i, DecidablePred (PoolBad i)]
    (sampled : ∀ i, Fin (k i) → Fin (C i))
    (SlotChecked : ∀ i, Fin (k i) → Prop)
    (hCompile : CS.Satisfies w → GlobalOK ∧ ChainOK ∧ ∀ i s, SlotChecked i s)
    (hLoc : ∀ i s, SlotChecked i s → ¬ PoolBad i (sampled i s))
    (hBad : ExtractedBad GlobalOK ChainOK C
      (fun i => (univ.filter (fun x : Fin (C i) => PoolBad i x)).card) σ)
    (hSat : CS.Satisfies w) :
    GlobalOK ∧ ChainOK ∧
      ∃ i : ι, 0 < C i ∧
        σ i * (C i : ℝ) ≤ ((univ.filter (fun x : Fin (C i) => PoolBad i x)).card : ℝ) ∧
        ∀ s : Fin (k i), ¬ PoolBad i (sampled i s) := by
  obtain ⟨hg, hc, hslots⟩ := hCompile hSat
  rcases hBad with hng | hnc | ⟨i, hCi, hfrac⟩
  · exact absurd hg hng
  · exact absurd hc hnc
  · refine ⟨hg, hc, i, hCi, hfrac, fun s => hLoc i s (hslots i s)⟩

theorem compiled_cs_doomed_of_bad_sample {ι WCol : Type*}
    (CS : CompiledCS WCol) (w : WCol)
    (GlobalOK ChainOK : Prop)
    (C : ι → ℕ) (σ : ι → ℝ) (k : ι → ℕ)
    (PoolBad : ∀ i, Fin (C i) → Prop) [∀ i, DecidablePred (PoolBad i)]
    (sampled : ∀ i, Fin (k i) → Fin (C i))
    (SlotChecked : ∀ i, Fin (k i) → Prop)
    (hCompile : CS.Satisfies w → GlobalOK ∧ ChainOK ∧ ∀ i s, SlotChecked i s)
    (hLoc : ∀ i s, SlotChecked i s → ¬ PoolBad i (sampled i s))
    (hBad : ExtractedBad GlobalOK ChainOK C
      (fun i => (univ.filter (fun x : Fin (C i) => PoolBad i x)).card) σ)
    (hNotEsc : ∀ i : ι, 0 < C i →
      σ i * (C i : ℝ) ≤ ((univ.filter (fun x : Fin (C i) => PoolBad i x)).card : ℝ) →
      ∃ s : Fin (k i), PoolBad i (sampled i s)) :
    ¬ CS.Satisfies w := by
  intro hSat
  obtain ⟨_, _, i, hCi, hfrac, hall⟩ :=
    compiled_cs_forces_all_good CS w GlobalOK ChainOK C σ k PoolBad sampled SlotChecked
      hCompile hLoc hBad hSat
  obtain ⟨s, hs⟩ := hNotEsc i hCi hfrac
  exact hall s hs

theorem first_escape_of_doomed_to_clear :
    ∀ {R : ℕ} (H : Fin (R + 1) → Prop),
      ¬ H 0 → H (Fin.last R) → ∃ r : Fin R, ¬ H r.castSucc ∧ H r.succ
  | 0, H, h0, hR => absurd hR h0
  | R + 1, H, h0, hR => by
    by_cases h1 : H ⟨1, by omega⟩
    · exact ⟨⟨0, by omega⟩, h0, h1⟩
    · obtain ⟨r, hr0, hr1⟩ := first_escape_of_doomed_to_clear (fun k => H k.succ) h1 hR
      exact ⟨r.succ, hr0, hr1⟩

structure ProtocolRBRSpec {Ω S : Type*} {Q : ℕ}
    (st : Fin Q → (Fin Q → Ω) → S)
    (Doomed : S → Prop) (Esc : S → Finset Ω)
    (BadAcc Fail : Finset (Fin Q → Ω)) where
  R : (Fin Q → Ω) → ℕ
  roundHolds : ∀ a : Fin Q → Ω, Fin (R a + 1) → Prop
  qIdx : ∀ a : Fin Q → Ω, Fin (R a) → Fin Q
  hInit : ∀ a ∈ BadAcc, a ∉ Fail → ¬ roundHolds a 0
  hFinal : ∀ a ∈ BadAcc, a ∉ Fail → roundHolds a (Fin.last (R a))
  hStep : ∀ a ∈ BadAcc, a ∉ Fail → ∀ r : Fin (R a),
    ¬ roundHolds a r.castSucc → roundHolds a r.succ →
    Doomed (st (qIdx a r) a) ∧ a (qIdx a r) ∈ Esc (st (qIdx a r) a)

theorem hpath_of_protocol_spec {Ω S : Type*} [DecidableEq Ω] {Q : ℕ}
    {st : Fin Q → (Fin Q → Ω) → S}
    {Doomed : S → Prop} {Esc : S → Finset Ω}
    {BadAcc Fail : Finset (Fin Q → Ω)}
    (spec : ProtocolRBRSpec st Doomed Esc BadAcc Fail) :
    ∀ a ∈ BadAcc, a ∈ Fail ∨ ∃ j : Fin Q, Doomed (st j a) ∧ a j ∈ Esc (st j a) := by
  classical
  intro a ha
  by_cases hf : a ∈ Fail
  · exact Or.inl hf
  · right
    obtain ⟨r, hr0, hr1⟩ :=
      first_escape_of_doomed_to_clear (spec.roundHolds a) (spec.hInit a ha hf) (spec.hFinal a ha hf)
    exact ⟨spec.qIdx a r, spec.hStep a ha hf r hr0 hr1⟩

/-- End-to-end protocol lifetime knowledge soundness theorem (`Pr[BadAccept] ≤ 2^{-40}` at `Q ≤ 2^64`). -/
theorem fpn_protocol_lifetime_soundness {Ω S ι : Type*} [Fintype Ω] [DecidableEq Ω] [Nonempty Ω]
    {Q : ℕ} (hQ : Q ≤ 2 ^ 64)
    (st : Fin Q → (Fin Q → Ω) → S)
    (hst : ∀ j a x, st j (Function.update a j x) = st j a)
    (Doomed : S → Prop) (Esc : S → Finset Ω) (round : S → ι) (ε : ι → ℝ)
    (m amap εalg Δ : ℝ) (hm0 : 0 ≤ m) (hm : m ≤ (1 / 2) ^ 105) (ha0 : 0 ≤ amap)
    (ha : amap ≤ (2 ^ 106)⁻¹) (he : εalg ≤ (2 ^ 106)⁻¹) (hΔ : Δ ≤ (2 ^ 42)⁻¹)
    (hεr : ∀ r, ε r ≤ max (m + amap) εalg)
    (hEsc : ∀ s, Doomed s → ((Esc s).card : ℝ) ≤ ε (round s) * Fintype.card Ω)
    (BadAcc Fail : Finset (Fin Q → Ω))
    (hFail : (Fail.card : ℝ) ≤ Δ * (Fintype.card Ω : ℝ) ^ Q)
    (spec : ProtocolRBRSpec st Doomed Esc BadAcc Fail) :
    (BadAcc.card : ℝ) ≤ (2 ^ 40)⁻¹ * (Fintype.card Ω : ℝ) ^ Q :=
  rbr_default_lifetime hQ st hst Doomed Esc round ε m amap εalg Δ
    hm0 hm ha0 ha he hΔ hεr hEsc BadAcc Fail hFail (hpath_of_protocol_spec spec)

end FlashProveNext

#print axioms FlashProveNext.goldilocks_p4_bounds
#print axioms FlashProveNext.delta_SL_p4_default
#print axioms FlashProveNext.sumcheck_round_esc_le
#print axioms FlashProveNext.hco_col_esc_le
#print axioms FlashProveNext.setup_col_esc_le
#print axioms FlashProveNext.tc_col_esc_le
#print axioms FlashProveNext.basefold_qry_esc_le
#print axioms FlashProveNext.atomBad_of_mem_lineage
#print axioms FlashProveNext.compiled_cs_forces_all_good
#print axioms FlashProveNext.compiled_cs_doomed_of_bad_sample
#print axioms FlashProveNext.first_escape_of_doomed_to_clear
#print axioms FlashProveNext.hpath_of_protocol_spec
#print axioms FlashProveNext.fpn_protocol_lifetime_soundness
