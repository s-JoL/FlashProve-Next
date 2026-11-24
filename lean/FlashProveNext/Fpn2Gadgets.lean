/-
M4-2: translation validation of the core IR operators (`constraint satisfied → integer relation holds`).

Reference: `prover/fpn_prover/ir/gadget_ir.py` (`Program.check_bounds`, `Program.nonneg`, `topk_tau`,
`attaining_rmx`), `prover/fpn_prover/ir/arith.py` (`BF16`, `install_bf16`, `Rsqrt`, `install_rsqrt`),
`prover/fpn_prover/circuit/lower.py` (`LimbPlan`, `Gate`, `_gate_rows`, `AIR_LIMB_BITS = 7`,
`AIR_GATE_FAMILIES`, `AIR_PREDICATE_RESIDUE`), `prover/fpn_prover/circuit/microprogram.py` (`_lift`) and
`prover/fpn_prover/circuit/blocks.py` (`rmsnorm_block`, `RESIDUAL_RMS_FLOOR = 1/1000`).

The field is `ZMod goldilocks`, `goldilocks = 2^64 - 2^32 + 1` (`RoundByRound.goldilocks`, prime by
`Fpn2Soundness.goldilocks_prime`). Six operator families:

1. Signed lift. `signed_lift_zero_iff`, `signed_lift_eq_of_sub_bound`, `signed_lift_mul_add_sound`
   (one Plonkish row `q_M a b + q_L a + q_R b + q_O c + q_C`), `gate_lift_sound` (any `Gate.from_poly`
   monomial list, any degree), `air_row_chain_sound` (the accumulator rows of `_gate_rows` are one
   equation), `canonical_coeff_cast` (`c % P` is the same field element as `c`),
   `signed_lift_bound_necessary` (the bound cannot be dropped: `p ↦ 0`).
2. 7-bit limbs. `limb7_recompose_bounds`, `limb7_tight_lt`, `tighten_gate_sound`,
   `signed_limb7_range_sound` (integer side), `limb7_field_range_sound` (field side: the recompose and
   tighten gates of `LimbPlan` give a unique integer in `[-2^(B-1), 2^(B-1))`), `nonneg_gate_sound`.
3. BF16. `bf16_decode_sound` (the `.finite` predicate of `install_bf16` against the IEEE-754 value),
   `bf16_signed_mantissa_sound`, `bf16_pack_unique`, `bf16_pack_field_sound`, `bf16_nz_indicator_sound`,
   `bf16_value_eq_zero_iff`, `bf16_decode_injective`.
4. MoE routing. `moe_bitmap_sound`, `moe_topk_margin_sound`, `moe_topk_exact_sound`, `moe_rmx_sound`,
   `moe_argmax_selected_of_full`, `moe_rmx_global_bound`, `moe_rmx_global_max` for the production gates of `_lift`; `topk_tau_schematic_sound` for the
   bitmap + threshold-slack + one-hot `rmx` schema.
5. RMSNorm. `rms_floor_sound`, `rsqrt_enclosure_sound`, `rsqrt_enclosure_floor`,
   `rsqrt_enclosure_floor_1e3`, `rsqrt_gates_sound`, `rsqrt_accepts_enclosure`.
6. Gated DeltaNet / LRec. `state_step_lift_sound`, `state_trajectory_lift_sound`, `gdn_gates_sound`,
   `gdn_decay_contraction`, `state_box_propagation`, `state_box_propagation_rounded`,
   `round_step_error`.

Scope. Families 3 (`bf16_decode`), 5 (`norm_rsqrt`) and the in-range state update of family 6 are
`REFERENCE_PREDICATE` residue in the production lowering (`AIR_PREDICATE_RESIDUE`): the relation-4
verifier does not check them as gates, so the theorems about them are statements about the reference
predicate, not about verifier-checked constraints. `install_bf16` (`.pack`) and `install_rsqrt`
(`.lower`, `.upper`) are reference lowerings that no production block installs. No emitted gate has
the shape `S_t = α_t S_{t-1} + u_t`; `state_step_lift_sound` is the lift for that shape, and
`gdn_gates_sound` covers the gates that are emitted. Families 1, 2, the MoE gates of family 4 and
`gdn_gates_sound` are about algebraic gates (`AIR_GATE_FAMILIES`, `LimbPlan`). Family 4 and
`gdn_gates_sound` are stated over `ℤ` after the lift of family 1. That the Python `Program` emits
exactly these constraints is read off the source, not extracted mechanically.
-/
import Mathlib
import FlashProveNext.Gadgets
import FlashProveNext.Fpn2Soundness

open scoped BigOperators

namespace FlashProveNext
namespace Fpn2Gadgets

/-! ## 0. Production parameters -/

/-- `AIR_LIMB_BITS`. -/
def limbBits : ℕ := 7

/-- Size of the relation-4 limb table `[0, 128)`. -/
def limbBase : ℕ := 128

/-- Signed lift bound exponent of `Program.check_bounds` (`min(P, 1 << 62)`). -/
def liftBits : ℕ := 62

theorem limbBase_eq : limbBase = 2 ^ limbBits := by norm_num [limbBase, limbBits]

theorem goldilocks_cast_eq : (goldilocks : ℤ) = 2 ^ 64 - 2 ^ 32 + 1 := by
  rw [goldilocks_eq]; norm_num

theorem lift_bound_lt_goldilocks : (2 : ℤ) ^ 62 < (goldilocks : ℤ) := by
  rw [goldilocks_cast_eq]; norm_num

/-- `min(P, 1 << 62) = 2^62`, and twice the bound still fits below `P`. -/
theorem lift_bound_eq_min :
    min (goldilocks : ℤ) (2 ^ liftBits) = 2 ^ 62 ∧ 2 * (2 : ℤ) ^ 62 < (goldilocks : ℤ) := by
  rw [goldilocks_cast_eq]; norm_num [liftBits]

theorem natCast_inj_small {a b : ℕ} (ha : a < goldilocks) (hb : b < goldilocks)
    (h : (a : ZMod goldilocks) = (b : ZMod goldilocks)) : a = b := by
  have h' := (ZMod.natCast_eq_natCast_iff' a b goldilocks).mp h
  rwa [Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] at h'

/-! ## 1. Signed lift `ZMod goldilocks → ℤ` -/

/-- A field zero whose integer preimage is below `p` in absolute value is an integer zero. -/
theorem signed_lift_zero_of_abs_lt_p {x : ℤ} (hx : |x| < (goldilocks : ℤ))
    (h : (x : ZMod goldilocks) = 0) : x = 0 :=
  Int.eq_zero_of_abs_lt_dvd ((ZMod.intCast_zmod_eq_zero_iff_dvd x goldilocks).mp h) hx

theorem signed_lift_zero_iff {x : ℤ} (hx : |x| < 2 ^ 62) :
    (x : ZMod goldilocks) = 0 ↔ x = 0 :=
  ⟨signed_lift_zero_of_abs_lt_p (hx.trans lift_bound_lt_goldilocks), fun h => by simp [h]⟩

theorem signed_lift_eq_of_sub_bound {a b : ℤ} (hab : |a - b| < 2 ^ 62)
    (h : (a : ZMod goldilocks) = (b : ZMod goldilocks)) : a = b := by
  have h0 : ((a - b : ℤ) : ZMod goldilocks) = 0 := by rw [Int.cast_sub, h, sub_self]
  have := (signed_lift_zero_iff hab).mp h0
  omega

/-- One Plonkish row `q_M a b + q_L a + q_R b + q_O c + q_C = 0` (`AirRow`): under the lift bound of
`Program.check_bounds` on the integer expression, the field equation is an integer equation. -/
theorem signed_lift_mul_add_sound {qM qL qR qO qC a b c : ℤ}
    (hb : |qM * a * b + qL * a + qR * b + qO * c + qC| < 2 ^ 62)
    (h : (qM : ZMod goldilocks) * a * b + qL * a + qR * b + qO * c + qC = 0) :
    qM * a * b + qL * a + qR * b + qO * c + qC = 0 := by
  refine (signed_lift_zero_iff hb).mp ?_
  push_cast
  exact h

/-- The canonical coefficient `c % P` of `Gate.from_poly` is the field element of the signed `c`. -/
theorem canonical_coeff_cast (c : ℤ) :
    ((c % (goldilocks : ℤ) : ℤ) : ZMod goldilocks) = (c : ZMod goldilocks) :=
  ZMod.intCast_mod c goldilocks

/-- `Gate.evaluate`: `∑_k coeff_k · ∏ wires_k` for a gate given as a list of (coefficient, monomial). -/
def gateEval {R : Type*} [CommRing R] {ι : Type*} (g : List (ℤ × List ι)) (w : ι → R) : R :=
  (g.map fun t => (t.1 : R) * (t.2.map w).prod).sum

theorem gateEval_cast {ι : Type*} (g : List (ℤ × List ι)) (w : ι → ℤ) :
    gateEval g (fun i => ((w i : ℤ) : ZMod goldilocks)) = ((gateEval g w : ℤ) : ZMod goldilocks) := by
  induction g with
  | nil => simp [gateEval]
  | cons t g ih =>
    have hm : ((t.2.map fun i => ((w i : ℤ) : ZMod goldilocks)).prod) =
        (((t.2.map w).prod : ℤ) : ZMod goldilocks) := by
      induction t.2 with
      | nil => simp
      | cons a l ihl => simp [ihl]
    simp only [gateEval, List.map_cons, List.sum_cons] at ih ⊢
    rw [ih, hm]
    push_cast
    ring

/-- Any gate of any degree: field witness `W`, integer witness `w` with `W = w` in the field (supplied by
`limb7_field_range_sound`), field gate satisfied and lift bound on the integer value give the integer
relation. -/
theorem gate_lift_sound {ι : Type*} (g : List (ℤ × List ι)) (W : ι → ZMod goldilocks) (w : ι → ℤ)
    (hW : ∀ i, W i = ((w i : ℤ) : ZMod goldilocks))
    (hb : |gateEval g w| < 2 ^ 62) (h : gateEval g W = 0) : gateEval g w = 0 := by
  refine (signed_lift_zero_iff hb).mp ?_
  rw [← gateEval_cast]
  have : W = fun i => ((w i : ℤ) : ZMod goldilocks) := funext hW
  rw [← this]
  exact h

/-- `_gate_rows`, the multi-row form of a gate with more than three linear terms: accumulator cells
`acc_0 = t_0 + t_1`, `acc_{j+1} = acc_j + t_{j+2}` and the closing row
`acc_n + t_{n+2} + t_{n+3} + const = 0` are together the single equation `∑ t_i + const = 0`. Here `t_i`
is a term `c_i · x_i`, or the cell `p` of a product row `c · u · v - p = 0`. Accumulators are plain field
cells with no range check; the elimination is field algebra and needs no lift. -/
theorem air_row_chain_sound {R : Type*} [CommRing R] (t acc : ℕ → R) (const : R) (n : ℕ)
    (h0 : t 0 + t 1 - acc 0 = 0)
    (hstep : ∀ j < n, acc j + t (j + 2) - acc (j + 1) = 0)
    (hlast : acc n + t (n + 2) + t (n + 3) + const = 0) :
    ∑ i ∈ Finset.range (n + 4), t i + const = 0 := by
  have hacc : ∀ j, j ≤ n → acc j = ∑ i ∈ Finset.range (j + 2), t i := by
    intro j
    induction j with
    | zero =>
      intro _
      rw [Finset.sum_range_succ, Finset.sum_range_one]
      linear_combination -h0
    | succ j ih =>
      intro hj
      have e : j + 1 + 2 = (j + 2) + 1 := by omega
      rw [e, Finset.sum_range_succ, ← ih (by omega)]
      linear_combination -(hstep j (by omega))
  have e4 : n + 4 = (n + 2) + 1 + 1 := by omega
  have e3 : n + 2 + 1 = n + 3 := by omega
  rw [e4, Finset.sum_range_succ, Finset.sum_range_succ, ← hacc n le_rfl, e3]
  linear_combination hlast

/-- The lift bound is necessary: `p` is a nonzero integer that is zero in the field. -/
theorem signed_lift_bound_necessary :
    (((goldilocks : ℤ)) : ZMod goldilocks) = 0 ∧ (goldilocks : ℤ) ≠ 0 := by
  refine ⟨by simp, ?_⟩
  rw [goldilocks_cast_eq]; norm_num

theorem signed_lift_nonvacuous :
    |(1 : ℤ) * 3 * 5 + 2 * 3 + 0 * 5 + (-1) * 21 + 0| < 2 ^ 62 ∧
      ((1 : ℤ) : ZMod goldilocks) * (3 : ℤ) * (5 : ℤ) + (2 : ℤ) * (3 : ℤ) + (0 : ℤ) * (5 : ℤ)
        + ((-1 : ℤ) : ZMod goldilocks) * (21 : ℤ) + (0 : ℤ) = 0 := by
  refine ⟨by norm_num, ?_⟩
  push_cast
  ring

/-! ## 2. 7-bit limb decomposition and range soundness -/

theorem limb7_recompose_lt (k : ℕ) (l : Fin k → ℕ) (hl : ∀ i, l i < 128) :
    ∑ i, l i * 128 ^ (i : ℕ) < 128 ^ k := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [Fin.sum_univ_castSucc]
    have h1 := ih (fun i => l i.castSucc) (fun i => hl _)
    have h2 : l (Fin.last k) ≤ 127 := by have := hl (Fin.last k); omega
    have h3 : l (Fin.last k) * 128 ^ k ≤ 127 * 128 ^ k := Nat.mul_le_mul_right _ h2
    simp only [Fin.val_castSucc, Fin.val_last] at h1 ⊢
    rw [pow_succ]
    omega

/-- `k` limbs in the table `[0, 128)` recompose to a value in `[0, 128^k)`. -/
theorem limb7_recompose_bounds (k : ℕ) (l : Fin k → ℕ) (hl : ∀ i, l i < 128) :
    0 ≤ ∑ i, l i * 128 ^ (i : ℕ) ∧ ∑ i, l i * 128 ^ (i : ℕ) < 128 ^ k :=
  ⟨Nat.zero_le _, limb7_recompose_lt k l hl⟩

/-- With the top limb below `2^t` the recomposition is below `2^t · 128^k`. -/
theorem limb7_tight_lt (k : ℕ) (l : Fin (k + 1) → ℕ) (hl : ∀ i, l i < 128) (t : ℕ)
    (ht : l (Fin.last k) < 2 ^ t) : ∑ i, l i * 128 ^ (i : ℕ) < 2 ^ t * 128 ^ k := by
  rw [Fin.sum_univ_castSucc]
  have h1 := limb7_recompose_lt k (fun i => l i.castSucc) (fun i => hl _)
  have h3 : (l (Fin.last k) + 1) * 128 ^ k ≤ 2 ^ t * 128 ^ k := Nat.mul_le_mul_right _ ht
  have h4 : (l (Fin.last k) + 1) * 128 ^ k = l (Fin.last k) * 128 ^ k + 128 ^ k := by ring
  simp only [Fin.val_castSucc, Fin.val_last] at h1 ⊢
  omega

/-- The `#tighten` gate `tight = top + (128 - 2^t)` with `tight` in the table forces `top < 2^t`. For
`t = 7` the gate is `tight = top` and production omits it. -/
theorem tighten_gate_sound {t top tight : ℕ} (ht7 : t ≤ 7) (htight : tight < 128)
    (hgate : tight = top + (128 - 2 ^ t)) : top < 2 ^ t := by
  have h : 2 ^ t ≤ 2 ^ 7 := Nat.pow_le_pow_right (by norm_num) ht7
  omega

theorem two_pow_split_nat {k t : ℕ} (ht1 : 1 ≤ t) :
    2 ^ t * 128 ^ k = 2 * 2 ^ (7 * k + t - 1) := by
  obtain ⟨b, rfl⟩ := Nat.exists_eq_add_of_le ht1
  have h : 7 * k + (1 + b) - 1 = 7 * k + b := by omega
  rw [h, show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_mul, pow_add, pow_add, pow_one]
  ring

theorem two_pow_split_int {B : ℕ} (hB : 1 ≤ B) : (2 : ℤ) ^ B = 2 * 2 ^ (B - 1) := by
  obtain ⟨b, rfl⟩ := Nat.exists_eq_add_of_le hB
  rw [Nat.add_sub_cancel_left, pow_add, pow_one]

/-- Integer side of `LimbPlan` for a wire of `B = 7k + t` signed bits (`k + 1` limbs, top limb of
`t` bits): the `#recompose` gate `x + 2^(B-1) = ∑ l_i 128^i` under its lift bound, limbs and tightening
cell in the table, give `-2^(B-1) ≤ x ≤ 2^(B-1) - 1` (`ColumnSpec.bounds`). -/
theorem signed_limb7_range_sound {k t : ℕ} (ht1 : 1 ≤ t) (ht7 : t ≤ 7)
    (x : ℤ) (l : Fin (k + 1) → ℕ) (s : ℕ) (hl : ∀ i, l i < 128) (hs : s < 128)
    (hlift : |x + 2 ^ (7 * k + t - 1) - ((∑ i, l i * 128 ^ (i : ℕ) : ℕ) : ℤ)| < 2 ^ 62)
    (hrec : ((x + 2 ^ (7 * k + t - 1) : ℤ) : ZMod goldilocks) =
      ((∑ i, l i * 128 ^ (i : ℕ) : ℕ) : ZMod goldilocks))
    (htight : s = l (Fin.last k) + (128 - 2 ^ t)) :
    -2 ^ (7 * k + t - 1) ≤ x ∧ x ≤ 2 ^ (7 * k + t - 1) - 1 := by
  have htop := tighten_gate_sound ht7 hs htight
  have hlt := limb7_tight_lt k l hl t htop
  rw [two_pow_split_nat ht1] at hlt
  have hlt' : ((∑ i, l i * 128 ^ (i : ℕ) : ℕ) : ℤ) < 2 * 2 ^ (7 * k + t - 1) := by exact_mod_cast hlt
  have hnn : (0 : ℤ) ≤ ((∑ i, l i * 128 ^ (i : ℕ) : ℕ) : ℤ) := Int.natCast_nonneg _
  rw [← Int.cast_natCast (R := ZMod goldilocks)] at hrec
  have heq := signed_lift_eq_of_sub_bound hlift hrec
  constructor <;> linarith

/-- Field side of `LimbPlan`, `B = 7k + t ≤ 62`: a field cell `X` whose `#recompose` and `#tighten`
gates hold with every limb and the tightening cell in the table `[0, 128)` is the image of exactly one
integer of `[-2^(B-1), 2^(B-1))`. No integer witness is assumed. -/
theorem limb7_field_range_sound {k t : ℕ} (ht1 : 1 ≤ t) (ht7 : t ≤ 7) (hB : 7 * k + t ≤ 62)
    (X : ZMod goldilocks) (l : Fin (k + 1) → ℕ) (s : ℕ) (hl : ∀ i, l i < 128) (hs : s < 128)
    (hrec : X + 2 ^ (7 * k + t - 1) - ∑ i, (l i : ZMod goldilocks) * 128 ^ (i : ℕ) = 0)
    (htight : (s : ZMod goldilocks) - (l (Fin.last k) : ZMod goldilocks) - (128 - 2 ^ t) = 0) :
    ∃! x : ℤ, (-2 ^ (7 * k + t - 1) ≤ x ∧ x < 2 ^ (7 * k + t - 1)) ∧ (x : ZMod goldilocks) = X := by
  have hg : 512 < goldilocks := by rw [goldilocks_eq]; norm_num
  have h2t : 2 ^ t ≤ 2 ^ 7 := Nat.pow_le_pow_right (by norm_num) ht7
  have hcast : ((s + 2 ^ t : ℕ) : ZMod goldilocks) = ((l (Fin.last k) + 128 : ℕ) : ZMod goldilocks) := by
    push_cast
    linear_combination htight
  have hl' := hl (Fin.last k)
  have hnat := natCast_inj_small (by omega) (by omega) hcast
  have htop : l (Fin.last k) < 2 ^ t := by omega
  have hlt := limb7_tight_lt k l hl t htop
  rw [two_pow_split_nat ht1] at hlt
  have hlt' : ((∑ i, l i * 128 ^ (i : ℕ) : ℕ) : ℤ) < 2 * 2 ^ (7 * k + t - 1) := by exact_mod_cast hlt
  have hnn : (0 : ℤ) ≤ ((∑ i, l i * 128 ^ (i : ℕ) : ℕ) : ℤ) := Int.natCast_nonneg _
  have hpow : (2 : ℤ) * 2 ^ (7 * k + t - 1) ≤ 2 ^ 62 := by
    rw [← two_pow_split_int (by omega : 1 ≤ 7 * k + t)]
    exact pow_le_pow_right₀ (by norm_num) hB
  refine ⟨((∑ i, l i * 128 ^ (i : ℕ) : ℕ) : ℤ) - 2 ^ (7 * k + t - 1), ⟨⟨by linarith, by linarith⟩, ?_⟩, ?_⟩
  · push_cast
    linear_combination -hrec
  · rintro y ⟨⟨hy1, hy2⟩, hy3⟩
    refine signed_lift_eq_of_sub_bound ?_ ?_
    · rw [abs_lt]
      constructor <;> linarith
    · rw [hy3]
      push_cast
      linear_combination hrec

/-- `Program.nonneg`: the gate `value = slack + 2^(B-1)` with `slack` in its signed `B`-bit range gives
`0 ≤ value < 2^B`. -/
theorem nonneg_gate_sound {B : ℕ} (hB : 1 ≤ B) {value slack : ℤ}
    (hgate : value - slack - 2 ^ (B - 1) = 0) (hlo : -2 ^ (B - 1) ≤ slack)
    (hhi : slack < 2 ^ (B - 1)) : 0 ≤ value ∧ value < 2 ^ B := by
  rw [two_pow_split_int hB]
  constructor <;> linarith

/-- A 9-bit wire (`k = 1`, `t = 2`, e.g. a BF16 exponent cell) holding `100`: limbs `(100, 2)`,
tightening cell `126`. -/
theorem limb7_field_range_nonvacuous :
    ∃ (X : ZMod goldilocks) (l : Fin 2 → ℕ) (s : ℕ), (∀ i, l i < 128) ∧ s < 128 ∧
      X + 2 ^ (7 * 1 + 2 - 1) - ∑ i, (l i : ZMod goldilocks) * 128 ^ (i : ℕ) = 0 ∧
      (s : ZMod goldilocks) - (l (Fin.last 1) : ZMod goldilocks) - (128 - 2 ^ 2) = 0 := by
  refine ⟨100, ![100, 2], 126, ?_, by norm_num, ?_, ?_⟩
  · intro i; fin_cases i <;> simp
  · simp [Fin.sum_univ_two]; norm_num
  · simp; norm_num

/-! ## 3. BF16 dyadic decode -/

/-- IEEE-754 value of a finite BF16 word with fields `(s, e, m)`: 1 sign bit, 8 exponent bits with
bias 127, 7 fraction bits. Subnormal `(-1)^s · 0.m · 2^(-126)`, normal `(-1)^s · 1.m · 2^(e-127)`. -/
def bf16Ieee (s e m : ℕ) : ℚ :=
  if e = 0 then (-1) ^ s * ((m : ℚ) / 128) * 2 ^ (-126 : ℤ)
  else (-1) ^ s * (1 + (m : ℚ) / 128) * 2 ^ ((e : ℤ) - 127)

/-- The `.finite` lookup predicate of `install_bf16` on the cells `(sign, exponent, mantissa,
significand, power)`. -/
def Bf16Finite (s e m : ℕ) (significand power : ℤ) : Prop :=
  s ≤ 1 ∧ e < 255 ∧ m < 128 ∧ significand = (m : ℤ) + (if e = 0 then 0 else 128) ∧
    power = max 1 (e : ℤ) - 134

/-- `BF16.value`: `(-1)^sign · significand · 2^power`. -/
def bf16Value (s : ℕ) (significand power : ℤ) : ℚ := (-1) ^ s * (significand : ℚ) * 2 ^ power

/-- The decoded dyadic `(significand, power)` of a finite word is its IEEE-754 value, normal and
subnormal. -/
theorem bf16_decode_sound {s e m : ℕ} {significand power : ℤ}
    (h : Bf16Finite s e m significand power) :
    bf16Value s significand power = bf16Ieee s e m := by
  obtain ⟨_, _, _, rfl, rfl⟩ := h
  unfold bf16Value bf16Ieee
  by_cases he : e = 0
  · subst he
    have h1 : max 1 ((0 : ℕ) : ℤ) - 134 = -126 - 7 := by norm_num
    rw [if_pos rfl, if_pos rfl, h1, zpow_sub₀ (by norm_num : (2 : ℚ) ≠ 0)]
    push_cast
    norm_num
    ring
  · have he1 : (1 : ℤ) ≤ (e : ℤ) := by omega
    have h1 : max 1 (e : ℤ) - 134 = ((e : ℤ) - 127) - 7 := by rw [max_eq_right he1]; ring
    rw [if_neg he, if_neg he, h1, zpow_sub₀ (by norm_num : (2 : ℚ) ≠ 0)]
    push_cast
    ring

/-- The signed-mantissa form: `M = (1 - 2s)(128·nz + m)`, `E = if e = 0 then 1 else e`, with the
indicator `nz = [e ≠ 0]`, has `M · 2^(E - 134)` equal to the IEEE-754 value. -/
theorem bf16_signed_mantissa_sound {s e m : ℕ} (hs : s ≤ 1) (nz : ℕ)
    (hnz : nz = if e = 0 then 0 else 1) :
    (((1 - 2 * (s : ℤ)) * (128 * (nz : ℤ) + m) : ℤ) : ℚ) *
        2 ^ (((if e = 0 then 1 else e : ℕ) : ℤ) - 134) = bf16Ieee s e m := by
  subst hnz
  have hsign : ((1 - 2 * (s : ℤ) : ℤ) : ℚ) = (-1) ^ s := by
    rcases (by omega : s = 0 ∨ s = 1) with rfl | rfl <;> norm_num
  unfold bf16Ieee
  by_cases he : e = 0
  · subst he
    have h1 : (((1 : ℕ) : ℤ) - 134) = -126 - 7 := by norm_num
    rw [if_pos rfl, if_pos rfl, if_pos rfl, h1, zpow_sub₀ (by norm_num : (2 : ℚ) ≠ 0), Int.cast_mul,
      hsign]
    push_cast
    norm_num
    ring
  · have h1 : ((e : ℤ) - 134) = ((e : ℤ) - 127) - 7 := by ring
    rw [if_neg he, if_neg he, if_neg he, h1, zpow_sub₀ (by norm_num : (2 : ℚ) ≠ 0), Int.cast_mul,
      hsign]
    push_cast
    ring

/-- The `.pack` gate `word = 32768 s + 128 e + m` with range-checked fields determines the fields as
the bit slices of `bf16_decode`. -/
theorem bf16_pack_unique {word s e m : ℕ} (hs : s ≤ 1) (he : e < 256) (hm : m < 128)
    (hpack : word = s * 32768 + e * 128 + m) :
    word < 65536 ∧ s = word / 32768 ∧ e = word / 128 % 256 ∧ m = word % 128 := by
  omega

/-- The `.pack` gate as a field equation lifts to the natural numbers. -/
theorem bf16_pack_field_sound {word s e m : ℕ} (hw : word < 65536) (hs : s ≤ 1) (he : e < 256)
    (hm : m < 128)
    (hpack : (word : ZMod goldilocks) = (s : ZMod goldilocks) * 32768 + e * 128 + m) :
    word = s * 32768 + e * 128 + m := by
  have hg : 2 ^ 17 < goldilocks := by rw [goldilocks_eq]; norm_num
  refine natCast_inj_small (by omega) (by omega) ?_
  push_cast
  exact hpack

/-- An inverse-based zero test in the field: `e (1 - nz) = 0` and `e · inv = nz` force `nz = [e ≠ 0]`.
Production keeps the indicator inside the `.finite` predicate; this is an algebraic lowering of it. -/
theorem bf16_nz_indicator_sound {e : ℕ} (he : e < 256) {nz inv : ZMod goldilocks}
    (h1 : (e : ZMod goldilocks) * (1 - nz) = 0) (h2 : (e : ZMod goldilocks) * inv = nz) :
    nz = if e = 0 then 0 else 1 := by
  by_cases h0 : e = 0
  · subst h0
    simpa using h2.symm
  · rw [if_neg h0]
    have hne : (e : ZMod goldilocks) ≠ 0 := by
      intro hz
      have hd := (ZMod.natCast_eq_zero_iff e goldilocks).mp hz
      have hle := Nat.le_of_dvd (Nat.pos_of_ne_zero h0) hd
      have hg : 256 < goldilocks := by rw [goldilocks_eq]; norm_num
      omega
    have := (mul_eq_zero.mp h1).resolve_left hne
    linear_combination -this

/-- Magnitude numerator of the value scaled by `2^134`: `significand · 2^max(1, e)`. -/
def bf16Mag (e m : ℕ) : ℕ := ((if e = 0 then 0 else 128) + m) * 2 ^ (max 1 e)

theorem bf16Ieee_eq_scaled {s e m : ℕ} (hs : s ≤ 1) :
    bf16Ieee s e m = (((1 - 2 * (s : ℤ)) * (bf16Mag e m : ℤ) : ℤ) : ℚ) / 2 ^ 134 := by
  have hsign : ((1 - 2 * (s : ℤ) : ℤ) : ℚ) = (-1) ^ s := by
    rcases (by omega : s = 0 ∨ s = 1) with rfl | rfl <;> norm_num
  unfold bf16Ieee bf16Mag
  rw [Int.cast_mul, hsign]
  by_cases he : e = 0
  · subst he
    rw [if_pos rfl, if_pos rfl, zpow_neg]
    push_cast
    norm_num
    ring
  · have he1 : max 1 e = e := Nat.max_eq_right (by omega)
    rw [if_neg he, if_neg he, he1, zpow_sub₀ (by norm_num : (2 : ℚ) ≠ 0), zpow_natCast]
    push_cast
    norm_num
    ring

/-- The value is zero exactly for the two zero words `e = 0 ∧ m = 0`. -/
theorem bf16_value_eq_zero_iff {s e m : ℕ} (hs : s ≤ 1) :
    bf16Ieee s e m = 0 ↔ e = 0 ∧ m = 0 := by
  rw [bf16Ieee_eq_scaled hs, div_eq_zero_iff]
  have h2 : ¬ ((2 : ℚ) ^ 134 = 0) := by positivity
  have hsg : (1 - 2 * (s : ℤ)) ≠ 0 := by omega
  have hp : 0 < 2 ^ (max 1 e) := by positivity
  constructor
  · rintro (h | h)
    · have h' : (1 - 2 * (s : ℤ)) * (bf16Mag e m : ℤ) = 0 := by exact_mod_cast h
      have hm0 : bf16Mag e m = 0 := by
        have := (mul_eq_zero.mp h').resolve_left hsg
        exact_mod_cast this
      unfold bf16Mag at hm0
      have hz := (Nat.mul_eq_zero.mp hm0).resolve_right (by omega)
      by_cases he : e = 0
      · rw [if_pos he] at hz; exact ⟨he, by omega⟩
      · rw [if_neg he] at hz; omega
    · exact absurd h h2
  · rintro ⟨rfl, rfl⟩
    left
    simp [bf16Mag]

theorem sign_mag_unique {s s' A A' : ℕ} (hs : s ≤ 1) (hs' : s' ≤ 1) (hA : 0 < A)
    (h : (1 - 2 * (s : ℤ)) * (A : ℤ) = (1 - 2 * (s' : ℤ)) * (A' : ℤ)) : s = s' ∧ A = A' := by
  rcases (by omega : s = 0 ∨ s = 1) with rfl | rfl <;>
    rcases (by omega : s' = 0 ∨ s' = 1) with rfl | rfl <;>
    constructor <;> push_cast at h <;> omega

theorem mag_unique_aux {g g' E d : ℕ} (hd : 1 ≤ d) (hg : g < 256)
    (hE : 128 ≤ g') (h : g * 2 ^ E = g' * 2 ^ (E + d)) : False := by
  have hp : 0 < 2 ^ E := by positivity
  have h1 : g * 2 ^ E = (g' * 2 ^ d) * 2 ^ E := by rw [h, pow_add]; ring
  have h2 : g = g' * 2 ^ d := Nat.eq_of_mul_eq_mul_right hp h1
  have h3 : 2 ≤ 2 ^ d := by
    calc 2 = 2 ^ 1 := by norm_num
      _ ≤ 2 ^ d := Nat.pow_le_pow_right (by norm_num) hd
  have h4 : g' * 2 ≤ g' * 2 ^ d := Nat.mul_le_mul_left g' h3
  omega

/-- Distinct finite nonzero BF16 field triples have distinct values: sign and magnitude are unique. -/
theorem bf16_decode_injective {s e m s' e' m' : ℕ} (hs : s ≤ 1) (he : e < 255) (hm : m < 128)
    (hs' : s' ≤ 1) (he' : e' < 255) (hm' : m' < 128) (hnz : e ≠ 0 ∨ m ≠ 0)
    (hval : bf16Ieee s e m = bf16Ieee s' e' m') : s = s' ∧ e = e' ∧ m = m' := by
  rw [bf16Ieee_eq_scaled hs, bf16Ieee_eq_scaled hs'] at hval
  have h2 : (2 : ℚ) ^ 134 ≠ 0 := by positivity
  have hint : (1 - 2 * (s : ℤ)) * (bf16Mag e m : ℤ) = (1 - 2 * (s' : ℤ)) * (bf16Mag e' m' : ℤ) := by
    have := (div_left_inj' h2).mp hval
    exact_mod_cast this
  have hpos : 0 < bf16Mag e m := by
    unfold bf16Mag
    have hp : 0 < 2 ^ (max 1 e) := by positivity
    refine Nat.mul_pos ?_ hp
    by_cases h0 : e = 0
    · rw [if_pos h0]; rcases hnz with h | h
      · exact absurd h0 h
      · omega
    · rw [if_neg h0]; omega
  obtain ⟨hss, hmag⟩ := sign_mag_unique hs hs' hpos hint
  refine ⟨hss, ?_⟩
  unfold bf16Mag at hmag
  rcases lt_trichotomy (max 1 e) (max 1 e') with hlt | heq | hgt
  · exfalso
    obtain ⟨d, hd⟩ := Nat.exists_eq_add_of_lt hlt
    have he0 : e' ≠ 0 := by
      intro h0
      have h1 : 1 ≤ max 1 e := le_max_left 1 e
      have h2 : max 1 e' = 1 := by rw [h0]; rfl
      omega
    have hE : 128 ≤ (if e' = 0 then 0 else 128) + m' := by rw [if_neg he0]; omega
    have hg : (if e = 0 then 0 else 128) + m < 256 := by split_ifs <;> omega
    rw [hd, add_assoc] at hmag
    exact mag_unique_aux (d := d + 1) (by omega) hg hE hmag
  · rw [heq] at hmag
    have hp : 0 < 2 ^ (max 1 e') := by positivity
    have hsig := Nat.eq_of_mul_eq_mul_right hp hmag
    by_cases h0 : e = 0 <;> by_cases h0' : e' = 0
    · subst h0; subst h0'; simp at hsig; exact ⟨rfl, hsig⟩
    · exfalso; rw [if_pos h0, if_neg h0'] at hsig; omega
    · exfalso; rw [if_neg h0, if_pos h0'] at hsig; omega
    · rw [if_neg h0, if_neg h0'] at hsig
      have h1 : max 1 e = e := Nat.max_eq_right (by omega)
      have h1' : max 1 e' = e' := Nat.max_eq_right (by omega)
      rw [h1, h1'] at heq
      exact ⟨heq, by omega⟩
  · exfalso
    obtain ⟨d, hd⟩ := Nat.exists_eq_add_of_lt hgt
    have he0 : e ≠ 0 := by
      intro h0
      have h1 : 1 ≤ max 1 e' := le_max_left 1 e'
      have h2 : max 1 e = 1 := by rw [h0]; rfl
      omega
    have hE : 128 ≤ (if e = 0 then 0 else 128) + m := by rw [if_neg he0]; omega
    have hg : (if e' = 0 then 0 else 128) + m' < 256 := by split_ifs <;> omega
    rw [hd, add_assoc] at hmag
    exact mag_unique_aux (d := d + 1) (by omega) hg hE hmag.symm

/-- `1.0 = 0x3F80`: `(s, e, m) = (0, 127, 0)`, decoded `128 · 2^(-7)`; and the smallest subnormal. -/
theorem bf16_decode_nonvacuous :
    Bf16Finite 0 127 0 128 (-7) ∧ bf16Ieee 0 127 0 = 1 ∧
      Bf16Finite 1 0 1 1 (-133) ∧ bf16Ieee 1 0 1 = -(1 / 2 ^ 133) := by
  refine ⟨⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩, ?_,
    ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩, ?_⟩
  · norm_num [bf16Ieee]
  · norm_num [bf16Ieee, zpow_neg]

/-! ## 4. MoE `TopK_τ` routing -/

/-- A Boolean vector with sum 1 is the indicator of one position. -/
theorem onehot_row_indicator {E : ℕ} (f : Fin E → ℤ) (hb : ∀ i, f i * (f i - 1) = 0)
    (hsum : ∑ i, f i = 1) : ∃ c, ∀ i, f i = if i = c then 1 else 0 := by
  have hb' : ∀ i, f i = 0 ∨ f i = 1 := fun i => by
    rcases mul_eq_zero.mp (hb i) with h | h
    · exact Or.inl h
    · exact Or.inr (by omega)
  have hex : ∃ c, f c = 1 := by
    by_contra hne
    push Not at hne
    have h0 : ∑ i, f i = 0 := Finset.sum_eq_zero fun i _ => (hb' i).resolve_right (hne i)
    rw [h0] at hsum
    exact absurd hsum (by norm_num)
  obtain ⟨c, hc⟩ := hex
  refine ⟨c, fun i => ?_⟩
  by_cases hic : i = c
  · rw [if_pos hic, hic, hc]
  · rw [if_neg hic]
    have hsplit := Finset.add_sum_erase Finset.univ f (Finset.mem_univ c)
    have hrest : ∑ j ∈ Finset.univ.erase c, f j = 0 := by linarith
    have hnn : ∀ j ∈ Finset.univ.erase c, 0 ≤ f j := fun j _ => by
      rcases hb' j with h | h <;> omega
    exact (Finset.sum_eq_zero_iff_of_nonneg hnn).mp hrest i
      (Finset.mem_erase.mpr ⟨hic, Finset.mem_univ i⟩)

/-- The selector-gated `moe_router_expert` gates of `_lift` that every profile emits, after the signed
lift: `E` experts, `k = top_k`, branch selector `m`, gated bitmap `on_i = m · active_i`, gated argmax
`best_i = m · argmax_i`, the `k × E` one-hot index matrix `h` and the index list. The `nonneg`
families are stated as the inequality that `nonneg_gate_sound` extracts. -/
structure MoeGates (E k : ℕ) (m : ℤ) (score active argmax on best : Fin E → ℤ)
    (h : Fin k → Fin E → ℤ) (index : Fin k → ℤ) (maximum : ℤ) : Prop where
  active_gate : ∀ i, on i = m * active i
  argmax_gate : ∀ i, best i = m * argmax i
  argmax_boolean : ∀ i, best i * (argmax i - 1) = 0
  argmax_one : ∑ i, best i = m
  argmax_attains : ∀ i, best i * (score i - maximum) = 0
  maximum_bound : ∀ i, 0 ≤ on i * (maximum - score i)
  index_onehot_boolean : ∀ r i, h r i * (h r i - 1) = 0
  index_row : ∀ r, ∑ i, h r i = m
  index_column : ∀ i, ∑ r, h r i = on i
  index_value : ∀ r, m * index r = ∑ i : Fin E, ((i : ℕ) : ℤ) * h r i
  index_order : ∀ r r' : Fin k, (r' : ℕ) = (r : ℕ) + 1 → 0 ≤ m * (index r' - index r - 1)

/-- (a) On the selected branch the bitmap is Boolean and selects exactly `top_k` distinct experts. Both
facts follow from the one-hot index matrix; production has no separate booleanity or cardinality
gate. -/
theorem moe_bitmap_sound {E k : ℕ} {m : ℤ} {score active argmax on best : Fin E → ℤ}
    {h : Fin k → Fin E → ℤ} {index : Fin k → ℤ} {maximum : ℤ}
    (g : MoeGates E k m score active argmax on best h index maximum) (hm : m = 1) :
    (∀ i, active i = 0 ∨ active i = 1) ∧
      (Finset.univ.filter fun i => active i = 1).card = k := by
  subst hm
  choose c hc using fun r => onehot_row_indicator (h r) (g.index_onehot_boolean r) (g.index_row r)
  have hidx : ∀ r, index r = ((c r : ℕ) : ℤ) := fun r => by
    have := g.index_value r
    rw [one_mul] at this
    rw [this]
    simp [hc]
  have hstep : ∀ r r' : Fin k, (r' : ℕ) = (r : ℕ) + 1 → index r + 1 ≤ index r' := fun r r' hr => by
    have := g.index_order r r' hr
    linarith
  have hmono : ∀ d : ℕ, ∀ r r' : Fin k, (r' : ℕ) = (r : ℕ) + d → index r + d ≤ index r' := by
    intro d
    induction d with
    | zero =>
      intro r r' hr
      have : r' = r := Fin.ext (by simpa using hr)
      rw [this]; simp
    | succ d ih =>
      intro r r' hr
      have hlt : (r : ℕ) + d < k := by have := r'.isLt; omega
      have h1 := ih r ⟨(r : ℕ) + d, hlt⟩ rfl
      have h2 := hstep ⟨(r : ℕ) + d, hlt⟩ r' (by simp [hr]; omega)
      push_cast
      linarith
  have hinj : Function.Injective c := by
    intro r r' hcc
    have hii : index r = index r' := by rw [hidx, hidx, hcc]
    apply Fin.ext
    rcases lt_trichotomy (r : ℕ) (r' : ℕ) with hlt | heq | hgt
    · exfalso
      obtain ⟨d, hd⟩ := Nat.exists_eq_add_of_lt hlt
      have := hmono (d + 1) r r' (by omega)
      push_cast at this
      linarith
    · exact heq
    · exfalso
      obtain ⟨d, hd⟩ := Nat.exists_eq_add_of_lt hgt
      have := hmono (d + 1) r' r (by omega)
      push_cast at this
      linarith
  have hcol : ∀ i, active i = ((Finset.univ.filter fun r => i = c r).card : ℤ) := fun i => by
    have h1 := g.index_column i
    rw [g.active_gate i, one_mul] at h1
    rw [← h1]
    simp only [hc]
    rw [Finset.sum_boole]
  have hle : ∀ i, (Finset.univ.filter fun r => i = c r).card ≤ 1 := fun i => by
    apply Finset.card_le_one.mpr
    intro a ha b hb
    exact hinj ((Finset.mem_filter.mp ha).2.symm.trans (Finset.mem_filter.mp hb).2)
  refine ⟨fun i => ?_, ?_⟩
  · have := hle i
    rw [hcol i]
    omega
  · have himg : (Finset.univ.filter fun i => active i = 1) = Finset.univ.image c := by
      ext i
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_image]
      constructor
      · intro hi
        rw [hcol i] at hi
        have hpos : 0 < (Finset.univ.filter fun r => i = c r).card := by omega
        obtain ⟨r, hr⟩ := Finset.card_pos.mp hpos
        exact ⟨r, (Finset.mem_filter.mp hr).2.symm⟩
      · rintro ⟨r, rfl⟩
        have hpos : 0 < (Finset.univ.filter fun r' => c r = c r').card :=
          Finset.card_pos.mpr ⟨r, Finset.mem_filter.mpr ⟨Finset.mem_univ r, rfl⟩⟩
        have := hle (c r)
        rw [hcol (c r)]
        omega
    rw [himg, Finset.card_image_of_injective _ hinj, Finset.card_univ, Fintype.card_fin]

/-- (b) The `topk_margin` gates (`top_k < experts`): with the certificate cell `threshold`, every
selected score is `≥ threshold - τ` and every unselected score is `≤ threshold + τ`, so the selected
set is a `2τ`-approximate Top-K set (`Gadgets.IsApproxTopK`, the relation `topk_tau`). -/
theorem moe_topk_margin_sound {E k : ℕ} {m : ℤ} {score active argmax on best : Fin E → ℤ}
    {h : Fin k → Fin E → ℤ} {index : Fin k → ℤ} {maximum threshold tau : ℤ}
    (g : MoeGates E k m score active argmax on best h index maximum) (hm : m = 1)
    (hmargin : ∀ i, 0 ≤ (2 * on i - m) * (score i - threshold) + m * tau) :
    IsApproxTopK score (2 * tau) k (Finset.univ.filter fun i => active i = 1) := by
  obtain ⟨hbool, hcard⟩ := moe_bitmap_sound g hm
  subst hm
  refine ⟨hcard, fun i hi j hj => ?_⟩
  have hai : active i = 1 := (Finset.mem_filter.mp hi).2
  have haj : active j = 0 := by
    rcases hbool j with h0 | h1
    · exact h0
    · exact absurd (Finset.mem_filter.mpr ⟨Finset.mem_univ j, h1⟩) hj
  have h1 := hmargin i
  have h2 := hmargin j
  rw [g.active_gate i, hai] at h1
  rw [g.active_gate j, haj] at h2
  linarith

/-- Exact Top-K at `τ = 0`: every selected expert beats or ties every unselected one. -/
theorem moe_topk_exact_sound {E k : ℕ} {m : ℤ} {score active argmax on best : Fin E → ℤ}
    {h : Fin k → Fin E → ℤ} {index : Fin k → ℤ} {maximum threshold : ℤ}
    (g : MoeGates E k m score active argmax on best h index maximum) (hm : m = 1)
    (hmargin : ∀ i, 0 ≤ (2 * on i - m) * (score i - threshold) + m * 0) :
    ∀ i, active i = 1 → ∀ j, active j ≠ 1 → score j ≤ score i := by
  intro i hi j hj
  have := (moe_topk_margin_sound g hm hmargin).2 i
    (Finset.mem_filter.mpr ⟨Finset.mem_univ i, hi⟩) j
    (fun hmem => hj (Finset.mem_filter.mp hmem).2)
  linarith

/-- (c) `rmx`: the cell `maximum` is attained by a selected expert and bounds every selected score, the
relation `attaining_rmx`. `hsel` is the gate family `argmax_selected`, emitted when `top_k < experts`
(see `moe_argmax_selected_of_full` for `top_k = experts`). -/
theorem moe_rmx_sound {E k : ℕ} {m : ℤ} {score active argmax on best : Fin E → ℤ}
    {h : Fin k → Fin E → ℤ} {index : Fin k → ℤ} {maximum : ℤ}
    (g : MoeGates E k m score active argmax on best h index maximum) (hm : m = 1)
    (hsel : ∀ i, best i * (active i - 1) = 0) :
    (∃ i, active i = 1 ∧ argmax i = 1 ∧ score i = maximum) ∧
      ∀ i, active i = 1 → score i ≤ maximum := by
  subst hm
  have hbest : ∀ i, best i = argmax i := fun i => by rw [g.argmax_gate i, one_mul]
  have hb : ∀ i, argmax i * (argmax i - 1) = 0 := fun i => by
    have := g.argmax_boolean i; rwa [hbest i] at this
  have hs : ∑ i, argmax i = 1 := by
    have := g.argmax_one; simpa [hbest] using this
  obtain ⟨c, hc⟩ := onehot_row_indicator argmax hb hs
  have hcc : argmax c = 1 := by rw [hc c, if_pos rfl]
  refine ⟨⟨c, ?_, hcc, ?_⟩, fun i hi => ?_⟩
  · have := hsel c
    rw [hbest c, hcc] at this
    linarith
  · have := g.argmax_attains c
    rw [hbest c, hcc] at this
    linarith
  · have := g.maximum_bound i
    rw [g.active_gate i, hi] at this
    linarith

/-- With `top_k = experts` every expert is selected, so `argmax_selected` holds without a gate. -/
theorem moe_argmax_selected_of_full {E : ℕ} {m : ℤ} {score active argmax on best : Fin E → ℤ}
    {h : Fin E → Fin E → ℤ} {index : Fin E → ℤ} {maximum : ℤ}
    (g : MoeGates E E m score active argmax on best h index maximum) (hm : m = 1) :
    (∀ i, active i = 1) ∧ ∀ i, best i * (active i - 1) = 0 := by
  obtain ⟨_, hcard⟩ := moe_bitmap_sound g hm
  have huniv : (Finset.univ.filter fun i => active i = 1) = Finset.univ :=
    Finset.eq_univ_of_card _ (by rw [hcard, Fintype.card_fin])
  have hall : ∀ i, active i = 1 := fun i => by
    have : i ∈ (Finset.univ.filter fun i => active i = 1) := by rw [huniv]; exact Finset.mem_univ i
    exact (Finset.mem_filter.mp this).2
  exact ⟨hall, fun i => by rw [hall i]; ring⟩

/-- Every score, selected or not, is at most `maximum + 2τ`. -/
theorem moe_rmx_global_bound {E k : ℕ} {m : ℤ} {score active argmax on best : Fin E → ℤ}
    {h : Fin k → Fin E → ℤ} {index : Fin k → ℤ} {maximum threshold tau : ℤ}
    (g : MoeGates E k m score active argmax on best h index maximum) (hm : m = 1)
    (hsel : ∀ i, best i * (active i - 1) = 0)
    (hmargin : ∀ i, 0 ≤ (2 * on i - m) * (score i - threshold) + m * tau) (htau : 0 ≤ tau) :
    ∀ j, score j ≤ maximum + 2 * tau := by
  obtain ⟨⟨c, hca, _, hcs⟩, hbound⟩ := moe_rmx_sound g hm hsel
  have happ := moe_topk_margin_sound g hm hmargin
  intro j
  by_cases hj : active j = 1
  · have := hbound j hj; linarith
  · have := happ.2 c (Finset.mem_filter.mpr ⟨Finset.mem_univ c, hca⟩) j
      (fun hmem => hj (Finset.mem_filter.mp hmem).2)
    linarith

/-- At `τ = 0` the cell `maximum` is the exact maximum of all scores. -/
theorem moe_rmx_global_max {E k : ℕ} {m : ℤ} {score active argmax on best : Fin E → ℤ}
    {h : Fin k → Fin E → ℤ} {index : Fin k → ℤ} {maximum threshold : ℤ}
    (g : MoeGates E k m score active argmax on best h index maximum) (hm : m = 1)
    (hsel : ∀ i, best i * (active i - 1) = 0)
    (hmargin : ∀ i, 0 ≤ (2 * on i - m) * (score i - threshold) + m * 0) :
    (∃ i, score i = maximum) ∧ ∀ j, score j ≤ maximum := by
  obtain ⟨⟨c, _, _, hcs⟩, _⟩ := moe_rmx_sound g hm hsel
  refine ⟨⟨c, hcs⟩, fun j => ?_⟩
  have := moe_rmx_global_bound g hm hsel hmargin (le_refl 0) j
  linarith

/-- The bitmap + cardinality + threshold-slack + one-hot `rmx` schema over `ℤ`: Boolean bitmap `b` with
`∑ b = k`, slack cells `u, v, w ≥ 0` with `b_i (score_i - τ - u_i) = 0`,
`(1 - b_i)(τ - score_i - v_i) = 0` and `rmx - score_i = w_i`, Boolean selector `sel` with `∑ sel = 1` and
`∑ sel_i (rmx - score_i) = 0`. The selected set has exactly `k` elements, every selected score is at
least every unselected one, and `rmx` is the maximum score. Booleanity of `sel` is needed: without it
`∑ sel_i (rmx - score_i) = 0` has solutions with no score equal to `rmx`. -/
theorem topk_tau_schematic_sound {E k : ℕ} (score b u v w sel : Fin E → ℤ) (τ rmx : ℤ)
    (hb : ∀ i, b i * (b i - 1) = 0) (hcard : ∑ i, b i = k)
    (hu : ∀ i, 0 ≤ u i) (hv : ∀ i, 0 ≤ v i) (hw : ∀ i, 0 ≤ w i)
    (hsel : ∀ i, b i * (score i - τ - u i) = 0)
    (hunsel : ∀ i, (1 - b i) * (τ - score i - v i) = 0)
    (hmax : ∀ i, rmx - score i = w i)
    (hselb : ∀ i, sel i * (sel i - 1) = 0) (hsel1 : ∑ i, sel i = 1)
    (hselmax : ∑ i, sel i * (rmx - score i) = 0) :
    (Finset.univ.filter fun i => b i = 1).card = k ∧
      (∀ i, b i = 1 → ∀ j, b j ≠ 1 → score j ≤ score i) ∧
      (∃ i, score i = rmx) ∧ ∀ i, score i ≤ rmx := by
  have hb' : ∀ i, b i = 0 ∨ b i = 1 := fun i => by
    rcases mul_eq_zero.mp (hb i) with h | h
    · exact Or.inl h
    · exact Or.inr (by omega)
  refine ⟨?_, ?_, ?_, fun i => by have := hmax i; have := hw i; linarith⟩
  · have hind : ∀ i, b i = if b i = 1 then 1 else 0 := fun i => by
      rcases hb' i with h | h <;> simp [h]
    have hsum : ∑ i, b i = ((Finset.univ.filter fun i => b i = 1).card : ℤ) := by
      rw [Finset.sum_congr rfl fun i _ => hind i, Finset.sum_boole]
    rw [hsum] at hcard
    exact_mod_cast hcard
  · intro i hi j hj
    have hj0 : b j = 0 := (hb' j).resolve_right hj
    have h1 := hsel i
    have h2 := hunsel j
    rw [hi, one_mul] at h1
    rw [hj0, sub_zero, one_mul] at h2
    have := hu i
    have := hv j
    linarith
  · obtain ⟨c, hc⟩ := onehot_row_indicator sel hselb hsel1
    refine ⟨c, ?_⟩
    have : ∑ i, sel i * (rmx - score i) = rmx - score c := by simp [hc]
    linarith

/-- The schema is satisfiable: two experts, `k = 1`, scores `(5, 3)`, `τ = 4`, `rmx = 5`. -/
theorem topk_tau_schematic_nonvacuous :
    ∃ (score b u v w sel : Fin 2 → ℤ) (τ rmx : ℤ),
      (∀ i, b i * (b i - 1) = 0) ∧ ∑ i, b i = ((1 : ℕ) : ℤ) ∧
      (∀ i, 0 ≤ u i) ∧ (∀ i, 0 ≤ v i) ∧ (∀ i, 0 ≤ w i) ∧
      (∀ i, b i * (score i - τ - u i) = 0) ∧ (∀ i, (1 - b i) * (τ - score i - v i) = 0) ∧
      (∀ i, rmx - score i = w i) ∧ (∀ i, sel i * (sel i - 1) = 0) ∧ ∑ i, sel i = 1 ∧
      ∑ i, sel i * (rmx - score i) = 0 := by
  refine ⟨![5, 3], ![1, 0], ![1, 0], ![0, 1], ![0, 2], ![1, 0], 4, 5, ?_⟩
  simp [Fin.forall_fin_two, Fin.sum_univ_two]

/-- Two experts, `top_k = 1`, scores `(5, 3)`: expert 0 selected and attaining. -/
theorem moe_gates_nonvacuous :
    MoeGates 2 1 1 ![5, 3] ![1, 0] ![1, 0] ![1, 0] ![1, 0] (fun _ => ![1, 0]) (fun _ => 0) 5 ∧
      (∀ i : Fin 2, (![1, 0] i : ℤ) * ((![1, 0] i : ℤ) - 1) = 0) ∧
      (∀ i : Fin 2, 0 ≤ (2 * (![1, 0] i : ℤ) - 1) * ((![5, 3] i : ℤ) - 4) + 1 * 0) := by
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_, ?_⟩
  · intro i; fin_cases i <;> simp
  · intro i; fin_cases i <;> simp
  · intro i; fin_cases i <;> simp
  · simp [Fin.sum_univ_two]
  · intro i; fin_cases i <;> simp
  · intro i; fin_cases i <;> simp
  · intro r i; fin_cases i <;> simp
  · intro r; simp [Fin.sum_univ_two]
  · intro i; fin_cases i <;> simp
  · intro r; simp [Fin.sum_univ_two]
  · intro r r' hr; exfalso; have := r.isLt; have := r'.isLt; omega
  · intro i; fin_cases i <;> simp
  · intro i; fin_cases i <;> simp

/-! ## 5. RMSNorm floor and `rsqrt` enclosure -/

/-- The residual floor of `rmsnorm_block`: `∑ x² ≥ width · floor²` gives `rms ≥ floor`. -/
theorem rms_floor_sound {width : ℕ} (hw : 0 < width) {sq f : ℝ} (hf : 0 < f)
    (h : (width : ℝ) * f ^ 2 ≤ sq) : f ≤ Real.sqrt (sq / width) := by
  have hw' : (0 : ℝ) < width := by exact_mod_cast hw
  rw [Real.le_sqrt' hf, le_div_iff₀ hw']
  linarith

/-- With `s > 0`, `y > 0` and `|y² s - 1| ≤ ε`: `|y - 1/√s| ≤ ε / (√s (y√s + 1)) ≤ ε / √s`. -/
theorem rsqrt_enclosure_sound {s y ε : ℝ} (hs : 0 < s) (hy : 0 < y) (h : |y ^ 2 * s - 1| ≤ ε) :
    |y - 1 / Real.sqrt s| ≤ ε / (Real.sqrt s * (y * Real.sqrt s + 1)) ∧
      ε / (Real.sqrt s * (y * Real.sqrt s + 1)) ≤ ε / Real.sqrt s := by
  have hr0 : 0 < Real.sqrt s := Real.sqrt_pos.mpr hs
  have hrr : Real.sqrt s ^ 2 = s := Real.sq_sqrt hs.le
  generalize Real.sqrt s = r at hr0 hrr ⊢
  subst hrr
  have hε : 0 ≤ ε := (abs_nonneg _).trans h
  have hpos : 0 < y * r + 1 := by positivity
  have hden : 0 < r * (y * r + 1) := by positivity
  have hid : y - 1 / r = (y ^ 2 * r ^ 2 - 1) / (r * (y * r + 1)) := by
    rw [eq_div_iff hden.ne']
    field_simp
    ring
  constructor
  · rw [hid, abs_div, abs_of_pos hden]
    exact div_le_div_of_nonneg_right h hden.le
  · apply div_le_div_of_nonneg_left hε hr0
    nlinarith [mul_pos hy hr0]

/-- The same with a floor `f² ≤ s`: `|y - 1/√s| ≤ ε / f`. -/
theorem rsqrt_enclosure_floor {s y ε f : ℝ} (hf : 0 < f) (hsf : f ^ 2 ≤ s) (hy : 0 < y)
    (h : |y ^ 2 * s - 1| ≤ ε) : |y - 1 / Real.sqrt s| ≤ ε / f := by
  have hs : 0 < s := lt_of_lt_of_le (by positivity) hsf
  obtain ⟨h1, h2⟩ := rsqrt_enclosure_sound hs hy h
  have hε : 0 ≤ ε := (abs_nonneg _).trans h
  have hfr : f ≤ Real.sqrt s := (Real.le_sqrt' hf).mpr hsf
  exact h1.trans (h2.trans (div_le_div_of_nonneg_left hε hf hfr))

/-- At the production floor `RESIDUAL_RMS_FLOOR = 1/1000`: `|y - 1/√s| ≤ 1000 ε`. -/
theorem rsqrt_enclosure_floor_1e3 {s y ε : ℝ} (hsf : ((1 : ℝ) / 1000) ^ 2 ≤ s) (hy : 0 < y)
    (h : |y ^ 2 * s - 1| ≤ ε) : |y - 1 / Real.sqrt s| ≤ 1000 * ε := by
  have := rsqrt_enclosure_floor (by norm_num : (0 : ℝ) < 1 / 1000) hsf hy h
  have h2 : ε / (1 / 1000) = 1000 * ε := by ring
  linarith

/-- The `.lower` / `.upper` gates of `install_rsqrt` with non-negative slack cells give the two-sided
test `u · low² ≤ scale² ≤ u · high²` of `Rsqrt.accepts` for `u = n / d`. -/
theorem rsqrt_gates_sound {n d low high scale ls us : ℤ} (hd : 0 < d) (hls : 0 ≤ ls) (hus : 0 ≤ us)
    (hlower : n * low * low + ls = d * scale ^ 2) (hupper : d * scale ^ 2 + us = n * high * high) :
    ((n : ℝ) / d) * (low : ℝ) ^ 2 ≤ (scale : ℝ) ^ 2 ∧ (scale : ℝ) ^ 2 ≤ ((n : ℝ) / d) * (high : ℝ) ^ 2 := by
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have h1 : (n : ℝ) * low * low + ls = d * (scale : ℝ) ^ 2 := by exact_mod_cast hlower
  have h2 : (d : ℝ) * (scale : ℝ) ^ 2 + us = n * high * high := by exact_mod_cast hupper
  have h3 : (0 : ℝ) ≤ ls := by exact_mod_cast hls
  have h4 : (0 : ℝ) ≤ us := by exact_mod_cast hus
  constructor
  · rw [div_mul_eq_mul_div, div_le_iff₀ hd']
    nlinarith
  · rw [div_mul_eq_mul_div, le_div_iff₀ hd']
    nlinarith

/-- `Rsqrt.accepts` ⟹ `Rsqrt.enclosure` contains `1/√x`: for `x = u · 2^(2k)`, `u > 0`,
`u · low² ≤ scale² ≤ u · high²`, the interval `[low, high] · 2^(-k) / scale` contains `1/√x`. -/
theorem rsqrt_accepts_enclosure {u x low high scale : ℝ} {k : ℤ} (hscale : 0 < scale) (hu : 0 < u)
    (hx : x = u * (2 : ℝ) ^ (2 * k)) (hlow : 0 ≤ low) (hhigh : 0 ≤ high)
    (hl : u * low ^ 2 ≤ scale ^ 2) (hh : scale ^ 2 ≤ u * high ^ 2) :
    low * ((2 : ℝ) ^ (-k) / scale) ≤ 1 / Real.sqrt x ∧
      1 / Real.sqrt x ≤ high * ((2 : ℝ) ^ (-k) / scale) := by
  have hP : (0 : ℝ) < (2 : ℝ) ^ k := by positivity
  have hsq : (2 : ℝ) ^ (2 * k) = ((2 : ℝ) ^ k) ^ 2 := by
    rw [two_mul, zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0), sq]
  have hr0 : 0 < Real.sqrt u := Real.sqrt_pos.mpr hu
  have hrr : Real.sqrt u ^ 2 = u := Real.sq_sqrt hu.le
  have hsx : Real.sqrt x = Real.sqrt u * (2 : ℝ) ^ k := by
    rw [hx, hsq, Real.sqrt_mul hu.le, Real.sqrt_sq hP.le]
  rw [hsx, zpow_neg]
  generalize Real.sqrt u = r at hr0 hrr ⊢
  generalize (2 : ℝ) ^ k = P at hP ⊢
  subst hrr
  have h1 : low * r ≤ scale := by
    by_contra hc
    push Not at hc
    nlinarith [mul_pos (sub_pos.mpr hc) (add_pos_of_nonneg_of_pos (mul_nonneg hlow hr0.le) hscale)]
  have h2 : scale ≤ high * r := by
    by_contra hc
    push Not at hc
    nlinarith [mul_pos (sub_pos.mpr hc) (add_pos_of_pos_of_nonneg hscale (mul_nonneg hhigh hr0.le))]
  have hden : 0 < r * P * scale := by positivity
  constructor
  · have hid : 1 / (r * P) - low * (P⁻¹ / scale) = (scale - low * r) / (r * P * scale) := by
      field_simp
    have : 0 ≤ 1 / (r * P) - low * (P⁻¹ / scale) := by
      rw [hid]; exact div_nonneg (by linarith) hden.le
    linarith
  · have hid : high * (P⁻¹ / scale) - 1 / (r * P) = (high * r - scale) / (r * P * scale) := by
      field_simp
    have : 0 ≤ high * (P⁻¹ / scale) - 1 / (r * P) := by
      rw [hid]; exact div_nonneg (by linarith) hden.le
    linarith

theorem rsqrt_enclosure_nonvacuous :
    ((1 : ℝ) / 1000) ^ 2 ≤ 4 ∧ (0 : ℝ) < 1 / 2 ∧ |((1 : ℝ) / 2) ^ 2 * 4 - 1| ≤ 0 := by
  norm_num

/-! ## 6. Gated DeltaNet / LRec state transition -/

/-- (a) A state-step equation `S_t = α_t S_{t-1} + u_t` in the field under the lift bound is the integer
recurrence. Production emits no gate of this shape (the in-range update is a rounding predicate). -/
theorem state_step_lift_sound {S Sp α u : ℤ} (hb : |S - (α * Sp + u)| < 2 ^ 62)
    (h : (S : ZMod goldilocks) = (α : ZMod goldilocks) * Sp + u) : S = α * Sp + u := by
  apply signed_lift_eq_of_sub_bound hb
  push_cast
  exact h

theorem state_trajectory_lift_sound (S α u : ℕ → ℤ) (T : ℕ)
    (hb : ∀ t < T, |S (t + 1) - (α t * S t + u t)| < 2 ^ 62)
    (h : ∀ t < T, (S (t + 1) : ZMod goldilocks) = (α t : ZMod goldilocks) * S t + u t) :
    ∀ t < T, S (t + 1) = α t * S t + u t :=
  fun t ht => state_step_lift_sound (hb t ht) (h t ht)

/-- The `gdn_lrec` gates of `_lift` on the selected branch (`g = 1`): `in_range` is Boolean and equals
`[position < extent]`; out of range the state is carried over unchanged and the convolution cell is 0;
in range `0 ≤ decay ≤ scale`. The in-range update itself is a rounding predicate, not a gate. -/
theorem gdn_gates_sound {inRange inside outside position extent state previous convolution decay
    scale : ℤ}
    (hin : inside = 1 * inRange) (hout : outside = 1 - inside)
    (hbool : inside * (inRange - 1) = 0)
    (hcmp : 0 ≤ inside * (extent - 1 - position) + outside * (position - extent))
    (hconv : outside * convolution = 0) (hstate : outside * (state - previous) = 0)
    (hdlo : 0 ≤ inside * decay) (hdhi : 0 ≤ inside * (scale - decay)) :
    (inRange = 0 ∨ inRange = 1) ∧ (inRange = 1 ↔ position < extent) ∧
      (inRange = 0 → state = previous ∧ convolution = 0) ∧
      (inRange = 1 → 0 ≤ decay ∧ decay ≤ scale) := by
  rw [one_mul] at hin
  subst hin
  subst hout
  rcases mul_eq_zero.mp hbool with h0 | h1
  · subst h0
    simp only [zero_mul, sub_zero, one_mul, zero_add] at hcmp hconv hstate
    refine ⟨Or.inl rfl, ⟨fun h => absurd h (by norm_num), fun h => by omega⟩,
      fun _ => ⟨by omega, hconv⟩, fun h => absurd h (by norm_num)⟩
  · have h1' : inside = 1 := by omega
    subst h1'
    simp only [one_mul, sub_self, zero_mul, add_zero] at hcmp hdlo hdhi
    refine ⟨Or.inr rfl, ⟨fun _ => by omega, fun _ => rfl⟩, fun h => absurd h (by norm_num),
      fun _ => ⟨hdlo, by omega⟩⟩

/-- The `gdn_lrec` gates are satisfiable in range: position 2 of extent 5, decay 200 of scale 256. -/
theorem gdn_gates_nonvacuous :
    ∃ inRange inside outside position extent state previous convolution decay scale : ℤ,
      inside = 1 * inRange ∧ outside = 1 - inside ∧ inside * (inRange - 1) = 0 ∧
      0 ≤ inside * (extent - 1 - position) + outside * (position - extent) ∧
      outside * convolution = 0 ∧ outside * (state - previous) = 0 ∧
      0 ≤ inside * decay ∧ 0 ≤ inside * (scale - decay) :=
  ⟨1, 1, 0, 2, 5, 9, 4, 3, 200, 256, by norm_num⟩

/-- A decay cell `0 ≤ decay ≤ scale` is a contraction factor `|decay / scale| ≤ 1`. -/
theorem gdn_decay_contraction {decay scale : ℤ} (hs : 0 < scale) (h0 : 0 ≤ decay)
    (h1 : decay ≤ scale) : |(decay : ℝ) / scale| ≤ 1 := by
  have hs' : (0 : ℝ) < scale := by exact_mod_cast hs
  have h0' : (0 : ℝ) ≤ decay := by exact_mod_cast h0
  have h1' : (decay : ℝ) ≤ scale := by exact_mod_cast h1
  rw [abs_of_nonneg (div_nonneg h0' hs'.le), div_le_one hs']
  exact h1'

/-- (b) Box propagation over any ordered commutative ring (`ℤ`, `ℚ`, `ℝ`): if the reference follows
`Ŝ_{t+1} = α_t Ŝ_t + u_t` with `|α_t| ≤ 1`, the observed states satisfy the step with local error
`δ_t`, and `|S_0 - Ŝ_0| ≤ r_0`, then `|S_n - Ŝ_n| ≤ r_0 + ∑_{t < n} δ_t` for every `n ≤ T`. -/
theorem state_box_propagation {K : Type*} [CommRing K] [LinearOrder K] [IsStrictOrderedRing K]
    (S Sh α u δ : ℕ → K) (r0 : K) (T : ℕ)
    (hα : ∀ t < T, |α t| ≤ 1)
    (hSh : ∀ t < T, Sh (t + 1) = α t * Sh t + u t)
    (hS : ∀ t < T, |S (t + 1) - (α t * S t + u t)| ≤ δ t)
    (h0 : |S 0 - Sh 0| ≤ r0) :
    ∀ n, n ≤ T → |S n - Sh n| ≤ r0 + ∑ t ∈ Finset.range n, δ t := by
  intro n
  induction n with
  | zero => intro _; simpa using h0
  | succ n ih =>
    intro hn
    have hlt : n < T := hn
    have ih' := ih (Nat.le_of_lt hlt)
    have hsplit : S (n + 1) - Sh (n + 1) =
        (S (n + 1) - (α n * S n + u n)) + α n * (S n - Sh n) := by
      rw [hSh n hlt]; ring
    have hmul : |α n * (S n - Sh n)| ≤ |S n - Sh n| := by
      rw [abs_mul]
      calc |α n| * |S n - Sh n| ≤ 1 * |S n - Sh n| :=
            mul_le_mul_of_nonneg_right (hα n hlt) (abs_nonneg _)
        _ = |S n - Sh n| := one_mul _
    rw [hsplit, Finset.sum_range_succ]
    calc |(S (n + 1) - (α n * S n + u n)) + α n * (S n - Sh n)|
        ≤ |S (n + 1) - (α n * S n + u n)| + |α n * (S n - Sh n)| := abs_add_le _ _
      _ ≤ δ n + (r0 + ∑ t ∈ Finset.range n, δ t) := add_le_add (hS n hlt) (hmul.trans ih')
      _ = r0 + (∑ t ∈ Finset.range n, δ t + δ n) := by ring

/-- With nearest rounding (`|S_{t+1} - exact| ≤ 1/2`, any tie rule) the drift after `T` steps is at
most `r_0 + T/2`. -/
theorem state_box_propagation_rounded (S : ℕ → ℤ) (Sh α u : ℕ → ℝ) (r0 : ℝ) (T : ℕ)
    (hα : ∀ t < T, |α t| ≤ 1)
    (hSh : ∀ t < T, Sh (t + 1) = α t * Sh t + u t)
    (hS : ∀ t < T, |(S (t + 1) : ℝ) - (α t * (S t : ℝ) + u t)| ≤ 1 / 2)
    (h0 : |(S 0 : ℝ) - Sh 0| ≤ r0) : |(S T : ℝ) - Sh T| ≤ r0 + (T : ℝ) / 2 := by
  have := state_box_propagation (fun t => (S t : ℝ)) Sh α u (fun _ => (1 / 2 : ℝ)) r0 T hα hSh hS h0
    T (le_refl T)
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul] at this
  linarith

/-- Mathlib's `round` is one admissible nearest rounding. -/
theorem round_step_error (x : ℝ) : |((round x : ℤ) : ℝ) - x| ≤ 1 / 2 := by
  rw [abs_sub_comm]; exact abs_sub_round x

theorem state_step_nonvacuous :
    |(7 : ℤ) - (1 * 5 + 2)| < 2 ^ 62 ∧
      ((7 : ℤ) : ZMod goldilocks) = ((1 : ℤ) : ZMod goldilocks) * (5 : ℤ) + (2 : ℤ) := by
  refine ⟨by norm_num, ?_⟩
  push_cast
  norm_num

end Fpn2Gadgets
end FlashProveNext

#print axioms FlashProveNext.Fpn2Gadgets.limbBase_eq
#print axioms FlashProveNext.Fpn2Gadgets.goldilocks_cast_eq
#print axioms FlashProveNext.Fpn2Gadgets.lift_bound_lt_goldilocks
#print axioms FlashProveNext.Fpn2Gadgets.lift_bound_eq_min
#print axioms FlashProveNext.Fpn2Gadgets.natCast_inj_small
#print axioms FlashProveNext.Fpn2Gadgets.signed_lift_zero_of_abs_lt_p
#print axioms FlashProveNext.Fpn2Gadgets.signed_lift_zero_iff
#print axioms FlashProveNext.Fpn2Gadgets.signed_lift_eq_of_sub_bound
#print axioms FlashProveNext.Fpn2Gadgets.signed_lift_mul_add_sound
#print axioms FlashProveNext.Fpn2Gadgets.canonical_coeff_cast
#print axioms FlashProveNext.Fpn2Gadgets.gateEval_cast
#print axioms FlashProveNext.Fpn2Gadgets.gate_lift_sound
#print axioms FlashProveNext.Fpn2Gadgets.air_row_chain_sound
#print axioms FlashProveNext.Fpn2Gadgets.signed_lift_bound_necessary
#print axioms FlashProveNext.Fpn2Gadgets.signed_lift_nonvacuous
#print axioms FlashProveNext.Fpn2Gadgets.limb7_recompose_lt
#print axioms FlashProveNext.Fpn2Gadgets.limb7_recompose_bounds
#print axioms FlashProveNext.Fpn2Gadgets.limb7_tight_lt
#print axioms FlashProveNext.Fpn2Gadgets.tighten_gate_sound
#print axioms FlashProveNext.Fpn2Gadgets.two_pow_split_nat
#print axioms FlashProveNext.Fpn2Gadgets.two_pow_split_int
#print axioms FlashProveNext.Fpn2Gadgets.signed_limb7_range_sound
#print axioms FlashProveNext.Fpn2Gadgets.limb7_field_range_sound
#print axioms FlashProveNext.Fpn2Gadgets.nonneg_gate_sound
#print axioms FlashProveNext.Fpn2Gadgets.limb7_field_range_nonvacuous
#print axioms FlashProveNext.Fpn2Gadgets.bf16_decode_sound
#print axioms FlashProveNext.Fpn2Gadgets.bf16_signed_mantissa_sound
#print axioms FlashProveNext.Fpn2Gadgets.bf16_pack_unique
#print axioms FlashProveNext.Fpn2Gadgets.bf16_pack_field_sound
#print axioms FlashProveNext.Fpn2Gadgets.bf16_nz_indicator_sound
#print axioms FlashProveNext.Fpn2Gadgets.bf16Ieee_eq_scaled
#print axioms FlashProveNext.Fpn2Gadgets.bf16_value_eq_zero_iff
#print axioms FlashProveNext.Fpn2Gadgets.sign_mag_unique
#print axioms FlashProveNext.Fpn2Gadgets.mag_unique_aux
#print axioms FlashProveNext.Fpn2Gadgets.bf16_decode_injective
#print axioms FlashProveNext.Fpn2Gadgets.bf16_decode_nonvacuous
#print axioms FlashProveNext.Fpn2Gadgets.onehot_row_indicator
#print axioms FlashProveNext.Fpn2Gadgets.moe_bitmap_sound
#print axioms FlashProveNext.Fpn2Gadgets.moe_topk_margin_sound
#print axioms FlashProveNext.Fpn2Gadgets.moe_topk_exact_sound
#print axioms FlashProveNext.Fpn2Gadgets.moe_rmx_sound
#print axioms FlashProveNext.Fpn2Gadgets.moe_argmax_selected_of_full
#print axioms FlashProveNext.Fpn2Gadgets.moe_rmx_global_bound
#print axioms FlashProveNext.Fpn2Gadgets.moe_rmx_global_max
#print axioms FlashProveNext.Fpn2Gadgets.topk_tau_schematic_sound
#print axioms FlashProveNext.Fpn2Gadgets.topk_tau_schematic_nonvacuous
#print axioms FlashProveNext.Fpn2Gadgets.moe_gates_nonvacuous
#print axioms FlashProveNext.Fpn2Gadgets.rms_floor_sound
#print axioms FlashProveNext.Fpn2Gadgets.rsqrt_enclosure_sound
#print axioms FlashProveNext.Fpn2Gadgets.rsqrt_enclosure_floor
#print axioms FlashProveNext.Fpn2Gadgets.rsqrt_enclosure_floor_1e3
#print axioms FlashProveNext.Fpn2Gadgets.rsqrt_gates_sound
#print axioms FlashProveNext.Fpn2Gadgets.rsqrt_accepts_enclosure
#print axioms FlashProveNext.Fpn2Gadgets.rsqrt_enclosure_nonvacuous
#print axioms FlashProveNext.Fpn2Gadgets.state_step_lift_sound
#print axioms FlashProveNext.Fpn2Gadgets.state_trajectory_lift_sound
#print axioms FlashProveNext.Fpn2Gadgets.gdn_gates_sound
#print axioms FlashProveNext.Fpn2Gadgets.gdn_gates_nonvacuous
#print axioms FlashProveNext.Fpn2Gadgets.gdn_decay_contraction
#print axioms FlashProveNext.Fpn2Gadgets.state_box_propagation
#print axioms FlashProveNext.Fpn2Gadgets.state_box_propagation_rounded
#print axioms FlashProveNext.Fpn2Gadgets.round_step_error
#print axioms FlashProveNext.Fpn2Gadgets.state_step_nonvacuous
