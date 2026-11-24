/-
Executable `verifyBytesRel4` for `FPN2` relation-4 containers (`FOLDED_GKR_AIR`,
`specs/11-wire-format.md` §5.5, §6) and its refinement to a typed specification. Stage 3 of the
Lean verifier (M4); Lean core only. The definitions follow `core/src/folded_air.rs` step by step and reuse the relation-agnostic parts of `Fpn2Rel3`
(masked GKR layers, Basefold round replay, fused coset-4 fold, coset Merkle paths).

1. `Rel4Layout` : the fixed-size layout, a function of `(n_vars, n_queries)` (`g`, `rows`, `s`,
   `folds`, `depth`, `gkrLen`, `zcLen`, `pcsLen`, `queryLen`, `bodyUsed`, `block`, `deposit`,
   `libraSlot`, `zcSlot`);
2. `rel4Terms`, `batchedClaim4`, `finalWeight4` : the batched linear claim of §5.5.5 on the witness
   vector `F` (`root_0`) and the circuit vector `C` (`root_circ`), and the final weight
   `W̃(θ, r_1, r')` of §5.5.6;
3. `Codec.airBody`, `parseBodyRel4`, `parseContainerRel4` : canonical decoding of the §5.5.7 layout
   (normative checks 9–12), with `parseContainerRel4_encode`,
   `parseContainerRel4_length_and_zero_pad`, `parseContainerRel4_injective_on_accepted`;
4. `gkrPhase4`, `zcPhase`, `pcsPhase4`, `queryPhase4`, `typedCheckRel4`, `verifyBytesRel4` :
   normative checks 13–19 in the transcript schedule of §5.5.8, with the spec reject codes; check
   13 is the mandatory circuit pin (`E_CIRCUIT_UNPINNED` / `E_CIRCUIT_PINNED`);
5. `GkrSpec4`, `ZcSpec`, `PcsSpec4`, `QuerySpec4`, `Rel4TypedVerify` : the declarative typed
   statement on a parsed container;
6. `verifyBytesRel4_refines_typed`, `verifyBytesRel4_complete`,
   `verifyBytesRel4_downgrade_resistant`, `verifyBytesRel4_injective_on_accepted`,
   `verifyBytesRel4_requires_circuit_pin`, `airBody_enc` : accepted bytes are a canonical,
   fixed-length, zero-padded container of a registered relation-4 profile whose `STMT.root_circ`
   is the caller's pin and on which `Rel4TypedVerify` holds, and conversely;

The relation dispatcher `verifyBytesAny` (one entry point over relations 1, 3, 4 and 5, like
`core/src/verify.rs::verify_bytes`) lives in `Fpn2Rel5`, which imports this module.

Scope. `Rel4TypedVerify` is the algebraic acceptance predicate of relation 4 over abstract hash
oracles. This module proves that the bytes verifier decides exactly that predicate (plus the
circuit pin) on exactly the canonical encodings; it does not prove that the predicate is a sound or
zero-knowledge argument for the Plonkish statement of §5.5.1 (see `THEOREM.md`).
-/
import FlashProveNext.Fpn2Wire
import FlashProveNext.Fpn2Transcript
import FlashProveNext.VerifyBytes
import FlashProveNext.Fpn2Rel3

namespace FlashProveNext

/-! ## Layout (spec §5.5.1, §5.5.7; `core/src/folded_air.rs::AirLayout`) -/

/-- Fixed-size relation-4 layout, a pure function of `(n_vars, n_queries)`. -/
structure Rel4Layout where
  nVars : Nat
  nQueries : Nat
  deriving DecidableEq, Repr

namespace Rel4Layout

/-- `AirLayout::of`: `sc_degree = 3`, `n` even with `12 ≤ n ≤ 28`, `1 ≤ Q ≤ 4096`. -/
def ofProfile (p : Fpn2Profile) : Option Rel4Layout :=
  if p.scDegree = 3 ∧ 12 ≤ p.nVars ∧ p.nVars % 2 = 0 ∧ p.nVars ≤ 28 ∧ 1 ≤ p.nQueries ∧
      p.nQueries ≤ 4096 then
    some ⟨p.nVars, p.nQueries⟩
  else none

/-- `g = n − 5`: the gate table has `K = 2^g` rows and `K` variables. -/
def g (lay : Rel4Layout) : Nat := lay.nVars - 5

/-- `K = 2^g`. -/
def rows (lay : Rel4Layout) : Nat := 2 ^ lay.g

/-- `S = g + 3`: the GKR tree has `2^(S+1)` leaves. -/
def s (lay : Rel4Layout) : Nat := lay.g + 3

/-- `T = n / 2` fused coset-4 folds. -/
def folds (lay : Rel4Layout) : Nat := lay.nVars / 2

/-- Merkle depth of folded layer `t`: `n + 1 − 2t`. -/
def depth (lay : Rel4Layout) (t : Nat) : Nat := lay.nVars + 1 - 2 * t

/-- The geometry of the folded opening (`L`, `L_t / 4`, `ω`) is that of relation 3 at the same
`(n, Q)`; the relation-3 fold and Merkle steps are reused through it. -/
def pcs (lay : Rel4Layout) : Rel3Layout := ⟨lay.nVars, lay.nQueries⟩

def gkrLen (lay : Rel4Layout) : Nat :=
  48 + 96 + ((List.range lay.s).map (fun i => Rel3Layout.layerLen (i + 1))).foldl (· + ·) 0

/-- `Z_sum ‖ g rounds × s(0..4) ‖ 9 finals`. -/
def zcLen (lay : Rel4Layout) : Nat := 24 + lay.g * 120 + 216

/-- `z_blind ‖ (n + 2) rounds ‖ (T − 1) fold roots ‖ FINAL`. -/
def pcsLen (lay : Rel4Layout) : Nat := 24 + 72 * (lay.nVars + 2) + 32 * (lay.folds - 1) + 24

/-- Per query: the `C` leaf opening of layer 0, plus one coset opening per folded layer. -/
def queryLen (lay : Rel4Layout) : Nat :=
  (96 + 32 * lay.depth 0) +
    ((List.range lay.folds).map (fun t => 96 + 32 * lay.depth t)).foldl (· + ·) 0

/-- Bytes used by the body before zero padding. -/
def bodyUsed (lay : Rel4Layout) : Nat :=
  96 + lay.gkrLen + lay.zcLen + lay.pcsLen + lay.nQueries * lay.queryLen

/-- Cell bits addressed by data coordinates, low to high: `0, 2, 3, …, n − 1`. -/
def freeBits (lay : Rel4Layout) : List Nat := 0 :: (List.range (lay.nVars - 2)).map (· + 2)

/-- `(fixed bits, free bits)` of the data-coordinate block `[off, off + 2^k)` in the half
`{b_1 = 1}`. -/
def block (lay : Rel4Layout) (off k : Nat) : List (Nat × Nat) × List Nat :=
  ((1, 1) :: (List.range (lay.freeBits.length - k)).map
      (fun i => (lay.freeBits.getD (k + i) 0, (off / 2 ^ (k + i)) % 2)),
    lay.freeBits.take k)

/-- Deposit the bits of `v` into the free bits of the Libra half `{b_1 = 0}`. -/
def deposit (lay : Rel4Layout) (v : Nat) : Nat :=
  (List.range lay.freeBits.length).foldl
    (fun idx j => idx + ((v / 2 ^ j) % 2) * 2 ^ (lay.freeBits.getD j 0)) 0

/-- Cell index of the GKR Libra coefficient `g_{k,i,e}`: slot `2k(k−1) + 4i + e`. -/
def libraSlot (lay : Rel4Layout) (k i e : Nat) : Nat := lay.deposit (2 * k * (k - 1) + 4 * i + e)

/-- Cell index of the zerocheck Libra coefficient `h_{i,e}`: slot `2S(S+1) + 5i + e`. -/
def zcSlot (lay : Rel4Layout) (i e : Nat) : Nat :=
  lay.deposit (2 * lay.s * (lay.s + 1) + 5 * i + e)

end Rel4Layout

/-- The layout of a relation-4 profile; `none` for any other relation or unwired parameters
(normative check 10). -/
def rel4Layout (p : Fpn2Profile) : Option Rel4Layout :=
  if p.relation = .foldedGkrAir then Rel4Layout.ofProfile p else none

/-- `fpn2-folded-air-v1`: `g = 9`, `K = 512`, `S = 12`, `T = 7`, body 247 176 bytes
(`core/src/folded_air.rs::tests::layout_of_registered_profiles`). -/
theorem rel4Layout_foldedAirV1 :
    rel4Layout foldedAirV1 = some ⟨14, 72⟩ ∧
      (Rel4Layout.g ⟨14, 72⟩, Rel4Layout.rows ⟨14, 72⟩, Rel4Layout.s ⟨14, 72⟩,
        Rel4Layout.folds ⟨14, 72⟩, Rel4Layout.bodyUsed ⟨14, 72⟩) = (9, 512, 12, 7, 247176) ∧
      headerLen + Rel4Layout.bodyUsed ⟨14, 72⟩ ≤ foldedAirV1.containerBytes := by
  decide

/-- `fpn2-folded-air-n12`: body 14 504 bytes, ending at 14 568. -/
theorem rel4Layout_foldedAirN12 :
    rel4Layout foldedAirN12 = some ⟨12, 2⟩ ∧ Rel4Layout.bodyUsed ⟨12, 2⟩ = 14504 ∧
      headerLen + Rel4Layout.bodyUsed ⟨12, 2⟩ ≤ foldedAirN12.containerBytes := by
  decide

/-- `fpn2-folded-air-n20-v1`: `g = 15`, `K = 32768`, `S = 18`, `T = 10`, body 424 104 bytes, ending
at 424 168 (`core/src/folded_air.rs::tests::layout_of_registered_profiles`). -/
theorem rel4Layout_foldedAirN20 :
    rel4Layout foldedAirN20V1 = some ⟨20, 72⟩ ∧
      (Rel4Layout.g ⟨20, 72⟩, Rel4Layout.rows ⟨20, 72⟩, Rel4Layout.s ⟨20, 72⟩,
        Rel4Layout.folds ⟨20, 72⟩, Rel4Layout.bodyUsed ⟨20, 72⟩) = (15, 32768, 18, 10, 424104) ∧
      headerLen + Rel4Layout.bodyUsed ⟨20, 72⟩ ≤ foldedAirN20V1.containerBytes := by
  decide

/-- The relation-3 and relation-4 parsers never both apply to one profile. -/
theorem rel4Layout_disjoint_rel3 (p : Fpn2Profile) (lay : Rel4Layout)
    (h : rel4Layout p = some lay) : rel3Layout p = none ∧ p.bodyUsed = none := by
  unfold rel4Layout at h
  split at h
  · rename_i hrel
    simp [rel3Layout, Fpn2Profile.bodyUsed, hrel]
  · cases h

/-! ## The batched linear claim (spec §5.5.5; `folded_air.rs::terms`) -/

/-- Five consecutive `Fp3` wire fields (one zerocheck round `s(0..4)`). -/
abbrev F3x5 := Fp3 × F3x4

def list5 (v : F3x5) : List Fp3 := v.1 :: list4 v.2

/-- `Σ_i 2^i · r_i`: the multilinear extension of the integer value of a bit string. -/
def intExt (l : List Fp3) : Fp3 :=
  (List.range l.length).foldl
    (fun acc i => acc.add ((l.getD i Fp3.zero).scale (fpOfNat (2 ^ i)))) Fp3.zero

/-- The sparse cells `coef · r^e` for `e < cnt` at `slot e`. -/
def powCells (slot : Nat → Nat) (cnt : Nat) (coef r : Fp3) : List (Nat × Fp3) :=
  ((List.range cnt).foldl
    (fun (acc : List (Nat × Fp3) × Fp3) e => (acc.1 ++ [(slot e, coef.mul acc.2)], acc.2.mul r))
    ([], Fp3.one)).1

/-- The sparse cells of one GKR Libra layer (4 coefficients per round). -/
def libraCells4 (lay : Rel4Layout) (k : Nat) (coef : Fp3) (rs : List Fp3) : List (Nat × Fp3) :=
  (List.range rs.length).flatMap fun i =>
    powCells (lay.libraSlot k i) 4 coef (rs.getD i Fp3.zero)

/-- The sparse cells of the zerocheck Libra mask (5 coefficients per round). -/
def zcCells (lay : Rel4Layout) (coef : Fp3) (zr : List Fp3) : List (Nat × Fp3) :=
  (List.range zr.length).flatMap fun i => powCells (lay.zcSlot i) 5 coef (zr.getD i Fp3.zero)

/-- `[c, c μ, c μ², …]` (`n` terms). -/
def geom (mu : Fp3) : Fp3 → Nat → List Fp3
  | _, 0 => []
  | c, n + 1 => c :: geom mu (c.mul mu) n

def airBlock (lay : Rel4Layout) (off k : Nat) (c : Fp3) (z : List Fp3) : Rel3Block :=
  ⟨(lay.block off k).1, (lay.block off k).2, c, z⟩

/-- The blinding block of mask point `z` on the Libra half `{b_1 = 0}`. -/
def airMaskBlock (lay : Rel4Layout) (coef z : Fp3) : Rel3Block :=
  let free := (List.range lay.nVars).filter (fun b => b != 1)
  ⟨[(1, 0)], free, coef.mul (Fp3.one.sub z), List.replicate free.length z⟩

/-- The batched linear claim of §5.5.5: public parts, the weight vector `a` on `F` (blocks and
sparse Libra cells), the weight vector `c` on `C`, the coefficients of the claimed evaluations
(`g_r(1..S)`, then the 9 zerocheck finals) and the coefficient of `z_blind`. -/
structure Rel4Terms where
  numPub : Fp3
  denPub : Fp3
  fBlocks : List Rel3Block
  fSparse : List (Nat × Fp3)
  cBlocks : List Rel3Block
  coefs : List Fp3
  zCoef : Fp3

/-- `terms(lay, point, beta, gamma, mu, libra, zc_point)`. -/
def rel4Terms (lay : Rel4Layout) (point : List Fp3) (beta gamma mu : Fp3)
    (libra : List LibraEntry) (zr : List Fp3) : Rel4Terms :=
  let g := lay.g
  let s := lay.s
  let k := lay.rows
  let rho0 := point.getD 0 Fp3.zero
  let rho := point.drop 1
  let low := rho.take g
  let top := (rho.drop g).take 3
  let ra := rho.take (s - 1)
  let rhoS := rho.getD (s - 1) Fp3.zero
  let eqb := fun (b : Nat) => eqEval top [bit3 b 0, bit3 b 1, bit3 b 2]
  let real := Fp3.one.sub rho0
  let numPub := real.mul (((eqb 0).add (eqb 1)).add (eqb 2))
  let s05 := (((((eqb 0).add (eqb 1)).add (eqb 2)).add (eqb 3)).add (eqb 4)).add (eqb 5)
  let denPub := real.mul
    (((beta.mul s05).sub ((gamma.mul (eqb 3)).mul (Fp3.one.add (intExt low)))).sub
      ((eqb 5).mul (intExt (low.take 7))))
  let lib := libra.foldl
    (fun (acc : List (Nat × Fp3) × List Fp3 × Fp3) (en : LibraEntry) =>
      (acc.1 ++ libraCells4 lay en.k acc.2.2 en.rs, acc.2.1 ++ [acc.2.2], acc.2.2.mul mu))
    ([], [], mu.mul mu)
  let fc := geom mu lib.2.2 10
  let cf := fun (j : Nat) => fc.getD j Fp3.zero
  let fBlocks :=
    [airBlock lay (4 * k) g (real.mul (eqb 5)).neg low,
     airBlock lay (5 * k) g (real.mul ((eqb 6).sub (eqb 7))) low,
     airBlock lay (8 * k) (s - 1) (rho0.mul (Fp3.one.sub (rhoS.add rhoS))) ra,
     airBlock lay 0 g (mu.mul (real.mul (eqb 0)).neg) low,
     airBlock lay k g (mu.mul (real.mul (eqb 1)).neg) low,
     airBlock lay (2 * k) g (mu.mul (real.mul (eqb 2)).neg) low,
     airBlock lay (3 * k) g (mu.mul (real.mul ((eqb 3).add (eqb 4))).neg) low,
     airBlock lay (6 * k) g (mu.mul (real.mul ((eqb 6).add (eqb 7)))) low,
     airBlock lay (12 * k) (s - 1) (mu.mul rho0) ra,
     airBlock lay 0 g (cf 0) zr,
     airBlock lay k g (cf 1) zr,
     airBlock lay (2 * k) g (cf 2) zr,
     airMaskBlock lay (cf 9) (Fp3.ofFp 2),
     airMaskBlock lay (cf 9) (Fp3.ofFp 3)]
  let cBlocks :=
    [airBlock lay (8 * k) g (real.mul (eqb 3)).neg low,
     airBlock lay (9 * k) g (real.mul (eqb 4)) low,
     airBlock lay (5 * k) g (mu.mul ((real.mul gamma).mul (eqb 0)).neg) low,
     airBlock lay (6 * k) g (mu.mul ((real.mul gamma).mul (eqb 1)).neg) low,
     airBlock lay (7 * k) g (mu.mul ((real.mul gamma).mul (eqb 2)).neg) low,
     airBlock lay 0 g (cf 3) zr,
     airBlock lay k g (cf 4) zr,
     airBlock lay (2 * k) g (cf 5) zr,
     airBlock lay (3 * k) g (cf 6) zr,
     airBlock lay (4 * k) g (cf 7) zr]
  ⟨numPub, denPub, fBlocks, lib.1 ++ zcCells lay (cf 8) zr, cBlocks, lib.2.1 ++ fc.take 9, cf 9⟩

/-- `Y = (num − num_pub) + μ (den − den_pub) + Σ_j coefs_j · claimed_j + zcoef · z_blind`, where
`claimed = g_r(1..S) ‖ (e_a, e_b, e_c, e_M, e_L, e_R, e_O, e_C, g_r)`. -/
def batchedClaim4 (tm : Rel4Terms) (num den mu : Fp3) (claimed : List Fp3) (zBlind : Fp3) : Fp3 :=
  ((tm.coefs.zip claimed).foldl (fun acc p => acc.add (p.1.mul p.2))
      ((num.sub tm.numPub).add (mu.mul (den.sub tm.denPub)))).add
    (tm.zCoef.mul zBlind)

/-- `W̃(θ, r_1, r') = (1 − r_1)((1 − θ) ã(r') + (2θ − 1) c̃(r'))` at the `n + 2` Basefold
challenges `rs = (θ, r_1, r')` (§5.5.6). -/
def finalWeight4 (lay : Rel4Layout) (tm : Rel4Terms) (rs : List Fp3) : Fp3 :=
  let theta := rs.getD 0 Fp3.zero
  let r1 := rs.getD 1 Fp3.zero
  let aR := weightsEval lay.pcs ⟨Fp3.zero, Fp3.zero, tm.fBlocks, tm.fSparse, Fp3.zero⟩ (rs.drop 2)
  let cR := weightsEval lay.pcs ⟨Fp3.zero, Fp3.zero, tm.cBlocks, [], Fp3.zero⟩ (rs.drop 2)
  (Fp3.one.sub r1).mul
    (((Fp3.one.sub theta).mul aR).add (((theta.add theta).sub Fp3.one).mul cR))

/-! ## Typed body and its codec (spec §5.5.7, normative check 12) -/

/-- `STMT` (96 bytes): `root_0 ‖ root_circ ‖ N_s`. -/
structure AirStmt where
  root0 : Bytes
  rootCirc : Bytes
  nS : Bytes

/-- `GKR`: root fraction, layer 0, the masked layers `1..S`. -/
structure AirGkr where
  rootNum : Fp3
  rootDen : Fp3
  layer0 : F3x4
  layers : List GkrLayer

/-- The 9 zerocheck finals `e_a, e_b, e_c, e_M, e_L, e_R, e_O, e_C, g_r`. -/
structure ZcFinals where
  ea : Fp3
  eb : Fp3
  ec : Fp3
  qm : Fp3
  ql : Fp3
  qr : Fp3
  qo : Fp3
  qc : Fp3
  gR : Fp3

/-- `ZEROCHECK`: `Z_sum`, the `g` degree-4 rounds, the finals. -/
structure AirZc where
  zSum : Fp3
  rounds : List F3x5
  finals : ZcFinals

/-- `PCS`: `z_blind`, the two batching rounds, the fused folds with their roots, the last two
rounds, `FINAL`. -/
structure AirPcs where
  zBlind : Fp3
  batch : F3x3 × F3x3
  head : List (F3x3 × F3x3 × Bytes)
  last : F3x3 × F3x3
  final : Fp3

/-- One query: the layer-0 coset of `F` (under `root_0`) and of `C` (under `root_circ`), then the
committed folded layers `1..T−1`. -/
structure AirQuery where
  fLeaf : CosetOpening
  cLeaf : CosetOpening
  rest : List CosetOpening

/-- Decoded body of a relation-4 (`FOLDED_GKR_AIR`) container, in wire order. -/
structure AirBody where
  stmt : AirStmt
  gkr : AirGkr
  zc : AirZc
  pcs : AirPcs
  queries : List AirQuery

/-- A fully parsed relation-4 container. -/
structure Rel4Container where
  header : Fpn2Header
  body : AirBody

def ZcFinals.list (f : ZcFinals) : List Fp3 :=
  [f.ea, f.eb, f.ec, f.qm, f.ql, f.qr, f.qo, f.qc, f.gR]

def Codec.fp3x5 : Codec F3x5 := Codec.fp3.pair Codec.fp3x4

def Codec.airStmt : Codec AirStmt :=
  ((Codec.rawN 32).pair ((Codec.rawN 32).pair (Codec.rawN 32))).map
    (fun t => ⟨t.1, t.2.1, t.2.2⟩) (fun st => (st.root0, st.rootCirc, st.nS)) (fun _ => rfl)

def Codec.airGkr (lay : Rel4Layout) : Codec AirGkr :=
  (Codec.fp3.pair (Codec.fp3.pair (Codec.fp3x4.pair
    (Codec.fam Codec.gkrLayer layerBytes (fun _ _ => rfl) 1 lay.s)))).map
    (fun t => ⟨t.1, t.2.1, t.2.2.1, t.2.2.2⟩)
    (fun G => (G.rootNum, G.rootDen, G.layer0, G.layers)) (fun _ => rfl)

def Codec.zcFinals : Codec ZcFinals :=
  (Codec.fp3x4.pair (Codec.fp3x4.pair Codec.fp3)).map
    (fun t => ⟨t.1.1, t.1.2.1, t.1.2.2.1, t.1.2.2.2, t.2.1.1, t.2.1.2.1, t.2.1.2.2.1,
      t.2.1.2.2.2, t.2.2⟩)
    (fun f => ((f.ea, f.eb, f.ec, f.qm), (f.ql, f.qr, f.qo, f.qc), f.gR)) (fun _ => rfl)

def Codec.airZc (lay : Rel4Layout) : Codec AirZc :=
  (Codec.fp3.pair ((Codec.fam (fun _ => Codec.fp3x5) Codec.fp3x5.enc (fun _ _ => rfl) 0 lay.g).pair
    Codec.zcFinals)).map
    (fun t => ⟨t.1, t.2.1, t.2.2⟩) (fun Z => (Z.zSum, Z.rounds, Z.finals)) (fun _ => rfl)

def Codec.airPcs (lay : Rel4Layout) : Codec AirPcs :=
  (Codec.fp3.pair ((Codec.fp3x3.pair Codec.fp3x3).pair
    ((Codec.fam (fun _ => Codec.pcsFold) pcsFoldBytes (fun _ _ => rfl) 0 (lay.folds - 1)).pair
      ((Codec.fp3x3.pair Codec.fp3x3).pair Codec.fp3)))).map
    (fun t => ⟨t.1, t.2.1, t.2.2.1, t.2.2.2.1, t.2.2.2.2⟩)
    (fun p => (p.zBlind, p.batch, p.head, p.last, p.final)) (fun _ => rfl)

/-- The wire bytes of one query (independent of the layout). -/
def airQueryBytes (q : AirQuery) : Bytes :=
  cosetBytes q.fLeaf ++ (cosetBytes q.cLeaf ++ q.rest.flatMap cosetBytes)

/-- One query: `F` leaf and `C` leaf with `n + 1` siblings each, then for `t = 1..T−1` 4 coset
values and `n + 1 − 2t` siblings. -/
def Codec.airQuery (lay : Rel4Layout) : Codec AirQuery :=
  ((Codec.coset (lay.depth 0)).pair ((Codec.coset (lay.depth 0)).pair
    (Codec.fam (fun t => Codec.coset (lay.depth t)) cosetBytes (fun _ _ => rfl) 1
      (lay.folds - 1)))).map
    (fun t => ⟨t.1, t.2.1, t.2.2⟩) (fun q => (q.fLeaf, q.cLeaf, q.rest)) (fun _ => rfl)

/-- The whole relation-4 body, sections back-to-back in wire order:
`STMT ‖ GKR ‖ ZEROCHECK ‖ PCS ‖ QUERIES`. Every `Fp3` field is decoded canonically in wire order,
so the first non-canonical limb determines `E_NONCANONICAL` (`folded_air.rs::decode`). -/
def Codec.airBody (lay : Rel4Layout) : Codec AirBody :=
  (Codec.airStmt.pair ((Codec.airGkr lay).pair ((Codec.airZc lay).pair ((Codec.airPcs lay).pair
    (Codec.fam (fun _ => Codec.airQuery lay) airQueryBytes (fun _ _ => rfl) 0 lay.nQueries))))).map
    (fun t => ⟨t.1, t.2.1, t.2.2.1, t.2.2.2.1, t.2.2.2.2⟩)
    (fun b => (b.stmt, b.gkr, b.zc, b.pcs, b.queries)) (fun _ => rfl)

/-- The 96 `STMT` wire bytes `root_0 ‖ root_circ ‖ N_s`. -/
def stmtBytesRel4 (st : AirStmt) : Bytes := st.root0 ++ (st.rootCirc ++ st.nS)

/-- The 48 `GKR_ROOT` wire bytes. -/
def rootBytesRel4 (G : AirGkr) : Bytes := Codec.fp3.enc G.rootNum ++ Codec.fp3.enc G.rootDen

/-- `GKR`: `GKR_ROOT ‖ layer 0 ‖ layers 1..S`. -/
def gkrBytesRel4 (G : AirGkr) : Bytes :=
  rootBytesRel4 G ++ (Codec.fp3x4.enc G.layer0 ++ G.layers.flatMap layerBytes)

/-- `ZEROCHECK`: `Z_sum ‖ rounds ‖ finals`. -/
def zcBytes (Z : AirZc) : Bytes :=
  Codec.fp3.enc Z.zSum ++ (Z.rounds.flatMap Codec.fp3x5.enc ++ Codec.zcFinals.enc Z.finals)

/-- `PCS`: `z_blind ‖ round 0 ‖ round 1 ‖ fused folds ‖ last two rounds ‖ FINAL`. -/
def pcsBytesRel4 (p : AirPcs) : Bytes :=
  Codec.fp3.enc p.zBlind ++ ((Codec.fp3x3.enc p.batch.1 ++ Codec.fp3x3.enc p.batch.2) ++
    (p.head.flatMap pcsFoldBytes ++ ((Codec.fp3x3.enc p.last.1 ++ Codec.fp3x3.enc p.last.2) ++
      Codec.fp3.enc p.final)))

/-- Canonical body serializer: the sections in wire order. It does not depend on the layout. The
transcript absorbs exactly these byte strings (`STMT`, `GKR_ROOT`, the pieces of each GKR layer,
`Z_sum`, each zerocheck round, the finals, `z_blind`, each PCS round and fold root, `FINAL`), and
the coset values hashed into Merkle leaves are the `QUERIES` bytes. -/
def encodeBodyRel4 (b : AirBody) : Bytes :=
  stmtBytesRel4 b.stmt ++ (gkrBytesRel4 b.gkr ++ (zcBytes b.zc ++ (pcsBytesRel4 b.pcs ++
    b.queries.flatMap airQueryBytes)))

/-- The encoder of the body codec is the section-wise serializer, for every layout: what
`Codec.airBody` accepts is the concatenation of the sections the verifier absorbs. -/
theorem airBody_enc (lay : Rel4Layout) (b : AirBody) :
    (Codec.airBody lay).enc b = encodeBodyRel4 b := by
  simp [Codec.airBody, Codec.airStmt, Codec.airGkr, Codec.airZc, Codec.airPcs, Codec.map,
    Codec.pair, Codec.rawN, Codec.fam, encodeBodyRel4, stmtBytesRel4, gkrBytesRel4, rootBytesRel4,
    zcBytes, pcsBytesRel4, List.append_assoc]

/-- The 216 wire bytes of the zerocheck finals are the nine `Fp3` encodings in order. -/
theorem zcFinals_enc (f : ZcFinals) :
    Codec.zcFinals.enc f = f.list.flatMap Codec.fp3.enc := by
  simp [Codec.zcFinals, Codec.fp3x4, Codec.fp3x3, Codec.map, Codec.pair, ZcFinals.list,
    List.append_assoc]

/-- Normative checks 9–12 for relation 4: exact length `C_p`, wired backend, all-zero `PAD`,
canonical decoding of every typed field in wire order. -/
def parseBodyRel4 (prof : Fpn2Profile) (l : Bytes) : Except RejectCode AirBody :=
  if l.length ≠ prof.containerBytes then .error .E_LENGTH
  else
    match rel4Layout prof with
    | none => .error .E_BACKEND_UNAVAILABLE
    | some lay =>
      if prof.containerBytes < headerLen + lay.bodyUsed then .error .E_LENGTH
      else if (l.drop headerLen).drop lay.bodyUsed ≠
          List.replicate (prof.containerBytes - headerLen - lay.bodyUsed) 0 then .error .E_PADDING
      else
        match (Codec.airBody lay).dec ((l.drop headerLen).take lay.bodyUsed) with
        | .error e => .error e
        | .ok (b, rest) => if rest = [] then .ok b else .error .E_LENGTH

/-- Strict parser against a pinned profile: header checks 1–5, the header names exactly the
pinned `profile_id` (else `E_PROFILE_PINNED`), then body checks 9–12. -/
def parseContainerRel4 (prof : Fpn2Profile) (raw : ByteArray) : Except RejectCode Rel4Container :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    if h.profileId ≠ prof.profileId then .error .E_PROFILE_PINNED
    else
      match parseBodyRel4 prof (toBytes raw) with
      | .error e => .error e
      | .ok b => .ok ⟨h, b⟩

/-- Zero padding length of a relation-4 profile (`0` when the relation is not wired). -/
def padLenRel4 (prof : Fpn2Profile) : Nat :=
  match rel4Layout prof with
  | none => 0
  | some lay => prof.containerBytes - headerLen - lay.bodyUsed

/-- Canonical serializer (inverse of `parseContainerRel4` on accepted inputs): `C_p` bytes. -/
def encodeContainerRel4 (prof : Fpn2Profile) (c : Rel4Container) : Bytes :=
  encodeHeader c.header ++ (encodeBodyRel4 c.body ++ List.replicate (padLenRel4 prof) 0)

theorem parseBodyRel4_spec {prof : Fpn2Profile} {l : Bytes} {b : AirBody}
    (hp : parseBodyRel4 prof l = .ok b) :
    l.length = prof.containerBytes ∧ ∃ lay, rel4Layout prof = some lay ∧
      headerLen + lay.bodyUsed ≤ prof.containerBytes ∧
      (l.drop headerLen).take lay.bodyUsed = encodeBodyRel4 b ∧
      (l.drop headerLen).drop lay.bodyUsed =
        List.replicate (prof.containerBytes - headerLen - lay.bodyUsed) 0 := by
  unfold parseBodyRel4 at hp
  split at hp
  · cases hp
  · rename_i hlen
    split at hp
    · cases hp
    · rename_i lay hlay
      split at hp
      · cases hp
      · rename_i hfit
        split at hp
        · cases hp
        · rename_i hpad
          split at hp
          · cases hp
          · rename_i b' rest hdec
            split at hp
            · rename_i hrest
              injection hp with hp
              subst hp hrest
              have hs := (Codec.airBody lay).sound _ _ _ hdec
              rw [List.append_nil, airBody_enc] at hs
              exact ⟨Decidable.not_not.mp hlen, lay, hlay, Nat.le_of_not_lt hfit, hs,
                Decidable.not_not.mp hpad⟩
            · cases hp

theorem parseContainerRel4_spec {prof : Fpn2Profile} {raw : ByteArray} {c : Rel4Container}
    (hp : parseContainerRel4 prof raw = .ok c) :
    parseHeader raw = .ok c.header ∧ c.header.profileId = prof.profileId ∧
      parseBodyRel4 prof (toBytes raw) = .ok c.body := by
  unfold parseContainerRel4 at hp
  cases hh : parseHeader raw with
  | error e => simp [hh] at hp
  | ok h =>
    simp only [hh] at hp
    split at hp
    · cases hp
    · rename_i hid
      cases hb : parseBodyRel4 prof (toBytes raw) with
      | error e => simp [hb] at hp
      | ok b =>
        simp only [hb] at hp
        injection hp with hp
        subst hp
        exact ⟨rfl, Decidable.not_not.mp hid, rfl⟩

/-- **Canonical wire representation.** An accepted byte string is exactly the canonical
serialization of the container it parses to. -/
theorem parseContainerRel4_encode {prof : Fpn2Profile} {raw : ByteArray} {c : Rel4Container}
    (hp : parseContainerRel4 prof raw = .ok c) : toBytes raw = encodeContainerRel4 prof c := by
  obtain ⟨hh, _, hb⟩ := parseContainerRel4_spec hp
  obtain ⟨_, lay, hlay, _, htake, hdrop⟩ := parseBodyRel4_spec hb
  have hhead := parseHeaderBytes_encode hh
  have e1 : toBytes raw = (toBytes raw).take headerLen ++ (toBytes raw).drop headerLen :=
    (List.take_append_drop _ _).symm
  have e2 : (toBytes raw).drop headerLen =
      ((toBytes raw).drop headerLen).take lay.bodyUsed ++
        ((toBytes raw).drop headerLen).drop lay.bodyUsed :=
    (List.take_append_drop _ _).symm
  have hpadLen : padLenRel4 prof = prof.containerBytes - headerLen - lay.bodyUsed := by
    simp [padLenRel4, hlay]
  rw [e1, e2, hhead, htake, hdrop, encodeContainerRel4, hpadLen]

/-- An accepted container has exactly `C_p` bytes and every padding byte is zero. -/
theorem parseContainerRel4_length_and_zero_pad {prof : Fpn2Profile} {raw : ByteArray}
    {c : Rel4Container} (hp : parseContainerRel4 prof raw = .ok c) :
    raw.size = prof.containerBytes ∧ ∃ lay, rel4Layout prof = some lay ∧
      headerLen + lay.bodyUsed ≤ prof.containerBytes ∧
      (toBytes raw).drop (headerLen + lay.bodyUsed) =
        List.replicate (prof.containerBytes - headerLen - lay.bodyUsed) 0 ∧
      ∀ i, headerLen + lay.bodyUsed ≤ i → i < raw.size → (toBytes raw)[i]? = some 0 := by
  obtain ⟨_, _, hb⟩ := parseContainerRel4_spec hp
  obtain ⟨hlen, lay, hlay, hfit, _, hdrop⟩ := parseBodyRel4_spec hb
  rw [toBytes_length] at hlen
  rw [List.drop_drop] at hdrop
  refine ⟨hlen, lay, hlay, hfit, hdrop, ?_⟩
  intro i hi hlt
  have h1 : ((toBytes raw).drop (headerLen + lay.bodyUsed))[i - (headerLen + lay.bodyUsed)]? =
      (toBytes raw)[i]? := by
    rw [List.getElem?_drop]
    congr 1
    omega
  rw [← h1, hdrop, List.getElem?_replicate]
  have : i - (headerLen + lay.bodyUsed) < prof.containerBytes - headerLen - lay.bodyUsed := by
    omega
  simp [this]

/-- **Uniqueness of the accepted encoding.** Two accepted byte arrays that parse to the same
container are byte-for-byte equal. -/
theorem parseContainerRel4_injective_on_accepted {prof : Fpn2Profile} {raw₁ raw₂ : ByteArray}
    {c : Rel4Container} (h₁ : parseContainerRel4 prof raw₁ = .ok c)
    (h₂ : parseContainerRel4 prof raw₂ = .ok c) : raw₁ = raw₂ :=
  toBytes_injective ((parseContainerRel4_encode h₁).trans (parseContainerRel4_encode h₂).symm)

/-! ## Checks 14–15: GKR root and masked GKR layers (spec §5.5.3, §5.5.8) -/

/-- Relation 4 calls the same two hash functions as relation 3 (`H_fs`, `B256_M`). -/
abbrev Rel4Oracles := Rel3Oracles

/-- What an accepted relation-4 container asserts: the profile and the `STMT` fields, in
particular the circuit commitment `root_circ` the statement is relative to. -/
structure Rel4Claim where
  profileId : Bytes
  root0 : Bytes
  rootCirc : Bytes
  nS : Bytes
  deriving DecidableEq, Repr

def Rel4Container.acceptedClaim (c : Rel4Container) : Rel4Claim :=
  ⟨c.header.profileId, c.body.stmt.root0, c.body.stmt.rootCirc, c.body.stmt.nS⟩

/-- The transcript after `init`, `absorb_bytes(header)`, `absorb_bytes(STMT)`: both `root_0` and
`root_circ` are bound before any challenge. -/
def rel4Start (O : Rel4Oracles) (pub : PublicInput) (c : Rel4Container) : Fpn2Transcript :=
  ((Fpn2Transcript.init O.hFs pub.transcriptLabel).absorbBytes O.hFs
    (encodeHeader c.header)).absorbBytes O.hFs (stmtBytesRel4 c.body.stmt)

/-- The lookup challenge `β`. -/
def rel4Beta (O : Rel4Oracles) (pub : PublicInput) (c : Rel4Container) : Fp3 :=
  ((rel4Start O pub c).squeezeFp3 O.hFs).1

/-- The wiring challenge `γ`, squeezed right after `β`, and the transcript after it. -/
def rel4GammaTr (O : Rel4Oracles) (pub : PublicInput) (c : Rel4Container) : Fp3 × Fpn2Transcript :=
  ((rel4Start O pub c).squeezeFp3 O.hFs).2.squeezeFp3 O.hFs

/-- The state after `β`, `γ` and `absorb_bytes(GKR_ROOT)`: empty point, claim = the root fraction. -/
def gkrInit4 (O : Rel4Oracles) (pub : PublicInput) (c : Rel4Container) : GkrState :=
  ⟨(rel4GammaTr O pub c).2.absorbBytes O.hFs (rootBytesRel4 c.body.gkr), [],
    c.body.gkr.rootNum, c.body.gkr.rootDen, [], 0⟩

/-- Normative checks 14 and 15: `root_num = 0 ∧ root_den ≠ 0` (else `E_GKR_ROOT`), then layer 0
and the masked layers `1..S` (the relation-3 layer steps, else `E_GKR`). -/
def gkrPhase4 (O : Rel4Oracles) (pub : PublicInput) (c : Rel4Container) :
    Except RejectCode GkrState :=
  if c.body.gkr.rootNum = Fp3.zero ∧ c.body.gkr.rootDen ≠ Fp3.zero then
    andThen (gkrLayer0 O (gkrInit4 O pub c) c.body.gkr.layer0) fun s0 =>
      runList (gkrLayerStep O) s0 c.body.gkr.layers
  else .error .E_GKR_ROOT

/-- The GKR part of the typed statement, ending in state `g` (point `ρ`, leaf claim, Libra
records). -/
def GkrSpec4 (O : Rel4Oracles) (pub : PublicInput) (c : Rel4Container) (g : GkrState) : Prop :=
  (c.body.gkr.rootNum = Fp3.zero ∧ c.body.gkr.rootDen ≠ Fp3.zero) ∧
    ∃ s0, GkrLayer0Spec O (gkrInit4 O pub c) c.body.gkr.layer0 s0 ∧
      RunList (GkrLayerSpec O) s0 c.body.gkr.layers g

theorem gkrPhase4_iff (O : Rel4Oracles) (pub : PublicInput) (c : Rel4Container) (g : GkrState) :
    gkrPhase4 O pub c = .ok g ↔ GkrSpec4 O pub c g := by
  unfold gkrPhase4 GkrSpec4
  rw [ite_ok_iff, andThen_ok_iff]
  constructor
  · rintro ⟨hroot, s0, h0, hl⟩
    exact ⟨hroot, s0, (gkrLayer0_iff _ _ _ _).mp h0,
      (runList_iff _ _ (gkrLayerStep_iff O) _ _ _).mp hl⟩
  · rintro ⟨hroot, s0, h0, hl⟩
    exact ⟨hroot, s0, (gkrLayer0_iff _ _ _ _).mpr h0,
      (runList_iff _ _ (gkrLayerStep_iff O) _ _ _).mpr hl⟩

/-! ## Check 16: Libra-masked zerocheck (spec §5.5.4, §5.5.8) -/

/-- `n` consecutive `squeeze_fp3()` calls. -/
def squeezeMany (O : Rel4Oracles) : Nat → Fpn2Transcript → List Fp3 × Fpn2Transcript
  | 0, tr => ([], tr)
  | n + 1, tr =>
    ((tr.squeezeFp3 O.hFs).1 :: (squeezeMany O n (tr.squeezeFp3 O.hFs).2).1,
      (squeezeMany O n (tr.squeezeFp3 O.hFs).2).2)

/-- The zerocheck point `z_0..z_{g−1}`, squeezed after the last GKR layer. -/
def zcPointTr (O : Rel4Oracles) (lay : Rel4Layout) (g : GkrState) : List Fp3 × Fpn2Transcript :=
  squeezeMany O lay.g g.tr

/-- `absorb_bytes(Z_sum)`, then the Libra challenge `λ_Z`. -/
def zcLamTr (O : Rel4Oracles) (lay : Rel4Layout) (c : Rel4Container) (g : GkrState) :
    Fp3 × Fpn2Transcript :=
  ((zcPointTr O lay g).2.absorbBytes O.hFs (Codec.fp3.enc c.body.zc.zSum)).squeezeFp3 O.hFs

/-- The zerocheck starts from `λ_Z · Z_sum` (the gate sum itself is claimed to be zero). -/
def zcInit (O : Rel4Oracles) (lay : Rel4Layout) (c : Rel4Container) (g : GkrState) : RoundState :=
  ⟨(zcLamTr O lay c g).2, (zcLamTr O lay c g).1.mul c.body.zc.zSum, []⟩

def zcRoundChal (O : Rel4Oracles) (s : RoundState) (sv : F3x5) : Fp3 :=
  ((s.tr.absorbBytes O.hFs (Codec.fp3x5.enc sv)).squeezeFp3 O.hFs).1

def zcRoundNext (O : Rel4Oracles) (s : RoundState) (sv : F3x5) : RoundState :=
  ⟨((s.tr.absorbBytes O.hFs (Codec.fp3x5.enc sv)).squeezeFp3 O.hFs).2,
    lagrangeEval (list5 sv) (zcRoundChal O s sv), s.rs ++ [zcRoundChal O s sv]⟩

/-- One degree-4 zerocheck round: `s(0) + s(1) = target`, else `E_ZEROCHECK`. -/
def zcRoundStep (O : Rel4Oracles) (s : RoundState) (sv : F3x5) : Except RejectCode RoundState :=
  if sv.1.add sv.2.1 = s.target then .ok (zcRoundNext O s sv) else .error .E_ZEROCHECK

def ZcRoundSpec (O : Rel4Oracles) (s : RoundState) (sv : F3x5) (s' : RoundState) : Prop :=
  sv.1.add sv.2.1 = s.target ∧ s' = zcRoundNext O s sv

theorem zcRoundStep_iff (O : Rel4Oracles) (s : RoundState) (sv : F3x5) (s' : RoundState) :
    zcRoundStep O s sv = .ok s' ↔ ZcRoundSpec O s sv s' :=
  guard_ok_iff _ _ _ _

/-- The Plonkish gate `q_M a b + q_L a + q_R b + q_O c + q_C` on the claimed evaluations. -/
def gateEval (f : ZcFinals) : Fp3 :=
  (((f.qm.mul (f.ea.mul f.eb)).add (f.ql.mul f.ea)).add
    ((f.qr.mul f.eb).add (f.qo.mul f.ec))).add f.qc

/-- `eq(z, r) · Gate(finals) + λ_Z · g_r`. -/
def zcExpect (O : Rel4Oracles) (lay : Rel4Layout) (c : Rel4Container) (g : GkrState)
    (zf : RoundState) : Fp3 :=
  ((eqEval (zcPointTr O lay g).1 zf.rs).mul (gateEval c.body.zc.finals)).add
    ((zcLamTr O lay c g).1.mul c.body.zc.finals.gR)

/-- Normative check 16: the `g` zerocheck round sums and the final gate equation, else
`E_ZEROCHECK`. The end state holds the gate point `r` and the transcript before the finals. -/
def zcPhase (O : Rel4Oracles) (lay : Rel4Layout) (c : Rel4Container) (g : GkrState) :
    Except RejectCode RoundState :=
  andThen (runList (zcRoundStep O) (zcInit O lay c g) c.body.zc.rounds) fun zf =>
    if zf.target = zcExpect O lay c g zf then .ok zf else .error .E_ZEROCHECK

def ZcSpec (O : Rel4Oracles) (lay : Rel4Layout) (c : Rel4Container) (g : GkrState)
    (zf : RoundState) : Prop :=
  RunList (ZcRoundSpec O) (zcInit O lay c g) c.body.zc.rounds zf ∧
    zf.target = zcExpect O lay c g zf

theorem zcPhase_iff (O : Rel4Oracles) (lay : Rel4Layout) (c : Rel4Container) (g : GkrState)
    (zf : RoundState) : zcPhase O lay c g = .ok zf ↔ ZcSpec O lay c g zf := by
  unfold zcPhase ZcSpec
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨p, hr, h⟩
    obtain ⟨hfin, hp⟩ := (guard_ok_iff _ _ _ _).mp h
    subst hp
    exact ⟨(runList_iff _ _ (zcRoundStep_iff O) _ _ _).mp hr, hfin⟩
  · rintro ⟨hr, hfin⟩
    exact ⟨zf, (runList_iff _ _ (zcRoundStep_iff O) _ _ _).mpr hr,
      (guard_ok_iff _ _ _ _).mpr ⟨hfin, rfl⟩⟩

/-! ## Checks 17–18: batched Basefold sumcheck and final evaluation (spec §5.5.5, §5.5.6) -/

/-- `absorb_bytes(finals)`, `absorb_bytes(z_blind)`, then the batching challenge `μ`. -/
def pcsMuTr4 (O : Rel4Oracles) (c : Rel4Container) (zf : RoundState) : Fp3 × Fpn2Transcript :=
  ((zf.tr.absorbBytes O.hFs (Codec.zcFinals.enc c.body.zc.finals)).absorbBytes O.hFs
    (Codec.fp3.enc c.body.pcs.zBlind)).squeezeFp3 O.hFs

def pcsTerms4 (O : Rel4Oracles) (pub : PublicInput) (lay : Rel4Layout) (c : Rel4Container)
    (g : GkrState) (zf : RoundState) : Rel4Terms :=
  rel4Terms lay g.point (rel4Beta O pub c) (rel4GammaTr O pub c).1 (pcsMuTr4 O c zf).1 g.libra
    zf.rs

/-- The Basefold sumcheck starts from the batched linear claim `Y` of §5.5.5. -/
def pcsInit4 (O : Rel4Oracles) (pub : PublicInput) (lay : Rel4Layout) (c : Rel4Container)
    (g : GkrState) (zf : RoundState) : PcsState :=
  ⟨(pcsMuTr4 O c zf).2,
    batchedClaim4 (pcsTerms4 O pub lay c g zf) g.num g.den (pcsMuTr4 O c zf).1
      (g.libra.map (fun en => en.gR) ++ c.body.zc.finals.list) c.body.pcs.zBlind, []⟩

/-- The `n + 2` rounds in wire order: the two batching rounds, then for `t < T − 1` rounds
`2 + 2t`, `3 + 2t` and `root_{t+1}`, then the last two rounds. -/
def pcsOps4 (p : AirPcs) : List PcsOp :=
  [.round p.batch.1, .round p.batch.2] ++
    (p.head.flatMap (fun f => [.round f.1, .round f.2.1, .root f.2.2]) ++
      [.round p.last.1, .round p.last.2])

/-- Normative checks 17 and 18: the `n + 2` Basefold round sums (else `E_SUMCHECK`), then
`y = FINAL · W̃(θ, r_1, r')` (else `E_FINAL`). -/
def pcsPhase4 (O : Rel4Oracles) (pub : PublicInput) (lay : Rel4Layout) (c : Rel4Container)
    (g : GkrState) (zf : RoundState) : Except RejectCode PcsState :=
  andThen (runList (pcsStep O) (pcsInit4 O pub lay c g zf) (pcsOps4 c.body.pcs)) fun pf =>
    if pf.y = c.body.pcs.final.mul (finalWeight4 lay (pcsTerms4 O pub lay c g zf) pf.rs) then
      .ok pf
    else .error .E_FINAL

def PcsSpec4 (O : Rel4Oracles) (pub : PublicInput) (lay : Rel4Layout) (c : Rel4Container)
    (g : GkrState) (zf : RoundState) (pf : PcsState) : Prop :=
  RunList (PcsOpSpec O) (pcsInit4 O pub lay c g zf) (pcsOps4 c.body.pcs) pf ∧
    pf.y = c.body.pcs.final.mul (finalWeight4 lay (pcsTerms4 O pub lay c g zf) pf.rs)

theorem pcsPhase4_iff (O : Rel4Oracles) (pub : PublicInput) (lay : Rel4Layout) (c : Rel4Container)
    (g : GkrState) (zf : RoundState) (pf : PcsState) :
    pcsPhase4 O pub lay c g zf = .ok pf ↔ PcsSpec4 O pub lay c g zf pf := by
  unfold pcsPhase4 PcsSpec4
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨p, hr, h⟩
    obtain ⟨hfin, hp⟩ := (guard_ok_iff _ _ _ _).mp h
    subst hp
    exact ⟨(runList_iff _ _ (pcsStep_iff O) _ _ _).mp hr, hfin⟩
  · rintro ⟨hr, hfin⟩
    exact ⟨pf, (runList_iff _ _ (pcsStep_iff O) _ _ _).mpr hr,
      (guard_ok_iff _ _ _ _).mpr ⟨hfin, rfl⟩⟩

/-! ## Check 19: dual layer-0 leaves, coset Merkle paths, fold consistency (spec §5.5.6) -/

/-- `root_0` from `STMT`, then the committed fold roots `root_1 .. root_{T−1}` of `H = F + θ C`. -/
def airRoots (b : AirBody) : List Bytes := b.stmt.root0 :: b.pcs.head.map (fun f => f.2.2)

/-- `u_k = v^F_k + θ · v^C_k`: the uncommitted layer-0 coset of `H = F + θ C`. -/
def combine4 (v cv : F3x4) (theta : Fp3) : F3x4 :=
  (v.1.add (theta.mul cv.1), v.2.1.add (theta.mul cv.2.1), v.2.2.1.add (theta.mul cv.2.2.1),
    v.2.2.2.add (theta.mul cv.2.2.2))

/-- The layer-0 coset leaf index of a query: `pos mod (L / 4)`. -/
def airLeafIdx (O : Rel4Oracles) (lay : Rel4Layout) (tr : Fpn2Transcript) : Nat :=
  queryPos lay.pcs (tr.squeezeBytes O.hFs 8).1 % lay.pcs.quarterAt 0

/-- The fold state after layer 0: the first fused fold of the combined coset. -/
def airLayer0 (O : Rel4Oracles) (lay : Rel4Layout) (rs : List Fp3) (theta : Fp3)
    (iotaInv : UInt64) (tr : Fpn2Transcript) (q : AirQuery) : FoldState :=
  ⟨1, airLeafIdx O lay tr,
    fold4 (combine4 q.fLeaf.values q.cLeaf.values theta)
      (foldXInv lay.pcs 0 (airLeafIdx O lay tr)) iotaInv
      (rs.getD 0 Fp3.zero) (rs.getD 1 Fp3.zero)⟩

/-- One query: squeeze the position; the `F` leaf authenticates to `root_0` and the `C` leaf to
the pinned `root_circ` at the same index (else `E_MERKLE`); fold `F + θ C`; walk the committed
layers `1..T−1` (relation-3 `foldStep`: `E_MERKLE` / `E_FOLD`); the last fold equals `FINAL`
(else `E_FOLD`). `rs` are the fold challenges `r' = r_2..r_{n+1}`. -/
def queryStep4 (O : Rel4Oracles) (lay : Rel4Layout) (roots : List Bytes) (rootCirc : Bytes)
    (rs : List Fp3) (theta final : Fp3) (iotaInv : UInt64) (tr : Fpn2Transcript) (q : AirQuery) :
    Except RejectCode Fpn2Transcript :=
  if cosetRoot O (merkleLeaf O (Codec.fp3x4.enc q.fLeaf.values)) (airLeafIdx O lay tr)
      q.fLeaf.siblings = roots.getD 0 [] then
    if cosetRoot O (merkleLeaf O (Codec.fp3x4.enc q.cLeaf.values)) (airLeafIdx O lay tr)
        q.cLeaf.siblings = rootCirc then
      andThen (runList (foldStep O lay.pcs roots rs iotaInv)
          (airLayer0 O lay rs theta iotaInv tr q) q.rest) fun ff =>
        if ff.prev = final then .ok (tr.squeezeBytes O.hFs 8).2 else .error .E_FOLD
    else .error .E_MERKLE
  else .error .E_MERKLE

def QueryStepSpec4 (O : Rel4Oracles) (lay : Rel4Layout) (roots : List Bytes) (rootCirc : Bytes)
    (rs : List Fp3) (theta final : Fp3) (iotaInv : UInt64) (tr : Fpn2Transcript) (q : AirQuery)
    (tr' : Fpn2Transcript) : Prop :=
  cosetRoot O (merkleLeaf O (Codec.fp3x4.enc q.fLeaf.values)) (airLeafIdx O lay tr)
      q.fLeaf.siblings = roots.getD 0 [] ∧
    cosetRoot O (merkleLeaf O (Codec.fp3x4.enc q.cLeaf.values)) (airLeafIdx O lay tr)
      q.cLeaf.siblings = rootCirc ∧
    ∃ ff, RunList (FoldSpec O lay.pcs roots rs iotaInv)
        (airLayer0 O lay rs theta iotaInv tr q) q.rest ff ∧
      ff.prev = final ∧ tr' = (tr.squeezeBytes O.hFs 8).2

theorem queryStep4_iff (O : Rel4Oracles) (lay : Rel4Layout) (roots : List Bytes) (rootCirc : Bytes)
    (rs : List Fp3) (theta final : Fp3) (iotaInv : UInt64) (tr : Fpn2Transcript) (q : AirQuery)
    (tr' : Fpn2Transcript) :
    queryStep4 O lay roots rootCirc rs theta final iotaInv tr q = .ok tr' ↔
      QueryStepSpec4 O lay roots rootCirc rs theta final iotaInv tr q tr' := by
  unfold queryStep4 QueryStepSpec4
  constructor
  · intro h
    obtain ⟨hF, h⟩ := (ite_ok_iff _ _ _ _).mp h
    obtain ⟨hC, h⟩ := (ite_ok_iff _ _ _ _).mp h
    obtain ⟨ff, hr, h⟩ := (andThen_ok_iff _ _ _).mp h
    exact ⟨hF, hC, ff, (runList_iff _ _ (foldStep_iff O lay.pcs roots rs iotaInv) _ _ _).mp hr,
      (guard_ok_iff _ _ _ _).mp h⟩
  · rintro ⟨hF, hC, ff, hr, h⟩
    refine (ite_ok_iff _ _ _ _).mpr ⟨hF, (ite_ok_iff _ _ _ _).mpr ⟨hC, ?_⟩⟩
    exact (andThen_ok_iff _ _ _).mpr
      ⟨ff, (runList_iff _ _ (foldStep_iff O lay.pcs roots rs iotaInv) _ _ _).mpr hr,
        (guard_ok_iff _ _ _ _).mpr h⟩

/-- The transcript the queries are squeezed from: after `absorb_bytes(FINAL)`. -/
def queryStart4 (O : Rel4Oracles) (c : Rel4Container) (pf : PcsState) : Fpn2Transcript :=
  pf.tr.absorbBytes O.hFs (Codec.fp3.enc c.body.pcs.final)

/-- Normative check 19 over all `Q` queries, with `θ = r_0` and fold challenges `r_2..r_{n+1}`. -/
def queryPhase4 (O : Rel4Oracles) (lay : Rel4Layout) (c : Rel4Container) (pf : PcsState) :
    Except RejectCode Unit :=
  andThen (ofOption rel3IotaInv .E_FINAL) fun iotaInv =>
    andThen (runList
      (queryStep4 O lay (airRoots c.body) c.body.stmt.rootCirc (pf.rs.drop 2)
        (pf.rs.getD 0 Fp3.zero) c.body.pcs.final iotaInv)
      (queryStart4 O c pf) c.body.queries) fun _ => .ok ()

def QuerySpec4 (O : Rel4Oracles) (lay : Rel4Layout) (c : Rel4Container) (pf : PcsState) : Prop :=
  ∃ iotaInv trF, rel3IotaInv = some iotaInv ∧
    RunList
      (QueryStepSpec4 O lay (airRoots c.body) c.body.stmt.rootCirc (pf.rs.drop 2)
        (pf.rs.getD 0 Fp3.zero) c.body.pcs.final iotaInv)
      (queryStart4 O c pf) c.body.queries trF

theorem queryPhase4_iff (O : Rel4Oracles) (lay : Rel4Layout) (c : Rel4Container) (pf : PcsState) :
    queryPhase4 O lay c pf = .ok () ↔ QuerySpec4 O lay c pf := by
  unfold queryPhase4 QuerySpec4
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨iotaInv, hi, h⟩
    rw [andThen_ok_iff] at h
    obtain ⟨trF, hr, _⟩ := h
    exact ⟨iotaInv, trF, (ofOption_ok_iff _ _ _).mp hi,
      (runList_iff _ _ (queryStep4_iff O lay _ _ _ _ _ iotaInv) _ _ _).mp hr⟩
  · rintro ⟨iotaInv, trF, hi, hr⟩
    refine ⟨iotaInv, (ofOption_ok_iff _ _ _).mpr hi, ?_⟩
    rw [andThen_ok_iff]
    exact ⟨trF, (runList_iff _ _ (queryStep4_iff O lay _ _ _ _ _ iotaInv) _ _ _).mpr hr, rfl⟩

/-! ## The typed specification and the bytes verifier -/

/-- Normative checks 14–19 on a parsed container with a known layout. -/
def typedCheckRel4 (O : Rel4Oracles) (pub : PublicInput) (lay : Rel4Layout) (c : Rel4Container) :
    Except RejectCode Unit :=
  andThen (gkrPhase4 O pub c) fun g =>
    andThen (zcPhase O lay c g) fun zf =>
      andThen (pcsPhase4 O pub lay c g zf) fun pf => queryPhase4 O lay c pf

/-- Checks 14–19 against a profile (its layout exists by check 10). -/
def rel4Check (O : Rel4Oracles) (pub : PublicInput) (prof : Fpn2Profile) (c : Rel4Container) :
    Except RejectCode Unit :=
  andThen (ofOption (rel4Layout prof) .E_BACKEND_UNAVAILABLE) fun lay => typedCheckRel4 O pub lay c

/-- **Typed acceptance predicate of relation 4** on a parsed container: the profile has a
relation-4 layout; the GKR root is `(0, root_den ≠ 0)` and all masked layer equations hold
(wiring + range fractional sum); the zerocheck round sums hold from `λ_Z · Z_sum` and end in
`eq(z, r) · Gate(finals) + λ_Z · g_r`; the Basefold round sums hold from the batched claim `Y` and
end in `FINAL · W̃(θ, r_1, r')`; every query's `F` leaf is anchored in `root_0`, its `C` leaf in
`STMT.root_circ`, the committed layers in `root_1 .. root_{T−1}`, and the fused folds of `F + θ C`
chain and end in `FINAL`. -/
def Rel4TypedVerify (O : Rel4Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (c : Rel4Container) : Prop :=
  ∃ lay g zf pf, rel4Layout prof = some lay ∧ GkrSpec4 O pub c g ∧ ZcSpec O lay c g zf ∧
    PcsSpec4 O pub lay c g zf pf ∧ QuerySpec4 O lay c pf

/-- The executable checks 14–19 decide exactly the typed acceptance predicate. -/
theorem rel4Check_iff (O : Rel4Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (c : Rel4Container) : rel4Check O pub prof c = .ok () ↔ Rel4TypedVerify O pub prof c := by
  unfold rel4Check Rel4TypedVerify typedCheckRel4
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨lay, hlay, h⟩
    rw [andThen_ok_iff] at h
    obtain ⟨g, hg, h⟩ := h
    rw [andThen_ok_iff] at h
    obtain ⟨zf, hz, h⟩ := h
    rw [andThen_ok_iff] at h
    obtain ⟨pf, hp, hq⟩ := h
    exact ⟨lay, g, zf, pf, (ofOption_ok_iff _ _ _).mp hlay, (gkrPhase4_iff _ _ _ _).mp hg,
      (zcPhase_iff _ _ _ _ _).mp hz, (pcsPhase4_iff _ _ _ _ _ _ _).mp hp,
      (queryPhase4_iff _ _ _ _).mp hq⟩
  · rintro ⟨lay, g, zf, pf, hlay, hg, hz, hp, hq⟩
    refine ⟨lay, (ofOption_ok_iff _ _ _).mpr hlay, ?_⟩
    rw [andThen_ok_iff]
    refine ⟨g, (gkrPhase4_iff _ _ _ _).mpr hg, ?_⟩
    rw [andThen_ok_iff]
    refine ⟨zf, (zcPhase_iff _ _ _ _ _).mpr hz, ?_⟩
    rw [andThen_ok_iff]
    exact ⟨pf, (pcsPhase4_iff _ _ _ _ _ _ _).mpr hp, (queryPhase4_iff _ _ _ _).mpr hq⟩

/-- **`verifyBytesRel4`**: full verification of raw relation-4 container bytes against a pinned
profile and a pinned circuit commitment, in the normative order of spec §6. `allowReduced` is
`--allow-reduced-security`, `circuitRoot` is `--circuit-root` (`Policy::expected_circuit_root`).

* 1–5   header: length `≥ 64`, magic, version, flags, reserved (`parseHeader`);
* 6     `profile_id` is registered, else `E_PROFILE_ID`;
* 7     it is the pinned profile, else `E_PROFILE_PINNED`;
* 8     tier is `STANDARD` or `allowReduced`, else `E_REDUCED_SECURITY`;
* 9–12  exact length `C_p`, relation-4 layout, zero `PAD`, canonical fields (`parseBodyRel4`);
* 13    `circuitRoot` is given (else `E_CIRCUIT_UNPINNED`) and equals `STMT.root_circ` (else
        `E_CIRCUIT_PINNED`): relation 4 fails closed;
* 14–19 GKR root, masked GKR layers, zerocheck, batched Basefold round sums, final evaluation,
        dual layer-0 leaves, coset Merkle paths and fold consistency (`rel4Check`).

A profile of another relation is `E_BACKEND_UNAVAILABLE` here (`verifyBytesAny` in `Fpn2Rel5`
dispatches). -/
def verifyBytesRel4 (O : Rel4Oracles) (pub : PublicInput) (prof : Fpn2Profile) (allowReduced : Bool)
    (circuitRoot : Option Bytes) (raw : ByteArray) : Except RejectCode Rel4Claim :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    match lookupProfile h.profileId with
    | none => .error .E_PROFILE_ID
    | some found =>
      if found ≠ prof then .error .E_PROFILE_PINNED
      else if (prof.isProduction || allowReduced) = false then .error .E_REDUCED_SECURITY
      else
        match parseBodyRel4 prof (toBytes raw) with
        | .error e => .error e
        | .ok b =>
          match circuitRoot with
          | none => .error .E_CIRCUIT_UNPINNED
          | some pin =>
            if pin ≠ b.stmt.rootCirc then .error .E_CIRCUIT_PINNED
            else
              match rel4Check O pub prof ⟨h, b⟩ with
              | .error e => .error e
              | .ok _ => .ok (Rel4Container.acceptedClaim ⟨h, b⟩)

/-- `verify_bytes` restricted to relation 4, with the caller policy of
`core/src/verify.rs::Policy`: an optional pinned `expected_profile_id` (check 7), the
header-selected registered profile, and the circuit pin. -/
def verifyBytesRel4Policy (O : Rel4Oracles) (pub : PublicInput) (allowReduced : Bool)
    (expectedId : Option Bytes) (circuitRoot : Option Bytes) (raw : ByteArray) :
    Except RejectCode Rel4Claim :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    match lookupProfile h.profileId with
    | none => .error .E_PROFILE_ID
    | some found =>
      if expectedId.isSome ∧ expectedId ≠ some h.profileId then .error .E_PROFILE_PINNED
      else verifyBytesRel4 O pub found allowReduced circuitRoot raw

/-- Everything `verifyBytesRel4` established on an accepted input, as one case analysis. -/
theorem verifyBytesRel4_spec {O : Rel4Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {circuitRoot : Option Bytes} {raw : ByteArray} {claim : Rel4Claim}
    (h : verifyBytesRel4 O pub prof allowReduced circuitRoot raw = .ok claim) :
    ∃ c : Rel4Container,
      parseHeader raw = .ok c.header ∧ lookupProfile c.header.profileId = some prof ∧
      (prof.isProduction || allowReduced) = true ∧ parseBodyRel4 prof (toBytes raw) = .ok c.body ∧
      circuitRoot = some c.body.stmt.rootCirc ∧
      rel4Check O pub prof c = .ok () ∧ claim = c.acceptedClaim := by
  unfold verifyBytesRel4 at h
  cases hh : parseHeader raw with
  | error e => simp [hh] at h
  | ok hd =>
    simp only [hh] at h
    cases hl : lookupProfile hd.profileId with
    | none => simp [hl] at h
    | some found =>
      simp only [hl] at h
      split at h
      · cases h
      · rename_i hfound
        split at h
        · cases h
        · rename_i hpol
          cases hb : parseBodyRel4 prof (toBytes raw) with
          | error e => simp [hb] at h
          | ok b =>
            simp only [hb] at h
            cases circuitRoot with
            | none => cases h
            | some pin =>
              simp only at h
              split at h
              · cases h
              · rename_i hpin
                cases ht : rel4Check O pub prof ⟨hd, b⟩ with
                | error e => simp [ht] at h
                | ok u =>
                  simp only [ht] at h
                  injection h with h
                  refine ⟨⟨hd, b⟩, rfl, ?_, ?_, rfl, ?_, ?_, h.symm⟩
                  · rw [← Decidable.not_not.mp hfound]
                    exact hl
                  · cases hv : (prof.isProduction || allowReduced) with
                    | false => exact absurd hv hpol
                    | true => rfl
                  · rw [Decidable.not_not.mp hpin]
                  · cases u; exact ht

/-- **Refinement theorem (relation 4).** For every byte string `raw`, if `verifyBytesRel4`
accepts it then:

(a) `raw` parses, against the pinned and registered profile, to a container `c`; `raw` has
    exactly `C_p` bytes, is byte-for-byte the canonical serialization of `c`, and every byte
    after the last typed field is zero;
(b) the profile is production-tier, or the caller explicitly opted into reduced security;
(c) the caller pinned a circuit commitment and it is exactly the `root_circ` carried in `STMT`;
(d) the typed algebraic checks `Rel4TypedVerify` hold on `c`, and the returned claim is the
    statement carried by `c`. -/
theorem verifyBytesRel4_refines_typed {O : Rel4Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {circuitRoot : Option Bytes} {raw : ByteArray} {claim : Rel4Claim}
    (h : verifyBytesRel4 O pub prof allowReduced circuitRoot raw = .ok claim) :
    ∃ c : Rel4Container,
      (parseContainerRel4 prof raw = .ok c ∧ prof ∈ profileRegistry ∧
        raw.size = prof.containerBytes ∧ toBytes raw = encodeContainerRel4 prof c ∧
        ∃ lay, rel4Layout prof = some lay ∧ headerLen + lay.bodyUsed ≤ prof.containerBytes ∧
          ∀ i, headerLen + lay.bodyUsed ≤ i → i < raw.size → (toBytes raw)[i]? = some 0) ∧
      (prof.isProduction = true ∨ allowReduced = true) ∧
      circuitRoot = some c.body.stmt.rootCirc ∧
      (Rel4TypedVerify O pub prof c ∧ claim = c.acceptedClaim) := by
  obtain ⟨c, hh, hl, hpol, hb, hpin, ht, hc⟩ := verifyBytesRel4_spec h
  obtain ⟨hreg, hid⟩ := lookupProfile_spec hl
  have hparse : parseContainerRel4 prof raw = .ok c := by
    unfold parseContainerRel4
    simp [hh, hid, hb]
  obtain ⟨hsize, lay, hlay, hfit, _, hzero⟩ := parseContainerRel4_length_and_zero_pad hparse
  refine ⟨c, ⟨hparse, hreg, hsize, parseContainerRel4_encode hparse, lay, hlay, hfit, hzero⟩, ?_,
    hpin, (rel4Check_iff _ _ _ _).mp ht, hc⟩
  simpa [Bool.or_eq_true] using hpol

/-- **Completeness of the refinement.** A canonical container of a registered profile that
passes the tier policy, whose `STMT.root_circ` is the caller's pin and which satisfies
`Rel4TypedVerify`, is accepted, with its own statement. -/
theorem verifyBytesRel4_complete {O : Rel4Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {circuitRoot : Option Bytes} {raw : ByteArray} {c : Rel4Container}
    (hparse : parseContainerRel4 prof raw = .ok c)
    (hreg : lookupProfile prof.profileId = some prof)
    (hpol : prof.isProduction = true ∨ allowReduced = true)
    (hpin : circuitRoot = some c.body.stmt.rootCirc)
    (ht : Rel4TypedVerify O pub prof c) :
    verifyBytesRel4 O pub prof allowReduced circuitRoot raw = .ok c.acceptedClaim := by
  obtain ⟨hh, hid, hb⟩ := parseContainerRel4_spec hparse
  have hpol' : (prof.isProduction || allowReduced) = true := by
    simpa [Bool.or_eq_true] using hpol
  subst hpin
  unfold verifyBytesRel4
  simp [hh, hid, hreg, hpol', hb, (rel4Check_iff O pub prof c).mpr ht]

/-- **No silent downgrade.** Without the explicit opt-in, only a production-tier profile can be
accepted. -/
theorem verifyBytesRel4_downgrade_resistant {O : Rel4Oracles} {pub : PublicInput}
    {prof : Fpn2Profile} {circuitRoot : Option Bytes} {raw : ByteArray} {claim : Rel4Claim}
    (h : verifyBytesRel4 O pub prof false circuitRoot raw = .ok claim) :
    prof.isProduction = true := by
  obtain ⟨_, _, hpol, _⟩ := verifyBytesRel4_refines_typed h
  rcases hpol with hp | hf
  · exact hp
  · cases hf

/-- A `REDUCED` relation-4 profile is rejected on every input unless the caller opts in. -/
theorem verifyBytesRel4_rejects_reduced (O : Rel4Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (circuitRoot : Option Bytes) (raw : ByteArray) (hred : prof.isProduction = false)
    (claim : Rel4Claim) : verifyBytesRel4 O pub prof false circuitRoot raw ≠ .ok claim := by
  intro h
  rw [verifyBytesRel4_downgrade_resistant h] at hred
  cases hred

/-- **No second accepted encoding.** Two accepted byte strings carrying the same parsed
container are equal; in particular flipping any padding byte of an accepted proof is rejected. -/
theorem verifyBytesRel4_injective_on_accepted {O : Rel4Oracles} {pub : PublicInput}
    {prof : Fpn2Profile} {a₁ a₂ : Bool} {r₁ r₂ : Option Bytes} {raw₁ raw₂ : ByteArray}
    {claim₁ claim₂ : Rel4Claim} {c : Rel4Container}
    (h₁ : verifyBytesRel4 O pub prof a₁ r₁ raw₁ = .ok claim₁)
    (h₂ : verifyBytesRel4 O pub prof a₂ r₂ raw₂ = .ok claim₂)
    (hc₁ : parseContainerRel4 prof raw₁ = .ok c) (hc₂ : parseContainerRel4 prof raw₂ = .ok c) :
    raw₁ = raw₂ ∧ claim₁ = claim₂ ∧ r₁ = r₂ := by
  obtain ⟨c₁, ⟨hp₁, _⟩, _, hpin₁, _, hcl₁⟩ := verifyBytesRel4_refines_typed h₁
  obtain ⟨c₂, ⟨hp₂, _⟩, _, hpin₂, _, hcl₂⟩ := verifyBytesRel4_refines_typed h₂
  rw [hc₁] at hp₁
  rw [hc₂] at hp₂
  injection hp₁ with e₁
  injection hp₂ with e₂
  subst e₁ e₂
  exact ⟨parseContainerRel4_injective_on_accepted hc₁ hc₂, hcl₁.trans hcl₂.symm,
    hpin₁.trans hpin₂.symm⟩

/-- **Mandatory circuit pin.** Relation 4 fails closed: an accepted container was verified
against a caller-supplied circuit commitment, and that commitment is the `root_circ` of the
accepted claim. In particular no input is accepted without a pin, and one container is never
accepted under two different pins. -/
theorem verifyBytesRel4_requires_circuit_pin {O : Rel4Oracles} {pub : PublicInput}
    {prof : Fpn2Profile} {allowReduced : Bool} {circuitRoot : Option Bytes} {raw : ByteArray}
    {claim : Rel4Claim}
    (h : verifyBytesRel4 O pub prof allowReduced circuitRoot raw = .ok claim) :
    circuitRoot = some claim.rootCirc ∧ circuitRoot ≠ none := by
  obtain ⟨c, _, _, _, _, hpin, _, hc⟩ := verifyBytesRel4_spec h
  have hroot : claim.rootCirc = c.body.stmt.rootCirc := by rw [hc]; rfl
  exact ⟨by rw [hpin, hroot], by rw [hpin]; intro h; cases h⟩

/-- An unpinned relation-4 verification never accepts. -/
theorem verifyBytesRel4_unpinned_rejects (O : Rel4Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (allowReduced : Bool) (raw : ByteArray) (claim : Rel4Claim) :
    verifyBytesRel4 O pub prof allowReduced none raw ≠ .ok claim := by
  intro h
  exact (verifyBytesRel4_requires_circuit_pin h).2 rfl

/-- An acceptance under the caller policy is an acceptance under a registered profile, which is
the pinned one whenever a pin was given. -/
theorem verifyBytesRel4Policy_spec {O : Rel4Oracles} {pub : PublicInput} {allowReduced : Bool}
    {expectedId : Option Bytes} {circuitRoot : Option Bytes} {raw : ByteArray} {claim : Rel4Claim}
    (h : verifyBytesRel4Policy O pub allowReduced expectedId circuitRoot raw = .ok claim) :
    ∃ prof, prof ∈ profileRegistry ∧
      verifyBytesRel4 O pub prof allowReduced circuitRoot raw = .ok claim ∧
      ∀ id, expectedId = some id → prof.profileId = id := by
  unfold verifyBytesRel4Policy at h
  cases hh : parseHeader raw with
  | error e => simp [hh] at h
  | ok hd =>
    simp only [hh] at h
    cases hl : lookupProfile hd.profileId with
    | none => simp [hl] at h
    | some found =>
      simp only [hl] at h
      split at h
      · cases h
      · rename_i hpin
        refine ⟨found, (lookupProfile_spec hl).1, h, ?_⟩
        intro id hid
        rw [(lookupProfile_spec hl).2]
        subst hid
        by_cases he : some id = some hd.profileId
        · injection he with he
          exact he.symm
        · exact absurd ⟨rfl, he⟩ hpin

end FlashProveNext

#print axioms FlashProveNext.rel4Layout_foldedAirV1
#print axioms FlashProveNext.rel4Layout_foldedAirN12
#print axioms FlashProveNext.rel4Layout_foldedAirN20
#print axioms FlashProveNext.rel4Layout_disjoint_rel3
#print axioms FlashProveNext.airBody_enc
#print axioms FlashProveNext.zcFinals_enc
#print axioms FlashProveNext.parseBodyRel4_spec
#print axioms FlashProveNext.parseContainerRel4_spec
#print axioms FlashProveNext.parseContainerRel4_encode
#print axioms FlashProveNext.parseContainerRel4_length_and_zero_pad
#print axioms FlashProveNext.parseContainerRel4_injective_on_accepted
#print axioms FlashProveNext.gkrPhase4_iff
#print axioms FlashProveNext.zcRoundStep_iff
#print axioms FlashProveNext.zcPhase_iff
#print axioms FlashProveNext.pcsPhase4_iff
#print axioms FlashProveNext.queryStep4_iff
#print axioms FlashProveNext.queryPhase4_iff
#print axioms FlashProveNext.rel4Check_iff
#print axioms FlashProveNext.verifyBytesRel4_spec
#print axioms FlashProveNext.verifyBytesRel4_refines_typed
#print axioms FlashProveNext.verifyBytesRel4_complete
#print axioms FlashProveNext.verifyBytesRel4_downgrade_resistant
#print axioms FlashProveNext.verifyBytesRel4_rejects_reduced
#print axioms FlashProveNext.verifyBytesRel4_injective_on_accepted
#print axioms FlashProveNext.verifyBytesRel4_requires_circuit_pin
#print axioms FlashProveNext.verifyBytesRel4_unpinned_rejects
#print axioms FlashProveNext.verifyBytesRel4Policy_spec
