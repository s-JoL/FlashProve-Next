/-
Executable `verifyBytesRel3` for `FPN2` relation-3 containers (`FOLDED_GKR_LOOKUP`,
`specs/11-wire-format.md` §5.4, §6) and its refinement to a typed specification. Stage 2 of the Lean
verifier (M4); Lean core only. The definitions follow `core/src/folded_gkr.rs` step by step.

1. `Rel3Layout` : the fixed-size layout, a function of `(n_vars, n_queries)` (`s`, `quarter`,
   `folds`, `depth`, `layerLen`, `gkrLen`, `pcsLen`, `queryLen`, `bodyUsed`, `dataBlock`, `libraSlot`);
2. `Fp3.neg`, `Fp3.inv`, `eqEval`, `evalOnBits`, `rel3Terms`, `weightsEval`, `fold4`, `cosetRoot` :
   the `F_{p^3}` helpers, the batched linear claim of §5.4.4, the fused coset-4 fold and the coset
   Merkle path of §5.4.5;
3. `Rel3Oracles` : the two hash functions the verifier calls and Lean does not re-execute
   (`H_fs` = BLAKE2b-512 personalized `"FPN2-FS-v2"`, `B256_M` = BLAKE2b-256 personalized
   `"FPN2-MERKLE-v1"`); the `0x00` / `0x01` leaf / node domain bytes are fixed in Lean;
4. `Codec.fam`, `Codec.foldedBody`, `parseBodyRel3`, `parseContainerRel3` : canonical decoding of
   the §5.4.7 layout (normative checks 9–12), with `parseContainerRel3_encode`,
   `parseContainerRel3_length_and_zero_pad`, `parseContainerRel3_injective_on_accepted`;
5. `gkrPhase`, `pcsPhase`, `queryPhase`, `typedCheckRel3`, `verifyBytesRel3` : normative checks
   13–17 in the transcript schedule of §5.4.8, with the spec reject codes;
6. `RunList`, `GkrSpec`, `PcsSpec`, `QuerySpec`, `Rel3TypedVerify` : the declarative typed statement
   (root shape, masked GKR layer equations, Basefold round sums, final evaluation, coset Merkle
   anchors and fold consistency along the Fiat–Shamir schedule) on a parsed container;
7. `verifyBytesRel3_refines_typed`, `verifyBytesRel3_complete`,
   `verifyBytesRel3_downgrade_resistant`, `verifyBytesRel3_injective_on_accepted`,
   `foldedBody_enc`, `layerBytes_sections` : accepted bytes are a canonical, fixed-length,
   zero-padded container of a registered relation-3 profile on which `Rel3TypedVerify` holds, and
   conversely; no silent downgrade; no second accepted encoding; the absorbed sections are exactly
   the wire bytes.

Scope. `Rel3TypedVerify` is the algebraic acceptance predicate of relation 3 over abstract hash
oracles. This module proves that the bytes verifier decides exactly that predicate on exactly the
canonical encodings; it does not prove that the predicate is a sound or zero-knowledge argument
(later stages, see `THEOREM.md`).
-/
import FlashProveNext.Fpn2Wire
import FlashProveNext.Fpn2Transcript
import FlashProveNext.VerifyBytes

namespace FlashProveNext

/-! ## Layout (spec §5.4.1, §5.4.7; `core/src/folded_gkr.rs::Layout`) -/

/-- Fixed-size relation-3 layout, a pure function of `(n_vars, n_queries)`. -/
structure Rel3Layout where
  nVars : Nat
  nQueries : Nat
  deriving DecidableEq, Repr

namespace Rel3Layout

/-- `Layout::of`: `sc_degree = 3`, `n` even with `8 ≤ n ≤ 28` (Goldilocks 2-adicity: `8·2^n = 2^(n+3) | P−1`), `1 ≤ Q ≤ 4096`. -/
def ofProfile (p : Fpn2Profile) : Option Rel3Layout :=
  if p.scDegree = 3 ∧ 8 ≤ p.nVars ∧ p.nVars % 2 = 0 ∧ p.nVars ≤ 28 ∧ 1 ≤ p.nQueries ∧
      p.nQueries ≤ 4096 then
    some ⟨p.nVars, p.nQueries⟩
  else none

/-- `s = n − 3`: the GKR tree has `2^(s+1)` leaves. -/
def s (lay : Rel3Layout) : Nat := lay.nVars - 3

/-- `Q4 = 2^(s−2)`: the number of private queries. -/
def quarter (lay : Rel3Layout) : Nat := 2 ^ (lay.s - 2)

/-- `T = n / 2` fused coset-4 folds. -/
def folds (lay : Rel3Layout) : Nat := lay.nVars / 2

/-- Merkle depth of folded layer `t`: `n + 1 − 2t`. -/
def depth (lay : Rel3Layout) (t : Nat) : Nat := lay.nVars + 1 - 2 * t

/-- Wire size of GKR layer `k ≥ 1`: `G_sum ‖ k rounds ‖ children ‖ g_r`. -/
def layerLen (k : Nat) : Nat := 24 + 96 * k + 120

def gkrLen (lay : Rel3Layout) : Nat :=
  48 + 96 + ((List.range lay.s).map (fun i => layerLen (i + 1))).foldl (· + ·) 0

def pcsLen (lay : Rel3Layout) : Nat := 24 + 72 * lay.nVars + 32 * (lay.folds - 1) + 24

def queryLen (lay : Rel3Layout) : Nat :=
  ((List.range lay.folds).map (fun t => 96 + 32 * lay.depth t)).foldl (· + ·) 0

/-- Bytes used by the body before zero padding. -/
def bodyUsed (lay : Rel3Layout) : Nat :=
  64 + lay.gkrLen + lay.pcsLen + lay.nQueries * lay.queryLen

/-- Codeword length `L = 8 · 2^n` (rate 1/8). -/
def codeLen (lay : Rel3Layout) : Nat := 8 * 2 ^ lay.nVars

/-- `L_t / 4 = L / 4^(t+1)`: the number of coset leaves of folded layer `t`. -/
def quarterAt (lay : Rel3Layout) (t : Nat) : Nat := lay.codeLen / 2 ^ (2 * t + 2)

/-- Cell bits addressed by data coordinates, low to high: `0, 2, 4, 5, …, n − 1`. -/
def freeBits (lay : Rel3Layout) : List Nat :=
  0 :: 2 :: (List.range (lay.nVars - 4)).map (· + 4)

/-- `(fixed bits, free bits)` of the data-coordinate block `[off, off + 2^k)`. -/
def dataBlock (lay : Rel3Layout) (off k : Nat) : List (Nat × Nat) × List Nat :=
  ([(1, 1), (3, 1)] ++
      (List.range (lay.freeBits.length - k)).map
        (fun i => (lay.freeBits.getD (k + i) 0, (off / 2 ^ (k + i)) % 2)),
    lay.freeBits.take k)

/-- Cell index of Libra coefficient `g_{k,i,e}`: slot `2k(k−1) + 4i + e` deposited into the free
bits of the subcube `{b_1 = b_3 = 0}`. -/
def libraSlot (lay : Rel3Layout) (k i e : Nat) : Nat :=
  (List.range lay.freeBits.length).foldl
    (fun idx j =>
      idx + (((2 * k * (k - 1) + 4 * i + e) / 2 ^ j) % 2) * 2 ^ (lay.freeBits.getD j 0))
    0

end Rel3Layout

/-- The layout of a relation-3 profile; `none` for any other relation or unwired parameters
(normative check 10). -/
def rel3Layout (p : Fpn2Profile) : Option Rel3Layout :=
  if p.relation = .foldedGkrLookup then Rel3Layout.ofProfile p else none

/-- `fpn2-folded-gkr-v1`: `s = 9`, `Q4 = 128`, `T = 6`, body 158 960 bytes, ending at 159 024. -/
theorem rel3Layout_foldedGkrV1 :
    rel3Layout foldedGkrV1 = some ⟨12, 72⟩ ∧
      (Rel3Layout.s ⟨12, 72⟩, Rel3Layout.quarter ⟨12, 72⟩, Rel3Layout.folds ⟨12, 72⟩,
        Rel3Layout.bodyUsed ⟨12, 72⟩) = (9, 128, 6, 158960) ∧
      headerLen + Rel3Layout.bodyUsed ⟨12, 72⟩ ≤ foldedGkrV1.containerBytes := by
  decide

/-- `fpn2-folded-gkr-n8`: body 5 392 bytes, ending at 5 456. -/
theorem rel3Layout_foldedGkrN8 :
    rel3Layout foldedGkrN8 = some ⟨8, 2⟩ ∧ Rel3Layout.bodyUsed ⟨8, 2⟩ = 5392 ∧
      headerLen + Rel3Layout.bodyUsed ⟨8, 2⟩ ≤ foldedGkrN8.containerBytes := by
  decide

/-! ## `F_{p^3}` helpers (spec §1, §5.4.4, §5.4.5) -/

namespace Fp3

def neg (a : Fp3) : Fp3 := zero.sub a

/-- Inverse via the norm map (`core/src/field.rs::Fp3::inv`); `none` for zero. -/
def inv (a : Fp3) : Option Fp3 :=
  let t0 := fpSub (fpMul a.a0 a.a0) (fpMul 2 (fpMul a.a1 a.a2))
  let t1 := fpSub (fpMul 2 (fpMul a.a2 a.a2)) (fpMul a.a0 a.a1)
  let t2 := fpSub (fpMul a.a1 a.a1) (fpMul a.a0 a.a2)
  match fpInv (fpAdd (fpMul a.a0 t0) (fpMul 2 (fpAdd (fpMul a.a2 t1) (fpMul a.a1 t2)))) with
  | none => none
  | some ni => some ⟨fpMul t0 ni, fpMul t1 ni, fpMul t2 ni⟩

end Fp3

/-- Three and four consecutive `Fp3` wire fields. -/
abbrev F3x3 := Fp3 × Fp3 × Fp3

abbrev F3x4 := Fp3 × Fp3 × Fp3 × Fp3

def list3 (v : F3x3) : List Fp3 := [v.1, v.2.1, v.2.2]

def list4 (v : F3x4) : List Fp3 := [v.1, v.2.1, v.2.2.1, v.2.2.2]

/-- `vals[slot]` for `slot < 4`. -/
def sel4 (v : F3x4) : Nat → Fp3
  | 0 => v.1
  | 1 => v.2.1
  | 2 => v.2.2.1
  | _ => v.2.2.2

/-- `eq1(a, b) = a b + (1 − a)(1 − b)`. -/
def eq1 (a b : Fp3) : Fp3 := (a.mul b).add ((Fp3.one.sub a).mul (Fp3.one.sub b))

/-- `eq(a, b) = Π_i eq1(a_i, b_i)` over the common prefix. -/
def eqEval (a b : List Fp3) : Fp3 :=
  (a.zip b).foldl (fun acc p => acc.mul (eq1 p.1 p.2)) Fp3.one

/-- Bit `i` of `v` as a field element. -/
def bit3 (v i : Nat) : Fp3 := if (v / 2 ^ i) % 2 = 1 then Fp3.one else Fp3.zero

/-- `eq(bits_n(idx), r)`: the multilinear Lagrange basis of cell `idx` evaluated at `r`. -/
def evalOnBits (idx n : Nat) (r : List Fp3) : Fp3 :=
  eqEval ((List.range n).map (bit3 idx)) r

/-- `a + r (b − a)`. -/
def mix (a b r : Fp3) : Fp3 := a.add (r.mul (b.sub a))

/-- One GKR layer's Libra record: layer index, round challenges, `g_r`. -/
structure LibraEntry where
  k : Nat
  rs : List Fp3
  gR : Fp3

/-- One subcube block `c · eq(z, b_free)` of the weight vector. -/
structure Rel3Block where
  fixed : List (Nat × Nat)
  free : List Nat
  coef : Fp3
  z : List Fp3

/-- The batched linear claim of §5.4.4 (`folded_gkr.rs::terms`). -/
structure Rel3Terms where
  numPub : Fp3
  denPub : Fp3
  blocks : List Rel3Block
  sparse : List (Nat × Fp3)
  coef : Fp3

def mkBlock (lay : Rel3Layout) (off k : Nat) (c : Fp3) (z : List Fp3) : Rel3Block :=
  ⟨(lay.dataBlock off k).1, (lay.dataBlock off k).2, c, z⟩

/-- The sparse cells `coef · r_i^e` (`e < 4`) of one Libra layer. -/
def libraCells (lay : Rel3Layout) (k : Nat) (coef : Fp3) (rs : List Fp3) : List (Nat × Fp3) :=
  (List.range rs.length).flatMap fun i =>
    let r := rs.getD i Fp3.zero
    let p0 := Fp3.one
    let p1 := p0.mul r
    let p2 := p1.mul r
    let p3 := p2.mul r
    [(lay.libraSlot k i 0, coef.mul p0), (lay.libraSlot k i 1, coef.mul p1),
     (lay.libraSlot k i 2, coef.mul p2), (lay.libraSlot k i 3, coef.mul p3)]

/-- The blinding block of mask point `z` on the subcube with the given fixed bits. -/
def maskBlock (lay : Rel3Layout) (coef z : Fp3) (fixed : List (Nat × Nat)) : Rel3Block :=
  let c := fixed.foldl (fun c f => c.mul (if f.2 = 1 then z else Fp3.one.sub z)) coef
  let free := (List.range lay.nVars).filter (fun b => !(fixed.any (fun f => f.1 == b)))
  ⟨fixed, free, c, List.replicate free.length z⟩

/-- `terms(lay, point, beta, libra, mu)`. -/
def rel3Terms (lay : Rel3Layout) (point : List Fp3) (beta : Fp3) (libra : List LibraEntry)
    (mu : Fp3) : Rel3Terms :=
  let s := lay.s
  let rho0 := point.getD 0 Fp3.zero
  let rho := point.drop 1
  let low := rho.take (s - 2)
  let lowp := rho.take (s - 3)
  let ra := rho.take (s - 1)
  let rhoA := rho.getD (s - 3) Fp3.zero
  let rhoB := rho.getD (s - 2) Fp3.zero
  let rhoC := rho.getD (s - 1) Fp3.zero
  let pair := Fp3.one.sub (rhoA.add rhoA)
  let eqq := fun (qd : Nat) => eqEval [rhoB, rhoC] [bit3 qd 0, bit3 qd 1]
  let real := Fp3.one.sub rho0
  let tExt := (List.range low.length).foldl
    (fun acc i => acc.add ((low.getD i Fp3.zero).scale (fpOfNat (2 ^ i)))) Fp3.zero
  let numPub := real.mul (eqq 0)
  let denPub := real.mul (((eqq 0).mul beta).add ((eqq 2).mul (beta.sub tExt)))
  let q4 := lay.quarter
  let h := q4 / 2
  let offA := 2 ^ s
  let offD := offA + 2 ^ (s - 1)
  let dataBlocks :=
    [mkBlock lay q4 (s - 2) (real.mul (eqq 2)).neg low,
     mkBlock lay (2 * q4) (s - 3) ((real.mul (eqq 1)).mul pair) lowp,
     mkBlock lay (3 * q4) (s - 3) ((real.mul (eqq 3)).mul pair) lowp,
     mkBlock lay offA (s - 1) (rho0.mul (Fp3.one.sub (rhoC.add rhoC))) ra,
     mkBlock lay 0 (s - 2) (mu.mul (real.mul (eqq 0)).neg) low,
     mkBlock lay (2 * q4 + h) (s - 3) (mu.mul (real.mul (eqq 1))) lowp,
     mkBlock lay (3 * q4 + h) (s - 3) (mu.mul (real.mul (eqq 3))) lowp,
     mkBlock lay offD (s - 1) (mu.mul rho0) ra]
  let lib := libra.foldl
    (fun (acc : List (Nat × Fp3) × Fp3) (en : LibraEntry) =>
      (acc.1 ++ libraCells lay en.k acc.2 en.rs, acc.2.mul mu))
    ([], mu.mul mu)
  let coef := lib.2
  let maskBlocks := [(2 : UInt64), 3].flatMap fun zv =>
    [maskBlock lay coef (Fp3.ofFp zv) [(1, 0)], maskBlock lay coef (Fp3.ofFp zv) [(1, 1), (3, 0)]]
  ⟨numPub, denPub, dataBlocks ++ maskBlocks, lib.1, coef⟩

/-- `Y = (num − num_pub) + μ (den − den_pub) + Σ_k μ^(k+1) g_r(k) + μ^(s+2) z_blind`. -/
def batchedClaim (tm : Rel3Terms) (num den mu : Fp3) (libra : List LibraEntry) (zBlind : Fp3) :
    Fp3 :=
  ((libra.foldl
      (fun (acc : Fp3 × Fp3) (en : LibraEntry) => (acc.1.add (acc.2.mul en.gR), acc.2.mul mu))
      ((num.sub tm.numPub).add (mu.mul (den.sub tm.denPub)), mu.mul mu)).1).add
    (tm.coef.mul zBlind)

/-- `w̃(r)`: the weight multilinear evaluated at the Basefold challenges. -/
def weightsEval (lay : Rel3Layout) (tm : Rel3Terms) (r : List Fp3) : Fp3 :=
  let acc := tm.blocks.foldl
    (fun acc b =>
      acc.add (b.fixed.foldl
        (fun term f =>
          term.mul (if f.2 = 1 then r.getD f.1 Fp3.zero else Fp3.one.sub (r.getD f.1 Fp3.zero)))
        (b.coef.mul (eqEval b.z (b.free.map (fun i => r.getD i Fp3.zero))))))
    Fp3.zero
  tm.sparse.foldl (fun acc c => acc.add (c.2.mul (evalOnBits c.1 lay.nVars r))) acc

/-- Fused coset-4 fold (`folded_gkr.rs::fold4`): `f_e = x^(−e)/4 · Σ_k ι^(−ek) v_k`, result
`f_0 + r_a f_1 + r_b f_2 + r_a r_b f_3`. -/
def fold4 (v : F3x4) (xInv iotaInv : UInt64) (ra rb : Fp3) : Fp3 :=
  let inv4 := (fpInv 4).getD 0
  let a := v.1.sub v.2.2.1
  let tb := (v.2.1.sub v.2.2.2).scale iotaInv
  let s0 := (v.1.add v.2.1).add (v.2.2.1.add v.2.2.2)
  let s1 := a.add tb
  let s2 := (v.1.add v.2.2.1).sub (v.2.1.add v.2.2.2)
  let s3 := a.sub tb
  let c1 := fpMul inv4 xInv
  let c2 := fpMul c1 xInv
  let c3 := fpMul c2 xInv
  (((s0.scale inv4).add (ra.mul (s1.scale c1))).add (rb.mul (s2.scale c2))).add
    ((ra.mul rb).mul (s3.scale c3))

/-! ## Hash oracles and coset Merkle paths (spec §2, §5.4.5) -/

/-- The hash functions the relation-3 verifier calls. They stay abstract in Stage 2. -/
structure Rel3Oracles where
  /-- `H_fs`: BLAKE2b-512, personal `"FPN2-FS-v2"`. -/
  hFs : ByteArray → ByteArray
  /-- `B256_M`: BLAKE2b-256, personal `"FPN2-MERKLE-v1" ‖ 0^2`. -/
  hMerkle : ByteArray → ByteArray

/-- `leaf = B256_M(0x00 ‖ 4 × enc_Fp3)`. -/
def merkleLeaf (O : Rel3Oracles) (raw : Bytes) : Bytes := toBytes (O.hMerkle (ofBytes (0 :: raw)))

/-- `node = B256_M(0x01 ‖ left ‖ right)`. -/
def merkleNode (O : Rel3Oracles) (l r : Bytes) : Bytes :=
  toBytes (O.hMerkle (ofBytes (1 :: (l ++ r))))

/-- Walk a coset authentication path from the leaf level upward (`folded_gkr.rs::merkle_ok`
compares the result with the root). -/
def cosetRoot (O : Rel3Oracles) : Bytes → Nat → List Bytes → Bytes
  | node, _, [] => node
  | node, idx, sib :: rest =>
    cosetRoot O (if idx % 2 = 1 then merkleNode O sib node else merkleNode O node sib) (idx / 2) rest

/-! ## Typed body and its codec (spec §5.4.7, normative check 12) -/

def decRawN (n : Nat) (l : Bytes) : Except RejectCode (Bytes × Bytes) :=
  if (l.take n).length = n then .ok (l.take n, l.drop n) else .error .E_LENGTH

theorem decRawN_sound (n : Nat) (l a rest : Bytes) (h : decRawN n l = .ok (a, rest)) :
    l = a ++ rest := by
  unfold decRawN at h
  split at h
  · injection h with h
    injection h with ha hr
    subst ha hr
    rw [List.take_append_drop]
  · cases h

/-- `n` opaque bytes (`Bytes32` for `n = 32`), read without measuring the whole remaining input. -/
def Codec.rawN (n : Nat) : Codec Bytes := ⟨decRawN n, id, decRawN_sound n⟩

/-- An index-dependent family of decoders sharing one encoder: items `i, i + 1, …, i + n − 1`. -/
def decFam {α : Type} (c : Nat → Codec α) : Nat → Nat → Bytes → Except RejectCode (List α × Bytes)
  | _, 0, l => .ok ([], l)
  | i, n + 1, l =>
    match (c i).dec l with
    | .error e => .error e
    | .ok (a, r) =>
      match decFam c (i + 1) n r with
      | .error e => .error e
      | .ok (as, r') => .ok (a :: as, r')

theorem decFam_sound {α : Type} (c : Nat → Codec α) (enc : α → Bytes)
    (henc : ∀ i a, (c i).enc a = enc a) (i n : Nat) (l : Bytes) (as : List α) (rest : Bytes)
    (h : decFam c i n l = .ok (as, rest)) : l = as.flatMap enc ++ rest := by
  induction n generalizing i l as rest with
  | zero =>
    simp only [decFam] at h
    injection h with h
    injection h with ha hr
    subst ha hr
    rfl
  | succ n ih =>
    simp only [decFam] at h
    cases h1 : (c i).dec l with
    | error e => simp [h1] at h
    | ok x =>
      obtain ⟨a, r⟩ := x
      simp only [h1] at h
      cases h2 : decFam c (i + 1) n r with
      | error e => simp [h2] at h
      | ok y =>
        obtain ⟨as', r'⟩ := y
        simp only [h2] at h
        injection h with h
        injection h with ha hr
        subst ha hr
        have e1 := (c i).sound _ _ _ h1
        have e2 := ih _ _ _ _ h2
        rw [e1, e2, List.flatMap_cons, henc, List.append_assoc]

/-- Items `start, …, start + n − 1` of a codec family whose encoders all equal `enc` (GKR layer
`k` has `k` rounds; folded layer `t` has `n + 1 − 2t` siblings). -/
def Codec.fam {α : Type} (c : Nat → Codec α) (enc : α → Bytes) (henc : ∀ i a, (c i).enc a = enc a)
    (start n : Nat) : Codec (List α) :=
  ⟨decFam c start n, fun as => as.flatMap enc, decFam_sound c enc henc start n⟩

def Codec.fp3x3 : Codec F3x3 := Codec.fp3.pair (Codec.fp3.pair Codec.fp3)

def Codec.fp3x4 : Codec F3x4 := Codec.fp3.pair Codec.fp3x3

/-- GKR layer `k ≥ 1` in wire order. -/
structure GkrLayer where
  gSum : Fp3
  rounds : List F3x4
  children : F3x4
  gR : Fp3

/-- `G_sum ‖ k rounds × g(0..3) ‖ children n_l, d_l, n_r, d_r ‖ g_r`. -/
def Codec.gkrLayer (k : Nat) : Codec GkrLayer :=
  (Codec.fp3.pair ((Codec.fp3x4.rep k).pair (Codec.fp3x4.pair Codec.fp3))).map
    (fun t => ⟨t.1, t.2.1, t.2.2.1, t.2.2.2⟩) (fun L => (L.gSum, L.rounds, L.children, L.gR))
    (fun _ => rfl)

/-- The wire bytes of one GKR layer (independent of the layer index). -/
def layerBytes (L : GkrLayer) : Bytes := (Codec.gkrLayer 0).enc L

/-- One coset opening: 4 values and the sibling digests, leaf level first. -/
structure CosetOpening where
  values : F3x4
  siblings : List Bytes

def Codec.coset (d : Nat) : Codec CosetOpening :=
  (Codec.fp3x4.pair ((Codec.rawN 32).rep d)).map
    (fun t => ⟨t.1, t.2⟩) (fun o => (o.values, o.siblings)) (fun _ => rfl)

def cosetBytes (o : CosetOpening) : Bytes := (Codec.coset 0).enc o

/-- One fused fold `t < T − 1`: round `2t`, round `2t + 1`, `root_{t+1}`. -/
def Codec.pcsFold : Codec (F3x3 × F3x3 × Bytes) :=
  Codec.fp3x3.pair (Codec.fp3x3.pair (Codec.rawN 32))

/-- Decoded body of a relation-3 (`FOLDED_GKR_LOOKUP`) container, in wire order. -/
structure FoldedBody where
  root0 : Bytes
  nS : Bytes
  rootNum : Fp3
  rootDen : Fp3
  layer0 : F3x4
  layers : List GkrLayer
  zBlind : Fp3
  pcsHead : List (F3x3 × F3x3 × Bytes)
  pcsLast : F3x3 × F3x3
  final : Fp3
  queries : List (List CosetOpening)

/-- A fully parsed relation-3 container. -/
structure Rel3Container where
  header : Fpn2Header
  body : FoldedBody

def Codec.gkrLayers (lay : Rel3Layout) : Codec (List GkrLayer) :=
  Codec.fam Codec.gkrLayer layerBytes (fun _ _ => rfl) 1 lay.s

/-- One query: for every folded layer `t < T`, 4 coset values and `n + 1 − 2t` siblings. -/
def Codec.query (lay : Rel3Layout) : Codec (List CosetOpening) :=
  Codec.fam (fun t => Codec.coset (lay.depth t)) cosetBytes (fun _ _ => rfl) 0 lay.folds

/-- The whole relation-3 body, sections back-to-back in wire order:
`STMT ‖ GKR_ROOT ‖ GKR_L0 ‖ GKR_L1..s ‖ PCS ‖ QUERIES`. Every `Fp3` field is decoded canonically in
wire order, so the first non-canonical limb determines `E_NONCANONICAL`. -/
def Codec.foldedBody (lay : Rel3Layout) : Codec FoldedBody :=
  ((Codec.rawN 32).pair ((Codec.rawN 32).pair (Codec.fp3.pair (Codec.fp3.pair (Codec.fp3x4.pair
    ((Codec.gkrLayers lay).pair (Codec.fp3.pair ((Codec.pcsFold.rep (lay.folds - 1)).pair
      ((Codec.fp3x3.pair Codec.fp3x3).pair (Codec.fp3.pair
        ((Codec.query lay).rep lay.nQueries))))))))))).map
    (fun t => ⟨t.1, t.2.1, t.2.2.1, t.2.2.2.1, t.2.2.2.2.1, t.2.2.2.2.2.1, t.2.2.2.2.2.2.1,
      t.2.2.2.2.2.2.2.1, t.2.2.2.2.2.2.2.2.1, t.2.2.2.2.2.2.2.2.2.1, t.2.2.2.2.2.2.2.2.2.2⟩)
    (fun b => (b.root0, b.nS, b.rootNum, b.rootDen, b.layer0, b.layers, b.zBlind, b.pcsHead,
      b.pcsLast, b.final, b.queries))
    (fun _ => rfl)

/-- The 64 `STMT` wire bytes `root_0 ‖ N_s`. -/
def stmtBytesRel3 (b : FoldedBody) : Bytes := b.root0 ++ b.nS

/-- The 48 `GKR_ROOT` wire bytes. -/
def rootBytesRel3 (b : FoldedBody) : Bytes := Codec.fp3.enc b.rootNum ++ Codec.fp3.enc b.rootDen

/-- The wire bytes of one fused fold `t < T − 1`. -/
def pcsFoldBytes (f : F3x3 × F3x3 × Bytes) : Bytes :=
  Codec.fp3x3.enc f.1 ++ (Codec.fp3x3.enc f.2.1 ++ f.2.2)

def queryBytes (q : List CosetOpening) : Bytes := q.flatMap cosetBytes

/-- Canonical body serializer (`folded_gkr.rs::encode`): the sections in wire order. It does not
depend on the layout. The transcript absorbs exactly these byte strings (`STMT`, `GKR_ROOT`, the
pieces of each GKR layer, `z_blind`, each PCS round and fold root, `FINAL`), and the coset values
hashed into Merkle leaves are the `QUERIES` bytes. -/
def encodeBodyRel3 (b : FoldedBody) : Bytes :=
  stmtBytesRel3 b ++ (rootBytesRel3 b ++ (Codec.fp3x4.enc b.layer0 ++
    (b.layers.flatMap layerBytes ++ (Codec.fp3.enc b.zBlind ++
      (b.pcsHead.flatMap pcsFoldBytes ++ ((Codec.fp3x3.enc b.pcsLast.1 ++
        Codec.fp3x3.enc b.pcsLast.2) ++ (Codec.fp3.enc b.final ++
          b.queries.flatMap queryBytes)))))))

/-- The wire bytes of one GKR layer are the chunks the verifier absorbs, in order: `G_sum`, each
round, then `children ‖ g_r`. -/
theorem layerBytes_sections (L : GkrLayer) :
    layerBytes L = Codec.fp3.enc L.gSum ++ (L.rounds.flatMap Codec.fp3x4.enc ++
      (Codec.fp3x4.enc L.children ++ Codec.fp3.enc L.gR)) := by
  have hr : encRep Codec.fp3x4 L.rounds = L.rounds.flatMap Codec.fp3x4.enc :=
    encRep_append_flatMap _ _
  rw [← hr]
  rfl

/-- The encoder of the body codec is the section-wise serializer, for every layout: what
`Codec.foldedBody` accepts is the concatenation of the sections the verifier absorbs. -/
theorem foldedBody_enc (lay : Rel3Layout) (b : FoldedBody) :
    (Codec.foldedBody lay).enc b = encodeBodyRel3 b := by
  have hh : encRep Codec.pcsFold b.pcsHead = b.pcsHead.flatMap pcsFoldBytes :=
    encRep_append_flatMap _ _
  have hq : encRep (Codec.query lay) b.queries = b.queries.flatMap queryBytes :=
    encRep_append_flatMap _ _
  unfold encodeBodyRel3
  rw [← hh, ← hq]
  simp [Codec.foldedBody, Codec.map, Codec.pair, Codec.rep, Codec.rawN, Codec.gkrLayers,
    Codec.fam, stmtBytesRel3, rootBytesRel3, List.append_assoc]

/-- Normative checks 9–12 for relation 3: exact length `C_p`, wired backend, all-zero `PAD`,
canonical decoding of every typed field in wire order. -/
def parseBodyRel3 (prof : Fpn2Profile) (l : Bytes) : Except RejectCode FoldedBody :=
  if l.length ≠ prof.containerBytes then .error .E_LENGTH
  else
    match rel3Layout prof with
    | none => .error .E_BACKEND_UNAVAILABLE
    | some lay =>
      if prof.containerBytes < headerLen + lay.bodyUsed then .error .E_LENGTH
      else if (l.drop headerLen).drop lay.bodyUsed ≠
          List.replicate (prof.containerBytes - headerLen - lay.bodyUsed) 0 then .error .E_PADDING
      else
        match (Codec.foldedBody lay).dec ((l.drop headerLen).take lay.bodyUsed) with
        | .error e => .error e
        | .ok (b, rest) => if rest = [] then .ok b else .error .E_LENGTH

/-- Strict parser against a pinned profile: header checks 1–5, the header names exactly the
pinned `profile_id` (else `E_PROFILE_PINNED`), then body checks 9–12. -/
def parseContainerRel3 (prof : Fpn2Profile) (raw : ByteArray) : Except RejectCode Rel3Container :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    if h.profileId ≠ prof.profileId then .error .E_PROFILE_PINNED
    else
      match parseBodyRel3 prof (toBytes raw) with
      | .error e => .error e
      | .ok b => .ok ⟨h, b⟩

/-- Zero padding length of a relation-3 profile (`0` when the relation is not wired). -/
def padLenRel3 (prof : Fpn2Profile) : Nat :=
  match rel3Layout prof with
  | none => 0
  | some lay => prof.containerBytes - headerLen - lay.bodyUsed

/-- Canonical serializer (inverse of `parseContainerRel3` on accepted inputs): `C_p` bytes. -/
def encodeContainerRel3 (prof : Fpn2Profile) (c : Rel3Container) : Bytes :=
  encodeHeader c.header ++ (encodeBodyRel3 c.body ++ List.replicate (padLenRel3 prof) 0)

theorem parseBodyRel3_spec {prof : Fpn2Profile} {l : Bytes} {b : FoldedBody}
    (hp : parseBodyRel3 prof l = .ok b) :
    l.length = prof.containerBytes ∧ ∃ lay, rel3Layout prof = some lay ∧
      headerLen + lay.bodyUsed ≤ prof.containerBytes ∧
      (l.drop headerLen).take lay.bodyUsed = encodeBodyRel3 b ∧
      (l.drop headerLen).drop lay.bodyUsed =
        List.replicate (prof.containerBytes - headerLen - lay.bodyUsed) 0 := by
  unfold parseBodyRel3 at hp
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
              have hs := (Codec.foldedBody lay).sound _ _ _ hdec
              rw [List.append_nil, foldedBody_enc] at hs
              exact ⟨Decidable.not_not.mp hlen, lay, hlay, Nat.le_of_not_lt hfit, hs,
                Decidable.not_not.mp hpad⟩
            · cases hp

theorem parseContainerRel3_spec {prof : Fpn2Profile} {raw : ByteArray} {c : Rel3Container}
    (hp : parseContainerRel3 prof raw = .ok c) :
    parseHeader raw = .ok c.header ∧ c.header.profileId = prof.profileId ∧
      parseBodyRel3 prof (toBytes raw) = .ok c.body := by
  unfold parseContainerRel3 at hp
  cases hh : parseHeader raw with
  | error e => simp [hh] at hp
  | ok h =>
    simp only [hh] at hp
    split at hp
    · cases hp
    · rename_i hid
      cases hb : parseBodyRel3 prof (toBytes raw) with
      | error e => simp [hb] at hp
      | ok b =>
        simp only [hb] at hp
        injection hp with hp
        subst hp
        exact ⟨rfl, Decidable.not_not.mp hid, rfl⟩

/-- **Canonical wire representation.** An accepted byte string is exactly the canonical
serialization of the container it parses to. -/
theorem parseContainerRel3_encode {prof : Fpn2Profile} {raw : ByteArray} {c : Rel3Container}
    (hp : parseContainerRel3 prof raw = .ok c) : toBytes raw = encodeContainerRel3 prof c := by
  obtain ⟨hh, _, hb⟩ := parseContainerRel3_spec hp
  obtain ⟨_, lay, hlay, _, htake, hdrop⟩ := parseBodyRel3_spec hb
  have hhead := parseHeaderBytes_encode hh
  have e1 : toBytes raw = (toBytes raw).take headerLen ++ (toBytes raw).drop headerLen :=
    (List.take_append_drop _ _).symm
  have e2 : (toBytes raw).drop headerLen =
      ((toBytes raw).drop headerLen).take lay.bodyUsed ++
        ((toBytes raw).drop headerLen).drop lay.bodyUsed :=
    (List.take_append_drop _ _).symm
  have hpadLen : padLenRel3 prof = prof.containerBytes - headerLen - lay.bodyUsed := by
    simp [padLenRel3, hlay]
  rw [e1, e2, hhead, htake, hdrop, encodeContainerRel3, hpadLen]

/-- An accepted container has exactly `C_p` bytes and every padding byte is zero. -/
theorem parseContainerRel3_length_and_zero_pad {prof : Fpn2Profile} {raw : ByteArray}
    {c : Rel3Container} (hp : parseContainerRel3 prof raw = .ok c) :
    raw.size = prof.containerBytes ∧ ∃ lay, rel3Layout prof = some lay ∧
      headerLen + lay.bodyUsed ≤ prof.containerBytes ∧
      (toBytes raw).drop (headerLen + lay.bodyUsed) =
        List.replicate (prof.containerBytes - headerLen - lay.bodyUsed) 0 ∧
      ∀ i, headerLen + lay.bodyUsed ≤ i → i < raw.size → (toBytes raw)[i]? = some 0 := by
  obtain ⟨_, _, hb⟩ := parseContainerRel3_spec hp
  obtain ⟨hlen, lay, hlay, hfit, _, hdrop⟩ := parseBodyRel3_spec hb
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
theorem parseContainerRel3_injective_on_accepted {prof : Fpn2Profile} {raw₁ raw₂ : ByteArray}
    {c : Rel3Container} (h₁ : parseContainerRel3 prof raw₁ = .ok c)
    (h₂ : parseContainerRel3 prof raw₂ = .ok c) : raw₁ = raw₂ :=
  toBytes_injective ((parseContainerRel3_encode h₁).trans (parseContainerRel3_encode h₂).symm)

/-! ## Sequencing combinators and their specifications -/

/-- Sequencing of checks: the first error wins. -/
def andThen {α β : Type} (x : Except RejectCode α) (f : α → Except RejectCode β) :
    Except RejectCode β :=
  match x with
  | .error e => .error e
  | .ok a => f a

def ofOption {α : Type} (o : Option α) (e : RejectCode) : Except RejectCode α :=
  match o with
  | none => .error e
  | some a => .ok a

/-- Run a step function over a list, threading the state; the first error wins. -/
def runList {σ α : Type} (step : σ → α → Except RejectCode σ) : σ → List α → Except RejectCode σ
  | s, [] => .ok s
  | s, a :: as => andThen (step s a) fun s' => runList step s' as

/-- The relational counterpart of `runList`: every step satisfies `R` and the states chain. -/
inductive RunList {σ α : Type} (R : σ → α → σ → Prop) : σ → List α → σ → Prop where
  | nil (s : σ) : RunList R s [] s
  | cons (s : σ) (a : α) (s' : σ) (as : List α) (sF : σ) :
      R s a s' → RunList R s' as sF → RunList R s (a :: as) sF

theorem andThen_ok_iff {α β : Type} (x : Except RejectCode α) (f : α → Except RejectCode β)
    (b : β) : andThen x f = .ok b ↔ ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e =>
    constructor
    · intro h
      cases h
    · rintro ⟨a, h, _⟩
      cases h
  | ok a =>
    constructor
    · intro h
      exact ⟨a, rfl, h⟩
    · rintro ⟨a', h, hf⟩
      injection h with h
      subst h
      exact hf

theorem ofOption_ok_iff {α : Type} (o : Option α) (e : RejectCode) (a : α) :
    ofOption o e = .ok a ↔ o = some a := by
  cases o with
  | none =>
    constructor
    · intro h
      cases h
    · intro h
      cases h
  | some v =>
    constructor
    · intro h
      injection h with h
      rw [h]
    · intro h
      injection h with h
      rw [h]
      rfl

theorem ok_eq_ok_iff {α : Type} (x a : α) :
    (Except.ok x : Except RejectCode α) = .ok a ↔ a = x := by
  constructor
  · intro h
    injection h with h
    exact h.symm
  · intro h
    rw [h]

/-- A guarded continuation succeeds iff the guard holds and the continuation succeeds. -/
theorem ite_ok_iff {α : Type} (c : Prop) [Decidable c] (x : Except RejectCode α) (e : RejectCode)
    (a : α) : (if c then x else .error e) = .ok a ↔ c ∧ x = .ok a := by
  by_cases hc : c
  · rw [if_pos hc]
    exact ⟨fun h => ⟨hc, h⟩, fun h => h.2⟩
  · rw [if_neg hc]
    constructor
    · intro h
      cases h
    · intro h
      exact absurd h.1 hc

theorem guard_ok_iff {α : Type} (c : Prop) [Decidable c] (x : α) (e : RejectCode) (a : α) :
    (if c then (Except.ok x : Except RejectCode α) else .error e) = .ok a ↔ c ∧ a = x := by
  rw [ite_ok_iff, ok_eq_ok_iff]

/-- `runList` decides `RunList` whenever each step decides its relation. -/
theorem runList_iff {σ α : Type} (step : σ → α → Except RejectCode σ) (R : σ → α → σ → Prop)
    (hstep : ∀ s a s', step s a = .ok s' ↔ R s a s') (s : σ) (as : List α) (sF : σ) :
    runList step s as = .ok sF ↔ RunList R s as sF := by
  constructor
  · intro h
    induction as generalizing s with
    | nil =>
      have h' : (Except.ok s : Except RejectCode σ) = .ok sF := h
      rw [ok_eq_ok_iff] at h'
      rw [h']
      exact .nil _
    | cons a as ih =>
      have h' : andThen (step s a) (fun s' => runList step s' as) = .ok sF := h
      rw [andThen_ok_iff] at h'
      obtain ⟨s', hs, hrest⟩ := h'
      exact .cons _ _ _ _ _ ((hstep _ _ _).mp hs) (ih _ hrest)
  · intro h
    induction h with
    | nil s => rfl
    | cons s a s' as sF hR _ ih =>
      show andThen (step s a) (fun s' => runList step s' as) = .ok sF
      rw [andThen_ok_iff]
      exact ⟨s', (hstep _ _ _).mpr hR, ih⟩

/-! ## Check 13–14: GKR root and masked GKR layers (spec §5.4.4, §5.4.8) -/

/-- What an accepted relation-3 container asserts: the profile and the `STMT` fields. -/
structure Rel3Claim where
  profileId : Bytes
  root0 : Bytes
  nS : Bytes
  deriving DecidableEq, Repr

def Rel3Container.acceptedClaim (c : Rel3Container) : Rel3Claim :=
  ⟨c.header.profileId, c.body.root0, c.body.nS⟩

/-- The transcript after `init`, `absorb_bytes(header)`, `absorb_bytes(STMT)`. -/
def rel3Start (O : Rel3Oracles) (pub : PublicInput) (c : Rel3Container) : Fpn2Transcript :=
  ((Fpn2Transcript.init O.hFs pub.transcriptLabel).absorbBytes O.hFs
    (encodeHeader c.header)).absorbBytes O.hFs (stmtBytesRel3 c.body)

/-- The lookup challenge `β`, squeezed once `root_0` is bound. -/
def rel3Beta (O : Rel3Oracles) (pub : PublicInput) (c : Rel3Container) : Fp3 :=
  ((rel3Start O pub c).squeezeFp3 O.hFs).1

/-- Running state of one layer sumcheck. -/
structure RoundState where
  tr : Fpn2Transcript
  target : Fp3
  rs : List Fp3

/-- Running state of the GKR: transcript, point, claim `(num, den)`, Libra records, layer index. -/
structure GkrState where
  tr : Fpn2Transcript
  point : List Fp3
  num : Fp3
  den : Fp3
  libra : List LibraEntry
  k : Nat

/-- The state after `β` and `absorb_bytes(GKR_ROOT)`: empty point, claim = the root fraction. -/
def gkrInit (O : Rel3Oracles) (pub : PublicInput) (c : Rel3Container) : GkrState :=
  ⟨((rel3Start O pub c).squeezeFp3 O.hFs).2.absorbBytes O.hFs (rootBytesRel3 c.body), [],
    c.body.rootNum, c.body.rootDen, [], 0⟩

/-- The challenge `r` of a round: squeezed after absorbing the round's raw bytes. -/
def gkrRoundChal (O : Rel3Oracles) (s : RoundState) (g : F3x4) : Fp3 :=
  ((s.tr.absorbBytes O.hFs (Codec.fp3x4.enc g)).squeezeFp3 O.hFs).1

def gkrRoundNext (O : Rel3Oracles) (s : RoundState) (g : F3x4) : RoundState :=
  ⟨((s.tr.absorbBytes O.hFs (Codec.fp3x4.enc g)).squeezeFp3 O.hFs).2,
    lagrangeEval (list4 g) (gkrRoundChal O s g), s.rs ++ [gkrRoundChal O s g]⟩

/-- One layer-sumcheck round: `g(0) + g(1) = target`, else `E_GKR`. -/
def gkrRoundStep (O : Rel3Oracles) (s : RoundState) (g : F3x4) : Except RejectCode RoundState :=
  if g.1.add g.2.1 = s.target then .ok (gkrRoundNext O s g) else .error .E_GKR

def GkrRoundSpec (O : Rel3Oracles) (s : RoundState) (g : F3x4) (s' : RoundState) : Prop :=
  g.1.add g.2.1 = s.target ∧ s' = gkrRoundNext O s g

theorem gkrRoundStep_iff (O : Rel3Oracles) (s : RoundState) (g : F3x4) (s' : RoundState) :
    gkrRoundStep O s g = .ok s' ↔ GkrRoundSpec O s g s' :=
  guard_ok_iff _ _ _ _

/-- `(n_l d_r + n_r d_l) + α · d_l d_r` for children `n_l, d_l, n_r, d_r`. -/
def childrenClaim (ch : F3x4) (alpha : Fp3) : Fp3 :=
  ((ch.1.mul ch.2.2.2).add (ch.2.2.1.mul ch.2.1)).add (alpha.mul (ch.2.1.mul ch.2.2.2))

/-- Layer 0 (no rounds, no Libra mask): the batching challenge `α`. -/
def layer0Alpha (O : Rel3Oracles) (s : GkrState) : Fp3 := (s.tr.squeezeFp3 O.hFs).1

def layer0Next (O : Rel3Oracles) (s : GkrState) (ch : F3x4) : GkrState :=
  let tr := (((s.tr.squeezeFp3 O.hFs).2.absorbBytes O.hFs (Codec.fp3x4.enc ch)).squeezeFp3 O.hFs)
  ⟨tr.2, [tr.1], mix ch.1 ch.2.2.1 tr.1, mix ch.2.1 ch.2.2.2 tr.1, s.libra, s.k + 1⟩

/-- Layer 0: `num + α den = eq(pt, []) · (children claim)`, else `E_GKR`. -/
def gkrLayer0 (O : Rel3Oracles) (s : GkrState) (ch : F3x4) : Except RejectCode GkrState :=
  if s.num.add ((layer0Alpha O s).mul s.den) =
      (eqEval s.point []).mul (childrenClaim ch (layer0Alpha O s)) then
    .ok (layer0Next O s ch)
  else .error .E_GKR

def GkrLayer0Spec (O : Rel3Oracles) (s : GkrState) (ch : F3x4) (s' : GkrState) : Prop :=
  s.num.add ((layer0Alpha O s).mul s.den) =
      (eqEval s.point []).mul (childrenClaim ch (layer0Alpha O s)) ∧
    s' = layer0Next O s ch

theorem gkrLayer0_iff (O : Rel3Oracles) (s : GkrState) (ch : F3x4) (s' : GkrState) :
    gkrLayer0 O s ch = .ok s' ↔ GkrLayer0Spec O s ch s' :=
  guard_ok_iff _ _ _ _

/-- Layer `k ≥ 1`: `α`, then `absorb_bytes(G_sum)` and the Libra challenge `λ`. -/
def layerAlpha (O : Rel3Oracles) (s : GkrState) : Fp3 := (s.tr.squeezeFp3 O.hFs).1

def layerLamTr (O : Rel3Oracles) (s : GkrState) (L : GkrLayer) : Fp3 × Fpn2Transcript :=
  ((s.tr.squeezeFp3 O.hFs).2.absorbBytes O.hFs (Codec.fp3.enc L.gSum)).squeezeFp3 O.hFs

/-- The layer sumcheck starts from `num + α den + λ G_sum`. -/
def layerRoundsInit (O : Rel3Oracles) (s : GkrState) (L : GkrLayer) : RoundState :=
  ⟨(layerLamTr O s L).2,
    (s.num.add ((layerAlpha O s).mul s.den)).add ((layerLamTr O s L).1.mul L.gSum), []⟩

/-- `eq(pt, r) · (children claim) + λ g_r`. -/
def layerExpect (O : Rel3Oracles) (s : GkrState) (L : GkrLayer) (rf : RoundState) : Fp3 :=
  ((eqEval s.point rf.rs).mul (childrenClaim L.children (layerAlpha O s))).add
    ((layerLamTr O s L).1.mul L.gR)

/-- After `absorb_bytes(children ‖ g_r)`: squeeze `branch`, extend the point, mix the children,
record the Libra evaluation. -/
def layerNext (O : Rel3Oracles) (s : GkrState) (L : GkrLayer) (rf : RoundState) : GkrState :=
  let tr := (rf.tr.absorbBytes O.hFs
    (Codec.fp3x4.enc L.children ++ Codec.fp3.enc L.gR)).squeezeFp3 O.hFs
  ⟨tr.2, tr.1 :: rf.rs, mix L.children.1 L.children.2.2.1 tr.1,
    mix L.children.2.1 L.children.2.2.2 tr.1, s.libra ++ [⟨s.k, rf.rs, L.gR⟩], s.k + 1⟩

/-- One masked GKR layer `k ≥ 1`: the round sums and the layer equation, else `E_GKR`. -/
def gkrLayerStep (O : Rel3Oracles) (s : GkrState) (L : GkrLayer) : Except RejectCode GkrState :=
  andThen (runList (gkrRoundStep O) (layerRoundsInit O s L) L.rounds) fun rf =>
    if rf.target = layerExpect O s L rf then .ok (layerNext O s L rf) else .error .E_GKR

def GkrLayerSpec (O : Rel3Oracles) (s : GkrState) (L : GkrLayer) (s' : GkrState) : Prop :=
  ∃ rf, RunList (GkrRoundSpec O) (layerRoundsInit O s L) L.rounds rf ∧
    rf.target = layerExpect O s L rf ∧ s' = layerNext O s L rf

theorem gkrLayerStep_iff (O : Rel3Oracles) (s : GkrState) (L : GkrLayer) (s' : GkrState) :
    gkrLayerStep O s L = .ok s' ↔ GkrLayerSpec O s L s' := by
  unfold gkrLayerStep GkrLayerSpec
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨rf, hr, h⟩
    exact ⟨rf, (runList_iff _ _ (gkrRoundStep_iff O) _ _ _).mp hr, (guard_ok_iff _ _ _ _).mp h⟩
  · rintro ⟨rf, hr, h⟩
    exact ⟨rf, (runList_iff _ _ (gkrRoundStep_iff O) _ _ _).mpr hr, (guard_ok_iff _ _ _ _).mpr h⟩

/-- Normative checks 13 and 14: `root_num = 0 ∧ root_den ≠ 0` (else `E_GKR_ROOT`), then layer 0
and the masked layers `1..s`. -/
def gkrPhase (O : Rel3Oracles) (pub : PublicInput) (c : Rel3Container) :
    Except RejectCode GkrState :=
  if c.body.rootNum = Fp3.zero ∧ c.body.rootDen ≠ Fp3.zero then
    andThen (gkrLayer0 O (gkrInit O pub c) c.body.layer0) fun s0 =>
      runList (gkrLayerStep O) s0 c.body.layers
  else .error .E_GKR_ROOT

/-- The GKR part of the typed statement: the root is `(0, root_den ≠ 0)` and every layer equation
holds along the Fiat–Shamir schedule, ending in state `g` (point `ρ`, leaf claim, Libra records). -/
def GkrSpec (O : Rel3Oracles) (pub : PublicInput) (c : Rel3Container) (g : GkrState) : Prop :=
  (c.body.rootNum = Fp3.zero ∧ c.body.rootDen ≠ Fp3.zero) ∧
    ∃ s0, GkrLayer0Spec O (gkrInit O pub c) c.body.layer0 s0 ∧
      RunList (GkrLayerSpec O) s0 c.body.layers g

theorem gkrPhase_iff (O : Rel3Oracles) (pub : PublicInput) (c : Rel3Container) (g : GkrState) :
    gkrPhase O pub c = .ok g ↔ GkrSpec O pub c g := by
  unfold gkrPhase GkrSpec
  rw [ite_ok_iff, andThen_ok_iff]
  constructor
  · rintro ⟨hroot, s0, h0, hl⟩
    exact ⟨hroot, s0, (gkrLayer0_iff _ _ _ _).mp h0,
      (runList_iff _ _ (gkrLayerStep_iff O) _ _ _).mp hl⟩
  · rintro ⟨hroot, s0, h0, hl⟩
    exact ⟨hroot, s0, (gkrLayer0_iff _ _ _ _).mpr h0,
      (runList_iff _ _ (gkrLayerStep_iff O) _ _ _).mpr hl⟩

/-! ## Check 15–16: fused Basefold sumcheck and final evaluation (spec §5.4.5, §5.4.8) -/

/-- Running state of the Basefold sumcheck: transcript, claim `y`, challenges so far. -/
structure PcsState where
  tr : Fpn2Transcript
  y : Fp3
  rs : List Fp3

/-- The PCS section in wire order: a degree-2 round `h(0), h(1), h(2)` or a fold root. -/
inductive PcsOp where
  | round (h : F3x3)
  | root (r : Bytes)

/-- For `t < T − 1`: round `2t`, round `2t + 1`, `root_{t+1}`; then the last two rounds. -/
def pcsOps (b : FoldedBody) : List PcsOp :=
  b.pcsHead.flatMap (fun f => [.round f.1, .round f.2.1, .root f.2.2]) ++
    [.round b.pcsLast.1, .round b.pcsLast.2]

def pcsRoundChal (O : Rel3Oracles) (s : PcsState) (h : F3x3) : Fp3 :=
  ((s.tr.absorbBytes O.hFs (Codec.fp3x3.enc h)).squeezeFp3 O.hFs).1

def pcsRoundNext (O : Rel3Oracles) (s : PcsState) (h : F3x3) : PcsState :=
  ⟨((s.tr.absorbBytes O.hFs (Codec.fp3x3.enc h)).squeezeFp3 O.hFs).2,
    lagrangeEval (list3 h) (pcsRoundChal O s h), s.rs ++ [pcsRoundChal O s h]⟩

def pcsRootNext (O : Rel3Oracles) (s : PcsState) (r : Bytes) : PcsState :=
  ⟨s.tr.absorbBytes O.hFs r, s.y, s.rs⟩

/-- One PCS step: a round with `h(0) + h(1) = y` (else `E_SUMCHECK`), or `absorb_bytes(root)`. -/
def pcsStep (O : Rel3Oracles) (s : PcsState) : PcsOp → Except RejectCode PcsState
  | .round h => if h.1.add h.2.1 = s.y then .ok (pcsRoundNext O s h) else .error .E_SUMCHECK
  | .root r => .ok (pcsRootNext O s r)

def PcsOpSpec (O : Rel3Oracles) (s : PcsState) : PcsOp → PcsState → Prop
  | .round h, s' => h.1.add h.2.1 = s.y ∧ s' = pcsRoundNext O s h
  | .root r, s' => s' = pcsRootNext O s r

theorem pcsStep_iff (O : Rel3Oracles) (s : PcsState) (op : PcsOp) (s' : PcsState) :
    pcsStep O s op = .ok s' ↔ PcsOpSpec O s op s' := by
  cases op with
  | round h => exact guard_ok_iff _ _ _ _
  | root r => exact ok_eq_ok_iff _ _

/-- `absorb_bytes(z_blind)`, then the batching challenge `μ`. -/
def pcsMuTr (O : Rel3Oracles) (c : Rel3Container) (g : GkrState) : Fp3 × Fpn2Transcript :=
  (g.tr.absorbBytes O.hFs (Codec.fp3.enc c.body.zBlind)).squeezeFp3 O.hFs

def pcsTerms (O : Rel3Oracles) (pub : PublicInput) (lay : Rel3Layout) (c : Rel3Container)
    (g : GkrState) : Rel3Terms :=
  rel3Terms lay g.point (rel3Beta O pub c) g.libra (pcsMuTr O c g).1

/-- The Basefold sumcheck starts from the batched linear claim `Y` of §5.4.4. -/
def pcsInit (O : Rel3Oracles) (pub : PublicInput) (lay : Rel3Layout) (c : Rel3Container)
    (g : GkrState) : PcsState :=
  ⟨(pcsMuTr O c g).2,
    batchedClaim (pcsTerms O pub lay c g) g.num g.den (pcsMuTr O c g).1 g.libra c.body.zBlind, []⟩

/-- Normative checks 15 and 16: the `n` Basefold round sums (else `E_SUMCHECK`), then
`y = FINAL · w̃(r)` (else `E_FINAL`). -/
def pcsPhase (O : Rel3Oracles) (pub : PublicInput) (lay : Rel3Layout) (c : Rel3Container)
    (g : GkrState) : Except RejectCode PcsState :=
  andThen (runList (pcsStep O) (pcsInit O pub lay c g) (pcsOps c.body)) fun pf =>
    if pf.y = c.body.final.mul (weightsEval lay (pcsTerms O pub lay c g) pf.rs) then .ok pf
    else .error .E_FINAL

def PcsSpec (O : Rel3Oracles) (pub : PublicInput) (lay : Rel3Layout) (c : Rel3Container)
    (g : GkrState) (pf : PcsState) : Prop :=
  RunList (PcsOpSpec O) (pcsInit O pub lay c g) (pcsOps c.body) pf ∧
    pf.y = c.body.final.mul (weightsEval lay (pcsTerms O pub lay c g) pf.rs)

theorem pcsPhase_iff (O : Rel3Oracles) (pub : PublicInput) (lay : Rel3Layout) (c : Rel3Container)
    (g : GkrState) (pf : PcsState) :
    pcsPhase O pub lay c g = .ok pf ↔ PcsSpec O pub lay c g pf := by
  unfold pcsPhase PcsSpec
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨p, hr, h⟩
    obtain ⟨hfin, hp⟩ := (guard_ok_iff _ _ _ _).mp h
    subst hp
    exact ⟨(runList_iff _ _ (pcsStep_iff O) _ _ _).mp hr, hfin⟩
  · rintro ⟨hr, hfin⟩
    exact ⟨pf, (runList_iff _ _ (pcsStep_iff O) _ _ _).mpr hr, (guard_ok_iff _ _ _ _).mpr ⟨hfin, rfl⟩⟩

/-! ## Check 17: coset Merkle paths and fold consistency (spec §5.4.5, §5.4.8) -/

/-- Running state inside one query: layer index, position, previous folded value. -/
structure FoldState where
  t : Nat
  pos : Nat
  prev : Fp3

/-- `root_0` from `STMT`, then the committed fold roots `root_1 .. root_{T−1}`. -/
def rel3Roots (b : FoldedBody) : List Bytes := b.root0 :: b.pcsHead.map (fun f => f.2.2)

/-- `pos = int_le(squeeze_bytes(8)) mod (L / 4)`. -/
def queryPos (lay : Rel3Layout) (o : Bytes) : Nat := leNat o % (lay.codeLen / 4)

/-- `ω = 7^((P−1)/L)`. -/
def rel3Omega (lay : Rel3Layout) : UInt64 := fpPow 7 ((goldilocksP.toNat - 1) / lay.codeLen)

/-- `ι^(−1)` for `ι = 7^((P−1)/4)`. -/
def rel3IotaInv : Option UInt64 := fpInv (fpPow 7 ((goldilocksP.toNat - 1) / 4))

/-- `x^(−1) = ω_t^(−j)` with `ω_t = ω^(4^t)`. -/
def foldXInv (lay : Rel3Layout) (t j : Nat) : UInt64 :=
  fpPow (fpPow (rel3Omega lay) (2 ^ (2 * t)))
    ((goldilocksP.toNat - 1 - j) % (goldilocksP.toNat - 1))

def foldNext (lay : Rel3Layout) (rs : List Fp3) (iotaInv : UInt64) (s : FoldState)
    (o : CosetOpening) : FoldState :=
  ⟨s.t + 1, s.pos % lay.quarterAt s.t,
    fold4 o.values (foldXInv lay s.t (s.pos % lay.quarterAt s.t)) iotaInv
      (rs.getD (2 * s.t) Fp3.zero) (rs.getD (2 * s.t + 1) Fp3.zero)⟩

/-- One folded layer of one query: the coset Merkle path to `root_t` (else `E_MERKLE`), then for
`t > 0` the value at the previous slot equals the previous fold (else `E_FOLD`). -/
def foldStep (O : Rel3Oracles) (lay : Rel3Layout) (roots : List Bytes) (rs : List Fp3)
    (iotaInv : UInt64) (s : FoldState) (o : CosetOpening) : Except RejectCode FoldState :=
  if cosetRoot O (merkleLeaf O (Codec.fp3x4.enc o.values)) (s.pos % lay.quarterAt s.t) o.siblings =
      roots.getD s.t [] then
    if s.t = 0 ∨ sel4 o.values (s.pos / lay.quarterAt s.t) = s.prev then
      .ok (foldNext lay rs iotaInv s o)
    else .error .E_FOLD
  else .error .E_MERKLE

def FoldSpec (O : Rel3Oracles) (lay : Rel3Layout) (roots : List Bytes) (rs : List Fp3)
    (iotaInv : UInt64) (s : FoldState) (o : CosetOpening) (s' : FoldState) : Prop :=
  cosetRoot O (merkleLeaf O (Codec.fp3x4.enc o.values)) (s.pos % lay.quarterAt s.t) o.siblings =
      roots.getD s.t [] ∧
    (s.t = 0 ∨ sel4 o.values (s.pos / lay.quarterAt s.t) = s.prev) ∧
    s' = foldNext lay rs iotaInv s o

theorem foldStep_iff (O : Rel3Oracles) (lay : Rel3Layout) (roots : List Bytes) (rs : List Fp3)
    (iotaInv : UInt64) (s : FoldState) (o : CosetOpening) (s' : FoldState) :
    foldStep O lay roots rs iotaInv s o = .ok s' ↔ FoldSpec O lay roots rs iotaInv s o s' := by
  unfold foldStep FoldSpec
  rw [ite_ok_iff, guard_ok_iff]

/-- One query: squeeze the position, walk the folded layers, and require the last fold to equal
`FINAL` (else `E_FOLD`). -/
def queryStep (O : Rel3Oracles) (lay : Rel3Layout) (roots : List Bytes) (rs : List Fp3)
    (final : Fp3) (iotaInv : UInt64) (tr : Fpn2Transcript) (q : List CosetOpening) :
    Except RejectCode Fpn2Transcript :=
  andThen (runList (foldStep O lay roots rs iotaInv)
      ⟨0, queryPos lay (tr.squeezeBytes O.hFs 8).1, Fp3.zero⟩ q) fun ff =>
    if ff.prev = final then .ok (tr.squeezeBytes O.hFs 8).2 else .error .E_FOLD

def QueryStepSpec (O : Rel3Oracles) (lay : Rel3Layout) (roots : List Bytes) (rs : List Fp3)
    (final : Fp3) (iotaInv : UInt64) (tr : Fpn2Transcript) (q : List CosetOpening)
    (tr' : Fpn2Transcript) : Prop :=
  ∃ ff, RunList (FoldSpec O lay roots rs iotaInv)
      ⟨0, queryPos lay (tr.squeezeBytes O.hFs 8).1, Fp3.zero⟩ q ff ∧
    ff.prev = final ∧ tr' = (tr.squeezeBytes O.hFs 8).2

theorem queryStep_iff (O : Rel3Oracles) (lay : Rel3Layout) (roots : List Bytes) (rs : List Fp3)
    (final : Fp3) (iotaInv : UInt64) (tr : Fpn2Transcript) (q : List CosetOpening)
    (tr' : Fpn2Transcript) :
    queryStep O lay roots rs final iotaInv tr q = .ok tr' ↔
      QueryStepSpec O lay roots rs final iotaInv tr q tr' := by
  unfold queryStep QueryStepSpec
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨ff, hr, h⟩
    exact ⟨ff, (runList_iff _ _ (foldStep_iff O lay roots rs iotaInv) _ _ _).mp hr,
      (guard_ok_iff _ _ _ _).mp h⟩
  · rintro ⟨ff, hr, h⟩
    exact ⟨ff, (runList_iff _ _ (foldStep_iff O lay roots rs iotaInv) _ _ _).mpr hr,
      (guard_ok_iff _ _ _ _).mpr h⟩

/-- The transcript the queries are squeezed from: after `absorb_bytes(FINAL)`. -/
def queryStart (O : Rel3Oracles) (c : Rel3Container) (pf : PcsState) : Fpn2Transcript :=
  pf.tr.absorbBytes O.hFs (Codec.fp3.enc c.body.final)

/-- Normative check 17 over all `Q` queries. -/
def queryPhase (O : Rel3Oracles) (lay : Rel3Layout) (c : Rel3Container) (pf : PcsState) :
    Except RejectCode Unit :=
  andThen (ofOption rel3IotaInv .E_FINAL) fun iotaInv =>
    andThen (runList (queryStep O lay (rel3Roots c.body) pf.rs c.body.final iotaInv)
      (queryStart O c pf) c.body.queries) fun _ => .ok ()

def QuerySpec (O : Rel3Oracles) (lay : Rel3Layout) (c : Rel3Container) (pf : PcsState) : Prop :=
  ∃ iotaInv trF, rel3IotaInv = some iotaInv ∧
    RunList (QueryStepSpec O lay (rel3Roots c.body) pf.rs c.body.final iotaInv)
      (queryStart O c pf) c.body.queries trF

theorem queryPhase_iff (O : Rel3Oracles) (lay : Rel3Layout) (c : Rel3Container) (pf : PcsState) :
    queryPhase O lay c pf = .ok () ↔ QuerySpec O lay c pf := by
  unfold queryPhase QuerySpec
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨iotaInv, hi, h⟩
    rw [andThen_ok_iff] at h
    obtain ⟨trF, hr, _⟩ := h
    exact ⟨iotaInv, trF, (ofOption_ok_iff _ _ _).mp hi,
      (runList_iff _ _ (queryStep_iff O lay _ _ _ iotaInv) _ _ _).mp hr⟩
  · rintro ⟨iotaInv, trF, hi, hr⟩
    refine ⟨iotaInv, (ofOption_ok_iff _ _ _).mpr hi, ?_⟩
    rw [andThen_ok_iff]
    exact ⟨trF, (runList_iff _ _ (queryStep_iff O lay _ _ _ iotaInv) _ _ _).mpr hr, rfl⟩

/-! ## The typed specification and the bytes verifier -/

/-- Normative checks 13–17 on a parsed container with a known layout. -/
def typedCheckRel3 (O : Rel3Oracles) (pub : PublicInput) (lay : Rel3Layout) (c : Rel3Container) :
    Except RejectCode Unit :=
  andThen (gkrPhase O pub c) fun g =>
    andThen (pcsPhase O pub lay c g) fun pf => queryPhase O lay c pf

/-- Checks 13–17 against a profile (its layout exists by check 10). -/
def rel3Check (O : Rel3Oracles) (pub : PublicInput) (prof : Fpn2Profile) (c : Rel3Container) :
    Except RejectCode Unit :=
  andThen (ofOption (rel3Layout prof) .E_BACKEND_UNAVAILABLE) fun lay => typedCheckRel3 O pub lay c

/-- **Typed acceptance predicate of relation 3** on a parsed container: the profile has a
relation-3 layout; the GKR root is `(0, root_den ≠ 0)` and all masked layer equations hold; the
Basefold round sums hold from the batched claim `Y` and end in `FINAL · w̃(r)`; every query's coset
openings are anchored in `root_0 .. root_{T−1}`, chain through the fused folds, and end in `FINAL`. -/
def Rel3TypedVerify (O : Rel3Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (c : Rel3Container) : Prop :=
  ∃ lay g pf, rel3Layout prof = some lay ∧ GkrSpec O pub c g ∧ PcsSpec O pub lay c g pf ∧
    QuerySpec O lay c pf

/-- The executable checks 13–17 decide exactly the typed acceptance predicate. -/
theorem rel3Check_iff (O : Rel3Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (c : Rel3Container) : rel3Check O pub prof c = .ok () ↔ Rel3TypedVerify O pub prof c := by
  unfold rel3Check Rel3TypedVerify typedCheckRel3
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨lay, hlay, h⟩
    rw [andThen_ok_iff] at h
    obtain ⟨g, hg, h⟩ := h
    rw [andThen_ok_iff] at h
    obtain ⟨pf, hp, hq⟩ := h
    exact ⟨lay, g, pf, (ofOption_ok_iff _ _ _).mp hlay, (gkrPhase_iff _ _ _ _).mp hg,
      (pcsPhase_iff _ _ _ _ _ _).mp hp, (queryPhase_iff _ _ _ _).mp hq⟩
  · rintro ⟨lay, g, pf, hlay, hg, hp, hq⟩
    refine ⟨lay, (ofOption_ok_iff _ _ _).mpr hlay, ?_⟩
    rw [andThen_ok_iff]
    refine ⟨g, (gkrPhase_iff _ _ _ _).mpr hg, ?_⟩
    rw [andThen_ok_iff]
    exact ⟨pf, (pcsPhase_iff _ _ _ _ _ _).mpr hp, (queryPhase_iff _ _ _ _).mpr hq⟩

/-- **`verifyBytesRel3`**: full verification of raw relation-3 container bytes against a pinned
profile, in the normative order of spec §6. `allowReduced` is `--allow-reduced-security`.

* 1–5   header: length `≥ 64`, magic, version, flags, reserved (`parseHeader`);
* 6     `profile_id` is registered, else `E_PROFILE_ID`;
* 7     it is the pinned profile, else `E_PROFILE_PINNED`;
* 8     tier is `STANDARD` or `allowReduced`, else `E_REDUCED_SECURITY`;
* 9–12  exact length `C_p`, relation-3 layout, zero `PAD`, canonical fields (`parseBodyRel3`);
* 13–17 GKR root, masked GKR layers, Basefold round sums, final evaluation, coset Merkle paths
        and fold consistency (`rel3Check`).

A profile of another relation is `E_BACKEND_UNAVAILABLE` here (relation 1 is `verifyBytes`,
relation 4 is `verifyBytesRel4`; `verifyBytesAny` in `Fpn2Rel5` dispatches). -/
def verifyBytesRel3 (O : Rel3Oracles) (pub : PublicInput) (prof : Fpn2Profile) (allowReduced : Bool)
    (raw : ByteArray) : Except RejectCode Rel3Claim :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    match lookupProfile h.profileId with
    | none => .error .E_PROFILE_ID
    | some found =>
      if found ≠ prof then .error .E_PROFILE_PINNED
      else if (prof.isProduction || allowReduced) = false then .error .E_REDUCED_SECURITY
      else
        match parseBodyRel3 prof (toBytes raw) with
        | .error e => .error e
        | .ok b =>
          match rel3Check O pub prof ⟨h, b⟩ with
          | .error e => .error e
          | .ok _ => .ok (Rel3Container.acceptedClaim ⟨h, b⟩)

/-- `verify_bytes` with the caller policy of `core/src/verify.rs::Policy`: an optional pinned
`expected_profile_id` (check 7) and the header-selected registered profile. -/
def verifyBytesRel3Policy (O : Rel3Oracles) (pub : PublicInput) (allowReduced : Bool)
    (expectedId : Option Bytes) (raw : ByteArray) : Except RejectCode Rel3Claim :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    match lookupProfile h.profileId with
    | none => .error .E_PROFILE_ID
    | some found =>
      if expectedId.isSome ∧ expectedId ≠ some h.profileId then .error .E_PROFILE_PINNED
      else verifyBytesRel3 O pub found allowReduced raw

/-- Everything `verifyBytesRel3` established on an accepted input, as one case analysis. -/
theorem verifyBytesRel3_spec {O : Rel3Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {raw : ByteArray} {claim : Rel3Claim}
    (h : verifyBytesRel3 O pub prof allowReduced raw = .ok claim) :
    ∃ c : Rel3Container,
      parseHeader raw = .ok c.header ∧ lookupProfile c.header.profileId = some prof ∧
      (prof.isProduction || allowReduced) = true ∧ parseBodyRel3 prof (toBytes raw) = .ok c.body ∧
      rel3Check O pub prof c = .ok () ∧ claim = c.acceptedClaim := by
  unfold verifyBytesRel3 at h
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
          cases hb : parseBodyRel3 prof (toBytes raw) with
          | error e => simp [hb] at h
          | ok b =>
            simp only [hb] at h
            cases ht : rel3Check O pub prof ⟨hd, b⟩ with
            | error e => simp [ht] at h
            | ok u =>
              simp only [ht] at h
              injection h with h
              refine ⟨⟨hd, b⟩, rfl, ?_, ?_, rfl, ?_, h.symm⟩
              · rw [← Decidable.not_not.mp hfound]
                exact hl
              · cases hv : (prof.isProduction || allowReduced) with
                | false => exact absurd hv hpol
                | true => rfl
              · cases u; exact ht

/-- **Refinement theorem (relation 3).** For every byte string `raw`, if `verifyBytesRel3`
accepts it then:

(a) `raw` parses, against the pinned and registered profile, to a container `c`; `raw` has
    exactly `C_p` bytes, is byte-for-byte the canonical serialization of `c`, and every byte
    after the last typed field is zero;
(b) the profile is production-tier, or the caller explicitly opted into reduced security;
(c) the typed algebraic checks `Rel3TypedVerify` hold on `c`, and the returned claim is the
    statement carried by `c`. -/
theorem verifyBytesRel3_refines_typed {O : Rel3Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {raw : ByteArray} {claim : Rel3Claim}
    (h : verifyBytesRel3 O pub prof allowReduced raw = .ok claim) :
    ∃ c : Rel3Container,
      (parseContainerRel3 prof raw = .ok c ∧ prof ∈ profileRegistry ∧
        raw.size = prof.containerBytes ∧ toBytes raw = encodeContainerRel3 prof c ∧
        ∃ lay, rel3Layout prof = some lay ∧ headerLen + lay.bodyUsed ≤ prof.containerBytes ∧
          ∀ i, headerLen + lay.bodyUsed ≤ i → i < raw.size → (toBytes raw)[i]? = some 0) ∧
      (prof.isProduction = true ∨ allowReduced = true) ∧
      (Rel3TypedVerify O pub prof c ∧ claim = c.acceptedClaim) := by
  obtain ⟨c, hh, hl, hpol, hb, ht, hc⟩ := verifyBytesRel3_spec h
  obtain ⟨hreg, hid⟩ := lookupProfile_spec hl
  have hparse : parseContainerRel3 prof raw = .ok c := by
    unfold parseContainerRel3
    simp [hh, hid, hb]
  obtain ⟨hsize, lay, hlay, hfit, _, hzero⟩ := parseContainerRel3_length_and_zero_pad hparse
  refine ⟨c, ⟨hparse, hreg, hsize, parseContainerRel3_encode hparse, lay, hlay, hfit, hzero⟩, ?_,
    (rel3Check_iff _ _ _ _).mp ht, hc⟩
  simpa [Bool.or_eq_true] using hpol

/-- **Completeness of the refinement.** A canonical container of a registered profile that
passes the tier policy and satisfies `Rel3TypedVerify` is accepted, with its own statement. -/
theorem verifyBytesRel3_complete {O : Rel3Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {raw : ByteArray} {c : Rel3Container}
    (hparse : parseContainerRel3 prof raw = .ok c)
    (hreg : lookupProfile prof.profileId = some prof)
    (hpol : prof.isProduction = true ∨ allowReduced = true)
    (ht : Rel3TypedVerify O pub prof c) :
    verifyBytesRel3 O pub prof allowReduced raw = .ok c.acceptedClaim := by
  obtain ⟨hh, hid, hb⟩ := parseContainerRel3_spec hparse
  have hpol' : (prof.isProduction || allowReduced) = true := by
    simpa [Bool.or_eq_true] using hpol
  unfold verifyBytesRel3
  simp [hh, hid, hreg, hpol', hb, (rel3Check_iff O pub prof c).mpr ht]

/-- **No silent downgrade.** Without the explicit opt-in, only a production-tier profile can be
accepted. -/
theorem verifyBytesRel3_downgrade_resistant {O : Rel3Oracles} {pub : PublicInput}
    {prof : Fpn2Profile} {raw : ByteArray} {claim : Rel3Claim}
    (h : verifyBytesRel3 O pub prof false raw = .ok claim) : prof.isProduction = true := by
  obtain ⟨_, _, hpol, _⟩ := verifyBytesRel3_refines_typed h
  rcases hpol with hp | hf
  · exact hp
  · cases hf

/-- A `REDUCED` relation-3 profile is rejected on every input unless the caller opts in. -/
theorem verifyBytesRel3_rejects_reduced (O : Rel3Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (raw : ByteArray) (hred : prof.isProduction = false) (claim : Rel3Claim) :
    verifyBytesRel3 O pub prof false raw ≠ .ok claim := by
  intro h
  rw [verifyBytesRel3_downgrade_resistant h] at hred
  cases hred

/-- **No second accepted encoding.** Two accepted byte strings carrying the same parsed
container are equal; in particular flipping any padding byte of an accepted proof is rejected. -/
theorem verifyBytesRel3_injective_on_accepted {O : Rel3Oracles} {pub : PublicInput}
    {prof : Fpn2Profile} {a₁ a₂ : Bool} {raw₁ raw₂ : ByteArray} {claim₁ claim₂ : Rel3Claim}
    {c : Rel3Container}
    (h₁ : verifyBytesRel3 O pub prof a₁ raw₁ = .ok claim₁)
    (h₂ : verifyBytesRel3 O pub prof a₂ raw₂ = .ok claim₂)
    (hc₁ : parseContainerRel3 prof raw₁ = .ok c) (hc₂ : parseContainerRel3 prof raw₂ = .ok c) :
    raw₁ = raw₂ ∧ claim₁ = claim₂ := by
  obtain ⟨c₁, ⟨hp₁, _⟩, _, _, hcl₁⟩ := verifyBytesRel3_refines_typed h₁
  obtain ⟨c₂, ⟨hp₂, _⟩, _, _, hcl₂⟩ := verifyBytesRel3_refines_typed h₂
  rw [hc₁] at hp₁
  rw [hc₂] at hp₂
  injection hp₁ with e₁
  injection hp₂ with e₂
  subst e₁ e₂
  exact ⟨parseContainerRel3_injective_on_accepted hc₁ hc₂, hcl₁.trans hcl₂.symm⟩

/-- An acceptance under the caller policy is an acceptance under a registered profile, which is
the pinned one whenever a pin was given. -/
theorem verifyBytesRel3Policy_spec {O : Rel3Oracles} {pub : PublicInput} {allowReduced : Bool}
    {expectedId : Option Bytes} {raw : ByteArray} {claim : Rel3Claim}
    (h : verifyBytesRel3Policy O pub allowReduced expectedId raw = .ok claim) :
    ∃ prof, prof ∈ profileRegistry ∧ verifyBytesRel3 O pub prof allowReduced raw = .ok claim ∧
      ∀ id, expectedId = some id → prof.profileId = id := by
  unfold verifyBytesRel3Policy at h
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

#print axioms FlashProveNext.rel3Layout_foldedGkrV1
#print axioms FlashProveNext.rel3Layout_foldedGkrN8
#print axioms FlashProveNext.decRawN_sound
#print axioms FlashProveNext.decFam_sound
#print axioms FlashProveNext.foldedBody_enc
#print axioms FlashProveNext.layerBytes_sections
#print axioms FlashProveNext.parseBodyRel3_spec
#print axioms FlashProveNext.parseContainerRel3_spec
#print axioms FlashProveNext.parseContainerRel3_encode
#print axioms FlashProveNext.parseContainerRel3_length_and_zero_pad
#print axioms FlashProveNext.parseContainerRel3_injective_on_accepted
#print axioms FlashProveNext.andThen_ok_iff
#print axioms FlashProveNext.ofOption_ok_iff
#print axioms FlashProveNext.ok_eq_ok_iff
#print axioms FlashProveNext.ite_ok_iff
#print axioms FlashProveNext.guard_ok_iff
#print axioms FlashProveNext.runList_iff
#print axioms FlashProveNext.gkrRoundStep_iff
#print axioms FlashProveNext.gkrLayer0_iff
#print axioms FlashProveNext.gkrLayerStep_iff
#print axioms FlashProveNext.gkrPhase_iff
#print axioms FlashProveNext.pcsStep_iff
#print axioms FlashProveNext.pcsPhase_iff
#print axioms FlashProveNext.foldStep_iff
#print axioms FlashProveNext.queryStep_iff
#print axioms FlashProveNext.queryPhase_iff
#print axioms FlashProveNext.rel3Check_iff
#print axioms FlashProveNext.verifyBytesRel3_spec
#print axioms FlashProveNext.verifyBytesRel3_refines_typed
#print axioms FlashProveNext.verifyBytesRel3_complete
#print axioms FlashProveNext.verifyBytesRel3_downgrade_resistant
#print axioms FlashProveNext.verifyBytesRel3_rejects_reduced
#print axioms FlashProveNext.verifyBytesRel3_injective_on_accepted
#print axioms FlashProveNext.verifyBytesRel3Policy_spec
