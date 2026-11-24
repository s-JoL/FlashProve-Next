/-
Executable `verifyBytesRel5` for `FPN2` relation-5 containers (`FOLDED_VEC_MATMUL`,
`specs/11-wire-format.md` §5.6, §6), its refinement to a typed specification, and the relation
dispatcher `verifyBytesAny` over relations 1, 3, 4 and 5. Stage 7 of the Lean verifier (M4); Lean
core only. The definitions follow `core/src/folded_vec.rs` step by step and reuse the batched
opening of relation 4 unchanged (`AirPcs`, `AirQuery`, `pcsOps4`, `queryStep4`: two batching
rounds, fused coset-4 folds, dual layer-0 leaves) together with the relation-3 round, fold and
Merkle steps.

1. `Rel5Layout` : the fixed-size layout, a function of `(n_vars, n_queries, max_hidden)` (`k`,
   `beta`, `tiles`, `kdim`, `d`, `folds`, `depth`, `ipLen`, `pcsLen`, `queryLen`, `bodyUsed`,
   `libraSlot`);
2. `rel5Terms`, `batchedClaim5`, `finalWeight5` : the batched linear claim of §5.6.4 on the witness
   vector `F` (`root_0`) and the public weight vector `C` (`weight_root`), and the final weight
   `W̃(θ, r_1, r')`;
3. `Codec.vecBody`, `parseBodyRel5`, `parseContainerRel5` : canonical decoding of the §5.6 layout
   (normative checks 9–12), with `parseContainerRel5_encode`,
   `parseContainerRel5_length_and_zero_pad`, `parseContainerRel5_injective_on_accepted`;
4. `ipPhase`, `pcsPhase5`, `queryPhase5`, `typedCheckRel5`, `verifyBytesRel5` : normative checks
   13–17 in the transcript schedule of `folded_vec.rs::verify`, with the spec reject codes; check
   13 is the mandatory weight pin (`E_CIRCUIT_UNPINNED` / `E_CIRCUIT_PINNED`);
5. `IpSpec`, `PcsSpec5`, `QuerySpec5`, `Rel5TypedVerify` : the declarative typed statement on a
   parsed container;
6. `verifyBytesRel5_refines_typed`, `verifyBytesRel5_complete`,
   `verifyBytesRel5_downgrade_resistant`, `verifyBytesRel5_injective_on_accepted`,
   `verifyBytesRel5_requires_weight_pin`, `verifyBytesRel5_unpinned_rejects`, `vecBody_enc` :
   accepted bytes are a canonical, fixed-length, zero-padded container of a registered relation-5
   profile whose `STMT.weight_root` is the caller's pin and on which `Rel5TypedVerify` holds, and
   conversely;
7. `verifyParsedRel5`, `verifyBytesRel5_eq_verifyParsedRel5`, `verifyBytesRel5_foldedVecN22` : the
   profile-generic core (parse, pin, `rel5Check`); `verifyBytesRel5` returns its result on every
   registered profile passing the tier policy when the header names it, in particular on the
   64-tile `fpn2-folded-vec-n22-v1` at the layout of `rel5Layout_foldedVecN22`;
8. `verifyBytesAny`, `verifyBytesAny_spec`, `verifyBytesAny_air_requires_circuit_pin`,
   `verifyBytesAny_matmul_requires_weight_pin`, `verifyBytesAny_downgrade_resistant` : one entry
   point dispatching on the registered profile's relation like
   `core/src/verify.rs::verify_bytes`.

Scope. `Rel5TypedVerify` is the algebraic acceptance predicate of relation 5 over abstract hash
oracles. This module proves that the bytes verifier decides exactly that predicate (plus the
weight pin) on exactly the canonical encodings; it does not prove that the predicate is a sound or
zero-knowledge argument for the statement `Y = W X` of §5.6 (see `THEOREM.md`).
-/
import FlashProveNext.Fpn2Wire
import FlashProveNext.Fpn2Transcript
import FlashProveNext.VerifyBytes
import FlashProveNext.Fpn2Rel3
import FlashProveNext.Fpn2Rel4

namespace FlashProveNext

/-! ## Layout (spec §5.6; `core/src/folded_vec.rs::VecLayout`) -/

/-- Fixed-size relation-5 layout, a pure function of `(n_vars, n_queries, max_hidden)`; `k` is
`log2 K` for the tile width `K = max_hidden`. -/
structure Rel5Layout where
  nVars : Nat
  nQueries : Nat
  k : Nat
  deriving DecidableEq, Repr

namespace Rel5Layout

/-- `ROW_BITS = 5`: a tile has `ROWS = 32` rows. -/
def rowBits : Nat := 5

/-- `some k` iff `kd = 2^k` with `k < 32` (`kd.is_power_of_two()` and `kd.trailing_zeros()` for
the `u32` field `max_hidden`). -/
def log2Exact (kd : Nat) : Option Nat := (List.range 32).find? (fun k => decide (2 ^ k = kd))

/-- `VecLayout::of`: `sc_degree = 3`, `n` even with `12 ≤ n ≤ 28`, `1 ≤ Q ≤ 4096`,
`K = max_hidden ≥ 32` a power of two, and `n ≥ 1 + 5 + log2 K`. -/
def ofProfile (p : Fpn2Profile) : Option Rel5Layout :=
  if p.scDegree = 3 ∧ 12 ≤ p.nVars ∧ p.nVars % 2 = 0 ∧ p.nVars ≤ 28 ∧ 1 ≤ p.nQueries ∧
      p.nQueries ≤ 4096 ∧ 32 ≤ p.maxHidden then
    match log2Exact p.maxHidden with
    | none => none
    | some k => if 1 + rowBits + k ≤ p.nVars then some ⟨p.nVars, p.nQueries, k⟩ else none
  else none

/-- The geometry of the batched opening (blocks, Libra deposit, PCS and query sections, fold
domain) is that of relation 4 at the same `(n, Q)`. -/
def air (lay : Rel5Layout) : Rel4Layout := ⟨lay.nVars, lay.nQueries⟩

/-- `log2 B`, the number of tile bits: `n − 1 − 5 − k`. -/
def beta (lay : Rel5Layout) : Nat := lay.nVars - 1 - rowBits - lay.k

/-- `B = 2^beta` tiles. -/
def tiles (lay : Rel5Layout) : Nat := 2 ^ lay.beta

/-- `K = 2^k` columns per tile. -/
def kdim (lay : Rel5Layout) : Nat := 2 ^ lay.k

/-- Inner-product sumcheck variables: `k` column bits, then `beta` tile bits. -/
def d (lay : Rel5Layout) : Nat := lay.k + lay.beta

/-- `T = n / 2` fused coset-4 folds. -/
def folds (lay : Rel5Layout) : Nat := lay.nVars / 2

/-- Merkle depth of folded layer `t`: `n + 1 − 2t`. -/
def depth (lay : Rel5Layout) (t : Nat) : Nat := lay.nVars + 1 - 2 * t

/-- `G_sum ‖ d rounds × s(0..3) ‖ e_W`. -/
def ipLen (lay : Rel5Layout) : Nat := 24 + 96 * lay.d + 24

/-- `z_blind ‖ (n + 2) rounds ‖ (T − 1) fold roots ‖ FINAL` (as relation 4). -/
def pcsLen (lay : Rel5Layout) : Nat := lay.air.pcsLen

/-- Per query: the weight-leaf opening of layer 0, plus one coset opening per folded layer. -/
def queryLen (lay : Rel5Layout) : Nat := lay.air.queryLen

/-- Bytes used by the body before zero padding. -/
def bodyUsed (lay : Rel5Layout) : Nat := 96 + lay.ipLen + lay.pcsLen + lay.nQueries * lay.queryLen

/-- Cell index of the inner-product Libra coefficient `g_{l,e}`: slot `4l + e` deposited into the
free bits of the half `{b_1 = 0}`. -/
def libraSlot (lay : Rel5Layout) (l e : Nat) : Nat := lay.air.deposit (4 * l + e)

end Rel5Layout

/-- The layout of a relation-5 profile; `none` for any other relation or unwired parameters
(normative check 10). -/
def rel5Layout (p : Fpn2Profile) : Option Rel5Layout :=
  if p.relation = .foldedVecMatmul then Rel5Layout.ofProfile p else none

/-- `fpn2-folded-vec-n16-v1`: `k = 10`, `beta = 0`, one tile, `d = 10`, `T = 8`, body 288 368
bytes (`core/src/folded_vec.rs::tests::layout_of_registered_profiles`). -/
theorem rel5Layout_foldedVecN16 :
    rel5Layout foldedVecN16V1 = some ⟨16, 72, 10⟩ ∧
      (Rel5Layout.k ⟨16, 72, 10⟩, Rel5Layout.beta ⟨16, 72, 10⟩, Rel5Layout.tiles ⟨16, 72, 10⟩,
        Rel5Layout.d ⟨16, 72, 10⟩, Rel5Layout.folds ⟨16, 72, 10⟩,
        Rel5Layout.bodyUsed ⟨16, 72, 10⟩) = (10, 0, 1, 10, 8, 288368) ∧
      Rel5Layout.tiles ⟨16, 72, 10⟩ * 32 * Rel5Layout.kdim ⟨16, 72, 10⟩ = 2 ^ 15 ∧
      headerLen + Rel5Layout.bodyUsed ⟨16, 72, 10⟩ ≤ foldedVecN16V1.containerBytes := by
  decide

/-- `fpn2-folded-vec-n22-v1`: `k = 10`, `beta = 6`, 64 tiles (2 097 152 multiply-adds), `d = 16`,
`T = 11`, body 469 184 bytes, ending at 469 248
(`core/src/folded_vec.rs::tests::layout_of_registered_profiles`). -/
theorem rel5Layout_foldedVecN22 :
    rel5Layout foldedVecN22V1 = some ⟨22, 72, 10⟩ ∧
      (Rel5Layout.k ⟨22, 72, 10⟩, Rel5Layout.beta ⟨22, 72, 10⟩, Rel5Layout.tiles ⟨22, 72, 10⟩,
        Rel5Layout.d ⟨22, 72, 10⟩, Rel5Layout.folds ⟨22, 72, 10⟩,
        Rel5Layout.bodyUsed ⟨22, 72, 10⟩) = (10, 6, 64, 16, 11, 469184) ∧
      Rel5Layout.tiles ⟨22, 72, 10⟩ * 32 * Rel5Layout.kdim ⟨22, 72, 10⟩ = 2097152 ∧
      headerLen + Rel5Layout.bodyUsed ⟨22, 72, 10⟩ ≤ foldedVecN22V1.containerBytes := by
  decide

/-- The relation-1, -3, -4 and -5 parsers never apply to one profile together. -/
theorem rel5Layout_disjoint (p : Fpn2Profile) (lay : Rel5Layout) (h : rel5Layout p = some lay) :
    rel3Layout p = none ∧ rel4Layout p = none ∧ p.bodyUsed = none := by
  unfold rel5Layout at h
  split at h
  · rename_i hrel
    simp [rel3Layout, rel4Layout, Fpn2Profile.bodyUsed, hrel]
  · cases h

/-- An admitted layout copies `(n, Q)` from the profile, has `K = max_hidden = 2^k ≥ 32`,
`1 + 5 + k ≤ n ≤ 28` and `sc_degree = 3`. -/
theorem rel5Layout_bounds (p : Fpn2Profile) (lay : Rel5Layout) (h : rel5Layout p = some lay) :
    lay.nVars = p.nVars ∧ lay.nQueries = p.nQueries ∧ 2 ^ lay.k = p.maxHidden ∧ 32 ≤ p.maxHidden ∧
      1 + Rel5Layout.rowBits + lay.k ≤ lay.nVars ∧ lay.nVars ≤ 28 ∧ p.scDegree = 3 := by
  unfold rel5Layout at h
  split at h
  · unfold Rel5Layout.ofProfile at h
    split at h
    · rename_i hc
      split at h
      · cases h
      · rename_i k hk
        split at h
        · rename_i hfit
          injection h with h
          subst h
          unfold Rel5Layout.log2Exact at hk
          have h2 := List.find?_some hk
          exact ⟨rfl, rfl, of_decide_eq_true h2, hc.2.2.2.2.2.2, hfit, hc.2.2.2.1, hc.1⟩
        · cases h
    · cases h
  · cases h

/-! ## The batched linear claim (spec §5.6.4; `folded_vec.rs::terms`) -/

/-- The weight vector `a` on `F` (blocks and sparse Libra cells) and the weight vector `c` on `C`
of `target + μ e_W + μ² z_blind`. -/
structure Rel5Terms where
  fBlocks : List Rel3Block
  fSparse : List (Nat × Fp3)
  cBlocks : List Rel3Block

/-- The sparse cells `λ · s_l^e` (`e < 4`) of the inner-product Libra mask. -/
def ipCells (lay : Rel5Layout) (lam : Fp3) (s : List Fp3) : List (Nat × Fp3) :=
  (List.range s.length).flatMap fun l => powCells (lay.libraSlot l) 4 lam (s.getD l Fp3.zero)

/-- `terms(lay, r_i, r_b, s, lam, e_w, mu)`. With `s = s_j ‖ s_b` (`k` column challenges, `beta`
tile challenges):

* `F`: `eq(r_b, s_b) e_W · X̃(s_j, s_b)` on the block `[0, B K)`;
  `−eq(r_b, s_b) e_s · Ỹ(s_j[..5], s_b)` on the block `[B K, B K + 32 B)`, where
  `e_s = eq(r_i, s_j[..5]) · Π_{t ≥ 5} (1 − s_j[t])`; the Libra cells; the two blinding blocks
  with coefficient `μ²`;
* `C`: `μ · W̃(s_j, r_i, s_b)` on the whole data half. -/
def rel5Terms (lay : Rel5Layout) (rI rB s : List Fp3) (lam eW mu : Fp3) : Rel5Terms :=
  let sJ := s.take lay.k
  let sB := s.drop lay.k
  let eqrb := eqEval rB sB
  let eS := (sJ.drop Rel5Layout.rowBits).foldl (fun acc t => acc.mul (Fp3.one.sub t))
    (eqEval rI (sJ.take Rel5Layout.rowBits))
  ⟨[airBlock lay.air 0 lay.d (eqrb.mul eW) s,
    airBlock lay.air (lay.tiles * lay.kdim) (Rel5Layout.rowBits + lay.beta) (eqrb.mul eS).neg
      (sJ.take Rel5Layout.rowBits ++ sB),
    airMaskBlock lay.air (mu.mul mu) (Fp3.ofFp 2),
    airMaskBlock lay.air (mu.mul mu) (Fp3.ofFp 3)],
   ipCells lay lam s,
   [airBlock lay.air 0 (lay.nVars - 1) mu (sJ ++ (rI ++ sB))]⟩

/-- `Y = target + μ e_W + μ² z_blind`, where `target` is the last inner-product sumcheck claim. -/
def batchedClaim5 (target mu eW zBlind : Fp3) : Fp3 :=
  (target.add (mu.mul eW)).add ((mu.mul mu).mul zBlind)

/-- `W̃(θ, r_1, r') = (1 − r_1)((1 − θ) ã(r') + (2θ − 1) c̃(r'))` at the `n + 2` Basefold
challenges `rs = (θ, r_1, r')`. -/
def finalWeight5 (lay : Rel5Layout) (tm : Rel5Terms) (rs : List Fp3) : Fp3 :=
  let theta := rs.getD 0 Fp3.zero
  let r1 := rs.getD 1 Fp3.zero
  let aR := weightsEval lay.air.pcs ⟨Fp3.zero, Fp3.zero, tm.fBlocks, tm.fSparse, Fp3.zero⟩
    (rs.drop 2)
  let cR := weightsEval lay.air.pcs ⟨Fp3.zero, Fp3.zero, tm.cBlocks, [], Fp3.zero⟩ (rs.drop 2)
  (Fp3.one.sub r1).mul
    (((Fp3.one.sub theta).mul aR).add (((theta.add theta).sub Fp3.one).mul cR))

/-! ## Typed body and its codec (spec §5.6, normative check 12) -/

/-- `STMT` (96 bytes): `root_0 ‖ weight_root ‖ N_s`. -/
structure VecStmt where
  root0 : Bytes
  weightRoot : Bytes
  nS : Bytes

/-- `IP`: `G_sum`, the `d` degree-3 rounds `s(0..3)`, the public evaluation `e_W`. -/
structure VecIp where
  gSum : Fp3
  rounds : List F3x4
  eW : Fp3

/-- Decoded body of a relation-5 (`FOLDED_VEC_MATMUL`) container, in wire order. The `PCS` and
`QUERIES` sections have the relation-4 shape. -/
structure VecBody where
  stmt : VecStmt
  ip : VecIp
  pcs : AirPcs
  queries : List AirQuery

/-- A fully parsed relation-5 container. -/
structure Rel5Container where
  header : Fpn2Header
  body : VecBody

def Codec.vecStmt : Codec VecStmt :=
  ((Codec.rawN 32).pair ((Codec.rawN 32).pair (Codec.rawN 32))).map
    (fun t => ⟨t.1, t.2.1, t.2.2⟩) (fun st => (st.root0, st.weightRoot, st.nS)) (fun _ => rfl)

def Codec.vecIp (lay : Rel5Layout) : Codec VecIp :=
  (Codec.fp3.pair ((Codec.fam (fun _ => Codec.fp3x4) Codec.fp3x4.enc (fun _ _ => rfl) 0 lay.d).pair
    Codec.fp3)).map
    (fun t => ⟨t.1, t.2.1, t.2.2⟩) (fun I => (I.gSum, I.rounds, I.eW)) (fun _ => rfl)

/-- The whole relation-5 body, sections back-to-back in wire order:
`STMT ‖ IP ‖ PCS ‖ QUERIES`. Every `Fp3` field is decoded canonically in wire order, so the first
non-canonical limb determines `E_NONCANONICAL` (`folded_vec.rs::decode`). -/
def Codec.vecBody (lay : Rel5Layout) : Codec VecBody :=
  (Codec.vecStmt.pair ((Codec.vecIp lay).pair ((Codec.airPcs lay.air).pair
    (Codec.fam (fun _ => Codec.airQuery lay.air) airQueryBytes (fun _ _ => rfl) 0
      lay.nQueries)))).map
    (fun t => ⟨t.1, t.2.1, t.2.2.1, t.2.2.2⟩)
    (fun b => (b.stmt, b.ip, b.pcs, b.queries)) (fun _ => rfl)

/-- The 96 `STMT` wire bytes `root_0 ‖ weight_root ‖ N_s`. -/
def stmtBytesRel5 (st : VecStmt) : Bytes := st.root0 ++ (st.weightRoot ++ st.nS)

/-- `IP`: `G_sum ‖ rounds ‖ e_W`. -/
def ipBytes (I : VecIp) : Bytes :=
  Codec.fp3.enc I.gSum ++ (I.rounds.flatMap Codec.fp3x4.enc ++ Codec.fp3.enc I.eW)

/-- Canonical body serializer: the sections in wire order. It does not depend on the layout. The
transcript absorbs exactly these byte strings (`STMT`, `G_sum`, each inner-product round, `e_W`,
`z_blind`, each PCS round and fold root, `FINAL`), and the coset values hashed into Merkle leaves
are the `QUERIES` bytes. -/
def encodeBodyRel5 (b : VecBody) : Bytes :=
  stmtBytesRel5 b.stmt ++ (ipBytes b.ip ++ (pcsBytesRel4 b.pcs ++ b.queries.flatMap airQueryBytes))

/-- The encoder of the body codec is the section-wise serializer, for every layout: what
`Codec.vecBody` accepts is the concatenation of the sections the verifier absorbs. -/
theorem vecBody_enc (lay : Rel5Layout) (b : VecBody) :
    (Codec.vecBody lay).enc b = encodeBodyRel5 b := by
  simp [Codec.vecBody, Codec.vecStmt, Codec.vecIp, Codec.airPcs, Codec.map, Codec.pair,
    Codec.rawN, Codec.fam, encodeBodyRel5, stmtBytesRel5, ipBytes, pcsBytesRel4,
    List.append_assoc]

/-- Normative checks 9–12 for relation 5: exact length `C_p`, wired backend, all-zero `PAD`,
canonical decoding of every typed field in wire order. -/
def parseBodyRel5 (prof : Fpn2Profile) (l : Bytes) : Except RejectCode VecBody :=
  if l.length ≠ prof.containerBytes then .error .E_LENGTH
  else
    match rel5Layout prof with
    | none => .error .E_BACKEND_UNAVAILABLE
    | some lay =>
      if prof.containerBytes < headerLen + lay.bodyUsed then .error .E_LENGTH
      else if (l.drop headerLen).drop lay.bodyUsed ≠
          List.replicate (prof.containerBytes - headerLen - lay.bodyUsed) 0 then .error .E_PADDING
      else
        match (Codec.vecBody lay).dec ((l.drop headerLen).take lay.bodyUsed) with
        | .error e => .error e
        | .ok (b, rest) => if rest = [] then .ok b else .error .E_LENGTH

/-- Strict parser against a pinned profile: header checks 1–5, the header names exactly the
pinned `profile_id` (else `E_PROFILE_PINNED`), then body checks 9–12. -/
def parseContainerRel5 (prof : Fpn2Profile) (raw : ByteArray) : Except RejectCode Rel5Container :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    if h.profileId ≠ prof.profileId then .error .E_PROFILE_PINNED
    else
      match parseBodyRel5 prof (toBytes raw) with
      | .error e => .error e
      | .ok b => .ok ⟨h, b⟩

/-- Zero padding length of a relation-5 profile (`0` when the relation is not wired). -/
def padLenRel5 (prof : Fpn2Profile) : Nat :=
  match rel5Layout prof with
  | none => 0
  | some lay => prof.containerBytes - headerLen - lay.bodyUsed

/-- Canonical serializer (inverse of `parseContainerRel5` on accepted inputs): `C_p` bytes. -/
def encodeContainerRel5 (prof : Fpn2Profile) (c : Rel5Container) : Bytes :=
  encodeHeader c.header ++ (encodeBodyRel5 c.body ++ List.replicate (padLenRel5 prof) 0)

theorem parseBodyRel5_spec {prof : Fpn2Profile} {l : Bytes} {b : VecBody}
    (hp : parseBodyRel5 prof l = .ok b) :
    l.length = prof.containerBytes ∧ ∃ lay, rel5Layout prof = some lay ∧
      headerLen + lay.bodyUsed ≤ prof.containerBytes ∧
      (l.drop headerLen).take lay.bodyUsed = encodeBodyRel5 b ∧
      (l.drop headerLen).drop lay.bodyUsed =
        List.replicate (prof.containerBytes - headerLen - lay.bodyUsed) 0 := by
  unfold parseBodyRel5 at hp
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
              have hs := (Codec.vecBody lay).sound _ _ _ hdec
              rw [List.append_nil, vecBody_enc] at hs
              exact ⟨Decidable.not_not.mp hlen, lay, hlay, Nat.le_of_not_lt hfit, hs,
                Decidable.not_not.mp hpad⟩
            · cases hp

theorem parseContainerRel5_spec {prof : Fpn2Profile} {raw : ByteArray} {c : Rel5Container}
    (hp : parseContainerRel5 prof raw = .ok c) :
    parseHeader raw = .ok c.header ∧ c.header.profileId = prof.profileId ∧
      parseBodyRel5 prof (toBytes raw) = .ok c.body := by
  unfold parseContainerRel5 at hp
  cases hh : parseHeader raw with
  | error e => simp [hh] at hp
  | ok h =>
    simp only [hh] at hp
    split at hp
    · cases hp
    · rename_i hid
      cases hb : parseBodyRel5 prof (toBytes raw) with
      | error e => simp [hb] at hp
      | ok b =>
        simp only [hb] at hp
        injection hp with hp
        subst hp
        exact ⟨rfl, Decidable.not_not.mp hid, rfl⟩

/-- **Canonical wire representation.** An accepted byte string is exactly the canonical
serialization of the container it parses to. -/
theorem parseContainerRel5_encode {prof : Fpn2Profile} {raw : ByteArray} {c : Rel5Container}
    (hp : parseContainerRel5 prof raw = .ok c) : toBytes raw = encodeContainerRel5 prof c := by
  obtain ⟨hh, _, hb⟩ := parseContainerRel5_spec hp
  obtain ⟨_, lay, hlay, _, htake, hdrop⟩ := parseBodyRel5_spec hb
  have hhead := parseHeaderBytes_encode hh
  have e1 : toBytes raw = (toBytes raw).take headerLen ++ (toBytes raw).drop headerLen :=
    (List.take_append_drop _ _).symm
  have e2 : (toBytes raw).drop headerLen =
      ((toBytes raw).drop headerLen).take lay.bodyUsed ++
        ((toBytes raw).drop headerLen).drop lay.bodyUsed :=
    (List.take_append_drop _ _).symm
  have hpadLen : padLenRel5 prof = prof.containerBytes - headerLen - lay.bodyUsed := by
    simp [padLenRel5, hlay]
  rw [e1, e2, hhead, htake, hdrop, encodeContainerRel5, hpadLen]

/-- An accepted container has exactly `C_p` bytes and every padding byte is zero. -/
theorem parseContainerRel5_length_and_zero_pad {prof : Fpn2Profile} {raw : ByteArray}
    {c : Rel5Container} (hp : parseContainerRel5 prof raw = .ok c) :
    raw.size = prof.containerBytes ∧ ∃ lay, rel5Layout prof = some lay ∧
      headerLen + lay.bodyUsed ≤ prof.containerBytes ∧
      (toBytes raw).drop (headerLen + lay.bodyUsed) =
        List.replicate (prof.containerBytes - headerLen - lay.bodyUsed) 0 ∧
      ∀ i, headerLen + lay.bodyUsed ≤ i → i < raw.size → (toBytes raw)[i]? = some 0 := by
  obtain ⟨_, _, hb⟩ := parseContainerRel5_spec hp
  obtain ⟨hlen, lay, hlay, hfit, _, hdrop⟩ := parseBodyRel5_spec hb
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
theorem parseContainerRel5_injective_on_accepted {prof : Fpn2Profile} {raw₁ raw₂ : ByteArray}
    {c : Rel5Container} (h₁ : parseContainerRel5 prof raw₁ = .ok c)
    (h₂ : parseContainerRel5 prof raw₂ = .ok c) : raw₁ = raw₂ :=
  toBytes_injective ((parseContainerRel5_encode h₁).trans (parseContainerRel5_encode h₂).symm)

/-! ## Check 14: Libra-masked inner-product sumcheck (spec §5.6.3) -/

/-- Relation 5 calls the same two hash functions as relations 3 and 4 (`H_fs`, `B256_M`). -/
abbrev Rel5Oracles := Rel3Oracles

/-- What an accepted relation-5 container asserts: the profile and the `STMT` fields, in
particular the public weight commitment `weight_root` the statement is relative to. -/
structure Rel5Claim where
  profileId : Bytes
  root0 : Bytes
  weightRoot : Bytes
  nS : Bytes
  deriving DecidableEq, Repr

def Rel5Container.acceptedClaim (c : Rel5Container) : Rel5Claim :=
  ⟨c.header.profileId, c.body.stmt.root0, c.body.stmt.weightRoot, c.body.stmt.nS⟩

/-- The transcript after `init`, `absorb_bytes(header)`, `absorb_bytes(STMT)`: both `root_0` and
`weight_root` are bound before any challenge. -/
def rel5Start (O : Rel5Oracles) (pub : PublicInput) (c : Rel5Container) : Fpn2Transcript :=
  ((Fpn2Transcript.init O.hFs pub.transcriptLabel).absorbBytes O.hFs
    (encodeHeader c.header)).absorbBytes O.hFs (stmtBytesRel5 c.body.stmt)

/-- The row point `r_i` (5 challenges), squeezed first. -/
def ipRowTr (O : Rel5Oracles) (pub : PublicInput) (c : Rel5Container) :
    List Fp3 × Fpn2Transcript :=
  squeezeMany O Rel5Layout.rowBits (rel5Start O pub c)

/-- The tile point `r_b` (`beta` challenges), squeezed after `r_i`. -/
def ipTileTr (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout) (c : Rel5Container) :
    List Fp3 × Fpn2Transcript :=
  squeezeMany O lay.beta (ipRowTr O pub c).2

/-- `absorb_bytes(G_sum)`, then the Libra challenge `λ`. -/
def ipLamTr (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout) (c : Rel5Container) :
    Fp3 × Fpn2Transcript :=
  ((ipTileTr O pub lay c).2.absorbBytes O.hFs (Codec.fp3.enc c.body.ip.gSum)).squeezeFp3 O.hFs

/-- The inner-product sumcheck starts from `λ · G_sum` (the masked sum itself is claimed to be
zero). -/
def ipInit (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout) (c : Rel5Container) :
    RoundState :=
  ⟨(ipLamTr O pub lay c).2, (ipLamTr O pub lay c).1.mul c.body.ip.gSum, []⟩

/-- One degree-3 inner-product round: `s(0) + s(1) = target`, else `E_ZEROCHECK`; the challenge
and the next target are those of a relation-3 layer round (`gkrRoundNext`: absorb the 96 raw
bytes, squeeze `r`, interpolate). -/
def ipRoundStep (O : Rel5Oracles) (s : RoundState) (sv : F3x4) : Except RejectCode RoundState :=
  if sv.1.add sv.2.1 = s.target then .ok (gkrRoundNext O s sv) else .error .E_ZEROCHECK

def IpRoundSpec (O : Rel5Oracles) (s : RoundState) (sv : F3x4) (s' : RoundState) : Prop :=
  sv.1.add sv.2.1 = s.target ∧ s' = gkrRoundNext O s sv

theorem ipRoundStep_iff (O : Rel5Oracles) (s : RoundState) (sv : F3x4) (s' : RoundState) :
    ipRoundStep O s sv = .ok s' ↔ IpRoundSpec O s sv s' :=
  guard_ok_iff _ _ _ _

/-- Normative check 14: the `d` inner-product round sums, else `E_ZEROCHECK`. There is no closing
equation here: the last target enters the batched claim `Y`. The end state holds the point `s`
and the transcript before `e_W`. -/
def ipPhase (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout) (c : Rel5Container) :
    Except RejectCode RoundState :=
  runList (ipRoundStep O) (ipInit O pub lay c) c.body.ip.rounds

def IpSpec (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout) (c : Rel5Container)
    (sf : RoundState) : Prop :=
  RunList (IpRoundSpec O) (ipInit O pub lay c) c.body.ip.rounds sf

theorem ipPhase_iff (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout) (c : Rel5Container)
    (sf : RoundState) : ipPhase O pub lay c = .ok sf ↔ IpSpec O pub lay c sf :=
  runList_iff _ _ (ipRoundStep_iff O) _ _ _

/-! ## Checks 15–16: batched Basefold sumcheck and final evaluation (spec §5.6.4) -/

/-- `absorb_bytes(e_W)`, `absorb_bytes(z_blind)`, then the batching challenge `μ`. -/
def pcsMuTr5 (O : Rel5Oracles) (c : Rel5Container) (sf : RoundState) : Fp3 × Fpn2Transcript :=
  ((sf.tr.absorbBytes O.hFs (Codec.fp3.enc c.body.ip.eW)).absorbBytes O.hFs
    (Codec.fp3.enc c.body.pcs.zBlind)).squeezeFp3 O.hFs

def pcsTerms5 (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout) (c : Rel5Container)
    (sf : RoundState) : Rel5Terms :=
  rel5Terms lay (ipRowTr O pub c).1 (ipTileTr O pub lay c).1 sf.rs (ipLamTr O pub lay c).1
    c.body.ip.eW (pcsMuTr5 O c sf).1

/-- The Basefold sumcheck starts from the batched linear claim `Y = target + μ e_W + μ² z_blind`. -/
def pcsInit5 (O : Rel5Oracles) (c : Rel5Container) (sf : RoundState) : PcsState :=
  ⟨(pcsMuTr5 O c sf).2,
    batchedClaim5 sf.target (pcsMuTr5 O c sf).1 c.body.ip.eW c.body.pcs.zBlind, []⟩

/-- Normative checks 15 and 16: the `n + 2` Basefold round sums in the relation-4 wire order
(`pcsOps4`, else `E_SUMCHECK`), then `y = FINAL · W̃(θ, r_1, r')` (else `E_FINAL`). -/
def pcsPhase5 (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout) (c : Rel5Container)
    (sf : RoundState) : Except RejectCode PcsState :=
  andThen (runList (pcsStep O) (pcsInit5 O c sf) (pcsOps4 c.body.pcs)) fun pf =>
    if pf.y = c.body.pcs.final.mul (finalWeight5 lay (pcsTerms5 O pub lay c sf) pf.rs) then
      .ok pf
    else .error .E_FINAL

def PcsSpec5 (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout) (c : Rel5Container)
    (sf : RoundState) (pf : PcsState) : Prop :=
  RunList (PcsOpSpec O) (pcsInit5 O c sf) (pcsOps4 c.body.pcs) pf ∧
    pf.y = c.body.pcs.final.mul (finalWeight5 lay (pcsTerms5 O pub lay c sf) pf.rs)

theorem pcsPhase5_iff (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout) (c : Rel5Container)
    (sf : RoundState) (pf : PcsState) :
    pcsPhase5 O pub lay c sf = .ok pf ↔ PcsSpec5 O pub lay c sf pf := by
  unfold pcsPhase5 PcsSpec5
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨p, hr, h⟩
    obtain ⟨hfin, hp⟩ := (guard_ok_iff _ _ _ _).mp h
    subst hp
    exact ⟨(runList_iff _ _ (pcsStep_iff O) _ _ _).mp hr, hfin⟩
  · rintro ⟨hr, hfin⟩
    exact ⟨pf, (runList_iff _ _ (pcsStep_iff O) _ _ _).mpr hr,
      (guard_ok_iff _ _ _ _).mpr ⟨hfin, rfl⟩⟩

/-! ## Check 17: dual layer-0 leaves, coset Merkle paths, fold consistency (spec §5.6.4) -/

/-- `root_0` from `STMT`, then the committed fold roots `root_1 .. root_{T−1}` of `H = F + θ C`. -/
def vecRoots (b : VecBody) : List Bytes := b.stmt.root0 :: b.pcs.head.map (fun f => f.2.2)

/-- The transcript the queries are squeezed from: after `absorb_bytes(FINAL)`. -/
def queryStart5 (O : Rel5Oracles) (c : Rel5Container) (pf : PcsState) : Fpn2Transcript :=
  pf.tr.absorbBytes O.hFs (Codec.fp3.enc c.body.pcs.final)

/-- Normative check 17 over all `Q` queries, with `θ = r_0` and fold challenges `r_2..r_{n+1}`.
Each query is the relation-4 step `queryStep4` with the pinned `weight_root` in the place of
`root_circ`: the `F` leaf authenticates to `root_0` and the weight leaf to `weight_root` at the
same coset (else `E_MERKLE`), `F + θ C` is folded, the committed layers are walked (`E_MERKLE` /
`E_FOLD`) and the last fold equals `FINAL` (else `E_FOLD`). -/
def queryPhase5 (O : Rel5Oracles) (lay : Rel5Layout) (c : Rel5Container) (pf : PcsState) :
    Except RejectCode Unit :=
  andThen (ofOption rel3IotaInv .E_FINAL) fun iotaInv =>
    andThen (runList
      (queryStep4 O lay.air (vecRoots c.body) c.body.stmt.weightRoot (pf.rs.drop 2)
        (pf.rs.getD 0 Fp3.zero) c.body.pcs.final iotaInv)
      (queryStart5 O c pf) c.body.queries) fun _ => .ok ()

def QuerySpec5 (O : Rel5Oracles) (lay : Rel5Layout) (c : Rel5Container) (pf : PcsState) : Prop :=
  ∃ iotaInv trF, rel3IotaInv = some iotaInv ∧
    RunList
      (QueryStepSpec4 O lay.air (vecRoots c.body) c.body.stmt.weightRoot (pf.rs.drop 2)
        (pf.rs.getD 0 Fp3.zero) c.body.pcs.final iotaInv)
      (queryStart5 O c pf) c.body.queries trF

theorem queryPhase5_iff (O : Rel5Oracles) (lay : Rel5Layout) (c : Rel5Container) (pf : PcsState) :
    queryPhase5 O lay c pf = .ok () ↔ QuerySpec5 O lay c pf := by
  unfold queryPhase5 QuerySpec5
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨iotaInv, hi, h⟩
    rw [andThen_ok_iff] at h
    obtain ⟨trF, hr, _⟩ := h
    exact ⟨iotaInv, trF, (ofOption_ok_iff _ _ _).mp hi,
      (runList_iff _ _ (queryStep4_iff O lay.air _ _ _ _ _ iotaInv) _ _ _).mp hr⟩
  · rintro ⟨iotaInv, trF, hi, hr⟩
    refine ⟨iotaInv, (ofOption_ok_iff _ _ _).mpr hi, ?_⟩
    rw [andThen_ok_iff]
    exact ⟨trF, (runList_iff _ _ (queryStep4_iff O lay.air _ _ _ _ _ iotaInv) _ _ _).mpr hr, rfl⟩

/-! ## The typed specification and the bytes verifier -/

/-- Normative checks 14–17 on a parsed container with a known layout. -/
def typedCheckRel5 (O : Rel5Oracles) (pub : PublicInput) (lay : Rel5Layout) (c : Rel5Container) :
    Except RejectCode Unit :=
  andThen (ipPhase O pub lay c) fun sf =>
    andThen (pcsPhase5 O pub lay c sf) fun pf => queryPhase5 O lay c pf

/-- Checks 14–17 against a profile (its layout exists by check 10). -/
def rel5Check (O : Rel5Oracles) (pub : PublicInput) (prof : Fpn2Profile) (c : Rel5Container) :
    Except RejectCode Unit :=
  andThen (ofOption (rel5Layout prof) .E_BACKEND_UNAVAILABLE) fun lay => typedCheckRel5 O pub lay c

/-- **Typed acceptance predicate of relation 5** on a parsed container: the profile has a
relation-5 layout; the `d` inner-product round sums hold from `λ · G_sum`, at challenges squeezed
after the row point `r_i` and the tile point `r_b`; the Basefold round sums hold from the batched
claim `Y = target + μ e_W + μ² z_blind` and end in `FINAL · W̃(θ, r_1, r')`; every query's `F`
leaf is anchored in `root_0`, its weight leaf in `STMT.weight_root`, the committed layers in
`root_1 .. root_{T−1}`, and the fused folds of `F + θ C` chain and end in `FINAL`. -/
def Rel5TypedVerify (O : Rel5Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (c : Rel5Container) : Prop :=
  ∃ lay sf pf, rel5Layout prof = some lay ∧ IpSpec O pub lay c sf ∧ PcsSpec5 O pub lay c sf pf ∧
    QuerySpec5 O lay c pf

/-- The executable checks 14–17 decide exactly the typed acceptance predicate. -/
theorem rel5Check_iff (O : Rel5Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (c : Rel5Container) : rel5Check O pub prof c = .ok () ↔ Rel5TypedVerify O pub prof c := by
  unfold rel5Check Rel5TypedVerify typedCheckRel5
  rw [andThen_ok_iff]
  constructor
  · rintro ⟨lay, hlay, h⟩
    rw [andThen_ok_iff] at h
    obtain ⟨sf, hs, h⟩ := h
    rw [andThen_ok_iff] at h
    obtain ⟨pf, hp, hq⟩ := h
    exact ⟨lay, sf, pf, (ofOption_ok_iff _ _ _).mp hlay, (ipPhase_iff _ _ _ _ _).mp hs,
      (pcsPhase5_iff _ _ _ _ _ _).mp hp, (queryPhase5_iff _ _ _ _).mp hq⟩
  · rintro ⟨lay, sf, pf, hlay, hs, hp, hq⟩
    refine ⟨lay, (ofOption_ok_iff _ _ _).mpr hlay, ?_⟩
    rw [andThen_ok_iff]
    refine ⟨sf, (ipPhase_iff _ _ _ _ _).mpr hs, ?_⟩
    rw [andThen_ok_iff]
    exact ⟨pf, (pcsPhase5_iff _ _ _ _ _ _).mpr hp, (queryPhase5_iff _ _ _ _).mpr hq⟩

/-- **`verifyBytesRel5`**: full verification of raw relation-5 container bytes against a pinned
profile and a pinned weight commitment, in the normative order of spec §6. `allowReduced` is
`--allow-reduced-security`, `weightPin` is `--circuit-root` (`Policy::expected_circuit_root`).

* 1–5   header: length `≥ 64`, magic, version, flags, reserved (`parseHeader`);
* 6     `profile_id` is registered, else `E_PROFILE_ID`;
* 7     it is the pinned profile, else `E_PROFILE_PINNED`;
* 8     tier is `STANDARD` or `allowReduced`, else `E_REDUCED_SECURITY`;
* 9–12  exact length `C_p`, relation-5 layout, zero `PAD`, canonical fields (`parseBodyRel5`);
* 13    `weightPin` is given (else `E_CIRCUIT_UNPINNED`) and equals `STMT.weight_root` (else
        `E_CIRCUIT_PINNED`): relation 5 fails closed;
* 14–17 inner-product round sums (`E_ZEROCHECK`), batched Basefold round sums (`E_SUMCHECK`),
        final evaluation (`E_FINAL`), dual layer-0 leaves, coset Merkle paths (`E_MERKLE`) and
        fold consistency (`E_FOLD`) (`rel5Check`).

A profile of another relation is `E_BACKEND_UNAVAILABLE` here (`verifyBytesAny` in `Fpn2Rel5`
dispatches). -/
def verifyBytesRel5 (O : Rel5Oracles) (pub : PublicInput) (prof : Fpn2Profile) (allowReduced : Bool)
    (weightPin : Option Bytes) (raw : ByteArray) : Except RejectCode Rel5Claim :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    match lookupProfile h.profileId with
    | none => .error .E_PROFILE_ID
    | some found =>
      if found ≠ prof then .error .E_PROFILE_PINNED
      else if (prof.isProduction || allowReduced) = false then .error .E_REDUCED_SECURITY
      else
        match parseBodyRel5 prof (toBytes raw) with
        | .error e => .error e
        | .ok b =>
          match weightPin with
          | none => .error .E_CIRCUIT_UNPINNED
          | some pin =>
            if pin ≠ b.stmt.weightRoot then .error .E_CIRCUIT_PINNED
            else
              match rel5Check O pub prof ⟨h, b⟩ with
              | .error e => .error e
              | .ok _ => .ok (Rel5Container.acceptedClaim ⟨h, b⟩)

/-- `verify_bytes` restricted to relation 5, with the caller policy of
`core/src/verify.rs::Policy`: an optional pinned `expected_profile_id` (check 7), the
header-selected registered profile, and the weight pin. -/
def verifyBytesRel5Policy (O : Rel5Oracles) (pub : PublicInput) (allowReduced : Bool)
    (expectedId : Option Bytes) (weightPin : Option Bytes) (raw : ByteArray) :
    Except RejectCode Rel5Claim :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    match lookupProfile h.profileId with
    | none => .error .E_PROFILE_ID
    | some found =>
      if expectedId.isSome ∧ expectedId ≠ some h.profileId then .error .E_PROFILE_PINNED
      else verifyBytesRel5 O pub found allowReduced weightPin raw

/-- Everything `verifyBytesRel5` established on an accepted input, as one case analysis. -/
theorem verifyBytesRel5_spec {O : Rel5Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {weightPin : Option Bytes} {raw : ByteArray} {claim : Rel5Claim}
    (h : verifyBytesRel5 O pub prof allowReduced weightPin raw = .ok claim) :
    ∃ c : Rel5Container,
      parseHeader raw = .ok c.header ∧ lookupProfile c.header.profileId = some prof ∧
      (prof.isProduction || allowReduced) = true ∧ parseBodyRel5 prof (toBytes raw) = .ok c.body ∧
      weightPin = some c.body.stmt.weightRoot ∧
      rel5Check O pub prof c = .ok () ∧ claim = c.acceptedClaim := by
  unfold verifyBytesRel5 at h
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
          cases hb : parseBodyRel5 prof (toBytes raw) with
          | error e => simp [hb] at h
          | ok b =>
            simp only [hb] at h
            cases weightPin with
            | none => cases h
            | some pin =>
              simp only at h
              split at h
              · cases h
              · rename_i hpin
                cases ht : rel5Check O pub prof ⟨hd, b⟩ with
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

/-- **Refinement theorem (relation 5).** For every byte string `raw`, if `verifyBytesRel5`
accepts it then:

(a) `raw` parses, against the pinned and registered profile, to a container `c`; `raw` has
    exactly `C_p` bytes, is byte-for-byte the canonical serialization of `c`, and every byte
    after the last typed field is zero;
(b) the profile is production-tier, or the caller explicitly opted into reduced security;
(c) the caller pinned a weight commitment and it is exactly the `weight_root` carried in `STMT`;
(d) the typed algebraic checks `Rel5TypedVerify` hold on `c`, and the returned claim is the
    statement carried by `c`. -/
theorem verifyBytesRel5_refines_typed {O : Rel5Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {weightPin : Option Bytes} {raw : ByteArray} {claim : Rel5Claim}
    (h : verifyBytesRel5 O pub prof allowReduced weightPin raw = .ok claim) :
    ∃ c : Rel5Container,
      (parseContainerRel5 prof raw = .ok c ∧ prof ∈ profileRegistry ∧
        raw.size = prof.containerBytes ∧ toBytes raw = encodeContainerRel5 prof c ∧
        ∃ lay, rel5Layout prof = some lay ∧ headerLen + lay.bodyUsed ≤ prof.containerBytes ∧
          ∀ i, headerLen + lay.bodyUsed ≤ i → i < raw.size → (toBytes raw)[i]? = some 0) ∧
      (prof.isProduction = true ∨ allowReduced = true) ∧
      weightPin = some c.body.stmt.weightRoot ∧
      (Rel5TypedVerify O pub prof c ∧ claim = c.acceptedClaim) := by
  obtain ⟨c, hh, hl, hpol, hb, hpin, ht, hc⟩ := verifyBytesRel5_spec h
  obtain ⟨hreg, hid⟩ := lookupProfile_spec hl
  have hparse : parseContainerRel5 prof raw = .ok c := by
    unfold parseContainerRel5
    simp [hh, hid, hb]
  obtain ⟨hsize, lay, hlay, hfit, _, hzero⟩ := parseContainerRel5_length_and_zero_pad hparse
  refine ⟨c, ⟨hparse, hreg, hsize, parseContainerRel5_encode hparse, lay, hlay, hfit, hzero⟩, ?_,
    hpin, (rel5Check_iff _ _ _ _).mp ht, hc⟩
  simpa [Bool.or_eq_true] using hpol

/-- **Completeness of the refinement.** A canonical container of a registered profile that
passes the tier policy, whose `STMT.weight_root` is the caller's pin and which satisfies
`Rel5TypedVerify`, is accepted, with its own statement. -/
theorem verifyBytesRel5_complete {O : Rel5Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {weightPin : Option Bytes} {raw : ByteArray} {c : Rel5Container}
    (hparse : parseContainerRel5 prof raw = .ok c)
    (hreg : lookupProfile prof.profileId = some prof)
    (hpol : prof.isProduction = true ∨ allowReduced = true)
    (hpin : weightPin = some c.body.stmt.weightRoot)
    (ht : Rel5TypedVerify O pub prof c) :
    verifyBytesRel5 O pub prof allowReduced weightPin raw = .ok c.acceptedClaim := by
  obtain ⟨hh, hid, hb⟩ := parseContainerRel5_spec hparse
  have hpol' : (prof.isProduction || allowReduced) = true := by
    simpa [Bool.or_eq_true] using hpol
  subst hpin
  unfold verifyBytesRel5
  simp [hh, hid, hreg, hpol', hb, (rel5Check_iff O pub prof c).mpr ht]

/-- **No silent downgrade.** Without the explicit opt-in, only a production-tier profile can be
accepted. -/
theorem verifyBytesRel5_downgrade_resistant {O : Rel5Oracles} {pub : PublicInput}
    {prof : Fpn2Profile} {weightPin : Option Bytes} {raw : ByteArray} {claim : Rel5Claim}
    (h : verifyBytesRel5 O pub prof false weightPin raw = .ok claim) :
    prof.isProduction = true := by
  obtain ⟨_, _, hpol, _⟩ := verifyBytesRel5_refines_typed h
  rcases hpol with hp | hf
  · exact hp
  · cases hf

/-- A `REDUCED` relation-5 profile is rejected on every input unless the caller opts in. -/
theorem verifyBytesRel5_rejects_reduced (O : Rel5Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (weightPin : Option Bytes) (raw : ByteArray) (hred : prof.isProduction = false)
    (claim : Rel5Claim) : verifyBytesRel5 O pub prof false weightPin raw ≠ .ok claim := by
  intro h
  rw [verifyBytesRel5_downgrade_resistant h] at hred
  cases hred

/-- **No second accepted encoding.** Two accepted byte strings carrying the same parsed
container are equal; in particular flipping any padding byte of an accepted proof is rejected. -/
theorem verifyBytesRel5_injective_on_accepted {O : Rel5Oracles} {pub : PublicInput}
    {prof : Fpn2Profile} {a₁ a₂ : Bool} {r₁ r₂ : Option Bytes} {raw₁ raw₂ : ByteArray}
    {claim₁ claim₂ : Rel5Claim} {c : Rel5Container}
    (h₁ : verifyBytesRel5 O pub prof a₁ r₁ raw₁ = .ok claim₁)
    (h₂ : verifyBytesRel5 O pub prof a₂ r₂ raw₂ = .ok claim₂)
    (hc₁ : parseContainerRel5 prof raw₁ = .ok c) (hc₂ : parseContainerRel5 prof raw₂ = .ok c) :
    raw₁ = raw₂ ∧ claim₁ = claim₂ ∧ r₁ = r₂ := by
  obtain ⟨c₁, ⟨hp₁, _⟩, _, hpin₁, _, hcl₁⟩ := verifyBytesRel5_refines_typed h₁
  obtain ⟨c₂, ⟨hp₂, _⟩, _, hpin₂, _, hcl₂⟩ := verifyBytesRel5_refines_typed h₂
  rw [hc₁] at hp₁
  rw [hc₂] at hp₂
  injection hp₁ with e₁
  injection hp₂ with e₂
  subst e₁ e₂
  exact ⟨parseContainerRel5_injective_on_accepted hc₁ hc₂, hcl₁.trans hcl₂.symm,
    hpin₁.trans hpin₂.symm⟩

/-- **Mandatory weight pin.** Relation 5 fails closed: an accepted container was verified
against a caller-supplied weight commitment, and that commitment is the `weight_root` of the
accepted claim. In particular no input is accepted without a pin, and one container is never
accepted under two different pins. -/
theorem verifyBytesRel5_requires_weight_pin {O : Rel5Oracles} {pub : PublicInput}
    {prof : Fpn2Profile} {allowReduced : Bool} {weightPin : Option Bytes} {raw : ByteArray}
    {claim : Rel5Claim}
    (h : verifyBytesRel5 O pub prof allowReduced weightPin raw = .ok claim) :
    weightPin = some claim.weightRoot ∧ weightPin ≠ none := by
  obtain ⟨c, _, _, _, _, hpin, _, hc⟩ := verifyBytesRel5_spec h
  have hroot : claim.weightRoot = c.body.stmt.weightRoot := by rw [hc]; rfl
  exact ⟨by rw [hpin, hroot], by rw [hpin]; intro h; cases h⟩

/-- An unpinned relation-5 verification never accepts. -/
theorem verifyBytesRel5_unpinned_rejects (O : Rel5Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (allowReduced : Bool) (raw : ByteArray) (claim : Rel5Claim) :
    verifyBytesRel5 O pub prof allowReduced none raw ≠ .ok claim := by
  intro h
  exact (verifyBytesRel5_requires_weight_pin h).2 rfl

/-- An acceptance under the caller policy is an acceptance under a registered profile, which is
the pinned one whenever a pin was given. -/
theorem verifyBytesRel5Policy_spec {O : Rel5Oracles} {pub : PublicInput} {allowReduced : Bool}
    {expectedId : Option Bytes} {weightPin : Option Bytes} {raw : ByteArray} {claim : Rel5Claim}
    (h : verifyBytesRel5Policy O pub allowReduced expectedId weightPin raw = .ok claim) :
    ∃ prof, prof ∈ profileRegistry ∧
      verifyBytesRel5 O pub prof allowReduced weightPin raw = .ok claim ∧
      ∀ id, expectedId = some id → prof.profileId = id := by
  unfold verifyBytesRel5Policy at h
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

/-! ## The profile-generic core of `verifyBytesRel5`, and the registered 64-tile profile -/

/-- Checks 1–5 and 9–17 of `verifyBytesRel5` as one function of an arbitrary profile: the strict
parse `parseContainerRel5` (header, the header names the profile, exact length, layout, zero
padding, canonical fields), the mandatory weight pin, then `rel5Check`. The registry lookup and
the tier policy (checks 6–8) are not part of it. `tests/Rel5Vectors.lean` runs this function on the
multi-tile fixture of an unregistered test profile. -/
def verifyParsedRel5 (O : Rel5Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (weightPin : Option Bytes) (raw : ByteArray) : Except RejectCode Rel5Claim :=
  match parseContainerRel5 prof raw with
  | .error e => .error e
  | .ok c =>
    match weightPin with
    | none => .error .E_CIRCUIT_UNPINNED
    | some pin =>
      if pin ≠ c.body.stmt.weightRoot then .error .E_CIRCUIT_PINNED
      else
        match rel5Check O pub prof c with
        | .error e => .error e
        | .ok _ => .ok c.acceptedClaim

/-- What an acceptance by `verifyParsedRel5` is. -/
theorem verifyParsedRel5_ok_iff {O : Rel5Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {weightPin : Option Bytes} {raw : ByteArray} {claim : Rel5Claim} :
    verifyParsedRel5 O pub prof weightPin raw = .ok claim ↔
      ∃ c, parseContainerRel5 prof raw = .ok c ∧ weightPin = some c.body.stmt.weightRoot ∧
        rel5Check O pub prof c = .ok () ∧ claim = c.acceptedClaim := by
  unfold verifyParsedRel5
  constructor
  · intro h
    cases hp : parseContainerRel5 prof raw with
    | error e => simp [hp] at h
    | ok c =>
      simp only [hp] at h
      cases weightPin with
      | none => cases h
      | some pin =>
        simp only at h
        split at h
        · cases h
        · rename_i hpin
          cases ht : rel5Check O pub prof c with
          | error e => simp [ht] at h
          | ok u =>
            simp only [ht] at h
            injection h with h
            refine ⟨c, rfl, ?_, ?_, h.symm⟩
            · rw [Decidable.not_not.mp hpin]
            · cases u; exact ht
  · rintro ⟨c, hp, rfl, ht, rfl⟩
    simp [hp, ht]

/-- **`verifyBytesRel5` is `verifyParsedRel5` behind the registry and the tier policy.** On a
registered profile that passes the tier policy, and on an input whose header names that profile,
the two functions return the same result: the same accepted claim or the same reject code. (On a
header naming another id they differ only in the code: `E_PROFILE_ID` / `E_PROFILE_PINNED` from the
registry lookup, against `E_PROFILE_PINNED` from the parser.) -/
theorem verifyBytesRel5_eq_verifyParsedRel5 {O : Rel5Oracles} {pub : PublicInput}
    {prof : Fpn2Profile} {allowReduced : Bool} {weightPin : Option Bytes} {raw : ByteArray}
    {h : Fpn2Header} (hreg : lookupProfile prof.profileId = some prof)
    (hpol : (prof.isProduction || allowReduced) = true)
    (hh : parseHeader raw = .ok h) (hid : h.profileId = prof.profileId) :
    verifyBytesRel5 O pub prof allowReduced weightPin raw =
      verifyParsedRel5 O pub prof weightPin raw := by
  cases hb : parseBodyRel5 prof (toBytes raw) with
  | error e =>
    simp [verifyBytesRel5, verifyParsedRel5, parseContainerRel5, hh, hid, hreg, hpol, hb]
  | ok b =>
    simp [verifyBytesRel5, verifyParsedRel5, parseContainerRel5, hh, hid, hreg, hpol, hb]

/-- On a registered profile that passes the tier policy, `verifyBytesRel5` accepts exactly what
`verifyParsedRel5` accepts, with the same claim, on every input. -/
theorem verifyBytesRel5_ok_iff_verifyParsedRel5 {O : Rel5Oracles} {pub : PublicInput}
    {prof : Fpn2Profile} {allowReduced : Bool} {weightPin : Option Bytes} {raw : ByteArray}
    {claim : Rel5Claim} (hreg : lookupProfile prof.profileId = some prof)
    (hpol : (prof.isProduction || allowReduced) = true) :
    verifyBytesRel5 O pub prof allowReduced weightPin raw = .ok claim ↔
      verifyParsedRel5 O pub prof weightPin raw = .ok claim := by
  constructor
  · intro h
    obtain ⟨c, hh, hl, _⟩ := verifyBytesRel5_spec h
    rw [← verifyBytesRel5_eq_verifyParsedRel5 hreg hpol hh (lookupProfile_spec hl).2.symm]
    exact h
  · intro h
    obtain ⟨c, hp, _⟩ := verifyParsedRel5_ok_iff.mp h
    obtain ⟨hh, hid, _⟩ := parseContainerRel5_spec hp
    rw [verifyBytesRel5_eq_verifyParsedRel5 hreg hpol hh hid]
    exact h

/-- `fpn2-folded-vec-n22-v1` is in the registry under its own id. -/
theorem lookupProfile_foldedVecN22 :
    lookupProfile foldedVecN22V1.profileId = some foldedVecN22V1 := by
  decide

/-- Checks 14–17 on `fpn2-folded-vec-n22-v1` are `typedCheckRel5` at the layout
`(n, Q, k) = (22, 72, 10)` of `rel5Layout_foldedVecN22`. -/
theorem rel5Check_foldedVecN22 (O : Rel5Oracles) (pub : PublicInput) (c : Rel5Container) :
    rel5Check O pub foldedVecN22V1 c = typedCheckRel5 O pub ⟨22, 72, 10⟩ c := by
  unfold rel5Check
  rw [rel5Layout_foldedVecN22.1]
  rfl

/-- **The registered 64-tile profile takes the profile-generic code path.** For
`fpn2-folded-vec-n22-v1` (`beta = 6`, 64 tiles, production tier), with or without
`--allow-reduced-security`:

(a) `verifyBytesRel5` accepts `raw` with `claim` iff `parseContainerRel5` parses it against the
    profile, the pin is the parsed `STMT.weight_root`, and `typedCheckRel5` holds at the layout
    `(22, 72, 10)`;
(b) on every input whose header names the profile, `verifyBytesRel5` returns exactly the result of
    `verifyParsedRel5` (claim or reject code), the function the multi-tile harness executes at
    `beta = 2`;
(c) that layout has `beta = 6` and `64` tiles.

This is a statement about which functions run, not an execution: no `beta = 6` container is run
through Lean. -/
theorem verifyBytesRel5_foldedVecN22 (O : Rel5Oracles) (pub : PublicInput) (allowReduced : Bool)
    (weightPin : Option Bytes) (raw : ByteArray) :
    (∀ claim, verifyBytesRel5 O pub foldedVecN22V1 allowReduced weightPin raw = .ok claim ↔
        ∃ c, parseContainerRel5 foldedVecN22V1 raw = .ok c ∧
          weightPin = some c.body.stmt.weightRoot ∧
          typedCheckRel5 O pub ⟨22, 72, 10⟩ c = .ok () ∧ claim = c.acceptedClaim) ∧
      (∀ h, parseHeader raw = .ok h → h.profileId = foldedVecN22V1.profileId →
        verifyBytesRel5 O pub foldedVecN22V1 allowReduced weightPin raw =
          verifyParsedRel5 O pub foldedVecN22V1 weightPin raw) ∧
      Rel5Layout.beta ⟨22, 72, 10⟩ = 6 ∧ Rel5Layout.tiles ⟨22, 72, 10⟩ = 64 := by
  have hpol : (foldedVecN22V1.isProduction || allowReduced) = true := by
    have hprod : foldedVecN22V1.isProduction = true := by decide
    rw [hprod]
    rfl
  refine ⟨fun claim => ?_, fun h hh hid =>
    verifyBytesRel5_eq_verifyParsedRel5 lookupProfile_foldedVecN22 hpol hh hid, by decide,
    by decide⟩
  rw [verifyBytesRel5_ok_iff_verifyParsedRel5 lookupProfile_foldedVecN22 hpol,
    verifyParsedRel5_ok_iff]
  simp only [rel5Check_foldedVecN22]

/-! ## One entry point: the relation dispatcher (`core/src/verify.rs::verify_bytes`) -/

/-- The caller policy of `core/src/verify.rs::Policy`. -/
structure VerifyPolicy where
  /-- `--allow-reduced-security`. -/
  allowReduced : Bool
  /-- `--profile`: pinned profile id. -/
  expectedProfileId : Option Bytes
  /-- `--circuit-root`: the pinned public commitment, relation 4's `root_circ` or relation 5's
  `weight_root`. Both fail closed. -/
  expectedCircuitRoot : Option Bytes

/-- The statement of an accepted container of any wired relation. -/
inductive AnyClaim where
  | harness (c : AcceptedClaim)
  | lookup (c : Rel3Claim)
  | air (c : Rel4Claim)
  | matmul (c : Rel5Claim)
  deriving DecidableEq, Repr

def liftClaim {α : Type} (f : α → AnyClaim) : Except RejectCode α → Except RejectCode AnyClaim
  | .error e => .error e
  | .ok a => .ok (f a)

theorem liftClaim_ok_iff {α : Type} (f : α → AnyClaim) (x : Except RejectCode α) (b : AnyClaim) :
    liftClaim f x = .ok b ↔ ∃ a, x = .ok a ∧ b = f a := by
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
      injection h with h
      exact ⟨a, rfl, h.symm⟩
    · rintro ⟨a', h, hb⟩
      injection h with h
      subst h hb
      rfl

/-- **`verifyBytesAny`**: full verification of raw container bytes of any relation, in the
normative order of spec §6, dispatching on the relation of the registered profile named by the
header exactly like `core/src/verify.rs::verify_bytes` (checks 1–7 here; 8–12 and the
relation-specific checks in the per-relation verifier). `O1` are the relation-1 oracles
(Poseidon2), `O` the BLAKE2b oracles of relations 3, 4 and 5. Relations 4 and 5 both take
`pol.expectedCircuitRoot` as their mandatory pin. Relation 2 (`LLM_INFERENCE_V2`) has no wired
backend: tier (8), exact length (9), then `E_BACKEND_UNAVAILABLE` (10). -/
def verifyBytesAny (O1 : Oracles) (O : Rel3Oracles) (pub : PublicInput) (pol : VerifyPolicy)
    (raw : ByteArray) : Except RejectCode AnyClaim :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    match lookupProfile h.profileId with
    | none => .error .E_PROFILE_ID
    | some found =>
      if pol.expectedProfileId.isSome ∧ pol.expectedProfileId ≠ some h.profileId then
        .error .E_PROFILE_PINNED
      else
        match found.relation with
        | .harnessSqSumcheck =>
          liftClaim AnyClaim.harness (verifyBytes O1 pub found pol.allowReduced raw)
        | .foldedGkrLookup =>
          liftClaim AnyClaim.lookup (verifyBytesRel3 O pub found pol.allowReduced raw)
        | .foldedGkrAir =>
          liftClaim AnyClaim.air
            (verifyBytesRel4 O pub found pol.allowReduced pol.expectedCircuitRoot raw)
        | .foldedVecMatmul =>
          liftClaim AnyClaim.matmul
            (verifyBytesRel5 O pub found pol.allowReduced pol.expectedCircuitRoot raw)
        | .llmInferenceV2 =>
          if (found.isProduction || pol.allowReduced) = false then .error .E_REDUCED_SECURITY
          else if raw.size ≠ found.containerBytes then .error .E_LENGTH
          else .error .E_BACKEND_UNAVAILABLE

/-- **Dispatcher specification.** An acceptance by `verifyBytesAny` is an acceptance by the
verifier of the relation of a registered profile, which is the pinned profile whenever a pin was
given; the claim constructor names that relation. -/
theorem verifyBytesAny_spec {O1 : Oracles} {O : Rel3Oracles} {pub : PublicInput}
    {pol : VerifyPolicy} {raw : ByteArray} {claim : AnyClaim}
    (h : verifyBytesAny O1 O pub pol raw = .ok claim) :
    ∃ prof, prof ∈ profileRegistry ∧ (∀ id, pol.expectedProfileId = some id → prof.profileId = id) ∧
      ((prof.relation = .harnessSqSumcheck ∧ ∃ c, claim = .harness c ∧
          verifyBytes O1 pub prof pol.allowReduced raw = .ok c) ∨
        (prof.relation = .foldedGkrLookup ∧ ∃ c, claim = .lookup c ∧
          verifyBytesRel3 O pub prof pol.allowReduced raw = .ok c) ∨
        (prof.relation = .foldedGkrAir ∧ ∃ c, claim = .air c ∧
          verifyBytesRel4 O pub prof pol.allowReduced pol.expectedCircuitRoot raw = .ok c) ∨
        (prof.relation = .foldedVecMatmul ∧ ∃ c, claim = .matmul c ∧
          verifyBytesRel5 O pub prof pol.allowReduced pol.expectedCircuitRoot raw = .ok c)) := by
  unfold verifyBytesAny at h
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
        have hid : ∀ id, pol.expectedProfileId = some id → found.profileId = id := by
          intro id hid
          rw [(lookupProfile_spec hl).2]
          by_cases he : some id = some hd.profileId
          · injection he with he
            exact he.symm
          · exact absurd ⟨by rw [hid]; rfl, by rw [hid]; exact he⟩ hpin
        refine ⟨found, (lookupProfile_spec hl).1, hid, ?_⟩
        cases hrel : found.relation with
        | harnessSqSumcheck =>
          simp only [hrel] at h
          obtain ⟨c, hv, hc⟩ := (liftClaim_ok_iff _ _ _).mp h
          exact Or.inl ⟨rfl, c, hc, hv⟩
        | foldedGkrLookup =>
          simp only [hrel] at h
          obtain ⟨c, hv, hc⟩ := (liftClaim_ok_iff _ _ _).mp h
          exact Or.inr (Or.inl ⟨rfl, c, hc, hv⟩)
        | foldedGkrAir =>
          simp only [hrel] at h
          obtain ⟨c, hv, hc⟩ := (liftClaim_ok_iff _ _ _).mp h
          exact Or.inr (Or.inr (Or.inl ⟨rfl, c, hc, hv⟩))
        | foldedVecMatmul =>
          simp only [hrel] at h
          obtain ⟨c, hv, hc⟩ := (liftClaim_ok_iff _ _ _).mp h
          exact Or.inr (Or.inr (Or.inr ⟨rfl, c, hc, hv⟩))
        | llmInferenceV2 =>
          simp only [hrel] at h
          split at h
          · cases h
          · split at h <;> cases h

/-- The dispatcher never accepts a relation-4 container without the circuit pin, and the accepted
claim's `root_circ` is that pin. -/
theorem verifyBytesAny_air_requires_circuit_pin {O1 : Oracles} {O : Rel3Oracles}
    {pub : PublicInput} {pol : VerifyPolicy} {raw : ByteArray} {c : Rel4Claim}
    (h : verifyBytesAny O1 O pub pol raw = .ok (.air c)) :
    pol.expectedCircuitRoot = some c.rootCirc := by
  obtain ⟨prof, _, _, hcase⟩ := verifyBytesAny_spec h
  rcases hcase with ⟨_, c', hc, _⟩ | ⟨_, c', hc, _⟩ | ⟨_, c', hc, hv⟩ | ⟨_, c', hc, _⟩
  · cases hc
  · cases hc
  · injection hc with hc
    subst hc
    exact (verifyBytesRel4_requires_circuit_pin hv).1
  · cases hc

/-- The dispatcher never accepts a relation-5 container without the weight pin, and the accepted
claim's `weight_root` is that pin. -/
theorem verifyBytesAny_matmul_requires_weight_pin {O1 : Oracles} {O : Rel3Oracles}
    {pub : PublicInput} {pol : VerifyPolicy} {raw : ByteArray} {c : Rel5Claim}
    (h : verifyBytesAny O1 O pub pol raw = .ok (.matmul c)) :
    pol.expectedCircuitRoot = some c.weightRoot := by
  obtain ⟨prof, _, _, hcase⟩ := verifyBytesAny_spec h
  rcases hcase with ⟨_, c', hc, _⟩ | ⟨_, c', hc, _⟩ | ⟨_, c', hc, _⟩ | ⟨_, c', hc, hv⟩
  · cases hc
  · cases hc
  · cases hc
  · injection hc with hc
    subst hc
    exact (verifyBytesRel5_requires_weight_pin hv).1

/-- **No silent downgrade through the dispatcher.** Without the explicit opt-in, an accepted
container names a registered production-tier profile. -/
theorem verifyBytesAny_downgrade_resistant {O1 : Oracles} {O : Rel3Oracles} {pub : PublicInput}
    {pol : VerifyPolicy} {raw : ByteArray} {claim : AnyClaim} (hpol : pol.allowReduced = false)
    (h : verifyBytesAny O1 O pub pol raw = .ok claim) :
    ∃ prof, prof ∈ profileRegistry ∧ prof.isProduction = true := by
  obtain ⟨prof, hreg, _, hcase⟩ := verifyBytesAny_spec h
  rw [hpol] at hcase
  refine ⟨prof, hreg, ?_⟩
  rcases hcase with ⟨_, c, _, hv⟩ | ⟨_, c, _, hv⟩ | ⟨_, c, _, hv⟩ | ⟨_, c, _, hv⟩
  · exact verifyBytes_downgrade_resistant hv
  · exact verifyBytesRel3_downgrade_resistant hv
  · exact verifyBytesRel4_downgrade_resistant hv
  · exact verifyBytesRel5_downgrade_resistant hv

end FlashProveNext

#print axioms FlashProveNext.rel5Layout_foldedVecN16
#print axioms FlashProveNext.rel5Layout_foldedVecN22
#print axioms FlashProveNext.rel5Layout_disjoint
#print axioms FlashProveNext.rel5Layout_bounds
#print axioms FlashProveNext.vecBody_enc
#print axioms FlashProveNext.parseBodyRel5_spec
#print axioms FlashProveNext.parseContainerRel5_spec
#print axioms FlashProveNext.parseContainerRel5_encode
#print axioms FlashProveNext.parseContainerRel5_length_and_zero_pad
#print axioms FlashProveNext.parseContainerRel5_injective_on_accepted
#print axioms FlashProveNext.ipRoundStep_iff
#print axioms FlashProveNext.ipPhase_iff
#print axioms FlashProveNext.pcsPhase5_iff
#print axioms FlashProveNext.queryPhase5_iff
#print axioms FlashProveNext.rel5Check_iff
#print axioms FlashProveNext.verifyBytesRel5_spec
#print axioms FlashProveNext.verifyBytesRel5_refines_typed
#print axioms FlashProveNext.verifyBytesRel5_complete
#print axioms FlashProveNext.verifyBytesRel5_downgrade_resistant
#print axioms FlashProveNext.verifyBytesRel5_rejects_reduced
#print axioms FlashProveNext.verifyBytesRel5_injective_on_accepted
#print axioms FlashProveNext.verifyBytesRel5_requires_weight_pin
#print axioms FlashProveNext.verifyBytesRel5_unpinned_rejects
#print axioms FlashProveNext.verifyBytesRel5Policy_spec
#print axioms FlashProveNext.verifyParsedRel5_ok_iff
#print axioms FlashProveNext.verifyBytesRel5_eq_verifyParsedRel5
#print axioms FlashProveNext.verifyBytesRel5_ok_iff_verifyParsedRel5
#print axioms FlashProveNext.lookupProfile_foldedVecN22
#print axioms FlashProveNext.rel5Check_foldedVecN22
#print axioms FlashProveNext.verifyBytesRel5_foldedVecN22
#print axioms FlashProveNext.liftClaim_ok_iff
#print axioms FlashProveNext.verifyBytesAny_spec
#print axioms FlashProveNext.verifyBytesAny_air_requires_circuit_pin
#print axioms FlashProveNext.verifyBytesAny_matmul_requires_weight_pin
#print axioms FlashProveNext.verifyBytesAny_downgrade_resistant
