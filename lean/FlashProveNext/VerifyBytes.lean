/-
Executable `verifyBytes` for `FPN2` relation-1 containers (`specs/11-wire-format.md` §5.3, §6) and
its refinement to a typed specification. Stage 1 of the Lean verifier (M4); Lean core only.

1. `fpAdd`, `fpMul`, `fpInv`, `Fp3.mul`, `lagrangeEval`, `basefoldFold`, `verifyPath` :
   executable Goldilocks / `F_{p^3}` arithmetic and the verification primitives of `core/src/verify.rs`;
2. `Oracles` : the three hash functions the verifier calls and Lean does not re-execute
   (`H_fs` = BLAKE2b-512, Poseidon2 leaf sponge and 2-to-1 compression), kept abstract so that
   a later stage can instantiate them with a random oracle;
3. `verifyBytes`, `verifyBytesUnpinned` : the normative 15-step check order with the spec codes;
4. `SumcheckRun`, `MerkleRun`, `TypedVerify` : the declarative typed statement (round sums,
   final evaluation, Merkle anchors along the Fiat–Shamir schedule) on a parsed container;
5. `sumcheckReplay_iff`, `merkleQueries_iff`, `typedCheck_iff` : the executable checks decide
   exactly the typed statement;
6. `verifyBytes_refines_typed` : every accepted byte string is a canonical, fixed-length,
   zero-padded container of a registered profile, accepted only at `STANDARD` tier or under an
   explicit opt-in, on which `TypedVerify` holds; `verifyBytes_complete` is the converse;
7. `verifyBytes_downgrade_resistant`, `verifyBytes_rejects_reduced`, `verifyBytes_injective_on_accepted`,
   `encodeContainer_sections` : corollaries (no silent downgrade, no second accepted encoding, the
   absorbed sections are exactly the wire bytes).

Scope. `TypedVerify` is the algebraic acceptance predicate of relation 1, a verification harness
whose binding `e = f̃(r)` is not enforced (spec §5.3 "Scope"); this module proves that the bytes
verifier refines that predicate, not that the predicate is a sound argument.
-/
import FlashProveNext.Fpn2Wire
import FlashProveNext.Fpn2Transcript

namespace FlashProveNext

/-! ## Goldilocks and `F_{p^3}` arithmetic (spec §1) -/

/-- Reduce a natural number into `[0, P)`. -/
def fpOfNat (n : Nat) : UInt64 := UInt64.ofNat (n % goldilocksP.toNat)

def fpAdd (a b : UInt64) : UInt64 := fpOfNat (a.toNat + b.toNat)

def fpSub (a b : UInt64) : UInt64 :=
  fpOfNat (a.toNat + goldilocksP.toNat - b.toNat % goldilocksP.toNat)

def fpMul (a b : UInt64) : UInt64 := fpOfNat (a.toNat * b.toNat)

/-- Square-and-multiply with explicit fuel (64 exponent bits). -/
def fpPowAux : Nat → UInt64 → Nat → UInt64 → UInt64
  | 0, _, _, acc => acc
  | fuel + 1, b, e, acc =>
    if e = 0 then acc
    else fpPowAux fuel (fpMul b b) (e / 2) (if e % 2 = 1 then fpMul acc b else acc)

def fpPow (b : UInt64) (e : Nat) : UInt64 := fpPowAux 64 b e 1

/-- Multiplicative inverse `a^(P−2)`; `none` for zero. -/
def fpInv (a : UInt64) : Option UInt64 :=
  if a.toNat % goldilocksP.toNat = 0 then none else some (fpPow a (goldilocksP.toNat - 2))

theorem fpOfNat_lt (n : Nat) : fpOfNat n < goldilocksP := by
  have hP : 0 < goldilocksP.toNat := by decide
  have hP' : goldilocksP.toNat < 2 ^ 64 := goldilocksP.toNat_lt
  have hmod := Nat.mod_lt n hP
  rw [UInt64.lt_iff_toNat_lt, fpOfNat, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  exact hmod

namespace Fp3

def zero : Fp3 := ⟨0, 0, 0⟩

def one : Fp3 := ⟨1, 0, 0⟩

def ofFp (x : UInt64) : Fp3 := ⟨x, 0, 0⟩

def add (a b : Fp3) : Fp3 := ⟨fpAdd a.a0 b.a0, fpAdd a.a1 b.a1, fpAdd a.a2 b.a2⟩

def sub (a b : Fp3) : Fp3 := ⟨fpSub a.a0 b.a0, fpSub a.a1 b.a1, fpSub a.a2 b.a2⟩

def scale (a : Fp3) (c : UInt64) : Fp3 := ⟨fpMul a.a0 c, fpMul a.a1 c, fpMul a.a2 c⟩

/-- Multiplication in `F_p[u]/(u^3 − 2)`:
`c0 = a0 b0 + 2 (a1 b2 + a2 b1)`, `c1 = a0 b1 + a1 b0 + 2 a2 b2`, `c2 = a0 b2 + a1 b1 + a2 b0`. -/
def mul (a b : Fp3) : Fp3 :=
  ⟨fpAdd (fpMul a.a0 b.a0) (fpMul 2 (fpAdd (fpMul a.a1 b.a2) (fpMul a.a2 b.a1))),
   fpAdd (fpAdd (fpMul a.a0 b.a1) (fpMul a.a1 b.a0)) (fpMul 2 (fpMul a.a2 b.a2)),
   fpAdd (fpAdd (fpMul a.a0 b.a2) (fpMul a.a1 b.a1)) (fpMul a.a2 b.a0)⟩

end Fp3

/-- Numerator `Π_{j ≠ i} (r − j)` and denominator `Π_{j ≠ i} (i − j)` of the `i`-th Lagrange
basis polynomial on the nodes `0, …, d − 1`. -/
def lagrangeBasis (d : Nat) (r : Fp3) (i : Nat) : Fp3 × UInt64 :=
  (List.range d).foldl
    (fun acc j =>
      if j = i then acc
      else (acc.1.mul (r.sub (Fp3.ofFp (fpOfNat j))), fpMul acc.2 (fpSub (fpOfNat i) (fpOfNat j))))
    (Fp3.one, 1)

/-- `lagrange_eval(g_0..g_d, r)`: the degree-`≤ d` polynomial with `g(j) = g_j`, evaluated at `r`. -/
def lagrangeEval (vals : List Fp3) (r : Fp3) : Fp3 :=
  (List.range vals.length).foldl
    (fun out i =>
      let t := lagrangeBasis vals.length r i
      out.add (((vals.getD i Fp3.zero).mul t.1).scale ((fpInv t.2).getD 0)))
    Fp3.zero

/-- One Basefold / FRI fold step `(f(x) + f(−x))/2 + r (f(x) − f(−x))/(2x)`
(`core/src/verify.rs::basefold_fold`). A reusable primitive for the relation-2 backend; the
relation-1 schedule of spec §5.3 has no fold check, so `verifyBytes` does not call it. -/
def basefoldFold (fx fmx : Fp3) (x : UInt64) (r : Fp3) : Option Fp3 :=
  match fpInv 2, fpInv (fpMul 2 x) with
  | some inv2, some inv2x => some (((fx.add fmx).scale inv2).add (r.mul ((fx.sub fmx).scale inv2x)))
  | _, _ => none

/-- Folding at `x = 0` is undefined (`2x` has no inverse). -/
theorem basefoldFold_zero (fx fmx r : Fp3) : basefoldFold fx fmx 0 r = none := by
  have h : fpInv (fpMul 2 0) = none := by decide
  unfold basefoldFold
  rw [h]
  cases fpInv 2 <;> rfl

/-! ## Hash oracles and Merkle anchors (spec §2–§3) -/

/-- The hash functions the verifier calls. They stay abstract in Stage 1. -/
structure Oracles where
  /-- `H_fs`: BLAKE2b-512, personal `"FPN2-FS-v2"`. -/
  hFs : ByteArray → ByteArray
  /-- `sponge_msg([v], DOM_LEAF = 64)`: the Poseidon2 leaf digest of a base-field value. -/
  leafDigest : UInt64 → Digest
  /-- Poseidon2 2-to-1 compression `compress(left, right)`. -/
  compress : Digest → Digest → Digest

/-- Walk an authentication path from the leaf level upward: at each level, if the low bit of
the index is 1 the running digest is the right child. Returns the root and the leftover index. -/
def merkleFold (compress : Digest → Digest → Digest) : Digest → Nat → List Digest → Digest × Nat
  | cur, idx, [] => (cur, idx)
  | cur, idx, sib :: rest =>
    merkleFold compress (if idx % 2 = 1 then compress sib cur else compress cur sib) (idx / 2) rest

/-- `verify_path`: the path leads from `leaf` at position `idx` to `root`, and `idx < 2^|path|`. -/
def verifyPath (O : Oracles) (root leaf : Digest) (idx : Nat) (path : List Digest) : Bool :=
  decide ((merkleFold O.compress leaf idx path).2 = 0 ∧ (merkleFold O.compress leaf idx path).1 = root)

theorem verifyPath_iff (O : Oracles) (root leaf : Digest) (idx : Nat) (path : List Digest) :
    verifyPath O root leaf idx path = true ↔ merkleFold O.compress leaf idx path = (root, 0) := by
  unfold verifyPath
  rw [decide_eq_true_iff]
  constructor
  · rintro ⟨h1, h2⟩
    exact Prod.ext h2 h1
  · intro h
    rw [h]
    exact ⟨rfl, rfl⟩

/-- `idx_q = int_le(squeeze_bytes(8)) mod 2^n`. -/
def queryIndex (n : Nat) (o : Bytes) : Nat := leNat o % 2 ^ n

/-! ## Statement, sections and the typed specification -/

/-- Out-of-band public parameters. Relation 1 carries its whole statement in-band (the `STMT`
section), so the only external input is the Fiat–Shamir domain label. -/
structure PublicInput where
  transcriptLabel : Bytes

/-- The normative public input: label `"FPN2/transcript/v2"`. -/
def PublicInput.v2 : PublicInput := ⟨labelV2⟩

/-- What an accepted container asserts: the profile and the `STMT` fields. -/
structure AcceptedClaim where
  profileId : Bytes
  rTab : Digest
  nS : Bytes
  stNew : Digest
  sum : Fp3
  deriving DecidableEq, Repr

def Fpn2Container.acceptedClaim (c : Fpn2Container) : AcceptedClaim :=
  ⟨c.header.profileId, c.body.rTab, c.body.nS, c.body.stNew, c.body.claim⟩

/-- The 120 `STMT` wire bytes. -/
def stmtBytes (b : HarnessBody) : Bytes := Codec.stmt.enc (b.rTab, b.nS, b.stNew, b.claim)

/-- The `(d+1)·24` wire bytes of one sumcheck round. -/
def encodeRound (g : List Fp3) : Bytes := encRep Codec.fp3 g

/-- The 24 `FINAL` wire bytes. -/
def finalBytes (b : HarnessBody) : Bytes := encodeFp3Bytes b.final

theorem encRep_append_flatMap {α : Type} (c : Codec α) (as : List α) :
    encRep c as = as.flatMap c.enc := by
  induction as with
  | nil => rfl
  | cons a as ih => simp [encRep, ih, List.flatMap_cons]

/-- The transcript absorbs raw wire bytes: the chunks absorbed by the verifier (`header`, `STMT`,
each round, `FINAL`), followed by the openings and the zero padding, concatenate to exactly the
canonical container, which by `parseContainer_encode` is the accepted input itself. -/
theorem encodeContainer_sections (prof : Fpn2Profile) (c : Fpn2Container) :
    encodeContainer prof c =
      encodeHeader c.header ++ (stmtBytes c.body ++ (c.body.rounds.flatMap encodeRound ++
        (finalBytes c.body ++ (encRep (Codec.opening prof) c.body.openings ++
          List.replicate (padLen prof) 0)))) := by
  have hr : encRep (Codec.round prof) c.body.rounds = c.body.rounds.flatMap encodeRound :=
    encRep_append_flatMap _ _
  rw [← hr]
  simp [encodeContainer, Codec.body, Codec.map, Codec.pair, Codec.rep, stmtBytes, finalBytes,
    encodeFp3Bytes, List.append_assoc]

/-- The transcript after `init`, `absorb_bytes(header)`, `absorb_bytes(STMT)`. -/
def startTranscript (O : Oracles) (pub : PublicInput) (c : Fpn2Container) : Fpn2Transcript :=
  ((Fpn2Transcript.init O.hFs pub.transcriptLabel).absorbBytes O.hFs
    (encodeHeader c.header)).absorbBytes O.hFs (stmtBytes c.body)

/-- The sumcheck schedule of spec §5.3 as a relation: from transcript `tr` and running `claim`,
each round absorbs its raw bytes, satisfies `g(0) + g(1) = claim`, squeezes `r` and continues with
`claim = g(r)`; the run ends in transcript `trF` with final claim `claimF`. -/
inductive SumcheckRun (O : Oracles) :
    Fpn2Transcript → Fp3 → List (List Fp3) → Fpn2Transcript → Fp3 → Prop where
  | nil (tr : Fpn2Transcript) (claim : Fp3) : SumcheckRun O tr claim [] tr claim
  | cons (tr : Fpn2Transcript) (claim g0 g1 : Fp3) (tl : List Fp3) (rest : List (List Fp3))
      (trF : Fpn2Transcript) (claimF : Fp3) :
      g0.add g1 = claim →
      SumcheckRun O
        ((tr.absorbBytes O.hFs (encodeRound (g0 :: g1 :: tl))).squeezeFp3 O.hFs).2
        (lagrangeEval (g0 :: g1 :: tl)
          ((tr.absorbBytes O.hFs (encodeRound (g0 :: g1 :: tl))).squeezeFp3 O.hFs).1)
        rest trF claimF →
      SumcheckRun O tr claim ((g0 :: g1 :: tl) :: rest) trF claimF

/-- The query schedule of spec §5.3 as a relation: every opening is a valid Merkle path from
`sponge_msg([f_q], 64)` at the squeezed index `idx_q` to the root `R_tab`. -/
inductive MerkleRun (O : Oracles) (root : Digest) (n : Nat) :
    Fpn2Transcript → List Opening → Prop where
  | nil (tr : Fpn2Transcript) : MerkleRun O root n tr []
  | cons (tr : Fpn2Transcript) (o : Opening) (rest : List Opening) :
      merkleFold O.compress (O.leafDigest o.value)
        (queryIndex n (tr.squeezeBytes O.hFs 8).1) o.siblings = (root, 0) →
      MerkleRun O root n (tr.squeezeBytes O.hFs 8).2 rest →
      MerkleRun O root n tr (o :: rest)

/-- **Typed acceptance predicate of relation 1** on a parsed container: all sumcheck round sums
hold along the Fiat–Shamir schedule, the last claim equals `e · e`, and every opening is anchored
in `R_tab`. -/
def TypedVerify (O : Oracles) (pub : PublicInput) (prof : Fpn2Profile) (c : Fpn2Container) : Prop :=
  ∃ trS claimF,
    SumcheckRun O (startTranscript O pub c) c.body.claim c.body.rounds trS claimF ∧
    claimF = c.body.final.mul c.body.final ∧
    MerkleRun O c.body.rTab prof.nVars (trS.absorbBytes O.hFs (finalBytes c.body)) c.body.openings

/-! ## Executable checks 13–15 -/

/-- Normative check 13 (`sumcheck_replay`): a round with fewer than two evaluations, or with
`g(0) + g(1) ≠ claim`, is `E_SUMCHECK`. -/
def sumcheckReplay (O : Oracles) :
    Fpn2Transcript → Fp3 → List (List Fp3) → Except RejectCode (Fpn2Transcript × Fp3)
  | tr, claim, [] => .ok (tr, claim)
  | _, _, [] :: _ => .error .E_SUMCHECK
  | _, _, [_] :: _ => .error .E_SUMCHECK
  | tr, claim, (g0 :: g1 :: tl) :: rest =>
    if g0.add g1 = claim then
      sumcheckReplay O
        ((tr.absorbBytes O.hFs (encodeRound (g0 :: g1 :: tl))).squeezeFp3 O.hFs).2
        (lagrangeEval (g0 :: g1 :: tl)
          ((tr.absorbBytes O.hFs (encodeRound (g0 :: g1 :: tl))).squeezeFp3 O.hFs).1)
        rest
    else .error .E_SUMCHECK

/-- Normative check 15: one squeezed index and one Merkle path per opening, else `E_MERKLE`. -/
def merkleQueries (O : Oracles) (root : Digest) (n : Nat) :
    Fpn2Transcript → List Opening → Except RejectCode Unit
  | _, [] => .ok ()
  | tr, o :: rest =>
    if verifyPath O root (O.leafDigest o.value)
        (queryIndex n (tr.squeezeBytes O.hFs 8).1) o.siblings = true then
      merkleQueries O root n (tr.squeezeBytes O.hFs 8).2 rest
    else .error .E_MERKLE

/-- Normative checks 13–15 on a parsed container. -/
def typedCheck (O : Oracles) (pub : PublicInput) (prof : Fpn2Profile) (c : Fpn2Container) :
    Except RejectCode Unit :=
  match sumcheckReplay O (startTranscript O pub c) c.body.claim c.body.rounds with
  | .error e => .error e
  | .ok (trS, claimF) =>
    if claimF = c.body.final.mul c.body.final then
      merkleQueries O c.body.rTab prof.nVars (trS.absorbBytes O.hFs (finalBytes c.body))
        c.body.openings
    else .error .E_FINAL

theorem sumcheckReplay_iff (O : Oracles) (tr : Fpn2Transcript) (claim : Fp3)
    (rounds : List (List Fp3)) (trF : Fpn2Transcript) (claimF : Fp3) :
    sumcheckReplay O tr claim rounds = .ok (trF, claimF) ↔
      SumcheckRun O tr claim rounds trF claimF := by
  constructor
  · intro h
    induction rounds generalizing tr claim with
    | nil =>
      simp only [sumcheckReplay] at h
      injection h with h
      injection h with h1 h2
      subst h1 h2
      exact .nil _ _
    | cons g rest ih =>
      match g, h with
      | [], h => simp [sumcheckReplay] at h
      | [_], h => simp [sumcheckReplay] at h
      | g0 :: g1 :: tl, h =>
        simp only [sumcheckReplay] at h
        split at h
        · rename_i hsum
          exact .cons _ _ _ _ _ _ _ _ hsum (ih _ _ h)
        · cases h
  · intro h
    induction h with
    | nil tr claim => simp [sumcheckReplay]
    | cons tr claim g0 g1 tl rest trF claimF hsum _ ih => simp [sumcheckReplay, hsum, ih]

theorem merkleQueries_iff (O : Oracles) (root : Digest) (n : Nat) (tr : Fpn2Transcript)
    (os : List Opening) :
    merkleQueries O root n tr os = .ok () ↔ MerkleRun O root n tr os := by
  constructor
  · intro h
    induction os generalizing tr with
    | nil => exact .nil _
    | cons o rest ih =>
      simp only [merkleQueries] at h
      split at h
      · rename_i hpath
        exact .cons _ _ _ ((verifyPath_iff _ _ _ _ _).mp hpath) (ih _ h)
      · cases h
  · intro h
    induction h with
    | nil tr => simp [merkleQueries]
    | cons tr o rest hpath _ ih =>
      simp [merkleQueries, (verifyPath_iff _ _ _ _ _).mpr hpath, ih]

/-- The executable checks 13–15 decide exactly the typed acceptance predicate. -/
theorem typedCheck_iff (O : Oracles) (pub : PublicInput) (prof : Fpn2Profile) (c : Fpn2Container) :
    typedCheck O pub prof c = .ok () ↔ TypedVerify O pub prof c := by
  unfold typedCheck TypedVerify
  constructor
  · intro h
    cases hs : sumcheckReplay O (startTranscript O pub c) c.body.claim c.body.rounds with
    | error e => simp [hs] at h
    | ok p =>
      obtain ⟨trS, claimF⟩ := p
      simp only [hs] at h
      split at h
      · rename_i hfin
        exact ⟨trS, claimF, (sumcheckReplay_iff _ _ _ _ _ _).mp hs, hfin,
          (merkleQueries_iff _ _ _ _ _).mp h⟩
      · cases h
  · rintro ⟨trS, claimF, hs, hfin, hm⟩
    rw [(sumcheckReplay_iff _ _ _ _ _ _).mpr hs]
    simp [hfin, (merkleQueries_iff _ _ _ _ _).mpr hm]

/-! ## The bytes verifier -/

/-- **`verifyBytes`**: full verification of raw container bytes against a pinned profile, in the
normative order of spec §6. `allowReduced` is the `--allow-reduced-security` policy flag.

* 1–5  header: length `≥ 64`, magic, version, flags, reserved (`parseHeader`);
* 6    `profile_id` is registered, else `E_PROFILE_ID`;
* 7    it is the pinned profile, else `E_PROFILE_PINNED`;
* 8    tier is `STANDARD` or `allowReduced`, else `E_REDUCED_SECURITY`;
* 9–12 exact length `C_p`, wired backend, zero `PAD`, canonical fields (`parseBody`);
* 13–15 sumcheck replay, final evaluation, Merkle anchors (`typedCheck`). -/
def verifyBytes (O : Oracles) (pub : PublicInput) (prof : Fpn2Profile) (allowReduced : Bool)
    (raw : ByteArray) : Except RejectCode AcceptedClaim :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    match lookupProfile h.profileId with
    | none => .error .E_PROFILE_ID
    | some found =>
      if found ≠ prof then .error .E_PROFILE_PINNED
      else if (prof.isProduction || allowReduced) = false then .error .E_REDUCED_SECURITY
      else
        match parseBody prof (toBytes raw) with
        | .error e => .error e
        | .ok b =>
          match typedCheck O pub prof ⟨h, b⟩ with
          | .error e => .error e
          | .ok _ => .ok (Fpn2Container.acceptedClaim ⟨h, b⟩)

/-- `verifyBytes` without a pin (`expected_profile_id` absent): the header selects the
registered profile, and check 7 is vacuous. -/
def verifyBytesUnpinned (O : Oracles) (pub : PublicInput) (allowReduced : Bool) (raw : ByteArray) :
    Except RejectCode AcceptedClaim :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    match lookupProfile h.profileId with
    | none => .error .E_PROFILE_ID
    | some found => verifyBytes O pub found allowReduced raw

/-- Everything `verifyBytes` established on an accepted input, as one case analysis. -/
theorem verifyBytes_spec {O : Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {raw : ByteArray} {claim : AcceptedClaim}
    (h : verifyBytes O pub prof allowReduced raw = .ok claim) :
    ∃ c : Fpn2Container,
      parseHeader raw = .ok c.header ∧ lookupProfile c.header.profileId = some prof ∧
      (prof.isProduction || allowReduced) = true ∧ parseBody prof (toBytes raw) = .ok c.body ∧
      typedCheck O pub prof c = .ok () ∧ claim = c.acceptedClaim := by
  unfold verifyBytes at h
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
          cases hb : parseBody prof (toBytes raw) with
          | error e => simp [hb] at h
          | ok b =>
            simp only [hb] at h
            cases ht : typedCheck O pub prof ⟨hd, b⟩ with
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

/-- **Refinement theorem.** For every byte string `raw`, if `verifyBytes` accepts it then:

(a) `raw` parses, against the pinned and registered profile, to a container `c`; `raw` has
    exactly `C_p` bytes, is byte-for-byte the canonical serialization of `c`, and every byte
    after the last typed field is zero;
(b) the profile is production-tier, or the caller explicitly opted into reduced security
    (downgrade resistance);
(c) the typed algebraic checks `TypedVerify` hold on `c`, and the returned claim is the
    statement carried by `c`. -/
theorem verifyBytes_refines_typed {O : Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {raw : ByteArray} {claim : AcceptedClaim}
    (h : verifyBytes O pub prof allowReduced raw = .ok claim) :
    ∃ c : Fpn2Container,
      (parseContainer prof raw = .ok c ∧ prof ∈ profileRegistry ∧
        raw.size = prof.containerBytes ∧ toBytes raw = encodeContainer prof c ∧
        ∃ used, prof.bodyUsed = some used ∧ headerLen + used ≤ prof.containerBytes ∧
          ∀ i, headerLen + used ≤ i → i < raw.size → (toBytes raw)[i]? = some 0) ∧
      (prof.isProduction = true ∨ allowReduced = true) ∧
      (TypedVerify O pub prof c ∧ claim = c.acceptedClaim) := by
  obtain ⟨c, hh, hl, hpol, hb, ht, hc⟩ := verifyBytes_spec h
  obtain ⟨hreg, hid⟩ := lookupProfile_spec hl
  have hparse : parseContainer prof raw = .ok c := by
    unfold parseContainer
    simp [hh, hid, hb]
  obtain ⟨hsize, used, hused, hfit, _, hzero⟩ := parseContainer_length_and_zero_pad hparse
  refine ⟨c, ⟨hparse, hreg, hsize, parseContainer_encode hparse, used, hused, hfit, hzero⟩, ?_,
    (typedCheck_iff _ _ _ _).mp ht, hc⟩
  simpa [Bool.or_eq_true] using hpol

/-- **Completeness of the refinement.** A canonical container of a registered profile that
passes the tier policy and satisfies `TypedVerify` is accepted, with its own statement. -/
theorem verifyBytes_complete {O : Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {allowReduced : Bool} {raw : ByteArray} {c : Fpn2Container}
    (hparse : parseContainer prof raw = .ok c)
    (hreg : lookupProfile prof.profileId = some prof)
    (hpol : prof.isProduction = true ∨ allowReduced = true)
    (ht : TypedVerify O pub prof c) :
    verifyBytes O pub prof allowReduced raw = .ok c.acceptedClaim := by
  obtain ⟨hh, hid, hb⟩ := parseContainer_spec hparse
  have hpol' : (prof.isProduction || allowReduced) = true := by
    simpa [Bool.or_eq_true] using hpol
  unfold verifyBytes
  simp [hh, hid, hreg, hpol', hb, (typedCheck_iff O pub prof c).mpr ht]

/-- **No silent downgrade.** Without the explicit opt-in, only a production-tier profile can be
accepted. -/
theorem verifyBytes_downgrade_resistant {O : Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {raw : ByteArray} {claim : AcceptedClaim}
    (h : verifyBytes O pub prof false raw = .ok claim) : prof.isProduction = true := by
  obtain ⟨_, _, hpol, _⟩ := verifyBytes_refines_typed h
  rcases hpol with hp | hf
  · exact hp
  · cases hf

/-- A `REDUCED` profile is rejected on every input unless the caller opts in. -/
theorem verifyBytes_rejects_reduced (O : Oracles) (pub : PublicInput) (prof : Fpn2Profile)
    (raw : ByteArray) (hred : prof.isProduction = false) (claim : AcceptedClaim) :
    verifyBytes O pub prof false raw ≠ .ok claim := by
  intro h
  rw [verifyBytes_downgrade_resistant h] at hred
  cases hred

/-- **No second accepted encoding.** Two accepted byte strings carrying the same parsed
container are equal; in particular flipping any padding byte of an accepted proof is rejected. -/
theorem verifyBytes_injective_on_accepted {O : Oracles} {pub : PublicInput} {prof : Fpn2Profile}
    {a₁ a₂ : Bool} {raw₁ raw₂ : ByteArray} {claim₁ claim₂ : AcceptedClaim} {c : Fpn2Container}
    (h₁ : verifyBytes O pub prof a₁ raw₁ = .ok claim₁)
    (h₂ : verifyBytes O pub prof a₂ raw₂ = .ok claim₂)
    (hc₁ : parseContainer prof raw₁ = .ok c) (hc₂ : parseContainer prof raw₂ = .ok c) :
    raw₁ = raw₂ ∧ claim₁ = claim₂ := by
  obtain ⟨c₁, ⟨hp₁, _⟩, _, _, hcl₁⟩ := verifyBytes_refines_typed h₁
  obtain ⟨c₂, ⟨hp₂, _⟩, _, _, hcl₂⟩ := verifyBytes_refines_typed h₂
  rw [hc₁] at hp₁
  rw [hc₂] at hp₂
  injection hp₁ with e₁
  injection hp₂ with e₂
  subst e₁ e₂
  exact ⟨parseContainer_injective_on_accepted hc₁ hc₂, hcl₁.trans hcl₂.symm⟩

/-- An unpinned acceptance is an acceptance under some registered profile. -/
theorem verifyBytesUnpinned_spec {O : Oracles} {pub : PublicInput} {allowReduced : Bool}
    {raw : ByteArray} {claim : AcceptedClaim}
    (h : verifyBytesUnpinned O pub allowReduced raw = .ok claim) :
    ∃ prof, prof ∈ profileRegistry ∧ verifyBytes O pub prof allowReduced raw = .ok claim := by
  unfold verifyBytesUnpinned at h
  cases hh : parseHeader raw with
  | error e => simp [hh] at h
  | ok hd =>
    simp only [hh] at h
    cases hl : lookupProfile hd.profileId with
    | none => simp [hl] at h
    | some found =>
      simp only [hl] at h
      exact ⟨found, (lookupProfile_spec hl).1, h⟩

end FlashProveNext

#print axioms FlashProveNext.fpOfNat_lt
#print axioms FlashProveNext.basefoldFold_zero
#print axioms FlashProveNext.verifyPath_iff
#print axioms FlashProveNext.encRep_append_flatMap
#print axioms FlashProveNext.encodeContainer_sections
#print axioms FlashProveNext.sumcheckReplay_iff
#print axioms FlashProveNext.merkleQueries_iff
#print axioms FlashProveNext.typedCheck_iff
#print axioms FlashProveNext.verifyBytes_spec
#print axioms FlashProveNext.verifyBytes_refines_typed
#print axioms FlashProveNext.verifyBytes_complete
#print axioms FlashProveNext.verifyBytes_downgrade_resistant
#print axioms FlashProveNext.verifyBytes_rejects_reduced
#print axioms FlashProveNext.verifyBytes_injective_on_accepted
#print axioms FlashProveNext.verifyBytesUnpinned_spec
