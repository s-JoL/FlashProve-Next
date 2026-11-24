/-
`FPN2` canonical wire format (`specs/11-wire-format.md`, `fpn-wire-2.0`): executable strict parser
with machine-checked canonical-encoding theorems. Stage 1 of the Lean `verifyBytes` verifier (M4).

This module is deliberately self-contained (Lean core only, no Mathlib import), so the trusted
executable base is small and the module compiles in seconds.

1. `leNat`, `natLE`, `decodeFpLE`, `encodeFpLE`, `decodeFp3LE`, `encodeFp3LE` :
   little-endian Goldilocks encodings (spec §1); a limb is accepted iff it is `< P`;
2. `Codec` : a decoder/encoder pair carrying the proof `dec l = ok (a, rest) → l = enc a ++ rest`,
   closed under pairing, fixed-count repetition and isomorphism (`Codec.pair`, `Codec.rep`, `Codec.map`);
3. `parseHeader`, `parseBody`, `parseContainer` : normative checks 1–5 and 9–12 of spec §6
   (the policy checks 6–8 live in `VerifyBytes`), with the spec reject codes;
4. `decodeFpLE_encodeFpLE`, `decodeFpLE_rejects_ge_P`, `parseHeader_magic_version_flags_reserved`,
   `parseContainer_length_and_zero_pad`, `parseContainer_encode`, `parseContainer_injective_on_accepted` :
   an accepted byte string is exactly `encodeContainer` of its parsed value, hence fixed-length,
   zero-padded and unique (no padding malleability, no second encoding of the same proof).

All parsing runs on `Bytes = List UInt8` (the `ByteArray` entry points convert once through `toBytes`).
-/

namespace FlashProveNext

/-! ## Bytes and little-endian integers -/

/-- Byte strings as lists; `ByteArray` entry points convert through `toBytes`. -/
abbrev Bytes := List UInt8

def toBytes (b : ByteArray) : Bytes := b.data.toList

def ofBytes (l : Bytes) : ByteArray := ⟨l.toArray⟩

theorem toBytes_ofBytes (l : Bytes) : toBytes (ofBytes l) = l := by
  simp [toBytes, ofBytes]

theorem ofBytes_toBytes (b : ByteArray) : ofBytes (toBytes b) = b := by
  cases b; simp [toBytes, ofBytes]

theorem toBytes_injective {a b : ByteArray} (h : toBytes a = toBytes b) : a = b := by
  rw [← ofBytes_toBytes a, ← ofBytes_toBytes b, h]

theorem toBytes_length (b : ByteArray) : (toBytes b).length = b.size := by
  cases b; rfl

/-- `int_le`: the natural number encoded little-endian by a byte string. -/
def leNat : Bytes → Nat
  | [] => 0
  | b :: bs => b.toNat + 256 * leNat bs

/-- `le_k(n)`: the `k`-byte little-endian encoding of `n` (truncating `n` modulo `256^k`). -/
def natLE : Nat → Nat → Bytes
  | 0, _ => []
  | k + 1, n => UInt8.ofNat (n % 256) :: natLE k (n / 256)

theorem natLE_length (k n : Nat) : (natLE k n).length = k := by
  induction k generalizing n with
  | zero => rfl
  | succ k ih => simp [natLE, ih]

theorem leNat_lt (l : Bytes) : leNat l < 256 ^ l.length := by
  induction l with
  | nil => simp [leNat]
  | cons b bs ih =>
    have hb : b.toNat < 256 := b.toNat_lt
    simp only [leNat, List.length_cons, Nat.pow_succ]
    omega

theorem leNat_natLE (k n : Nat) (h : n < 256 ^ k) : leNat (natLE k n) = n := by
  induction k generalizing n with
  | zero =>
    have : n = 0 := by simpa using h
    simp [natLE, leNat, this]
  | succ k ih =>
    have h' : n / 256 < 256 ^ k := by rw [Nat.pow_succ] at h; omega
    have hb : (UInt8.ofNat (n % 256)).toNat = n % 256 := by
      rw [UInt8.toNat_ofNat']; omega
    simp only [natLE, leNat, ih _ h', hb]
    omega

theorem natLE_leNat (l : Bytes) : natLE l.length (leNat l) = l := by
  induction l with
  | nil => rfl
  | cons b bs ih =>
    have hb : b.toNat < 256 := b.toNat_lt
    have h1 : (b.toNat + 256 * leNat bs) % 256 = b.toNat := by omega
    have h2 : (b.toNat + 256 * leNat bs) / 256 = leNat bs := by omega
    simp only [List.length_cons, natLE, leNat, h1, h2, ih, UInt8.ofNat_toNat]

/-- `le_k` is injective on `[0, 256^k)`. -/
theorem natLE_injective (k n m : Nat) (hn : n < 256 ^ k) (hm : m < 256 ^ k)
    (h : natLE k n = natLE k m) : n = m := by
  rw [← leNat_natLE k n hn, ← leNat_natLE k m hm, h]

/-- `l[a .. a+b) ‖ l[a+b .. a+b+c) = l[a .. a+b+c)`. -/
theorem take_drop_append (l : Bytes) (a b c : Nat) :
    (l.drop a).take b ++ (l.drop (a + b)).take c = (l.drop a).take (b + c) := by
  rw [List.take_add, List.drop_drop]

/-! ## Reject codes (spec §6) -/

/-- The twenty stable reject codes of `specs/11-wire-format.md` §6, in the order of
`core/src/verify.rs::RejectCode` (`E_GKR_ROOT`, `E_GKR` are raised only by relations 3 and 4,
`Fpn2Rel3` / `Fpn2Rel4`; `E_FOLD` by relations 3, 4 and 5; the last three only by relations 4 and 5,
`Fpn2Rel4` / `Fpn2Rel5`). -/
inductive RejectCode where
  | E_LENGTH
  | E_MAGIC
  | E_VERSION
  | E_FLAGS
  | E_RESERVED
  | E_PROFILE_ID
  | E_PROFILE_PINNED
  | E_REDUCED_SECURITY
  | E_BACKEND_UNAVAILABLE
  | E_PADDING
  | E_NONCANONICAL
  | E_SUMCHECK
  | E_FINAL
  | E_MERKLE
  | E_GKR_ROOT
  | E_GKR
  | E_FOLD
  | E_ZEROCHECK
  | E_CIRCUIT_UNPINNED
  | E_CIRCUIT_PINNED
  deriving DecidableEq, Repr

/-- The exact code string printed by every implementation (`REJECTED: <code>`). -/
def RejectCode.toString : RejectCode → String
  | .E_LENGTH => "E_LENGTH"
  | .E_MAGIC => "E_MAGIC"
  | .E_VERSION => "E_VERSION"
  | .E_FLAGS => "E_FLAGS"
  | .E_RESERVED => "E_RESERVED"
  | .E_PROFILE_ID => "E_PROFILE_ID"
  | .E_PROFILE_PINNED => "E_PROFILE_PINNED"
  | .E_REDUCED_SECURITY => "E_REDUCED_SECURITY"
  | .E_BACKEND_UNAVAILABLE => "E_BACKEND_UNAVAILABLE"
  | .E_PADDING => "E_PADDING"
  | .E_NONCANONICAL => "E_NONCANONICAL"
  | .E_SUMCHECK => "E_SUMCHECK"
  | .E_FINAL => "E_FINAL"
  | .E_MERKLE => "E_MERKLE"
  | .E_GKR_ROOT => "E_GKR_ROOT"
  | .E_GKR => "E_GKR"
  | .E_FOLD => "E_FOLD"
  | .E_ZEROCHECK => "E_ZEROCHECK"
  | .E_CIRCUIT_UNPINNED => "E_CIRCUIT_UNPINNED"
  | .E_CIRCUIT_PINNED => "E_CIRCUIT_PINNED"

instance : ToString RejectCode := ⟨RejectCode.toString⟩

/-! ## Field encodings (spec §1) -/

/-- Goldilocks `P = 2^64 − 2^32 + 1` (the same number as `RoundByRound.goldilocks`). -/
def goldilocksP : UInt64 := 0xffffffff00000001

theorem goldilocksP_toNat : goldilocksP.toNat = 2 ^ 64 - 2 ^ 32 + 1 := by decide

/-- `goldilocksP.toNat` as the literal used by `RoundByRound.goldilocks_eq`. -/
theorem goldilocksP_toNat_eq : goldilocksP.toNat = 18446744069414584321 := by decide

/-- Element `a0 + a1 u + a2 u^2` of `E = F_p[u]/(u^3 − 2)`; limb 0 is the constant term. -/
structure Fp3 where
  a0 : UInt64
  a1 : UInt64
  a2 : UInt64
  deriving DecidableEq, Repr

/-- A Poseidon2 digest: four `Fp` limbs. -/
structure Digest where
  d0 : UInt64
  d1 : UInt64
  d2 : UInt64
  d3 : UInt64
  deriving DecidableEq, Repr

/-- The little-endian `u64` value of a byte string (reduced modulo `2^64`). -/
def toUInt64LE (b : ByteArray) : UInt64 := UInt64.ofNat (leNat (toBytes b))

/-- Canonical `Fp` decoding of exactly eight bytes: `x ≥ P` is rejected, never reduced. -/
def decodeFpBytes (l : Bytes) : Except RejectCode UInt64 :=
  if l.length = 8 then
    if leNat l < goldilocksP.toNat then .ok (UInt64.ofNat (leNat l)) else .error .E_NONCANONICAL
  else .error .E_LENGTH

def encodeFpBytes (x : UInt64) : Bytes := natLE 8 x.toNat

def decodeFpLE (b : ByteArray) : Except RejectCode UInt64 := decodeFpBytes (toBytes b)

def encodeFpLE (x : UInt64) : ByteArray := ofBytes (encodeFpBytes x)

theorem encodeFpBytes_length (x : UInt64) : (encodeFpBytes x).length = 8 := natLE_length 8 _

theorem decodeFpBytes_encodeFpBytes (x : UInt64) (h : x < goldilocksP) :
    decodeFpBytes (encodeFpBytes x) = .ok x := by
  have hx : x.toNat < 256 ^ 8 := x.toNat_lt
  have hlt : x.toNat < goldilocksP.toNat := UInt64.lt_iff_toNat_lt.mp h
  simp [decodeFpBytes, encodeFpBytes, natLE_length, leNat_natLE 8 _ hx, hlt]

/-- Whatever `decodeFpBytes` accepts is the canonical encoding of the value it returns. -/
theorem encodeFpBytes_of_decodeFpBytes {l : Bytes} {x : UInt64} (h : decodeFpBytes l = .ok x) :
    encodeFpBytes x = l ∧ x < goldilocksP := by
  unfold decodeFpBytes at h
  split at h
  · rename_i h8
    split at h
    · rename_i hlt
      injection h with h
      have hP : goldilocksP.toNat < 2 ^ 64 := goldilocksP.toNat_lt
      have hx : x.toNat = leNat l := by
        rw [← h, UInt64.toNat_ofNat']; omega
      refine ⟨?_, UInt64.lt_iff_toNat_lt.mpr (by omega)⟩
      have := natLE_leNat l
      rw [h8] at this
      simpa [encodeFpBytes, hx] using this
    · cases h
  · cases h

/-- Round trip: every canonical `x < P` decodes back from its 8-byte encoding. -/
theorem decodeFpLE_encodeFpLE (x : UInt64) (h : x < goldilocksP) :
    decodeFpLE (encodeFpLE x) = .ok x := by
  simp [decodeFpLE, encodeFpLE, toBytes_ofBytes, decodeFpBytes_encodeFpBytes x h]

/-- Any 8-byte word whose little-endian `u64` value is `≥ P` (in particular `P`, `P + 1`,
`2^64 − 1`) is rejected with `E_NONCANONICAL` (the spec's code for a non-canonical `Fp`). -/
theorem decodeFpLE_rejects_ge_P (b : ByteArray) (h8 : b.size = 8) (h : goldilocksP ≤ toUInt64LE b) :
    decodeFpLE b = .error .E_NONCANONICAL := by
  have hl : (toBytes b).length = 8 := by rw [toBytes_length, h8]
  have hlt : leNat (toBytes b) < 256 ^ 8 := by have := leNat_lt (toBytes b); rwa [hl] at this
  have hge : goldilocksP.toNat ≤ leNat (toBytes b) := by
    have := UInt64.le_iff_toNat_le.mp h
    rw [toUInt64LE, UInt64.toNat_ofNat'] at this
    omega
  have hn : ¬ leNat (toBytes b) < goldilocksP.toNat := by omega
  simp [decodeFpLE, decodeFpBytes, hl, hn]

/-- An accepted 8-byte word is the unique canonical encoding of its value. -/
theorem encodeFpLE_of_decodeFpLE {b : ByteArray} {x : UInt64} (h : decodeFpLE b = .ok x) :
    encodeFpLE x = b ∧ x < goldilocksP := by
  obtain ⟨he, hx⟩ := encodeFpBytes_of_decodeFpBytes h
  exact ⟨by rw [encodeFpLE, he, ofBytes_toBytes], hx⟩

/-! ## Codecs: decoders that provably invert an encoder -/

/-- A decoder together with the encoder it inverts on every accepted input. `sound` is the
canonical-encoding property: an accepted prefix is exactly the encoding of the decoded value. -/
structure Codec (α : Type) where
  dec : Bytes → Except RejectCode (α × Bytes)
  enc : α → Bytes
  sound : ∀ l a rest, dec l = .ok (a, rest) → l = enc a ++ rest

def decFp (l : Bytes) : Except RejectCode (UInt64 × Bytes) :=
  match decodeFpBytes (l.take 8) with
  | .error e => .error e
  | .ok x => .ok (x, l.drop 8)

theorem decFp_sound (l : Bytes) (x : UInt64) (rest : Bytes) (h : decFp l = .ok (x, rest)) :
    l = encodeFpBytes x ++ rest := by
  unfold decFp at h
  cases hd : decodeFpBytes (l.take 8) with
  | error e => simp [hd] at h
  | ok y =>
    simp only [hd] at h
    injection h with h
    injection h with hx hr
    subst hx hr
    rw [(encodeFpBytes_of_decodeFpBytes hd).1, List.take_append_drop]

/-- One canonical `Fp` limb (8 bytes, `< P`). -/
def Codec.fp : Codec UInt64 := ⟨decFp, encodeFpBytes, decFp_sound⟩

def decRaw (n : Nat) (l : Bytes) : Except RejectCode (Bytes × Bytes) :=
  if n ≤ l.length then .ok (l.take n, l.drop n) else .error .E_LENGTH

theorem decRaw_sound (n : Nat) (l a rest : Bytes) (h : decRaw n l = .ok (a, rest)) :
    l = a ++ rest := by
  unfold decRaw at h
  split at h
  · injection h with h
    injection h with ha hr
    subst ha hr
    rw [List.take_append_drop]
  · cases h

/-- `n` opaque bytes (`Bytes32` for `n = 32`): always canonical. -/
def Codec.raw (n : Nat) : Codec Bytes := ⟨decRaw n, id, decRaw_sound n⟩

def decPair {α β : Type} (c₁ : Codec α) (c₂ : Codec β) (l : Bytes) :
    Except RejectCode ((α × β) × Bytes) :=
  match c₁.dec l with
  | .error e => .error e
  | .ok (a, r) =>
    match c₂.dec r with
    | .error e => .error e
    | .ok (b, r') => .ok ((a, b), r')

theorem decPair_sound {α β : Type} (c₁ : Codec α) (c₂ : Codec β) (l : Bytes) (p : α × β)
    (rest : Bytes) (h : decPair c₁ c₂ l = .ok (p, rest)) :
    l = (c₁.enc p.1 ++ c₂.enc p.2) ++ rest := by
  unfold decPair at h
  cases h1 : c₁.dec l with
  | error e => simp [h1] at h
  | ok x =>
    obtain ⟨a, r⟩ := x
    simp only [h1] at h
    cases h2 : c₂.dec r with
    | error e => simp [h2] at h
    | ok y =>
      obtain ⟨b, r'⟩ := y
      simp only [h2] at h
      injection h with h
      injection h with hp hr
      subst hp hr
      have e1 := c₁.sound _ _ _ h1
      have e2 := c₂.sound _ _ _ h2
      rw [e1, e2, List.append_assoc]

/-- Two fields back-to-back, first-failure-wins in wire order. -/
def Codec.pair {α β : Type} (c₁ : Codec α) (c₂ : Codec β) : Codec (α × β) :=
  ⟨decPair c₁ c₂, fun p => c₁.enc p.1 ++ c₂.enc p.2, decPair_sound c₁ c₂⟩

def decRep {α : Type} (c : Codec α) : Nat → Bytes → Except RejectCode (List α × Bytes)
  | 0, l => .ok ([], l)
  | n + 1, l =>
    match c.dec l with
    | .error e => .error e
    | .ok (a, r) =>
      match decRep c n r with
      | .error e => .error e
      | .ok (as, r') => .ok (a :: as, r')

def encRep {α : Type} (c : Codec α) : List α → Bytes
  | [] => []
  | a :: as => c.enc a ++ encRep c as

theorem decRep_sound {α : Type} (c : Codec α) (n : Nat) (l : Bytes) (as : List α) (rest : Bytes)
    (h : decRep c n l = .ok (as, rest)) : l = encRep c as ++ rest := by
  induction n generalizing l as rest with
  | zero =>
    simp only [decRep] at h
    injection h with h
    injection h with ha hr
    subst ha hr
    rfl
  | succ n ih =>
    simp only [decRep] at h
    cases h1 : c.dec l with
    | error e => simp [h1] at h
    | ok x =>
      obtain ⟨a, r⟩ := x
      simp only [h1] at h
      cases h2 : decRep c n r with
      | error e => simp [h2] at h
      | ok y =>
        obtain ⟨as', r'⟩ := y
        simp only [h2] at h
        injection h with h
        injection h with ha hr
        subst ha hr
        have e1 := c.sound _ _ _ h1
        have e2 := ih _ _ _ h2
        rw [e1, e2, encRep, List.append_assoc]

theorem decRep_length {α : Type} (c : Codec α) (n : Nat) (l : Bytes) (as : List α) (rest : Bytes)
    (h : decRep c n l = .ok (as, rest)) : as.length = n := by
  induction n generalizing l as rest with
  | zero =>
    simp only [decRep] at h
    injection h with h
    injection h with ha hr
    subst ha
    rfl
  | succ n ih =>
    simp only [decRep] at h
    cases h1 : c.dec l with
    | error e => simp [h1] at h
    | ok x =>
      obtain ⟨a, r⟩ := x
      simp only [h1] at h
      cases h2 : decRep c n r with
      | error e => simp [h2] at h
      | ok y =>
        obtain ⟨as', r'⟩ := y
        simp only [h2] at h
        injection h with h
        injection h with ha hr
        subst ha
        simp [ih _ _ _ h2]

/-- Exactly `n` consecutive items; the count is a profile constant, never read from the wire. -/
def Codec.rep {α : Type} (c : Codec α) (n : Nat) : Codec (List α) :=
  ⟨decRep c n, encRep c, decRep_sound c n⟩

def decMap {α β : Type} (c : Codec α) (f : α → β) (l : Bytes) : Except RejectCode (β × Bytes) :=
  match c.dec l with
  | .error e => .error e
  | .ok (a, r) => .ok (f a, r)

theorem decMap_sound {α β : Type} (c : Codec α) (f : α → β) (g : β → α) (hgf : ∀ a, g (f a) = a)
    (l : Bytes) (b : β) (rest : Bytes) (h : decMap c f l = .ok (b, rest)) :
    l = c.enc (g b) ++ rest := by
  unfold decMap at h
  cases h1 : c.dec l with
  | error e => simp [h1] at h
  | ok x =>
    obtain ⟨a, r⟩ := x
    simp only [h1] at h
    injection h with h
    injection h with hb hr
    subst hb hr
    rw [hgf]
    exact c.sound _ _ _ h1

/-- Transport a codec along `f` with left inverse `g` (tuples to named structures). -/
def Codec.map {α β : Type} (c : Codec α) (f : α → β) (g : β → α) (hgf : ∀ a, g (f a) = a) :
    Codec β :=
  ⟨decMap c f, fun b => c.enc (g b), decMap_sound c f g hgf⟩

/-- `Fp3`: three canonical limbs, 24 bytes. -/
def Codec.fp3 : Codec Fp3 :=
  (Codec.fp.pair (Codec.fp.pair Codec.fp)).map
    (fun t => ⟨t.1, t.2.1, t.2.2⟩) (fun a => (a.a0, a.a1, a.a2)) (fun _ => rfl)

/-- `Digest`: four canonical limbs, 32 bytes. -/
def Codec.digest : Codec Digest :=
  (Codec.fp.pair (Codec.fp.pair (Codec.fp.pair Codec.fp))).map
    (fun t => ⟨t.1, t.2.1, t.2.2.1, t.2.2.2⟩) (fun d => (d.d0, d.d1, d.d2, d.d3)) (fun _ => rfl)

/-- Canonical `Fp3` decoding of exactly 24 bytes. -/
def decodeFp3LE (b : ByteArray) : Except RejectCode Fp3 :=
  if b.size = 24 then
    match Codec.fp3.dec (toBytes b) with
    | .error e => .error e
    | .ok (a, _) => .ok a
  else .error .E_LENGTH

def encodeFp3Bytes (a : Fp3) : Bytes := Codec.fp3.enc a

def encodeFp3LE (a : Fp3) : ByteArray := ofBytes (encodeFp3Bytes a)

theorem encodeFp3Bytes_eq (a : Fp3) :
    encodeFp3Bytes a = encodeFpBytes a.a0 ++ (encodeFpBytes a.a1 ++ encodeFpBytes a.a2) := rfl

theorem encodeFp3Bytes_length (a : Fp3) : (encodeFp3Bytes a).length = 24 := by
  simp [encodeFp3Bytes_eq, encodeFpBytes_length]

/-- An accepted 24-byte word is the unique canonical encoding of its value. -/
theorem encodeFp3LE_of_decodeFp3LE {b : ByteArray} {a : Fp3} (h : decodeFp3LE b = .ok a) :
    encodeFp3LE a = b := by
  unfold decodeFp3LE at h
  split at h
  · rename_i h24
    cases hd : Codec.fp3.dec (toBytes b) with
    | error e => simp [hd] at h
    | ok x =>
      obtain ⟨a', rest⟩ := x
      simp only [hd] at h
      injection h with h
      subst h
      have hs := Codec.fp3.sound _ _ _ hd
      have hlen : (toBytes b).length = 24 := by rw [toBytes_length, h24]
      have hrest : rest = [] := by
        have := congrArg List.length hs
        rw [hlen, List.length_append] at this
        have h24' : (Codec.fp3.enc a').length = 24 := encodeFp3Bytes_length a'
        exact List.eq_nil_of_length_eq_zero (by omega)
      rw [hrest, List.append_nil] at hs
      rw [encodeFp3LE, encodeFp3Bytes, ← hs, ofBytes_toBytes]
  · cases h

/-! ## Header (spec §5.1) -/

def headerLen : Nat := 64

/-- `"FPN2"`. -/
def magicBytes : Bytes := [0x46, 0x50, 0x4E, 0x32]

def wireVersion : Nat := 2

/-- The only free header field; `magic`, `version`, `flags`, `reserved` are constants. -/
structure Fpn2Header where
  profileId : Bytes
  deriving DecidableEq, Repr

/-- Normative checks 1–5, in order. -/
def parseHeaderBytes (l : Bytes) : Except RejectCode Fpn2Header :=
  if l.length < headerLen then .error .E_LENGTH
  else if l.take 4 ≠ magicBytes then .error .E_MAGIC
  else if leNat ((l.drop 4).take 2) ≠ wireVersion then .error .E_VERSION
  else if leNat ((l.drop 6).take 2) ≠ 0 then .error .E_FLAGS
  else if (l.drop 40).take 24 ≠ List.replicate 24 0 then .error .E_RESERVED
  else .ok ⟨(l.drop 8).take 32⟩

def parseHeader (raw : ByteArray) : Except RejectCode Fpn2Header := parseHeaderBytes (toBytes raw)

/-- The 64 header bytes: `magic ‖ le_2(2) ‖ le_2(0) ‖ profile_id ‖ 0^24`. -/
def encodeHeader (h : Fpn2Header) : Bytes :=
  magicBytes ++ (natLE 2 wireVersion ++ (natLE 2 0 ++ (h.profileId ++ List.replicate 24 0)))

theorem parseHeaderBytes_spec {l : Bytes} {h : Fpn2Header} (hp : parseHeaderBytes l = .ok h) :
    headerLen ≤ l.length ∧ l.take 4 = magicBytes ∧ leNat ((l.drop 4).take 2) = wireVersion ∧
      leNat ((l.drop 6).take 2) = 0 ∧ (l.drop 40).take 24 = List.replicate 24 0 ∧
      h.profileId = (l.drop 8).take 32 := by
  unfold parseHeaderBytes at hp
  split at hp
  · cases hp
  · rename_i h1
    split at hp
    · cases hp
    · rename_i h2
      split at hp
      · cases hp
      · rename_i h3
        split at hp
        · cases hp
        · rename_i h4
          split at hp
          · cases hp
          · rename_i h5
            injection hp with hp
            subst hp
            exact ⟨Nat.le_of_not_lt h1, Decidable.not_not.mp h2, Decidable.not_not.mp h3,
              Decidable.not_not.mp h4, Decidable.not_not.mp h5, rfl⟩

/-- An accepted header is byte-for-byte `encodeHeader` of the parsed value. -/
theorem parseHeaderBytes_encode {l : Bytes} {h : Fpn2Header} (hp : parseHeaderBytes l = .ok h) :
    l.take headerLen = encodeHeader h := by
  obtain ⟨hlen, hm, hv, hf, hr, hid⟩ := parseHeaderBytes_spec hp
  have hlen' : 64 ≤ l.length := hlen
  have l2 : ∀ a, a + 2 ≤ l.length → ((l.drop a).take 2).length = 2 := by
    intro a ha; simp only [List.length_take, List.length_drop]; omega
  have ev : (l.drop 4).take 2 = natLE 2 wireVersion := by
    have := natLE_leNat ((l.drop 4).take 2)
    rw [l2 4 (by omega), hv] at this
    exact this.symm
  have ef : (l.drop 6).take 2 = natLE 2 0 := by
    have := natLE_leNat ((l.drop 6).take 2)
    rw [l2 6 (by omega), hf] at this
    exact this.symm
  have split : l.take 64 =
      l.take 4 ++ ((l.drop 4).take 2 ++ ((l.drop 6).take 2 ++
        ((l.drop 8).take 32 ++ (l.drop 40).take 24))) := by
    rw [take_drop_append l 8 32 24, take_drop_append l 6 2 56, take_drop_append l 4 2 58]
    have := take_drop_append l 0 4 60
    simpa using this.symm
  show l.take 64 = _
  rw [split, hm, ev, ef, hr, encodeHeader, hid]

/-- An accepted header has the exact magic `"FPN2"`, `version = 2`, `flags = 0`, and 24 zero
reserved bytes; the only free bytes are the 32-byte `profile_id` at offset 8. -/
theorem parseHeader_magic_version_flags_reserved {raw : ByteArray} {h : Fpn2Header}
    (hp : parseHeader raw = .ok h) :
    headerLen ≤ raw.size ∧
      (toBytes raw).take 4 = magicBytes ∧
      leNat (((toBytes raw).drop 4).take 2) = 2 ∧
      leNat (((toBytes raw).drop 6).take 2) = 0 ∧
      ((toBytes raw).drop 40).take 24 = List.replicate 24 0 ∧
      h.profileId = ((toBytes raw).drop 8).take 32 ∧
      (toBytes raw).take headerLen = encodeHeader h := by
  obtain ⟨hlen, hm, hv, hf, hr, hid⟩ := parseHeaderBytes_spec hp
  rw [toBytes_length] at hlen
  exact ⟨hlen, hm, hv, hf, hr, hid, parseHeaderBytes_encode hp⟩

/-! ## Profiles (spec §5.2) -/

inductive SecurityTier where
  | reduced
  | standard
  deriving DecidableEq, Repr

inductive Relation where
  | harnessSqSumcheck
  | llmInferenceV2
  | foldedGkrLookup
  | foldedGkrAir
  | foldedVecMatmul
  deriving DecidableEq, Repr

def SecurityTier.toNat : SecurityTier → Nat
  | .reduced => 0
  | .standard => 1

def Relation.toNat : Relation → Nat
  | .harnessSqSumcheck => 1
  | .llmInferenceV2 => 2
  | .foldedGkrLookup => 3
  | .foldedGkrAir => 4
  | .foldedVecMatmul => 5

/-- A profile: public scale upper bounds and protocol constants only (no architecture fields).
`profileId` is the pinned `B256(canonicalBytes)`; BLAKE2b is not re-executed in Lean, so the pin
is a registry constant exactly as in `core/src/profile.rs::REGISTRY`. -/
structure Fpn2Profile where
  name : String
  securityTier : SecurityTier
  relation : Relation
  securityBits : Nat
  maxLayers : Nat
  maxHidden : Nat
  maxSeqLen : Nat
  maxVocab : Nat
  maxNewTokens : Nat
  stopIds : List Nat
  nVars : Nat
  scDegree : Nat
  nQueries : Nat
  containerBytes : Nat
  profileId : Bytes
  deriving DecidableEq, Repr

namespace Fpn2Profile

/-- `"FPN2PROF"`. -/
def magic : Bytes := [0x46, 0x50, 0x4E, 0x32, 0x50, 0x52, 0x4F, 0x46]

/-- The 128 canonical profile bytes of spec §5.2 (for a well-formed profile: name of 1–32
bytes, at most 8 stop ids). -/
def canonicalBytes (p : Fpn2Profile) : Bytes :=
  let nm := toBytes p.name.toUTF8
  magic ++ natLE 2 1 ++ nm ++ List.replicate (32 - nm.length) 0 ++
    [UInt8.ofNat p.securityTier.toNat, UInt8.ofNat p.relation.toNat] ++ natLE 2 p.securityBits ++
    natLE 4 p.maxLayers ++ natLE 4 p.maxHidden ++ natLE 4 p.maxSeqLen ++ natLE 4 p.maxVocab ++
    natLE 4 p.maxNewTokens ++ [UInt8.ofNat p.stopIds.length] ++
    p.stopIds.flatMap (natLE 4) ++ List.replicate (4 * (8 - p.stopIds.length)) 0 ++
    [UInt8.ofNat p.nVars, UInt8.ofNat p.scDegree] ++ natLE 2 p.nQueries ++
    natLE 8 p.containerBytes ++ List.replicate 17 0

/-- A production (`STANDARD`-tier) profile; `REDUCED` profiles need an explicit opt-in. -/
def isProduction (p : Fpn2Profile) : Bool := p.securityTier == .standard

def stmtLen : Nat := 120

def finalLen : Nat := 24

def roundLen (p : Fpn2Profile) : Nat := (p.scDegree + 1) * 24

def openingLen (p : Fpn2Profile) : Nat := 8 + 32 * p.nVars

/-- Bytes used by a relation-1 body before zero padding; `none` when the relation is not wired in
the relation-1 parser. Relations 3, 4 and 5 have their own layouts (`Rel3Layout.bodyUsed` in
`Fpn2Rel3`, `Rel4Layout.bodyUsed` in `Fpn2Rel4`, `Rel5Layout.bodyUsed` in `Fpn2Rel5`) and parsers
(`parseBodyRel3`, `parseBodyRel4`, `parseBodyRel5`), so they are `none` here. -/
def bodyUsed (p : Fpn2Profile) : Option Nat :=
  match p.relation with
  | .harnessSqSumcheck =>
    some (stmtLen + p.nVars * p.roundLen + finalLen + p.nQueries * p.openingLen)
  | .llmInferenceV2 => none
  | .foldedGkrLookup => none
  | .foldedGkrAir => none
  | .foldedVecMatmul => none

end Fpn2Profile

/-- `fpn2-harness-sq-n4` (`REDUCED`, relation 1, `C_p = 2048`). -/
def harnessSqN4 : Fpn2Profile where
  name := "fpn2-harness-sq-n4"
  securityTier := .reduced
  relation := .harnessSqSumcheck
  securityBits := 0
  maxLayers := 0
  maxHidden := 0
  maxSeqLen := 0
  maxVocab := 0
  maxNewTokens := 0
  stopIds := []
  nVars := 4
  scDegree := 2
  nQueries := 4
  containerBytes := 2048
  profileId :=
    [0xe2, 0x4f, 0x71, 0x8c, 0xf8, 0xc2, 0xf4, 0xd2, 0x14, 0x17, 0xf9, 0xe8, 0xa3, 0x03, 0x8b, 0x99,
     0x93, 0xaa, 0x41, 0x8c, 0xa0, 0xf7, 0x20, 0x5a, 0x5c, 0xf6, 0xa8, 0xc5, 0x3a, 0x34, 0x42, 0xdb]

/-- `fpn2-llm-std-v1` (`STANDARD`, relation 2, `C_p = 8 MiB`; provisional constants, spec §5.2). -/
def llmStdV1 : Fpn2Profile where
  name := "fpn2-llm-std-v1"
  securityTier := .standard
  relation := .llmInferenceV2
  securityBits := 100
  maxLayers := 128
  maxHidden := 16384
  maxSeqLen := 131072
  maxVocab := 262144
  maxNewTokens := 32768
  stopIds := [151643, 151645]
  nVars := 0
  scDegree := 0
  nQueries := 0
  containerBytes := 8388608
  profileId :=
    [0x58, 0x86, 0x59, 0xa6, 0xe4, 0x6c, 0x51, 0xc0, 0x21, 0xcd, 0x65, 0xdd, 0x28, 0x7c, 0x89, 0x68,
     0x56, 0x2b, 0x63, 0x7d, 0x20, 0xaf, 0xaf, 0x0d, 0x68, 0xf4, 0xad, 0x57, 0xb1, 0x6d, 0xb9, 0xa1]

/-- `fpn2-folded-gkr-v1` (`STANDARD`, relation 3, `n = 12`, `Q = 72`, `C_p = 163840`). -/
def foldedGkrV1 : Fpn2Profile where
  name := "fpn2-folded-gkr-v1"
  securityTier := .standard
  relation := .foldedGkrLookup
  securityBits := 100
  maxLayers := 0
  maxHidden := 0
  maxSeqLen := 0
  maxVocab := 0
  maxNewTokens := 0
  stopIds := []
  nVars := 12
  scDegree := 3
  nQueries := 72
  containerBytes := 163840
  profileId :=
    [0xe2, 0x38, 0xdb, 0x30, 0x63, 0x48, 0xde, 0x19, 0x56, 0x1c, 0x2f, 0xb4, 0x89, 0x6e, 0xa4, 0xa2,
     0x49, 0xbd, 0x8c, 0xe7, 0xbd, 0x41, 0xbe, 0xb1, 0xf7, 0x15, 0xb6, 0xfc, 0xb3, 0x05, 0xe9, 0x86]

/-- `fpn2-folded-gkr-n8` (`REDUCED`, relation 3, `n = 8`, `Q = 2`, `C_p = 6144`). -/
def foldedGkrN8 : Fpn2Profile where
  name := "fpn2-folded-gkr-n8"
  securityTier := .reduced
  relation := .foldedGkrLookup
  securityBits := 0
  maxLayers := 0
  maxHidden := 0
  maxSeqLen := 0
  maxVocab := 0
  maxNewTokens := 0
  stopIds := []
  nVars := 8
  scDegree := 3
  nQueries := 2
  containerBytes := 6144
  profileId :=
    [0xd7, 0xa3, 0xd1, 0x69, 0xb6, 0x5f, 0x8c, 0x5e, 0x32, 0x89, 0x11, 0xf4, 0xa7, 0xed, 0xb3, 0x75,
     0x0c, 0x32, 0x44, 0xca, 0x02, 0x02, 0xa9, 0xb2, 0x59, 0xa7, 0x32, 0x5d, 0x57, 0x05, 0xc1, 0x47]

/-- `fpn2-folded-air-v1` (`STANDARD`, relation 4, `n = 14`, `Q = 72`, `C_p = 262144`). -/
def foldedAirV1 : Fpn2Profile where
  name := "fpn2-folded-air-v1"
  securityTier := .standard
  relation := .foldedGkrAir
  securityBits := 100
  maxLayers := 0
  maxHidden := 0
  maxSeqLen := 0
  maxVocab := 0
  maxNewTokens := 0
  stopIds := []
  nVars := 14
  scDegree := 3
  nQueries := 72
  containerBytes := 262144
  profileId :=
    [0xf6, 0xc6, 0xb4, 0xca, 0x3f, 0x42, 0x92, 0xe6, 0xe8, 0x91, 0x7d, 0x12, 0x85, 0xf6, 0x10, 0x5b,
     0xde, 0x16, 0x8e, 0xd7, 0x51, 0x7b, 0x71, 0xaa, 0x65, 0xe5, 0xc5, 0x62, 0x93, 0x3d, 0x0a, 0x97]

/-- `fpn2-folded-air-n12` (`REDUCED`, relation 4, `n = 12`, `Q = 2`, `C_p = 16384`). -/
def foldedAirN12 : Fpn2Profile where
  name := "fpn2-folded-air-n12"
  securityTier := .reduced
  relation := .foldedGkrAir
  securityBits := 0
  maxLayers := 0
  maxHidden := 0
  maxSeqLen := 0
  maxVocab := 0
  maxNewTokens := 0
  stopIds := []
  nVars := 12
  scDegree := 3
  nQueries := 2
  containerBytes := 16384
  profileId :=
    [0x5f, 0x66, 0x1e, 0xc4, 0xea, 0x1a, 0xb8, 0xb6, 0x7d, 0xe2, 0xed, 0x9d, 0x13, 0xa1, 0x3d, 0x14,
     0x8b, 0x4e, 0xd6, 0x39, 0xa2, 0x42, 0xd3, 0x0c, 0x81, 0xfd, 0xb4, 0xf2, 0x70, 0x5c, 0xed, 0x7d]

/-- `fpn2-folded-air-n20-v1` (`STANDARD`, relation 4, `n = 20`, `K = 32768`, `Q = 72`,
`C_p = 524288`; `core/src/profile.rs::FOLDED_AIR_N20_V1`). -/
def foldedAirN20V1 : Fpn2Profile where
  name := "fpn2-folded-air-n20-v1"
  securityTier := .standard
  relation := .foldedGkrAir
  securityBits := 100
  maxLayers := 0
  maxHidden := 0
  maxSeqLen := 0
  maxVocab := 0
  maxNewTokens := 0
  stopIds := []
  nVars := 20
  scDegree := 3
  nQueries := 72
  containerBytes := 524288
  profileId :=
    [0x96, 0x6d, 0xec, 0x06, 0xad, 0x5f, 0xb4, 0x0e, 0xf7, 0x84, 0x3f, 0xe9, 0x13, 0x10, 0x68, 0x4a,
     0x12, 0x37, 0x89, 0x90, 0xbb, 0xe5, 0x34, 0x83, 0xce, 0x77, 0x18, 0xef, 0x60, 0x6b, 0xa6, 0x0d]

/-- `fpn2-folded-vec-n16-v1` (`STANDARD`, relation 5, one full-width `32 × 1024` GEMM tile:
`K = max_hidden = 1024`, `n = 16`, `Q = 72`, `C_p = 393216`; `core/src/profile.rs::FOLDED_VEC_N16_V1`). -/
def foldedVecN16V1 : Fpn2Profile where
  name := "fpn2-folded-vec-n16-v1"
  securityTier := .standard
  relation := .foldedVecMatmul
  securityBits := 100
  maxLayers := 0
  maxHidden := 1024
  maxSeqLen := 0
  maxVocab := 0
  maxNewTokens := 0
  stopIds := []
  nVars := 16
  scDegree := 3
  nQueries := 72
  containerBytes := 393216
  profileId :=
    [0xa4, 0x8d, 0xfb, 0x0a, 0x1b, 0x0f, 0x17, 0x58, 0x0c, 0xba, 0xb6, 0xb7, 0x56, 0x67, 0x82, 0x38,
     0x20, 0x0c, 0x1b, 0x30, 0x28, 0xa2, 0x6f, 0x92, 0x89, 0x84, 0x30, 0xd8, 0xda, 0x80, 0x13, 0x27]

/-- `fpn2-folded-vec-n22-v1` (`STANDARD`, relation 5, 64 full-width `32 × 1024` tiles: `K = 1024`,
`n = 22`, `Q = 72`, `C_p = 524288`; `core/src/profile.rs::FOLDED_VEC_N22_V1`). -/
def foldedVecN22V1 : Fpn2Profile where
  name := "fpn2-folded-vec-n22-v1"
  securityTier := .standard
  relation := .foldedVecMatmul
  securityBits := 100
  maxLayers := 0
  maxHidden := 1024
  maxSeqLen := 0
  maxVocab := 0
  maxNewTokens := 0
  stopIds := []
  nVars := 22
  scDegree := 3
  nQueries := 72
  containerBytes := 524288
  profileId :=
    [0xba, 0x72, 0x06, 0x68, 0x88, 0xc8, 0xfc, 0x3c, 0xf0, 0xca, 0x31, 0xc8, 0x65, 0x27, 0xa5, 0x29,
     0x27, 0x59, 0xeb, 0x62, 0x9c, 0x0e, 0xed, 0x13, 0x1d, 0x81, 0x26, 0x23, 0xb3, 0xe6, 0xb3, 0xef]

/-- The pinned registry (spec §5.2), in the order of `core/src/profile.rs::REGISTRY`. -/
def profileRegistry : List Fpn2Profile :=
  [harnessSqN4, llmStdV1, foldedGkrV1, foldedGkrN8, foldedAirV1, foldedAirN12, foldedAirN20V1,
    foldedVecN16V1, foldedVecN22V1]

/-- Normative check 6: the registered profile with this pinned id, if any. -/
def lookupProfile (id : Bytes) : Option Fpn2Profile :=
  profileRegistry.find? (fun p => decide (p.profileId = id))

theorem lookupProfile_spec {id : Bytes} {p : Fpn2Profile} (h : lookupProfile id = some p) :
    p ∈ profileRegistry ∧ p.profileId = id := by
  unfold lookupProfile at h
  exact ⟨List.mem_of_find?_eq_some h, by simpa using List.find?_some h⟩

/-! ## Relation-1 body (spec §5.3) -/

/-- One query opening: the leaf value `f_q` and its `n` sibling digests, leaf level first. -/
structure Opening where
  value : UInt64
  siblings : List Digest
  deriving DecidableEq, Repr

/-- Decoded body of a relation-1 (`HARNESS_SQ_SUMCHECK`) container, in wire order. -/
structure HarnessBody where
  rTab : Digest
  nS : Bytes
  stNew : Digest
  claim : Fp3
  rounds : List (List Fp3)
  final : Fp3
  openings : List Opening
  deriving DecidableEq, Repr

/-- A fully parsed relation-1 container. -/
structure Fpn2Container where
  header : Fpn2Header
  body : HarnessBody
  deriving DecidableEq, Repr

/-- `STMT` (120 bytes): `R_tab ‖ N_s ‖ St_new ‖ S`. -/
def Codec.stmt : Codec (Digest × Bytes × Digest × Fp3) :=
  Codec.digest.pair ((Codec.raw 32).pair (Codec.digest.pair Codec.fp3))

/-- One sumcheck round: `g_i(0), …, g_i(d)` as `Fp3`. -/
def Codec.round (prof : Fpn2Profile) : Codec (List Fp3) := Codec.fp3.rep (prof.scDegree + 1)

/-- One opening: `f_q ‖ sib_0 … sib_{n-1}`. -/
def Codec.opening (prof : Fpn2Profile) : Codec Opening :=
  (Codec.fp.pair (Codec.digest.rep prof.nVars)).map
    (fun t => ⟨t.1, t.2⟩) (fun o => (o.value, o.siblings)) (fun _ => rfl)

/-- The whole relation-1 body, sections back-to-back in wire order:
`STMT ‖ SUMCHECK ‖ FINAL ‖ OPENINGS`. Every typed field is decoded canonically in wire order, so
the first non-canonical limb determines `E_NONCANONICAL` (normative check 12). -/
def Codec.body (prof : Fpn2Profile) : Codec HarnessBody :=
  (Codec.stmt.pair (((Codec.round prof).rep prof.nVars).pair
      (Codec.fp3.pair ((Codec.opening prof).rep prof.nQueries)))).map
    (fun t => ⟨t.1.1, t.1.2.1, t.1.2.2.1, t.1.2.2.2, t.2.1, t.2.2.1, t.2.2.2⟩)
    (fun b => ((b.rTab, b.nS, b.stNew, b.claim), b.rounds, b.final, b.openings))
    (fun _ => rfl)

/-- Normative checks 9–12 against a known profile: exact length `C_p`, wired backend, all-zero
`PAD`, canonical decoding of every typed field in wire order. -/
def parseBody (prof : Fpn2Profile) (l : Bytes) : Except RejectCode HarnessBody :=
  if l.length ≠ prof.containerBytes then .error .E_LENGTH
  else
    match prof.bodyUsed with
    | none => .error .E_BACKEND_UNAVAILABLE
    | some used =>
      if prof.containerBytes < headerLen + used then .error .E_LENGTH
      else if (l.drop headerLen).drop used ≠
          List.replicate (prof.containerBytes - headerLen - used) 0 then .error .E_PADDING
      else
        match (Codec.body prof).dec ((l.drop headerLen).take used) with
        | .error e => .error e
        | .ok (b, rest) => if rest = [] then .ok b else .error .E_LENGTH

/-- Strict parser against a pinned profile: header checks 1–5, the header names exactly the
pinned `profile_id` (else `E_PROFILE_PINNED`), then body checks 9–12. The registry and tier
policy checks 6–8 are applied by `verifyBytes` between the two halves. -/
def parseContainer (prof : Fpn2Profile) (raw : ByteArray) : Except RejectCode Fpn2Container :=
  match parseHeader raw with
  | .error e => .error e
  | .ok h =>
    if h.profileId ≠ prof.profileId then .error .E_PROFILE_PINNED
    else
      match parseBody prof (toBytes raw) with
      | .error e => .error e
      | .ok b => .ok ⟨h, b⟩

/-- Zero padding length of a profile (`0` when the relation has no wired backend). -/
def padLen (prof : Fpn2Profile) : Nat := prof.containerBytes - headerLen - prof.bodyUsed.getD 0

/-- Canonical serializer (inverse of `parseContainer` on accepted inputs): exactly `C_p` bytes. -/
def encodeContainer (prof : Fpn2Profile) (c : Fpn2Container) : Bytes :=
  encodeHeader c.header ++ ((Codec.body prof).enc c.body ++ List.replicate (padLen prof) 0)

theorem parseBody_spec {prof : Fpn2Profile} {l : Bytes} {b : HarnessBody}
    (hp : parseBody prof l = .ok b) :
    l.length = prof.containerBytes ∧ ∃ used, prof.bodyUsed = some used ∧
      headerLen + used ≤ prof.containerBytes ∧
      (l.drop headerLen).take used = (Codec.body prof).enc b ∧
      (l.drop headerLen).drop used = List.replicate (prof.containerBytes - headerLen - used) 0 := by
  unfold parseBody at hp
  split at hp
  · cases hp
  · rename_i hlen
    split at hp
    · cases hp
    · rename_i used hused
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
              have hs := (Codec.body prof).sound _ _ _ hdec
              rw [List.append_nil] at hs
              exact ⟨Decidable.not_not.mp hlen, used, hused, Nat.le_of_not_lt hfit, hs,
                Decidable.not_not.mp hpad⟩
            · cases hp

theorem parseContainer_spec {prof : Fpn2Profile} {raw : ByteArray} {c : Fpn2Container}
    (hp : parseContainer prof raw = .ok c) :
    parseHeader raw = .ok c.header ∧ c.header.profileId = prof.profileId ∧
      parseBody prof (toBytes raw) = .ok c.body := by
  unfold parseContainer at hp
  cases hh : parseHeader raw with
  | error e => simp [hh] at hp
  | ok h =>
    simp only [hh] at hp
    split at hp
    · cases hp
    · rename_i hid
      cases hb : parseBody prof (toBytes raw) with
      | error e => simp [hb] at hp
      | ok b =>
        simp only [hb] at hp
        injection hp with hp
        subst hp
        exact ⟨rfl, Decidable.not_not.mp hid, rfl⟩

/-- **Canonical wire representation.** An accepted byte string is exactly the canonical
serialization of the container it parses to. -/
theorem parseContainer_encode {prof : Fpn2Profile} {raw : ByteArray} {c : Fpn2Container}
    (hp : parseContainer prof raw = .ok c) : toBytes raw = encodeContainer prof c := by
  obtain ⟨hh, _, hb⟩ := parseContainer_spec hp
  obtain ⟨_, used, hused, _, htake, hdrop⟩ := parseBody_spec hb
  have hhead := parseHeaderBytes_encode hh
  have e1 : toBytes raw = (toBytes raw).take headerLen ++ (toBytes raw).drop headerLen :=
    (List.take_append_drop _ _).symm
  have e2 : (toBytes raw).drop headerLen =
      ((toBytes raw).drop headerLen).take used ++ ((toBytes raw).drop headerLen).drop used :=
    (List.take_append_drop _ _).symm
  rw [e1, e2, hhead, htake, hdrop, encodeContainer, padLen, hused]
  rfl

/-- An accepted container has exactly `C_p` bytes and every padding byte is zero: the bytes
after the last typed field carry no information (no padding malleability, no covert channel). -/
theorem parseContainer_length_and_zero_pad {prof : Fpn2Profile} {raw : ByteArray}
    {c : Fpn2Container} (hp : parseContainer prof raw = .ok c) :
    raw.size = prof.containerBytes ∧ ∃ used, prof.bodyUsed = some used ∧
      headerLen + used ≤ prof.containerBytes ∧
      (toBytes raw).drop (headerLen + used) =
        List.replicate (prof.containerBytes - headerLen - used) 0 ∧
      ∀ i, headerLen + used ≤ i → i < raw.size → (toBytes raw)[i]? = some 0 := by
  obtain ⟨_, _, hb⟩ := parseContainer_spec hp
  obtain ⟨hlen, used, hused, hfit, _, hdrop⟩ := parseBody_spec hb
  rw [toBytes_length] at hlen
  rw [List.drop_drop] at hdrop
  refine ⟨hlen, used, hused, hfit, hdrop, ?_⟩
  intro i hi hlt
  have h1 : ((toBytes raw).drop (headerLen + used))[i - (headerLen + used)]? = (toBytes raw)[i]? := by
    rw [List.getElem?_drop]
    congr 1
    omega
  rw [← h1, hdrop, List.getElem?_replicate]
  have : i - (headerLen + used) < prof.containerBytes - headerLen - used := by omega
  simp [this]

/-- **Uniqueness of the accepted encoding.** Two accepted byte arrays that parse to the same
container are byte-for-byte equal. -/
theorem parseContainer_injective_on_accepted {prof : Fpn2Profile} {raw₁ raw₂ : ByteArray}
    {c : Fpn2Container} (h₁ : parseContainer prof raw₁ = .ok c)
    (h₂ : parseContainer prof raw₂ = .ok c) : raw₁ = raw₂ :=
  toBytes_injective ((parseContainer_encode h₁).trans (parseContainer_encode h₂).symm)

end FlashProveNext

#print axioms FlashProveNext.toBytes_ofBytes
#print axioms FlashProveNext.ofBytes_toBytes
#print axioms FlashProveNext.toBytes_injective
#print axioms FlashProveNext.toBytes_length
#print axioms FlashProveNext.natLE_length
#print axioms FlashProveNext.leNat_lt
#print axioms FlashProveNext.leNat_natLE
#print axioms FlashProveNext.natLE_leNat
#print axioms FlashProveNext.natLE_injective
#print axioms FlashProveNext.take_drop_append
#print axioms FlashProveNext.goldilocksP_toNat
#print axioms FlashProveNext.goldilocksP_toNat_eq
#print axioms FlashProveNext.encodeFpBytes_length
#print axioms FlashProveNext.decodeFpBytes_encodeFpBytes
#print axioms FlashProveNext.encodeFpBytes_of_decodeFpBytes
#print axioms FlashProveNext.decodeFpLE_encodeFpLE
#print axioms FlashProveNext.decodeFpLE_rejects_ge_P
#print axioms FlashProveNext.encodeFpLE_of_decodeFpLE
#print axioms FlashProveNext.decFp_sound
#print axioms FlashProveNext.decRaw_sound
#print axioms FlashProveNext.decPair_sound
#print axioms FlashProveNext.decRep_sound
#print axioms FlashProveNext.decRep_length
#print axioms FlashProveNext.decMap_sound
#print axioms FlashProveNext.encodeFp3Bytes_eq
#print axioms FlashProveNext.encodeFp3Bytes_length
#print axioms FlashProveNext.encodeFp3LE_of_decodeFp3LE
#print axioms FlashProveNext.parseHeaderBytes_spec
#print axioms FlashProveNext.parseHeaderBytes_encode
#print axioms FlashProveNext.parseHeader_magic_version_flags_reserved
#print axioms FlashProveNext.lookupProfile_spec
#print axioms FlashProveNext.parseBody_spec
#print axioms FlashProveNext.parseContainer_spec
#print axioms FlashProveNext.parseContainer_encode
#print axioms FlashProveNext.parseContainer_length_and_zero_pad
#print axioms FlashProveNext.parseContainer_injective_on_accepted
