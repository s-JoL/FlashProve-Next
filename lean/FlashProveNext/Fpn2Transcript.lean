/-
`Fpn2Transcript` (`specs/11-wire-format.md` §4): the raw-byte Fiat–Shamir hash chain with 1-byte
domain tags, as an executable state machine over an abstract hash oracle `H : ByteArray → ByteArray`
(`H_fs` = BLAKE2b-512 personalized `"FPN2-FS-v2"` in the Rust and Python verifiers; a random
oracle in the Stage 3–4 soundness proofs). Lean core only, no Mathlib import.

1. `frameInit`, `frameAbsorb`, `frameSqueeze`, `frameRatchet` : the exact byte strings hashed by
   each operation (`st ‖ tag ‖ le_8(|b|) ‖ b`, and so on);
2. `Fpn2Transcript.init`, `absorbBytes`, `absorbFp`, `absorbFp3`, `squeezeBytes`, `squeezeFp3`,
   `squeezeU192` : the operations of the spec table, `run` : a whole operation schedule;
3. `frameAbsorb_injective`, `frameSqueeze_injective`, `frameInit_injective`, `frame_split`,
   `frameAbsorb_ne_frameSqueeze`, `frameAbsorb_ne_frameRatchet`, `frameSqueeze_ne_frameRatchet`,
   `frameInit_ne_frameAbsorb` : the framing is injective and the domains are pairwise separated,
   so two distinct operations never hash the same string (no claim about `H` itself is made);
4. `absorb_eq_imp_eq_or_collision`, `absorbBytes_eq_imp_eq_or_collision` : equal successor states
   come from equal (state, tag, data) or exhibit an explicit collision of `H`;
5. `run_append`, `squeezeBytes_length`, `squeezeFp3_canonical` : schedules compose sequentially,
   a squeeze returns exactly `n` bytes for a 64-byte oracle, and squeezed limbs are canonical.
-/
import FlashProveNext.Fpn2Wire

namespace FlashProveNext

/-! ## Domain tags and framing -/

def tagInit : UInt8 := 0x00
def tagBytes : UInt8 := 0x01
def tagFp : UInt8 := 0x02
def tagFp3 : UInt8 := 0x03
def tagSqBytes : UInt8 := 0x10
def tagSqFp3 : UInt8 := 0x11
def tagSqU192 : UInt8 := 0x12
def tagRatchet : UInt8 := 0x7F

/-- The container transcript label `"FPN2/transcript/v2"` (18 ASCII bytes). -/
def labelV2 : Bytes :=
  [0x46, 0x50, 0x4e, 0x32, 0x2f, 0x74, 0x72, 0x61, 0x6e, 0x73, 0x63, 0x72, 0x69, 0x70, 0x74, 0x2f,
   0x76, 0x32]

/-- `le_8(n)`. -/
def le8 (n : Nat) : Bytes := natLE 8 n

theorem le8_length (n : Nat) : (le8 n).length = 8 := natLE_length 8 n

theorem le8_injective {n m : Nat} (hn : n < 2 ^ 64) (hm : m < 2 ^ 64) (h : le8 n = le8 m) :
    n = m :=
  natLE_injective 8 n m hn hm h

/-- `0x00 ‖ le_8(|label|) ‖ label`. -/
def frameInit (label : Bytes) : Bytes := tagInit :: (le8 label.length ++ label)

/-- `st ‖ tag ‖ le_8(|data|) ‖ data`. -/
def frameAbsorb (st : Bytes) (tag : UInt8) (data : Bytes) : Bytes :=
  st ++ tag :: (le8 data.length ++ data)

/-- `st ‖ tag ‖ le_8(n) ‖ le_8(i)`: output block `i` of an `n`-byte squeeze. -/
def frameSqueeze (st : Bytes) (tag : UInt8) (n i : Nat) : Bytes :=
  st ++ tag :: (le8 n ++ le8 i)

/-- `st ‖ 0x7F ‖ tag ‖ le_8(n)`: the ratchet after an `n`-byte squeeze. -/
def frameRatchet (st : Bytes) (tag : UInt8) (n : Nat) : Bytes :=
  st ++ tagRatchet :: tag :: le8 n

/-- Any two frames over equal-length states split identically: same state, same tag byte, same
payload. This is the single fact behind every domain-separation statement below. -/
theorem frame_split {st st' : Bytes} {t t' : UInt8} {x y : Bytes} (hlen : st.length = st'.length)
    (h : st ++ t :: x = st' ++ t' :: y) : st = st' ∧ t = t' ∧ x = y := by
  obtain ⟨h1, h2⟩ := List.append_inj h hlen
  injection h2 with ht hx
  exact ⟨h1, ht, hx⟩

/-- Absorb framing is injective in (state, tag, data) for equal-length states. -/
theorem frameAbsorb_injective {st st' : Bytes} {tag tag' : UInt8} {d d' : Bytes}
    (hlen : st.length = st'.length) (h : frameAbsorb st tag d = frameAbsorb st' tag' d') :
    st = st' ∧ tag = tag' ∧ d = d' := by
  obtain ⟨h1, h2, h3⟩ := frame_split hlen h
  obtain ⟨_, h4⟩ := List.append_inj h3 (by rw [le8_length, le8_length])
  exact ⟨h1, h2, h4⟩

/-- Squeeze framing is injective in (state, tag, length, block index) below `2^64`. -/
theorem frameSqueeze_injective {st st' : Bytes} {tag tag' : UInt8} {n n' i i' : Nat}
    (hlen : st.length = st'.length) (hn : n < 2 ^ 64) (hn' : n' < 2 ^ 64) (hi : i < 2 ^ 64)
    (hi' : i' < 2 ^ 64) (h : frameSqueeze st tag n i = frameSqueeze st' tag' n' i') :
    st = st' ∧ tag = tag' ∧ n = n' ∧ i = i' := by
  obtain ⟨h1, h2, h3⟩ := frame_split hlen h
  obtain ⟨h4, h5⟩ := List.append_inj h3 (by rw [le8_length, le8_length])
  exact ⟨h1, h2, le8_injective hn hn' h4, le8_injective hi hi' h5⟩

theorem frameInit_injective {l l' : Bytes} (h : frameInit l = frameInit l') : l = l' := by
  unfold frameInit at h
  injection h with _ h
  exact (List.append_inj h (by rw [le8_length, le8_length])).2

/-- Absorbs and squeezes are separated by their tag byte. -/
theorem frameAbsorb_ne_frameSqueeze {st st' : Bytes} {tag tag' : UInt8} {d : Bytes} {n i : Nat}
    (hlen : st.length = st'.length) (htag : tag ≠ tag') :
    frameAbsorb st tag d ≠ frameSqueeze st' tag' n i :=
  fun h => htag (frame_split hlen h).2.1

/-- Absorbs are separated from ratchets (`0x7F`). -/
theorem frameAbsorb_ne_frameRatchet {st st' : Bytes} {tag tag' : UInt8} {d : Bytes} {n : Nat}
    (hlen : st.length = st'.length) (htag : tag ≠ tagRatchet) :
    frameAbsorb st tag d ≠ frameRatchet st' tag' n :=
  fun h => htag (frame_split hlen h).2.1

/-- Squeeze output blocks are separated from ratchets (`0x7F`). -/
theorem frameSqueeze_ne_frameRatchet {st st' : Bytes} {tag tag' : UInt8} {n i n' : Nat}
    (hlen : st.length = st'.length) (htag : tag ≠ tagRatchet) :
    frameSqueeze st tag n i ≠ frameRatchet st' tag' n' :=
  fun h => htag (frame_split hlen h).2.1

/-- The `init` frame of a short label (`< 56` bytes, in particular the 18-byte container label)
is shorter than any frame over a 64-byte state. -/
theorem frameInit_ne_frameAbsorb {label st : Bytes} {tag : UInt8} {d : Bytes}
    (hst : st.length = 64) (hlabel : label.length < 56) :
    frameInit label ≠ frameAbsorb st tag d := by
  intro h
  have := congrArg List.length h
  simp only [frameInit, frameAbsorb, List.length_cons, List.length_append, le8_length, hst] at this
  omega

/-! ## The transcript state machine -/

/-- Transcript state (`{0,1}^512` for the BLAKE2b-512 instantiation). -/
structure Fpn2Transcript where
  state : ByteArray

namespace Fpn2Transcript

theorem ext {T T' : Fpn2Transcript} (h : T.state = T'.state) : T = T' := by
  cases T; cases T'; cases h; rfl

/-- `init(label)`: `st ← H(0x00 ‖ le_8(|label|) ‖ label)`. -/
def init (H : ByteArray → ByteArray) (label : Bytes) : Fpn2Transcript :=
  ⟨H (ofBytes (frameInit label))⟩

/-- `st ← H(st ‖ tag ‖ le_8(|data|) ‖ data)`. -/
def absorb (H : ByteArray → ByteArray) (T : Fpn2Transcript) (tag : UInt8) (data : Bytes) :
    Fpn2Transcript :=
  ⟨H (ofBytes (frameAbsorb (toBytes T.state) tag data))⟩

/-- `absorb_bytes(b)`, tag `0x01`. -/
def absorbBytes (H : ByteArray → ByteArray) (T : Fpn2Transcript) (b : Bytes) : Fpn2Transcript :=
  T.absorb H tagBytes b

/-- `absorb_fp(x)`, tag `0x02` (`x` canonical). -/
def absorbFp (H : ByteArray → ByteArray) (T : Fpn2Transcript) (x : UInt64) : Fpn2Transcript :=
  T.absorb H tagFp (encodeFpBytes x)

/-- `absorb_fp3(a)`, tag `0x03` (`a` canonical). -/
def absorbFp3 (H : ByteArray → ByteArray) (T : Fpn2Transcript) (a : Fp3) : Fpn2Transcript :=
  T.absorb H tagFp3 (encodeFp3Bytes a)

/-- `squeeze(tag, n)`: the first `n` bytes of `H(st‖tag‖le_8(n)‖le_8(0)) ‖ H(st‖tag‖le_8(n)‖le_8(1)) ‖ …`
(`⌈n/64⌉` blocks), then the ratchet `st ← H(st ‖ 0x7F ‖ tag ‖ le_8(n))`. -/
def squeeze (H : ByteArray → ByteArray) (T : Fpn2Transcript) (tag : UInt8) (n : Nat) :
    Bytes × Fpn2Transcript :=
  (((List.range ((n + 63) / 64)).flatMap
      (fun i => toBytes (H (ofBytes (frameSqueeze (toBytes T.state) tag n i))))).take n,
    ⟨H (ofBytes (frameRatchet (toBytes T.state) tag n))⟩)

/-- `squeeze_bytes(n)`, tag `0x10`. -/
def squeezeBytes (H : ByteArray → ByteArray) (T : Fpn2Transcript) (n : Nat) :
    Bytes × Fpn2Transcript :=
  T.squeeze H tagSqBytes n

/-- Limb `k` of a 48-byte squeeze: `int_le(o[16k .. 16k+16]) mod P`. -/
def limb128 (o : Bytes) (k : Nat) : UInt64 :=
  UInt64.ofNat (leNat ((o.drop (16 * k)).take 16) % goldilocksP.toNat)

/-- `squeeze_fp3()`, tag `0x11`: three 128-bit words reduced modulo `P`. -/
def squeezeFp3 (H : ByteArray → ByteArray) (T : Fpn2Transcript) : Fp3 × Fpn2Transcript :=
  let r := T.squeeze H tagSqFp3 48
  (⟨limb128 r.1 0, limb128 r.1 1, limb128 r.1 2⟩, r.2)

/-- `squeeze_u192()`, tag `0x12`: `int_le` of 24 bytes. -/
def squeezeU192 (H : ByteArray → ByteArray) (T : Fpn2Transcript) : Nat × Fpn2Transcript :=
  let r := T.squeeze H tagSqU192 24
  (leNat r.1, r.2)

/-! ### Operation schedules -/

/-- One transcript operation of the spec table (`init` starts a schedule and is not an `Op`). -/
inductive Op where
  | absorbBytes (b : Bytes)
  | absorbFp (x : UInt64)
  | absorbFp3 (a : Fp3)
  | squeezeBytes (n : Nat)
  | squeezeFp3
  | squeezeU192

/-- Execute one operation; the second component is the raw squeezed bytes (`[]` for absorbs). -/
def step (H : ByteArray → ByteArray) (T : Fpn2Transcript) : Op → Fpn2Transcript × Bytes
  | .absorbBytes b => (T.absorbBytes H b, [])
  | .absorbFp x => (T.absorbFp H x, [])
  | .absorbFp3 a => (T.absorbFp3 H a, [])
  | .squeezeBytes n => ((T.squeeze H tagSqBytes n).2, (T.squeeze H tagSqBytes n).1)
  | .squeezeFp3 => ((T.squeeze H tagSqFp3 48).2, (T.squeeze H tagSqFp3 48).1)
  | .squeezeU192 => ((T.squeeze H tagSqU192 24).2, (T.squeeze H tagSqU192 24).1)

/-- Execute a schedule, collecting every squeezed output in order. -/
def run (H : ByteArray → ByteArray) (T : Fpn2Transcript) : List Op → Fpn2Transcript × List Bytes
  | [] => (T, [])
  | op :: ops =>
    ((run H (step H T op).1 ops).1, (step H T op).2 :: (run H (step H T op).1 ops).2)

/-- Schedules compose sequentially: the transcript after `ops₁ ++ ops₂` is the transcript after
`ops₂` started from the state reached by `ops₁`, and the outputs concatenate. The state reached
and every challenge are functions of the oracle, the start state and the operation list only. -/
theorem run_append (H : ByteArray → ByteArray) (T : Fpn2Transcript) (ops₁ ops₂ : List Op) :
    run H T (ops₁ ++ ops₂) =
      ((run H (run H T ops₁).1 ops₂).1, (run H T ops₁).2 ++ (run H (run H T ops₁).1 ops₂).2) := by
  induction ops₁ generalizing T with
  | nil => rfl
  | cons op ops ih => simp [run, ih]

/-! ### Domain separation lifted to the oracle -/

/-- Two absorbs that reach the same state either absorbed the same tagged data from the same
state, or exhibit an explicit collision `x ≠ y`, `H x = H y`. -/
theorem absorb_eq_imp_eq_or_collision (H : ByteArray → ByteArray) {T T' : Fpn2Transcript}
    {tag tag' : UInt8} {d d' : Bytes} (hlen : T.state.size = T'.state.size)
    (h : T.absorb H tag d = T'.absorb H tag' d') :
    (T = T' ∧ tag = tag' ∧ d = d') ∨ ∃ x y : ByteArray, x ≠ y ∧ H x = H y := by
  have hH : H (ofBytes (frameAbsorb (toBytes T.state) tag d)) =
      H (ofBytes (frameAbsorb (toBytes T'.state) tag' d')) := congrArg Fpn2Transcript.state h
  by_cases hf : frameAbsorb (toBytes T.state) tag d = frameAbsorb (toBytes T'.state) tag' d'
  · left
    obtain ⟨h1, h2, h3⟩ :=
      frameAbsorb_injective (by rw [toBytes_length, toBytes_length, hlen]) hf
    exact ⟨ext (toBytes_injective h1), h2, h3⟩
  · right
    refine ⟨_, _, ?_, hH⟩
    intro he
    apply hf
    have := congrArg toBytes he
    rwa [toBytes_ofBytes, toBytes_ofBytes] at this

theorem absorbBytes_eq_imp_eq_or_collision (H : ByteArray → ByteArray) {T T' : Fpn2Transcript}
    {b b' : Bytes} (hlen : T.state.size = T'.state.size)
    (h : T.absorbBytes H b = T'.absorbBytes H b') :
    (T = T' ∧ b = b') ∨ ∃ x y : ByteArray, x ≠ y ∧ H x = H y := by
  rcases absorb_eq_imp_eq_or_collision H hlen h with ⟨h1, _, h3⟩ | hc
  · exact Or.inl ⟨h1, h3⟩
  · exact Or.inr hc

/-! ### Output shape -/

theorem flatMap_length_const {α : Type} (l : List α) (f : α → Bytes) (k : Nat)
    (hf : ∀ a, (f a).length = k) : (l.flatMap f).length = k * l.length := by
  induction l with
  | nil => simp
  | cons a as ih => simp [List.flatMap_cons, hf, ih, Nat.mul_succ, Nat.add_comm]

/-- With a 64-byte oracle a squeeze returns exactly `n` bytes. -/
theorem squeeze_length (H : ByteArray → ByteArray) (hH : ∀ x, (H x).size = 64)
    (T : Fpn2Transcript) (tag : UInt8) (n : Nat) : (T.squeeze H tag n).1.length = n := by
  have hlen := flatMap_length_const (List.range ((n + 63) / 64))
    (fun i => toBytes (H (ofBytes (frameSqueeze (toBytes T.state) tag n i)))) 64
    (fun i => by rw [toBytes_length, hH])
  simp only [squeeze, List.length_take, hlen, List.length_range]
  omega

theorem squeezeBytes_length (H : ByteArray → ByteArray) (hH : ∀ x, (H x).size = 64)
    (T : Fpn2Transcript) (n : Nat) : (T.squeezeBytes H n).1.length = n :=
  squeeze_length H hH T tagSqBytes n

theorem limb128_lt (o : Bytes) (k : Nat) : limb128 o k < goldilocksP := by
  have hP : 0 < goldilocksP.toNat := by decide
  have hP' : goldilocksP.toNat < 2 ^ 64 := goldilocksP.toNat_lt
  have hmod := Nat.mod_lt (leNat ((o.drop (16 * k)).take 16)) hP
  rw [UInt64.lt_iff_toNat_lt, limb128, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  exact hmod

/-- Every squeezed challenge limb is canonical (`< P`), for any oracle. -/
theorem squeezeFp3_canonical (H : ByteArray → ByteArray) (T : Fpn2Transcript) :
    (T.squeezeFp3 H).1.a0 < goldilocksP ∧ (T.squeezeFp3 H).1.a1 < goldilocksP ∧
      (T.squeezeFp3 H).1.a2 < goldilocksP :=
  ⟨limb128_lt _ _, limb128_lt _ _, limb128_lt _ _⟩

end Fpn2Transcript

end FlashProveNext

#print axioms FlashProveNext.le8_length
#print axioms FlashProveNext.le8_injective
#print axioms FlashProveNext.frame_split
#print axioms FlashProveNext.frameAbsorb_injective
#print axioms FlashProveNext.frameSqueeze_injective
#print axioms FlashProveNext.frameInit_injective
#print axioms FlashProveNext.frameAbsorb_ne_frameSqueeze
#print axioms FlashProveNext.frameAbsorb_ne_frameRatchet
#print axioms FlashProveNext.frameSqueeze_ne_frameRatchet
#print axioms FlashProveNext.frameInit_ne_frameAbsorb
#print axioms FlashProveNext.Fpn2Transcript.ext
#print axioms FlashProveNext.Fpn2Transcript.run_append
#print axioms FlashProveNext.Fpn2Transcript.absorb_eq_imp_eq_or_collision
#print axioms FlashProveNext.Fpn2Transcript.absorbBytes_eq_imp_eq_or_collision
#print axioms FlashProveNext.Fpn2Transcript.flatMap_length_const
#print axioms FlashProveNext.Fpn2Transcript.squeeze_length
#print axioms FlashProveNext.Fpn2Transcript.squeezeBytes_length
#print axioms FlashProveNext.Fpn2Transcript.limb128_lt
#print axioms FlashProveNext.Fpn2Transcript.squeezeFp3_canonical
