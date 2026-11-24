/-
Shared test code of the executable parity harnesses (`Rel3Vectors.lean`, `Rel4Vectors.lean`): a
concrete BLAKE2b (RFC 7693, personalized, unkeyed) written in Lean, the `Rel3Oracles` instance it
gives (`H_fs`, `B256_M`; relation 4 uses the same two functions), and hex helpers.

This file is test code: it is not imported by `FlashProveNext.lean` and nothing is proved about
the BLAKE2b below (`check_rel3_vectors.py` checks it against `hashlib`, including the KATs of
`fpn2_primitives.json`). The Python drivers compile it to `tests/FpnTestLib.olean` in a temporary
directory, so the harnesses import it as `tests.FpnTestLib`.
-/
import FlashProveNext.Fpn2Rel3

open FlashProveNext

namespace FpnTest

def iv : Array UInt64 :=
  #[0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179]

def sigma : Array (Array Nat) :=
  #[#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    #[14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    #[11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    #[7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    #[9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    #[2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    #[12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    #[13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    #[6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    #[10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0]]

def rotr (x : UInt64) (n : UInt64) : UInt64 := (x >>> n) ||| (x <<< (64 - n))

def mixG (v : Array UInt64) (a b c d : Nat) (x y : UInt64) : Array UInt64 :=
  let va := v[a]! + v[b]! + x
  let vd := rotr (v[d]! ^^^ va) 32
  let vc := v[c]! + vd
  let vb := rotr (v[b]! ^^^ vc) 24
  let va := va + vb + y
  let vd := rotr (vd ^^^ va) 16
  let vc := vc + vd
  let vb := rotr (vb ^^^ vc) 63
  (((v.set! a va).set! b vb).set! c vc).set! d vd

/-- Little-endian `u64` at byte offset `off`; bytes past the end read as zero (block padding). -/
def loadWord (b : ByteArray) (off : Nat) : UInt64 :=
  (List.range 8).foldl
    (fun acc i =>
      if off + i < b.size then acc ||| ((b.get! (off + i)).toUInt64 <<< (UInt64.ofNat (8 * i)))
      else acc) 0

/-- The compression function `F(h, m, t, last)` on the block starting at byte `off`. -/
def compress (h : Array UInt64) (msg : ByteArray) (off t : Nat) (last : Bool) : Array UInt64 :=
  let m : Array UInt64 := ((List.range 16).map (fun i => loadWord msg (off + 8 * i))).toArray
  let v := h ++ iv
  let v := v.set! 12 (v[12]! ^^^ UInt64.ofNat (t % 2 ^ 64))
  let v := v.set! 13 (v[13]! ^^^ UInt64.ofNat (t / 2 ^ 64))
  let v := if last then v.set! 14 (~~~ v[14]!) else v
  let v := (List.range 12).foldl
    (fun v r =>
      let s := sigma[r % 10]!
      let v := mixG v 0 4 8 12 m[s[0]!]! m[s[1]!]!
      let v := mixG v 1 5 9 13 m[s[2]!]! m[s[3]!]!
      let v := mixG v 2 6 10 14 m[s[4]!]! m[s[5]!]!
      let v := mixG v 3 7 11 15 m[s[6]!]! m[s[7]!]!
      let v := mixG v 0 5 10 15 m[s[8]!]! m[s[9]!]!
      let v := mixG v 1 6 11 12 m[s[10]!]! m[s[11]!]!
      let v := mixG v 2 7 8 13 m[s[12]!]! m[s[13]!]!
      mixG v 3 4 9 14 m[s[14]!]! m[s[15]!]!)
    v
  ((List.range 8).map (fun i => h[i]! ^^^ v[i]! ^^^ v[i + 8]!)).toArray

/-- Unkeyed BLAKE2b with digest length `nn` and a 16-byte personalization (zero-padded). -/
def blake2b (nn : Nat) (person : ByteArray) (msg : ByteArray) : ByteArray :=
  let h := iv.set! 0 (iv[0]! ^^^ 0x01010000 ^^^ UInt64.ofNat nn)
  let h := h.set! 6 (h[6]! ^^^ loadWord person 0)
  let h := h.set! 7 (h[7]! ^^^ loadWord person 8)
  let nBlocks := if msg.size = 0 then 1 else (msg.size + 127) / 128
  let h := (List.range nBlocks).foldl
    (fun h i =>
      if i + 1 = nBlocks then compress h msg (128 * i) msg.size true
      else compress h msg (128 * i) (128 * (i + 1)) false)
    h
  let out := (List.range nn).map
    (fun i => (h[i / 8]! >>> (UInt64.ofNat (8 * (i % 8)))).toUInt8)
  ⟨out.toArray⟩

def personFs : ByteArray := "FPN2-FS-v2".toUTF8

def personMerkle : ByteArray := "FPN2-MERKLE-v1".toUTF8

/-- The concrete oracles of the Rust and Python verifiers. -/
def oracles : Rel3Oracles := ⟨blake2b 64 personFs, blake2b 32 personMerkle⟩

def hexDigit (c : Char) : Nat :=
  if '0' ≤ c ∧ c ≤ '9' then c.toNat - '0'.toNat
  else if 'a' ≤ c ∧ c ≤ 'f' then c.toNat - 'a'.toNat + 10
  else 0

def hexPairs : List Char → List UInt8
  | a :: b :: rest => UInt8.ofNat (16 * hexDigit a + hexDigit b) :: hexPairs rest
  | _ => []

def fromHex (s : String) : ByteArray := if s = "-" then ⟨#[]⟩ else ⟨(hexPairs s.toList).toArray⟩

def hexChar (n : Nat) : Char := Char.ofNat (if n < 10 then '0'.toNat + n else 'a'.toNat + n - 10)

def toHex (l : List UInt8) : String :=
  String.ofList (l.flatMap fun b => [hexChar (b.toNat / 16), hexChar (b.toNat % 16)])

end FpnTest
