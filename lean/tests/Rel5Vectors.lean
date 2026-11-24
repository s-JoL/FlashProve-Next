/-
Executable parity harness for `verifyBytesRel5` and the four-relation dispatcher `verifyBytesAny`
(`FlashProveNext/Fpn2Rel5.lean`): instantiates the abstract oracles with the concrete BLAKE2b of
`tests/FpnTestLib.lean` (the one `Rel3Vectors.lean` and `Rel4Vectors.lean` use) and runs the
verified verifiers on raw container files. Driven by `check_rel5_vectors.py`, which materializes
`specs/vectors/fpn2_folded_vec.json` and compares every printed result.

This file is test code: it is not imported by `FlashProveNext.lean`.

Manifest lines (space separated), one result line per input line:
  `profile <name>`  ->  `profile <name> <B256(canonical bytes)> <pinned id> <relation number>`
  `layout <name>`   ->  `layout <name> <header + body bytes | none>`
  `verify5 <name> <0|1> <pin-hex|-> <weight-root-hex|-> <file>`
        ->  `verify5 <name> <verifyBytesRel5Policy result> <verifyBytesAny result>`
  `weight <name> <n> <k> <3 limbs per Fp3: r_i (5), r_b (beta), s (d), lam, e_W, mu, rs (n + 2)>`
        ->  `weight <name> <a0> <a1> <a2>` of `finalWeight5 (rel5Terms …) rs`
  `tprof <name> <tier> <bits> <n> <Q> <K> <C_p>`
        ->  `tprof <name> <B256(canonical bytes)> <beta> <tiles> <d> <header + body bytes | none>`
  `tile5 <case> <name> <tier> <bits> <n> <Q> <K> <C_p> <weight-root-hex|-> <file>`
        ->  `tile5 <case> <verifyParsedRel5 result>`
  `tchal <name> <tier> <bits> <n> <Q> <K> <C_p> <file>`
        ->  `tchal <name> <3 limbs per Fp3: r_i (5), r_b (beta), lam, s (d), mu>` (or `tchal <name> <code>`)
with results `VERIFIED` or the reject code.

The last three lines run a relation-5 container of a test-only, unregistered profile (the multi-tile
fixture `tests/rel5_multitile.json`, `beta = 2`) through `verifyParsedRel5` (`Fpn2Rel5.lean`):
`parseContainerRel5`, the weight pin and `rel5Check`. The registry lookup and the tier policy
(checks 6-8) are not involved. `verifyBytesRel5_eq_verifyParsedRel5` proves that `verifyBytesRel5`
returns the result of this same function on every registered profile passing the tier policy when
the header names it, and `verifyBytesRel5_foldedVecN22` states it for `fpn2-folded-vec-n22-v1`
(`beta = 6`, 64 tiles), whose bytes are not executed here.
-/
import FlashProveNext.Fpn2Rel5
import tests.FpnTestLib

open FlashProveNext
open FpnTest

namespace Rel5Test

/-- Relation 1 needs Poseidon2, which has no Lean implementation: these oracles are never called
on the relation-5 branch of `verifyBytesAny` exercised here. -/
def harnessOracles : Oracles := ⟨oracles.hFs, fun _ => ⟨0, 0, 0, 0⟩, fun a _ => a⟩

def showResult {α : Type} : Except RejectCode α → String
  | .ok _ => "VERIFIED"
  | .error e => e.toString

def optHex (s : String) : Option Bytes := if s = "-" then none else some (toBytes (fromHex s))

/-- Consecutive limb triples as `Fp3` values. -/
def fp3s : List String → List Fp3
  | a :: b :: c :: rest => ⟨fpOfNat a.toNat!, fpOfNat b.toNat!, fpOfNat c.toNat!⟩ :: fp3s rest
  | _ => []

/-- A relation-5 profile that is not in the registry (test only); its `profileId` is the BLAKE2b-256
of its canonical bytes, as for the registered profiles. -/
def testProfile (name tier bits n q kdim cp : String) : Fpn2Profile :=
  let p0 : Fpn2Profile :=
    { name := name, securityTier := if tier = "1" then .standard else .reduced,
      relation := .foldedVecMatmul, securityBits := bits.toNat!, maxLayers := 0,
      maxHidden := kdim.toNat!, maxSeqLen := 0, maxVocab := 0, maxNewTokens := 0, stopIds := [],
      nVars := n.toNat!, scDegree := 3, nQueries := q.toNat!, containerBytes := cp.toNat!,
      profileId := [] }
  { p0 with profileId := toBytes (blake2b 32 ⟨#[]⟩ (ofBytes p0.canonicalBytes)) }

def limbs (l : List Fp3) : String :=
  " ".intercalate (l.flatMap fun a => [toString a.a0.toNat, toString a.a1.toNat, toString a.a2.toNat])

def runLine (line : String) : IO Unit := do
  match line.splitOn " " with
  | ["profile", name] =>
    match profileRegistry.find? (fun p => p.name == name) with
    | none => IO.println s!"profile {name} unknown"
    | some p =>
      let id := blake2b 32 ⟨#[]⟩ (ofBytes p.canonicalBytes)
      IO.println s!"profile {name} {toHex (toBytes id)} {toHex p.profileId} {p.relation.toNat}"
  | ["layout", name] =>
    match (profileRegistry.find? (fun p => p.name == name)).bind rel5Layout with
    | none => IO.println s!"layout {name} none"
    | some lay => IO.println s!"layout {name} {headerLen + lay.bodyUsed}"
  | ["verify5", name, allow, pin, wroot, path] =>
    let raw ← IO.FS.readBinFile path
    let r := verifyBytesRel5Policy oracles PublicInput.v2 (allow = "1") (optHex pin) (optHex wroot)
      raw
    let a := verifyBytesAny harnessOracles oracles PublicInput.v2
      ⟨allow = "1", optHex pin, optHex wroot⟩ raw
    IO.println s!"verify5 {name} {showResult r} {showResult a}"
    (← IO.getStdout).flush
  | "weight" :: name :: n :: k :: limbs =>
    let lay : Rel5Layout := ⟨n.toNat!, 1, k.toNat!⟩
    let v := fp3s limbs
    let rI := v.take 5
    let rB := (v.drop 5).take lay.beta
    let s := (v.drop (5 + lay.beta)).take lay.d
    let tail := v.drop (5 + lay.beta + lay.d)
    let tm := rel5Terms lay rI rB s (tail.getD 0 Fp3.zero) (tail.getD 1 Fp3.zero)
      (tail.getD 2 Fp3.zero)
    let w := finalWeight5 lay tm (tail.drop 3)
    IO.println s!"weight {name} {w.a0.toNat} {w.a1.toNat} {w.a2.toNat}"
  | ["tprof", name, tier, bits, n, q, kdim, cp] =>
    let prof := testProfile name tier bits n q kdim cp
    match rel5Layout prof with
    | none => IO.println s!"tprof {name} {toHex prof.profileId} none"
    | some lay =>
      IO.println
        s!"tprof {name} {toHex prof.profileId} {lay.beta} {lay.tiles} {lay.d} {headerLen + lay.bodyUsed}"
  | ["tile5", case, name, tier, bits, n, q, kdim, cp, wroot, path] =>
    let raw ← IO.FS.readBinFile path
    let r := verifyParsedRel5 oracles PublicInput.v2 (testProfile name tier bits n q kdim cp)
      (optHex wroot) raw
    IO.println s!"tile5 {case} {showResult r}"
    (← IO.getStdout).flush
  | ["tchal", name, tier, bits, n, q, kdim, cp, path] =>
    let raw ← IO.FS.readBinFile path
    let prof := testProfile name tier bits n q kdim cp
    match parseContainerRel5 prof raw, rel5Layout prof with
    | .ok c, some lay =>
      match ipPhase oracles PublicInput.v2 lay c with
      | .ok sf =>
        let v := (ipRowTr oracles PublicInput.v2 c).1 ++ (ipTileTr oracles PublicInput.v2 lay c).1 ++
          [(ipLamTr oracles PublicInput.v2 lay c).1] ++ sf.rs ++ [(pcsMuTr5 oracles c sf).1]
        IO.println s!"tchal {name} {limbs v}"
      | .error e => IO.println s!"tchal {name} {e.toString}"
    | .error e, _ => IO.println s!"tchal {name} {e.toString}"
    | _, none => IO.println s!"tchal {name} none"
  | _ => pure ()

end Rel5Test

def main (args : List String) : IO UInt32 := do
  match args with
  | [manifest] =>
    for line in (← IO.FS.readFile manifest).splitOn "\n" do
      Rel5Test.runLine line
    return 0
  | _ =>
    IO.eprintln "usage: Rel5Vectors <manifest>"
    return 2
