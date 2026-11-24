/-
Executable parity harness for `verifyBytesRel3` (`FlashProveNext/Fpn2Rel3.lean`): instantiates the
abstract `Rel3Oracles` with the concrete BLAKE2b of `tests/FpnTestLib.lean` (RFC 7693, personalized,
unkeyed, written in Lean), and runs the verified verifier on raw container files. Driven by `check_rel3_vectors.py`,
which materializes `specs/vectors/fpn2_folded_gkr.json` and compares every printed result.

This file is test code: it is not imported by `FlashProveNext.lean` and nothing is proved about
the BLAKE2b it imports (it is checked against the `hashlib` KATs of `fpn2_primitives.json`).

Manifest lines (space separated), one result line per input line:
  `hash <nn> <person-hex|-> <msg-hex|->`        ->  `hash <digest-hex>`
  `fp3inv <a0> <a1> <a2>`                       ->  `fp3inv <b0> <b1> <b2> <x·x⁻¹ = 1>` or `fp3inv none`
  `profile <name>`                              ->  `profile <name> <B256(canonical bytes)> <pinned id>`
  `verify <name> <0|1> <pin-hex|-> <file>`      ->  `verify <name> <VERIFIED|E_...>`
-/
import FlashProveNext.Fpn2Rel3
import tests.FpnTestLib

open FlashProveNext

open FpnTest

namespace Rel3Test

def resultString : Except RejectCode Rel3Claim → String
  | .ok _ => "VERIFIED"
  | .error e => e.toString

def runLine (line : String) : IO Unit := do
  match line.splitOn " " with
  | ["hash", nn, person, msg] =>
    IO.println s!"hash {toHex (toBytes (blake2b nn.toNat! (fromHex person) (fromHex msg)))}"
  | ["fp3inv", a0, a1, a2] =>
    let x : Fp3 := ⟨fpOfNat a0.toNat!, fpOfNat a1.toNat!, fpOfNat a2.toNat!⟩
    match x.inv with
    | none => IO.println "fp3inv none"
    | some y => IO.println s!"fp3inv {y.a0.toNat} {y.a1.toNat} {y.a2.toNat} {decide (x.mul y = Fp3.one)}"
  | ["profile", name] =>
    match profileRegistry.find? (fun p => p.name == name) with
    | none => IO.println s!"profile {name} unknown"
    | some p =>
      let id := blake2b 32 ⟨#[]⟩ (ofBytes p.canonicalBytes)
      IO.println s!"profile {name} {toHex (toBytes id)} {toHex p.profileId}"
  | ["verify", name, allow, pin, path] =>
    let raw ← IO.FS.readBinFile path
    let expected : Option Bytes := if pin = "-" then none else some (toBytes (fromHex pin))
    let r := verifyBytesRel3Policy oracles PublicInput.v2 (allow = "1") expected raw
    IO.println s!"verify {name} {resultString r}"
    (← IO.getStdout).flush
  | _ => pure ()

end Rel3Test

def main (args : List String) : IO UInt32 := do
  match args with
  | [manifest] =>
    for line in (← IO.FS.readFile manifest).splitOn "\n" do
      Rel3Test.runLine line
    return 0
  | _ =>
    IO.eprintln "usage: Rel3Vectors <manifest>"
    return 2
