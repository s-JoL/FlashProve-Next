/-
Executable parity harness for `verifyBytesRel4` (`FlashProveNext/Fpn2Rel4.lean`) and the dispatcher
`verifyBytesAny` (`FlashProveNext/Fpn2Rel5.lean`, since relation 5 joined it): instantiates the abstract oracles with the concrete BLAKE2b of
`tests/FpnTestLib.lean` (the one `Rel3Vectors.lean` uses) and runs the verified verifiers on raw
container files. Driven by `check_rel4_vectors.py`, which materializes
`specs/vectors/fpn2_folded_air.json` (and, for the dispatcher, `fpn2_folded_gkr.json`) and
compares every printed result.

This file is test code: it is not imported by `FlashProveNext.lean`.

Manifest lines (space separated), one result line per input line:
  `profile <name>`                                    ->  `profile <name> <B256(canonical bytes)> <pinned id>`
  `layout <name>`                                     ->  `layout <name> <header + body bytes | none>`
  `verify4 <name> <0|1> <pin-hex|-> <circ-hex|-> <file>`
        ->  `verify4 <name> <verifyBytesRel4Policy result> <verifyBytesAny result>`
  `any <name> <0|1> <pin-hex|-> <circ-hex|-> <file>`  ->  `any <name> <verifyBytesAny result>`
with results `VERIFIED` or the reject code.
-/
import FlashProveNext.Fpn2Rel5
import tests.FpnTestLib

open FlashProveNext
open FpnTest

namespace Rel4Test

/-- Relation 1 needs Poseidon2, which has no Lean implementation: these oracles are never called
on the relation-3 / relation-4 branches of `verifyBytesAny` exercised here. -/
def harnessOracles : Oracles := ⟨oracles.hFs, fun _ => ⟨0, 0, 0, 0⟩, fun a _ => a⟩

def showResult {α : Type} : Except RejectCode α → String
  | .ok _ => "VERIFIED"
  | .error e => e.toString

def optHex (s : String) : Option Bytes := if s = "-" then none else some (toBytes (fromHex s))

def runLine (line : String) : IO Unit := do
  match line.splitOn " " with
  | ["profile", name] =>
    match profileRegistry.find? (fun p => p.name == name) with
    | none => IO.println s!"profile {name} unknown"
    | some p =>
      let id := blake2b 32 ⟨#[]⟩ (ofBytes p.canonicalBytes)
      IO.println s!"profile {name} {toHex (toBytes id)} {toHex p.profileId}"
  | ["layout", name] =>
    match (profileRegistry.find? (fun p => p.name == name)).bind rel4Layout with
    | none => IO.println s!"layout {name} none"
    | some lay => IO.println s!"layout {name} {headerLen + lay.bodyUsed}"
  | ["verify4", name, allow, pin, circ, path] =>
    let raw ← IO.FS.readBinFile path
    let r := verifyBytesRel4Policy oracles PublicInput.v2 (allow = "1") (optHex pin) (optHex circ) raw
    let a := verifyBytesAny harnessOracles oracles PublicInput.v2
      ⟨allow = "1", optHex pin, optHex circ⟩ raw
    IO.println s!"verify4 {name} {showResult r} {showResult a}"
    (← IO.getStdout).flush
  | ["any", name, allow, pin, circ, path] =>
    let raw ← IO.FS.readBinFile path
    let a := verifyBytesAny harnessOracles oracles PublicInput.v2
      ⟨allow = "1", optHex pin, optHex circ⟩ raw
    IO.println s!"any {name} {showResult a}"
    (← IO.getStdout).flush
  | _ => pure ()

end Rel4Test

def main (args : List String) : IO UInt32 := do
  match args with
  | [manifest] =>
    for line in (← IO.FS.readFile manifest).splitOn "\n" do
      Rel4Test.runLine line
    return 0
  | _ =>
    IO.eprintln "usage: Rel4Vectors <manifest>"
    return 2
