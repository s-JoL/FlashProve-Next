#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify Lean 4 declaration tags, relative Markdown links, Lean module/declaration counts, CI unit-test shards
and paper PDF sync.

The leak and prose pattern lists are local-only (`.git/info/local_check.json`); without that file, as on a CI
runner or in a linked worktree, those two checks do not run and the final line says so.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_CFG = ROOT / ".git" / "info" / "local_check.json"
PROSE_EXTS = {".md", ".html", ".tex"}

DECL_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+)*"
    r"(?:theorem|lemma|def|abbrev|structure|inductive)\s+([A-Za-z0-9_']+)",
    re.MULTILINE,
)
LEAN_TAG_RE = re.compile(r"\[Lean[^:\]]*:\s*([^\]]+)\]")
BACKTICK_IDENT_RE = re.compile(r"`([A-Za-z0-9_.']+)`")
MD_LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
EXTERNAL_LINK_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#)", re.IGNORECASE)
AXIOM_LINE_RE = re.compile(r"^#print axioms\b", re.MULTILINE)
LEAN_BADGE_RE = re.compile(r"badge/Lean_4\.29\.1-(\d+)_[^_]+_·_(\d+)_")


def collect_lean_decls() -> tuple[dict[str, set[str]], set[str]]:
    by_mod: dict[str, set[str]] = {}
    all_short: set[str] = set()
    lean_dir = ROOT / "lean" / "FlashProveNext"
    for p in sorted(lean_dir.glob("*.lean")):
        mod = p.stem
        txt = p.read_text(encoding="utf-8")
        names = set(DECL_RE.findall(txt))
        by_mod[mod] = names
        all_short.update(names)
    return by_mod, all_short


def candidate_files() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        paths = [ROOT / ln.strip() for ln in out.splitlines() if ln.strip()]
        return [p for p in paths if p.is_file()]
    except Exception:
        return [
            p for p in ROOT.rglob("*")
            if p.is_file()
            and ".git" not in p.parts
            and ".lake" not in p.parts
            and ".olean" not in p.parts
        ]


def check_lean_counts(errors: list[str]) -> tuple[int, int]:
    """The module and axiom-audited declaration counts stated in the docs must equal the Lean sources."""
    lean_dir = ROOT / "lean"
    sources = sorted((lean_dir / "FlashProveNext").glob("*.lean"))
    n_mod = len(sources)
    n_decl = sum(len(AXIOM_LINE_RE.findall(p.read_text(encoding="utf-8"))) for p in sources)

    m = re.search(r'^MODULES="([^"]*)"', (lean_dir / "check.sh").read_text(encoding="utf-8"), re.MULTILINE)
    listed = set(m.group(1).split()) if m else set()
    if listed != {p.stem for p in sources}:
        diff = sorted(listed ^ {p.stem for p in sources})
        errors.append(f"[LEAN-COUNT] lean/check.sh MODULES differs from lean/FlashProveNext/*.lean: {diff}")

    theorem = (lean_dir / "THEOREM.md").read_text(encoding="utf-8")
    m = re.search(r"Theorem Inventory \((\d+) Modules, (\d+) Axiom-Checked Declarations\)", theorem)
    if not m or (int(m.group(1)), int(m.group(2))) != (n_mod, n_decl):
        stated = m.groups() if m else None
        errors.append(f"[LEAN-COUNT] lean/THEOREM.md inventory header states {stated}, sources have ({n_mod}, {n_decl})")

    for name in ("README.md", "README.zh-CN.md"):
        path = ROOT / name
        if not path.exists():
            continue
        m = LEAN_BADGE_RE.search(path.read_text(encoding="utf-8"))
        if not m or (int(m.group(1)), int(m.group(2))) != (n_mod, n_decl):
            stated = m.groups() if m else None
            errors.append(f"[LEAN-COUNT] {name} Lean badge states {stated}, sources have ({n_mod}, {n_decl})")
    return n_mod, n_decl


def check_ci_shards(errors: list[str]) -> None:
    """The CI matrix must run exactly the shards that scripts/ci_unit_shards.py defines."""
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    script = ROOT / "scripts" / "ci_unit_shards.py"
    if not workflow.exists() or not script.exists():
        return
    m = re.search(r"^\s*shard:\s*\[([^\]]*)\]", workflow.read_text(encoding="utf-8"), re.MULTILINE)
    in_ci = {s.strip() for s in m.group(1).split(",")} if m else set()
    out = subprocess.run([sys.executable, str(script), "--list"], cwd=ROOT, text=True, capture_output=True)
    if out.returncode != 0:
        errors.append(f"[CI-SHARDS] scripts/ci_unit_shards.py --list failed: {(out.stdout + out.stderr).strip()}")
        return
    defined = {ln.split(":", 1)[0] for ln in out.stdout.splitlines() if ":" in ln}
    if in_ci != defined:
        errors.append(f"[CI-SHARDS] ci.yml matrix runs {sorted(in_ci)}, scripts/ci_unit_shards.py defines {sorted(defined)}")


def main() -> int:
    errors: list[str] = []

    leak_pats: list[str] = []
    prose_pats: list[str] = []
    disallowed_tracked: set[str] = set()
    if LOCAL_CFG.exists():
        cfg = json.loads(LOCAL_CFG.read_text(encoding="utf-8"))
        leak_pats = list(cfg.get("leak_patterns") or [])
        prose_pats = list(cfg.get("prose_patterns") or [])
        disallowed_tracked = set(cfg.get("disallowed_tracked") or [])

    if disallowed_tracked:
        try:
            tracked = set(
                subprocess.check_output(
                    ["git", "ls-files"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
                ).splitlines()
            )
            for item in sorted(disallowed_tracked & tracked):
                errors.append(f"local-only file {item} must not be tracked in git")
        except Exception:
            pass

    by_mod, all_short = collect_lean_decls()
    files = candidate_files()

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel in disallowed_tracked:
            continue
        if path.suffix in {".pdf", ".proof", ".npz", ".npy", ".so", ".dylib", ".png", ".jpg", ".gz"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for pat in leak_pats:
            for m in re.finditer(pat, text):
                line_no = text.count("\n", 0, m.start()) + 1
                errors.append(f"[LEAK] {rel}:{line_no}: matched {pat!r} ({m.group(0)!r})")

        if path.suffix in PROSE_EXTS:
            for pat in prose_pats:
                for m in re.finditer(pat, text):
                    line_no = text.count("\n", 0, m.start()) + 1
                    errors.append(f"[STYLE] {rel}:{line_no}: matched {pat!r} ({m.group(0)!r})")

        if path.suffix == ".md":
            for m in MD_LINK_RE.finditer(text):
                target = m.group(1)
                if EXTERNAL_LINK_RE.match(target):
                    continue
                target_path = target.split("#", 1)[0]
                if target_path and not (path.parent / target_path).exists():
                    line_no = text.count("\n", 0, m.start()) + 1
                    errors.append(f"[LINK] {rel}:{line_no}: relative link target {target!r} does not exist")

        if path.suffix == ".md":
            for m in LEAN_TAG_RE.finditer(text):
                body = m.group(1)
                line_no = text.count("\n", 0, m.start()) + 1
                for ident in BACKTICK_IDENT_RE.findall(body):
                    if ident.endswith("*") or ident in ("sorry", "name", "Module.decl"):
                        continue
                    if "." in ident:
                        mod, decl = ident.split(".", 1)
                        if mod in by_mod and decl not in by_mod[mod]:
                            errors.append(f"[LEAN-TAG] {rel}:{line_no}: `{ident}` not found in {mod}.lean")
                        elif mod not in by_mod:
                            errors.append(f"[LEAN-TAG] {rel}:{line_no}: unknown Lean module `{mod}` in `{ident}`")
                    else:
                        if ident not in by_mod and ident not in all_short:
                            errors.append(f"[LEAN-TAG] {rel}:{line_no}: `{ident}` not found in any Lean module")

    pdf_main = ROOT / "paper" / "flashprove-next.pdf"
    pdf_docs = ROOT / "docs" / "paper" / "FlashProve-Next.pdf"
    if pdf_main.exists() and pdf_docs.exists():
        if pdf_main.read_bytes() != pdf_docs.read_bytes():
            errors.append("[PDF-SYNC] paper/flashprove-next.pdf != docs/paper/FlashProve-Next.pdf")

    n_mod, n_decl = check_lean_counts(errors)
    check_ci_shards(errors)

    if errors:
        for e in errors[:50]:
            print(e, file=sys.stderr)
        print(f"FAIL: {len(errors)} check error(s)", file=sys.stderr)
        return 1

    patterns = "leak/prose patterns checked" if LOCAL_CFG.exists() else "leak/prose patterns NOT checked: no local config"
    print(
        f"OK: verified {len(files)} files "
        f"(all [Lean: ...] tags resolved across {len(by_mod)} modules, relative Markdown links resolve, "
        f"docs state {n_mod} Lean modules / {n_decl} declarations, PDF synced; {patterns})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
