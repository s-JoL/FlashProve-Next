#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify Lean 4 declaration tag resolution across docs/paper/specs and paper PDF synchronization."""
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

    if errors:
        for e in errors[:50]:
            print(e, file=sys.stderr)
        print(f"FAIL: {len(errors)} check error(s)", file=sys.stderr)
        return 1

    print(
        f"OK: verified {len(files)} files "
        f"(all [Lean: ...] tags resolved across {len(by_mod)} modules, PDF synced)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
