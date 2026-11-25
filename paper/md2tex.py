#!/usr/bin/env python3
"""Convert paper/flashprove-next.md to arXiv/pdflatex & tectonic-compatible LaTeX.

Usage: python3 paper/md2tex.py [in.md] [out.tex]
"""
import re
import sys

UNICODE = {
    "→": r"$\to$",
    "≤": r"$\le$",
    "≥": r"$\ge$",
    "×": r"$\times$",
    "≈": r"$\approx$",
    "∎": r"$\blacksquare$",
    "…": r"\ldots{}",
    "−": r"$-$",
    "⇒": r"$\Rightarrow$",
    "±": r"$\pm$",
    "′": r"$'$",
    "≠": r"$\neq$",
    "∈": r"$\in$",
    "–": "--",
    "—": "---",
    "“": "``",
    "”": "''",
    "’": "'",
    "‘": "`",
    "§": r"\S{}",
    "·": r"\textperiodcentered{}",
    "á": r"\'a",
    "ö": r'\"o',
    "ą": r"\k{a}",
    "ć": r"\'c",
}

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[letterpaper,margin=1in]{geometry}
\usepackage{iftex}
\ifPDFTeX
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
  \usepackage{microtype}
\fi
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{array,tabularx,booktabs}
\usepackage{enumitem}
\usepackage{fancyvrb}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!55!black,citecolor=blue!55!black,urlcolor=blue!55!black]{hyperref}
\setlist{itemsep=2pt,topsep=4pt}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\renewcommand{\arraystretch}{1.15}
\emergencystretch=3em
\hbadness=5000
\sloppy
"""


def esc_text(s: str) -> str:
    out = []
    for ch in s:
        if ch in "&%#_{}$":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\^{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in UNICODE:
            out.append(UNICODE[ch])
        else:
            out.append(ch)
    return "".join(out)


def esc_code(s: str) -> str:
    t = esc_text(s)
    t = t.replace(r"\_", r"\_\allowbreak{}").replace(".", r".\allowbreak{}")
    return r"\texttt{" + t + "}"


def esc_math(s: str) -> str:
    out = []
    for ch in s:
        if ch in UNICODE:
            if ch == "–":
                out.append("--")
            elif ch == "—":
                out.append("---")
            elif ch == "′":
                out.append("'")
            else:
                out.append(UNICODE[ch])
        else:
            out.append(ch)
    return "".join(out)


DOLLAR = "\x00D\x00"


def collect_ref_keys(md: str) -> set[str]:
    keys: set[str] = set()
    in_refs = False
    for ln in md.split("\n"):
        s = ln.strip()
        if s == "## References":
            in_refs = True
            continue
        if in_refs and s.startswith("## "):
            break
        if in_refs:
            m = re.match(r"^- \[([A-Za-z0-9+_-]+)\]\s+", s)
            if m:
                keys.add(m.group(1))
    return keys


def inline(s: str, ref_keys: set[str] | None = None, link_arxiv: bool = False) -> str:
    st = s.strip()
    if st.startswith("*") and st.endswith("*") and not st.startswith("**") and len(st) > 2:
        return r"\emph{" + inline(st[1:-1], ref_keys, link_arxiv) + "}"
    s = s.replace(r"\$", DOLLAR)
    tokens: list[str] = []

    def _stash_raw(latex: str) -> str:
        idx = len(tokens)
        tokens.append(latex)
        return f"\x01T{idx}T\x01"

    def _stash_code_math(m: re.Match) -> str:
        tok = m.group(0)
        if tok.startswith("`"):
            return _stash_raw(esc_code(tok[1:-1]))
        return _stash_raw(esc_math(tok))

    # 1. Stash inline code and math first
    s = re.sub(r"`[^`]+`|\$[^$]+\$", _stash_code_math, s)

    # 2. Stash markdown links [label](https://...) and bare <https://...>
    def _stash_mdlink(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        return _stash_raw(r"\href{" + url + "}{" + esc_text(label) + "}")

    s = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", _stash_mdlink, s)
    s = re.sub(r"<(https?://[^>\s]+)>", lambda m: _stash_raw(r"\url{" + m.group(1) + "}"), s)

    # 3. Stash citations [KEY1, KEY2] when every item is a known bib key
    if ref_keys:
        def _stash_cite(m: re.Match) -> str:
            parts = [p.strip() for p in m.group(1).split(",")]
            if parts and all(p in ref_keys for p in parts):
                return _stash_raw(r"\cite{" + ",".join(parts) + "}")
            return m.group(0)

        s = re.sub(r"\[([A-Za-z0-9+_, -]+)\]", _stash_cite, s)

    # 4. Optionally hyperlink arXiv IDs in References
    if link_arxiv:
        s = re.sub(
            r"\barXiv:(\d{4}\.\d{4,5})\b",
            lambda m: _stash_raw(
                r"\href{https://arxiv.org/abs/" + m.group(1) + "}{arXiv:" + m.group(1) + "}"
            ),
            s,
        )

    # 5. Pure ASCII smart quotes
    s = re.sub(r'"([^"]*)"', r"``\1''", s)

    t = esc_text(s)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", t)
    for idx, val in enumerate(tokens):
        t = t.replace(f"\x01T{idx}T\x01", val)
    return t.replace(DOLLAR, r"\$")


def split_row(line: str):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells, cur, in_math, in_code = [], [], False, False
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line) and line[i + 1] == "$":
            cur.append("\\$")
            i += 2
            continue
        if c == "$" and not in_code:
            in_math = not in_math
        elif c == "`" and not in_math:
            in_code = not in_code
        if c == "|" and not in_math and not in_code:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def _col_spec(header: list[str], body: list[list[str]]) -> str:
    n = len(header)
    if n == 6 and "Full-reply" in header[0]:
        return (
            r">{\hsize=1.55\hsize}Y"
            r">{\hsize=0.95\hsize}Y"
            r">{\hsize=0.70\hsize}Y"
            r">{\hsize=1.00\hsize}Y"
            r">{\hsize=1.10\hsize}Y"
            r">{\hsize=0.70\hsize}Y"
        )
    if n == 2:
        # Give wider second column for Definition/Statement tables
        avg0 = sum(len(r[0]) for r in body) / max(len(body), 1)
        avg1 = sum(len(r[1]) for r in body) / max(len(body), 1)
        if avg1 > 2.2 * max(avg0, 1):
            return r">{\hsize=0.55\hsize}Y>{\hsize=1.45\hsize}Y"
    return "Y" * n


def table(rows, ref_keys: set[str]):
    header = split_row(rows[0])
    body = [split_row(r) for r in rows[2:]]
    n = len(header)
    spec = _col_spec(header, body)
    out = [r"\begin{center}\small", r"\begin{tabularx}{\linewidth}{" + spec + "}", r"\toprule"]
    out.append(" & ".join("{}" + r"\textbf{" + inline(c, ref_keys) + "}" for c in header) + r" \\")
    out.append(r"\midrule")
    for r in body:
        r = (r + [""] * n)[:n]
        out.append(" & ".join("{}" + inline(c, ref_keys) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{center}"]
    return "\n".join(out)


def heading(text: str) -> str:
    return re.sub(r"^\d+(\.\d+)*\s+", "", text.strip())


def convert(md: str) -> str:
    ref_keys = collect_ref_keys(md)
    lines = md.split("\n")
    out = [PREAMBLE]
    title, note = "", ""
    i = 0
    body = []
    in_abstract = False
    in_refs = False
    para = []

    def flush_para():
        nonlocal para
        if para:
            body.append(inline(" ".join(x.strip() for x in para), ref_keys))
            body.append("")
            para = []

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if ln.startswith("# ") and not title:
            title = ln[2:].strip()
            i += 1
            continue
        if not note and s.startswith("*") and s.endswith("*") and not body and title:
            note = s[1:-1]
            i += 1
            continue
        if s.startswith("```"):
            flush_para()
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code.append(lines[j])
                j += 1
            body.append(r"\begin{center}")
            body.append(r"\begin{Verbatim}[fontsize=\fontsize{6.2}{7.2}\selectfont]")
            body.extend(code)
            body.append(r"\end{Verbatim}")
            body.append(r"\end{center}")
            i = j + 1
            continue
        if s.startswith("## "):
            flush_para()
            h = heading(s[3:])
            if in_abstract:
                body.append(r"\end{abstract}")
                in_abstract = False
            if in_refs:
                body.append(r"\end{thebibliography}")
                in_refs = False
            if h == "Abstract":
                body.append(r"\begin{abstract}")
                in_abstract = True
            elif h == "References":
                widest = max(ref_keys, key=len) if ref_keys else "XXXX99"
                body.append(r"\begin{thebibliography}{" + widest + "}")
                in_refs = True
            else:
                body.append(r"\section{" + inline(h, ref_keys) + "}")
            i += 1
            continue
        if s.startswith("### "):
            flush_para()
            body.append(r"\subsection{" + inline(heading(s[4:]), ref_keys) + "}")
            i += 1
            continue
        if in_refs and s.startswith("- ["):
            m = re.match(r"^- \[([A-Za-z0-9+_-]+)\]\s+(.*)", s)
            if m:
                key, rest = m.group(1), m.group(2)
                body.append(
                    r"\bibitem[" + key + "]{" + key + "} " + inline(rest, None, link_arxiv=True)
                )
                i += 1
                continue
        if s.startswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            body.append(table(rows, ref_keys))
            body.append("")
            continue
        if s.startswith("$$"):
            flush_para()
            buf = s[2:]
            if buf.endswith("$$") and len(buf) >= 2:
                content = buf[:-2]
                i += 1
            else:
                parts = [buf]
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    parts.append(lines[i])
                    i += 1
                parts.append(lines[i].strip()[:-2])
                i += 1
                content = "\n".join(parts)
            body.append(r"\[" + esc_math(content.strip()) + r"\]")
            body.append("")
            continue
        if re.match(r"^(- |\d+\. )", s):
            flush_para()
            numbered = bool(re.match(r"^\d+\. ", s))
            env = "enumerate" if numbered else "itemize"
            body.append(r"\begin{" + env + "}")
            item = None
            while i < len(lines):
                t = lines[i]
                ts = t.strip()
                m = re.match(r"^(- |\d+\. )(.*)", ts)
                if m and (bool(re.match(r"^\d+\. ", ts)) == numbered):
                    if item is not None:
                        body.append(r"\item{} " + inline(" ".join(item), ref_keys))
                    item = [m.group(2)]
                elif ts and item is not None and t.startswith("  "):
                    item.append(ts)
                else:
                    break
                i += 1
            if item is not None:
                body.append(r"\item{} " + inline(" ".join(item), ref_keys))
            body.append(r"\end{" + env + "}")
            body.append("")
            continue
        if s.startswith(">"):
            flush_para()
            q = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q.append(lines[i].strip()[1:].strip())
                i += 1
            body.append(r"\begin{quote}" + inline(" ".join(q), ref_keys) + r"\end{quote}")
            continue
        if not s:
            flush_para()
            i += 1
            continue
        para.append(ln)
        i += 1
    flush_para()
    if in_abstract:
        body.append(r"\end{abstract}")
    if in_refs:
        body.append(r"\end{thebibliography}")
    out.append(r"\title{" + inline(title, ref_keys) + "}")
    out.append(
        r"\author{Liang Song\\" "\n"
        r"\small Independent Researcher\\" "\n"
        r"\small \texttt{sl12160010@gmail.com}}"
    )
    out.append(r"\date{" + inline(note, ref_keys) + "}")
    out.append(r"\begin{document}")
    out.append(r"\maketitle")
    out.extend(body)
    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "paper/flashprove-next.md"
    dst = sys.argv[2] if len(sys.argv) > 2 else "paper/flashprove-next.tex"
    with open(src, encoding="utf-8") as f:
        md = f.read()
    with open(dst, "w", encoding="utf-8") as f:
        f.write(convert(md))


if __name__ == "__main__":
    main()
