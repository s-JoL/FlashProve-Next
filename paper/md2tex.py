#!/usr/bin/env python3
"""Convert paper/flashprove-next.md to paper/flashprove-next.tex.

Usage: python3 paper/md2tex.py [in.md] [out.tex]
"""
import re
import sys

UNICODE = {
    "→": r"$\to$", "≤": r"$\le$", "≥": r"$\ge$", "×": r"$\times$", "≈": r"$\approx$",
    "∎": r"$\blacksquare$", "…": r"\ldots{}", "−": r"$-$", "⇒": r"$\Rightarrow$",
    "±": r"$\pm$", "′": r"$'$", "≠": r"$\neq$", "∈": r"$\in$",
}

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[letterpaper,margin=1in]{geometry}
\usepackage{fontspec}
\usepackage{amsmath,amssymb}
\usepackage{array,tabularx,booktabs}
\usepackage{enumitem}
\usepackage{fancyvrb}
\usepackage[hidelinks]{hyperref}
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
        if ch in "&%#_{}":
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


DOLLAR = "\x00D\x00"


def inline(s: str) -> str:
    st = s.strip()
    if st.startswith("*") and st.endswith("*") and not st.startswith("**") and len(st) > 2:
        return r"\emph{" + inline(st[1:-1]) + "}"
    s = re.sub(r'"([^"$`]*)"', "\u201c\\1\u201d", s)
    s = s.replace(r"\$", DOLLAR)
    parts = re.split(r"(\$[^$]+\$|`[^`]+`)", s)
    res = []
    for p in parts:
        if not p:
            continue
        if p.startswith("$") and p.endswith("$") and len(p) > 1:
            res.append(p)
        elif p.startswith("`") and p.endswith("`"):
            res.append(esc_code(p[1:-1]))
        else:
            t = esc_text(p)
            t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
            t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", t)
            res.append(t)
    return "".join(res).replace(DOLLAR, r"\$")


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
            cur.append("\\$"); i += 2; continue
        if c == "$" and not in_code:
            in_math = not in_math
        elif c == "`" and not in_math:
            in_code = not in_code
        if c == "|" and not in_math and not in_code:
            cells.append("".join(cur).strip()); cur = []
        else:
            cur.append(c)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def table(rows):
    header = split_row(rows[0])
    body = [split_row(r) for r in rows[2:]]
    n = len(header)
    spec = "Y" * n
    out = [r"\begin{center}\small", r"\begin{tabularx}{\linewidth}{" + spec + "}", r"\toprule"]
    out.append(" & ".join("{}" + r"\textbf{" + inline(c) + "}" for c in header) + r" \\")
    out.append(r"\midrule")
    for r in body:
        r = (r + [""] * n)[:n]
        out.append(" & ".join("{}" + inline(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabularx}", r"\end{center}"]
    return "\n".join(out)


def heading(text: str) -> str:
    return re.sub(r"^\d+(\.\d+)*\s+", "", text.strip())


def convert(md: str) -> str:
    lines = md.split("\n")
    out = [PREAMBLE]
    title, note = "", ""
    i = 0
    body = []
    in_abstract = False
    para = []

    def flush_para():
        nonlocal para
        if para:
            body.append(inline(" ".join(x.strip() for x in para)))
            body.append("")
            para = []

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if ln.startswith("# ") and not title:
            title = ln[2:].strip(); i += 1; continue
        if not note and s.startswith("*") and s.endswith("*") and not body and title:
            note = s[1:-1]; i += 1; continue
        if s.startswith("```"):
            flush_para()
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code.append(lines[j]); j += 1
            body.append(r"\begin{center}")
            body.append(r"\begin{Verbatim}[fontsize=\fontsize{6.2}{7.2}\selectfont]")
            body.extend(code)
            body.append(r"\end{Verbatim}")
            body.append(r"\end{center}")
            i = j + 1; continue
        if s.startswith("## "):
            flush_para()
            h = heading(s[3:])
            if in_abstract:
                body.append(r"\end{abstract}"); in_abstract = False
            if h == "Abstract":
                body.append(r"\begin{abstract}"); in_abstract = True
            elif h == "References":
                body.append(r"\section*{References}")
                body.append(r"\addcontentsline{toc}{section}{References}")
            else:
                body.append(r"\section{" + inline(h) + "}")
            i += 1; continue
        if s.startswith("### "):
            flush_para()
            body.append(r"\subsection{" + inline(heading(s[4:])) + "}")
            i += 1; continue
        if s.startswith("|"):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i]); i += 1
            body.append(table(rows)); body.append("")
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
                    parts.append(lines[i]); i += 1
                parts.append(lines[i].strip()[:-2]); i += 1
                content = "\n".join(parts)
            body.append(r"\[" + content.strip() + r"\]")
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
                        body.append(r"\item{} " + inline(" ".join(item)))
                    item = [m.group(2)]
                elif ts and item is not None and t.startswith("  "):
                    item.append(ts)
                else:
                    break
                i += 1
            if item is not None:
                body.append(r"\item{} " + inline(" ".join(item)))
            body.append(r"\end{" + env + "}")
            body.append("")
            continue
        if s.startswith(">"):
            flush_para()
            q = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                q.append(lines[i].strip()[1:].strip()); i += 1
            body.append(r"\begin{quote}" + inline(" ".join(q)) + r"\end{quote}")
            continue
        if not s:
            flush_para(); i += 1; continue
        para.append(ln)
        i += 1
    flush_para()
    if in_abstract:
        body.append(r"\end{abstract}")
    out.append(r"\title{" + inline(title) + "}")
    out.append(r"\author{Liang Song}")
    out.append(r"\date{" + inline(note) + "}")
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
