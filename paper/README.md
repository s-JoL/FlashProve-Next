# Paper (`paper/`)

Technical report for **FlashProve-Next**:

- **[`flashprove-next.md`](flashprove-next.md)** — *FlashProve-Next: Verifiable Inference for Production LLM APIs* (English source, tagging every claim by explicit evidence level: `[Def]`, `[Paper]`, `[Lean: name]`, `[Assume]`, `[Meas: model, machine]`, `[Est]`, `[Open]`).
- **[`flashprove-next.pdf`](flashprove-next.pdf)** — Compiled PDF (mirrored at [`docs/paper/FlashProve-Next.pdf`](../docs/paper/FlashProve-Next.pdf) for the project homepage).

## Rebuild PDF

Requires Python 3 and [`tectonic`](https://tectonic-typesetting.github.io/):

```bash
python3 paper/md2tex.py paper/flashprove-next.md paper/flashprove-next.tex
cd paper && tectonic flashprove-next.tex
cp flashprove-next.pdf ../docs/paper/FlashProve-Next.pdf
```
