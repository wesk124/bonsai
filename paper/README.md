# LaTeX manuscript and result figures

`manuscript.pdf` is the rendered version of the updated manuscript in
`docs/PAPER_FUSED_DRAFT.md`. The LaTeX source includes the measured retrieval,
communication, latency, codec, and memory results. Figures are 300-dpi PNGs.

## Build the PDF

From this directory:

```bash
bash build.sh
```

The build uses `latexmk`, `pdflatex`, BibTeX, and standard TeX Live packages.
The supplied figure PNGs allow PDF compilation without Python or GPU access.
The finished paper has eight pages and six result figures.

## Regenerate the figures

```bash
python make_figures.py
```

Figure generation uses the validated data snapshots in `data/`. Their manifest
records original paths and hashes. Numerical dependencies and exact input
provenance are recorded with the figure-generation outputs. Regeneration
requires NumPy, Matplotlib, and Seaborn; the tested environment uses Python
3.11 and Seaborn 0.13.2.

Exact tested Python versions are listed in `requirements-figures.txt`.
`python validate_pdf.py` checks the compiled PDF's authors, key result values,
embedded figures, and resolved references; it also requires `pypdf`.

## Evidence scope

The original study contains 900 unique final prompts and 6,300 predictions.
The optimized implementation's performance pilot contains nine unique prompts
repeated in three process blocks, with 378 prediction observations. Codec-only
measurements and complete handoff measurements have separate timing scopes.
Empirical pilot percentiles and block ranges are descriptive. The original
two-percentage-point quality-preservation margin remains statistically
inconclusive despite no observed retrieval loss.

The original frozen experiments and manuscripts are preserved. This directory
contains the LaTeX conversion, portable figure inputs, generated plots, and PDF.
