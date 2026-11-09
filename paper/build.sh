#!/usr/bin/env bash
set -euo pipefail
paper_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$paper_dir"
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error manuscript.tex
