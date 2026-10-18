#!/bin/bash
# Build the manuscript and supplementary PDFs (see HANDOFF.md).
set -u
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode elsevier_manuscript.tex >/dev/null 2>&1
bibtex elsevier_manuscript >/dev/null 2>&1
pdflatex -interaction=nonstopmode elsevier_manuscript.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode elsevier_manuscript.tex > build_main.log 2>&1
pdflatex -interaction=nonstopmode supplementary.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode supplementary.tex > build_supp.log 2>&1
pdflatex -interaction=nonstopmode highlights.tex >/dev/null 2>&1
grep -E "^!|undefined" build_main.log build_supp.log | head
grep "Output written" build_main.log build_supp.log
