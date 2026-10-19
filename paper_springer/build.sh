#!/bin/bash
# Build the Springer Nature manuscript and Online Resource 1.
set -u
cd "$(dirname "$0")"
PY="${PY:-/c/Users/vinh.dq4/AppData/Local/anaconda3/envs/py313/python.exe}"
"$PY" make_sections.py || exit 1
pdflatex -interaction=nonstopmode sn_manuscript.tex >/dev/null 2>&1
bibtex -include-directory=bst sn_manuscript >/dev/null 2>&1
pdflatex -interaction=nonstopmode sn_manuscript.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode sn_manuscript.tex > build_main.log 2>&1
pdflatex -interaction=nonstopmode online_resource_1.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode online_resource_1.tex > build_or1.log 2>&1
grep -E "^!|undefined|Overfull" build_main.log build_or1.log | head -30
grep "Output written" build_main.log build_or1.log
