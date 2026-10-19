# Springer Nature (SN Computer Science) version

Build: `bash build.sh` (needs pdflatex, bibtex and the py313 Python; set `PY` to override). It runs `make_sections.py`, then builds `sn_manuscript.pdf` and `online_resource_1.pdf`.
`make_sections.py` regenerates `sections/`, `tables/` and `references.bib` from `../paper/` (never edit those copies): it rewrites "Supplementary Section/Table/Algorithm S#" to "Online Resource 1, ...", drops linespread hacks and wraps `\resizebox` tables in `varwidth` (needed by the sn-jnl table environment).
Title page, structured abstract, keywords and Declarations are in `sn_manuscript.tex`; the Online Resource 1 header is in `online_resource_1.tex`.
Still manual: ORCID, affiliation address details (street, postcode), funding statement, and a final check of the author-year/numbered style against the journal's current guidelines.
