"""Generate Springer-adapted copies of the shared manuscript sources.

Reads paper/sec*.tex (main text) and paper/supplementary_*.tex plus
paper/tables/*.tex, and writes adapted copies under paper_springer/.
Edit the prose in paper/ only; re-run this script to propagate.

Adaptations:
  * 'Supplementary Section/Table/Algorithm S#' -> 'Online Resource 1, ...'
    (first mention in each file: 'Online Resource 1 (Section S#)', or the
    comma form when already inside a parenthesis)
  * drop linespread hacks that fight the class; wrap resizebox tables in varwidth
"""
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "paper"

KIND = r"(Sections?|Tables?|Algorithms?)"
PATTERN = re.compile(r"Supplementary\s+" + KIND + r"(~|\s)")


def rewrite_online_resource(text: str) -> str:
    out, last, first = [], 0, True
    for m in PATTERN.finditer(text):
        prefix = text[last:m.start()]
        out.append(prefix)
        kind, sep = m.group(1), m.group(2)
        line_start = text.rfind("\n", 0, m.start()) + 1
        before = text[line_start:m.start()]
        in_paren = before.count("(") > before.count(")")
        if first and not in_paren:
            # Wrap the first reference and its following list in parentheses.
            end = re.compile(r"S\d+(?:(?:--|,\s*|\s+and~?\s*)S\d+)*").match(text, m.end())
            if end:
                out.append("Online Resource~1 (" + kind + "~" + text[m.end():end.end()] + ")")
                last = end.end()
                first = False
                continue
        out.append("Online Resource~1, " + kind + "~")
        last = m.end()
        first = False
    out.append(text[last:])
    return "".join(out)


RESIZE = re.compile(r"\\resizebox\{\\textwidth\}\{!\}\{%\n(\\input\{[^}]*\}%?)\n\}")


def strip_hacks(text: str) -> str:
    text = re.sub(r"^\\linespread\{1\}\\selectfont\s*\n", "", text, flags=re.M)
    # The sn-jnl table environment (threeparttable) breaks a tabular inside a
    # bare \resizebox; a varwidth box of natural width avoids that.
    return RESIZE.sub(
        lambda m: "\\resizebox{\\textwidth}{!}{\\begin{varwidth}{\\maxdimen}\\noindent%\n"
        + m.group(1) + "\n\\end{varwidth}}",
        text,
    )


def main() -> None:
    (HERE / "sections").mkdir(exist_ok=True)
    for f in sorted(SRC.glob("sec*.tex")):
        t = strip_hacks(rewrite_online_resource(f.read_text(encoding="utf-8")))
        (HERE / "sections" / f.name).write_text(t, encoding="utf-8")
    for name in ("supplementary_body.tex", "supplementary_extra.tex"):
        t = (SRC / name).read_text(encoding="utf-8")
        (HERE / "sections" / ("or1_" + name[len("supplementary_"):])).write_text(t, encoding="utf-8")
    dst = HERE / "tables"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC / "tables", dst)
    shutil.copyfile(SRC / "references.bib", HERE / "references.bib")


if __name__ == "__main__":
    main()
