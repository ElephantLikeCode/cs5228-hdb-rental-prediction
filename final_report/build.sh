#!/usr/bin/env bash
# Typeset committed report assets; no training or data lookup.
set -euo pipefail
task_project="$(cd "$(dirname "$0")/.." && pwd)"
task_build="$task_project/artifacts/final_report_build"
task_output="$task_project/output/pdf/CS5228_Final_Report.pdf"
mkdir -p "$task_build"
cd "$task_project/final_report"
if [[ -n "${TECTONIC:-}" ]] || command -v tectonic >/dev/null 2>&1; then
    task_compiler="${TECTONIC:-tectonic}"
    "$task_compiler" --keep-logs --outdir "$task_build" main.tex
    cp "$task_build/main.pdf" "$task_output"
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$task_build" -jobname=CS5228_Final_Report main.tex
    # BibTeX's restricted output policy rejects absolute output filenames.
    # Run beside the aux file while retaining access to the source bibliography.
    (
        cd "$task_build"
        BIBINPUTS="$task_project/final_report:${BIBINPUTS:-}" bibtex CS5228_Final_Report
    )
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$task_build" -jobname=CS5228_Final_Report main.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$task_build" -jobname=CS5228_Final_Report main.tex
    cp "$task_build/CS5228_Final_Report.pdf" "$task_output"
else
    echo 'Install Tectonic or a TeX distribution with pdflatex and bibtex; optionally set TECTONIC=/path/to/tectonic.' >&2
    exit 1
fi
echo "Report: $task_output"
