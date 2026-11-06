# Compile the stable final-report PDF without requiring latexmk/Perl.
# This only typesets existing evidence. Run from any directory:
# ./final_report/build.ps1
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskBuild = Join-Path $taskProject 'artifacts/final_report_build'
$taskOutput = Join-Path $taskProject 'final_report/project-report-team44.pdf'
New-Item -ItemType Directory -Path $taskBuild -Force | Out-Null
Push-Location $PSScriptRoot
try {
    & pdflatex '--interaction=nonstopmode' '--halt-on-error' "--output-directory=$taskBuild" '--jobname=CS5228_Final_Report' main.tex
    if ($LASTEXITCODE -ne 0) { throw 'First LaTeX pass failed.' }
    & bibtex (Join-Path $taskBuild 'CS5228_Final_Report')
    if ($LASTEXITCODE -ne 0) { throw 'Bibliography pass failed.' }
    foreach ($taskPass in 1..2) {
        & pdflatex '--interaction=nonstopmode' '--halt-on-error' "--output-directory=$taskBuild" '--jobname=CS5228_Final_Report' main.tex
        if ($LASTEXITCODE -ne 0) { throw "LaTeX pass $taskPass failed." }
    }
    Copy-Item -LiteralPath (Join-Path $taskBuild 'CS5228_Final_Report.pdf') -Destination $taskOutput
    Write-Output "Report: $taskOutput"
} finally {
    Pop-Location
}
