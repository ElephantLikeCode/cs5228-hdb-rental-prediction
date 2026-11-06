# CS5228 HDB Rental Prediction

**ELEPHANT_in_the_Data (44)** — National University of Singapore, CS5228.

Predict recorded monthly HDB rents from flat, location and historical-comparable
information. This is an empirical course project, not a causal housing-price
model. Local chronological evaluation is primary; Kaggle is supplementary.

## Reports

- [Final report — 8 body + 1 references + 2 appendix pages](final_report/project-report-team44.pdf)
- [Progress report](progress_report/project-progress-team44.pdf)
- [Research journey and decisions](docs/PROGRESS.md)
- [Detailed reproduction commands](docs/REPRODUCIBILITY.md)
- [Experiment evidence index](reports/README.md)

The final report contains member names/IDs, the team name, AI declaration and
workload statement. Responsibilities and the final submission require team
approval. The Canvas report deadline is **6 November 2026, 23:59 Singapore time**;
the later Kaggle closing date is not the course-report deadline.

## Selected result

Eight equally weighted CatBoost components: normal history with local trends
and stale-history-augmented history, each with direct/residual targets and
seeds 42/2026. Every component has 1,600 trees, depth 8, learning rate 0.04 and
L2 regularisation 6. Training retains all labels. At inference both views use
all permitted training history. Inputs have 78/74 features.

| Local development window | Selected RMSE | Gain over four-component adaptive reference |
| --- | ---: | ---: |
| Jan–Mar 2025 | 440.763 | +4.756 |
| Apr 2024–Mar 2025 | 466.757 | +2.783 |
| Apr 2023–Mar 2024 | 590.347 | −9.811 |
| Dec 2023–Mar 2025 | 483.434 | +3.208 |

Recent MAE is 316.549. These are repeatedly inspected, overlapping development
windows, not independent final tests. Eight versus four components is not
equal total compute. The older-year regression remains unresolved.

Recorded public RMSE is **432.14881** for the mainline and **432.04286** for the
optional recent-window stacking probe. Stacking is not the default because
its local improvement is small and temporal transfer is inconsistent.
The two candidates share the original fitted components, not separate codebases.

Canonical text prevents aliases such as `3-room`/`3 room` from splitting groups.
Same recorded fields do not establish an individual flat, so duplicate-looking
transactions remain. Target histories exclude the whole query month and all
outer scoring labels. Static amenity snapshots and revised macro vintages remain
point-in-time limitations. No test-period public rental targets enter the
modeling, tuning or submission workflow.

## Folder map

```text
final_report/          Final PDF, LaTeX, bibliography, figures and numeric table fragments
progress_report/       Progress PDF and editable slide source/assets
docs/                  Concise cumulative progress + detailed reproduction guide
reports/               Small CSV/JSON experiment evidence and provenance (not trained models)
configs/               Selected recipe and audited MRT calendar
src/                   Production, evaluation and figure-generation Python
  research/            Executed optional studies retained to explain negative results
tests/                 Data contracts, label isolation, model/output safeguards and research tests
requirements.txt       Pinned main environment
requirements-research.txt  Optional CPU/LightGBM research environment
```

Private/local only, excluded from Git and the course ZIP: `data/`, `artifacts/`,
`submissions/*.csv`, virtual environments, temporary/build files and local AI
notes. Detailed retired notes and non-modeling prototypes are recoverable under
ignored `artifacts/local_archive/`; earlier shared versions remain in Git history.
No raw data, CBMs, prediction caches or target-visible lookup code is distributed.

## Quick start

Use Python 3.12. Install dependencies in a fresh environment, obtain the original
course files privately and restore the `data/raw/` layout documented in the guide.

```bash
python -m pip install -r requirements.txt
python src/check_setup.py
python -m pytest -q
```

The eight mandatory CSVs have fixed checksums. The optional example submission
is validated when present; otherwise the same ordered `Id,Predicted` layout is
constructed in memory, without fabricating a course data file.
The current suite has 142 passing tests; four optional LightGBM tests skip when
that research package is absent. New-process tests cover package and script
imports, including the repaired drawing-module entry point.

## Reproduce the model and evaluation

**Run training/regeneration in a separate checkout.** Default training can write
model/metadata/submission outputs; do not overwrite historical evidence to claim
a fresh run is the original one.

```bash
python src/run_experiments.py --compare-candidates --candidates hierarchical market_residual horizon_direct horizon_residual local_trend_direct local_trend_residual --windows short_recent long_recent long_older horizon16 --comparison-seeds 42 2026 --trial-iterations 1600 --task-type GPU --resume
python src/generate_figures.py --temporal-only --reference-candidate hierarchical_market_blend --bootstrap --error-groups
python src/train_final_model.py --task-type GPU
```

These are full-budget runs. An isolated execution smoke check is:

```bash
python src/train_final_model.py --task-type GPU --iterations 5 --model-prefix acceptance_smoke --submission submissions/acceptance_smoke.csv --metadata reports/acceptance_smoke_metadata.json
```

Smoke scores are not scientific results. Frozen evidence, raw row IDs and hashes
must match when reproducing the reported numbers. GPU refitting need not be
bit-identical. An independent same-device/new-environment full-budget repeat
already completed: 8 × 1,600 trees, 50,000 predictions, recent RMSE 440.6423;
it did not replace the frozen 440.7629 result. See
`reports/full_budget_reproduction.json`.

Other entry points:
`src/generate_eda.py` for EDA,
`src/run_experiments.py` for historical baselines,
`src/feedback_experiments.py` for controlled information/model screens,
`src/followup_analysis.py` for transfer/access checks, and
`python -m src.research.<module>` for the bounded research studies.
Their exact prerequisites, source windows, cache families and commands are in
[the guide](docs/REPRODUCIBILITY.md). Cached CPU PKL/NPY studies are not
interchangeable with temporal NPZ predictions. Load only trusted caches.

## Compile the paper

The existing figures/tables allow compilation without any private data/caches.

```powershell
./final_report/build.ps1
```

On macOS/Linux: `bash final_report/build.sh` (Tectonic or pdflatex/BibTeX).
Both output `final_report/project-report-team44.pdf`.
Report statistics can be regenerated from frozen sources with
`python final_report/prepare_report.py`; this needs course data and caches.
Figure 2 preserves the reviewed illustration with publication-size vector labels;
its optional rebuild additionally needs XeLaTeX and Times New Roman.
Other plots are code-generated from saved numerical evidence.
The progress report remains independently editable under `progress_report/`.

## Submission and continuation

Canvas accepts a ZIP containing the report and source, or a report with an
accessible repository link. A private link alone may be inaccessible to markers.
Use a source-only package: no environments, data or fitted models; aim below
10 MB. Only one authorised member submits for the team. Do not confuse a
Kaggle prediction CSV with the final course report/code submission.

Keep one selected recipe, the existing evidence ledgers and the same report.
New work should update this README and append a concise, evidence-linked entry
to `docs/PROGRESS.md`, not create stage folders, duplicated handoffs or claims of
improvement based solely on one small public-score change.
