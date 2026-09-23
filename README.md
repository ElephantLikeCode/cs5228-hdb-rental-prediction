# CS5228 HDB Rental Prediction

Private group-project repository for predicting Singapore HDB monthly rent.
The evaluation metric is RMSE, and chronological validation is the primary
local evaluation method.

## Current mainline and retained reference

The mainline is the **history-diversity CatBoost ensemble**: normal-history and
six-month stale-history training views, each with direct-rent and market-residual
targets and seeds 42/2026 (eight equally weighted components). At inference all
components receive the full permitted training history. It retains all labels.

The pipeline includes canonical category cleanup, core/spatial features,
strictly past-only comparable rents, adaptive market estimates, rolling
validation, baseline/feature ablations, grouped errors, EDA and a validated
50,000-row submission generator. Twenty-four automated tests cover data contracts,
feature/validation behavior and output protection.

### Model comparison — January-March 2025 validation

This is the single main results table. Update it when the selected pipeline
changes; retain previous iteration results in `docs/PROGRESS.md`.

| Model | Validation RMSE | Evaluation |
|---|---:|---|
| Global median | 844.60 | Historical reference |
| Town + flat-type median | 546.81 | Historical reference |
| HistGradientBoosting | 463.36 | Historical reference |
| CatBoost core | 455.71 | Historical reference |
| CatBoost + spatial and past-rent features (seed 42) | 450.04 | Historical reference |
| Retained reference: adaptive-market direct/residual ensemble (four components) | 445.52 | Fixed 1,600 trees/component |
| **Current mainline: normal/stale-history ensemble (eight components)** | **440.89** | **Fixed 1,600 trees/component; temporal limitations below** |
| Candidate: local-trend/stale-history ensemble (eight components) | 440.76 | Fixed 1,600 trees/component; awaiting platform test |

The final two rows use identical validation rows and seeds, with no scoring-label
early stopping. Historical baselines used a different early-stopping protocol
and do not measure this update's gain. The eight-component model costs more.

The mainline improves the recent three-month, recent 12-month and 16-month
windows by 4.63, 2.65 and 2.81 RMSE, but **worsens Apr 2023-Mar 2024 by 10.55**,
with all 12 months worsening and systematic overprediction. The windows overlap;
they are development evidence, not independent tests or a private-score guarantee.
Full results remain in `reports/temporal_summary.csv` and `docs/PROGRESS.md`.

The new candidate replaces only the normal-history half with four extra
town/flat-type momentum features. Relative to the current mainline it improves
the same four windows by 0.13, 0.13, 0.74 and 0.40 RMSE. Two additional
three-month cutoffs improve the trend head by 0.17 and 1.47 over its matched
reference. The older-year deficit is reduced but remains 9.81 RMSE, so the
candidate is generated for one platform test rather than treated as proven.

The browser-verified public score on 21 September 2026 is **432.47187, rank 3 of
16**, improved from 436.05684 by 3.58497. The uploaded filename was
`submission_history_diversity.csv`. Its exact bytes are now also the mainline
`submissions/final_submission.csv`; SHA-256:
`b464777c16870c3f1484dccf8a418465b239d6f22ded97965bb5553c81f113b7`.
Promotion copied the verified CSV and eight CBMs, without retraining.
The former four-component pipeline is retained locally as a reproducible reference,
not discarded. The Keep/Remove submissions also remain unchanged.

Public scores use approximately 50% of test rows; private evaluation uses the
other half, with no verified random/temporal split rule. Up to two final
submissions may be selected; the two slots do not require two separate codebases.
No selection has been changed by this agent. Exact submitted scores are in
`reports/kaggle_submissions.csv`.

The official Kaggle ZIP was checked on 18 September 2026; its eight CSVs match
the existing raw files byte-for-byte. Raw data is never modified.
Spatial amenities use the supplied snapshot (MRT status `open`), not historical
opening dates per lease. This prediction-time availability limitation must be
disclosed in the report; target-history leakage tests do not resolve it.

## Setup from a fresh clone

Ask the repository owner to add your GitHub account as a collaborator, then:

```bash
git clone https://github.com/ElephantLikeCode/cs5228-hdb-rental-prediction.git
cd cs5228-hdb-rental-prediction
```

Use Python 3.12. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\Activate.ps1'
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Course/Kaggle data is not stored on GitHub. Obtain the official files privately
and use this exact layout:

```text
data/raw/
|-- train.csv
|-- test.csv
|-- example-submission.csv
`-- auxiliary/
    |-- sg-coe-prices.csv
    |-- sg-hdb-block.csv
    |-- sg-mrt-stations.csv
    |-- sg-schools.csv
    |-- sg-shopping-malls.csv
    `-- sg-stock-prices.csv
```

Validate the environment, exact data release, and schemas:

```bash
python src/check_setup.py
```

Do not continue until it prints `SETUP CHECK PASSED`. Before receiving the
private data, use `python src/check_setup.py --skip-data` to check only Python
and packages.

## Reproduce the current result

Run from the repository root:

```bash
python src/data_audit.py
python -m pytest -q
python src/generate_eda.py
python src/run_experiments.py --compare-candidates --candidates hierarchical market_residual horizon_direct horizon_residual --windows short_recent long_recent long_older horizon16 --comparison-seeds 42 2026 --task-type GPU --resume
python src/generate_figures.py --temporal-only --reference-candidate hierarchical_market_blend --bootstrap --error-groups
python src/update_progress.py
python src/train_final_model.py --task-type GPU
```

`configs/model.json` selects the current submission recipe. Both candidate
evaluation and submission use `src/modeling.py`, so feature construction and
fixed tree counts match. Candidate evaluation never supplies scoring labels to
early stopping. The default comparison covers the latest three months and two
12-month windows; `--windows horizon16` adds the latest 16-month stress test.
These windows overlap and are development comparisons, not independent tests.
For a second-seed confirmation, add `--comparison-seeds 42 2026 --resume`.

| Candidate | What changes from the reference |
|---|---|
| `legacy` | All-time detailed comparable rents; reference used for Keep/Remove |
| `recent` | Adds trailing-24-month means, medians and counts |
| `hierarchical` | Adds 12-month half-life comparables, flat-type market normalization, address-to-street-to-town fallbacks and history age |
| `market_residual` | Uses hierarchical features and learns rent minus the estimated local market baseline |
| `legacy_depth6`, `hierarchical_depth6` | Same respective features with depth 6 instead of 8 |
| `weighted_direct`, `weighted_residual` | Hierarchical features; 18-month half-life training weights |
| `horizon_direct`, `horizon_residual` | Hierarchical features; add a six-month stale-history training view, preserving each observation's total weight |
| `relative_rent` | Learn a market-normalized residual; squared-scale weights preserve the rent-MSE objective |
| `local_trend_direct`, `local_trend_residual` | Add smoothed town/flat-type relative-rent changes between the latest and preceding six months |
| `composition_trend_direct`, `composition_trend_residual` | Test the same local trend after shrinking area/lease-age composition cells; retained as negative evidence |

The retained `hierarchical_market_blend` reference averages hierarchical and
market-residual predictions 50:50, then averages seeds. It adds 18 adaptive
features to the existing 56; category cleanup and raw data remain unchanged.

The selected `history_diversity_blend` in `configs/model.json` averages that
reference and the stale-history ensemble 50:50 (eight equal-weight components).
Its weights were fixed after initial screening, then checked with the second
seed and two additional windows; they were not optimized over a weight grid.
At inference every component receives all available permitted training history;
the latest six months are hidden only in the extra training view.
Promotion followed the public result and review of the old-year regression;
that limitation remains open. Recent-sample weighting and the relative-rent
learner failed their stability screens and are not part of the mainline.

Adaptive market forecasts use only prior months, cap monthly log-growth at
plus/minus 1%, and damp the extrapolation horizon toward six months. This is a
bounded hypothesis, not an assumption that rents must increase. Training-label
removal is not part of these temporal comparisons; both previously submitted
Keep/Remove files are retained locally.

Every comparison is appended to `reports/temporal_comparison.csv`, including
RMSE, MSE, MAE, bias, diagnostic rent slices, timing, exact configuration, and
data/code hashes. Local paired predictions live under ignored
`artifacts/temporal_comparison/`. Resume skips only identical recorded runs.
`python src/generate_figures.py --temporal-only --reference-candidate hierarchical_market_blend --bootstrap --error-groups` reconstructs matched-seed
ensemble comparisons, including the fixed 50:50 direct/residual combination,
in `reports/temporal_summary.csv`. Complete-window RMSE is the selection metric;
the label-defined low-rent slices are diagnostics only.
The reference option explicitly compares candidates with the submitted
four-component pipeline. Frozen reference predictions can come from an earlier
source revision; matching row IDs, labels, training cutoff and data hashes are
checked, and both source hashes are recorded. `--error-groups` writes regenerable
month/town/flat-type/area diagnostics locally to
`artifacts/temporal_comparison/error_groups.csv`, not another GitHub report.
The optional 1,000-resample address-cluster bootstrap measures paired uncertainty
within these development samples; it does not establish future-market accuracy.
`--trial-iterations 5 --candidates legacy --windows short_recent` is a quick
GPU smoke run; its scores are not suitable for model selection.

Generate the current local-trend candidate without changing the mainline:

```bash
python src/run_experiments.py --compare-candidates --candidates local_trend_direct local_trend_residual --windows short_recent long_recent long_older horizon16 --comparison-seeds 42 2026 --trial-iterations 1600 --task-type GPU --resume
python src/generate_figures.py --temporal-only --reference-candidate hierarchical_market_blend --bootstrap --error-groups
python src/train_final_model.py --task-type GPU --history-views local_trend:0 hierarchical:6 --submission submissions/submission_local_trend_diversity.csv --model-prefix candidates/local_trend_diversity/model --metadata artifacts/candidates/local_trend_diversity/metadata.json
```

This uses local-trend features only for the normal-history view and the proven
hierarchical recipe for the six-month stale-history view. Mixed views are
recorded explicitly in metadata; experimental output paths remain isolated.
Each accepted recipe has a distinct model suffix, and residual/normalized
targets are rejected unless every view supplies the required adaptive baseline.

The original classroom baseline/ablation study remains reproducible with
`python src/run_experiments.py --task-type GPU` and its `--long-horizon` option.
Those commands refresh the historical reference tables and use early stopping;
keep their scores separate from the fixed-budget comparison.

For the current low-rent-label sensitivity test, generate two otherwise
identical submissions. The second command removes only 3-room-or-larger labels
below 50% of the same-month, same-town, same-flat-type median and recomputes all
past-rent features from the retained training rows. It never removes or edits
competition test rows.

```bash
python src/train_final_model.py --candidate legacy --task-type GPU --outlier-policy keep --submission submissions/submission_keep_all_labels.csv --model-prefix candidate_keep_all_labels --metadata artifacts/candidate_keep_all_labels_metadata.json
python src/train_final_model.py --candidate legacy --task-type GPU --outlier-policy remove --outlier-ratio 0.5 --submission submissions/submission_remove_suspected_low_rents.csv --model-prefix candidate_remove_suspected_low_rents --metadata artifacts/candidate_remove_suspected_low_rents_metadata.json
```

Expected results:

- `24 passed` from pytest;
- candidate scores and adoption evidence in `docs/PROGRESS.md`;
- `submissions/final_submission.csv` with columns `Id,Predicted` and
  exactly 50,000 rows;
- eight loadable models under `artifacts/final_model*.cbm`: two seeds,
  direct/residual targets and normal/`_lag6` training views;
- `reports/final_model_metadata.json` with component modes, weights, timings,
  model/code/data hashes, and the submission hash.

When loading a residual model, use `predict_model` from `src/modeling.py` with
its metadata component's `residual_target` flag: a raw CBM prediction is a rent
difference and must have the market baseline added back. The training command
automatically reloads every saved model and checks 100 predictions before
writing and round-trip-checking the submission CSV.

The mainline command is `python src/train_final_model.py --task-type GPU`.
An experimental override (candidate/config, seeds, trees, outlier settings or
history views) must specify separate `--submission`, `--metadata` and
`--model-prefix` paths; otherwise the command fails before training.
Output paths are restricted to their project output directories.

Before replacing an existing complete run, the training command verifies and
snapshots its CSV, metadata and every model into ignored
`artifacts/submitted/<submission-sha256>/`, mirroring repository-relative paths.
An incomplete or hash-mismatched previous run is rejected. A failed later fit
can still leave partial working outputs; the verified snapshot is the recovery
source, not an atomic deployment guarantee.

The retained 436.05684 reference is under:
`artifacts/submitted/cb6ec7a6d60cf2e491ff044faa202ccf1873adb962a8ef810ccf437f4c85b2d3/`.
Resolve its metadata's model/submission paths relative to that snapshot directory.
Its original training recipe is preserved in that metadata. The exact uploaded
432.47187 candidate also remains locally under
`artifacts/candidates/history_diversity/` and its original submission filename.
This retention is for reproducibility and rollback, not an additional active branch.

Use `--task-type CPU` without NVIDIA hardware. GPU and CPU CatBoost are not
bit-identical; even fixed-seed GPU reruns can differ. The promoted artifact is
byte-exact, while retraining reproduces the recipe rather than a guaranteed hash.
The existing eight-component run took 709.32 seconds on the RTX 4060; the retained
four-component run took 336.92 seconds. No new fit was run during promotion.
Private data, models, figures and submission CSVs remain ignored by Git.

## Repository structure

```text
docs/PROJECT_PLAN.md       Project scope, deadlines, risks, and next work
docs/PROGRESS.md           Single cumulative record of current progress
docs/KAGGLE_AUDIT.md       Official data/rules audit and planning implications
src/check_setup.py         Read-only environment and private-data validation
src/data_audit.py          Reproducible data audit
src/features.py            Leakage-safe core, spatial, and past-rent features
src/modeling.py            Shared feature recipes, fixed-budget fitting and prediction
configs/model.json         Current submission recipe and hyperparameters
src/validation.py          Rolling time folds and regression/error metrics
src/generate_eda.py        Reproducible EDA figures and summaries
src/run_experiments.py     Baselines, rolling validation, and feature ablations
src/generate_figures.py    Report-ready experiment and error plots
src/update_progress.py     Refresh current results in the cumulative progress log
src/train_final_model.py   Train the selected model and validate its submission
reports/temporal_comparison.csv   Current fixed-budget per-fit evidence
reports/temporal_summary.csv      Matched-seed and fixed-blend comparisons
reports/final_model_metadata.json Exact local model and submission provenance
reports/kaggle_submissions.csv    Only actually submitted public scores
tests/                     Automated data, feature, and validation tests
requirements.txt           Exact verified direct dependencies
```

Other experiment/error tables in `reports/` retain the historical baseline,
ablation and rejected-candidate evidence; they do not select the current model.
Use `configs/model.json` and the latest cumulative entry in `docs/PROGRESS.md`
for the current recipe. Detailed paired predictions, CBMs and plot images stay
local under ignored paths rather than accumulating in the GitHub repository.

## Team workflow

Do not work directly on `main`. Start each task from the latest main branch:

```bash
git switch main
git pull --ff-only
git switch -c feature/short-description
```

Before opening a pull request, run `python src/check_setup.py` and
`python -m pytest -q`. Keep reusable logic in `src/`, use notebooks only for
exploration, and never commit course data, templates, credentials, trained
models, or generated submissions.

## Course deliverables and next work

Canvas rechecked on 21 September 2026:

- Methodological quality 60%, report quality 30%, reproducibility 10%.
- Progress report: approximately 10-15 slides as PDF; final 1-2 slides describe
  next steps. The page says Sunday 4 October (not a hard deadline).
- Final report: at most eight main-text pages, excluding references/appendix,
  plus code or a GitHub link and member workload breakdown.
- The final deadline remains the placeholder `Nov XX, 11.59 pm`; do not treat the
  former local 6 November date, or the Kaggle closing date, as a confirmed course deadline.
- Source: https://canvas.nus.edu.sg/courses/93866/pages/project

No further experiments are scheduled by this closeout. When the team resumes:

1. Reproduce setup/tests and agree on member ownership.
2. Investigate the old-window regression with more chronological cutoffs and
   monthly/town/flat-type errors before tuning against public scores.
3. Maintain one mainline and, only when justified, a complementary final candidate;
   preserve both outlier-policy submissions as historical evidence.
4. Prepare the progress report from existing EDA, baselines, ablations and failures.
5. Verify the final deadline and freeze/reproduce the final pipeline before delivery.

Keep one cumulative `docs/PROGRESS.md`, one main results table here, and the
existing evidence tables. See `docs/PROJECT_PLAN.md` for scope and responsibilities.
