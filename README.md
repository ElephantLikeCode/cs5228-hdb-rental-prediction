# CS5228 HDB Rental Prediction

Private group-project repository for predicting Singapore HDB monthly rent.
The evaluation metric is RMSE, and chronological validation is the primary
local evaluation method.

## Current validated pipeline

The repository currently provides:

- a reproducible data audit;
- eighteen automated data-contract and feature/validation tests;
- a January-March 2025 chronological holdout;
- global-median and town/flat-type median baselines;
- Linear, Ridge, Decision Tree, Random Forest, HistGradientBoosting, and CatBoost comparisons;
- generation and validation of a 50,000-row Kaggle submission;
- three-fold rolling chronological validation and segmented error analysis;
- strict block/MRT/mall/school/all-spatial ablations with facility-radius counts;
- reproducible EDA figures;
- HDB block and nearest open-MRT, mall, and school spatial features;
- canonicalized categorical values, including year-dependent flat-type aliases;
- leakage-safe past-only comparable rents split by address, flat type, area,
  street, town, and flat model;
- a validated four-component CatBoost ensemble: direct rent and market-residual
  predictions, each using seeds 42 and 2026, with equal weights.

### Model comparison — January-March 2025 validation

This is the single main results table. Update it when the selected pipeline
changes; retain previous iteration results in `docs/PROGRESS.md`, not in another
competing README results table.

| Model | Validation RMSE | Evaluation |
|---|---:|---|
| Global median | 844.60 | Historical reference |
| Town + flat-type median | 546.81 | Historical reference |
| HistGradientBoosting | 463.36 | Historical reference |
| CatBoost core | 455.71 | Historical reference |
| CatBoost + spatial and past-rent features (seed 42) | 450.04 | Historical reference |
| **Current retained: adaptive-market direct/residual ensemble (four components)** | **445.52** | **Fixed 1,600 trees/component** |
| Experimental: normal/stale-history ensemble (eight components; not adopted) | 440.89 | Fixed 1,600 trees/component; stability limitation below |

The final two rows use identical validation rows and seeds. The experimental
combination improves this short window by **4.63 RMSE**, but worsens the older
12-month window by **10.55 RMSE**, with systematic overprediction. It is therefore
not the default model. Neither uses scoring labels for early stopping, and tree
budgets match deployment; the candidate uses twice as many components. Historical
baseline experiments allowed validation-driven early stopping where applicable
and are not used to calculate this update's improvement.

The current pipeline also improves both 12-month windows and the 16-month
stress window. Full multi-window results, previous iteration scores, and the
adoption evidence are in `docs/PROGRESS.md` and `reports/temporal_summary.csv`.
Those development windows overlap and do not guarantee a Kaggle improvement.
The new model keeps all labels. Latest-window MAE rises from 318.67 to 319.29,
and recent-12-month MAE rises from 343.39 to 347.69; four components also cost
more than the former two.

The current `final_submission.csv` scored **436.05684** on Kaggle, browser-verified
on 21 September 2026 at rank **7 of 16**. This improves the previous best
436.58736 by 0.53052. The preserved removal and keep-all-labels candidates scored
436.58736 and 436.75718 respectively. Public scores cover approximately half the
test set, so they support rather than replace chronological validation. Exact
submitted scores and the latest CSV hash are in `reports/kaggle_submissions.csv`.

For one optional platform test, use `submissions/submission_history_diversity.csv`.
It is an unsubmitted experimental candidate, not a replacement for
`submissions/final_submission.csv`. The four-component submitted model, both
Keep/Remove alternatives and their exact CSVs remain unchanged. Candidate CBMs
and metadata are isolated in ignored `artifacts/candidates/history_diversity/`.

The official Kaggle ZIP was downloaded and checked on 18 September 2026. All
eight CSVs match the existing raw files byte-for-byte. See
`docs/KAGGLE_AUDIT.md`; the private ZIP is kept under ignored `data/source/`.

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

The selected `hierarchical_market_blend` in `configs/model.json` averages the
hierarchical and market-residual predictions 50:50, then averages seeds. These
weights were fixed, not fitted to validation labels. It adds 18 adaptive features
to the existing 56, rather than replacing the raw data or categorical cleanup.

The experimental `history_diversity_blend` averages that existing ensemble and
the stale-history ensemble 50:50 (eight equal-weight components in total).
Its weights were fixed after initial screening, then checked with the second
seed and two additional windows; they were not optimized over a weight grid.
At inference every component receives all available permitted training history;
the latest six months are hidden only in the extra training view.
The old-year regression prevents automatic adoption. Recent-sample weighting
and the relative-rent learner also failed their stability screens.

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

- `18 passed` from pytest;
- candidate scores and adoption evidence in `docs/PROGRESS.md`;
- `submissions/final_submission.csv` with columns `Id,Predicted` and
  exactly 50,000 rows;
- four loadable models: `artifacts/final_model_seed_{42,2026}.cbm` and
  `artifacts/final_model_residual_seed_{42,2026}.cbm`;
- `reports/final_model_metadata.json` with component modes, weights, timings,
  model/code/data hashes, and the submission hash.

When loading a residual model, use `predict_model` from `src/modeling.py` with
its metadata component's `residual_target` flag: a raw CBM prediction is a rent
difference and must have the market baseline added back. The training command
automatically reloads every saved model and checks 100 predictions before
writing and round-trip-checking the submission CSV.

To reproduce the optional candidate without changing the default configuration
or its artifacts:

```bash
python src/train_final_model.py --task-type GPU --augmentation-lags 0 6 --submission submissions/submission_history_diversity.csv --model-prefix candidates/history_diversity/model --metadata artifacts/candidates/history_diversity/metadata.json
```

The candidate directory contains its eight model components, not old model
versions. Successful generation reloads each CBM and validates the 50,000-row
CSV. The main configuration and `final_submission.csv` still select the retained
four-component model.
The verified eight-component candidate run took 709.32 seconds (about 11.8
minutes) on the RTX 4060; its file hash is recorded in `docs/PROGRESS.md`.

Use `--task-type CPU` on machines without a compatible NVIDIA GPU. GPU and CPU
CatBoost are not bit-identical, and GPU reductions can differ between reruns even
with the same seed. Record task type, configuration and hashes with every result.
The verified four-component full-data GPU run took 336.92 seconds (about 5.6
minutes) on this RTX 4060 laptop, including feature construction and save checks.
Private data, model files, figures, and Kaggle submissions are ignored by Git.

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

## Next work

1. Give every member repository and Kaggle-team access and reproduce the fixed-budget comparison.
2. Preserve both outlier policies; use 436.05684 as the public reference, not as
   evidence that hidden labels have fewer anomalies.
3. Optionally test the history-diversity candidate once on Kaggle. Keep the current
   model unless further evidence justifies the market-regime tradeoff.
4. Test COE and stock features only through isolated chronological ablations.
5. Freeze the final pipeline, reproduce it cleanly, and prepare the reports.

The detailed schedule, risk controls, role suggestions, and methodological plan
are in `docs/PROJECT_PLAN.md`.
