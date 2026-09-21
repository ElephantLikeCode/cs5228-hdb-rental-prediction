# CS5228 HDB Rental Prediction

Private group-project repository for predicting Singapore HDB monthly rent.
The evaluation metric is RMSE, and chronological validation is the primary
local evaluation method.

## Current validated pipeline

The repository currently provides:

- a reproducible data audit;
- fourteen automated data-contract and feature/validation tests;
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
| Detailed-comparable GPU CatBoost ensemble (two seeds; re-evaluated reference) | 447.57 | Fixed 1,600 trees/component |
| **Current: adaptive-market direct/residual CatBoost ensemble (four components)** | **445.52** | **Fixed 1,600 trees/component** |

The final two rows are the matched comparison for this update: identical
validation rows and seeds, giving a **2.05 RMSE improvement**. Neither uses the
scoring labels for early stopping, so validation uses the same tree budget as
final training. Historical model results have not been rerun under this new
protocol; their original experiments allowed validation-driven early stopping
where applicable. Do not use them to calculate this update's improvement.

The current pipeline also improves both 12-month windows and the 16-month
stress window. Full multi-window results, previous iteration scores, and the
adoption evidence are in `docs/PROGRESS.md` and `reports/temporal_summary.csv`.
Those development windows overlap and do not guarantee a Kaggle improvement.
The new model keeps all labels. Latest-window MAE rises from 318.67 to 319.29,
and recent-12-month MAE rises from 343.39 to 347.69; four components also cost
more than the former two.

The new `final_submission.csv` has not been submitted. The best recorded public
score remains **436.58736**, from the preserved low-rent-removal candidate;
the preserved keep-all-labels candidate scored **436.75718**. These public
scores belong to the previous model, not the current pipeline. Their small
0.16982 difference is supporting evidence, not a replacement for chronological
validation. All submitted scores remain in `reports/kaggle_submissions.csv`.

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
python src/run_experiments.py --compare-candidates --candidates legacy hierarchical market_residual --windows short_recent long_recent long_older horizon16 --comparison-seeds 42 2026 --task-type GPU --resume
python src/generate_figures.py --temporal-only --bootstrap
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

The selected `hierarchical_market_blend` in `configs/model.json` averages the
hierarchical and market-residual predictions 50:50, then averages seeds. These
weights were fixed, not fitted to validation labels. It adds 18 adaptive features
to the existing 56, rather than replacing the raw data or categorical cleanup.

Adaptive market forecasts use only prior months, cap monthly log-growth at
plus/minus 1%, and damp the extrapolation horizon toward six months. This is a
bounded hypothesis, not an assumption that rents must increase. Training-label
removal is not part of these temporal comparisons; both previously submitted
Keep/Remove files are retained locally.

Every comparison is appended to `reports/temporal_comparison.csv`, including
RMSE, MSE, MAE, bias, diagnostic rent slices, timing, exact configuration, and
data/code hashes. Local paired predictions live under ignored
`artifacts/temporal_comparison/`. Resume skips only identical recorded runs.
`python src/generate_figures.py --temporal-only --bootstrap` reconstructs matched-seed
ensemble comparisons, including the fixed 50:50 direct/residual combination,
in `reports/temporal_summary.csv`. Complete-window RMSE is the selection metric;
the label-defined low-rent slices are diagnostics only.
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

- `14 passed` from pytest;
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
2. Preserve both outlier policies; use 436.58736 as the public reference, not as
   evidence that hidden labels have fewer anomalies.
3. Check the new temporal ensemble once on Kaggle, recording its exact CSV hash;
   do not tune repeatedly against public-leaderboard scores.
4. Test COE and stock features only through isolated chronological ablations.
5. Freeze the final pipeline, reproduce it cleanly, and prepare the reports.

The detailed schedule, risk controls, role suggestions, and methodological plan
are in `docs/PROJECT_PLAN.md`.
