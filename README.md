# CS5228 HDB Rental Prediction

Private group-project repository for predicting Singapore HDB monthly rent.
The evaluation metric is RMSE, and chronological validation is the primary
local evaluation method.

## Current validated pipeline

The repository currently provides:

- a reproducible data audit;
- eight automated data-contract and feature/validation tests;
- a January-March 2025 chronological holdout;
- global-median and town/flat-type median baselines;
- Linear, Ridge, Decision Tree, Random Forest, HistGradientBoosting, and CatBoost comparisons;
- generation and validation of a 50,000-row Kaggle submission;
- three-fold rolling chronological validation and subgroup error analysis;
- strict block/MRT/mall/school/all-spatial ablations with facility-radius counts;
- reproducible EDA figures;
- HDB block and nearest open-MRT, mall, and school spatial features.
- leakage-safe past-only rent statistics for address and property groups.

Verified results with Python 3.12 and seed 42:

| Model | Validation RMSE |
|---|---:|
| Global median | 844.60 |
| Town + flat-type median | 546.81 |
| HistGradientBoosting | 463.36 |
| CatBoost core | 455.71 |
| CatBoost + spatial and past-rent features | **450.04** |

Across three consecutive three-month validation windows, CatBoost improves from
mean RMSE 460.49 with core features to 456.44 with the selected features. The latest
window is January-March 2025; do not compare the three-fold mean directly with a
single-window score.

Two additional 12-month robustness windows also favor the selected history
pipeline. The first Kaggle submission scored `438.94488` on the public half of
the test set (rank 9 of 14 at submission time); this is supporting evidence,
not a replacement for chronological validation.

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
python src/run_experiments.py
python src/run_experiments.py --long-horizon
python src/generate_figures.py
python src/update_progress.py
python src/train_final_model.py
```

Expected results:

- `8 passed` from pytest;
- latest-window CatBoost validation RMSE near `450.04`;
- `submissions/final_submission.csv` with columns `Id,Predicted` and
  exactly 50,000 rows;
- a loadable model at `artifacts/final_model.cbm`.

The full experiment comparison takes roughly 35-45 minutes and final-model training
about five minutes on the verified Windows machine. Private data, model files,
figures, and Kaggle submissions are ignored by Git.

## Repository structure

```text
docs/PROJECT_PLAN.md       Project scope, deadlines, risks, and next work
docs/PROGRESS.md           Single cumulative record of current progress
docs/KAGGLE_AUDIT.md       Official data/rules audit and planning implications
src/check_setup.py         Read-only environment and private-data validation
src/data_audit.py          Reproducible data audit
src/features.py            Leakage-safe core, spatial, and past-rent features
src/validation.py          Rolling time folds and regression/error metrics
src/generate_eda.py        Reproducible EDA figures and summaries
src/run_experiments.py     Baselines, rolling validation, and feature ablations
src/generate_figures.py    Report-ready experiment and error plots
src/update_progress.py     Refresh current results in the cumulative progress log
src/train_final_model.py   Train the selected model and validate its submission
tests/                     Automated data, feature, and validation tests
requirements.txt           Exact verified direct dependencies
```

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

1. Give every member repository and Kaggle-team access and reproduce RMSE 450.04.
2. Review and submit the current candidate once, then record its public score.
3. Strengthen explicit long-horizon time-trend tests.
4. Investigate high-error towns and Executive flats with residual diagnostics.
5. Test alternative CatBoost learning-rate/regularization pairs around the selected budget.
6. Test COE and stock features only through isolated chronological ablations.
7. Freeze the final pipeline, reproduce it cleanly, and prepare the reports.

The detailed schedule, risk controls, role suggestions, and methodological plan
are in `docs/PROJECT_PLAN.md`.
