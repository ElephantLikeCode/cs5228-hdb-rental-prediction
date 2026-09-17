# CS5228 HDB Rental Prediction

Private group-project repository for predicting Singapore HDB monthly rent.
The evaluation metric is RMSE, and chronological validation is the primary
local evaluation method.

## Current checkpoint

The repository currently provides:

- a reproducible data audit;
- three automated data-contract tests;
- a January-March 2025 chronological holdout;
- global-median and town/flat-type median baselines;
- a CatBoost reference model;
- generation and validation of a 50,000-row Kaggle submission.
- three-fold rolling chronological validation and subgroup error analysis;
- course-aligned Ridge, Decision Tree, and Random Forest comparisons;
- reproducible EDA figures;
- HDB block and nearest open-MRT, mall, and school spatial features.

Verified results with Python 3.12 and seed 42:

| Model | Validation RMSE |
|---|---:|
| Global median | 844.60 |
| Town + flat-type median | 546.81 |
| CatBoost prototype | 457.08 |
| CatBoost + spatial features (stage 2) | 453.10 |

Across three consecutive three-month validation windows, CatBoost improves from
mean RMSE 461.07 with core features to 458.58 with spatial features. The latest
window is January-March 2025; do not compare the three-fold mean directly with a
single-window score.

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
python src/train_prototype.py
python src/generate_eda.py
python src/run_stage2_experiments.py
python src/generate_stage2_figures.py
python src/train_spatial_v2.py
```

Expected results:

- `3 passed` from pytest;
- CatBoost validation RMSE near `457.08`;
- `submissions/prototype_catboost_v1.csv` with columns `Id,Predicted` and
  exactly 50,000 rows;
- a loadable model at `artifacts/prototype_catboost_v1.cbm`.

The full reference run takes about three minutes on the verified Windows
machine. Generated audits, models, figures, and submissions are local artifacts
and are ignored by Git.

## Repository structure

```text
docs/PROJECT_PLAN.md       Project scope, phases, deadlines, and next work
src/check_setup.py         Read-only environment/data validation
src/data_audit.py          Reproducible data audit
src/train_prototype.py     Baselines, CatBoost model, and submission creation
src/project_features.py    Shared leakage-safe core and spatial features
src/validation.py          Rolling time folds and regression/error metrics
src/generate_eda.py        Reproducible EDA figures and summaries
src/run_stage2_experiments.py  Course baselines and spatial ablations
src/generate_stage2_figures.py Report-ready experiment and error plots
src/train_spatial_v2.py    Train the validated v2 model and submission
tests/                     Automated data-contract tests
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

1. Give every member repository and Kaggle-team access and reproduce RMSE 457.08.
2. Review and submit the spatial v2 candidate once, then record its public score.
3. Separate block attributes from amenity-distance features in the next ablation.
4. Add leakage-safe historical/location statistics and explicit time-trend tests.
5. Investigate high-error towns and Executive flats with residual diagnostics.
6. Run bounded tuning only after the next feature ablations are stable.
7. Freeze the final pipeline, reproduce it cleanly, and prepare the reports.

The detailed schedule, risk controls, role suggestions, and methodological plan
are in `docs/PROJECT_PLAN.md`.
