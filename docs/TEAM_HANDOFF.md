# Team handoff: current state and next work

Last verified: 17 September 2026

## What is already working

- Python 3.12 environment specification and automated setup check.
- Immutable local layout for supplied train, test, sample-submission, and six
  auxiliary datasets.
- Reproducible data audit and Markdown/JSON audit outputs.
- Three automated data-contract tests.
- Chronological validation using January-March 2025 as the holdout window.
- Global-median and town/flat-type median baselines.
- CatBoost reference model using core, date, lease-age, and address features.
- Schema-validated generation of a 50,000-row Kaggle submission.

The latest verified local results are:

| Model | Chronological validation RMSE |
|---|---:|
| Global median | 844.60 |
| Town + flat-type median | 546.81 |
| CatBoost prototype | 457.08 |

The complete reference run uses seed 42 and 700 CatBoost iterations. On the
reference Windows machine it completes in approximately three minutes.

## How to reproduce the checkpoint

After completing `README.md` setup and passing `src/check_setup.py`:

```bash
python src/data_audit.py
python -m pytest -q
python src/train_prototype.py
```

Expected outcomes:

- all three tests pass;
- `reports/prototype_metrics.json` reports CatBoost RMSE near `457.08`;
- `submissions/prototype_catboost_v1.csv` has columns `Id,Predicted` and
  exactly 50,000 rows;
- `artifacts/prototype_catboost_v1.cbm` can be loaded by CatBoost.

Small floating-point or runtime differences across operating systems are
acceptable. Large metric differences require investigation before comparison.

## Files that are deliberately local only

The GitHub repository does not contain course/Kaggle data, supplied report
templates, trained models, generated submissions, credentials, or virtual
environments. Each member must obtain the same data from the official channel.
See `data/README.md` for the exact file names and layout.

## Immediate next work, in order

1. Give all members private-repository and Kaggle-team access; have each member
   reproduce the setup check, tests, and reference RMSE.
2. Submit the reviewed prototype once and record the public leaderboard score
   in `docs/EXPERIMENT_LOG.csv`.
3. Extract chronological splitting and common evaluation into reusable modules
   so every experiment uses the same protocol.
4. Implement spatial features from the supplied HDB, MRT, school, and shopping
   mall data. Evaluate each feature family with controlled ablations.
5. Add focused EDA figures for time drift, geography, flat attributes, target
   distribution, and residual/error slices.
6. Compare transparent and course-aligned alternatives against CatBoost; tune
   only after the feature and validation pipeline is stable.
7. Run robustness/error analysis, freeze a final model, and build the progress
   and final reports from logged evidence.

## Decisions that must remain consistent

- Chronological validation is primary; random splits are diagnostic only.
- Never use hidden/post-release target information or reconstruct test labels.
- Fit all preprocessing and target-derived statistics on training folds only.
- Add auxiliary data through documented, leakage-safe joins and ablations.
- Keep reusable logic in `src/`; notebooks are for exploration/presentation.
- Record all meaningful results, including failed experiments.

Open policy questions are tracked in `docs/OPEN_QUESTIONS.md`. Method and scope
decisions are tracked in `docs/DECISIONS.md`.

