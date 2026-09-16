# Prototype v1 Results

Run date: 17 September 2026

## Goal

Create the smallest defensible end-to-end experiment that uses a future-month validation design, compares transparent baselines with a task-suitable model, and produces a Kaggle-compatible file without auxiliary-data complexity.

## Validation design

- Training: January 2021 through December 2024 (141,076 rows).
- Validation: January through March 2025 (8,924 rows).
- Metric: RMSE, matching Kaggle.
- Random seed: 42.

## Results

| Method | Validation RMSE |
| --- | ---: |
| Global training median | 844.60 |
| Town + flat-type median with fallbacks | 546.81 |
| CatBoost prototype | 457.08 |

CatBoost improves RMSE by 16.4% relative to the grouped-median baseline. The model was still improving at iteration 700, so this is a starting point rather than a tuned final result.

## Prototype features

- Original non-target core attributes.
- Rent year, month, and month index.
- Lease age at the rental month.
- Combined block and street address category.
- Constant `FURNISHED` and `FEE` fields removed.

No auxiliary data, external data, historical target aggregates, or hidden-label reconstruction was used.

## Generated artifacts

- Submission: `submissions/prototype_catboost_v1.csv`
- Model: `artifacts/prototype_catboost_v1.cbm`
- Machine-readable metrics: `reports/prototype_metrics.json`

The submission contains 50,000 unique IDs in order, two required columns (`Id`, `Predicted`), no missing/non-finite values, and predictions ranging from approximately 1,681.72 to 5,383.97 SGD.

## Interpretation and next experiment

This result demonstrates that nonlinear interactions and direct categorical handling add substantial value over simple location/flat-type summaries. It does not establish that CatBoost is the final model. The next high-value experiment is to join the supplied HDB block coordinates and attributes, followed by separate MRT, mall, and school distance/count ablations. A lighter linear/tree baseline should also be retained for report interpretation.

The file has not been uploaded to Kaggle. Its public leaderboard score must be recorded only after an intentional team submission.
