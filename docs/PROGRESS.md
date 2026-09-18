# Project Progress

This is the single cumulative technical progress record for the project. New
work updates the current validated pipeline in this file; unsuccessful trials
remain in machine-readable experiment records rather than creating a new
versioned summary document.

## Initial reproducible foundation (18 September 2026)

Established the private Git repository, environment/data checks, a chronological
holdout, global and town/flat-type median baselines, an initial CatBoost reference, and a
validated 50,000-row submission generator. The initial CatBoost reference
achieved approximately 457.08 RMSE on the January-March 2025 holdout.

Key lesson: random splitting is not the primary evidence because the test period
is mostly later than training. Subsequent model selection uses chronological
rolling validation.

<!-- CURRENT_RESULTS_START -->
## Current validated pipeline (18 September 2026)

Three expanding chronological folds are used, with every validation month
strictly later than its training months. Linear Regression, Ridge, Decision
Tree, Random Forest, HistGradientBoosting, and CatBoost share a common latest
comparison window. Historical rent statistics use only strictly earlier months;
spatial features never use the target.

Best latest-window candidate: **catboost_seed_ensemble with all_spatial_history**,
RMSE **449.20 SGD**, MAE **319.29 SGD**, signed bias
**18.57 SGD**.

### Latest-window comparison (January-March 2025)

| model | feature_set | rmse | mae | r2 | bias | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| catboost_seed_ensemble | all_spatial_history | 449.20 | 319.29 | 0.52 | 18.57 | 957.19 |
| catboost | all_spatial_history | 450.04 | 320.03 | 0.52 | 18.69 | 441.37 |
| catboost | all_spatial | 452.62 | 334.22 | 0.51 | -44.98 | 182.97 |
| catboost | block_mrt | 452.89 | 334.84 | 0.51 | -48.09 | 174.78 |
| catboost | block_school | 453.71 | 335.49 | 0.51 | -48.81 | 192.83 |
| catboost | block_mall | 453.97 | 335.41 | 0.51 | -46.89 | 178.38 |
| catboost | block | 454.19 | 335.86 | 0.51 | -48.44 | 165.54 |
| catboost | core | 455.71 | 337.09 | 0.50 | -45.78 | 130.82 |
| hist_gradient_boosting | core | 463.36 | 336.75 | 0.49 | -8.27 | 1.38 |
| random_forest | core | 470.19 | 348.52 | 0.47 | -50.59 | 4.95 |
| decision_tree | core | 485.90 | 358.59 | 0.44 | -25.19 | 0.79 |
| ridge | core | 520.01 | 366.58 | 0.35 | 242.14 | 1.14 |
| linear_regression | core | 523.20 | 373.09 | 0.35 | 243.67 | 4.30 |
| town_flat_type_median | raw | 546.81 | 425.11 | 0.29 | -199.79 | 0.12 |
| global_median | raw | 844.60 | 691.33 | -0.70 | -542.46 | 0.00 |

### Rolling CatBoost stability

| pipeline | runs | mean_rmse | std_rmse | mean_mae | mean_bias |
| --- | --- | --- | --- | --- | --- |
| all_spatial_history_2seed_ensemble | 3 | 455.70 | 15.50 | 328.06 | 0.15 |
| all_spatial_history_seed42 | 3 | 456.44 | 15.37 | 328.28 | 3.54 |
| all_spatial | 2 | 460.41 | 22.35 | 337.36 | -32.36 |
| core | 2 | 462.89 | 20.25 | 340.99 | -37.13 |

### Error by actual rent band

| group_value | rows | rmse | mae | bias |
| --- | --- | --- | --- | --- |
| <=2500 | 1458 | 822.01 | 656.98 | 629.55 |
| 2501-3000 | 2525 | 299.26 | 226.18 | 70.29 |
| 3001-3500 | 2823 | 263.01 | 206.16 | -96.72 |
| 3501-4000 | 1541 | 356.64 | 295.86 | -246.27 |
| >4000 | 577 | 585.91 | 489.55 | -480.25 |

### Seen versus unseen addresses

| group_value | rows | rmse | mae | bias |
| --- | --- | --- | --- | --- |
| seen | 8901 | 448.75 | 318.87 | 19.23 |
| unseen | 23 | 596.64 | 482.98 | -235.68 |

### Findings

- Past-only address, town/flat-type, town, flat-type, and flat-model rent
  statistics improve the selected pipeline without using same-month targets.
- Equal-weight averaging of seeds 42 and 2026 improves the retained single
  CatBoost on all three short folds and both 12-month robustness folds.
- The selected ensemble reaches a three-fold mean RMSE of 455.70 versus 456.44
  for the seed-42 history model; the gain is small but consistent.
- Error is concentrated at the lowest and highest rent bands. The unseen-address
  slice contains only 23 rows, so its estimate is informative but uncertain.
- The official Kaggle ZIP matches the existing raw files byte-for-byte. Kaggle
  confirms RMSE, `Id,Predicted`, a roughly 50/50 public/private test split, and
  permits additional data collection.
- The current Kaggle score 438.94488 belongs to the earlier seed-42 submission;
  the new ensemble submission has not yet been uploaded.
<!-- CURRENT_RESULTS_END -->

## Robustness and first Kaggle result (18 September 2026)

The selected spatial/history pipeline was compared with the spatial-only model
on two consecutive 12-month validation windows. It improved RMSE from 595.21 to
580.75 on April 2023-March 2024 and from 473.72 to 472.67 on April 2024-March
2025, while reducing underprediction bias in both windows. Results are stored in
`reports/long_horizon_results.csv` and reproduced with
`python src/run_experiments.py --long-horizon`.

The first Kaggle submission scored **438.94488 public RMSE**, ranking 9 of 14 at
submission time. The public leaderboard covers approximately half of the test
set and is treated only as supporting evidence.

Three additional candidates were rejected on the January-March 2025 window:
lagged COE/stock indicators (452.68 RMSE), recent-12-month group statistics
(450.28), and a 2,200-tree lower-learning-rate configuration (450.59), versus
450.04 for the retained pipeline.

## Current next steps

1. Add errors by rent band, seen/unseen address, and sample density.
2. Investigate the remaining long-horizon underprediction in the earlier window.
3. Test residual blending only if a genuinely complementary model is identified.
4. Review robustness and leaderboard evidence with the team before another submission.
