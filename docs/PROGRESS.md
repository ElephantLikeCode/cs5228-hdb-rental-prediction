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

Best latest-window candidate: **catboost with all_spatial_history**,
RMSE **450.04 SGD**, MAE **320.03 SGD**, signed bias
**18.69 SGD**.

### Latest-window comparison (January-March 2025)

| model | feature_set | rmse | mae | r2 | bias | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| catboost | all_spatial_history | 450.04 | 320.03 | 0.52 | 18.69 | 281.05 |
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

| feature_set | runs | mean_rmse | std_rmse | mean_mae | mean_bias |
| --- | --- | --- | --- | --- | --- |
| all_spatial_history | 3 | 456.44 | 15.37 | 328.28 | 3.54 |
| all_spatial | 3 | 457.81 | 16.43 | 336.31 | -36.56 |
| core | 3 | 460.49 | 14.91 | 339.69 | -40.01 |

### Highest-error towns

| group_value | rows | rmse | mae | bias |
| --- | --- | --- | --- | --- |
| jurong west | 620 | 551.27 | 384.48 | 72.29 |
| kallang/whampoa | 342 | 540.79 | 404.49 | 45.92 |
| bukit merah | 530 | 509.15 | 358.73 | -50.52 |
| bishan | 174 | 496.93 | 330.83 | 67.57 |
| toa payoh | 341 | 494.79 | 377.80 | 35.99 |

### Findings

- Past-only address, town/flat-type, town, flat-type, and flat-model rent
  statistics improve the selected pipeline without using same-month targets.
- Increasing the selected CatBoost limit from 1,200 to 1,600 trees improved the
  latest window further; two earlier folds stopped early.
- The selected history pipeline improves two of three folds and reduces the
  three-fold mean, although the middle fold is slightly worse.
- The official Kaggle ZIP matches the existing raw files byte-for-byte. Kaggle
  confirms RMSE, `Id,Predicted`, a roughly 50/50 public/private test split, and
  permits additional data collection.
- The test horizon extends well beyond training, so long-horizon trend robustness
  remains a higher priority than broad model search.
<!-- CURRENT_RESULTS_END -->

## Current next steps

1. Stress-test long-horizon time extrapolation beyond three-month validation windows.
2. Test alternative CatBoost learning-rate and regularization pairs around 1,600 trees.
3. Test COE and stock indicators in isolated chronological ablations.
4. Add errors by rent band, seen/unseen address, and sample density.
5. Review the candidate as a team before the first Kaggle submission.
