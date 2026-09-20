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
## Reference evaluation (historical early-stopping protocol)

Three expanding chronological folds are used, with every validation month
strictly later than its training months. Linear Regression, Ridge, Decision
Tree, Random Forest, HistGradientBoosting, and CatBoost share a common latest
comparison window. Historical comparable-rent statistics use only strictly
earlier months; spatial features never use the target.

Best latest-window candidate: **catboost_seed_ensemble with all_spatial_detailed_history**,
RMSE **448.06 SGD**, MAE **318.82 SGD**, signed bias
**21.15 SGD**.

### Latest-window comparison (January-March 2025)

| model | feature_set | rmse | mae | r2 | bias | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| catboost_seed_ensemble | all_spatial_detailed_history | 448.06 | 318.82 | 0.52 | 21.15 | 97.02 |
| catboost | all_spatial_detailed_history | 448.16 | 318.93 | 0.52 | 21.74 | 55.83 |
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
| detailed_comparables_2seed_ensemble | 3 | 454.31 | 15.54 | 326.55 | 7.00 |
| detailed_comparables_seed42 | 3 | 454.67 | 15.92 | 326.75 | 8.83 |
| all_spatial | 2 | 460.41 | 22.35 | 337.36 | -32.36 |
| core | 2 | 462.89 | 20.25 | 340.99 | -37.13 |

### Error by actual rent band

| group_value | rows | rmse | mae | bias |
| --- | --- | --- | --- | --- |
| 2501-3000 | 2525 | 308.93 | 234.08 | 65.37 |
| 3001-3500 | 2823 | 269.10 | 210.20 | -94.35 |
| 3501-4000 | 1541 | 350.42 | 289.46 | -228.16 |
| <=2500 | 1458 | 817.88 | 649.67 | 618.13 |
| >4000 | 577 | 561.01 | 463.44 | -449.86 |

### Seen versus unseen addresses

| group_value | rows | rmse | mae | bias |
| --- | --- | --- | --- | --- |
| seen | 8901 | 447.55 | 318.31 | 21.84 |
| unseen | 23 | 613.71 | 515.17 | -244.96 |

### Twelve-month robustness

| model | feature_set | fold | rmse | mae | bias |
| --- | --- | --- | --- | --- | --- |
| catboost | all_spatial_detailed_history | 2023-04_to_2024-03 | 578.71 | 459.44 | -31.60 |
| catboost_seed_ensemble | all_spatial_detailed_history | 2023-04_to_2024-03 | 578.30 | 459.01 | -30.99 |
| catboost | all_spatial_detailed_history | 2024-04_to_2025-03 | 470.85 | 343.01 | 2.04 |
| catboost_seed_ensemble | all_spatial_detailed_history | 2024-04_to_2025-03 | 470.41 | 342.84 | 0.06 |

### Recorded public submissions

| submitted_at | submission_file | public_rmse |
| --- | --- | --- |
| 2026-09-18 | final_submission.csv | 438.94 |
| 2026-09-21 | submission_keep_all_labels.csv | 436.76 |
| 2026-09-21 | submission_remove_suspected_low_rents.csv | 436.59 |

These reference metrics use validation-driven early stopping. Fixed-budget
comparisons and adoption decisions are recorded in the cumulative entries below
and in `reports/temporal_comparison.csv`. Do not mix the two protocols.
<!-- CURRENT_RESULTS_END -->

## Recorded findings before fixed-budget evaluation

- Category strings are canonicalized before modeling, including `3 room` versus
  `3-room`, which were year-dependent aliases in the raw data.
- Past-only comparables now separate address by flat type and floor area, and
  add street/flat-type and town/flat-type/model groups. They improve every
  three-month, 12-month, and 16-month window tested.
- The selected GPU ensemble reaches latest-window RMSE 448.06 and three-fold
  mean 454.31, versus 449.20 and 455.70 for the previous CPU ensemble.
- Unconstrained recent/market-trend features reached 445.49 on the latest
  window but were rejected because they degraded some long-horizon windows.
- XGBoost reached 470.13 and worsened every tested blend; CatBoost remains the
  retained model because of validation evidence, not by assumption.
- Error is concentrated at the lowest and highest rent bands. The unseen-address
  slice contains only 23 rows, so its estimate is informative but uncertain.
- The official Kaggle ZIP matches the existing raw files byte-for-byte. Kaggle
  confirms RMSE, `Id,Predicted`, a roughly 50/50 public/private test split, and
  permits additional data collection.
- The earlier seed-42 submission scored 438.94488. The detailed-comparable
  two-seed GPU ensemble now scores 436.75718 when retaining all labels.
- A fixed low-rent sensitivity rule flags 954 of 150,000 training labels
  (0.636%): only 3-room-or-larger or executive flats below 50% of the median
  for the same month, town, and canonical flat type. Raw files and all 50,000
  competition test rows remain unchanged.
- On the latest chronological window, excluding the 73 similarly flagged
  validation labels changes single-seed RMSE from 448.16 to 417.78. However,
  removing flagged training labels and recomputing histories gives 418.98 on
  that cleaned validation slice and 451.38 on the full slice. This supports a
  platform A/B sensitivity test, but does not prove that label removal is the
  better general pipeline.
- Two equal-configuration GPU candidates are ready locally: one keeps all
  labels and one removes the 954 flagged training labels. Their 50,000 test
  predictions correlate at 0.9961; the removal candidate is S$7.66 higher on
  average. Their public Kaggle RMSE values are 436.75718 (retain) and 436.58736
  (remove), so removal wins by 0.16982 on the public half. This is weak positive
  evidence for the low-anomaly hypothesis, not proof that the private half has
  the same composition or that all flagged labels are erroneous.


## Prior robustness and first Kaggle result (18 September 2026)

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

## Next steps recorded before the temporal comparison

1. Have each member reproduce the ten tests and the selected GPU result.
2. Keep the fixed-rule removal candidate as the current public-leaderboard best,
   while testing a softer robust-loss or label-weighting treatment that does not
   assume every flagged record is erroneous.
3. Revisit market trend only with bounded, horizon-aware extrapolation.
4. Test optional economic features through isolated chronological ablations.
5. Freeze the strongest stable pipeline before preparing the final report.

## Adaptive market comparables and complementary predictions (21 September 2026)

### Decision and implementation

Adopt `hierarchical_market_blend` as the current configuration, retaining all
training labels. Preserve the existing Keep/Remove submissions and their CBM
files unchanged; their public scores are 436.75718 and 436.58736 respectively.
The new final submission has no public score yet. Raw files and all 50,000 test
rows are unchanged, and target-defined validation slices are never used to
remove rows from the primary metric.

The new features normalize past rent by its own month and canonical flat type,
then estimate local relative premiums with 12-month half-life weights and
address/flat/area -> address/flat -> street/flat -> town/flat fallbacks. A bounded
market forecast caps monthly log-growth at +/-1% and damps long-horizon growth.
Only strictly earlier reference months contribute. Eighteen adaptive features
augment the existing 56 spatial/core/history features.

Two complementary prediction modes use these same 74 features: direct rent,
and rent minus the estimated local market level (added back at prediction).
Each uses seeds 42 and 2026, depth 8, 1,600 trees, learning rate 0.04 and L2 6.
The four predictions receive equal weights; blend weights were not optimized
against the validation targets. The additional components increase compute.

Shared logic is in `src/modeling.py`; `configs/model.json` selects the one
current recipe. Both validation and final training call the same feature and
model functions. No stage directories or per-round summary documents were added.

### Validation and actual results

Ran 31 full GPU validation fits on the RTX 4060: six candidate recipes were
screened with seed 42, then the reference and selected direct/residual recipes
were confirmed with seed 2026 across four windows. Model fitting plus prediction
took 2,088 seconds in aggregate, excluding feature construction and reporting.
Every scoring window is later than its training cutoff. The scoring labels are
not passed to CatBoost early stopping; all fits use the deployment budget of
1,600 trees. Consequently the historical 448.06 early-stopped result is not the
matched reference for this round.

| Validation period | Rows | Matched previous RMSE | New RMSE | RMSE gain | MSE reduction | Months improved |
|---|---:|---:|---:|---:|---:|---:|
| Jan-Mar 2025 | 8,924 | 447.5714 | 445.5185 | 2.0529 | 0.915% | 3/3 |
| Apr 2024-Mar 2025 | 35,158 | 471.1416 | 469.5399 | 1.6017 | 0.679% | 8/12 |
| Apr 2023-Mar 2024 | 35,882 | 585.5931 | 580.5366 | 5.0565 | 1.720% | 12/12 |
| Dec 2023-Mar 2025 | 47,171 | 491.6239 | 486.6420 | 4.9818 | 2.016% | 16/16 |

Paired address-cluster bootstraps (1,000 resamples) give 95% RMSE-gain intervals
of [0.46, 3.46], [0.63, 2.67], [3.82, 6.32], and [4.05, 5.76], respectively.
These quantify development-sample uncertainty, not future market uncertainty;
the windows overlap and candidate selection used these windows. A second seed
checks training variation, not a new independent holdout.

Rejected alternatives remain as evidence in `reports/temporal_comparison.csv`:
the trailing-24-month feature recipe worsened the older 12-month window by
4.28 RMSE (seed 42), despite a small recent-window gain. Depth-6 models did not
beat the selected combination on the screened short and 16-month windows.
The ledger records configuration, dates, metrics, runtime and data/code hashes;
`reports/temporal_summary.csv` contains matched-seed ensemble results, and paired
row predictions remain in ignored `artifacts/temporal_comparison/`.

Limitations: latest-window MAE worsens 318.67 -> 319.29, recent-12-month MAE
worsens 343.39 -> 347.69, and four of that window's months regress (worst monthly
RMSE change -2.59). That window also has a -34.95 mean prediction bias. The
decision follows the competition's full-window RMSE objective, not a claim
that every error measure improves. No new external dataset was introduced.

### Reproduction and next check

```bash
python src/check_setup.py
python -m pytest -q
python src/run_experiments.py --compare-candidates --candidates legacy hierarchical market_residual --windows short_recent long_recent long_older horizon16 --comparison-seeds 42 2026 --task-type GPU --resume
python src/generate_figures.py --temporal-only --bootstrap
python src/train_final_model.py --task-type GPU
```

Fourteen tests pass, including flat-type normalization, detailed comparable
separation, current/future-label exclusion, hierarchical fallbacks, long-horizon
history availability and saved direct/residual prediction round trips. Setup
checks confirm all nine required data files and their original checksums.
GPU reruns can differ slightly despite fixed seeds.

The full-data GPU training and submission round trip completed successfully in
336.92 seconds (150,000 training rows; four 1,600-tree components). The resulting
50,000-row CSV has ordered `Id,Predicted` columns and finite positive predictions
from 1493.17 to 5428.01. Each saved model passed a 100-row reload check during
generation; an independent 128-row sample spanning the test-file order then
reproduced the four-model ensemble to within 9.1e-13 SGD. All four model hashes,
the raw-data checksums, and both preserved A/B submission hashes were verified.
The resume command reused all 24 matching confirmation runs successfully.
`update_progress.py` also preserves the manually appended history on refresh.

New final submission SHA-256:
`cb6ec7a6d60cf2e491ff044faa202ccf1873adb962a8ef810ccf437f4c85b2d3`.
Component hashes, flags and timings are in `reports/final_model_metadata.json`.

Next: submit the new `final_submission.csv` once and record its exact hash and
public score, comparing it with 436.58736 without assuming the local improvement
will transfer one-for-one. Preserve both anomaly-policy candidates. If the new
pipeline transfers, investigate the remaining long-window bias and property
segments; do not start by repeatedly tweaking leaderboard-facing weights.
