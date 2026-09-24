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
| 2026-09-21 | final_submission.csv | 436.06 |
| 2026-09-21 | submission_history_diversity.csv | 432.47 |

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

## Forecast-horizon training checks (21 September 2026)

Read-only browser verification confirms the adaptive-market submission scored
436.05684, rank 7 of 16: 0.53052 better than the removal candidate and 0.70034
better than the old keep-all-labels candidate. Sixth place is 433.60708 and first
place is 430.68326 at this observation. The submitted file matches the SHA-256
recorded above. No submissions or other website mutations were performed.

The current four-component predictions have latest-window bias +8.94 SGD, but
recent-12-month bias -34.95; that window's March 2025 bias reaches -80.48. The
same town does not have a stable correction across cutoffs (e.g. Jurong West
is +70.90 on the short window and -97.09 on the recent 12-month window). Therefore
do not fit a town/month correction on these scoring labels or add a global
constant chosen from the validation errors.

Two pre-specified hypotheses are being checked with unchanged scoring rows:
18-month half-life training weights, and six-month stale-history augmentation.
Augmentation retains true query dates but hides the latest six months of prior
labels; original/augmented observations share their original total weight.
Only rows with at least six remaining months of reference history are augmented.
Both direct and residual prediction modes use identical rules. No raw rows or
competition test rows are deleted, and the current submitted artifacts remain
unchanged until a candidate is accepted.

### Completed comparisons and decision

Completed 22 GPU validation fits: four recent-weighting fits, sixteen stale-history
fits (two prediction modes, two seeds, four windows), and two relative-rent fits.
Recorded model fitting/prediction time totals 1,637.16 seconds, excluding feature
construction. The extra relative-rent hypothesis divides residuals by the
past-only market level and multiplies loss weights by its square, preserving
the squared-error objective in SGD. It neither deletes labels nor changes RMSE
into a relative-error objective.

Recent weighting worsened matched seed-42 RMSE by 0.23 on the short window and
0.46 on the 16-month window. Relative-rent normalization improved the short
window by only 0.15 but worsened the 16-month window by 1.57. Neither is adopted.
Stale-history training alone improved the short window but degraded long
horizons. A fixed 50:50 blend with the original ensemble was then checked:

| Window | Submitted four-model RMSE | Experimental eight-model RMSE | RMSE gain | Months improved |
|---|---:|---:|---:|---:|
| Jan-Mar 2025 | 445.5185 | 440.8896 | +4.6288 | 3/3 |
| Apr 2024-Mar 2025 | 469.5399 | 466.8866 | +2.6533 | 11/12 |
| Apr 2023-Mar 2024 | 580.5366 | 591.0835 | -10.5469 | 0/12 |
| Dec 2023-Mar 2025 | 486.6420 | 483.8323 | +2.8097 | 15/16 |

All rows use seeds 42 and 2026, fixed 1,600-tree budgets per component and
identical scoring rows. This is not an equal-compute comparison: the candidate
doubles the component count. The blend was selected after the first-seed short
and 16-month screen; subsequent seeds are variation checks, not independent
test sets. Address-cluster bootstrap 95% gain intervals are [3.55, 5.69],
[2.03, 3.23], [-11.80, -9.33] and [2.22, 3.43], respectively. They describe
these development samples, not future-market uncertainty or selection bias.

Latest-window MAE improves 319.29 -> 316.69 and bias +8.94 -> +0.71. However,
older-year bias rises +8.89 -> +83.62 and every month in that window worsens.
Recent-12-month bias also worsens -34.95 -> -41.47 despite the lower overall
RMSE. These are systematic regime sensitivities, not justification for claiming
the method always improves or adding a validation-derived price correction.

Decision: keep `configs/model.json`, the submitted four CBMs and
`submissions/final_submission.csv` unchanged. Retain the new combination as an
explicit experimental candidate for one optional platform test, not as a proven
replacement. Its CSV is `submissions/submission_history_diversity.csv`; its eight
CBMs and metadata are isolated under ignored `artifacts/candidates/history_diversity/`.
The previous Keep/Remove files are also unchanged. No new source files, stage
directories or patch files were created for this round.

### Reproduction and checks

Use the existing comparison command with `--candidates horizon_direct horizon_residual`
and `--windows short_recent horizon16 long_recent long_older --comparison-seeds 42 2026`.
The two rejected additional probes use `--candidates weighted_direct weighted_residual relative_rent`
with `--windows short_recent horizon16 --comparison-seeds 42`. Both use
`--compare-candidates --task-type GPU --resume`.

`python src/generate_figures.py --temporal-only --reference-candidate hierarchical_market_blend --bootstrap --error-groups`
reconstructs paired comparisons. Reference predictions from the earlier submitted
revision are reused with explicit data/source hashes; raw row IDs, labels and
training cutoffs must match. The default features were also checked against the
submitted implementation across sampled months and matched exactly. Grouped
month/town/flat/area diagnostics are regenerable local evidence at
`artifacts/temporal_comparison/error_groups.csv`, not another GitHub report.

Eighteen tests pass, including hidden-history exclusion, preservation of each
augmented observation's total weight, residual/relative CBM prediction round trips,
eight-component aggregation, mismatched-row rejection and SGD-loss equivalence.
The optional candidate generation command is:

```bash
python src/train_final_model.py --task-type GPU --augmentation-lags 0 6 --submission submissions/submission_history_diversity.csv --model-prefix candidates/history_diversity/model --metadata artifacts/candidates/history_diversity/metadata.json
```

Candidate generation completed end-to-end on the RTX 4060 in 709.32 seconds.
The normal training view has 150,000 rows; the augmented view has 262,381 rows,
but both have total observation weight 150,000. At prediction time all eight
components receive the full permitted history; recent data is not discarded.
The eight component CBMs passed generation-time reload checks, and an independent
128-row prediction check reproduced the final ensemble within 4.55e-13 SGD.
The CSV has exactly 50,000 ordered `Id,Predicted` rows and finite positive values
between 1448.98 and 5487.47. Compared with the retained submission, the mean
prediction change is -2.92 SGD, with standard deviation 40.92; this is a scale
sanity check, not evidence of hidden-label accuracy.

Candidate SHA-256:
`b464777c16870c3f1484dccf8a418465b239d6f22ded97965bb5553c81f113b7`.
All eight candidate model hashes were verified. The original four submitted
model hashes, original final CSV, both Keep/Remove CSVs and the default config
remain unchanged. Setup checks confirm the original nine data checksums, and
18 tests pass. Automatic progress refresh also preserves the cumulative notes.

Next action: optionally submit only `submission_history_diversity.csv` once,
record the returned public score against this hash, and discuss the old-regime
tradeoff before promotion. Do not overwrite the default or claim leaderboard
improvement before that result. If it does not transfer, retain the 436.05684
pipeline and move to a different learner or local-comparable representation,
rather than repeatedly tuning this blend against the public leaderboard.

## Mainline promotion and maintenance closeout — 21 September 2026

### Decision and evidence

The user submitted `submission_history_diversity.csv`. Read-only browser
verification showed public RMSE **432.47187**, rank **3/16**, compared with
436.05684 for the retained reference: a gain of **3.58497**. This is new evidence
on the public half, not a guarantee for the private half. The April 2023-March
2024 validation regression (+10.55 RMSE; all 12 months worse) remains unresolved.
The local multi-window results above were not rerun or changed.

Following the user's closeout request, the history-diversity ensemble becomes
the one current mainline. `configs/model.json` selects both history views [0, 6],
both targets and seeds 42/2026. Eight already-verified CBMs and the exact uploaded
CSV were copied to the standard final paths, with no retraining or new experiment.
The final CSV hash is
`b464777c16870c3f1484dccf8a418465b239d6f22ded97965bb5553c81f113b7`.
Metadata retains original training hashes/timings and records promotion separately.

The previous 436.05684 run is fully backed up at
`artifacts/submitted/cb6ec7a6d60cf2e491ff044faa202ccf1873adb962a8ef810ccf437f4c85b2d3/`,
including its four models, exact CSV and metadata with repository-relative paths.
Use that snapshot directory as the root when reading those paths.
The original eight-model candidate and both Keep/Remove submissions remain
unchanged. No user files or raw data were deleted.

### Architecture and safeguards

Keep the existing ten Python modules and directories. Current recipes belong
in config/modeling; historical experiments remain evidence rather than parallel
active pipelines. No new Python files, stage folders or separate summary reports.
README keeps one primary results table; this file remains the cumulative log.

Experimental training overrides now require isolated submission, metadata and
model-prefix destinations. Output path checks exclude raw data and paths outside
the project output folders. Before overwriting a complete run, the generator
verifies and snapshots its CSV, metadata and every CBM. Incomplete/hash-mismatched
runs are refused. This enables recovery but is not an atomic multi-file deployment:
interrupted fitting may leave partial working outputs, so use the snapshot.

Maintenance verification covers 21 tests, setup checks of nine raw files, hash
and schema checks of mainline/reference artifacts, 50,000 ordered prediction rows,
and availability of all 53 uniquely keyed experiment prediction files. The three
added tests cover output isolation, backup integrity and eight-component config.
Data, CBMs, submission CSVs, generated figures and local AI notes remain Git-ignored.

### Course deliverables and deferred work

Canvas was checked read-only. Methodology contributes 60%, report quality 30%,
reproducibility 10%. The progress report is 10-15 PDF slides with final 1-2 slides
for next steps; the stated target is Sunday 4 October (not a hard deadline).
The final report has at most eight main-text pages plus references/appendix,
code or GitHub link and workload attribution. Its date still reads
`Nov XX, 11.59 pm`. Earlier local 8 October / 6 November entries are superseded.

Document the auxiliary snapshot limitation: MRT status is not filtered by its
opening date at each historical observation. Do not conflate past-only rent
history checks with a complete point-in-time auxiliary-data audit.

No further model training is part of this closeout. When work resumes, prioritize
the older-window stability investigation and progress-report evidence. One
complementary final submission can be retained alongside the mainline; two
selection slots do not require two codebases. Browser selections remain unchanged.

## Town/flat-type momentum experiment — 23 September 2026

### Hypothesis and implementation

The 432.47187 mainline improves recent periods but overpredicts the older
April 2023-March 2024 window. Error grouping showed that the regression grew
over time and varied materially by town. A new leakage-safe feature group tests
whether each town/flat-type segment's premium over its contemporaneous flat-type
market changed recently. It compares the latest six prior months with the
preceding six, shrinks both ratios toward one with strength 20, clips the log
change to plus/minus 0.15, and records the smaller period's support. Same-month
and future labels are excluded. Four features are added only to the normal
history view; the lag-six view keeps the submitted hierarchical recipe.

The existing XGBoost experiment was not repeated: it scored 470.13 on the
January-March 2025 window and every tested CatBoost/XGBoost blend was worse than
CatBoost alone. Cached component predictions were first used to test stale-view
weights. A 25% stale weight was more stable but weaker than the submitted 50%
weight on recent windows; it remains a possible conservative final-selection
alternative, not the performance candidate.

### Fixed-budget evidence

All comparisons use 1,600 trees, identical validation rows, seeds 42 and 2026
where available, and no validation-label early stopping.

| Window | Current mainline RMSE | Local-trend candidate RMSE | Gain |
|---|---:|---:|---:|
| Jan-Mar 2025 | 440.8896 | 440.7629 | +0.1268 |
| Apr 2024-Mar 2025 | 466.8866 | 466.7566 | +0.1300 |
| Apr 2023-Mar 2024 | 591.0835 | 590.3473 | +0.7361 |
| Dec 2023-Mar 2025 | 483.8323 | 483.4345 | +0.3979 |

Two extra single-seed cutoffs, introduced before the second-seed confirmation,
also favored the local-trend head: July-September 2024 improved by 0.17 RMSE and
September-November 2023 by 1.47. The feature head alone improves all six checked
windows after the two-seed results are used where available. The complete mixed
candidate improves all four core windows, but still loses 9.81 RMSE to the
four-component reference in the oldest 12-month regime. This is a reduction of
the known risk, not its resolution.

### Generated candidate and decision

The full eight-model GPU run completed in 669.44 seconds. It uses 78 features
for `local_trend:0` and 74 for `hierarchical:6`; each view preserves total
training weight 150,000. All CBMs passed save/reload checks. The submission has
50,000 ordered finite positive predictions from 1461.83 to 5437.83 SGD.

Candidate file: `submissions/submission_local_trend_diversity.csv`
SHA-256: `a88da3be518a4793486765f5d9ff44c9b27b3248bb195e6f2d0264ae7dabb57a`

Compared with the 432.47187 mainline, prediction changes average +2.66 SGD with
standard deviation 12.69 and correlation 0.99964. The change is bounded rather
than a broad market-level shift. Keep the submitted 432.47187 artifact as the
mainline until this candidate receives a platform score; one small public test
is justified by consistent local improvements. No Kaggle upload or final-slot
selection was performed automatically.

Twenty-two tests pass, including future/same-month exclusion for the new feature,
mixed-history output isolation and existing artifact safeguards. Setup checks
still verify all nine raw files. No new Python module, stage directory or report
file was created.

## Review fixes and composition-controlled trend — 23 September 2026

### Review corrections

A direct review of the mixed-history training entry point found two boundary
bugs that did not affect the generated local-trend candidate: some other accepted
recipe combinations could resolve to the same CBM filename, and residual mode
was checked against the top-level recipe rather than every history view. Model
suffixes are now recipe-specific where needed, and every residual or normalized
view must provide adaptive market features. Two tests cover these cases. The
existing candidate remains valid: all eight model paths are unique and their
recorded hashes match.

The earlier statement that the local-trend candidate consistently improves the
current mainline needs qualification. A 2,000-replicate address-cluster paired
bootstrap gives mainline-to-candidate gain intervals of -0.13 to 0.36 RMSE for
January-March 2025 and -0.04 to 0.30 for April 2024-March 2025. Those two small
gains are directionally useful but not statistically stable within these
development samples. The older-year and 16-month gains are stronger at 0.51 to
0.96 and 0.24 to 0.58 respectively. Overlapping windows remain non-independent.

### Composition-control experiment

The follow-up hypothesis was that changing area and lease-age composition within
a town/flat-type segment could masquerade as local rent momentum. A new
past-only recipe estimates each historical row relative to a smoothed
flat-type/month/10-square-metre/10-year-age cell before comparing the latest six
months with the preceding six. The same fixed 1,600-tree budget, validation rows,
seeds 42/2026 and no validation-label early stopping were used.

| Window | Current mainline | Prior local-trend candidate | Composition candidate | Versus mainline |
|---|---:|---:|---:|---:|
| Jan-Mar 2025 | 440.8896 | 440.7629 | 440.8613 | +0.0284 |
| Apr 2024-Mar 2025 | 466.8866 | 466.7566 | 466.5409 | +0.3457 |
| Apr 2023-Mar 2024 | 591.0835 | 590.3473 | 591.2047 | -0.1213 |
| Dec 2023-Mar 2025 | 483.8323 | 483.4345 | 483.5960 | +0.2363 |

The composition candidate is rejected: it improves the recent 12-month window
but is worse than the prior local-trend candidate on three of four windows and
slightly worse than the current mainline on the oldest window. No submission or
final models were generated for it. Its per-fit evidence remains in the existing
temporal reports, without a new stage directory or report file.

### Time-bias diagnostic and decision

The current mainline's mean error changes sign across consecutive rolling
windows: +83.62 SGD in April 2023-March 2024 and -41.47 SGD in
April 2024-March 2025. Applying the older window's month-ahead bias corrections
to the next 12-month window worsens RMSE from 466.89 to 489.67. A fixed horizon
offset is therefore rejected as a regime-dependent correction, not promoted as
a leaderboard candidate.

Correction after code review: the composition recipe above changed the shared
historical rent ratios as well as trend columns, hence it also changed the
residual-target baseline. Its result rejects that combined implementation; it
does not isolate or reject composition control of trend alone. Similarly, the
489.67 diagnostic rejects only the tested unshrunk previous-year horizon offsets,
not all temporal calibration methods. The requested current-feature comparison
with a different learner remained outstanding at that checkpoint.

Twenty-four tests and all nine setup/data checks pass. The public-verified
432.47187 history-diversity artifact remains the mainline, and
`submission_local_trend_diversity.csv` remains the only unsubmitted candidate
supported for one platform comparison. The next model experiment should learn
regime adjustment from predictors available at forecast time and prove transfer
from earlier cutoffs to later cutoffs before final training.

## Isolated trend ablation and current-feature learner comparison — 23 September 2026

The isolated `composition_trend_only` recipe gives only the four trend columns
their composition-adjusted ratios. All other adaptive columns, including the
residual offset, must be exactly equal to `local_trend`; a synthetic composition
shift test verifies both this invariant and same/future-label exclusion.

Experiment scope fixed before the formal results: CatBoost direct/residual
heads at 1,600 trees, seeds 42/2026 and the same four core temporal windows.
XGBoost uses exactly the `local_trend` feature table, native categorical splits
with training-only vocabularies, the same rows, labels and 1,600-tree budget.
Its bounded screen is depth 6/8, direct/residual, seed 42, on `short_recent`
and `horizon16`. Only a promising head or pre-specified 25% XGBoost / 75% current-mainline blend
advances to further windows and seed confirmation; no public scores select it.
Equal tree counts are a controlled budget, not equal model capacity or compute.

GPU compatibility: the initially installed XGBoost 3.4.1 used CUDA 13.3 and
fell back to CPU under driver 560.94. The experiment aborted instead of recording
that as a GPU fit. XGBoost 3.0.5, built with CUDA 12.5, runs on `cuda:0` and is
pinned. A small regressor adapter preserves its JSON metadata compatibility
with sklearn 1.9; categories and rent-unit predictions survive save/reload.
Learner versions enter both the run identity and the per-fit evidence ledger.

The XGBoost screen completed all eight planned fits using actual `cuda:0`.
The table below uses seed 42 on both sides, not a two-seed CatBoost ensemble
against single-seed XGBoost. Each head averages its direct/residual models.

| Window | XGB depth | CatBoost local-trend head | XGB head | Mainline | 25% XGB + 75% mainline |
|---|---:|---:|---:|---:|---:|
| Jan-Mar 2025 | 6 | 445.4677 | 455.9069 | 440.7998 | 442.1007 |
| Jan-Mar 2025 | 8 | 445.4677 | 456.7098 | 440.7998 | 442.1462 |
| Dec 2023-Mar 2025 | 6 | 487.0015 | 503.0877 | 484.0099 | 484.6837 |
| Dec 2023-Mar 2025 | 8 | 487.0015 | 505.2490 | 484.0099 | 484.9992 |

XGBoost/mainline residual correlations were 0.957-0.971. Both depths and both
fixed cross-learner blends failed the screen, so no further seeds or final models
were trained for XGBoost. This is current-feature negative evidence, not a claim
that CatBoost is universally best or that broader XGBoost searches cannot work.

### Isolated-ablation results and closeout

All 16 planned isolated CatBoost fits completed (four windows, two targets and
two seeds). The final combination retains the identical frozen stale-history
components and equal weights; only the normal-history trend columns change.

| Window | Prior local-trend candidate | Isolated-trend combination | Gain versus prior | Paired 95% gain interval |
|---|---:|---:|---:|---|
| Jan-Mar 2025 | 440.7629 | 440.7610 | +0.0019 | -0.2098 to +0.2020 |
| Apr 2024-Mar 2025 | 466.7566 | 466.9105 | -0.1539 | -0.3138 to -0.0055 |
| Apr 2023-Mar 2024 | 590.3473 | 590.6350 | -0.2876 | -0.4700 to -0.1008 |
| Dec 2023-Mar 2025 | 483.4345 | 483.7752 | -0.3407 | -0.5031 to -0.1811 |

Intervals use the existing 1,000-replicate paired address-cluster bootstrap.
They describe these overlapping development samples, not future or private-score
uncertainty. Reproduce them with the existing summary command and
`--reference-candidate local_trend_history_diversity_blend --bootstrap`.
The canonical `temporal_summary.csv` was then restored to the four-component
hierarchical reference so historical comparisons retain their documented meaning.

The isolated head alone reached 579.7247 on the older year, slightly better than
the prior local-trend head's 579.8459, but its full combination was worse. Head
improvements therefore do not substitute for evaluating the final combination.
Reject this isolated implementation for promotion: its latest-quarter gain is
effectively a tie and it loses on the other three windows. This does not invalidate
the composition-control idea in general.

This round completed 24 formal GPU fits with 1,600 trees each (16 CatBoost and
8 XGBoost), totaling 1,206.56 seconds in the per-fit model timers, excluding
feature construction, smoke tests and reporting. GPU smoke runs, all 26 tests,
and the nine-file setup/checksum check passed. Tests verify exact preservation
of all non-trend adaptive features and residual offsets, attenuation of a known
synthetic composition shift, exclusion of same/future labels, XGBoost category
mapping/JSON reload, and the fixed cross-learner blend's weights/model count.
A separate 90-query check spanning all 51 training months reproduced the frozen
`2be9063` local-trend adaptive features exactly. All 97 pre-existing per-fit rows
retain their metrics and prediction paths.

Mainline and retained-candidate model hashes and CSV hashes were rechecked; both
eight-model runs remain intact. Mainline SHA-256 remains
`b464777c16870c3f1484dccf8a418465b239d6f22ded97965bb5553c81f113b7`; candidate remains
`a88da3be518a4793486765f5d9ff44c9b27b3248bb195e6f2d0264ae7dabb57a`.
No final training, new submission, raw-data edit or browser action occurred.
The original module/report layout and cumulative progress file are retained.

## Local-trend mainline promotion — 24 September 2026

The user reported public RMSE **432.14881** for
`submission_local_trend_diversity.csv`, an improvement of **0.32306** over the
432.47187 history-diversity mainline. This agrees directionally with the four
pre-recorded chronological windows, where the candidate improved by 0.13, 0.13,
0.74 and 0.40 RMSE. It is supporting evidence on the public half, not a guarantee
for the private half; the older-year deficit versus the four-component reference
remains 9.81 RMSE.

The submitted CSV and its eight already-verified CBMs were promoted byte-exactly
without retraining. The default config now records the mixed views explicitly:
`local_trend:0` and `hierarchical:6`, crossed with direct/residual targets and
seeds 42/2026. `submissions/final_submission.csv` now has SHA-256
`a88da3be518a4793486765f5d9ff44c9b27b3248bb195e6f2d0264ae7dabb57a`.
The previous 432.47187 artifact remains reproducible under the existing
`artifacts/candidates/history_diversity/` directory and its original submission
filename. No raw data, final-submission selection or experiment results changed.

## Time-transferred stacking experiment — 24 September 2026

The proposed stacking idea was tested using the four distinct two-seed heads in
the 432.14881 mainline: local-trend direct, local-trend residual, stale-history
direct and stale-history residual. Every meta model was trained only on cached
out-of-fold predictions from months strictly earlier than its evaluation period.
No base model was scored on rows used to train that base model, and no evaluation
label selected a meta-model weight.

Three fixed transfer checks were used: Apr 2023-Mar 2024 to Apr 2024-Mar 2025;
Apr-Nov 2023 to Dec 2023-Mar 2025; and Apr-Dec 2024 to Jan-Mar 2025. The tested
meta models were exact nonnegative simplex weights, a 50% shrink toward equal
weights, and Ridge with its regularization selected on the final three months of
the meta-training period.

| Meta model | Older year -> recent year | Pre-Dec 2023 -> 16 months | Pre-Jan 2025 -> recent quarter |
|---|---:|---:|---:|
| Equal-weight mainline | 466.7566 | 483.4345 | 440.7629 |
| Simplex gain | -2.4271 | -2.8193 | +0.0692 |
| Half-shrunk simplex gain | -0.4737 | -0.2427 | +0.0723 |
| Temporal Ridge gain | -8.7075 | -1.8337 | -0.9461 |

Positive gain means lower RMSE than the equal-weight mainline. Address-cluster
bootstrap intervals for the recent-quarter half-shrunk gain were -0.05 to +0.19;
both longer-transfer simplex losses were clear. The older regimes learned to put
most weight on the local-trend residual head and almost discard the stale-history
heads, but that relationship did not transfer. Equal weighting remains more
robust across regime changes.

The result rejects these static four-head meta models for promotion. It does not
reject stacking with genuinely diverse base predictions or a separately validated
regime-aware gating model. No final model, submission CSV, raw data or default
configuration changed. Reproduce the tracked result with
commit `bab7c46`; the current `--stacking-only` command runs the expanded matched-
horizon study below. Machine-readable current evidence is in
`reports/stacking_results.csv`.

## Matched-horizon four-head versus eight-model stacking — 24 September 2026

The initial stacking check above left two important questions open: two transfer
pairs did not use equal forecast horizons, and seed averaging could conceal useful
model-level differences. The follow-up therefore trained 40 missing GPU OOF fits
and froze 64 source predictions covering eight time windows, four candidate
families and seeds 42/2026. `reports/stacking_sources.csv` records each run ID,
training/validation boundary, configuration, data/code provenance, prediction
path and SHA-256. It is verified rather than silently reselected on later runs.

Six forward tests now have matched horizons: four successive three-month tests
and two successive twelve-month tests. Meta-training months always end before
the evaluation window and raw row IDs cannot overlap. Four seed-averaged heads
and all eight individual predictions were compared under identical evaluation
rows. The methods were equal weighting, nonnegative simplex, 50%-shrunk simplex,
an eight-input seed-pair shrinkage variant, and a strongly regularized temporal
Ridge correction. Address-cluster bootstrap intervals were also recorded.

| Forward evaluation | Equal-weight RMSE | Half-shrunk 4-head gain | Half-shrunk 8-model gain |
|---|---:|---:|---:|
| Sep-Nov 2023 | 574.2864 | +0.9341 | +0.8973 |
| Jul-Sep 2024 | 469.9204 | -0.1075 | -0.0595 |
| Oct-Dec 2024 | 436.8152 | -0.4952 | -0.5632 |
| Jan-Mar 2025 | 440.7629 | -0.4144 | -0.4153 |
| Apr 2023-Mar 2024 | 590.3473 | +6.8809 | +7.2101 |
| Apr 2024-Mar 2025 | 466.7566 | -0.4864 | -0.8903 |

Positive gain means lower RMSE than equal weighting. The comparison must not
cherry-pick a different meta method after seeing each evaluation period; its
purpose is stability. The apparent annual gain did not transfer to the next
year. On the two latest quarterly windows every learned four-head and eight-model
method lost to equal weighting. The individual seeds were also nearly duplicates
in recent windows, so eight inputs added degrees of freedom without enough new
signal. Direct eight-model stacking is technically valid but is rejected for
promotion under current evidence.

All 28 automated tests pass, including a new check that four-head columns equal
the paired-seed averages and that four- and eight-input equal-weight predictions
are identical. The mainline config, eight CBMs and `final_submission.csv` remain
unchanged; submission SHA-256 is still
`a88da3be518a4793486765f5d9ff44c9b27b3248bb195e6f2d0264ae7dabb57a`.

## Recency-aware stacking follow-up — 24 September 2026

The same frozen 64 OOF predictions were reused to test whether old regimes were
causing the static weights to fail. No CatBoost model was retrained. Two rules
were fixed across every forward test: learn the simplex from only the latest
available meta-training window, or use all meta-training rows with a 12-month
exponential half-life. Both were shrunk 50% toward equal weights. Bootstrap
comparisons use identical address-cluster draws across methods.

The latest-window four-head rule improved Jul-Sep 2024 by 0.045 RMSE and Jan-Mar
2025 by 0.106, but their confidence intervals crossed zero. It lost 0.229 on
Oct-Dec 2024 and 0.474 on Apr 2024-Mar 2025. With all eight seed-level inputs it
lost 0.054 on Jan-Mar 2025 and 0.442 on the recent annual window. Fixed time
decay lost 0.184/0.239 on the latest quarter for four/eight inputs and also lost
on the recent annual window.

Recency reduces several earlier losses, so regime drift remains a useful lead,
but neither fixed rule transfers consistently enough to replace equal weighting.
No model, config or submission changed. All 29 tests pass; the added test verifies
that weighted simplex fitting responds to the intended sample emphasis.

## Fit-record correction and competition-horizon check — 24 September 2026

The result schema now distinguishes the full set of earlier OOF rows available
at each origin from the exact rows used by a method. `meta_fit_start/end`,
`meta_fit_rows`, `meta_fit_source_run_ids` and `meta_fit_weighted` make this
explicit. `train_rmse` and `meta_fit_rmse` are now calculated on those actual fit
rows; the time-decay version uses the same sample weights as its objective. This
corrects a reporting-only issue where recent-window weights were evaluated across
all earlier windows. Evaluation RMSE, weights and promotion decisions were not
affected.

Eight existing horizon predictions were added to the frozen manifest, bringing
it to 72 exact sources. A seventh forward check learns weights from Apr 2022-Mar
2023 OOF predictions and evaluates Dec 2023-Mar 2025, a 16-month span close to
the competition test's 17 months. Equal weighting scored 483.4345. The recent-
window half-shrunk rule scored 483.9728 with four heads (gain -0.5384, 95% address-
cluster interval -0.8758 to -0.1697) and 484.6230 with eight inputs (gain -1.1885,
interval -1.5872 to -0.7733). Fixed time decay was also worse.

At this point no candidate was generated and the mainline remained unchanged.
The initial interpretation that this removed the case for a leaderboard probe
was too strong: this stress check also changes the fitting-window length and
historical regime. See the clarification below.

## User-requested leaderboard probe artifact — 24 September 2026

At the user's request, the fixed recent-window four-head rule was exported for a
single public-leaderboard comparison despite the negative long-horizon evidence.
No base model was retrained. The generator verifies the raw data and all eight CBM
hashes, rebuilds their 50,000 test predictions, and confirms that the original
equal weights reproduce the 432.14881 source CSV before applying new weights.

`submissions/submission_recent_window_stacking.csv` has SHA-256
`05bf5b69408b0b8d61a35cc54590277ee48e41cb8c30a371a594a6f360ce9cff`.
Its head weights are 0.138908 local-trend direct, 0.320003 local-trend residual,
0.267238 stale-history direct and 0.273851 stale-history residual, divided equally
between seeds 42/2026 within each head. Compared with the mainline its test
predictions have mean change -1.247 SGD and RMS change 5.618 SGD.

This is a diagnostic submission, not a promoted model. The mainline files remain
unchanged pending the reported Kaggle score. Exact component order and provenance
are recorded in `reports/submission_recent_window_stacking_metadata.json`.

## Candidate protection and interpretation correction — 24 September 2026

The 16-month check uses a twelve-month fit window ending Mar 2023, while the
exported probe freezes weights fitted on Oct-Dec 2024. The former is evidence of
historical long-horizon risk, not an isolated test of recent-quarter weighting.
Address-cluster intervals describe sampling within that historical evaluation;
they do not quantify future regime uncertainty or guarantee a private score.

The probe's weights were deliberately retained from the recorded quarterly
experiment. They were not refitted on the available Jan-Mar 2025 OOF predictions,
so it is a fixed historical-weight probe, not a latest-data refitted stack.
README and the current plan now state this and that the probe score is pending.

Reweighted generation now requires fresh CSV and metadata paths, rejecting both
complete and partial existing runs before loading models. Exclusive file creation
also prevents overwriting if a file appears after preflight. The source CSV hash
and ID order are verified in addition to model/data hashes. A partially failed
new run is retained and must use new paths on retry; no atomic-pair guarantee is
claimed. Four regression cases cover existing/partial output and source-metadata
protection. Original candidate metadata retains the original generator provenance.
The candidate CSV, mainline CSV and eight CBMs were not regenerated or replaced.

## Fixed-weight probe public result — 24 September 2026

The user supplied the completed Kaggle result for
`submission_recent_window_stacking.csv`: **432.04286**. The score is recorded as
user-reported, not newly browser-verified; no current rank was provided.
The local CSV SHA-256 is
`05bf5b69408b0b8d61a35cc54590277ee48e41cb8c30a371a594a6f360ce9cff`.
It improves the 432.14881 mainline by 0.10595 and the first 438.94488 submission
by 6.90202. This is the best recorded public submission, while the default
equal-weight mainline remains unchanged pending stronger stability evidence.

The public gain is close in magnitude to the +0.10638 quarterly OOF gain, but
this agreement does not prove generalization to the private half or establish
that all stacking/recency schemes work. The quarterly interval still crosses
zero; losses on other historical origins remain part of the evidence. The
annual-fit sixteen-month stress test also changes regime and fit-window length,
so it is not an isolated rejection of quarterly-refit deployment.

Next work is recorded in PROJECT_PLAN: explain the gain/loss groups using cached
predictions, perform at most one prespecified quarterly-refit follow-up, and
prioritize a complementary past-only comparable-property predictor and market
calibration. Avoid prolonged public-score weight tuning. This update records
the score and plan only; no training, candidate regeneration, promotion or
Kaggle final-selection action was performed.
