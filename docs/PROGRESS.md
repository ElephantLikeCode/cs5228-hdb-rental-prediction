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

## Progress-report slide deck - 25 September 2026

Canvas was rechecked read-only. The progress report requires an approximately
10-15 slide PDF that covers the goals, task/data understanding, initial design
and results, current obstacles, and one or two final slides of next steps. The
assignment accepts PDF files only and states that one group member may submit.

A 14-slide English draft now lives in `progress_report/main.tex`, adapted from
the user-selected NUS LaTeX Beamer template. Five selected report figures and
the NUS logo are stored in `progress_report/assets/` so a clean clone does not
depend on ignored local plot files. The compiled deliverable is
`output/pdf/CS5228_Progress_Report.pdf`. It covers the data audit, chronological
validation, preprocessing and leakage controls, baseline comparison, spatial
and historical features, current ensemble, error analysis, submitted public
scores, limitations, and two final planning slides. It also includes the AI-use
declaration requested on the Canvas assignment page.

No model, experiment table, submission file or mainline selection changed. The
title now lists all four members in a consistent surname-first romanisation and
includes their student IDs. A final group review remains necessary before
submission.

The report evidence was then audited against the saved experiment artifacts.
The baseline chart now compares CatBoost with the other learners on core inputs,
the staged-results slide no longer presents successive pipeline revisions as a
single-feature ablation, and the rent-band chart is rebuilt from the selected
440.7629-RMSE mainline predictions. The validation slide now states that its
stress windows overlap, while the ensemble slide documents the weighted stale-
history augmentation and full-history inference behavior. The results slide
also distinguishes the selected 432.14881 mainline from the unpromoted
432.04286 stacking candidate and quantifies the 9.81-RMSE older-period deficit.

The final report audit then tightened four interpretations. The initial model
screen now distinguishes its validation-based CatBoost early stopping from the
later fixed 1,600-tree candidate protocol. The rent-band discussion labels the
entire at-or-below-SGD-2,500 group rather than treating every row as an anomaly,
and identifies the 954-row removal as an earlier sensitivity test. The
older-year comparison now names the current equal-weight mainline and the
earlier adaptive-market reference. The stacking result records that the weights
were fitted on Oct-Dec 2024 and evaluated on Jan-Mar 2025, with a 0.106-RMSE
gain and an address-bootstrap interval of -0.034 to 0.226 that crosses zero.
The slide labels this explicitly as a 95% interval. LaTeX log files created
inside `progress_report/` are ignored as build intermediates.

The title slide now lists the four members vertically in alphabetical order by
surname-first romanised name: Han Linxuan (A0350911M), Sun Shenghe
(A0354849L), Zhang Yuhong (A0356795H), and Zhang Zhexuan (A0355290A). The
balanced two-column block is centred as a whole. All slide source notes use one
bottom-left alignment rule, including figure slides that otherwise centre their
main content. The plan slide was also updated to remove the completed
student-ID task.

## Professor feedback and bounded experiments - 6 October 2026

The team supplied feedback confirming that the project direction is sound,
while recommending clearer categorical encoding and candidate definitions,
analysis of which flat types/areas generate extreme errors, possible specialist
models, and early final-report drafting. The instructor's announcement clarifies
that dates are ordered but there is no unique individual-flat identity. Block
histories represent comparable transactions. The 425 identical rows are not
proven duplicate observations and remain retained. Chronological validation still
matches the competition split, and query/validation targets must not enter
prediction features. The user deprioritised stacking and kept it as an alternative.

A bounded screen was predefined: nearest comparable rents with three market
forecasts; input-flat-type specialists; and a soft three-band rental mixture.
The latter uses a feature-based classifier at inference, never true query rent.
Fixed 25% blends with the frozen eight-component mainline are included. The
trained screens use seed 42 and 1,600 trees without outer early stopping.
Three-month and sixteen-month windows screen candidates first; promising
candidates need recent-year/older-year confirmation and a second trained seed.
Promotion requires a meaningful recent gain (about 2 RMSE), positive long-
horizon gain and no material regression in confirmation windows. The screen
uses fewer ensemble members than the mainline and is not an equal-capacity
causal ablation. All validation windows remain development evidence.

The entry point is `src/feedback_experiments.py`. Results accumulate in
`reports/feedback_results.csv`; hash-identified configurations, matched predictions
and subgroup diagnostics stay in ignored `artifacts/feedback_study/` directories.
A GPU smoke run successfully exercised all three families on 3,000 training and
300 validation rows with 40 trees. Its metrics are execution checks only and do
not enter the formal ledger. Four additional tests cover canonical routing,
mixture probabilities, constant-rent comparables and rejection of same-month
reference labels. The complete test suite passed: 37 tests.

A diagnostic on Jan-Mar 2025 found 2,408 of 8,924 rows in canonical identical-
input groups, with a maximum within-group rent span of SGD 2,600. The best
constant prediction within each identical-input group, calculated using the
validation labels themselves, gives an optimistic sample RMSE lower bound of
171.85636. Singleton groups are fitted perfectly in this diagnostic. It is not
a deployable predictor, population noise-floor estimate, or evidence about other
teams' methods; external individual-flat information could change the grouping.
It illustrates the ambiguity emphasized by the instructor. Other teams' public
scores near 173 cannot be attributed to a particular method from the leaderboard.

The formal GPU screen completed with run ID `d0e607697ec39679`, two validation
origins, 22 score rows (including two frozen references), 14 regressor fits and
two classifiers. Recent validation uses 141,076 training / 8,924 validation rows;
the sixteen-month origin uses 102,829 / 47,171. No validation labels were provided
to fitting, early stopping, reference-rent construction or mixture routing.
Classifier budgets are 1,000 depth-6 trees; expert/regressor budgets are 1,600
depth-8 trees. The ledger's `trees` describes the regressor budget, not the total
ensemble size. Frozen-reference `model_seconds=0` means cached predictions were
reused, not that the mainline trains for free. Blend times describe the new
family's computation only. `seed=42` identifies this screening run; the frozen
reference still averages seeds 42 and 2026 across four heads. `feature_group`
names the constructed local-trend/spatial pool, not the nearest method's actual
inputs: nearest uses flat type, approval month, coordinates, area and lease age
plus training-only observed rents. The frozen reference also includes its
unchanged stale hierarchical-history view, as its source run IDs specify.
Unique new-family computation totalled 1,238.96
seconds, plus 61.22 seconds for shared feature construction; do not sum duplicated
blend/nearest rows as independent training runs. RTX 4060 GPU execution was
confirmed; the user connected power and adjusted power settings during the run.

| Fixed recipe | Jan-Mar 2025 RMSE | Dec 2023-Mar 2025 RMSE |
|---|---:|---:|
| Frozen current eight-component mainline | 440.7629 | 483.4345 |
| Nearest comparables, flat market | 455.7328 | 493.4052 |
| Nearest comparables, damped market trend | 455.8490 | 491.8830 |
| Nearest comparables, bounded linear trend | 455.9122 | 506.2170 |
| Input-flat-type specialists | 449.5262 | 506.4313 |
| Soft rental-band mixture | 451.5176 | 487.3899 |
| Mainline 75% + nearest/flat 25% | 441.8423 | 482.1983 |
| Mainline 75% + nearest/damped 25% | 441.8372 | 482.5230 |
| Mainline 75% + nearest/linear 25% | 441.8366 | 483.7023 |
| Mainline 75% + flat specialists 25% | 441.2614 | 483.3562 |
| Mainline 75% + rental mixture 25% | 441.9618 | 482.4875 |

No configuration passed the predefined recent-and-long-window gate. No second
seed, additional confirmation-window fit, final retraining or new submission
was warranted. The nearest/flat blend gains 1.2361 on the long window but loses
1.0794 on the recent quarter; the soft-mixture blend gains 0.9469 and loses
1.1989, respectively. This is evidence against promoting these specific recipes,
not proof that specialists, nearest comparables or mixture models cannot work.
The recent mixture blend improves MAE only from 316.5491 to 316.2118 while its
RMSE worsens; no claim of overall superiority follows.

Grouped diagnostics refine the next research question. In recent validation,
1,458 rows with true rent <= SGD 2,500 (16.34%) contribute 52.23% of mainline
squared error; the long window also attributes 50.86% to that band. It is not an
anomaly label or an inference-time routing variable. Executive flats have recent
RMSE 506.73, but only 462 rows and 6.84% of total squared error. Four-room and
three-room flats contribute 35.11% and 29.76%, respectively. Jurong West five-
and four-room groups contribute 4.31% and 3.65%; Bukit Merah five-room flats have
RMSE 612.09 across 106 rows but contribute only 2.29%. High subgroup RMSE alone
does not identify the largest total improvement opportunity. The classifier-
routed mixture did not resolve low-band overprediction in the recent quarter.

Next, investigate cross-fitted, training-only comparable-flat premiums and
dispersion: train-row encodings exclude the entire held-out inner fold, whereas
outer validation uses only the outer training partition. This would explicitly
test the cross-sectional interpretation and reduce the mismatch between sparse
early training histories and full-history inference, without assuming a unique
flat identity or accessing future validation labels. Keep the existing history
and target recipe fixed in the first ablation. Examine low-rent support and
observable region/type/area groups, rather than routing by true query rent.
Whether more permissible information exists should be checked against course
rules; the leaderboard alone cannot explain other teams' large score gains.

Post-run checks: all 37 automated tests pass; all eight selected CBM checksums,
both existing submission checksums and raw train/test checksums match their
recorded provenance. Raw data, `configs/model.json`, mainline models and existing
CSVs remain unchanged. This round's code, result ledger and cumulative document
updates remain local and have not been committed or pushed. Existing unrelated
report/slide work was preserved.

### Comparable-statistics follow-up - 6 October 2026

The user authorised the information audit and isolated cross-fitting study.
The existing entry point now supports `--families crossfit` and `--audit-only`.
`reports/information_audit.csv` records input cardinalities, missingness, canonical
category counts, date ranges and production spatial-join coverage. Source snapshots
are retained inside ignored run directories, including the previous formal run.
No raw input rows are removed. FEE and FURNISHED are constant in both supplied
datasets; coordinates, maximum floors and completion years have no missing joined
rows in either dataset.

Predefined comparison: keep the original local-trend feature pool, residual
offset, seed and fixed 1,600-tree/depth-8 budget. Add comparable relative-rent
statistics at town/flat, street/flat, address/flat and address/flat/rounded-area
levels. Reference rents are normalised by reference-only month/town/flat market
means shrunk toward month/flat means (strength 20). Group estimates shrink toward
reference flat-type priors with strength 20. The means view adds eight mean/count
features; the distribution view adds eight standard-deviation/relative-low-rate
features as well. Relative-low means a reference rent ratio below 0.65, not an
assertion that a transaction is incorrect.

Five inner GroupKFold partitions group identical canonical input rows (including
month) together. Every training encoding excludes the whole query fold. Outer
validation encodings use the outer training data only. A strict-past distribution
control uses identical statistic definitions and identical validation features,
but each training query uses only earlier training months. The earliest month
gets label-independent mean=1 and count/std/low-rate=0 defaults. This contrast
isolates the training-encoding rule; the means/distribution contrast studies the
additional dispersion features. Direct and residual targets are each compared
against their frozen same-seed original components, not only the larger ensemble.

Each window fits six GPU regressors. A seed-specific ensemble comparison replaces
exactly two normal-history components in the existing eight-component prediction;
the remaining six stay frozen. It does not assume that an untrained second seed
will behave identically. Screen on Jan-Mar 2025 and Dec 2023-Mar 2025 first. A
clear two-component recent gain around 1 RMSE with positive long-window evidence
can justify second-seed confirmation; promotion still requires about 2 RMSE recent
gain for the complete candidate, positive long-window gain and no material
confirmation-window regression. Additional windows/seeds are conditional rather
than automatically fitting every failed recipe. Smoke scores are execution QA.

Additional audit findings: canonical town/flat/model/street categories have zero
unseen test rows against the full training set. At month precision, March 2025
appears in both datasets: 3,032 training and 380 test rows. This is only 0.76% of
test rows and does not establish that individual event ordering was violated.
The production past-only rule excludes those March training labels for March
queries; an available-training-only same-month study is a separate future
hypothesis, not a plausible explanation of a 432-to-173 score gap by itself.
The supplied COE and stock-price tables are not consumed by the current feature
pipeline; their potential economic information remains untested. No external
target lookup was performed.

A validation-label diagnostic uses the actual same-month/town/canonical-flat
median only to describe relative price tails, never to route predictions or
construct training features. On the frozen recent mainline:

| Actual rent / validation-group median | Rows | RMSE | Mean prediction minus rent | Fraction of total squared error |
|---|---:|---:|---:|---:|
| <= 0.50 | 93 | 1,747.62 | +1,710.19 | 16.38% |
| > 0.50 and <= 0.75 | 641 | 941.11 | +891.14 | 32.75% |
| > 0.75 and <= 1.25 | 8,091 | 314.26 | -79.36 | 46.09% |
| > 1.25 | 99 | 914.71 | -819.53 | 4.78% |

The 734 rows at/below 75% are 8.22% of validation but account for 49.13% of
squared error. Jurong West 5/4-room, Ang Mo Kio 3-room, Kallang/Whampoa 3-room
and Bukit Merah 3-room are among their largest absolute SSE contributors. These
are label-defined research groups, not observable routing rules or proven data
errors. Even setting all 734 residuals to zero while leaving every other residual
unchanged would yield approximately 314 RMSE on all 8,924 rows. Resolving only
this tail would therefore not reproduce a public score near 173 on this window.
This arithmetic is an optimistic what-if calculation, not a achievable-model
claim or a public-test estimate.

The formal follow-up completed on GPU with run ID `02e104f0ffdf57c5`: twelve
1,600-tree regressors, 32 scored rows and 5,408 subgroup rows. The result ledger
now has 54 formal rows including the previous study; no smoke rows are included.
New regressor fitting/prediction totalled 735.80 seconds, production feature
construction 46.89 seconds and additional statistic construction 54.02 seconds.
Source snapshots, fold assignments, sixteen encoded feature columns and exact
query predictions are retained under that ignored run directory. Entry command:

```bash
python src/feedback_experiments.py --families crossfit --windows short_recent horizon16 --seed 42 --task-type GPU --resume
```

Matched seed-42 normal-history pairs (direct/residual mean):

| Training encoding / extra features | Jan-Mar 2025 RMSE | Dec 2023-Mar 2025 RMSE |
|---|---:|---:|
| Original normal-history pair | 445.4677 | 487.0015 |
| Inner-fold-excluded mean/support statistics | 465.0201 | 512.2764 |
| Inner-fold-excluded distribution statistics | 463.4765 | 509.4779 |
| Strict-past distribution statistics | 446.2969 | 489.2022 |

Actual eight-component predictions, replacing only the two seed-42 normal heads:

| Eight-component configuration | Jan-Mar 2025 RMSE | Dec 2023-Mar 2025 RMSE |
|---|---:|---:|
| Frozen current mainline | 440.7629 | 483.4345 |
| Replace with cross-fitted mean/support heads | 442.5545 | 483.8341 |
| Replace with cross-fitted distribution heads | 442.3438 | 483.6721 |
| Replace with strict-past distribution heads | 440.6410 | 483.2988 |

The encoding-rule contrast is adverse: cross-fitting distribution statistics
loses 17.1796 recent RMSE and 20.2758 long-window RMSE against the same-statistic
strict-past pair. Adding dispersion to the cross-fitted means does recover 1.5435
and 2.7985 versus cross-fitted means alone, but both remain worse than the original
pair. The strict-past pair is itself worse by 0.8292 / 2.2006; its tiny ensemble
gain (0.1219 / 0.1357) reflects complementary errors rather than a better stand-
alone predictor. Its long-window MAE worsens from 355.3968 to 355.7491. No recipe
passes the predefined advancement gate, so no second seed, annual confirmation,
final retraining, submission generation or mainline promotion was performed.

Recent price-band diagnostics also contradict a simple global-offset explanation:
cross-fitted distribution-pair bias is approximately zero overall, while low,
middle and high rent RMSE all worsen. High-rent RMSE rises from 567.59 to 631.84,
and Executive RMSE from 512.92 to 555.40. Address/flat/area encoding support is
larger at validation (log-count median 2.708 versus 2.485 in cross-fitted training),
as expected when all reference rows become available. Mean/std/low-rate marginal
quantiles are relatively similar. These summaries do not establish the exact
cause of failure; temporal heterogeneity, fold noise and model reliance on the
new statistics are hypotheses, not proven explanations. The experiment rejects
this encoding/statistic/budget configuration, not all cross-fitting methods or
the instructor's cross-sectional interpretation.

Next priority: evaluate information already provided but unused. The stock table
contains 77 symbols from 4 January 2021 to 30 July 2026; COE spans 2021-2026 with
four categories. A bounded market-level calibration study can test monthly
economic summaries alongside the original baseline, with explicit data-availability
assumptions and same-budget/feature controls. Further independent-learner or
targeted residual studies remain possible; repeated weight/threshold searches
and another full cross-fit configuration are lower priority after this result.

Final verification: 41 tests pass, including perturbing every label in an inner
query fold and perturbing current/future-month labels in the strict-past control.
Formal metrics are finite, temporal cutoffs and ledger keys are valid, source
hashes match the saved run, and the eight original CBMs, two submitted CSVs and
raw train/test hashes remain unchanged. All changes remain local; no Git commit
or push has been performed. Existing unrelated slide/report modifications were
preserved. README retains its one main comparison table; this cumulative entry
and the existing result ledger hold the new rejected-candidate evidence.

## Supplied market information, matched learners and targeted corrections — 6 October 2026

Completed the remaining bounded experiments before closing the work. Formal run
`8050c7a295047db8` covers Jan-Mar 2025 (141,076 training / 8,924 validation rows)
and Dec 2023-Mar 2025 (102,829 / 47,171 rows), seed 42. Training ends Dec 2024
and Nov 2023 respectively. All rows and rental labels are retained. The frozen
eight-component mainline is the deployment reference; its same-seed two
normal-history heads are the feature/learner comparison reference. No outer
validation labels enter fitting, encoders, sample selection or prediction routing.

Research uses the existing entry point `src/feedback_experiments.py`, with
reusable market/correction logic isolated in `src/market_experiments.py` rather
than creating multiple competing training pipelines. Production features,
configuration, eight selected CBMs and submission generator are unchanged.

### Experiments and availability assumptions

- COE: monthly mean premium, three-month premium change and bids/quota ratios
  for categories A/B/C/E, parsed from the supplied currency/comma fields.
- Stocks: monthly CLOSE-based median-return cumulative index, one-/three-month
  returns, three-month volatility, positive-return breadth and symbol coverage.
  No rent labels or externally downloaded stock information is used. Snapshot
  membership introduces survivorship/availability limitations; CLOSE avoids
  adjusted-close hindsight but does not remove all corporate-action effects.
- Each query joins the preceding month. No backward fill or future-fitted
  normalization is used. The COE/stock single-group screens share an aggregate
  source-missingness indicator; they are not completely isolated missingness
  ablations. The combined view contains 19 extra numeric columns.
- COE alone, stocks alone, both and an origin-only frozen future-covariate
  control each train direct/residual CatBoost heads (1,600 trees, depth 8).
  Supplied-snapshot variants assume future-month economic information is
  permitted in this offline competition. Origin-only does not use it. A monthly
  lag is not verified release-time availability. The CatBoost control uses
  separate same-seed fits, so small GPU fit variation remains a confound.
- GPU XGBoost (original/market inputs, direct/residual, fixed 1,600-tree depth-8
  budget) uses training-only categorical vocabularies. CPU Ridge (alpha 100)
  uses training-only numeric imputation/scaling and one-hot encoding, with the
  unchanged market-residual target. Same rows/features are compared, not equal
  algorithm capacity. XGBoost confirms actual CUDA training; its host-input
  prediction-transfer warning is not a CPU-training fallback.
- Targeted correction uses the latest 12 months of existing, honest earlier-
  origin OOF mainline predictions. Overlaps retain the most recent origin,
  chosen without errors. It fits 600-tree depth-4 residual regressors, or a
  classifier plus two residual experts for rent below 65% of the OOF prediction.
  Validation routing is probabilistic and uses observable features/base
  prediction only. Predefined correction weights are 25%/50%. The recent/long
  correction fits use 35,239/23,869 OOF rows, never fitted-training residuals.

These families trained 32 GPU models and four CPU Ridge models. Family runtime
was 1,913.81 seconds, excluding 72.92 seconds of outer feature construction and
initial data loading. RTX 4060 usage was confirmed. Pair/replacement/blend
result rows reuse these fits and must not be counted as additional models.

### Results and adoption decision

`replace_seed` replaces only two of eight frozen components, not the untrained
second seed. The guarded Ridge row is a fixed 25% blend, and correction rows
apply a fixed 25% residual strength. These are deployment-candidate checks.

| Configuration | Jan-Mar 2025 RMSE | Dec 2023-Mar 2025 RMSE |
|---|---:|---:|
| Frozen current mainline | 440.7629 | 483.4345 |
| COE heads: replace two components | 440.8770 | 483.4336 |
| Stock heads: replace two components | 440.8159 | 483.5355 |
| Combined market heads: replace two components | 440.8101 | 483.4322 |
| Origin-only market heads: replace two components | 440.7727 | 483.5769 |
| Original-feature XGBoost: replace two components | 441.8813 | 485.2444 |
| Market-feature XGBoost: replace two components | 442.1336 | 484.8956 |
| Guarded, origin-only Ridge: fixed 25% blend | 442.3978 | 481.5146 |
| OOF residual correction, strength 25% | 440.9220 | 483.3395 |
| OOF relative-price mixture correction, strength 25% | 441.0134 | 483.3793 |

The COE pair improves its matched normal-history pair by only 0.207 long-window
RMSE and loses 0.863 recent RMSE. Combined-head replacement gains 0.0023 on the
long window and loses 0.0472 recently; the paired intervals cross zero. These
are not meaningful promotion evidence. Both XGBoost replacements lose in both
windows. Stronger 50% residual/mixture corrections also lose in both windows.
No candidate passes the recent-and-long advancement gate; no second seed,
additional promotion windows, final training, new CSV or mainline replacement
was warranted. This rejects these fixed configurations, not every use of market
information, every linear model, or every specialist method.

### Follow-up: diagnose unbounded linear extrapolation

Rather than stop at the first poor Ridge result, run `e3bf7c874aee98da` tested a
training-input-range clipping guard on both windows. It trained four additional
CPU Ridge models (20.13 seconds plus 61.10 seconds of outer features). The
origin-only market prediction reuses the SAME fitted market Ridge. The original
permitted rental offset is retained after prediction, not its clipped copy.
Training `ADAPT_HORIZON` ranges from 0 to 1; recent validation ranges 1 to 3,
with 67.84% outside the training range. Other future time/market inputs also
exceed training ranges. Unbounded original/market Ridge RMSE of 601.83/1207.32
recently becomes 464.97/466.41 with the joint clipping guard. Long-window
unbounded RMSE is 1251.85/3308.71. This supports extrapolation instability as
an issue, but does not identify a single causal feature because all numeric
inputs are clipped together. Even the best guarded blend loses 1.635 recently
despite gaining 1.920 on the long window. Address-cluster gain intervals are
[-2.456,-0.841] and [1.595,2.264], respectively. It remains rejected.

### Low-rent findings beyond RMSE

For the operational event `actual rent < 0.65 * frozen prediction`, prevalence
is 2.992% recently and 3.364% on the long window. These rows contribute 32.47%
and 31.27% of mainline squared error. The classifier's recent/long AUC is
0.5761/0.5433, average precision 0.04161/0.04061 versus prevalence baselines
0.02992/0.03364. There is weak ranking signal, not effective separation;
Brier scores 0.02911/0.03296 are also recorded. Neither the softer mixture nor
the direct residual correction improves the recent window. Unobserved contract
conditions remain a hypothesis, not a proven explanation or grounds to delete
these records. All target-defined groups stay diagnostic, not inference routes.

Observable-group diagnostics do not point mainly to Executive flats: of the
267 recent low-event rows, 94 are 3-room and 91 are 4-room (69.3% combined).
Their groups also contribute 64.87% of total mainline SSE, partly because they
are common. Two-room flats have the highest low-event rate among adequately
represented flat types, 7.02% (17/242), versus 3.22% for 3-room and 2.84% for
4-room. Among towns with at least 200 validation rows, Kallang/Whampoa and
Jurong West have rates 5.56% and 4.68%. These are overlapping label-derived
diagnostics, not held-out discoveries: most rows in every group are not low
events, so group membership alone cannot justify deletion or hard low-rent routing.

### Evidence, external-data boundary and synchronization

The existing `reports/feedback_results.csv` now holds 158 cumulative rows,
including 104 new formal score rows; none is a smoke result. Per-run predictions,
low-rent probabilities, OOF training row/source-origin records, monthly economic
features, grouped errors, configurations and original source snapshots stay in
ignored `artifacts/feedback_study/`. No new model artifacts were retained for
rejected candidates. Sources and raw/auxiliary hashes were verified, and every
non-reference saved prediction reproduces its ledger RMSE. The summary command
appends 1,000-resample address-cluster gain intervals to the SAME existing table.
It repairs per-model budget/device metadata: Ridge runs on CPU with zero trees,
targeted correction models use 600 trees; primary heads use 1,600. Original
run-local records/snapshots remain unchanged. Intervals are exploratory,
post-development, within overlapping windows; they are not independent tests or
future-market uncertainty bounds.

Official external-source investigation is recorded in `docs/KAGGLE_AUDIT.md`.
HDB open rental records now overlap the competition test period, but were NOT
downloaded/matched/used. Additional-data permission does not explicitly settle
direct test-period rental-label lookup; instructor clarification is still
required. CPI/URA macro sources were located, not trained; release timing and
revisions require a separate point-in-time audit. No conclusions about other
teams' methods or misconduct follow from public scores near 173.

Final QA: 50 automated tests pass, including source-fingerprint rejection,
metadata repair, train-only clipping and paired-interval sign checks. Mixed-
family GPU smoke succeeded before the formal run, and the final entry point
also received a guarded-Ridge smoke. Raw train/test and the two selected CSV
hashes remain unchanged. The 12 existing local final-model CBMs (including the
eight selected components and retained references) are protected by a before/
after hash check. README retains its single main results table; experiment
code, aggregate tables and cumulative documentation are the synchronization
scope, with data/models/submissions/artifacts and unrelated slide/plot edits
excluded.

Next priority is instructor clarification of the external-data boundary and
final-report preparation. A new predictive-information hypothesis should precede
another large model/weight search. Current mainline remains public 432.14881;
the retained stacking alternative remains public 432.04286.

### User-directed research closeout — 6 October 2026

The user has declined using public transaction rental labels overlapping the
competition test period. No such labels were imported or matched. This closes
that proposed route without requiring instructor clarification and without
attributing any method to other teams. CPI means an aggregate consumer price
index; the proposed URA series measures private-residential rents, not individual
HDB test rents. Neither optional series has been imported or trained.

Low-rent identification is a completed unsuccessful screen, not a requirement
to keep tuning. Wrong labels, special contracts and unobserved circumstances
remain hypotheses. Preserve the raw rows and current keep-label mainline.
Retain recent/long-horizon trade-off experiments for later reconsideration via
their existing code, settings, scores and local predictions; no rejected final
CBMs or new submission files are created. The user does not need an urgent
deadline check at this point. Keep report evidence accumulating in this log.

Read-only inspection confirmed the supplied MRT columns are CODE, NAME, STATUS,
LATITUDE, LONGITUDE, SUBZONE, PLANNING_AREA and REGION. There is no opening date.
The pipeline filters STATUS=open once and builds block-level station distances,
nearest names and radius counts. It does not rebuild these by rental month or
explicitly estimate opening-before/after rental effects. Static spatial
association must not be described as a measured causal opening-date effect.

Repository closeout inventory: before this documentation update, outstanding
tracked edits were .gitignore (+10 lines of temporary/LaTeX-intermediate rules),
reports/temporal_summary.csv (+16 bridge-window result rows), and
src/generate_figures.py (rent-band diagnostics and report-chart selection).
Temporal means time-based validation, not temporary data. These tracked source
and result changes are pending review/synchronization, not files covered by the
local-only data/model policy. Existing progress_report/ source/assets and the
output/pdf/CS5228_Progress_Report.pdf are also untracked; README's assertion that
they were already versioned was corrected. The report includes student IDs, so
its publication scope should be explicit. Local-only course data, trained
models, submission CSVs, detailed artifacts and AI notes remain local-only.

This closeout edits existing documentation only: no directory moves, deletions,
training, model/configuration changes, commits or pushes. The documentation edits
themselves remain uncommitted, alongside the pre-existing pending changes.

### Official CPI/URA and MRT service-time experiments — 6 October 2026

The user requested aggregate price/rental indices and MRT opening-time features,
followed by synchronization of maintained project files. External test-period
individual rental-label lookup and more low-rent classifier fitting remain out
of scope. No raw rental rows were removed.

The existing `feedback_experiments.py` entry point now supports `--families public`
and `--download-public`; reusable logic remains in `market_experiments.py`.
No extra Python entry point or numbered-stage directory was created.
`configs/mrt_openings.json` is the new shared configuration. Downloaded JSONs,
station audits, source snapshots and paired predictions remain ignored/local.
Exact official URLs, byte hashes and availability assumptions are in KAGGLE_AUDIT.

Inputs: CPI All Items (2024 base) and continuous URA Non-Landed / Whole Island
indices. The official All Residential series ends at 2021-Q2; do not treat it
as a complete current series or splice property types. Features include index
level, annual/short-period change, missingness and publication age. CPI activates
at observation month +2 and URA at quarter-end month +2. Latest historical
vintages are used; revisions and CPI rebasing are not reconstructed.

The MRT calendar verifies TEL2, TEL3, TEL4, Punggol Coast and Hume service and
announcement dates. Month-only rental dates use the first full month after
opening. Unknown/future codes are excluded; interchange rows are deduplicated
by physical station name. Legacy flags mean operational before the 2021 study,
not invented exact opening dates. Coverage is limited to supplied coordinates,
especially the incomplete LRT catalog. Fifty-five supplied `open` rows fall
outside the verified service calendar. The January 2025 catalog has 143 physical
names; filtering and deduplication change nearest distance by over 10 metres for
1,691/9,464 HDB blocks (17.87%). This combines two repairs, not two isolated
causal ablations. The unchanged mainline's MRT fields are now explicitly described
as supplied-snapshot proxies rather than certified operational service.

#### Origin-available screen

Run `878fed7edebe94c5`: seven variants, direct/residual targets, two windows,
seed 42, fixed 1,600 trees/depth 8: 28 GPU fits. Recent validation is Jan-Mar
2025 (141,076 train / 8,924 validation); long validation is Dec 2023-Mar 2025
(102,829 / 47,171). Original past-rent baselines and residual offsets stay fixed.
Macro values freeze at the validation origin; rail dates must have been announced
before the origin. The static rail control uses the origin snapshot even during
fitting. Dynamic rail reconstructs rental-month access; the event variant adds
distance-decayed new-line access, recent-line access and months since opening.
Line additions at existing interchanges can affect exposure without a new station.

Below each candidate replaces only the seed-42 direct/residual pair in the frozen
eight-component ensemble at unchanged 1/8 weights. Separate heads, pairs and
fixed 25% blends are retained in the existing ledger; blends are not equal-
capacity comparisons.

| Candidate | Recent RMSE | Sixteen-month RMSE |
|---|---:|---:|
| Current mainline | 440.7629 | 483.4345 |
| CPI | 440.8597 | 483.5572 |
| URA non-landed rental index | 440.8779 | 483.5280 |
| Both indices | 440.9309 | 483.7531 |
| Corrected static rail catalog | 440.6306 | 483.4176 |
| Rental-month rail access | 440.7152 | 483.5387 |
| Rail access plus opening exposure | 440.7603 | 483.3522 |
| Both indices plus opening exposure | 440.9139 | 483.6136 |

The largest recent replacement gain is 0.1322 for corrected static rail,
address-cluster interval [-0.0378, 0.2993]; its long gain is 0.0169, interval
[-0.1127, 0.1480]. Event replacement gains 0.0026/0.0823 with both intervals
crossing zero. Some fixed 25% blends gain up to 0.2556 long-window RMSE, but all
lose recently. None passes the meaningful promotion gate.

#### Same-model monthly-availability control

For dated cross-sectional transactions, separately test covariates released and
service operational by EACH rental month. Run `f0d15ed2879e3f3b` trains eight
GPU heads for joint indices and opening exposure across both windows. The SAME
fitted models predict both frozen and monthly inputs; no second model is fitted
for the query control. This is contemporaneous-covariate evaluation, not a
single-origin future forecast. Publication lags remain; validation rental labels
never enter features or fitting.

| Same-model replacement | Recent RMSE | Sixteen-month RMSE |
|---|---:|---:|
| Joint indices, origin frozen | 440.7983 | 483.7478 |
| Joint indices, monthly available | 440.7908 | 483.4784 |
| Opening exposure, origin available | 440.7400 | 483.3659 |
| Opening exposure, monthly available | 440.7491 | 483.4402 |

Monthly indices gain 0.2694 long-window replacement RMSE relative to their OWN
frozen predictions, interval [0.1826, 0.3563], but still lose 0.0439 to the mainline
and lose 0.0279 recently. Recent availability gain is 0.0075, interval [-0.0640,
0.0790]. Monthly rail inputs lose 0.0092/0.0743 relative to their own origin-
available models. No setting qualifies for promotion. This does not disprove
inflation or MRT effects in general, or establish a causal opening-date premium.

Repeated nominal seed-42 GPU fits vary: joint-index recent origin replacement
changes from 440.9309 to 440.7983. Rebuilt training/frozen-query matrices from
BOTH original source snapshots have identical hashes; fitting settings, seed,
raw/auxiliary and external-data hashes also match. Changes around 0.1 require
caution. CatBoost documents non-deterministic GPU summation:
https://catboost.ai/docs/en/features/training-on-gpu . Saved predictions reproduce
all recorded metrics; bit-identical GPU refitting is not promised. Same-model
query controls isolate availability without refitting noise. Intervals remain
exploratory evidence in overlapping development windows, not independent tests.

#### Verification and synchronization scope

Total: 36 formal GPU fits, 2,262.12 seconds of per-fit training/prediction time
plus 85.88 seconds of common feature building. Family runtimes also include
extra rail construction/control prediction; these are not total wall-clock times.
Successful GPU smoke runs are excluded from the aggregate ledger. The existing
feedback table now has 284 cumulative score rows, 126 new (references/pairs/blends
are score rows, not new fits). No duplicate run/window/method keys, nonfinite core
metrics, chronological violations or smoke rows were found. Fifty-five tests and
the environment/data setup check pass. Both runs' original source/data/external
snapshots and saved predictions were verified. Dedicated `availability_gain`
columns record paired same-model query-control effects. Source snapshots preserve
the actual fitting code when reporting helpers are added afterward.

No second-seed promotion study, final retraining or new submission was justified.
Raw train/test, existing model components and retained submission files remain
protected by hash checks. User-reported mainline public score remains 432.14881;
stacking alternative remains 432.04286. Trade-off candidates retain evidence,
without new final CBMs.

Synchronization includes pending .gitignore, 16 bridge-window temporal-summary
rows, report plotting code, maintained docs, new research code/config/results,
and the existing September report source/assets/PDF. All 16 pending rows reproduce
from saved component predictions (maximum RMSE difference 5.7e-14). Plot generation
succeeded into separate ignored `artifacts/plot_check/` (five figures), preserving
archived report assets. The report has 14 frames and all six graphics resolve.
It stays the September checkpoint; October evidence belongs in this cumulative
log for the final report. No moves or deletions occurred. Course data, economic
JSONs, models, submission CSVs, detailed artifacts, build logs and local AI notes
remain excluded from GitHub.

## Final scientific-report draft -- 23 October 2026

The user authorised a complete first draft, requested all generated project
materials remain in the unique D-drive workspace, and clarified that local
evaluation should lead the narrative. Workload attribution belongs in its
eventual appendix position but must remain unassigned until the team provides it.

### Requirements verified from Canvas

The Chrome course page, project page, Policies, 8 October feedback announcement
and Assignments list were read through the supported browser interface. Canvas
was not edited and nothing was posted or submitted. Direct sources:

- Project: https://canvas.nus.edu.sg/courses/93866/pages/project
- Policies: https://canvas.nus.edu.sg/courses/93866/pages/policies
- Feedback: https://canvas.nus.edu.sg/courses/93866/discussion_topics/501350

The final deliverable is a scientific-paper PDF, at most eight main-text pages
including figures/tables, excluding references and a sparingly used appendix.
The body must stand alone. Names, student IDs and team name are required;
workload attribution may be placed in the appendix. PDF plus code is uploaded
to Canvas as a ZIP, with a GitHub link allowed instead of packaged source.
Methodology/report/reproducibility weights remain 60/30/10. No final presentation
is required. The final date remains a placeholder (`Xxx, Nov XX, 11.59 pm`), and
no Final Report entry was visible in Assignments. No deadline was invented.
The Policies page explicitly permits group-project AI use with documentation;
the paper contains a factual assistance statement, not a new uploaded AI log.

### Template, writing guidance and structure

Downloaded the official Canvas file 9565679 via its Download templates button.
Its SHA-256 matches the previous course archive exactly:
`ab2974691f767969dd6b2b1de18217ccaa7829f4439c5232a3fbbaa03bbfda54`.
The ZIP and downloaded example report remain under ignored `templates/`.
The official IEEEtran conference class is copied unchanged into `final_report/`.
The unmodified example compiled in the built-in LaTeX editor before authoring.

Searched public writing skills and applied the repository-to-paper, citation
verification and writing guidance from KalarisLabs' MIT-licensed ml-paper-writing:
https://github.com/KalarisLabs/research-agent-skills/tree/main/skills/ml-paper-writing
Read its SKILL.md and the required repository, writing, template and checklist
references. The user's approved course-report scope overrides conference-specific
anonymisation, template and intermediate-approval defaults. No new plugin or
global skill installation, model fitting or unrequested external account access
was required. Guidance influenced claim/evidence organisation and citation checks;
all scientific results come from this project, not the skill's example facts.

Two public previous-course reports informed structure and concise interpretation:
https://ltdk.me/assets/docs/CS5228.pdf and
https://github.com/ParasharaRamesh/Home-sweet-home/blob/main/report.pdf .
Their editions/data/validation differ. No text, experiment numbers, code or
unjustified duplicate-removal rule was adopted as current-project evidence.

The single evolving source is `final_report/main.tex`, with these sections:
Introduction; Data Understanding and Preparation; Methods; Experimental Design;
Results and Interpretation; Limitations and Reproducibility; Conclusion.
Appendix A maps reproducibility and baseline settings, Appendix B reserves
workload attribution, and Appendix C documents AI assistance/draft status.
Names/IDs match the archived progress-report source in surname-first order.
Group 44 is retained as the team identifier; no new team nickname is invented.

### Evidence and maintenance

`final_report/prepare_report.py` reads existing source tables and verified frozen
predictions, produces three vector figures and five aggregate numeric tables,
and writes `generated/evidence.json`. No training is performed. The recent
mainline RMSE/MAE/bias recompute as 440.762863/316.549083/+0.787674. All 8,924
validation rows remain. The low actual-rent band contributes 52.2320% of SSE.
The raw file audit, not the course overview, establishes train Jan 2021-Mar 2025
and test Mar 2025-Jul 2026, including the observable March overlap. Historical
early-stopping results are separated from fixed-budget comparisons. The older
annual regression, overlapping development windows, one-seed screens, revised
macro vintages and static mainline MRT limitations remain explicit. Recorded
public scores are secondary and are not described as newly checked ranks.

Research BibTeX entries were retrieved from NeurIPS and Crossref, with titles,
authors and publication details checked against publication/arXiv/author sources.
Official course, CPI, URA and LTA sources support the corresponding references.
No fabricated citation or unverified experimental result is used.

`final_report/build.ps1` compiles the stable
`output/pdf/CS5228_Final_Report.pdf` using existing pdflatex/BibTeX. Local latexmk
cannot run without Perl, so no new installation was attempted; the direct build
and built-in editor compiler both succeed. Build artifacts and page previews
remain under ignored `artifacts/`. The paper has eight total pages: main text
ends on page seven, with references on that page and appendices on page eight.
All pages were rendered for visual review; remaining intentional placeholders
are the workload entries and team review status. Fifty-five existing tests pass.
The selected model, raw data and competition submissions are unchanged.

Report source/class, references, preparation/build commands, derived aggregate
assets and draft PDF are in scope for GitHub maintenance, alongside this
cumulative entry, README and the requirements audit. Course files, example PDFs,
detailed predictions, compiler intermediates and local AI notes remain excluded.
Next work is team review and eventual workload completion, followed by edits to
this same paper as new project evidence becomes available; this draft has not
been submitted to Canvas.

## Final paper review revision -- 23 October 2026

Applied the user's instruction to select from the 41-point review and improve
the existing paper. No new model fitting, dataset edits or promotion occurred.
The selected local RMSE/MAE/bias remain 440.762863/316.549083/+0.787674.

Review disposition (numbers refer to the review in this conversation):

| Points | Decision and implementation |
|---|---|
| 1-4 | Adopt research-decision framing and course page counting; retain truthful workload placeholder and Group 44 identifier. |
| 5-12 | Fix clipped axis title; use installed Times New Roman at 8pt for plot labels/legends; grouped bars replace non-temporal connected lines; use labelled panels, thousand counts and a protected legend; retain subdued colours and official 10pt/Letter/two-column class. Show page numbers consistently, remove policy footnote in favour of a verified bibliography entry, shorten captions and remove the appendix's broken template-description sentence. |
| 13-19 | Shorten abstract; answer RQ1-3 explicitly; add research-decision table and an eight-component TikZ diagram; define reference/diversity/trend/stacking candidates and seed-averaged heads; distinguish empirical budgets from pragmatic smoothing choices. Preserve the professor's individual-flat identity caveat. |
| 20-25 | Explain task-dependent model selection and retained robustness reference; discuss older-year MAE 455.20 to 454.38 despite RMSE regression; distinguish historical early stopping from fixed-budget comparisons, acknowledge total-compute differences, and add selected-model reference plus B/R/C application modes to research table. Negative screens are interpreted as bounded hypotheses rather than universal algorithm rejection. |
| 26-30 | Recompute observable-flat-type counts/RMSE/bias/SSE shares from the same frozen predictions; qualify the tiny one-room sample and target-conditioned bias interpretation; distinguish rail semantic correction from score gains; retain development-window reuse disclosure and explain stopping. No new causal or irreducible-error claim. |
| 31-32 | Compress Kaggle to a supplementary selected-versus-stacking observation and short outlier sensitivity note; remove historical score procession and other-team speculation. Clarify that inputs are visible but labels/public membership are hidden. |
| 33-39 | Preserve existing directories, add final-report paths and README navigation; clarify historical refresh script's docstring/map label; document standalone compilation versus cache regeneration, GPU variation and protected source manifests. Workload/team review and teacher access to the private repository remain pre-submission tasks, not invented completed checks. |
| 40-41 | Retain previously reviewed public reports/ML-writing guidance as presentation references; use course rules over conference defaults. No unverified citation expansion, global skill install or new experiment was needed. |

The revised PDF has seven body pages, a separate balanced reference page and one
appendix page (nine total); the course limit is eight body pages excluding the
last two categories. Main source/class and stable PDF path are retained. New
aggregate flat_type_evidence.csv and flat_types.tex live in the existing generated
directory. Figures/tables regenerate only from verified retained predictions;
course data, CBMs, submission CSVs and detailed artifacts remain local.

Validation: report assets regenerate successfully; the protected mainline metric
and data hashes match the original evidence. Direct LaTeX/BibTeX compilation
succeeds, with no overfull boxes or unresolved references. Page images and embedded
fonts are reviewed after the final layout pass. Team workload remains intentionally
unfilled; this revision is not a Canvas submission or clean-machine GPU retrain.

## Focused final-paper evidence review -- 23 October 2026

Reviewed the fourteen follow-up suggestions and applied the useful changes to
the existing report. This is a report clarification using retained experiment
evidence; no training, promotion or new competition submission was performed.

| Review points | Disposition |
|---|---|
| 1 | Retain the research narrative and existing structure; no wholesale rewrite. |
| 2 | Add selected-versus-adaptive RMSE gain 4.76 with recorded 95% paired cluster interval [3.64, 5.87], and older-year gain -9.81 with interval [-11.08, -8.59]. Existing source metrics, 1,000 resamples and development-window caveats remain explicit. |
| 3 | Add each executed fit's training cutoff to the temporal table; state January 2021 training start and fixed models/rental-label histories throughout scoring. No validation rent is assimilated. Cutoffs derive from temporal_comparison.csv. |
| 4 | Identify the depth-4, 600-tree CatBoost low-event detector, learning rate .04 and L2 20; document 35,239 Jan-Dec 2024 earlier-origin OOF rows, observable/base-prediction inputs and Jan-Mar 2025 evaluation. Distinguish it from the rent-band routing classifier. |
| 5 | Define the 64-neighbour same-type comparable method, recency weighting and reported constant-market variant; define specialist groups/fallback and the three rent-band experts with predicted routing. Preserve the fixed 25% application rule. |
| 6 | Clarify that retaining the supplied MRT snapshot preserves the recorded offline recipe; it does not establish historical operational availability. Corrected-calendar results remain separate. |
| 7 | Explain acceptance of eight rather than four components for recent/task-period gains, retaining the simpler reference and the older-window failure. No equal-compute or universal-dominance claim. |
| 8 | Define month indices, horizon, market baseline/slope, floor-area units and 0.1 sqm comparable keys against the implementation. |
| 9-10 | Replace internal mainline/gate/screen wording in the paper; compress repeated framing and caveats while preserving limitations, methodological distinctions and negative evidence. The acceptance threshold is explicitly operational, not a significance level. |
| 11 | Move the temporal table earlier in the source and use a top float; it now appears on page 5 beside the principal results instead of page 6. Keep legible standard table fonts. |
| 12 | Format row counts with separators and round signed zero bias to 0.0. Retain additional RMSE digits only for the small-gain research comparisons and recorded competition values. |
| 13 | Keep the unchanged official class, fonts, subdued colours, separate references and seven body pages. No padding or type-size reduction. |
| 14 | Keep workload attribution and team-review status pending, as instructed. Teacher access to a private repository remains a submission-time check, not an asserted completed action. |

An additional implementation check corrected the earlier description of the
main ensemble bootstrap: generate_figures.py groups the recorded raw
town/block/street strings, not canonicalised addresses. The paper now discloses
that formatting variants can split one building across keys. Recorded intervals
are preserved rather than relabelled as a new bootstrap experiment.

The existing aggregate evidence.json now records the four selected-versus-adaptive
gain intervals and the additional cutoff-source table. Numerical table generation
handles signed zero and count formatting; no new experiment/report directories
were introduced. README describes the clarification. The draft remains seven
body pages, one reference page and one appendix page (nine PDF pages total).

Validation: the asset generator rechecks original raw-data and frozen-prediction
hashes, all 8,924 scoring rows and RMSE/MAE/bias
440.762863/316.549083/+0.787674. Direct pdflatex/BibTeX compilation succeeds;
rendered pages are inspected for clipping, placement and readability, with no
overfull boxes or unresolved references. Fonts, model configuration, training
code, original course data and competition CSVs retain their previous scope.
Workload completion, actual team approval and course submission remain pending.

Follow-up wording correction (23 October 2026): explicitly map the local-trend
increment to each validation window: recent quarter 0.13, recent year 0.13,
sixteen months 0.40 and older year 0.74 RMSE. This removes the ambiguous
"respectively" ordering; recorded metrics, tables and selected predictions do
not change. The same PDF is rebuilt and its affected page and page count checked.
Workload attribution remains reserved.

## Pre-submission acceptance -- 29 October 2026

Completed the user-authorised A-category closeout. Canvas Overview lists Friday
6 November 2026, 23:59 Singapore time, despite the Project page placeholder.
Source: https://canvas.nus.edu.sg/courses/93866/pages/overview . README and the
current plan now use that deadline; historical uncertainty entries are retained.

Corrected the primary reproduction sequence to include local_trend_direct and
local_trend_residual alongside the four reference/diversity candidates. Removed
the historical-only progress refresh from this current-result sequence.

Independent local clone under ignored tmp/submission_acceptance, with a new
Python 3.12 virtual environment and requirements installed afresh: installation,
pip check, all 55 tests and all nine data/checksum/schema checks pass. The clone
starts without original experiment caches. Its five-tree local-trend GPU
validation run scores RMSE 698.938768, MAE 560.424343 on 8,924 rows; these are
smoke scores, not model-selection evidence. Its full-input eight-component
five-tree GPU submission workflow completes in 126.63 seconds, including model
save/reload checks and CSV validation for all 50,000 ordered IDs. Clone PDF
compilation succeeds without prediction caches. Full-budget historical runs and
another physical machine were not retested. An initial check launched before
installation finished failed on missing packages; the post-install checks pass.
Local environment snapshot and all smoke outputs remain inside the ignored clone.

Added final_report/check_statistics.py and aggregate
reports/pre_submission_checks.json for reproducible canonical-address bootstrap
sensitivity on the frozen predictions. At 1,000 resamples with seed 42, intervals
are recent quarter [3.6216,5.8467], recent year [2.1484,3.4801], older year
[-11.1226,-8.5921], sixteen months [2.5390,3.8435]. Gain directions and selection
conclusions are unchanged; canonical keys merge formatting variants. The paper
retains the original intervals and adds this sensitivity check explicitly.

All eight production CBM hashes and the final submission hash match metadata.
No model promotion or production submission change. Review retains older-window
regression, development-window reuse, differing experimental budgets, weak
low-rent detection and temporal availability limitations. Workload, team approval
and teacher repository access remain pending user/team decisions. The paper
continues to use seven body pages plus references and appendix; no Canvas upload.

## Full-budget reproduction and targeted follow-up -- 29 October 2026

Completed the user-requested selected-model full-budget repeat and bounded
follow-up research. No production model/submission promotion occurred.

### Reproduction

The independent local clone/new Python 3.12 environment now repeats all eight
selected components at 1,600 trees on 150,000 training rows. Fit takes 679.83 s;
all CBM hashes/reloads and 50,000 ordered finite CSV predictions pass. Eight
fresh recent-window fits at the same budget give RMSE 440.642256, MAE 316.123729,
bias +2.599434, versus frozen RMSE 440.762863. This is refit variation, not a
feature gain. Test predictions differ by mean absolute 4.397426 SGD and maximum
34.836542 SGD; exact old submission bytes remain protected. Aggregate evidence:
reports/full_budget_reproduction.json. Full repeat covers selected all-data
training and recent validation, not every historical experiment or another
physical computer. The user will arrange cross-device checking.

### Transfer and low-rent diagnostics

src/followup_analysis.py is the single new research entry point. Frozen paired
predictions give older-year normal-view RMSE 579.845889/bias +7.077102 versus
stale-view 624.076255/+158.353556. All six canonical types deteriorate versus
the adaptive reference (one-room has only eight rows); 94.41% of excess squared
error falls in August 2023-March 2024. Recent stale-view RMSE is instead
441.783033 versus normal-view 445.203240. This localises prediction behaviour,
not an isolated causal augmentation effect because view features also differ.

A predeclared half-bias correction learned from earlier OOF origins worsens
older-year RMSE to 596.270758 and recent RMSE to 440.774835. Earlier bias has the
opposite sign in the old window, so constant inherited calibration is inadequate.
Town/type low-event rates use earlier OOF labels only and shrink with 30 prior
observations. Recent AUC/AP are .513370/.030949 versus prevalence .029919;
older-year AUC is .517631. This offers no useful stable routing rule, and does
not prove transactions are wrong or all such events inherently unpredictable.
Aggregates: followup_transfer.csv, followup_groups.csv, followup_low_rent.csv.

### Operational interchange access

Tested nearest operating interchange distance and distinct supplied line-code
prefixes within 1/2 km, using the existing audited calendar and announcements
known before each validation origin. Original static covariates remain; these
three additions are not track routes or actual commuting times. Six seed-42
1,600-tree GPU fits (direct/residual in recent, older and sixteen-month windows)
test corresponding component replacement and a fixed 25% pair blend.

Replacement gains: -.083768/-.558694/-.117775 RMSE; blend gains:
-.654862/+4.937799/+.099853 in the same window order. The older blend still
loses to the adaptive reference. No candidate passes the recent/long-window
promotion criterion; no second-seed expansion or new Kaggle CSV is warranted.
Aggregate provenance, timing and paired intervals: reports/followup_transit.csv.
Detailed predictions/features stay under ignored artifacts/followup.

Pre-fit review caught MONTHLY_RENT retained in the draft feature-cache input;
it was removed before any transit fit, label-free assertions added, and the
three discarded draft caches deleted. Formal inputs have 81 columns (78+3).
No leaky model/result was fitted or used. Repository pytest initially scanned
the ignored local clone too; pytest.ini now limits discovery to maintained
tests. Two new meaningful risk/access tests bring the suite to 57 passing.

### Report and remaining scope

The same paper now records these positive/negative findings and actual full-
budget reproduction, replacing the obsolete unestablished-retraining wording.
Original headline metrics stay frozen and the result cutoff is 29 October.
This follow-up comprises 22 full-budget GPU fits: eight full-data, eight recent
validation and six access fits. It provides substantive post-progress-report
analysis without manufacturing an improvement. Workload/names/team review,
teacher access and actual submission are user/team tasks. Exact commuting
features need routing data; truly new-period validation needs new permitted
labels. Their implementation and robust regime-adaptive modeling remain open.

Final typesetting: eight body pages, one reference page and one appendix page; ten total. All pages rendered and checked; original production hashes and all nine data checks remain valid.

## 2 November 2026 - Reconciled Claude/Codex experiments and report update

This entry supersedes the earlier report cutoff and the original handoff's
whole-project submission status. It preserves the historical mainline and its
recorded metrics. The consolidated Chinese explanation and artifact ownership
are in `docs/RESEARCH_NOTES.md` and the audited
`docs/RESEARCH_NOTES.md`.

### Executed evidence

- Claude's supplementary 91-feature study has 13 result rows across three windows,
  including two simple baselines, three reused-prediction blends and one
  target-visible diagnostic. It is not a theoretical ceiling experiment.
  The seven fitted configurations use seed42 and 1600 trees per regressor.
  All 39 aggregate rows are in `reports/november_simplified_20261102.csv`.
- Natural-log targets with training-set Duan smearing reduce sixteen-month bias
  from +85.21 to +27.98 in the supplementary recipe, but not recent RMSE.
  Integrated with the main feature workflow, the six-component mean scores
  442.1204/467.2721/583.5154 across recent/recent-year/older-year windows,
  versus the distinct four-component CPU reference
  441.8429/467.3043/585.8437. It fails the declared all-window rule.
- Codex's initial CPU comparisons, LightGBM variants and embedding MLP do not
  justify replacing the mainline. A fixed relative-LightGBM blend reaches
  441.5953 locally but fails the advancement rule. Budget and preprocessing
  differences prevent interpreting these as a universal learner ranking.
- Seventeen lagged resale, sixteen current building-stock and nineteen
  contemporaneous supplied-macro features yield 441.8273, 441.7830 and 441.9767
  against CPU reference 441.8429. Snapshot and release assumptions remain explicit.
- The strict-past-reference low-rent event is distinct from earlier event
  definitions: 3.160% of rows contribute 32.598% of reference squared error.
  Adding housing data changes AUC 0.65761 to 0.65835; a relative two-expert mixture
  and its fixed 25% blend score 453.993 and 442.223, not an improvement.
- Five additional CatBoost fits test duplicate-mean compression, a market-
  normalized target, 4357 attribute-matched historical-only source additions,
  deeper symmetric trees and asymmetric trees. Their selected composition
  scores 441.7347/441.6300/441.8761/441.7568/442.2445. None passes the recorded gate.
  Exact compositions and cluster intervals are in
  `reports/november_cpu_comparisons_20261102.csv`.

### Actual public probes and decisions

The user separately requested two submissions despite the local screen not
passing. The four-head seed42 CPU building-stock probe scores 432.70689;
the full-data market-normalized/damped6 probe scores 434.72053. Both are verified
Complete on the signed-in Kaggle page; hashes and descriptions are in
`reports/kaggle_submissions.csv`. Neither changes the best 432.04286,
the selected 432.14881 recipe, or final-submission selection. Claude's abandoned
eight-head stock prototype produced no submission and is not the executed
four-head stock script.

The separately saved public-source diagnostic is target-visible and was not
used for training/tuning or uploaded. The instructor's 30 October clarification
restricts publicly matched test rents to final evaluation. Source availability
does not identify competitors' methods. The diagnostic's near189 proxy retains
its evaluated row's source rent and is not out-of-sample prediction evidence.
The script now requires explicit diagnostic-only opt-in; its CSV remains ignored.

No 435--440 irreducible-error bound is established. Coarse scoring-label group
medians, repeated-input dispersion and deleting low scoring rows do not prove
such a bound. Course attributes are compatible with coarse building/type joins,
but the exact generation rule and per-lease room/unit attributes remain unknown.

### Integration, validation and remaining team action

The original final-report source and PDF now include November findings; complete
numeric tables are in its appendix, with eight main-text pages retained.
`final_report/prepare_november.py` exports only aggregates from saved evidence;
`final_report/build.sh` adds a macOS/Linux compilation path without fitting.
The November suite passes 74 checks, with one optional example-template test
skipped because the official ZIP omits that file. Loss-equivalence, log-target
output units and exclusion of future source targets are checked. No fabricated
example CSV is substituted for the checksum-locked course original.

Code, process documents, aggregate results and the updated PDF are shared;
raw records, per-row predictions, fitted weights, source-lookup CSVs and local
logs remain excluded. Human responsibilities and percentages must still be
confirmed by the team. This integration is not a claim of team approval or
Canvas submission.

## 3 November 2026 - Research structure consolidation

Integrated remote commit 72a8620 by fast-forward after confirming a clean local
checkout. No experiment was rerun or promoted. Moved the twenty dated research
scripts into semantic modules: nineteen under src/research and the target-visible
diagnostic under src/audits. Production entry points/configuration remain unchanged.
Removed three machine-specific shell wrappers; supported invocation is now
python -m src.research.<module> from the repository root. Seven overlapping
handoffs/study notes are preserved together in docs/RESEARCH_NOTES.md, with the
reconciled summary first. Historical aggregate tables, hashes and artifact IDs
retain their original bytes and dates for traceability.

LightGBM is lazily imported by the common comparison helper; only its four
specific smoke tests skip in the historical environment. Added relocation and
default-denial checks. The abandoned eight-component prototype is explicitly
guarded, and the quarantined public-source audit requires an explicit input path
instead of a personal Desktop path. Neither is executed by normal imports/tests.
Research cache-dependent probes are documented separately from fresh-clone entry
points; historical macOS commands in the retained notes are not current guidance.

Paper decision: retain concise evidence on alternative learners/log targets,
lagged resale/building attributes, and low-rent identifiability. Do not present
target-visible lookup as a predictive result or an error floor. No paper text,
tables, PDF, model, raw data or submission changed during this structural cleanup.
The remote paper remains 8 body + 1 reference + 4 appendix pages; compression
and final layout review are deliberately the next task, not claimed complete.

Validation after relocation: 74 tests pass; four optional LightGBM tests skip in
the unchanged historical environment. Both research/audit packages compile;
the comparison CLI help and data-audit mode run successfully. All nine required
data/schema/hash checks pass, and all eight production CBMs plus the selected
submission match the existing metadata. No new training or public submission.

## 3 November 2026 - Integrate follow-up evidence into the original paper

Reworked the same final_report/main.tex instead of appending another study.
Retained attribute-identity evidence in data understanding, housing covariate and
log-target definitions in methods, the separate CPU reference/progression rules
in experimental design, and the new risk-detector result in error analysis.
A compact three-window log-target table shows the recent/older trade-off;
learner and external-attribute results are concise negative evidence in results.
CPU four-component and GPU eight-component numbers remain explicitly separate.
Public probes remain supplementary; target-visible lookup is only a boundary
statement, not a predictive result or an explanation of competitors' scores.

Removed the unreferenced chronological appendix and its three generated table
fragments; they remain recoverable in Git. All aggregate CSV/JSON evidence is
unchanged. prepare_november.py still exports historical aggregates from private
caches, but no longer regenerates unused appendix tables. Workload is intentionally
unfilled. No model, configuration, raw data, or submission was changed.

The revised paper has 8 body pages + 1 references + 1 appendix (10 total), using
the unchanged official IEEEtran class. Full PDF rendering and visual review cover
all pages. The full-budget reproduction statement remains scoped to the selected
recipe and the same physical computer; cross-device verification is a team task.
