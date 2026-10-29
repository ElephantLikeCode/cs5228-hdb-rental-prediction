# CS5228 HDB Rental Prediction

Private group-project repository for predicting Singapore HDB monthly rent.
The evaluation metric is RMSE, and chronological validation is the primary
local evaluation method.

Quick navigation: [setup](#setup-from-a-fresh-clone),
[run the selected model](#reproduce-the-current-result),
[repository map](#repository-structure),
[final paper and build instructions](#final-report-working-draft),
and [cumulative research decisions](docs/PROGRESS.md).

## Current mainline and retained reference

The mainline is the **local-trend/history-diversity CatBoost ensemble**: a
normal-history view with four past-only town/flat-type momentum features and a
six-month stale hierarchical-history view, each with direct-rent and
market-residual targets and seeds 42/2026 (eight equally weighted components).
At inference all components receive the full permitted training history. It
retains all labels.

The pipeline includes canonical category cleanup, core/spatial features,
strictly past-only comparable rents, adaptive market estimates, rolling
validation, baseline/feature ablations, grouped errors, EDA and a validated
50,000-row submission generator. Automated tests cover data contracts,
feature/validation behavior and output protection.

### Model comparison — January-March 2025 validation

This is the single main results table. Update it when the selected pipeline
changes; retain previous iteration results in `docs/PROGRESS.md`.

| Model | Validation RMSE | Evaluation |
|---|---:|---|
| Global median | 844.60 | Historical reference |
| Town + flat-type median | 546.81 | Historical reference |
| HistGradientBoosting | 463.36 | Historical reference |
| CatBoost core | 455.71 | Historical reference |
| CatBoost + spatial and past-rent features (seed 42) | 450.04 | Historical reference |
| Retained reference: adaptive-market direct/residual ensemble (four components) | 445.52 | Fixed 1,600 trees/component |
| Retained reference: normal/stale-history ensemble (eight components) | 440.89 | Fixed 1,600 trees/component; public RMSE 432.47187 |
| **Current mainline: local-trend/stale-history ensemble (eight components)** | **440.76** | **Fixed 1,600 trees/component; public RMSE 432.14881** |
| Public-best candidate: fixed recent-window four-head stacking | 440.66 | Same eight CBMs; public RMSE 432.04286; not promoted |

The final three rows use identical validation rows and seeds, with no scoring-label
early stopping. Historical baselines used a different early-stopping protocol
and do not measure this update's gain. The eight-component model costs more.

The retained normal/stale reference improves the recent three-month, recent 12-month and 16-month
windows by 4.63, 2.65 and 2.81 RMSE, but **worsens Apr 2023-Mar 2024 by 10.55**,
with all 12 months worsening and systematic overprediction. The windows overlap;
they are development evidence, not independent tests or a private-score guarantee.
Full results remain in `reports/temporal_summary.csv` and `docs/PROGRESS.md`.

The 6 October feedback-driven GPU screen tested five predictors and their fixed
25% blends on the recent quarter and sixteen-month window. None improved both
windows, so no mainline replacement or new submission was generated. Exact
RMSE/MAE/bias, cutoffs, hashes and incremental runtimes are recorded in
`reports/feedback_results.csv`; definitions and subgroup insights are appended
to `docs/PROGRESS.md`. The specialists/mixture use one screening seed, compared
with the frozen eight-component reference; this is a deployment-candidate screen,
not an equal-capacity causal comparison. The mixture classifier has 1,000 trees;
its three regression experts each have 1,600. Cached-reference fit time is zero
in the ledger because it was reused; blend times exclude original mainline fitting.

The current mainline replaces only the normal-history half with four extra
town/flat-type momentum features. Relative to the retained reference it improves
the same four windows by 0.13, 0.13, 0.74 and 0.40 RMSE. Two additional
three-month cutoffs improve the trend head by 0.17 and 1.47 over its matched
reference. The older-year deficit is reduced but remains 9.81 RMSE, so temporal
regime sensitivity remains unresolved despite the public-score improvement.

The subsequent isolated composition-trend ablation preserved every historical
baseline and residual offset. Relative to the retained candidate, its final
eight-component combination improved Jan-Mar 2025 by only 0.002 RMSE and worsened
the recent-year, older-year and 16-month windows by 0.154, 0.288 and 0.341.
It was not adopted. The earlier joint composition experiment and this isolated
experiment remain separate evidence; neither rejects all composition-control
methods. The bounded current-feature XGBoost screen also failed, including its
pre-specified cross-learner blends. The subsequent time-transferred stacking study
now compares four seed-averaged heads with all eight model outputs directly.
Across six matched-horizon forward tests, learned weights were not stable: the
best eight-input method gained 7.21 RMSE in the first annual transfer but lost
0.89 in the next, and lost 0.56/0.42 on the final two quarterly transfers. A
follow-up using only the latest meta-training window reduced some losses but did
not make the result stable: its four-head gain on Jan-Mar 2025 was only 0.106
RMSE with an interval crossing zero, while its eight-input result was -0.054.
Fixed 12-month time decay also lost on the latest quarter. In a 16-month forward
stress check, weights fitted on Apr 2022-Mar 2023 lost 0.538 RMSE with four heads
and 1.189 with eight inputs. This check changes the historical regime, fitting
window length and prediction horizon together; it is long-horizon risk evidence,
not a clean rejection of quarterly recency weighting. At the user's request,
`submission_recent_window_stacking.csv` was generated as a fixed-weight probe.
Its weights were fitted on Oct-Dec 2024 and were not refitted on Jan-Mar 2025.
The user reported public RMSE **432.04286** on 24 September, improving the
mainline by **0.10595**. This is the best recorded public candidate, with SHA-256
`05bf5b69408b0b8d61a35cc54590277ee48e41cb8c30a371a594a6f360ce9cff`.
The result supports testing fixed head weighting but does not establish private-
score or cross-regime improvement. The default mainline remains 432.14881.

The user-reported public score on 24 September 2026 is **432.14881**, improving
the prior 432.47187 mainline by **0.32306**. The uploaded filename was
`submission_local_trend_diversity.csv`. Its exact bytes are now also the mainline
`submissions/final_submission.csv`; SHA-256:
`a88da3be518a4793486765f5d9ff44c9b27b3248bb195e6f2d0264ae7dabb57a`.
Promotion copied the submitted CSV and eight verified CBMs, without retraining.
The score is recorded as user-reported; no current rank was supplied.
The former four-component pipeline is retained locally as a reproducible reference,
not discarded. The Keep/Remove submissions also remain unchanged.

Public scores use approximately 50% of test rows; private evaluation uses the
other half, with no verified random/temporal split rule. Up to two final
submissions may be selected; the two slots do not require two separate codebases.
No final-submission selection was changed by this agent. Exact submitted scores are in
`reports/kaggle_submissions.csv`.

The official Kaggle ZIP was checked on 18 September 2026; its eight CSVs match
the existing raw files byte-for-byte. Raw data is never modified.
Spatial amenities use the supplied snapshot (MRT status `open`), not historical
opening dates per lease. This prediction-time availability limitation must be
disclosed in the report; target-history leakage tests do not resolve it.
The 6 October opening-calendar audit also found 55 supplied `open` rows outside
the verified service calendar (future or unverified station-line entries).
The mainline's MRT fields are therefore supplied-snapshot proxies, not certified
historically operational service. Research variants reconstruct rental-month
access and deduplicate interchange entries by station name.

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

The sequence below includes all six component candidates needed for the selected
ensemble and its four-component reference. Full-size runs are expensive; for an
isolated end-to-end smoke test use a separate checkout and run
`python src/train_final_model.py --task-type GPU --iterations 5 --model-prefix acceptance_smoke --submission submissions/acceptance_smoke.csv --metadata reports/acceptance_smoke_metadata.json`.
This exercises all eight configured components and output validation, but its
scores are not evidence of final model quality.

Run from the repository root:

```bash
python src/data_audit.py
python -m pytest -q
python src/generate_eda.py
python src/run_experiments.py --compare-candidates --candidates hierarchical market_residual horizon_direct horizon_residual local_trend_direct local_trend_residual --windows short_recent long_recent long_older horizon16 --comparison-seeds 42 2026 --trial-iterations 1600 --task-type GPU --resume
python src/generate_figures.py --temporal-only --reference-candidate hierarchical_market_blend --bootstrap --error-groups
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
| `weighted_direct`, `weighted_residual` | Hierarchical features; 18-month half-life training weights |
| `horizon_direct`, `horizon_residual` | Hierarchical features; add a six-month stale-history training view, preserving each observation's total weight |
| `relative_rent` | Learn a market-normalized residual; squared-scale weights preserve the rent-MSE objective |
| `local_trend_direct`, `local_trend_residual` | Add smoothed town/flat-type relative-rent changes between the latest and preceding six months |
| `composition_trend_direct`, `composition_trend_residual` | Jointly change trend and historical baselines using area/lease-age cells; rejected combined experiment, not an isolated ablation |
| `composition_trend_only_direct`, `composition_trend_only_residual` | Change only trend columns using composition cells; preserve original historical levels and residual targets exactly |
| `xgb_trend_depth6_*`, `xgb_trend_depth8_*` | GPU XGBoost direct/residual comparison with the identical local-trend features, training-only category vocabularies and fixed tree budget |

The retained `hierarchical_market_blend` reference averages hierarchical and
market-residual predictions 50:50, then averages seeds. It adds 18 adaptive
features to the existing 56; category cleanup and raw data remain unchanged.

The selected `local_trend_history_diversity_blend` in `configs/model.json`
averages the local-trend normal-history head and the stale hierarchical-history
head 50:50 (eight equal-weight components). Its weights were inherited from the
validated history-diversity design rather than optimized over a weight grid.
At inference every component receives all available permitted training history;
the latest six months are hidden only in the extra training view.
Promotion followed the public result and review of the old-year regression;
that limitation remains open. Recent-sample weighting and the relative-rent
learner failed their stability screens and are not part of the mainline.

Adaptive market forecasts use only prior months, cap monthly log-growth at
plus/minus 1%, and damp the extrapolation horizon toward six months. This is a
bounded hypothesis, not an assumption that rents must increase. Training-label
removal is not part of these temporal comparisons; both previously submitted
Keep/Remove files are retained locally.

Every comparison is appended to `reports/temporal_comparison.csv`, including
RMSE, MSE, MAE, bias, diagnostic rent slices, timing, exact configuration, and
data/code hashes. Local paired predictions live under ignored
`artifacts/temporal_comparison/`. Resume skips only identical recorded runs.
`python src/generate_figures.py --temporal-only --reference-candidate hierarchical_market_blend --bootstrap --error-groups` reconstructs matched-seed
ensemble comparisons, including the fixed 50:50 direct/residual combination,
in `reports/temporal_summary.csv`. Complete-window RMSE is the selection metric;
the label-defined low-rent slices are diagnostics only.
The reference option explicitly compares candidates with the submitted
four-component pipeline. Frozen reference predictions can come from an earlier
source revision; matching row IDs, labels, training cutoff and data hashes are
checked, and both source hashes are recorded. `--error-groups` writes regenerable
month/town/flat-type/area diagnostics locally to
`artifacts/temporal_comparison/error_groups.csv`, not another GitHub report.
The optional 1,000-resample address-cluster bootstrap measures paired uncertainty
within these development samples; it does not establish future-market accuracy.
`--trial-iterations 5 --candidates legacy --windows short_recent` is a quick
GPU smoke run; its scores are not suitable for model selection.

The explicit commands used to generate the now-promoted local-trend run were:

```bash
python src/run_experiments.py --compare-candidates --candidates local_trend_direct local_trend_residual --windows short_recent long_recent long_older horizon16 --comparison-seeds 42 2026 --trial-iterations 1600 --task-type GPU --resume
python src/generate_figures.py --temporal-only --reference-candidate hierarchical_market_blend --bootstrap --error-groups
python src/train_final_model.py --task-type GPU --history-views local_trend:0 hierarchical:6 --submission submissions/submission_local_trend_diversity.csv --model-prefix candidates/local_trend_diversity/model --metadata artifacts/candidates/local_trend_diversity/metadata.json
```

This uses local-trend features only for the normal-history view and the proven
hierarchical recipe for the six-month stale-history view. Mixed views are
recorded explicitly in metadata; experimental output paths remain isolated.
Each accepted recipe has a distinct model suffix, and residual/normalized
targets are rejected unless every view supplies the required adaptive baseline.

Reproduce the isolated ablation and bounded alternative-learner screen:

```bash
python src/run_experiments.py --compare-candidates --candidates composition_trend_only_direct composition_trend_only_residual --windows short_recent long_recent long_older horizon16 --comparison-seeds 42 2026 --trial-iterations 1600 --task-type GPU --resume
python src/run_experiments.py --compare-candidates --candidates xgb_trend_depth6_direct xgb_trend_depth6_residual xgb_trend_depth8_direct xgb_trend_depth8_residual --windows short_recent horizon16 --comparison-seeds 42 --trial-iterations 1600 --task-type GPU --resume
python src/generate_figures.py --temporal-only --reference-candidate hierarchical_market_blend --bootstrap --error-groups
```

Reproduce the leakage-safe four-head versus eight-model stacking comparison.
The first command creates any missing matched-horizon out-of-fold predictions;
the second freezes their exact provenance and evaluates the meta models:

```bash
python src/run_experiments.py --compare-candidates --candidates local_trend_direct local_trend_residual horizon_direct horizon_residual --windows bridge_2022 bridge_2023 bridge_2024 short_middle short_recent long_early long_older long_recent horizon16 --comparison-seeds 42 2026 --trial-iterations 1600 --task-type GPU --resume
python src/generate_figures.py --stacking-only --refresh-stacking-sources
```

This learns static, latest-window and fixed 12-month-decay meta weights on months
strictly before each evaluation period and writes seven forward transfer
checks to `reports/stacking_results.csv`.
`reports/stacking_sources.csv` records the 72 exact source runs and prediction
hashes. Later reruns without `--refresh-stacking-sources` verify and reuse that
frozen source set. It does not fit a meta model on in-sample base-model predictions.
The result table distinguishes all available meta-training rows from the exact
rows, dates and source runs used to fit each method; its training RMSE is measured
on the latter, with time-decay weights applied when relevant.

The summary includes pre-specified 25% XGBoost / 75% mainline blends, paired on
the same seed and validation rows. XGBoost's bounded screen did not improve either
window, alone or blended, so it did not advance to more seeds. This rejects only
the tested configurations. Native category vocabularies are fitted on training
rows; unseen values become missing. XGBoost JSON files preserve these vocabularies.
The pinned XGBoost 3.0.5 CUDA-12.5 build works with this laptop's 560.94 driver;
newer CUDA-13 builds did not. GPU requests fail explicitly if training falls back
to CPU. Equal tree counts do not imply equal model capacity or runtime.
Categorical handling follows the [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html),
with explicit saved vocabularies for the pinned version. An input-device warning
during prediction means pandas data is transferred through a DMatrix; each
recorded XGBoost training device is independently checked to be `cuda:0`.
For the exact isolated-versus-prior-candidate uncertainty comparison, run the
same summary command with `--reference-candidate local_trend_history_diversity_blend`.
This refreshes the existing summary for that reference; rerun the documented
hierarchical-reference command to restore the canonical report view.

Feedback-driven experiments use one separate entry point and never overwrite
the selected model or submission:

```bash
python src/feedback_experiments.py --smoke --windows short_recent --task-type GPU
python src/feedback_experiments.py --windows short_recent horizon16 --families nearest specialists mixture --task-type GPU --resume
python src/feedback_experiments.py --audit-only
python src/feedback_experiments.py --families crossfit --windows short_recent --task-type GPU --smoke
python src/feedback_experiments.py --families crossfit --windows short_recent horizon16 --seed 42 --task-type GPU --resume
python src/feedback_experiments.py --families market learners correction --windows short_recent horizon16 --task-type GPU --smoke
python src/feedback_experiments.py --families market learners correction --windows short_recent horizon16 --seed 42 --task-type GPU --resume
python src/feedback_experiments.py --families ridge_guarded --windows short_recent horizon16 --task-type GPU --resume
```

The nearest-comparable predictor matches canonical flat type and uses location,
area, lease age and recency. Its three market forecasts are a constant level,
the existing six-month damping, and bounded linear extrapolation. Flat-type
specialists split 3-room, 4-room, 5-room/executive and smaller flats; sparse
groups fall back to the frozen mainline. The rent mixture trains three target-
band regressors and a feature-based classifier, then combines the regressors
using predicted probabilities. Validation rent never chooses a prediction route.
Each alternative is tested alone and in a predefined 25% blend with the mainline.
The GPU screen uses seed 42 and 1,600 trees per regressor, versus the frozen
eight-component mainline; it is a practical candidate screen, not an isolated
equal-capacity ablation. Promising trained candidates need a second seed and
matched longer/older-period confirmation before promotion. Smoke runs use
reduced data and 40 trees and are excluded from formal result tables.

The cross-fitting follow-up preserves original history features and targets.
Four comparable groups receive eight relative-rent mean/support features, or
sixteen features including dispersion and the rate below 65% of the reference
market level. Five input-group folds exclude every query-fold label; canonical
identical inputs share a fold. A strict-past control computes the same statistics
from earlier training months. Both encodings use identical outer-training-only
statistics for future validation. The contrast separates the training-encoding
rule from the added feature definitions. Original direct/residual same-seed
predictions supply matched references. `comparison_gain` compares a component
or normal-history pair with its matched reference; `gain` compares with the full
mainline; `crossfit_vs_past_gain` compares identical distribution features under
the two encoding rules. `replace_seed` changes only two of the eight components.
It requires a second-seed and other-window check before any full promotion.
New-fit times are recorded per regressor; paired/ensemble rows reuse those fits.
Cross-fit statistic computation time is recorded separately. Audit summaries
stay in `reports/information_audit.csv`; run-specific sources, fold assignments,
encoded features, predictions and subgroup tables remain in ignored artifacts.

The market screen adds supplied monthly COE premiums, three-month changes and
bid/quota demand ratios, plus a median-return stock index, returns, volatility,
breadth and coverage. Each rental row uses the PREVIOUS month, without backward
filling. The `origin_only` control freezes future economic inputs before the
forecast origin; other variants assume the provided future-month snapshot is
permitted for this offline task. Neither assumption establishes real-time release
availability. Stock membership is snapshot-based; CLOSE avoids ex-post dividend
adjustments but not every corporate-action distortion. The COE-only/stock-only
screens share a missing-data-availability indicator across the supplied sources.

The matched learner screen compares original and market-augmented inputs with
GPU XGBoost (native categorical vocabularies fitted on training only) and CPU
Ridge (training-only imputation/scaling and one-hot encoding, alpha 100). A
follow-up clips Ridge inference inputs to training-input ranges; the original
rent offset is preserved. This checks unbounded extrapolation, not a new target
or cleaned validation subset. The guarded Ridge origin-only control reuses the
same fitted market model. CatBoost origin-only and supplied-snapshot controls
use separate same-seed fits; small GPU fit variation remains a confound.

The targeted correction uses the latest twelve months of honest earlier-origin
OOF predictions, not fitted-training residuals. It trains a 600-tree depth-4
residual regressor, or a classifier plus two residual experts for rents below
65% of the OOF base prediction. Routing at validation uses predicted probabilities
only. Fixed correction strengths are 25% and 50%. Its geographic/flat/area/age
inputs denote comparable transactions, not individual-flat identity. Label-based
rent groups are diagnosis only. A source-verified saved-run summary can append
descriptive address-cluster gain intervals without training:

```bash
python src/feedback_experiments.py --summarize-runs 8050c7a295047db8 e3bf7c874aee98da
```

This command requires those original ignored local run artifacts. Canonical
ledger `trees` is the per-regressor tree budget (Ridge/nearest neighbours: zero;
targeted corrections: 600); paired/replacement/blend rows reuse fits. The earlier
absolute-rent mixture classifier has 1,000 trees in addition to its three
1,600-tree experts. `actual_device` distinguishes CPU Ridge from GPU boosters.
Intervals are exploratory within overlapping development windows and are not
independent confirmation or estimates of private leaderboard performance.

The market/learner/correction screen and guarded-Ridge follow-up are complete:
32 GPU fits and eight CPU Ridge fits across two windows, with 104 new formal
score rows. No candidate improves both windows enough to advance. Mainline
RMSE remains 440.7629/483.4345; the best long-window guarded Ridge blend reaches
481.5146 but loses recently (442.3978). Relative-low-rent detection is weak
(AUC 0.576/0.543), and neither tested correction improves the recent window.
No new CSV or production replacement was made. Exact negative results and
external-data boundaries are recorded in PROGRESS and KAGGLE_AUDIT.

The cross-fitting follow-up is complete (seed 42, both windows, twelve regressor fits).
The cross-fitted distribution pair loses 17.18/20.28 RMSE to the identical-
statistic strict-past control. Replacing two mainline heads with strict-past
statistics gains only 0.122/0.136, below the advancement gate; the mainline and
submissions are retained. Details and subgroup interpretations are in PROGRESS.
The ledger distinguishes standalone, paired and actual two-head-replacement
scores. A cached same-seed reference is not retrained. Raw records remain retained;
the audit confirms complete supplied block joins, constant FEE/FURNISHED and no
unseen canonical town/flat/model/street categories in the supplied test set.

`reports/feedback_results.csv` records metrics, dates, time, feature groups,
source prediction IDs and hashes. Exact configurations, candidate predictions
and group diagnostics stay locally under `artifacts/feedback_study/`.

Categorical preprocessing is model-specific: canonical text is passed through
CatBoost's native categorical handling; Linear/Ridge use one-hot encoding;
the original classical tree baselines use ordinal encoding. An address/flat-type
history denotes comparable transactions, not an identified individual flat.
Identical rows are retained because the dataset cannot establish whether they
refer to the same flat.

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

- `33 passed` from pytest;
- candidate scores and adoption evidence in `docs/PROGRESS.md`;
- `submissions/final_submission.csv` with columns `Id,Predicted` and
  exactly 50,000 rows;
- eight active model paths listed in `reports/final_model_metadata.json`: two
  seeds, direct/residual targets and local-trend/`_lag6` training views;
- `reports/final_model_metadata.json` with component modes, weights, timings,
  model/code/data hashes, and the submission hash.

When loading a residual model, use `predict_model` from `src/modeling.py` with
its metadata component's `residual_target` flag: a raw CBM prediction is a rent
difference and must have the market baseline added back. The training command
automatically reloads every saved model and checks 100 predictions before
writing and round-trip-checking the submission CSV.

The mainline command is `python src/train_final_model.py --task-type GPU`.
An experimental override (candidate/config, seeds, trees, outlier settings or
history views) must specify separate `--submission`, `--metadata` and
`--model-prefix` paths; otherwise the command fails before training.
Output paths are restricted to their project output directories.

To reproduce the user-requested fixed historical-weight stacking CSV without retraining,
reuse the eight verified mainline CBMs with the recorded component-order weights:

```bash
python src/train_final_model.py --reuse-metadata reports/final_model_metadata.json --component-weights 0.06945411339312821 0.1600015151383129 0.06945411339312821 0.1600015151383129 0.13361900687086095 0.13692536459769795 0.13361900687086095 0.13692536459769795 --weight-label three_month_2025/four_heads/recent_window_half_shrunk_simplex --model-prefix candidate_recent_stack --submission submissions/submission_recent_window_stacking.csv --metadata reports/submission_recent_window_stacking_metadata.json
```

The command verifies model/data/source-CSV hashes and IDs, then reproduces the
source equal-weight prediction before writing the candidate. Both output paths
must be fresh: existing complete or partial candidates are rejected before model
loading, and exclusive file creation prevents overwriting. The command above
documents the original artifact; use two new output paths to rerun it locally.
The fixed weights came from Oct-Dec 2024 OOF predictions; they were not refitted
on the available Jan-Mar 2025 OOF predictions. This is an experimental probe.

Before replacing an existing complete run, the training command verifies and
snapshots its CSV, metadata and every model into ignored
`artifacts/submitted/<submission-sha256>/`, mirroring repository-relative paths.
An incomplete or hash-mismatched previous run is rejected. A failed later fit
can still leave partial working outputs; the verified snapshot is the recovery
source, not an atomic deployment guarantee.

The retained 436.05684 reference is under:
`artifacts/submitted/cb6ec7a6d60cf2e491ff044faa202ccf1873adb962a8ef810ccf437f4c85b2d3/`.
Resolve its metadata's model/submission paths relative to that snapshot directory.
Its original training recipe is preserved in that metadata. The exact uploaded
432.47187 candidate also remains locally under
`artifacts/candidates/history_diversity/` and its original submission filename.
This retention is for reproducibility and rollback, not an additional active branch.

Use `--task-type CPU` without NVIDIA hardware. GPU and CPU CatBoost are not
bit-identical; even fixed-seed GPU reruns can differ. The promoted artifact is
byte-exact, while retraining reproduces the recipe rather than a guaranteed hash.
The current eight-component run took 669.44 seconds on the RTX 4060; the retained
four-component run took 336.92 seconds. No new fit was run during promotion.
Private data, models, figures and submission CSVs remain ignored by Git.

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
src/feedback_experiments.py   Entry point for information audit and feedback-driven experiments
src/market_experiments.py     Research-only economic indices, MRT dates, matched learners and OOF corrections
src/followup_analysis.py      Transfer decomposition, low-rent risk and interchange-access checks
configs/mrt_openings.json     Official service/announcement dates for recent MRT openings
src/generate_figures.py    Report-ready experiment and error plots
src/update_progress.py     Refresh the historical early-stopping reference block only
src/train_final_model.py   Train the selected model and validate its submission
reports/temporal_comparison.csv   Current fixed-budget per-fit evidence
reports/temporal_summary.csv      Matched-seed and fixed-blend comparisons
reports/feedback_results.csv      Feedback-driven screens; not a mainline selection
reports/information_audit.csv     Aggregate input and spatial-join audit
reports/followup_*.csv            Post-feedback transfer/risk/access aggregates
reports/full_budget_reproduction.json  Independent full-budget selected-recipe check
reports/final_model_metadata.json Exact local model and submission provenance
reports/kaggle_submissions.csv    Only actually submitted public scores
progress_report/main.tex          Editable source for the 14-slide progress report
progress_report/assets/           NUS logo and the five figures used by the report
output/pdf/CS5228_Progress_Report.pdf   Submission-ready progress-report PDF
final_report/main.tex             Single evolving final scientific-paper source
final_report/prepare_report.py    Derive paper figures/tables from frozen evidence
final_report/check_statistics.py  Raw/canonical address bootstrap sensitivity
final_report/figures/             Vector plots committed for standalone compilation
final_report/generated/           Aggregate evidence and generated table fragments
output/pdf/CS5228_Final_Report.pdf     Current final-report draft for team review
tests/                     Automated data, feature, and validation tests
pytest.ini                 Collect only maintained tests, excluding local clone copies
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
exploration, and never commit course data, credentials, trained models, or
generated submissions. The adapted progress-report source, selected figures and
PDF are included in the shared project under the user's 6 October authorization.
They preserve the submitted progress checkpoint; later experiments belong in the
cumulative progress log rather than retroactively changing that PDF.

## Official macro indices and MRT opening experiments

Completed 6 October: 36 formal GPU fits across the recent quarter and sixteen-
month window, with 55 tests passing. No setting met the promotion rule. Corrected
static rail replacement gained only 0.132/0.017 RMSE, with intervals crossing
zero; the other variants were negative or similarly small. Monthly released-
input controls did not change the decision. Mainline/submissions are unchanged;
full positive and negative evidence is in the existing cumulative PROGRESS log
and feedback ledger. GPU refits are not bit-deterministic; the same-model controls
and saved prediction/source hashes support the recorded comparisons.

Use the existing experiment entry point; no separate training codebase is needed:

```bash
python src/feedback_experiments.py --download-public
python src/feedback_experiments.py --families public --windows short_recent --task-type GPU --seed 42 --smoke
python src/feedback_experiments.py --families public --windows short_recent horizon16 --task-type GPU --seed 42 --resume
python src/feedback_experiments.py --families public --public-variants macro_both rail_events --public-monthly-controls --windows short_recent horizon16 --task-type GPU --seed 42 --resume
```

The downloader caches only official aggregate CPI and URA private non-landed
whole-island rental indices under ignored `data/external/official_macro/`.
Existing snapshots are preserved; downloads from another date may have revised
values, so compare source hashes before claiming exact numerical reproduction.
No external individual rental labels are downloaded or matched.

Seven fixed variants test CPI, URA, both indices, corrected static rail access,
rental-month rail access, rail opening exposure, and the combined features.
Each uses the same original historical-rent recipe and direct/residual targets.
Besides paired-head results, the ledger records replacing two of eight mainline
components and a fixed 25% blend. These are deployment screens; blends are not
equal-capacity model comparisons. `--public-variants` selects a subset for a
confirmation run, and `--summarize-runs RUN_ID` verifies predictions and appends
descriptive address-bootstrap intervals to the existing results table.

At rental-month start, CPI activates two months after its observation month;
URA activates two months after quarter end. All validation macro values are
frozen at the forecast origin. Latest downloaded historical vintages are used;
release lags do not reconstruct the original revisions or CPI rebasing.
The optional monthly control predicts with the SAME fitted models using indices
already published and rail service already operational by each rental month.
This evaluates available contemporaneous covariates for dated transactions;
it is separately labelled and does not establish a single-origin future forecast.
Rail features use the first full month after opening and only opening dates
announced before the validation origin. Unmapped stations stay excluded.
The static rail control uses an origin snapshot for both fit/query and is not a
claim of historical availability. Legacy service flags only mean operational
before the 2021 study, not exact original opening dates. The station catalog
remains limited to the supplied coordinates, especially its incomplete LRT rows.
The opening features test predictive association, not a causal rent premium.

Configurations, source snapshots and paired predictions remain local under
`artifacts/feedback_study/`; aggregate outcomes use the existing
`reports/feedback_results.csv` and `docs/PROGRESS.md`. To check report plots
without overwriting archived figures:

```bash
python src/generate_figures.py --output-dir artifacts/plot_check
```

## Course deliverables and next work

Canvas project page, policies, announcements and assignment list rechecked read-only on 23 October 2026:

- Methodological quality 60%, report quality 30%, reproducibility 10%.
- Progress report: approximately 10-15 slides as PDF; final 1-2 slides describe
  next steps. The page says Sunday 4 October (not a hard deadline).
- Final report: at most eight main-text pages, excluding references/appendix,
  plus code or a GitHub link and member workload breakdown.
- The final PDF should use scientific-paper form and include the team name,
  all member names and student IDs. Tables/plots count toward the eight pages;
  the main argument must be understandable without the appendix.
- The 8 October announcement confirms no final presentation and that Kaggle
  score is not the grading criterion. Group-project AI assistance is permitted
  with appropriate documentation, as stated on the course Policies page.
- Final report deadline: **Friday 6 November 2026, 23:59 Singapore time**, listed
  in the Canvas Overview project table (checked 29 October). The Project page
  still contains a placeholder; the Overview supplies the date.
- Deadline source: https://canvas.nus.edu.sg/courses/93866/pages/overview
- Source: https://canvas.nus.edu.sg/courses/93866/pages/project

The current 14-slide draft is `output/pdf/CS5228_Progress_Report.pdf`. Edit
`progress_report/main.tex`, then rebuild with a LaTeX installation containing
Beamer:

```bash
cd progress_report
latexmk -pdf -outdir=../output/pdf -jobname=CS5228_Progress_Report main.tex
```

The title page lists all four team members and their student IDs. Before
submission, let every member confirm the spelling and ID pairing, and review
the claims and AI declaration.

### Final report working draft

The English paper is `output/pdf/CS5228_Final_Report.pdf`. The official Canvas
LaTeX template was downloaded and verified against the existing course ZIP
(SHA-256 `ab2974691f767969dd6b2b1de18217ccaa7829f4439c5232a3fbbaa03bbfda54`).
Its unchanged `IEEEtran.cls` is bundled under `final_report/` so the report can
compile independently of the ignored template archive.

Edit the single evolving `final_report/main.tex` and `references.bib`. Figures
and numeric table fragments are under `final_report/figures/` and `generated/`.
The draft prioritises local temporal evaluation and subgroup interpretation;
Kaggle scores are supplementary historical observations. It currently has
eight main-text pages, one reference page and one appendix page (ten total).
Workload attribution is explicitly reserved in Appendix B and remains unfilled.
It is a review draft, not a final submission or an assertion of team approval.

The focused review adds recorded address-cluster gain intervals, explicit
training cutoffs and fixed rental-history evaluation, and definitions of the
low-rent detector and nearest-comparable/specialist/mixture candidates. The
main temporal table appears beside the main results. PROGRESS records the
disposition of all fourteen suggestions; workload completion and team review
remain pending. The official template is retained; the 29 October follow-up
expands the body to eight pages, within the course limit.

Compile the committed text, figures and tables in the built-in LaTeX editor,
or on Windows with existing MiKTeX/TeX tools:

```powershell
./final_report/build.ps1
```

This uses `pdflatex` and `bibtex`, without requiring latexmk/Perl. On Overleaf,
upload the contents of `final_report/` and choose `main.tex` as the main file.
To regenerate the paper's figures/tables from local course data and the existing
temporal prediction caches, after following the earlier cache-generation steps:

```bash
python final_report/prepare_report.py
```

Compiling the committed report does not require training data or local caches.
Regenerating its evidence does: frozen sources are checked against prediction
and data hashes, and the selected metric is protected by an explicit check.
GPU retraining is not expected to reproduce every bit. To report a fresh run,
review its provenance, refresh the source manifest using the documented workflow,
and update the report's protected metric and claims together. Do not bypass checks
to make an unrelated cache appear to be the original evidence. Full-budget
selected-model retraining has now completed in a fresh local environment;
see the 29 October follow-up below for its exact scope. Cross-device checking
remains a separate team task.

The figure generator requires Times New Roman (installed by default on Windows);
install an authorised copy on other systems or use the committed vector figures.

`python final_report/check_statistics.py` repeats the 1,000-resample raw versus
canonical-address sensitivity check using the protected stacking source manifest
and paired reference predictions. Results are in `reports/pre_submission_checks.json`.
It needs the original local prediction caches; no model is retrained.

That command reads existing evidence and trains no models. Cached predictions,
course data, downloaded historical example reports and compiler intermediates
stay local. The report source, aggregate tables/figures and final draft PDF are
maintained in Git. Future project changes should revise this same report and
append the existing progress log rather than create another report version.

Acceptance check (29 October 2026): a local clone with a fresh Python 3.12 venv
installs the pinned requirements, passes `pip check`, 55 tests and the nine-file
data audit. Without original caches it completes a five-tree GPU validation run
and the eight-component five-tree workflow on all 50,000 test inputs (126.63 s),
including model reload and CSV checks. This is same-device, reduced-budget
acceptance, not a full-budget retraining or a different-machine verification.
The cloned report also compiles without private experiment caches. Detailed
smoke outputs stay local; PROGRESS records the commands, results and limitations.

The earlier closeout is followed by the bounded experiments below.

### Follow-up diagnostics and full-budget reproduction (29 October)

`python src/followup_analysis.py` decomposes retained normal/stale and
direct/residual predictions, reports monthly/town/type errors, tests a fixed
half-bias correction learned at an earlier origin, and evaluates smoothed
town/type low-rent-event rates on later labels. Aggregates are maintained in
`reports/followup_transfer.csv`, `followup_groups.csv` and `followup_low_rent.csv`.
These are reused development windows, not new unseen-period evidence.

`python src/followup_analysis.py --train-transit` tests nearest operational
interchange distance and distinct line prefixes accessible within 1/2 km.
It uses the audited station calendar and announcements known before each origin;
these geographic access measures are not actual network routes or commute times.
The fixed one-seed, direct/residual 1,600-tree screen covers the recent quarter,
older year and sixteen-month window. Detailed features/predictions stay under
ignored `artifacts/followup`; aggregates are in `reports/followup_transit.csv`.

Full-budget reproduction is performed in the ignored independent local clone
`tmp/submission_acceptance`, using its new venv and original course data, without
original model/prediction caches. From that clone's root run:

```bash
python src/train_final_model.py --task-type GPU --model-prefix acceptance_full --submission submissions/acceptance_full.csv --metadata reports/acceptance_full_metadata.json
python src/run_experiments.py --compare-candidates --candidates local_trend_direct local_trend_residual horizon_direct horizon_residual --windows short_recent --comparison-seeds 42 2026 --trial-iterations 1600 --task-type GPU
```

From the maintained checkout, `python src/followup_analysis.py --summarize-reproduction`
audits that clone's eight full-budget CBMs, all ordered test predictions and the
eight-component recent validation result. It writes aggregate evidence to
`reports/full_budget_reproduction.json`. Refit variation does not trigger model
promotion; frozen production artifacts remain the submission reference.

Remaining team actions and longer-term research:

1. Confirm member ownership and review the reproduction evidence above.
2. Investigate the old-window regression with more chronological cutoffs and
   monthly/town/flat-type errors before tuning against public scores.
3. Maintain one mainline and, only when justified, a complementary final candidate;
   preserve both outlier-policy submissions as historical evidence.
4. Review and revise the existing final-report draft as further evidence becomes available.
5. Verify the final deadline and freeze/reproduce the final pipeline before delivery.

Keep one cumulative `docs/PROGRESS.md`, one main results table here, and the
existing evidence tables. See `docs/PROJECT_PLAN.md` for scope and responsibilities.
