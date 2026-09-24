# CS5228 HDB Rental Prediction

Private group-project repository for predicting Singapore HDB monthly rent.
The evaluation metric is RMSE, and chronological validation is the primary
local evaluation method.

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
50,000-row submission generator. Thirty-three automated tests cover data contracts,
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
python src/run_experiments.py --compare-candidates --candidates hierarchical market_residual horizon_direct horizon_residual --windows short_recent long_recent long_older horizon16 --comparison-seeds 42 2026 --task-type GPU --resume
python src/generate_figures.py --temporal-only --reference-candidate hierarchical_market_blend --bootstrap --error-groups
python src/update_progress.py
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
src/generate_figures.py    Report-ready experiment and error plots
src/update_progress.py     Refresh current results in the cumulative progress log
src/train_final_model.py   Train the selected model and validate its submission
reports/temporal_comparison.csv   Current fixed-budget per-fit evidence
reports/temporal_summary.csv      Matched-seed and fixed-blend comparisons
reports/final_model_metadata.json Exact local model and submission provenance
reports/kaggle_submissions.csv    Only actually submitted public scores
tests/                     Automated data, feature, and validation tests
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
exploration, and never commit course data, templates, credentials, trained
models, or generated submissions.

## Course deliverables and next work

Canvas rechecked on 21 September 2026:

- Methodological quality 60%, report quality 30%, reproducibility 10%.
- Progress report: approximately 10-15 slides as PDF; final 1-2 slides describe
  next steps. The page says Sunday 4 October (not a hard deadline).
- Final report: at most eight main-text pages, excluding references/appendix,
  plus code or a GitHub link and member workload breakdown.
- The final deadline remains the placeholder `Nov XX, 11.59 pm`; do not treat the
  former local 6 November date, or the Kaggle closing date, as a confirmed course deadline.
- Source: https://canvas.nus.edu.sg/courses/93866/pages/project

No further experiments are scheduled by this closeout. When the team resumes:

1. Reproduce setup/tests and agree on member ownership.
2. Investigate the old-window regression with more chronological cutoffs and
   monthly/town/flat-type errors before tuning against public scores.
3. Maintain one mainline and, only when justified, a complementary final candidate;
   preserve both outlier-policy submissions as historical evidence.
4. Prepare the progress report from existing EDA, baselines, ablations and failures.
5. Verify the final deadline and freeze/reproduce the final pipeline before delivery.

Keep one cumulative `docs/PROGRESS.md`, one main results table here, and the
existing evidence tables. See `docs/PROJECT_PLAN.md` for scope and responsibilities.
