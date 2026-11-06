# Reproduction guide

This is the technical command reference. Start with the root README for the
selected result and folder map. Commands that train models or regenerate
reports must run in an isolated checkout; they must not overwrite the frozen
submission evidence. Course data and caches are intentionally not distributed.

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
|-- example-submission.csv    (optional; only if provided by the course)
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

The official Kaggle ZIP has eight required CSVs (train, test and six auxiliary
files), not the optional example template. You may import it with
`python src/import_official_data.py path/to/official.zip`.
When the template is absent, setup/audit and both final-submission entry paths
use the verified `Id,Predicted` contract with zero-based positional IDs in memory.
No substitute CSV is written into raw data or claimed to be a course file.
If a template is present it must pass its checks; malformed templates are not
silently ignored. Raw-file checksums remain enforced.

### Explicit research cache inputs

Do not rename a fresh run to an old hash ID. Use the actual run folder printed by
`compare_models`, which contains both features and predictions, not just its parent:

```bash
python -m src.research.compare_models --candidates reference --windows short_recent --output artifacts/research_repeat
python -m src.research.train_hdb_attributes --candidates resale --reference-run-dir artifacts/research_repeat/ACTUAL_RUN_ID --output-dir artifacts/attribute_repeat
python -m src.research.probe_rmse_structure --windows short_recent --reference-run-dir artifacts/research_repeat/ACTUAL_RUN_ID --output-dir artifacts/rmse_repeat
```

These commands **train research models** and are not part of report compilation.
Use the separate research environment for LightGBM candidates/dependent probes;
CatBoost-only comparison no longer requires LightGBM package metadata.

`analyze_hdb_attributes`, `train_hdb_attributes`, `probe_historical_augmentation`,
`check_low_rent_information`, `probe_low_rent_mixture`, `probe_rmse_structure` and
`probe_cat_capacity` accept `--reference-run-dir` and `--output-dir`.
The two low-rent scripts additionally use `--housing-run-dir` for a completed
resale-feature run; the mixture uses `--classifier-run-dir` for saved risk scores.
For attribute analysis, `--output-dir` selects the attribute-study root being
analysed (its ensemble files live in child run folders).
For different reference folders across windows, invoke the probe once per window,
each with its corresponding reference and a fresh output directory.
`submit_market_normalized` accepts `--stock-run-dir` and `--selected-run-dir`;
it still requires the corresponding completed four-component stock and
market-normalized model artifacts, not a generic reference cache.
The MLP, sparse-LightGBM and comparison-analysis scripts retain their positional
`run_folder` argument and now check prerequisites before reading caches.

Missing inputs list their exact paths and the prerequisite command. Historical
defaults remain compatible, but are not automatically searched or substituted.
Only use trusted team caches: pickle loading can execute code. Shapes/labels
and protocol checks in the existing probes still apply. Path parameters do not
make unrelated protocols interchangeable or bypass missing external-source data.
The `artifacts/temporal_comparison/` NPZ directory is a different cache family;
it cannot replace a CPU `compare_models` folder with its required PKL/NPY files.

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
external-data boundaries are recorded in PROGRESS and PROGRESS.

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

- a passing test suite (6 November check: `142 passed, 4 skipped`; skips are
  optional LightGBM checks when that research dependency is not installed);
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

## Build the final report

The English paper is `final_report/project-report-team44.pdf`. The official Canvas
LaTeX template was downloaded and verified against the existing course ZIP
(SHA-256 `ab2974691f767969dd6b2b1de18217ccaa7829f4439c5232a3fbbaa03bbfda54`).
Its unchanged `IEEEtran.cls` is bundled under `final_report/` so the report can
compile independently of the ignored template archive.

Edit the single evolving `final_report/main.tex` and `references.bib`. Figures
and numeric table fragments are under `final_report/figures/` and `generated/`.
The five figures cover the data profile, full selected-model architecture,
chronological validation windows, temporal RMSE trade-offs and error concentration.
`python final_report/prepare_report.py` regenerates them from protected local
evidence (including retained prediction caches); compilation alone needs no data.
Figure 2 uses the reviewed image-model illustration in
`final_report/figures/model_overview.png`; it is a raster illustration, not a
vector schematic. Its `.tex` wrapper preserves the saved pixels and overlays
publication-size Times New Roman labels. With XeLaTeX, Times New Roman and the
standalone/fontspec/TikZ packages available, run
`python final_report/prepare_report.py --overview-only` to rebuild its PDF.
This does not call an image API, regenerate numeric tables, load course data or
train models. The committed figure PDF is used by the normal report build, so
readers need not regenerate it to compile. Other report figures remain vector.
Repeated diagram-footer notes are covered by the caption/body instead. The
reviewed labels remain at least 8 pt in the report; float placement follows
first citation in two-column reading order, with top/bottom placement allowed.
The draft prioritises local temporal evaluation and subgroup interpretation;
Kaggle scores are supplementary historical observations. The reviewed PDF has
eight body pages, one reference page and two appendix pages. Its shrinkage table,
intervals and attribution now use the independent CPU repetition in
`reports/history_research/dispersion_shrinkage_cpu_reproduction_20261106/`.
The old team aggregates are preserved separately; they are not presented as an
exactly reproduced run. Dates belong to data periods and provenance, not an
article update log. Rebuild and recheck pages after any subsequent edits.
Workload attribution is in Appendix C; AI use, including Figure 2, is declared in Appendix B.
All members must confirm their responsibilities before the authorised submission.

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
existing evidence tables. See `docs/PROGRESS.md` for scope and responsibilities.

## Historical reliability studies (3–5 November)

The following are completed development studies, not new production recipes.
Detailed artifacts are local/ignored; a fresh clone must restore the original
course data and validated caches before repeating prediction-based diagnostics.
All windows overlap; they are not independent final tests.

The public evidence paths below are relative to `reports/history_research/`.
They can be read without local caches; row-level reproduction still needs the
private inputs described below.

| Study | Committed aggregate evidence | Result / decision |
| --- | --- | --- |
| Horizon-dependent weights | `horizon_blend_v1/results.csv` | Recent RMSE 441.083 vs 440.763; no promotion |
| Support-group weights | `support_blend_v1/results.csv` | Recent RMSE 441.091 vs 440.763; no promotion |
| Historical reliability | `history_reliability_diagnostic_v1/group_metrics.csv`, `stratified_contrasts.csv` | High dispersion is a relatively stable risk marker; support/age are confounded |
| Parent–child references | `parent_child_reference_v1/paired_comparisons.csv` | Finest-level parent advantage motivated a bounded experiment, not deployment |
| Dispersion shrinkage | `dispersion_shrinkage_cpu_v1/results.csv`, `local_evaluation_check.json` | Two-head recent gain 0.103, 16-month loss 0.317 RMSE; no promotion |
| Saved-prediction attribution | `dispersion_shrinkage_attribution_v1/error_changes.csv` | Loss starts in the first month; target modes and monthly effects differ |

### Tests and independent verification (6 November)

```bash
python -m pytest tests/test_history_research.py -q
python -m pytest -q
```

The 57 focused checks cover all six research modules: source hashes and ledger
consistency, complete/aligned validation rows, forward-only meta-training,
bounded/clamped weights, sparse-group fallback, past-only normalized support
and dispersion, the two-column shrinkage formula and unchanged residual target,
address-cluster bootstrap units/sign, and additive SSE attribution. Synthetic
end-to-end cases exercise blending and attribution, plus a three-tree CPU
CatBoost fit. They require neither course data nor original research caches;
synthetic scores are never model-quality evidence. The full suite currently
passes 142 checks with four optional LightGBM skips.

Separately, the original 72 frozen OOF caches passed hash, label, row and forward
split validation on this machine. Both blending studies were recomputed into
ignored `tmp/history_research_verification_20261106/`; all 21 rows per study
matched the committed aggregates (`rtol=1e-12`, `atol=1e-9`). This is a
recalculation from saved predictions, not retraining or independent test data.
The teammate's original shrinkage caches remain unavailable. A subsequent fresh
eight-fit CPU repetition is now complete; its saved models, predictions, feature
intervention, metrics and new address-cluster intervals were verified locally.
This independently validates the currently committed recipe, not the exact old
binaries or every old score. Maximum old-versus-new RMSE difference is 0.199056;
its cause cannot be established without the original provenance/caches. Both
runs support keeping this candidate research-only. Public evidence is in
`reports/history_research/dispersion_shrinkage_cpu_reproduction_20261106/`.

### Reuse predictions before retraining

The blending studies require `reports/stacking_sources.csv`, its original
`artifacts/temporal_comparison/` prediction caches and `data/raw/train.csv`.
Use fresh output directories; the examples below do not overwrite completed v1
studies. Run only the study needed, not this entire block as a training pipeline.

```bash
python -m src.research.horizon_blend --audit-only
python -m src.research.horizon_blend --output artifacts/horizon_blend_v2
python -m src.research.support_blend --output artifacts/support_blend_v2
python -m src.research.history_reliability --output artifacts/history_reliability_diagnostic_v2
python -m src.research.parent_child_reference --output artifacts/parent_child_reference_v2
```

The first two blending modules fit combination weights using earlier labels;
the reliability and parent–child modules fit no models or weights. No command
above retrains CatBoost or generates a submission. Missing or mismatched caches
must be resolved rather than silently substituting new predictions.

### Finest-level dispersion shrinkage (completed; not promoted)

`src/research/dispersion_shrinkage.py` is a separate paired CatBoost study.
Only high past-dispersion records (at least five comparables and adjusted
IQR/median > 0.25) change the Address/Flat/Area parent strength from 5 to 10.
Only its LEVEL and PREMIUM input columns change. Original residual offsets
remain fixed in both recipes. Training statistics exclude the entire query
month; validation histories are frozen at the training origin. No production
configuration, submission or frozen GPU comparison is changed.

```bash
python -m src.research.dispersion_shrinkage --task-type CPU --windows short_recent horizon16 --iterations 1600 --seed 42 --output artifacts/dispersion_shrinkage_cpu_v2
```

This fits eight models: two windows x original/candidate x direct/residual.
Compare the same-device two-head pair rows in results.csv, not the historical
GPU eight-model RMSE. groups.csv reports high/low/other/all slices (overlapping
slices are not additive); feature_changes.csv records changed-row counts.
Models, original offsets in feature caches, predictions and provenance are
saved only in the new directory. Add `--resume` to the identical command to
reuse completed fits; changed code/data/options are rejected. Interrupted
in-progress fits restart. Local pickle feature caches must not be replaced with
untrusted files. Small --iterations runs are execution checks, not score evidence.
No automatic second seed, promotion or Kaggle submission is performed.

The command above starts a **new training run** in a fresh directory; it is not
needed to read completed v1 results. The original team run and independent
6 November repetition use the same stated seed-42 CPU comparison:

| Validation window | Team baseline | Team candidate | Repetition baseline | Repetition candidate |
| --- | ---: | ---: | ---: | ---: |
| Jan–Mar 2025 | 445.759 | 445.656 | 445.759 | 445.488 |
| Dec 2023–Mar 2025 | 486.644 | 486.962 | 486.476 | 486.907 |

Fresh baseline-minus-candidate gains are +0.270473 and -0.431249 respectively.
Their 1,000-replicate address-cluster 95% intervals are [-0.423042, 0.921257]
and [-0.906902, 0.027248]: both cross zero. The candidate remains research-only;
these two-head scores must not be compared as a method gain against the
eight-component GPU mainline. These are repeated development windows, not new
holdouts. See the progress log for attribution and the limits of this comparison.

To repeat the full study, use fresh directories (the recorded ones now exist):

```bash
python -m src.research.dispersion_shrinkage --task-type CPU --windows short_recent horizon16 --iterations 1600 --seed 42 --output artifacts/dispersion_shrinkage_cpu_reproduction_NEW
python -m src.research.analyze_shrinkage_errors --source-dir artifacts/dispersion_shrinkage_cpu_reproduction_NEW --output reports/history_research/dispersion_shrinkage_cpu_reproduction_NEW --verify-models --bootstrap 1000 --reference-results reports/history_research/dispersion_shrinkage_cpu_v1/results.csv
```

The public repetition folder contains `results.csv`, `reference_comparison.csv`,
`local_evaluation_check.json`, `error_changes.csv`, `monthly_data_profile.csv`
and `audit.json`. The audit embeds the fresh source/environment signature and
all eight model-reload checks. Models, feature pickles and row predictions stay
under ignored `artifacts/`; only locally generated/trusted pickles may be loaded.

`python -m src.research.analyze_shrinkage_errors` reads the existing
`artifacts/dispersion_shrinkage_cpu_v1/` predictions and feature caches by default
and refuses to overwrite its existing output. For a fresh completed run, specify
`--source-dir` and a new `--output`; `--verify-models` reloads all eight CBMs,
checks their budget/seed/device, verifies cached predictions and the feature
intervention, and recomputes metrics. `--bootstrap` defaults to 1,000 whole-address
resamples. Optional `--reference-results` records prior-versus-fresh differences;
it never substitutes the prior predictions or forces matching scores.

The shared original-run summary is
`reports/history_research/dispersion_shrinkage_attribution_v1/error_changes.csv`.
Detailed monthly profiles and audit records remain local to the original run;
restore trusted inputs before reproducing them. Do not delete completed results
just to rerun the analysis.
