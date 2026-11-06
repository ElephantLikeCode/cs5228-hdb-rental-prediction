# Numerical evidence index

These small CSV/JSON files are experiment results and provenance, not trained
models or prediction caches. They support the paper and are intentionally kept.
Do not delete or rewrite earlier unsuccessful results to make the latest model
appear uniformly better. Values with different protocols are not a common ranking.

## Start here: evidence cited in the final report

| File / directory | Meaning |
| --- | --- |
| `experiment_results.csv` | Historical learner and spatial-feature baselines (Table II); earlier early-stopping protocol |
| `temporal_comparison.csv` | Fixed-budget per-component fits, configurations, cutoffs, runtime and cache hashes |
| `temporal_summary.csv` | Matched ensembles and the four core local windows (Table I / Figure 4) |
| `feedback_results.csv` | Candidate information/model screens, actual replacement/blend compositions (Table V) |
| `stacking_results.csv` | Forward-only four-head/eight-output stacking and transfer checks |
| `stacking_sources.csv` | Frozen prediction-source manifest; required to regenerate reported model comparisons |
| `final_model_metadata.json` | Exact production component paths, hashes, weights and submitted-CSV provenance |
| `submission_recent_window_stacking_metadata.json` | Optional stacking weights and verified source-model/CSV hashes |
| `kaggle_submissions.csv` | Actually submitted public scores; supplementary, not local test labels |
| `full_budget_reproduction.json` | Independent checkout/new-environment selected-model repeat, same physical device |
| `pre_submission_checks.json` | Raw/canonical address-cluster bootstrap sensitivity |
| `history_research/dispersion_shrinkage_cpu_reproduction_20261106/` | Independently repeated shrinkage results, intervals, model/feature checks and attribution (Table VI) |

Paper Table VIII keeps these original paths, so readers and code can locate the
same evidence. Core model configuration remains at `configs/model.json`.

## Other evidence, grouped by purpose

- `eda_*.csv`, `eda_summary.csv`, `information_audit.csv`: data understanding,
  category distributions, date coverage and auxiliary joins.
- `experiment_summary.csv`, `experiment_metadata.json`, `error_analysis.csv`,
  `feature_trials.csv`, `tuning_results.csv`, `robustness_trials.csv`,
  `gpu_iteration_trials.csv`, `long_horizon_results.csv`: historical baseline,
  feature, budget and robustness checks. Some are consumed by regeneration code.
- `followup_*.csv`: view decomposition, earlier-origin bias/risk tests and
  operational-interchange proxies. No mainline promotion.
- `november_*.csv`, `november_evidence_20261102.json`, `rmse_followup_*.csv`:
  supplementary CPU learners, log targets, housing attributes, error analysis
  and unsuccessful candidate compositions. The four-head CPU reference differs
  from the eight-component GPU mainline. Rows explicitly marked target-visible
  are diagnostics, never held-out prediction results or an error-floor estimate.
- `history_research/horizon_blend_v1/`, `support_blend_v1/`: learned history-view
  weights; do not improve the recent/long-window advancement screen.
- `history_research/history_reliability_diagnostic_v1/`,
  `parent_child_reference_v1/`: support/dispersion associations and matched
  historical-reference comparisons, not gains of the final CatBoost model.
- `history_research/dispersion_shrinkage_cpu_v1/` and
  `dispersion_shrinkage_attribution_v1/`: original team-run aggregates. They are
  preserved separately, not substituted for the later independently repeated
  numbers. Maximum discrepancy is 0.199 RMSE; both runs support non-adoption.

Private raw records, per-row predictions and fitted weights are not included.
See `docs/REPRODUCIBILITY.md` for commands, cache prerequisites and interpretation.
