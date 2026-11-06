# Project Progress

Single concise, cumulative research record. Current state is summarised first;
the chronological table preserves the reasoning and unsuccessful directions.
Full pre-cleanup notes are retained locally under ignored
`artifacts/local_archive/submission_cleanup_20261106/` and in earlier Git history.
New work should append evidence-linked decisions here, not duplicate handoffs.

## Current checkpoint — 6 November 2026

- Team: **ELEPHANT_in_the_Data (44)**, four members.
- Selected recipe: eight equal-weight local-trend/history-diversity CatBoost
  components, seeds 42/2026, direct/residual targets, 1,600 trees each.
  All labels retained; normal/stale views have 78/74 input features.
- Recent local RMSE/MAE: **440.762863 / 316.549083**.
  Recorded public scores: mainline **432.14881**, optional stacking **432.04286**.
  Stacking remains a backup, not an adopted replacement.
- Recent-year and sixteen-month gains coexist with an older-year loss of 9.811
  RMSE versus the four-component adaptive reference. No universal dominance,
  causal transport premium or irreducible-error floor is claimed.
- Independently repeated CPU shrinkage: recent **445.758752 → 445.488279**;
  sixteen-month **486.475878 → 486.907127**. Gains +0.270473 / −0.431249,
  both paired intervals crossing zero; not promoted. These are two-head CPU
  comparisons, not the full GPU ensemble.
- Final paper: `final_report/project-report-team44.pdf`, **8 body + 1 references
  + 2 appendix pages**. Names/IDs, team name, AI declaration including the
  model illustration, and actual workload text are present. Team members must
  confirm the human contribution statements.
- Course report deadline: **6 November 2026, 23:59 Singapore time**, confirmed
  on the official Canvas assignment and same-day announcement. ZIP upload,
  one authorised member per team. The Kaggle closing date is separate.
- Reproduction instructions: `docs/REPRODUCIBILITY.md`.
  Numerical evidence: `reports/README.md`.
  Training data, fitted models and detailed prediction caches remain local.

## Research journey and decisions

| Period | Question / work | Observed outcome and decision | Evidence |
| --- | --- | --- | --- |
| 18 Sep | Establish environment, data audit, chronological folds and simple references | Initial CatBoost recent RMSE about 457.08. Raw data frozen; compare against medians and linear/tree baselines. | `experiment_results.csv`, `experiment_metadata.json` |
| 18–21 Sep | Improve preprocessing and spatial/history features | Canonicalise eleven raw flat-type labels to six; preserve repeat-looking transactions; join blocks, MRT, schools and malls. Detailed early-stopped two-seed model reaches 448.06. | `src/features.py`, `experiment_results.csv`, `information_audit.csv` |
| 21 Sep | Are very low rents label errors? | Keep/remove training-label sensitivity yields public 436.75718 / 436.58736: only 0.17 apart. Does not establish erroneous labels or justify deleting scoring rows. Both historical artifacts retained locally. | `kaggle_submissions.csv` |
| 21 Sep | Separate market movement from local rent premiums | Hierarchical market-normalised comparables and direct/residual averaging improve all four initial fixed-budget windows. Four-head recent RMSE 445.5185, public 436.05684. | `temporal_comparison.csv`, `temporal_summary.csv` |
| 21–24 Sep | Simulate stale information during training | Six-month history-gap augmentation preserves row weights. Equal original/augmented ensemble improves recent and long windows but worsens the old year. Promotion follows public 432.47187 with the trade-off disclosed. | `src/modeling.py`, `temporal_summary.csv` |
| 23–24 Sep | Add local market momentum | Four smoothed town/flat-type trend features in the normal view reach recent 440.7629 and public 432.14881. Exact submitted bytes/models promoted; this remains the mainline. | `configs/model.json`, `final_model_metadata.json` |
| 24 Sep | Does composition control explain local trends? | Earlier joint changes were confounded. Isolated trend-only ablation preserves history levels/residual targets: essentially no recent gain and worse other windows. No promotion; not a rejection of every composition method. | `temporal_comparison.csv` |
| 24 Sep | Learn component weights instead of equal averaging | Four-head/eight-output forward stacking is inconsistent across periods. Recent four-head gain is only 0.106 RMSE with interval crossing zero; public 432.04286 retained as an optional probe. | `stacking_results.csv`, `stacking_sources.csv` |
| 6 Oct | Audit unused fields; test comparables, specialists and cross-fitting | Neighbour, room-type and rent-expert blends trade recent loss for long-window gain. Full-interval cross-fitting loses; strict-past distribution replacement gains only about 0.12/0.14. No replacement. | `information_audit.csv`, `feedback_results.csv` |
| 6 Oct | Add COE, stock prices, alternative learners and residual corrections | Lagged indicators and matched-input XGBoost do not improve both windows. Guarded Ridge stabilises extrapolation but remains period-dependent. Low-rent ranking is weak, not proven impossible. | `feedback_results.csv`, `src/market_experiments.py` |
| 6 Oct | CPI/URA releases and historical MRT access | 36 formal GPU fits compare origin-frozen/monthly-available indices and audited rail access. Corrected static MRT gains 0.1322/0.0169, comparable to refit variation; no stable replacement or causal claim. | `feedback_results.csv`, `configs/mrt_openings.json` |
| 23–29 Oct | Build scientific report and test reproducibility | Official IEEEtran layout; local evidence primary. Canonical-address bootstrap preserves gain directions. Separate checkout/new Python environment completes eight full-budget GPU fits in 679.83 s, recent RMSE 440.6423; not promoted. | `full_budget_reproduction.json`, `pre_submission_checks.json` |
| 29 Oct | Diagnose old-period failure and interchange access | Normal/stale decomposition and earlier-OOF bias/risk corrections do not resolve transfer. Six interchange-access fits fail the recent/long-window screen despite some old-year gain. | `followup_*.csv` |
| 2–3 Nov | Integrate teammate learner/target/housing studies | LightGBM/MLP, log targets, resale/building-stock data, low-rent mixtures, capacity and historical-only augmentation probes add negative evidence, not a new production recipe. Distinct CPU/GPU protocols kept separate. | `november_*.csv`, `rmse_followup_*.csv` |
| 2 Nov | Supplementary public checks | Four-head housing/building-stock and market-normalised probes score 432.70689 / 434.72053. Neither promoted. Public-source target matching is excluded from modeling and public code distribution. | `kaggle_submissions.csv` |
| 3–5 Nov | Learn horizon/support-based view weights | Improve three of seven overlapping comparisons but lose recent/long-window acceptance. Historical dispersion identifies risk; parent-level reference accuracy does not itself imply model improvement. | `history_research/horizon_blend_v1/`, `support_blend_v1/`, `parent_child_reference_v1/` |
| 5–6 Nov | Test stronger finest-level shrinkage on high-dispersion groups | Two windows × original/candidate × direct/residual CPU fits. Original team run gains 0.103 / loses 0.317. Independently repeated run gains 0.270 / loses 0.431; max old/new discrepancy 0.199 cannot be attributed without original caches. Both reject promotion. | `history_research/dispersion_shrinkage_cpu_reproduction_20261106/` |
| 6 Nov | Validate the added research and report claims | Tests cover cache/row/hash matching, forward-label isolation, weighting, shrinkage column scope, fixed residual offsets, bootstrap and error totals. Paper distinguishes risk diagnostics from corrections and observed labels from normative fair rent. | `tests/test_history_research.py`, repetition `audit.json` |
| 6 Nov | Human paper review and template alignment | Accepted prose frozen. Fix first-reference float ordering, body acronym definitions, equation-reference style, proper-noun capitalisation and Figure 2 labels (at least 8 pt). References verified; no model/result change. | `final_report/` |
| 6 Nov | Repair isolated drawing-module import | Package-relative/direct-script imports supported. Existing stacking test now launches three fresh-process checks, so prior test imports cannot conceal the failure. 142 tests pass, four optional skips. | `src/generate_figures.py`, `tests/test_project_features.py` |

All evidence paths in the table are under `reports/` unless qualified otherwise.
Historic early-stopping scores and later fixed-budget scores are deliberately
not treated as one algorithm ranking. Dataset assumptions follow the professor's
clarification: block histories identify comparables, not a verified individual
flat time series. Identical rows are retained.

## What the teammate work contributed

Git history establishes what was shared, not human contribution percentages.

- The 2 November contribution broadened learner/target comparisons, housing
  and resale attributes, low-rent detectability, mixtures, capacity and
  historical augmentation; two requested public probes did not beat the mainline.
- The 5 November contribution added forward horizon/support gates, reliability
  and parent/child reference diagnostics, then the isolated shrinkage experiment.
- Integration preserved the original production pipeline. Executed methods live
  under `src/research/`, numerical evidence stays under `reports/`, and the
  paper organises findings by research question rather than a dated addendum.
- Responsibility statements in the submitted paper are team attestations;
  commit authorship and AI assistance do not independently prove ownership.

## Why model exploration stops here

Candidate gains are small or period-dependent. More tuning would repeatedly
reuse development labels without resolving the older-year loss. Stronger claims
need new observable evidence and additional, preferably non-overlapping periods,
not deleted validation rows, target lookup or a tiny public-score improvement.

## Submission cleanup and preservation

The report PDF now lives beside its source in `final_report/`; the progress PDF
lives in `progress_report/`. Build scripts and links use those stable paths.
No public `output/pdf/` layer, duplicated stage/handoff document or raw cache is
needed. Full detailed pre-cleanup documents, the superseded progress refresher,
a disabled target-visible diagnostic and an abandoned submission prototype have
recoverable local copies in ignored `artifacts/local_archive/`. They are not
necessary for reproducing the legitimate model/experiments reported in the paper.

All files explicitly named by the paper's evidence map are retained at their
original paths, including research modules, result ledgers, configuration and
this progress file. Source and tests supporting unsuccessful studies remain:
negative evidence is part of the research journey, not disposable clutter.
Only superseded notes/prototypes were removed from public distribution.

### Post-cleanup verification — 6 November 2026

- A separate source copy passed **142 tests, 4 optional skips** in 53.79 seconds.
  Course data and frozen prediction caches were linked only into that local
  verification copy, not into the school source package.
- All 20 explicit paper file/directory references resolve. The 19 distributable
  references are also present in the source package; `data/raw/` remains private
  and its restoration layout is documented. All eight regenerated numeric/table
  files match the reviewed originals exactly after normalising line endings.
- Recompilation succeeded at **11 pages: 8 body, 1 references, 2 appendices**.
  Extracted text matches the reviewed PDF on every page; the LaTeX log contains
  no overfull-box, undefined-reference or multiply-defined-reference warnings.
  The supplied final PDF is the unchanged reviewed original, not the scratch build.
- An isolated GPU smoke run completed all **8 components × 5 trees**, using
  150,000 original training rows and producing 50,000 predictions in 145.3 seconds.
  This checks execution, not full-budget scientific reproduction. The earlier
  full-budget repeat remains separately recorded in
  `reports/full_budget_reproduction.json`.
- The selected recipe, production training/feature code, model provenance and
  both protected submission CSV hashes are unchanged by this cleanup.

GitHub and the school package share the same distributable source tree. The
school ZIP additionally includes the reviewed PDF at its top level; private raw
data, model binaries, caches, environments and local archives are excluded.
No new scientific result or automatic Canvas upload is implied. Final member
approval and the actual upload remain the team's responsibility.
