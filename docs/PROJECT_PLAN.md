# Project Plan

## 1. Project objective

Build a reproducible regression pipeline that predicts the monthly rent of an HDB flat from its date, location, flat characteristics, block information, nearby amenities, and selected economic indicators. The report must emphasize understanding, methodological justification, evaluation, interpretation, and limitations rather than leaderboard rank alone.

## 2. Definition of success

The project is successful when it produces all of the following:

1. A leak-resistant local validation design that reflects future-month prediction.
2. At least one transparent baseline and one competitive nonlinear model.
3. A documented data-cleaning and feature-engineering pipeline.
4. An ablation study showing which feature groups help.
5. Error analysis by time, town, flat type, price band, and seen/unseen location.
6. A valid Kaggle submission file once the competition opens.
7. A reproducible repository and an 8-page final report.
8. A 10-15 slide progress report with concrete findings and next steps.

## 3. Guiding principles from CS5228

- Let EDA inform preprocessing and model choices.
- Compare alternatives and justify decisions.
- Use an experimental setup and metrics appropriate to the task.
- Interpret failures and limitations, not only aggregate scores.
- Prefer a clear, reproducible method over an unexplained leaderboard improvement.
- Use additional data only when it is available at prediction time and permitted by the competition.

## 4. Phased implementation

### Phase 0 - Requirements and governance

Status: complete. Kaggle overview, data, rules, leaderboard protocol, and
discussion page were checked on 18 September 2026; see `docs/KAGGLE_AUDIT.md`.

Deliverables:

- confirmed Canvas requirements;
- documented report-date status and confirmed Kaggle rules (final date still pending);
- isolated, shareable folder;
- data-sharing and leakage rules.

Exit condition met: RMSE, `Id,Predicted`, public/private leaderboard split,
external-data permission, and official data checksums are documented.

### Phase 1 - Data audit and reproducible skeleton

Status: implemented in `src/data_audit.py`.

Tasks:

- verify schemas, sizes, date coverage, missingness, constants, and duplicates;
- validate the example-submission contract;
- inspect train/test category mismatch;
- measure HDB-block join coverage;
- record raw-file hashes;
- establish automated tests.

Exit condition: `python src/data_audit.py` and `python -m pytest` pass on a clean copy.

### Phase 2 - EDA and validation design

Status: first reproducible EDA and three rolling three-month folds implemented.

Planned outputs:

- target distribution and monthly trend;
- rent by town, flat type, floor area, flat model, and lease age;
- record counts and category drift over time;
- analysis of exact-feature duplicates with different target values;
- chronological holdout and rolling-origin splits.

Primary validation design:

- three expanding-window chronological folds;
- each validation window contains three consecutive months;
- latest validation window: 2025-01 through 2025-03;
- report the latest common-window comparison plus rolling mean and variability.

Kaggle confirms RMSE as the metric. Random cross-validation may be retained as
a secondary diagnostic but not the main evidence.

### Phase 3 - Baselines and early strong reference

Status: global/group medians, Linear Regression, Ridge, Decision Tree, Random
Forest, HistGradientBoosting, and CatBoost have been compared on the identical
latest chronological window. CatBoost remains strongest; HistGradientBoosting
is the strongest low-cost course-aligned alternative.

Models:

1. Global median.
2. Hierarchical median using month/trend, town, and flat type with fallbacks.
3. Linear or Ridge Regression as an interpretable parametric baseline.
4. Decision Tree or Random Forest as a tree-based baseline.
5. CatBoost as an early strong reference because the dataset contains many mixed and high-cardinality categorical attributes.
6. k-Nearest Neighbour only as a bounded pilot if scaling/encoding and runtime are reasonable; it is not required merely because it appears in the syllabus.

Required outputs:

- validation score and runtime;
- residual distribution;
- subgroup errors;
- one valid submission generator.

Exit condition: a single command can train, evaluate, save the model metadata, and create a schema-valid submission.

### Phase 4 - Core feature engineering

Status: strict spatial ablations implemented for block-only, block + MRT,
block + mall, block + school, and the combined set. Distance and 0.5/1/2 km
facility-count features are included. Past-only rent statistics were then added
for address, town-flat-type, town, flat type, and flat model. The selected
pipeline reduced latest-window RMSE from 452.62 to 450.04 and three-fold mean
RMSE from 457.81 to 456.44. A two-seed ensemble then reduced these to 449.20
and 455.70. Canonical category cleaning and finer past-only comparables by
address/flat type/area, street/flat type, and town/flat type/model subsequently
reduced them to 448.06 and 454.31.

Feature groups, in priority order:

1. Time: year, month, monotonic month index, quarter.
2. Flat: normalized flat type/model, floor area, lease age.
3. Block: coordinates, completion year, maximum floor, subzone, planning area, region.
4. Spatial: distance to nearest open MRT, mall, primary school, CBD, and counts within selected radii.
5. Historical group statistics: past-only rent summaries for address, address-flat-type-area, street-flat-type, town-flat-type-model, town, flat type, and flat model.

Historical rent summaries are implemented with strict monthly cutoffs: training
rows cannot use their own month, while validation and test rows can use only
strictly earlier reference months.

Exit condition: each feature group has a documented ablation result.

### Phase 5 - Model comparison and tuning

Status: model comparison complete. A bounded iteration test improved the latest
window from 450.73 at 1,200 trees to 450.04 at 1,600 trees. Learning-rate and
regularization testing at 2,200 trees did not improve the retained configuration.
Equal-weight averaging of seeds 42 and 2026 subsequently improved all three
short validation folds and both 12-month robustness folds, so it is retained.
The selected implementation now uses GPU CatBoost for faster iteration. XGBoost
native categorical training and CatBoost/XGBoost blends were tested and rejected;
the XGBoost result was materially worse and residuals were highly correlated.

Candidate models:

- global/group baselines plus Linear/Ridge Regression for transparent reference points;
- Decision Tree or Random Forest for an interpretable tree comparison;
- CatBoost as a leading candidate, motivated by mixed numeric and high-cardinality categorical attributes;
- LightGBM/XGBoost as alternative boosted-tree candidates if encoding choices and expected value justify the extra comparison;
- kNN as an optional diagnostic rather than a compulsory model, given the dataset size and categorical dimensionality.

The project does not restrict methods to those taught in class. Model inclusion is determined by task fit, methodological value, validation evidence, implementation cost, and the eight-page report limit. Advanced methods must be motivated and referenced, as required by the project page.

Tuning policy:

- tune only after the baseline and validation design are stable;
- use bounded searches and record every run;
- choose the final model from rolling validation, not one lucky split;
- consider blending only if residuals are complementary.

### Phase 6 - Optional economic features

Evaluate COE and stock-derived monthly indicators only after the core pipeline is stable.

Requirements:

- aggregate to information available by the rent approval month;
- avoid using future values;
- compare against a no-economic-feature model;
- remove the features if gains are negligible or unstable.

### Phase 7 - Interpretation and robustness

Required analyses:

- regression coefficients, tree importance and residual analysis; SHAP only as an optional advanced extension;
- error by month, town, flat type, rent band, and sample density;
- performance on new versus previously seen addresses;
- sensitivity to outliers and duplicate observations;
- discussion of unobserved variables such as unit floor, condition, furnishing, and negotiation.

### Phase 8 - Reports and submission

Progress report outline:

1. Problem and stakeholders.
2. Dataset and prediction setup.
3. Important EDA findings.
4. Cleaning decisions.
5. Validation design.
6. Baseline methods and initial results.
7. Challenges and risks.
8. Planned feature/model experiments.
9. Final 1-2 slides: next steps and schedule.

Final report outline:

1. Abstract.
2. Motivation and problem formulation.
3. Data and exploratory analysis.
4. Preprocessing and feature engineering.
5. Methods and experimental setup.
6. Results, ablations, and interpretation.
7. Error analysis, limitations, and future work.
8. Conclusion.
9. References and a minimal appendix, including workload breakdown.

## 5. Suggested four-person ownership

| Workstream | Primary owner | Reviewer | Main outputs |
| --- | --- | --- | --- |
| Data audit, cleaning, validation | Member A | Member C | pipeline, schemas, splits |
| Spatial and auxiliary features | Member B | Member A | joins, distances, ablations |
| Models and tuning | Member C | Member B | course baselines, ensembles, comparisons |
| Evaluation, report, reproducibility | Member D | All | figures, report, run instructions |

Everyone should contribute code or analysis and review at least one other workstream. Ownership is not a silo: major design decisions should be discussed by the full team.

## 6. Meeting and review cadence

- One short technical sync and one results/report sync each week.
- Freeze new feature ideas at least seven days before the final deadline.
- Freeze model training at least three days before the deadline.
- Reserve the final two days for reproduction, report checks, and packaging.

## 7. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Deadline/metric not yet published | Keep metric configurable and track Canvas/Kaggle updates |
| Random-split score inflation | Use chronological rolling validation |
| Target leakage in rent aggregates | Past-only and out-of-fold construction |
| Category formatting drift | Central normalization functions and tests |
| Auxiliary features consume too much time | Baseline first; feature-group ablations |
| One member becomes a bottleneck | Small interfaces, code review, reproducible commands |
| Report becomes an afterthought | Draft figures and findings throughout the project |
| Public data reveals hidden test labels | Do not reconstruct targets from post-release sources |

## 8. Course requirements and calendar

Rechecked read-only on 21 September 2026 against
https://canvas.nus.edu.sg/courses/93866/pages/project.
Methodological quality is 60%, report quality 30%, reproducibility 10%.
The progress report is approximately 10-15 PDF slides, ending with 1-2 slides of
next steps. It is not explicitly graded, but omission negatively affects the
final grade. The final paper has at most eight main-text pages (references and
appendix excluded), includes workload attribution, and is submitted with code
or a GitHub link. Ranking alone is not the course objective.

The former 8 October / 6 November dates below were unverified planning entries;
they are superseded, not official deadlines.

| Timing | Main target / status |
| --- | --- |
| Completed through 24 Sep | Audit, EDA, baselines, spatial ablations, chronological comparisons, 432.14881 mainline |
| Before / around 4 Oct | Prepare 10-15 slide progress-report PDF from existing evidence |
| Sun 4 Oct | Canvas progress-report target; explicitly not a hard deadline |
| After progress feedback | Resume stability/model experiments when requested; maintain one mainline and justified alternative |
| Final report deadline | Canvas still says `Nov XX, 11.59 pm`; exact date unconfirmed |
| Before confirmed final deadline | Reserve time for freezing, clean reproduction, report review and packaging |

## 9. Current checkpoint and deferred queue

Best recorded public candidate: fixed recent-window four-head stacking,
user-reported RMSE 432.04286 on 24 September, a 0.10595 improvement. It reuses the
eight mainline CBMs with Oct-Dec 2024 weights and is retained without promotion.

Mainline: eight-component local-trend/history-diversity ensemble, user-reported
public RMSE 432.14881 on 24 September. Default config, final submission, eight
CBMs and metadata agree. The exact 436.05684 reference and 432.47187
history-diversity run are retained locally, alongside the unchanged Keep/Remove
submissions. Promotion is a decision based on current evidence, not a claim of
uniform temporal improvement: the older 12-month deficit versus the
four-component reference remains 9.81 RMSE.

The 23 September momentum experiment added four past-only town/flat-type trend
features and produced a locally stronger candidate. Its public RMSE of 432.14881
then improved the 432.47187 reference by 0.32306 and it was promoted. A follow-up composition
control using area and lease-age cells was rejected because it was less stable
than that candidate. That joint experiment also changed historical baselines;
it was not a pure trend ablation. The subsequent isolated trend-only ablation
preserved those baselines exactly but lost to the retained candidate on three
of four windows. A current-feature GPU XGBoost depth-6/8 direct/residual screen
and fixed 25% cross-learner blends also failed their two-window screen.
Thirty-three tests cover features, validation, mixed-history outputs, isolated
target preservation, category mappings and model reloads. The two requested
follow-up comparisons are complete; neither produced an additional submission.

A leakage-safe stacking follow-up now compares four seed-averaged heads with all
eight individual model predictions over four matched three-month transfers and
two matched twelve-month transfers. Exact source runs and prediction hashes are
frozen in `reports/stacking_sources.csv`. The half-shrunk eight-input method gained
7.21 RMSE on the first annual transfer but failed to transfer, losing 0.89 on the
next annual window; all learned variants lost on the final two quarters.
The equal-weight 432.14881 pipeline therefore remains the mainline.

A recency follow-up reused the frozen predictions. Learning only from the latest
meta-training window produced a small +0.106 RMSE four-head result on Jan-Mar
2025, but its interval crossed zero and it lost on the preceding quarter and the
recent annual window. Eight-input recent-only and fixed 12-month-decay variants
also failed in the tested configurations. Other recency schedules and regime
models remain untested; the result does not rule them out.

A 16-month forward check was then added to approximate the 17-month competition
test horizon. Recent-window four-head stacking lost 0.538 RMSE and its eight-input
version lost 1.189, with both address-cluster intervals entirely below zero. The
result record now reports each method's actual fit dates, rows, sources and fit
RMSE. The stress check also changes the fitting window to Apr 2022-Mar 2023
(twelve months), ending eight months before the evaluation cutoff. It is not a
controlled test of the recent-quarter rule and cannot establish that a public
leaderboard probe has no value.

The user requested one controlled public-leaderboard probe, so the fixed recent-
window four-head candidate is available as
`submissions/submission_recent_window_stacking.csv`. Treat its public score as
supporting evidence alongside the negative 16-month stress result; do not promote
it solely from a small public-half improvement. This probe freezes Oct-Dec 2024
weights rather than refitting the second layer on Jan-Mar 2025. Its user-reported
public RMSE is 432.04286. Candidate generation requires fresh CSV and metadata paths to preserve
already-generated artifacts.

When the user resumes work:
1. Preserve the 432.14881 mixed-view run as the current mainline and use public
   leaderboard results only as supporting evidence.
2. Retain the 432.04286 fixed-weight candidate as the public-best alternative.
   First decompose its cached OOF gains/losses by month, town, flat type, price
   band (diagnostic only) and history support. Assess systematic signed error as
   well as RMSE; do not tune rules using public scores or target-only groups.
3. For one bounded stacking follow-up, predefine the latest-quarter four-head
   refit with the same 50% shrinkage. Evaluate equivalent historical refit rules
   on quarterly and longer horizons, including recent-quarter fitting at the
   long-horizon origin. Compare with equal weights and the fixed probe, then
   refit on Jan-Mar 2025 OOF predictions only for deployment. Do not repeatedly
   search public-leaderboard weights. The existing annual-fit long-horizon test
   alone cannot decide this quarterly-refit question.
4. Prioritize additional predictive information over many small weight changes:
   test a past-only nearest-comparable rent-premium predictor using matching
   flat type, similar area, nearby blocks and recency. Benchmark it alone and
   with one fixed small blend on the same rows. Existing hierarchical history
   features already cover much of this information, so verify incremental value.
5. Investigate a model-based, past-only correction for market regime change;
   the tested unshrunk previous-year horizon offsets failed because bias changed
   sign. Other temporal calibration methods remain untested, not disproven.
6. Reproduce setup/tests as a team, confirm member ownership, and prepare the
   progress slides from EDA, cleaning, baselines, ablations,
   quantitative gains, negative results and unresolved limitations.
7. Audit auxiliary snapshot timing: MRT `open` status is not historical
   opening-date filtering. Record this limitation rather than claiming full
   point-in-time availability.
8. Reconfirm the final report deadline and reproduce/package the chosen pipeline.

The single cumulative technical record remains `docs/PROGRESS.md`; README keeps
one main comparison table. Historical reports are evidence, not active recipes.

## Feedback-driven priority update - 6 October 2026

Stacking stays as the retained alternative; additional weight searches are low
priority. The bounded nearest-comparable, flat-type specialist and probabilistic
rent-mixture screen is complete: none passed the recent-and-long-window gate.
Keep the current mainline and do not extrapolate these failures to every possible
configuration of these methods. Explain gains
by observable groups and distinguish those groups from target-price diagnostics.
Only candidates with a meaningful recent improvement and consistent long-window
evidence advance to a second seed and potential mainline replacement. Keep all
raw labels and identical rows in the primary comparison. Record rejected methods
as well as accepted ones in the cumulative PROGRESS record.

Start the final report alongside experiments: clarify the absence of individual-
flat IDs, native CatBoost versus classical-model encoding, precise candidate
definitions, and the limits of chronological and identical-input diagnostics.
Professor feedback is advice rather than a requirement to adopt a particular model.

Next priority is an isolated test of inner-fold-excluded comparable-flat premium
and dispersion features. Outer validation labels remain completely unavailable
to encoders and fitting; all input-only flat/location groupings are explicitly
not unique-flat identities. Compare the same original history/target recipe
with and without these features before changing any other component. Diagnose
low-rent overprediction and total squared-error contribution rather than focusing
only on the highest-RMSE or executive-flat subgroup. Neither a public score near
173 nor the validation-label oracle diagnostic establishes a feasible score for
our current inputs. The full completed screen and numerical evidence are in
`docs/PROGRESS.md` and `reports/feedback_results.csv`.

The user has completed the audit/cross-fitting priority. The aggregate audit is now
available in `reports/information_audit.csv`; `--families crossfit` runs the
paired mean, distribution and strict-past-statistic controls through the existing
experiment entry point. Preserve the 5-fold assignments and source snapshots
locally for report reproducibility. The protocol and relative-price error
diagnostic are appended to PROGRESS. The two-window seed-42 screen is complete:
full-interval cross-fitted pairs lost substantially, and strict-past-statistic
replacement gained only 0.122/0.136 RMSE in the eight-component ensemble. None
passes the advancement gate. Keep the mainline; no new submission is needed for
these configurations. Next test unused supplied COE/stock monthly information in
a bounded market-level study with explicit availability assumptions, followed if
justified by targeted residual correction or a matched independent learner. Do
not interpret this result as rejecting every cross-fitting/encoding method.

## Market/learner/correction study closeout — 6 October 2026

The unused supplied COE/stocks study, matched original/augmented XGBoost and
Ridge comparison, honest OOF residual correction and relative-low-rent mixture
screen are complete on the recent quarter and sixteen-month window. A bounded
Ridge extrapolation follow-up is also complete. No configuration passes the
recent-and-long advancement gate; preserve the current mainline and the fixed
stacking alternative. Do not schedule second-seed confirmation or generate a
submission for these rejected configurations. Exact results, limitations and
50-test QA are appended to the same PROGRESS record and feedback result ledger.

The low-rent event has weak observable ranking signal, but neither tested
correction improves recent RMSE. Do not claim it is completely unpredictable,
automatically erroneous, or solved. Raw rows and labels remain retained.

The official external-source/rules investigation identified overlapping public
HDB rental records; no external rent labels were downloaded or used. Ask the
instructor to clarify whether test-period rental records are permitted before
considering any matching experiment. Do not infer how other teams achieved low
scores. CPI/URA indicators remain optional future work requiring release/revision
timing checks, not a completed imported-data experiment.

Next actions: start the final report from the accumulated positive/negative
evidence; specify candidate definitions, category encoding, comparable-group
identity limitations and macro availability assumptions. Confirm the final
deadline and team ownership. Resume modeling with a new information hypothesis,
not repeated tiny weight or threshold searches. The six original directions
were addressed by audit and bounded experiments; useful low-rent identification
and permission for overlapping external rental labels remain unresolved.

## User-directed closeout — 6 October 2026

This update supersedes the external-label clarification priority above. The user
has chosen not to obtain or match public transaction rental labels covering the
competition test period. Close that route rather than treating permission as a
blocker. This is a project choice, not a claim about other teams or a newly
verified course prohibition.

Keep the mainline and stacking alternative unchanged. Retain code, configurations,
scores and saved predictions for recent/long-window trade-off candidates as
research alternatives, not promoted submission models. Rejected candidates were
not saved as new final CBMs. Do not prioritize further low-rent classifier tuning
without a new observable signal; incorrect labels and unobserved contract terms
remain possible explanations, not established facts. CPI (general price inflation)
and URA private-residential rental indices remain optional aggregate covariates,
not transaction-label lookup, and have not been imported or tested.

No urgent final-report deadline action or new training is scheduled in this
closeout. Continue accumulating evidence for later reporting. MRT opening-date
features remain unimplemented: the supplied table has status but no opening-date
column, and the current pipeline uses a static open-station snapshot.

## CPI/URA and MRT validation closeout — 6 October 2026

The requested aggregate-index and MRT-opening implementation is complete in the
existing experiment pipeline. Both origin-frozen and same-model monthly-input
controls were validated in two windows (36 formal GPU fits; 55 tests pass).
Definitions, results, timing assumptions and GPU refit variation are appended
to PROGRESS and the existing feedback ledger. Corrected static rail gains only
0.132/0.017; other settings lose or show similarly tiny gains. No setting passes
the promotion gate. Retain the mainline, stacking alternative and trade-off
evidence; do not generate another submission for these configurations.

Earlier "CPI/URA not imported" and "MRT dates not implemented" statuses are
superseded by this completed research. The selected production recipe still
uses the supplied spatial snapshot as a proxy; its operational-status limitation
is explicit. No external test-period individual rental labels or further low-
rent classifier fitting were used.

Synchronize maintained code/configuration, summaries, cumulative docs and the
September report source/assets/PDF as authorized. Raw data, index JSONs, models,
submission CSVs, detailed predictions, build intermediates and local AI notes
stay local. Preserve the report as its submitted checkpoint; accumulate October
evidence for later reporting. Future modeling needs a new predictive hypothesis.
If revisiting transit, assess commuting/connectivity or anticipation features
with explicit availability checks; opening-exposure results neither establish
a causal premium nor reject every transit feature.

## Final report draft -- 23 October 2026

The initial scientific-paper draft now exists at
`output/pdf/CS5228_Final_Report.pdf`, with the single evolving source under
`final_report/`. Use the verified official Canvas IEEEtran format. Local temporal
evaluation, subgroup interpretation and limitations lead; public scores are
supplementary. The current eight-page PDF includes the main text through page
seven and an appendix page, within the eight-page body limit.

Workload attribution is reserved in Appendix B, without assigning responsibilities
or percentages. Confirm that information with the team later. All four
surname-first names and student IDs match the progress-report source. Complete
team review before submission; no new submission or training is scheduled.
The course policy's AI-documentation requirement is addressed by a concise
report statement; the local AI log stays ignored. The final deadline remains
unconfirmed in the page and assignment list checked today.

Future experiments should revise this same report and append PROGRESS, preserving
the distinctions between historical early-stopping references, fixed-budget
results, one-seed research screens and untested proposals. Regenerate only the
relevant aggregate figures/tables when underlying evidence changes.

### Report review revision -- 23 October 2026

The initial eight-total-page count above is superseded: the revised paper has
seven main-text pages, one reference page and one appendix page. It adopts the
review's formatting fixes and research-decision narrative, adds a component
diagram and observable-flat-type errors from frozen predictions, and keeps local
development evidence primary. Workload remains reserved for the team. No models,
raw data or selected submissions were changed. The cumulative PROGRESS record
documents the disposition of all 41 review points.
