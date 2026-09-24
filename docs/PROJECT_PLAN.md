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
Twenty-six tests cover features, validation, mixed-history outputs, isolated
target preservation, category mappings and model reloads. The two requested
follow-up comparisons are complete; neither produced an additional submission.

When the user resumes work:
1. Preserve the 432.14881 mixed-view run as the current mainline and use public
   leaderboard results only as supporting evidence.
2. Keep one mainline and consider one genuinely complementary final candidate,
   without tuning repeatedly against the public leaderboard.
3. Investigate a model-based, past-only correction for market regime change;
   the tested unshrunk previous-year horizon offsets failed because bias changed
   sign. Other temporal calibration methods remain untested, not disproven.
4. Reproduce setup/tests as a team, confirm member ownership, and prepare the
   progress slides from EDA, cleaning, baselines, ablations,
   quantitative gains, negative results and unresolved limitations.
5. Audit auxiliary snapshot timing: MRT `open` status is not historical
   opening-date filtering. Record this limitation rather than claiming full
   point-in-time availability.
6. Reconfirm the final report deadline and reproduce/package the chosen pipeline.

The single cumulative technical record remains `docs/PROGRESS.md`; README keeps
one main comparison table. Historical reports are evidence, not active recipes.
