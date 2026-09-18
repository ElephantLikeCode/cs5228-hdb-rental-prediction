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
- confirmed report dates and Kaggle rules;
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
and 455.70 respectively without changing the feature set.

Feature groups, in priority order:

1. Time: year, month, monotonic month index, quarter.
2. Flat: normalized flat type/model, floor area, lease age.
3. Block: coordinates, completion year, maximum floor, subzone, planning area, region.
4. Spatial: distance to nearest open MRT, mall, primary school, CBD, and counts within selected radii.
5. Historical group statistics: past-only rent summaries for address, town, town-flat-type, flat type, and flat model.

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

## 8. Calendar

| Dates | Main target |
| --- | --- |
| 16-20 Sep | Finish audit, chronological split utilities and core EDA |
| 21-27 Sep | Global/group baselines, kNN, linear regression and regression tree |
| 28 Sep-4 Oct | Bagging/boosting baseline, initial block/spatial features and ablations |
| 5-7 Oct | Freeze results and finish progress-report slides |
| 8 Oct | Submit progress report by 11:59 pm |
| 9-18 Oct | Refine spatial features and evaluate optional historical summaries |
| 19-25 Oct | Bounded tuning and model comparison; optional advanced model only if justified |
| 26-30 Oct | Error analysis, final figures and report draft |
| 31 Oct-3 Nov | Freeze model and complete final report |
| 4-5 Nov | Clean-environment reproduction and packaging |
| 6 Nov | Submit final report/code by 11:59 pm |

## 9. Immediate implementation queue

Completed checkpoint: official Kaggle audit and data verification, data-contract
tests, three rolling folds, EDA, course-aligned model comparison, strict spatial
ablations, past-only rent statistics, segmented errors, and a validated submission generator.

1. Have every member reproduce `check_setup.py`, eight tests, and the current experiment results.
2. Review the first public score (438.94488) and long-horizon results as a team.
3. Investigate the strong regression-to-the-mean pattern in low/high rent bands.
4. Test COE and stock features through isolated chronological ablations.
5. Freeze the strongest stable pipeline before preparing the final report.
