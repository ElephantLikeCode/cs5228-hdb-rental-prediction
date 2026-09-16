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

Status: mostly complete.

Deliverables:

- confirmed Canvas requirements;
- confirmed report dates and list of unresolved Kaggle rules;
- isolated, shareable folder;
- data-sharing and leakage rules.

Exit condition: the official Kaggle link, evaluation metric and submission rules have been added when released.

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

Planned outputs:

- target distribution and monthly trend;
- rent by town, flat type, floor area, flat model, and lease age;
- record counts and category drift over time;
- analysis of exact-feature duplicates with different target values;
- chronological holdout and rolling-origin splits.

Primary validation proposal:

- development training: 2021-01 through 2024-09;
- main validation: 2024-10 through 2025-03;
- two additional rolling backtests using earlier six-month validation windows;
- report mean and variability across windows.

This split is provisional until the Kaggle metric is published. Random cross-validation may be retained as a secondary diagnostic but not the main evidence.

### Phase 3 - Baselines and early strong reference

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

Feature groups, in priority order:

1. Time: year, month, monotonic month index, quarter.
2. Flat: normalized flat type/model, floor area, lease age.
3. Block: coordinates, completion year, maximum floor, subzone, planning area, region.
4. Spatial: distance to nearest open MRT, mall, primary school, CBD, and counts within selected radii.
5. Optional historical group statistics: past-only rent summaries for town, address, town-flat-type, and area band.

Historical rent summaries are optional because they can easily leak the target. If tested, training rows must be constructed without using their own target and validation/test rows may use only earlier dates.

Exit condition: each feature group has a documented ablation result.

### Phase 5 - Model comparison and tuning

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

1. Run and review the data audit.
2. Confirm the official Kaggle metric/link and exact deadlines when released.
3. Implement chronological split utilities and EDA figures.
4. Implement global/hierarchical median, kNN, linear regression and regression-tree baselines.
5. Implement course-covered ensemble baselines and the submission validator.
6. Add block and distance features with ablations.
7. Train an early CatBoost reference and retain or replace it according to chronological validation, interpretability and runtime.
