# Course Alignment

This document maps relevant CS5228 concepts to the project without treating the syllabus as a restriction. The professor asks teams to select appropriate techniques and explicitly says that advanced methods or models may be used with relevant references. Task fit and methodological evidence therefore take priority over whether a method appeared in a lecture.

## Core project activities and their course basis

| Project activity | CS5228 basis | Status in our plan |
| --- | --- | --- |
| Data types, quality checks, EDA, cleaning and transformation | Week 1: Introduction and Data Preparation; project marking guide explicitly assesses data preparation and visualization | Required core |
| Regression evaluation and experimental design | Week 4-6: Classification and Regression; project guide explicitly assesses correct experimental setup and metrics | Required core |
| Global/group median baseline | Basic descriptive analysis and a necessary reference model | Required core |
| k-Nearest Neighbour regression | Week 4: k-Nearest Neighbor | Course-model baseline |
| Decision Tree regression | Week 4: Decision and Regression Trees | Course-model baseline |
| Linear Regression | Week 5: Linear models | Course-model baseline |
| Bagging/Random Forest and boosting | Week 6: model ensembles, bagging and boosting | Main nonlinear models after Week 6 |
| Time-aware holdout | Required by the temporal train/test setup and the project demand for correct evaluation; uses a chronological holdout rather than advanced time-series forecasting | Required evaluation safeguard |
| HDB block and landmark features | Project page explicitly supplies block, MRT, mall and school data for enrichment; feature generation is named in report guidance | Required feature-engineering experiment |
| COE and stock indicators | Explicitly supplied by the project page as possible economic indicators | Optional ablation |
| PCA or t-SNE | Week 8: Dimensionality Reduction | Optional visualization/diagnostic only |
| Classical time-series models | Week 11: Time Series Analysis, taught shortly before the final deadline | Optional and low priority |

## Methods deliberately not forced into the project

- Clustering, association-rule mining, recommender systems and graph mining are course topics, but they are not necessary for the stated regression task. We will not insert them merely to demonstrate syllabus coverage.
- NetworkX is a course package, but no graph problem is required by the project. It will not be used unless a clear, useful question emerges.
- Deep learning is outside the stated syllabus focus and is not planned.

## Method-selection policy

The project page permits flexibility and says advanced methods may be used with relevant references. Therefore:

- A small number of transparent baselines will establish reference points; completing every lecture model is not required.
- CatBoost is a justified early candidate because the data contains mixed numeric and high-cardinality categorical attributes.
- LightGBM/XGBoost may be used when their encoding and tuning cost is justified by a distinct experimental question.
- SHAP is optional; the default interpretation uses coefficients, tree importance, partial comparisons and residual plots.
- Leakage-prone target encoding is not a core feature. If historical group-rent statistics are tested, they must be past-only, compared by ablation, and clearly justified.
- Any advanced method included in the report must be motivated, referenced, evaluated on the same chronological splits, and discussed in terms of limitations.

## Recommended report narrative

1. Start from EDA and data-quality findings.
2. Establish a small set of transparent baselines.
3. Explain why nonlinear tree ensembles may fit interactions among location, flat type, area and time.
4. Add project-provided spatial/economic data one feature group at a time.
5. Use ablation and error analysis to justify what remains in the final model.
6. Treat leaderboard performance as supporting evidence, not the project objective.
