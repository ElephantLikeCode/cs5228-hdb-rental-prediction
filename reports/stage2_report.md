# Stage 2: Validation, Baselines, and First Spatial Ablation

Run date: 18 September 2026

## Experimental design

Three expanding chronological folds were used, each with a three-month validation
window:

1. July-September 2024;
2. October-December 2024;
3. January-March 2025.

All features are available at prediction time. Spatial features are joined through
normalized town/block/street keys with complete coverage. MRT distance uses only
stations marked open in the supplied data.

## Results

| Model | Features | Validation scope | RMSE |
| --- | --- | --- | ---: |
| Global median | Raw | Jan-Mar 2025 | 844.60 |
| Town + flat-type median | Raw | Jan-Mar 2025 | 546.81 |
| Ridge | Core | Jan-Mar 2025 | 520.01 |
| Decision Tree | Core | Jan-Mar 2025 | 485.90 |
| Random Forest | Core | Jan-Mar 2025 | 470.19 |
| CatBoost | Core | Jan-Mar 2025 | 456.66 |
| CatBoost | Core + spatial | Jan-Mar 2025 | **453.10** |

Across all three folds:

| CatBoost features | Mean RMSE | RMSE standard deviation | Mean MAE |
| --- | ---: | ---: | ---: |
| Core | 461.07 | 14.75 | 340.33 |
| Core + spatial | **458.58** | 16.25 | **337.27** |

The spatial feature set improved every fold, by 0.66, 3.27, and 3.56 RMSE
respectively. The average improvement is 2.50 RMSE (0.54%). This is a stable but
modest gain, so the next experiment should separate HDB block attributes from each
amenity group rather than treating the combined set as proven in full.

## Error findings

The latest spatial model has a mean signed error of -48.99 SGD, indicating modest
systematic underprediction. Among flat types with meaningful sample sizes,
Executive flats are hardest (RMSE 535.69). High-error towns include Jurong West,
Kallang/Whampoa, Bukit Merah, Toa Payoh, and Bishan.

## Interpretation

The model ranking is consistent with the task: a linear model cannot adequately
capture location/flat/time interactions, a single tree has high variance, and the
ensembles perform better. CatBoost remains justified because of the mixed numerical
and high-cardinality categorical inputs. Spatial context adds information beyond the
address category, but most remaining error is not explained by the first nearest-
facility distances.

## Next hypotheses

1. Separate block properties, MRT, malls, and schools in controlled ablations.
2. Add facility counts within fixed radii and distance to the CBD.
3. Test a leakage-safe time trend or past-only location rent summaries.
4. Diagnose high-error towns and Executive flats before hyperparameter tuning.
5. Evaluate leaderboard performance only after team review; keep local rolling
   validation as the main model-selection evidence.
