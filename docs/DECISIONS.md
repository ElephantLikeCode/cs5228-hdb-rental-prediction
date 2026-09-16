# Decision Log

## D001 Isolated workspace

- Date: 2026-09-16
- Decision: Store all new work in `C:\Users\User\CS5228_Project_Workspace` outside OneDrive.
- Reason: The folder can be zipped, shared with team members, or removed without affecting course files.

## D002 Raw data is immutable

- Date: 2026-09-16
- Decision: Never edit `data/raw/`; all derived data goes to `data/processed/`.
- Reason: Reproducibility and safe comparison with the supplied files.

## D003 Chronological validation is primary

- Date: 2026-09-16
- Decision: Use rolling chronological validation as the primary evaluation design.
- Reason: The local training set ends in March 2025 and most test rows are later months.

## D004 Core model before auxiliary complexity

- Date: 2026-09-16
- Decision: Establish transparent descriptive/model baselines and an early task-suitable strong model before spatial/economic enrichment.
- Reason: This creates a useful reference score and prevents auxiliary feature work from blocking progress.

## D005 No reconstruction of hidden labels

- Date: 2026-09-16
- Decision: Do not use post-release public HDB rental records to recover hidden Kaggle labels.
- Reason: Academic integrity and leakage prevention.

## D006 Task-first method selection

- Date: 2026-09-16
- Decision: Use a small set of transparent baselines, then choose the main models according to task fit, validation evidence, interpretability, runtime and report value; methods need not be limited to the syllabus.
- Reason: The professor asks teams to select appropriate techniques and explicitly permits advanced methods when they are motivated and referenced.

## D007 Advanced methods require evidence, not prior exclusion

- Date: 2026-09-16
- Decision: Treat CatBoost as an early leading candidate; consider LightGBM/XGBoost or SHAP only where they answer a distinct need. Historical target-derived encodings remain optional because of leakage risk.
- Reason: Advanced methods are allowed, but every added component must earn its place through a clear hypothesis, references, fair validation and useful interpretation.
