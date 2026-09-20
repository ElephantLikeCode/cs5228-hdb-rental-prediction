"""Update current validated results inside the single cumulative progress file."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROGRESS = ROOT / "docs" / "PROGRESS.md"
START = "<!-- CURRENT_RESULTS_START -->"
END = "<!-- CURRENT_RESULTS_END -->"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows: list[str] = []
    for _, row in frame[columns].iterrows():
        values = []
        for value in row:
            values.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def current_results_text() -> str:
    results = pd.read_csv(REPORTS / "experiment_results.csv")
    metadata = json.loads(
        (REPORTS / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    latest_fold = metadata["folds"][-1]["name"]
    latest = results.loc[
        (results["fold"] == latest_fold)
        & results["protocol"].isin(
            [
                "latest_fold_comparison",
                "latest_spatial_ablation",
                "rolling_history_single",
                "rolling_selected_pipeline",
            ]
        )
    ].sort_values("rmse")
    rolling = results.loc[
        results["protocol"].isin(
            ["rolling_stability", "rolling_history_single", "rolling_selected_pipeline"]
        )
    ].copy()
    rolling["pipeline"] = rolling["feature_set"]
    rolling.loc[
        rolling["protocol"] == "rolling_history_single", "pipeline"
    ] = "detailed_comparables_seed42"
    rolling.loc[
        rolling["protocol"] == "rolling_selected_pipeline", "pipeline"
    ] = "detailed_comparables_2seed_ensemble"
    rolling_summary = (
        rolling.groupby("pipeline", as_index=False)
        .agg(
            runs=("fold", "count"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_bias=("bias", "mean"),
        )
        .sort_values("mean_rmse")
    )
    groups = pd.read_csv(REPORTS / "error_analysis.csv")
    rent_bands = groups.loc[groups["group_column"] == "RENT_BAND"]
    address_status = groups.loc[groups["group_column"] == "ADDRESS_STATUS"]
    long_horizon = pd.read_csv(REPORTS / "long_horizon_results.csv")
    long_selected = long_horizon.loc[
        long_horizon["feature_set"].str.contains("history")
    ]
    best = latest.iloc[0]
    submissions = pd.read_csv(REPORTS / "kaggle_submissions.csv")
    submission_table = markdown_table(submissions, ["submitted_at", "submission_file", "public_rmse"])
    return f"""{START}
## Reference evaluation (historical early-stopping protocol)

Three expanding chronological folds are used, with every validation month
strictly later than its training months. Linear Regression, Ridge, Decision
Tree, Random Forest, HistGradientBoosting, and CatBoost share a common latest
comparison window. Historical comparable-rent statistics use only strictly
earlier months; spatial features never use the target.

Best latest-window candidate: **{best['model']} with {best['feature_set']}**,
RMSE **{best['rmse']:.2f} SGD**, MAE **{best['mae']:.2f} SGD**, signed bias
**{best['bias']:.2f} SGD**.

### Latest-window comparison (January-March 2025)

{markdown_table(latest, ['model', 'feature_set', 'rmse', 'mae', 'r2', 'bias', 'runtime_seconds'])}

### Rolling CatBoost stability

{markdown_table(rolling_summary, ['pipeline', 'runs', 'mean_rmse', 'std_rmse', 'mean_mae', 'mean_bias'])}

### Error by actual rent band

{markdown_table(rent_bands, ['group_value', 'rows', 'rmse', 'mae', 'bias'])}

### Seen versus unseen addresses

{markdown_table(address_status, ['group_value', 'rows', 'rmse', 'mae', 'bias'])}

### Twelve-month robustness

{markdown_table(long_selected, ['model', 'feature_set', 'fold', 'rmse', 'mae', 'bias'])}

### Recorded public submissions

{submission_table}

These reference metrics use validation-driven early stopping. Fixed-budget
comparisons and adoption decisions are recorded in the cumulative entries below
and in `reports/temporal_comparison.csv`. Do not mix the two protocols.
<!-- CURRENT_RESULTS_END -->"""


def main() -> None:
    text = PROGRESS.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError("Current-results markers are missing")
    before, remainder = text.split(START, maxsplit=1)
    previous, after = remainder.split(END, maxsplit=1)
    # One-time migration: preserve manually written findings outside the
    # generated section, so future refreshes cannot erase the project history.
    if "### Findings" in previous:
        findings = previous.split("### Findings", maxsplit=1)[1].strip()
        after = "\n\n## Recorded findings before fixed-budget evaluation\n\n" + findings + "\n" + after
    PROGRESS.write_text(before + current_results_text() + after, encoding="utf-8")
    print(PROGRESS)


if __name__ == "__main__":
    main()
