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
            ["latest_fold_comparison", "latest_spatial_ablation", "rolling_selected_pipeline"]
        )
    ].sort_values("rmse")
    rolling = results.loc[
        (results["model"] == "catboost")
        & results["feature_set"].isin(["core", "all_spatial", "all_spatial_history"])
    ]
    rolling_summary = (
        rolling.groupby("feature_set", as_index=False)
        .agg(
            runs=("fold", "count"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_bias=("bias", "mean"),
        )
        .sort_values("mean_rmse")
    )
    groups = pd.read_csv(REPORTS / "subgroup_errors.csv")
    worst_towns = groups.loc[
        (groups["group_column"] == "TOWN") & (groups["rows"] >= 100)
    ].nlargest(5, "rmse")
    best = latest.iloc[0]
    return f"""{START}
## Current validated pipeline (18 September 2026)

Three expanding chronological folds are used, with every validation month
strictly later than its training months. Linear Regression, Ridge, Decision
Tree, Random Forest, HistGradientBoosting, and CatBoost share a common latest
comparison window. Historical rent statistics use only strictly earlier months;
spatial features never use the target.

Best latest-window candidate: **{best['model']} with {best['feature_set']}**,
RMSE **{best['rmse']:.2f} SGD**, MAE **{best['mae']:.2f} SGD**, signed bias
**{best['bias']:.2f} SGD**.

### Latest-window comparison (January-March 2025)

{markdown_table(latest, ['model', 'feature_set', 'rmse', 'mae', 'r2', 'bias', 'runtime_seconds'])}

### Rolling CatBoost stability

{markdown_table(rolling_summary, ['feature_set', 'runs', 'mean_rmse', 'std_rmse', 'mean_mae', 'mean_bias'])}

### Highest-error towns

{markdown_table(worst_towns, ['group_value', 'rows', 'rmse', 'mae', 'bias'])}

### Findings

- Past-only address, town/flat-type, town, flat-type, and flat-model rent
  statistics improve the selected pipeline without using same-month targets.
- Increasing the selected CatBoost limit from 1,200 to 1,600 trees improved the
  latest window further; two earlier folds stopped early.
- The selected history pipeline improves two of three folds and reduces the
  three-fold mean, although the middle fold is slightly worse.
- The official Kaggle ZIP matches the existing raw files byte-for-byte. Kaggle
  confirms RMSE, `Id,Predicted`, a roughly 50/50 public/private test split, and
  permits additional data collection.
- The test horizon extends well beyond training, so long-horizon trend robustness
  remains a higher priority than broad model search.
<!-- CURRENT_RESULTS_END -->"""


def main() -> None:
    text = PROGRESS.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError("Current-results markers are missing")
    before, remainder = text.split(START, maxsplit=1)
    _, after = remainder.split(END, maxsplit=1)
    PROGRESS.write_text(before + current_results_text() + after, encoding="utf-8")
    print(PROGRESS)


if __name__ == "__main__":
    main()
