"""Create report-ready plots from the stage 2 experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    results = pd.read_csv(REPORTS / "stage2_experiment_results.csv")

    latest = results.loc[results["fold"] == "2025-01_to_2025-03"].copy()
    latest["label"] = latest["model"].str.replace("_", " ").str.title()
    latest.loc[latest["feature_set"] == "core_spatial", "label"] += " + spatial"
    latest = latest.sort_values("rmse", ascending=False)
    plt.figure(figsize=(8.5, 4.8))
    ax = sns.barplot(data=latest, x="rmse", y="label", color="#3D7EA6")
    ax.bar_label(ax.containers[0], fmt="%.1f", padding=4)
    plt.xlabel("RMSE (SGD; lower is better)")
    plt.ylabel("")
    plt.title("Latest chronological validation window (Jan-Mar 2025)")
    save("stage2_latest_model_comparison.png")

    catboost = results.loc[results["model"] == "catboost"].copy()
    catboost["Features"] = catboost["feature_set"].map(
        {"core": "Core", "core_spatial": "Core + spatial"}
    )
    plt.figure(figsize=(8.5, 4.8))
    sns.lineplot(
        data=catboost,
        x="fold",
        y="rmse",
        hue="Features",
        marker="o",
        linewidth=2,
    )
    plt.xlabel("Validation window")
    plt.ylabel("RMSE (SGD)")
    plt.title("Spatial-feature ablation across rolling time folds")
    plt.xticks(rotation=15)
    save("stage2_spatial_ablation.png")

    groups = pd.read_csv(REPORTS / "stage2_subgroup_errors.csv")
    towns = groups.loc[(groups["group_column"] == "TOWN") & (groups["rows"] >= 100)]
    towns = towns.nlargest(10, "rmse").sort_values("rmse")
    plt.figure(figsize=(8.5, 5.5))
    ax = sns.barplot(data=towns, x="rmse", y="group_value", color="#C96A4A")
    ax.bar_label(ax.containers[0], fmt="%.0f", padding=4)
    plt.xlabel("RMSE (SGD)")
    plt.ylabel("Town")
    plt.title("Highest-error towns (at least 100 validation rows)")
    save("stage2_high_error_towns.png")
    print(f"Wrote stage 2 figures to {FIGURES}")


if __name__ == "__main__":
    main()
