"""Create report-ready plots from the current experiment outputs."""

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
    results = pd.read_csv(REPORTS / "experiment_results.csv")

    latest = results.loc[
        (results["fold"] == "2025-01_to_2025-03")
        & ((results["model"] != "catboost") | (results["feature_set"] == "all_spatial"))
    ].copy()
    latest["label"] = latest["model"].str.replace("_", " ").str.title()
    latest.loc[latest["feature_set"] == "all_spatial", "label"] += " + spatial"
    latest = latest.sort_values("rmse", ascending=False)
    plt.figure(figsize=(8.5, 4.8))
    ax = sns.barplot(data=latest, x="rmse", y="label", color="#3D7EA6")
    ax.bar_label(ax.containers[0], fmt="%.1f", padding=4)
    plt.xlabel("RMSE (SGD; lower is better)")
    plt.ylabel("")
    plt.title("Latest chronological validation window (Jan-Mar 2025)")
    save("latest_model_comparison.png")

    catboost = results.loc[
        (results["model"] == "catboost")
        & (results["feature_set"].isin(["core", "all_spatial"]))
    ].copy()
    catboost["Features"] = catboost["feature_set"].map(
        {"core": "Core", "all_spatial": "Core + all spatial"}
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
    plt.title("Core versus spatial features across rolling time folds")
    plt.xticks(rotation=15)
    save("rolling_stability.png")

    ablation = results.loc[
        (results["model"] == "catboost")
        & (results["protocol"] == "latest_spatial_ablation")
    ].copy()
    ablation["Features"] = ablation["feature_set"].map(
        {
            "core": "Core",
            "block": "Core + block",
            "block_mrt": "Block + MRT",
            "block_mall": "Block + mall",
            "block_school": "Block + school",
            "all_spatial": "All spatial",
        }
    )
    ablation = ablation.sort_values("rmse", ascending=False)
    plt.figure(figsize=(8.5, 4.8))
    ax = sns.barplot(data=ablation, x="rmse", y="Features", color="#5B8E7D")
    ax.bar_label(ax.containers[0], fmt="%.1f", padding=4)
    plt.xlabel("RMSE (SGD; lower is better)")
    plt.ylabel("")
    plt.title("Strict spatial ablation (Jan-Mar 2025)")
    save("spatial_ablation.png")

    groups = pd.read_csv(REPORTS / "subgroup_errors.csv")
    towns = groups.loc[(groups["group_column"] == "TOWN") & (groups["rows"] >= 100)]
    towns = towns.nlargest(10, "rmse").sort_values("rmse")
    plt.figure(figsize=(8.5, 5.5))
    ax = sns.barplot(data=towns, x="rmse", y="group_value", color="#C96A4A")
    ax.bar_label(ax.containers[0], fmt="%.0f", padding=4)
    plt.xlabel("RMSE (SGD)")
    plt.ylabel("Town")
    plt.title("Highest-error towns (at least 100 validation rows)")
    save("high_error_towns.png")
    print(f"Wrote current experiment figures to {FIGURES}")


if __name__ == "__main__":
    main()
