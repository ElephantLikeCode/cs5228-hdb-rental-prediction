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
        & (
            (results["model"] != "catboost")
            | (results["feature_set"] == "all_spatial_history")
        )
    ].copy()
    latest["label"] = latest["model"].str.replace("_", " ").str.title()
    latest.loc[latest["model"] == "catboost", "label"] = "CatBoost (seed 42)"
    latest.loc[
        latest["model"] == "catboost_seed_ensemble", "label"
    ] = "CatBoost (2-seed ensemble)"
    latest = latest.sort_values("rmse", ascending=False)
    plt.figure(figsize=(8.5, 4.8))
    ax = sns.barplot(data=latest, x="rmse", y="label", color="#3D7EA6")
    ax.bar_label(ax.containers[0], fmt="%.1f", padding=4)
    plt.xlabel("RMSE (SGD; lower is better)")
    plt.ylabel("")
    plt.title("Latest chronological validation window (Jan-Mar 2025)")
    save("latest_model_comparison.png")

    catboost = results.loc[
        results["protocol"].isin(
            ["rolling_stability", "rolling_history_single", "rolling_selected_pipeline"]
        )
    ].copy()
    catboost["Features"] = catboost["feature_set"].map(
        {"core": "Core", "all_spatial": "Core + all spatial"}
    )
    catboost.loc[
        catboost["protocol"] == "rolling_history_single", "Features"
    ] = "Spatial + history (seed 42)"
    catboost.loc[
        catboost["protocol"] == "rolling_selected_pipeline", "Features"
    ] = "Spatial + history (2-seed ensemble)"
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
    plt.title("Selected feature sets across rolling time folds")
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

    groups = pd.read_csv(REPORTS / "error_analysis.csv")
    rent_bands = groups.loc[groups["group_column"] == "RENT_BAND"].copy()
    rent_bands["group_value"] = pd.Categorical(
        rent_bands["group_value"],
        categories=["<=2500", "2501-3000", "3001-3500", "3501-4000", ">4000"],
        ordered=True,
    )
    rent_bands = rent_bands.sort_values("group_value")
    plt.figure(figsize=(8.5, 5.5))
    ax = sns.barplot(data=rent_bands, x="group_value", y="rmse", color="#C96A4A")
    ax.bar_label(ax.containers[0], fmt="%.0f", padding=4)
    plt.xlabel("Actual monthly rent band (SGD)")
    plt.ylabel("RMSE (SGD)")
    plt.title("Selected ensemble error by rent band")
    save("error_by_rent_band.png")
    print(f"Wrote current experiment figures to {FIGURES}")


if __name__ == "__main__":
    main()
