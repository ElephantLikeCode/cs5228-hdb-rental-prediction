"""Create report-ready plots from the current experiment outputs."""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from features import suspected_low_rent_mask


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def summarize_temporal(bootstrap=False):
    """Compare exact same rows and seeds using saved out-of-fold predictions."""
    results = pd.read_csv(REPORTS / "temporal_comparison.csv")
    results = results.loc[results["iterations"] >= 1600].copy()
    # Keep the latest execution per configuration identity, not smoke results.
    identity = ["candidate", "window", "seed", "iterations", "task_type", "code_sha256", "data_sha256"]
    results = results.drop_duplicates(identity, keep="last")
    # A pre-specified 50:50 combination, never an optimized validation weight.
    blended = []
    for _, direct in results.loc[results.candidate == "hierarchical"].iterrows():
        matched = results.loc[results.candidate == "market_residual"]
        for key in identity[1:]:
            matched = matched.loc[matched[key] == direct[key]]
        if matched.empty:
            continue
        residual = matched.iloc[-1]
        with np.load(ROOT / direct.prediction_file) as left, np.load(ROOT / residual.prediction_file) as right:
            if not (np.array_equal(left["row_id"], right["row_id"]) and np.array_equal(left["y"], right["y"])):
                raise ValueError("Cannot average components from different validation rows")
            blend_path = ROOT / "artifacts" / "temporal_comparison" / f"{direct.run_id}_{residual.run_id}_equal_heads.npz"
            np.savez_compressed(blend_path, row_id=left["row_id"], y=left["y"], month=left["month"],
                                prediction=(left["prediction"] + right["prediction"]) / 2)
        row = direct.to_dict()
        row.update(candidate="hierarchical_market_blend", prediction_file=str(blend_path.relative_to(ROOT)))
        blended.append(row)
    if blended:
        results = pd.concat([results, pd.DataFrame(blended)], ignore_index=True)
    summaries = []
    for (candidate, window, iterations, device, code, data), group in results.groupby(
        ["candidate", "window", "iterations", "task_type", "code_sha256", "data_sha256"], sort=False
    ):
        base = results.loc[(results.candidate == "legacy") & (results.window == window)
                           & (results.iterations == iterations) & (results.task_type == device)
                           & (results.code_sha256 == code) & (results.data_sha256 == data)]
        seeds = sorted(set(group.seed) & set(base.seed))
        if not seeds:
            continue
        predictions, baseline = [], []
        target = row_ids = months = None
        for seed in seeds:
            for frame, bucket in [(group, predictions), (base, baseline)]:
                path = ROOT / frame.loc[frame.seed == seed].iloc[-1].prediction_file
                with np.load(path) as values:
                    if target is None:
                        target, row_ids, months = values["y"], values["row_id"], values["month"]
                    if not (np.array_equal(target, values["y"]) and np.array_equal(row_ids, values["row_id"])):
                        raise ValueError("Cannot compare mismatched validation rows")
                    bucket.append(values["prediction"])
        pred = np.mean(predictions, axis=0)
        ref = np.mean(baseline, axis=0)
        rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
        baseline_rmse = float(np.sqrt(np.mean((ref - target) ** 2)))
        monthly_gains = []
        for month in np.unique(months):
            mask = months == month
            monthly_gains.append(float(np.sqrt(np.mean((ref[mask] - target[mask]) ** 2))
                                       - np.sqrt(np.mean((pred[mask] - target[mask]) ** 2))))
        uncertainty = {}
        if bootstrap and candidate == "hierarchical_market_blend" and len(seeds) == 2:
            # Resample whole addresses to retain within-building dependence.
            # This diagnoses this development sample, not future-market uncertainty.
            raw_keys = pd.read_csv(ROOT / "data" / "raw" / "train.csv",
                                   usecols=["TOWN", "BLOCK", "STREET"]).iloc[row_ids]
            address = raw_keys.astype(str).agg("|".join, axis=1)
            group_id, _ = pd.factorize(address)
            counts = np.bincount(group_id)
            base_sse = np.bincount(group_id, weights=(ref-target) ** 2)
            new_sse = np.bincount(group_id, weights=(pred-target) ** 2)
            draws = np.random.default_rng(42).integers(len(counts), size=(1000, len(counts)))
            totals = counts[draws].sum(axis=1)
            gains = np.sqrt(base_sse[draws].sum(axis=1) / totals) - np.sqrt(new_sse[draws].sum(axis=1) / totals)
            uncertainty = dict(address_bootstrap_replicates=1000,
                               gain_ci_lower=float(np.quantile(gains, .025)),
                               gain_ci_upper=float(np.quantile(gains, .975)))
        summaries.append(dict(candidate=candidate, window=window, seeds="+".join(map(str, seeds)),
                              model_count=len(seeds) * (2 if candidate == "hierarchical_market_blend" else 1),
                              iterations=iterations, task_type=device, rows=len(target),
                              rmse=rmse, mse=rmse ** 2, baseline_rmse=baseline_rmse,
                              rmse_improvement=baseline_rmse-rmse,
                              mae=float(np.mean(np.abs(pred-target))), bias=float(np.mean(pred-target)),
                              months_improved=sum(g > 0 for g in monthly_gains), months=len(monthly_gains),
                              worst_month_improvement=min(monthly_gains),
                              code_sha256=code, data_sha256=data, **uncertainty))
    summary = pd.DataFrame(summaries)
    summary.to_csv(REPORTS / "temporal_summary.csv", index=False)
    print(summary[["candidate", "window", "seeds", "rmse", "rmse_improvement", "months_improved", "months"]].round(3).to_string(index=False))
    if not summary.empty:
        chart = summary.loc[summary.candidate != "legacy"].copy()
        if not chart.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=chart, x="rmse_improvement", y="candidate", hue="window", errorbar=None)
            plt.axvline(0, color="black", linewidth=1)
            plt.xlabel("RMSE improvement over matched legacy model (positive is better)")
            plt.ylabel("")
            plt.title("Fixed-budget temporal comparison; identical rows and seeds")
            save("temporal_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-only", action="store_true")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Add paired address-bootstrap intervals for the two-seed head blend.")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    if args.temporal_only:
        summarize_temporal(bootstrap=args.bootstrap)
        return
    results = pd.read_csv(REPORTS / "experiment_results.csv")

    latest = results.loc[
        (results["fold"] == "2025-01_to_2025-03")
        & (
            (results["model"] != "catboost")
            | (results["feature_set"].str.contains("history"))
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

    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    target = train["MONTHLY_RENT"].astype(float)
    raw_features = train.drop(columns=["MONTHLY_RENT"])
    suspected = suspected_low_rent_mask(raw_features, target, ratio_threshold=0.5)
    diagnostic = pd.DataFrame(
        {
            "month": pd.to_datetime(raw_features["RENT_APPROVAL_DATE"]),
            "rent": target,
            "status": "Retained label",
        }
    )
    diagnostic.loc[suspected, "status"] = "Suspected low-rent label"
    monthly_rate = (
        diagnostic.assign(suspected=suspected)
        .groupby("month", as_index=False)["suspected"]
        .mean()
    )
    monthly_rate["suspected_percent"] = 100 * monthly_rate["suspected"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.lineplot(
        data=monthly_rate,
        x="month",
        y="suspected_percent",
        marker="o",
        color="#C96A4A",
        ax=axes[0],
    )
    axes[0].set_title("Flagged-label share over time")
    axes[0].set_xlabel("Approval month")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].set_ylabel("Flagged rows (%)")
    sns.boxplot(
        data=diagnostic,
        x="status",
        y="rent",
        hue="status",
        palette=["#5B8E7D", "#C96A4A"],
        showfliers=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Rent distributions under the fixed rule")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Monthly rent (SGD)")
    axes[1].tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "outlier_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote current experiment figures to {FIGURES}")


if __name__ == "__main__":
    main()
