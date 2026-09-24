"""Create report-ready plots from the current experiment outputs."""

from __future__ import annotations

from pathlib import Path
import argparse
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge

from features import suspected_low_rent_mask


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
STACK_HEADS = (
    "local_trend_direct",
    "local_trend_residual",
    "horizon_direct",
    "horizon_residual",
)


def save(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close()


def simplex_least_squares(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit nonnegative ensemble weights constrained to sum to one."""
    predictions = np.asarray(predictions, dtype=float)
    target = np.asarray(target, dtype=float)
    if predictions.ndim != 2 or target.shape != (len(predictions),):
        raise ValueError("Stacking predictions and target have incompatible shapes")
    scale = max(float(np.nanmedian(np.abs(target))), 1.0)
    X, y = predictions / scale, target / scale
    best_weights, best_loss = None, np.inf
    for size in range(1, X.shape[1] + 1):
        for active in combinations(range(X.shape[1]), size):
            subset = X[:, active]
            gram = subset.T @ subset
            kkt = np.block([[gram, np.ones((size, 1))],
                            [np.ones((1, size)), np.zeros((1, 1))]])
            rhs = np.r_[subset.T @ y, 1.0]
            solution = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:-1]
            if np.any(solution < -1e-9):
                continue
            weights = np.zeros(X.shape[1])
            weights[list(active)] = np.maximum(solution, 0.0)
            weights /= weights.sum()
            loss = float(np.mean((predictions @ weights - target) ** 2))
            if loss < best_loss:
                best_weights, best_loss = weights, loss
    if best_weights is None:
        raise ValueError("No feasible stacking weights")
    return best_weights


def fit_temporal_ridge(predictions: np.ndarray, target: np.ndarray,
                       months: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Choose Ridge strength on the final three meta-training months only."""
    unique_months = sorted(np.unique(months))
    if len(unique_months) < 6:
        raise ValueError("Ridge stacking needs at least six meta-training months")
    inner_validation = set(unique_months[-3:])
    inner_train = np.array([month not in inner_validation for month in months])
    inner_valid = ~inner_train
    ratios = (0.01, 0.1, 1.0, 10.0)
    best_ratio, best_rmse = None, np.inf
    for ratio in ratios:
        mean = predictions[inner_train].mean(axis=0)
        scale = predictions[inner_train].std(axis=0)
        scale[scale == 0] = 1.0
        model = Ridge(alpha=float(inner_train.sum()) * ratio)
        model.fit((predictions[inner_train]-mean) / scale, target[inner_train])
        estimate = model.predict((predictions[inner_valid]-mean) / scale)
        rmse = float(np.sqrt(np.mean((estimate-target[inner_valid]) ** 2)))
        if rmse < best_rmse:
            best_ratio, best_rmse = ratio, rmse
    mean = predictions.mean(axis=0)
    scale = predictions.std(axis=0)
    scale[scale == 0] = 1.0
    model = Ridge(alpha=float(len(target)) * best_ratio)
    model.fit((predictions-mean) / scale, target)
    weights = model.coef_ / scale
    intercept = float(model.intercept_ - mean @ weights)
    return weights, intercept, float(best_ratio)


def load_stacking_window(results: pd.DataFrame, window: str) -> dict:
    """Load four matched two-seed out-of-fold prediction heads."""
    heads, target, row_ids, months = [], None, None, None
    for candidate in STACK_HEADS:
        rows = results.loc[(results.candidate == candidate)
                           & (results.window == window)
                           & (results.iterations >= 1600)].copy()
        if rows.empty:
            raise ValueError(f"Missing stacking head {candidate} for {window}")
        rows = rows.drop_duplicates("seed", keep="last")
        if set(rows.seed.astype(int)) != {42, 2026}:
            raise ValueError(f"Stacking requires seeds 42 and 2026 for {candidate}/{window}")
        seed_predictions = []
        for _, row in rows.sort_values("seed").iterrows():
            with np.load(ROOT / row.prediction_file) as values:
                if target is None:
                    target = values["y"].copy()
                    row_ids = values["row_id"].copy()
                    months = values["month"].astype(str)
                if not (np.array_equal(target, values["y"])
                        and np.array_equal(row_ids, values["row_id"])
                        and np.array_equal(months, values["month"].astype(str))):
                    raise ValueError("Cannot stack predictions from different validation rows")
                seed_predictions.append(values["prediction"].copy())
        heads.append(np.mean(seed_predictions, axis=0))
    return dict(predictions=np.column_stack(heads), target=target,
                row_id=row_ids, month=months)


def stacking_gain_interval(reference: np.ndarray, prediction: np.ndarray,
                           target: np.ndarray, row_ids: np.ndarray,
                           random_seed: int = 42) -> tuple[float, float]:
    """Paired address-cluster interval for RMSE gain on one evaluation period."""
    raw_keys = pd.read_csv(ROOT / "data" / "raw" / "train.csv",
                           usecols=["TOWN", "BLOCK", "STREET"]).iloc[row_ids]
    address = raw_keys.astype(str).agg("|".join, axis=1)
    group_id, _ = pd.factorize(address)
    counts = np.bincount(group_id)
    ref_sse = np.bincount(group_id, weights=(reference-target) ** 2)
    new_sse = np.bincount(group_id, weights=(prediction-target) ** 2)
    draws = np.random.default_rng(random_seed).integers(len(counts), size=(1000, len(counts)))
    totals = counts[draws].sum(axis=1)
    gains = np.sqrt(ref_sse[draws].sum(axis=1) / totals) - np.sqrt(new_sse[draws].sum(axis=1) / totals)
    return float(np.quantile(gains, .025)), float(np.quantile(gains, .975))


def evaluate_stacking() -> pd.DataFrame:
    """Learn meta weights on earlier OOF months and evaluate on later months."""
    results = pd.read_csv(REPORTS / "temporal_comparison.csv")
    plans = [
        ("older_to_recent", "long_older", None, "long_recent"),
        ("predec_2023_to_horizon16", "long_older", "2023-11", "horizon16"),
        ("prejan_2025_to_short", "long_recent", "2024-12", "short_recent"),
    ]
    records = []
    equal = np.full(len(STACK_HEADS), 1.0 / len(STACK_HEADS))
    for plan, train_window, train_end, eval_window in plans:
        train = load_stacking_window(results, train_window)
        evaluation = load_stacking_window(results, eval_window)
        train_mask = np.ones(len(train["target"]), dtype=bool)
        if train_end is not None:
            train_mask &= train["month"] <= train_end
        if not train_mask.any():
            raise ValueError(f"Empty meta-training period for {plan}")
        latest_train_month = sorted(train["month"][train_mask])[-1]
        earliest_eval_month = sorted(evaluation["month"])[0]
        if latest_train_month >= earliest_eval_month:
            raise ValueError(f"Stacking periods overlap for {plan}")
        learned = simplex_least_squares(train["predictions"][train_mask],
                                        train["target"][train_mask])
        ridge_weights, ridge_intercept, ridge_alpha_ratio = fit_temporal_ridge(
            train["predictions"][train_mask], train["target"][train_mask],
            train["month"][train_mask],
        )
        methods = {
            "equal_weight_reference": (equal, 0.0, np.nan),
            "simplex": (learned, 0.0, np.nan),
            "half_shrunk_simplex": (0.5 * learned + 0.5 * equal, 0.0, np.nan),
            "temporal_ridge": (ridge_weights, ridge_intercept, ridge_alpha_ratio),
        }
        reference = evaluation["predictions"] @ equal
        reference_rmse = float(np.sqrt(np.mean((reference-evaluation["target"]) ** 2)))
        for method, (weights, intercept, alpha_ratio) in methods.items():
            train_prediction = train["predictions"][train_mask] @ weights + intercept
            prediction = evaluation["predictions"] @ weights + intercept
            eval_rmse = float(np.sqrt(np.mean((prediction-evaluation["target"]) ** 2)))
            monthly_gains = []
            for month in np.unique(evaluation["month"]):
                mask = evaluation["month"] == month
                old = np.sqrt(np.mean((reference[mask]-evaluation["target"][mask]) ** 2))
                new = np.sqrt(np.mean((prediction[mask]-evaluation["target"][mask]) ** 2))
                monthly_gains.append(float(old-new))
            lower = upper = np.nan
            if method != "equal_weight_reference":
                lower, upper = stacking_gain_interval(
                    reference, prediction, evaluation["target"], evaluation["row_id"],
                    random_seed=42 + len(records),
                )
            record = dict(
                plan=plan, meta_train_window=train_window,
                meta_train_start=sorted(train["month"][train_mask])[0],
                meta_train_end=latest_train_month, evaluation_window=eval_window,
                evaluation_start=earliest_eval_month,
                evaluation_end=sorted(evaluation["month"])[-1], method=method,
                train_rows=int(train_mask.sum()), evaluation_rows=len(evaluation["target"]),
                train_rmse=float(np.sqrt(np.mean((train_prediction-train["target"][train_mask]) ** 2))),
                evaluation_rmse=eval_rmse,
                rmse_improvement=reference_rmse-eval_rmse,
                months_improved=sum(value > 0 for value in monthly_gains),
                months=len(monthly_gains), gain_ci_lower=lower, gain_ci_upper=upper,
                intercept=intercept, ridge_alpha_ratio=alpha_ratio,
            )
            record.update({f"weight_{name}": float(weight)
                           for name, weight in zip(STACK_HEADS, weights)})
            records.append(record)
    summary = pd.DataFrame(records)
    summary.to_csv(REPORTS / "stacking_results.csv", index=False)
    print(summary[["plan", "method", "evaluation_rmse", "rmse_improvement",
                   "months_improved", "months", "gain_ci_lower", "gain_ci_upper"]]
          .round(4).to_string(index=False))
    return summary


def summarize_temporal(bootstrap=False, reference_candidate="legacy", error_groups=False):
    """Compare exact same rows and seeds using saved out-of-fold predictions."""
    results = pd.read_csv(REPORTS / "temporal_comparison.csv")
    results = results.loc[results["iterations"] >= 1600].copy()
    # Keep the latest execution per configuration identity, not smoke results.
    identity = ["candidate", "window", "seed", "iterations", "task_type", "code_sha256", "data_sha256"]
    results = results.drop_duplicates(identity, keep="last")
    results["components_per_seed"] = 1
    # Fixed head weights and one pre-specified cross-learner weight; no grid search.
    pairs = [("hierarchical", "market_residual", "hierarchical_market_blend", 2),
             ("weighted_direct", "weighted_residual", "weighted_blend", 2),
             ("horizon_direct", "horizon_residual", "horizon_blend", 2),
             ("local_trend_direct", "local_trend_residual", "local_trend_blend", 2),
             ("composition_trend_direct", "composition_trend_residual",
              "composition_trend_blend", 2),
             ("composition_trend_only_direct", "composition_trend_only_residual",
              "composition_trend_only_blend", 2),
             ("xgb_trend_depth6_direct", "xgb_trend_depth6_residual", "xgb_trend_depth6_blend", 2),
             ("xgb_trend_depth8_direct", "xgb_trend_depth8_residual", "xgb_trend_depth8_blend", 2),
             ("horizon_blend", "hierarchical_market_blend", "history_diversity_blend", 4),
             ("horizon_blend", "local_trend_blend",
              "local_trend_history_diversity_blend", 4),
             ("horizon_blend", "composition_trend_blend",
              "composition_trend_history_diversity_blend", 4),
             ("horizon_blend", "composition_trend_only_blend",
              "composition_trend_only_history_diversity_blend", 4),
             ("xgb_trend_depth6_blend", "history_diversity_blend",
              "xgb_depth6_quarter_mainline_blend", 6),
             ("xgb_trend_depth8_blend", "history_diversity_blend",
              "xgb_depth8_quarter_mainline_blend", 6)]
    for direct_name, residual_name, blend_name, component_count in pairs:
        left_weight = 0.25 if "quarter_mainline" in blend_name else 0.5
        blended = []
        for _, direct in results.loc[results.candidate == direct_name].iterrows():
            matched = results.loc[results.candidate == residual_name]
            for key in identity[1:]:
                if key != "code_sha256" or component_count == 2:
                    matched = matched.loc[matched[key] == direct[key]]
            if matched.empty:
                continue
            residual = matched.iloc[-1]
            if direct.train_end != residual.train_end:
                raise ValueError("Cannot average different training cutoffs")
            with np.load(ROOT / direct.prediction_file) as left, np.load(ROOT / residual.prediction_file) as right:
                if not (np.array_equal(left["row_id"], right["row_id"]) and np.array_equal(left["y"], right["y"])):
                    raise ValueError("Cannot average components from different validation rows")
                suffix = "quarter_xgb" if left_weight == 0.25 else "equal_heads"
                blend_path = ROOT / "artifacts" / "temporal_comparison" / f"{direct.run_id}_{residual.run_id}_{suffix}.npz"
                np.savez_compressed(blend_path, row_id=left["row_id"], y=left["y"], month=left["month"],
                                    prediction=left_weight * left["prediction"] + (1-left_weight) * right["prediction"])
            row = direct.to_dict()
            row.update(candidate=blend_name, prediction_file=str(blend_path.relative_to(ROOT)),
                       components_per_seed=component_count)
            if direct.code_sha256 != residual.code_sha256:
                row["code_sha256"] = direct.code_sha256 + "+" + residual.code_sha256
            blended.append(row)
        if blended:
            results = pd.concat([results, pd.DataFrame(blended)], ignore_index=True)
    summaries, diagnostics = [], []
    raw = pd.read_csv(ROOT / "data" / "raw" / "train.csv") if error_groups else None
    for (candidate, window, iterations, device, code, data), group in results.groupby(
        ["candidate", "window", "iterations", "task_type", "code_sha256", "data_sha256"], sort=False
    ):
        base = results.loc[(results.candidate == reference_candidate) & (results.window == window)
                           & (results.iterations == iterations) & (results.task_type == device)
                           & (results.data_sha256 == data)]
        if base.empty:
            continue
        # Reuse immutable paired predictions from the submitted reference when
        # additive experimental code changes its source hash. Record provenance;
        # row IDs, labels and training cutoff must still match exactly.
        reference_code = code if base.code_sha256.eq(code).any() else base.iloc[-1].code_sha256
        base = base.loc[base.code_sha256.eq(reference_code)]
        if not set(group.train_end).issubset(set(base.train_end)):
            raise ValueError("Cannot compare different training cutoffs")
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
        if bootstrap and candidate.endswith("blend") and len(seeds) == 2 and candidate != reference_candidate:
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
                              model_count=len(seeds) * int(group.components_per_seed.fillna(1).iloc[0]),
                              iterations=iterations, task_type=device, rows=len(target),
                              rmse=rmse, mse=rmse ** 2, baseline_rmse=baseline_rmse,
                              rmse_improvement=baseline_rmse-rmse,
                              mae=float(np.mean(np.abs(pred-target))), bias=float(np.mean(pred-target)),
                              months_improved=sum(g > 0 for g in monthly_gains), months=len(monthly_gains),
                              worst_month_improvement=min(monthly_gains),
                              code_sha256=code, data_sha256=data,
                              baseline_candidate=reference_candidate, baseline_code_sha256=reference_code,
                              **uncertainty))
        if error_groups and candidate.endswith("blend"):
            from features import normalize_text
            frame = raw.iloc[row_ids].reset_index(drop=True)
            frame["MONTH"] = months
            for field in ["TOWN", "FLAT_TYPE"]:
                frame[field] = normalize_text(frame[field])
            frame["AREA_BAND"] = pd.cut(frame.FLOOR_AREA_SQM, [0, 60, 90, 110, 130, np.inf]).astype(str)
            frame["error"], frame["squared_error"] = pred-target, (pred-target)**2
            frame["baseline_squared_error"] = (ref-target)**2
            for field in ["MONTH", "TOWN", "FLAT_TYPE", "AREA_BAND"]:
                grouped = frame.groupby(field).agg(rows=("error", "size"), bias=("error", "mean"),
                                                   mse=("squared_error", "mean"),
                                                   baseline_mse=("baseline_squared_error", "mean"),
                                                   sse=("squared_error", "sum")).reset_index()
                grouped = grouped.rename(columns={field: "group_value"})
                grouped["rmse"] = np.sqrt(grouped.mse)
                grouped["rmse_improvement"] = np.sqrt(grouped.baseline_mse)-grouped.rmse
                grouped["sse_fraction"] = grouped.sse / frame.squared_error.sum()
                diagnostics.append(grouped.assign(candidate=candidate, window=window,
                                                   seeds="+".join(map(str, seeds)), group_column=field,
                                                   code_sha256=code, baseline_code_sha256=reference_code))
    summary = pd.DataFrame(summaries)
    if summary.empty:
        raise ValueError("No paired runs match the requested reference candidate")
    summary.to_csv(REPORTS / "temporal_summary.csv", index=False)
    if diagnostics:
        pd.concat(diagnostics, ignore_index=True).to_csv(
            ROOT / "artifacts" / "temporal_comparison" / "error_groups.csv", index=False)
    print(summary[["candidate", "window", "seeds", "rmse", "rmse_improvement", "months_improved", "months"]].round(3).to_string(index=False))
    if not summary.empty:
        chart = summary.loc[summary.candidate != reference_candidate].copy()
        if not chart.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(data=chart, x="rmse_improvement", y="candidate", hue="window", errorbar=None)
            plt.axvline(0, color="black", linewidth=1)
            plt.xlabel(f"RMSE improvement over {reference_candidate} (positive is better)")
            plt.ylabel("")
            plt.title("Fixed-budget temporal comparison; identical rows and seeds")
            save("temporal_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal-only", action="store_true")
    parser.add_argument("--stacking-only", action="store_true",
                        help="Evaluate time-transferred meta weights from saved OOF predictions.")
    parser.add_argument("--reference-candidate", default="legacy")
    parser.add_argument("--error-groups", action="store_true",
                        help="Write local paired errors by month, town, flat type and area.")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Add paired address-bootstrap intervals for the two-seed head blend.")
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    if args.stacking_only:
        evaluate_stacking()
        return
    if args.temporal_only:
        summarize_temporal(bootstrap=args.bootstrap, reference_candidate=args.reference_candidate,
                           error_groups=args.error_groups)
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
