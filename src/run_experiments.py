"""Run course baselines, rolling validation, and strict spatial ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from features import (
    add_past_rent_features,
    add_spatial_features,
    build_block_lookup,
    make_core_features,
    normalize_text,
    prepare_for_catboost,
)
from validation import regression_metrics, rolling_time_folds, subgroup_metrics
from modeling import CANDIDATES, build_features, fit_model, predict_model


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TARGET = "MONTHLY_RENT"
FEATURE_SETS = [
    "core",
    "block",
    "block_mrt",
    "block_mall",
    "block_school",
    "all_spatial",
]
LEAKAGE_CHECK = "train_month<validation_month;target-free spatial features;train-fitted preprocessing"
HISTORY_LEAKAGE_CHECK = LEAKAGE_CHECK + ";historical targets use strictly earlier months"


def group_median_predict(
    train: pd.DataFrame, validation: pd.DataFrame, target: pd.Series
) -> np.ndarray:
    keys = ["TOWN", "FLAT_TYPE"]
    lookup = train.assign(_target=target.to_numpy()).groupby(keys)["_target"].median()
    fallback = train.assign(_target=target.to_numpy()).groupby("FLAT_TYPE")["_target"].median()
    global_median = float(target.median())
    return np.asarray(
        [
            lookup.get((town, flat), fallback.get(flat, global_median))
            for town, flat in validation[keys].itertuples(index=False, name=None)
        ],
        dtype=float,
    )


def classical_preprocessor(frame: pd.DataFrame, linear: bool) -> ColumnTransformer:
    categorical = frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [column for column in frame.columns if column not in categorical]
    if linear:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", min_frequency=10)
        numeric_transformer = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    else:
        categorical_transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        numeric_transformer = SimpleImputer(strategy="median")
    return ColumnTransformer(
        [("numeric", numeric_transformer, numeric), ("categorical", categorical_transformer, categorical)],
        sparse_threshold=0.3,
    )


def classical_models(frame: pd.DataFrame, seed: int) -> dict[str, object]:
    """Course-aligned models sharing the same latest chronological fold."""
    return {
        "linear_regression": make_pipeline(
            classical_preprocessor(frame, linear=True), LinearRegression()
        ),
        "ridge": make_pipeline(
            classical_preprocessor(frame, linear=True), Ridge(alpha=10.0)
        ),
        "decision_tree": make_pipeline(
            classical_preprocessor(frame, linear=False),
            DecisionTreeRegressor(max_depth=18, min_samples_leaf=20, random_state=seed),
        ),
        "random_forest": make_pipeline(
            classical_preprocessor(frame, linear=False),
            RandomForestRegressor(
                n_estimators=180,
                max_features=0.7,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            classical_preprocessor(frame, linear=False),
            HistGradientBoostingRegressor(
                learning_rate=0.08,
                max_iter=250,
                max_leaf_nodes=31,
                l2_regularization=2.0,
                random_state=seed,
            ),
        ),
    }


def fit_predict_catboost(
    train: pd.DataFrame,
    y_train: pd.Series,
    validation: pd.DataFrame,
    y_validation: pd.Series,
    iterations: int,
    seed: int,
    learning_rate: float = 0.04,
    depth: int = 8,
    l2_leaf_reg: float = 6.0,
    task_type: str = "CPU",
) -> tuple[np.ndarray, int]:
    train = prepare_for_catboost(train)
    validation = prepare_for_catboost(validation)
    categorical = train.select_dtypes(include=["object"]).columns.tolist()
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=seed,
        thread_count=-1,
        task_type=task_type,
        devices="0" if task_type == "GPU" else None,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(
        train,
        y_train,
        cat_features=categorical,
        eval_set=(validation, y_validation),
        early_stopping_rounds=100,
        use_best_model=True,
        verbose=False,
    )
    return model.predict(validation), int(model.get_best_iteration())


def record(
    rows: list[dict[str, object]],
    model: str,
    feature_set: str,
    protocol: str,
    fold,
    y_true: pd.Series,
    prediction: np.ndarray,
    elapsed: float,
    best_iteration: int | None = None,
    iterations_limit: int | None = None,
    leakage_checks: str = LEAKAGE_CHECK,
) -> None:
    rows.append(
        {
            "model": model,
            "feature_set": feature_set,
            "protocol": protocol,
            "fold": fold.name,
            "train_end": fold.train_end,
            "validation_start": fold.validation_start,
            "validation_end": fold.validation_end,
            "train_rows": int(fold.train_mask.sum()),
            "validation_rows": int(fold.validation_mask.sum()),
            **regression_metrics(y_true, prediction),
            "best_iteration": best_iteration,
            "iterations_limit": iterations_limit,
            "runtime_seconds": round(elapsed, 3),
            "leakage_checks": leakage_checks,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-candidates", action="store_true",
                        help="Fixed-budget temporal comparison; no scoring-label early stopping.")
    parser.add_argument("--candidates", nargs="+", choices=list(CANDIDATES),
                        default=["legacy", "recent", "hierarchical", "market_residual"])
    parser.add_argument("--windows", nargs="+", default=["short_recent", "long_recent", "long_older"],
                        choices=["short_recent", "short_middle", "long_recent", "long_older", "horizon16"])
    parser.add_argument("--comparison-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--trial-iterations", type=int, default=1600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--history-iterations", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-seeds", type=int, nargs="+", default=[42, 2026])
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument(
        "--refresh-selected",
        action="store_true",
        help="Reuse existing baseline rows and recompute only the selected pipeline.",
    )
    parser.add_argument(
        "--long-horizon",
        action="store_true",
        help="Run two 12-month robustness folds and write long_horizon_results.csv.",
    )
    return parser.parse_args()


def run_long_horizon(
    args: argparse.Namespace,
    train_raw: pd.DataFrame,
    y: pd.Series,
    raw_features: pd.DataFrame,
    core_features: pd.DataFrame,
    block_lookup: pd.DataFrame,
    reports: Path,
) -> None:
    """Compare spatial and selected history pipelines on two 12-month windows."""
    rows: list[dict[str, object]] = []
    folds = rolling_time_folds(train_raw, validation_months=12, folds=2)
    all_spatial = add_spatial_features(core_features, block_lookup, "all_spatial")
    for fold in folds:
        fold_y_train = y.loc[fold.train_mask].reset_index(drop=True)
        fold_y_valid = y.loc[fold.validation_mask].reset_index(drop=True)
        X_train_base = all_spatial.loc[fold.train_mask].reset_index(drop=True)
        X_valid_base = all_spatial.loc[fold.validation_mask].reset_index(drop=True)
        started = time.perf_counter()
        prediction, best_iteration = fit_predict_catboost(
            X_train_base,
            fold_y_train,
            X_valid_base,
            fold_y_valid,
            args.iterations,
            args.seed,
            task_type=args.task_type,
        )
        record(
            rows,
            "catboost",
            "all_spatial",
            "long_horizon_12m",
            fold,
            fold_y_valid,
            prediction,
            time.perf_counter() - started,
            best_iteration,
            args.iterations,
            LEAKAGE_CHECK,
        )
        fold_train_raw = raw_features.loc[fold.train_mask].reset_index(drop=True)
        fold_valid_raw = raw_features.loc[fold.validation_mask].reset_index(drop=True)
        history_train = add_past_rent_features(
            fold_train_raw,
            fold_y_train,
            fold_train_raw,
            detailed=True,
            market_features=False,
        )
        history_valid = add_past_rent_features(
            fold_train_raw,
            fold_y_train,
            fold_valid_raw,
            detailed=True,
            market_features=False,
        )
        history_columns = [
            column for column in history_train if column.startswith("HIST_")
        ]
        X_train = X_train_base.copy()
        X_valid = X_valid_base.copy()
        X_train[history_columns] = history_train[history_columns]
        X_valid[history_columns] = history_valid[history_columns]
        predictions: list[np.ndarray] = []
        total_elapsed = 0.0
        for seed in args.ensemble_seeds:
            started = time.perf_counter()
            prediction, best_iteration = fit_predict_catboost(
                X_train,
                fold_y_train,
                X_valid,
                fold_y_valid,
                args.history_iterations,
                seed,
                task_type=args.task_type,
            )
            elapsed = time.perf_counter() - started
            total_elapsed += elapsed
            predictions.append(prediction)
            if seed == args.seed:
                record(
                    rows,
                    "catboost",
                    "all_spatial_detailed_history",
                    "long_horizon_12m",
                    fold,
                    fold_y_valid,
                    prediction,
                    elapsed,
                    best_iteration,
                    args.history_iterations,
                    HISTORY_LEAKAGE_CHECK,
                )
        ensemble_prediction = np.mean(predictions, axis=0)
        record(
            rows,
            "catboost_seed_ensemble",
            "all_spatial_detailed_history",
            "long_horizon_12m",
            fold,
            fold_y_valid,
            ensemble_prediction,
            total_elapsed,
            None,
            args.history_iterations,
            HISTORY_LEAKAGE_CHECK,
        )
    results = pd.DataFrame(rows)
    results.to_csv(reports / "long_horizon_results.csv", index=False)
    print(results.to_string(index=False))


def run_candidate_comparison(args):
    """Append auditable paired predictions and metrics without replacing old reports.

    Windows overlap and are development evidence, not independent test sets.
    Tree counts are set BEFORE evaluation, with no eval_set/early stopping.
    """
    windows = {
        "short_recent": ("2025-01", "2025-03"),
        "short_middle": ("2024-10", "2024-12"),
        "long_recent": ("2024-04", "2025-03"),
        "long_older": ("2023-04", "2024-03"),
        "horizon16": ("2023-12", "2025-03"),
    }
    from features import suspected_low_rent_mask
    source = RAW / "train.csv"
    data_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    code_hash = hashlib.sha256(b"".join(
        (ROOT / "src" / file).read_bytes()
        for file in ["features.py", "modeling.py", "run_experiments.py"]
    )).hexdigest()
    raw = pd.read_csv(source)
    target = raw.pop(TARGET)
    lookup = build_block_lookup(RAW / "auxiliary")
    ledger = ROOT / "reports" / "temporal_comparison.csv"
    cache_dir = ROOT / "artifacts" / "temporal_comparison"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(ledger).to_dict("records") if ledger.exists() else []
    for window in args.windows:
        start, end = windows[window]
        train_mask = raw.RENT_APPROVAL_DATE.lt(start)
        valid_mask = raw.RENT_APPROVAL_DATE.between(start, end)
        train = raw.loc[train_mask].reset_index(drop=True)
        valid = raw.loc[valid_mask].reset_index(drop=True)
        y_train = target.loc[train_mask].reset_index(drop=True)
        y_valid = target.loc[valid_mask].reset_index(drop=True)
        flagged = suspected_low_rent_mask(valid, y_valid)
        feature_cache = {}
        for name in args.candidates:
            config = dict(CANDIDATES[name], iterations=args.trial_iterations)
            recipe = config["history"]
            for seed in args.comparison_seeds:
                identity = json.dumps(dict(config=config, seed=seed, window=window,
                                           device=args.task_type, data=data_hash,
                                           code=code_hash), sort_keys=True)
                run_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
                prediction_path = cache_dir / f"{run_id}.npz"
                if args.resume and prediction_path.exists() and any(r["run_id"] == run_id for r in rows):
                    print(f"REUSE {name} {window} seed={seed}", flush=True)
                    continue
                if recipe not in feature_cache:
                    feature_started = time.perf_counter()
                    X_train = build_features(train, y_train, train, lookup, recipe)
                    X_valid = build_features(train, y_train, valid, lookup, recipe)
                    feature_cache[recipe] = (X_train, X_valid, time.perf_counter() - feature_started)
                X_train, X_valid, feature_seconds = feature_cache[recipe]
                print(f"FIT {name} {window} seed={seed} rows={len(train)} trees={config['iterations']}", flush=True)
                started = time.perf_counter()
                model = fit_model(X_train, y_train, config, seed, args.task_type)
                prediction = predict_model(model, X_valid, config)
                elapsed = time.perf_counter() - started
                np.savez_compressed(prediction_path, row_id=raw.index[valid_mask].to_numpy(),
                                    y=y_valid.to_numpy(), prediction=prediction,
                                    month=valid.RENT_APPROVAL_DATE.to_numpy(dtype=str))
                metrics = regression_metrics(y_valid, prediction)
                row = dict(run_id=run_id, candidate=name, window=window, train_end=train.RENT_APPROVAL_DATE.max(),
                           validation_start=start, validation_end=end, train_rows=len(train),
                           validation_rows=len(valid), task_type=args.task_type, seed=seed,
                           iterations=config["iterations"], actual_trees=model.tree_count_,
                           features=recipe, residual_target=config["residual_target"],
                           depth=config["depth"], feature_count=X_train.shape[1],
                           **metrics, mse=metrics["rmse"] ** 2,
                           flagged_validation_rows=int(flagged.sum()),
                           retained_slice_rmse=regression_metrics(y_valid[~flagged], prediction[~flagged])["rmse"],
                           flagged_slice_rmse=regression_metrics(y_valid[flagged], prediction[flagged])["rmse"] if flagged.any() else np.nan,
                           feature_seconds=round(feature_seconds, 3), model_seconds=round(elapsed, 3),
                           data_sha256=data_hash, code_sha256=code_hash,
                           config_json=json.dumps(config, sort_keys=True),
                           prediction_file=str(prediction_path.relative_to(ROOT)),
                           protocol="fixed_budget_no_outer_early_stopping")
                rows = [r for r in rows if r["run_id"] != run_id] + [row]
                pd.DataFrame(rows).to_csv(ledger, index=False)
                print(f"DONE {name} {window} seed={seed} RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} seconds={elapsed:.1f}", flush=True)
                del model
    print(f"Results: {ledger}", flush=True)


def main() -> None:
    args = parse_args()
    if args.compare_candidates:
        run_candidate_comparison(args)
        return
    if not args.ensemble_seeds or len(set(args.ensemble_seeds)) != len(
        args.ensemble_seeds
    ):
        raise ValueError("--ensemble-seeds must contain unique seed values")
    if args.seed not in args.ensemble_seeds:
        raise ValueError("--ensemble-seeds must include --seed for the single-model control")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    train_raw = pd.read_csv(RAW / "train.csv")
    y = train_raw[TARGET].astype(float)
    raw_features = train_raw.drop(columns=[TARGET])
    core_features = make_core_features(raw_features)
    block_lookup = build_block_lookup(RAW / "auxiliary")
    if args.long_horizon:
        run_long_horizon(
            args,
            train_raw,
            y,
            raw_features,
            core_features,
            block_lookup,
            reports,
        )
        return
    folds = rolling_time_folds(
        train_raw,
        validation_months=args.validation_months,
        folds=args.folds,
    )
    latest = folds[-1]
    if args.refresh_selected:
        existing_path = reports / "experiment_results.csv"
        if not existing_path.exists():
            raise FileNotFoundError("--refresh-selected requires existing experiment results")
        existing = pd.read_csv(existing_path)
        existing = existing.loc[
            ~existing["feature_set"].astype(str).str.contains("history")
        ]
        rows = existing.to_dict(orient="records")
    else:
        rows: list[dict[str, object]] = []

    y_train = y.loc[latest.train_mask].reset_index(drop=True)
    y_valid = y.loc[latest.validation_mask].reset_index(drop=True)
    if not args.skip_classical and not args.refresh_selected:
        train_raw_fold = raw_features.loc[latest.train_mask].reset_index(drop=True)
        valid_raw_fold = raw_features.loc[latest.validation_mask].reset_index(drop=True)
        started = time.perf_counter()
        record(
            rows,
            "global_median",
            "raw",
            "latest_fold_comparison",
            latest,
            y_valid,
            np.full(len(y_valid), y_train.median()),
            time.perf_counter() - started,
        )
        started = time.perf_counter()
        record(
            rows,
            "town_flat_type_median",
            "raw",
            "latest_fold_comparison",
            latest,
            y_valid,
            group_median_predict(train_raw_fold, valid_raw_fold, y_train),
            time.perf_counter() - started,
        )

        X_train = core_features.loc[latest.train_mask].reset_index(drop=True)
        X_valid = core_features.loc[latest.validation_mask].reset_index(drop=True)
        for name, model in classical_models(X_train, args.seed).items():
            started = time.perf_counter()
            model.fit(X_train, y_train)
            prediction = model.predict(X_valid)
            record(
                rows,
                name,
                "core",
                "latest_fold_comparison",
                latest,
                y_valid,
                prediction,
                time.perf_counter() - started,
            )

    latest_history_prediction: np.ndarray | None = None
    for fold in folds:
        fold_y_train = y.loc[fold.train_mask].reset_index(drop=True)
        fold_y_valid = y.loc[fold.validation_mask].reset_index(drop=True)
        feature_sets = FEATURE_SETS if fold.name == latest.name else ["core", "all_spatial"]
        for feature_set in ([] if args.refresh_selected else feature_sets):
            feature_frame = add_spatial_features(core_features, block_lookup, feature_set)
            X_train = feature_frame.loc[fold.train_mask].reset_index(drop=True)
            X_valid = feature_frame.loc[fold.validation_mask].reset_index(drop=True)
            started = time.perf_counter()
            prediction, best_iteration = fit_predict_catboost(
                X_train,
                fold_y_train,
                X_valid,
                fold_y_valid,
                args.iterations,
                args.seed,
                task_type=args.task_type,
            )
            protocol = (
                "latest_spatial_ablation"
                if fold.name == latest.name
                else "rolling_stability"
            )
            record(
                rows,
                "catboost",
                feature_set,
                protocol,
                fold,
                fold_y_valid,
                prediction,
                time.perf_counter() - started,
                best_iteration,
                args.iterations,
            )

        fold_train_raw = raw_features.loc[fold.train_mask].reset_index(drop=True)
        fold_valid_raw = raw_features.loc[fold.validation_mask].reset_index(drop=True)
        history_train = add_past_rent_features(
            fold_train_raw,
            fold_y_train,
            fold_train_raw,
            detailed=True,
            market_features=False,
        )
        history_valid = add_past_rent_features(
            fold_train_raw,
            fold_y_train,
            fold_valid_raw,
            detailed=True,
            market_features=False,
        )
        history_columns = [
            column for column in history_train if column.startswith("HIST_")
        ]
        all_spatial = add_spatial_features(core_features, block_lookup, "all_spatial")
        X_train = all_spatial.loc[fold.train_mask].reset_index(drop=True)
        X_valid = all_spatial.loc[fold.validation_mask].reset_index(drop=True)
        X_train[history_columns] = history_train[history_columns]
        X_valid[history_columns] = history_valid[history_columns]
        predictions: list[np.ndarray] = []
        total_elapsed = 0.0
        for seed in args.ensemble_seeds:
            started = time.perf_counter()
            prediction, best_iteration = fit_predict_catboost(
                X_train,
                fold_y_train,
                X_valid,
                fold_y_valid,
                args.history_iterations,
                seed,
                task_type=args.task_type,
            )
            elapsed = time.perf_counter() - started
            total_elapsed += elapsed
            predictions.append(prediction)
            if seed == args.seed:
                record(
                    rows,
                    "catboost",
                    "all_spatial_detailed_history",
                    "rolling_history_single",
                    fold,
                    fold_y_valid,
                    prediction,
                    elapsed,
                    best_iteration,
                    args.history_iterations,
                    HISTORY_LEAKAGE_CHECK,
                )
        prediction = np.mean(predictions, axis=0)
        record(
            rows,
            "catboost_seed_ensemble",
            "all_spatial_detailed_history",
            "rolling_selected_pipeline",
            fold,
            fold_y_valid,
            prediction,
            total_elapsed,
            None,
            args.history_iterations,
            HISTORY_LEAKAGE_CHECK,
        )
        if fold.name == latest.name:
            latest_history_prediction = prediction

    results = pd.DataFrame(rows)
    results.to_csv(reports / "experiment_results.csv", index=False)
    summary = (
        results.groupby(["protocol", "model", "feature_set"], as_index=False)
        .agg(
            runs=("fold", "count"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
            mean_bias=("bias", "mean"),
            total_runtime_seconds=("runtime_seconds", "sum"),
        )
        .sort_values(["protocol", "mean_rmse"])
    )
    summary.to_csv(reports / "experiment_summary.csv", index=False)
    if latest_history_prediction is None:
        raise AssertionError("Latest selected CatBoost prediction was not produced")
    validation_raw = train_raw.loc[latest.validation_mask].reset_index(drop=True)
    error_groups = subgroup_metrics(
        validation_raw,
        y_valid,
        latest_history_prediction,
        ["RENT_APPROVAL_DATE", "TOWN", "FLAT_TYPE", "FLAT_MODEL"],
    )
    diagnostics = validation_raw.assign(
        _target=y_valid.to_numpy(),
        _prediction=latest_history_prediction,
        RENT_BAND=pd.cut(
            y_valid,
            bins=[-np.inf, 2500, 3000, 3500, 4000, np.inf],
            labels=["<=2500", "2501-3000", "3001-3500", "3501-4000", ">4000"],
        ).astype(str),
        ADDRESS_STATUS=np.where(
            (
                normalize_text(validation_raw["BLOCK"]).fillna("missing")
                + " "
                + normalize_text(validation_raw["STREET"]).fillna("missing")
            ).isin(
                normalize_text(raw_features.loc[latest.train_mask, "BLOCK"])
                .fillna("missing")
                + " "
                + normalize_text(raw_features.loc[latest.train_mask, "STREET"])
                .fillna("missing")
            ),
            "seen",
            "unseen",
        ),
    )
    density = np.expm1(history_valid["HIST_ADDRESS_LOG_COUNT"].to_numpy())
    diagnostics["ADDRESS_HISTORY_DENSITY"] = pd.cut(
        density,
        bins=[-np.inf, 0, 5, 20, np.inf],
        labels=["0", "1-5", "6-20", ">20"],
    ).astype(str)
    extra_groups = subgroup_metrics(
        diagnostics,
        diagnostics["_target"],
        diagnostics["_prediction"],
        ["RENT_BAND", "ADDRESS_STATUS", "ADDRESS_HISTORY_DENSITY"],
    )
    pd.concat([error_groups, extra_groups], ignore_index=True).to_csv(
        reports / "error_analysis.csv", index=False
    )
    metadata = {
        "folds": [
            {
                "name": fold.name,
                "train_end": fold.train_end,
                "validation_start": fold.validation_start,
                "validation_end": fold.validation_end,
                "train_rows": int(fold.train_mask.sum()),
                "validation_rows": int(fold.validation_mask.sum()),
            }
            for fold in folds
        ],
        "iterations_limit": args.iterations,
        "selected_pipeline": {
            "model": "catboost_seed_ensemble",
            "feature_set": "all_spatial_detailed_history",
            "history_groups": [
                "address",
                "address_flat_type",
                "address_flat_type_area",
                "street_flat_type",
                "town_flat_type",
                "town_flat_type_model",
                "town",
                "flat_type",
                "flat_model",
            ],
            "recent_or_market_adjusted_features": False,
            "seeds": args.ensemble_seeds,
            "weights": [1.0 / len(args.ensemble_seeds)] * len(args.ensemble_seeds),
            "iterations_limit": args.history_iterations,
            "learning_rate": 0.04,
            "depth": 8,
            "l2_leaf_reg": 6.0,
            "task_type": args.task_type,
        },
        "seed": args.seed,
        "leakage_controls": [
            "validation months are strictly later than all training months",
            "spatial joins never use MONTHLY_RENT",
            "only MRT stations marked open in supplied data are used",
            "encoders, imputers, and scalers are fitted inside each training fold",
            "historical rent statistics use only rows from strictly earlier months",
        ],
        "kaggle_data": {
            "competition": "cs-5228-2610-project",
            "downloaded_zip_sha256": "BB7FF2EE4AE7EABE73DAF929D352B27F98592CD9AEA89EAA99804011612F3BC6",
            "official_files_match_existing_raw_data": True,
        },
    }
    (reports / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
