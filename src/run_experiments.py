"""Run course baselines, rolling validation, and strict spatial ablations."""

from __future__ import annotations

import argparse
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
    prepare_for_catboost,
)
from validation import regression_metrics, rolling_time_folds, subgroup_metrics


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
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--history-iterations", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensemble-seeds", type=int, nargs="+", default=[42, 2026])
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
            fold_train_raw, fold_y_train, fold_train_raw
        )
        history_valid = add_past_rent_features(
            fold_train_raw, fold_y_train, fold_valid_raw
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
            )
            elapsed = time.perf_counter() - started
            total_elapsed += elapsed
            predictions.append(prediction)
            if seed == args.seed:
                record(
                    rows,
                    "catboost",
                    "all_spatial_history",
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
            "all_spatial_history",
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


def main() -> None:
    args = parse_args()
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
        existing = existing.loc[existing["feature_set"] != "all_spatial_history"]
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
            fold_train_raw, fold_y_train, fold_train_raw
        )
        history_valid = add_past_rent_features(
            fold_train_raw, fold_y_train, fold_valid_raw
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
            )
            elapsed = time.perf_counter() - started
            total_elapsed += elapsed
            predictions.append(prediction)
            if seed == args.seed:
                record(
                    rows,
                    "catboost",
                    "all_spatial_history",
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
            "all_spatial_history",
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
            validation_raw["ADDRESS"].isin(
                raw_features.loc[latest.train_mask, "ADDRESS"]
            ),
            "seen",
            "unseen",
        ),
    )
    density = history_valid["HIST_ADDRESS_COUNT"].to_numpy()
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
            "feature_set": "all_spatial_history",
            "seeds": args.ensemble_seeds,
            "weights": [1.0 / len(args.ensemble_seeds)] * len(args.ensemble_seeds),
            "iterations_limit": args.history_iterations,
            "learning_rate": 0.04,
            "depth": 8,
            "l2_leaf_reg": 6.0,
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
