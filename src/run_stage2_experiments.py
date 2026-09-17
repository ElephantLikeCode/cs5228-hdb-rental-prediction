"""Run course baselines and CatBoost spatial-feature ablations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from project_features import (
    add_spatial_features,
    build_block_lookup,
    make_core_features,
    prepare_for_catboost,
)
from validation import regression_metrics, rolling_time_folds, subgroup_metrics


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TARGET = "MONTHLY_RENT"


def group_median_predict(
    train: pd.DataFrame, validation: pd.DataFrame, target: pd.Series
) -> np.ndarray:
    keys = ["TOWN", "FLAT_TYPE"]
    lookup = train.assign(_target=target.to_numpy()).groupby(keys)["_target"].median()
    fallback = train.assign(_target=target.to_numpy()).groupby("FLAT_TYPE")["_target"].median()
    global_median = float(target.median())
    return np.asarray(
        [lookup.get((town, flat), fallback.get(flat, global_median)) for town, flat in validation[keys].itertuples(index=False, name=None)],
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


def fit_predict_catboost(
    train: pd.DataFrame,
    y_train: pd.Series,
    validation: pd.DataFrame,
    y_validation: pd.Series,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    train = prepare_for_catboost(train)
    validation = prepare_for_catboost(validation)
    categorical = train.select_dtypes(include=["object"]).columns.tolist()
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=6.0,
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
        verbose=200,
    )
    return model.predict(validation), int(model.get_best_iteration())


def record(
    rows: list[dict[str, object]],
    model: str,
    feature_set: str,
    fold_name: str,
    y_true: pd.Series,
    prediction: np.ndarray,
    elapsed: float,
    best_iteration: int | None = None,
) -> None:
    rows.append(
        {
            "model": model,
            "feature_set": feature_set,
            "fold": fold_name,
            **regression_metrics(y_true, prediction),
            "best_iteration": best_iteration,
            "elapsed_seconds": round(elapsed, 2),
            "validation_rows": len(y_true),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-classical", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    train_raw = pd.read_csv(RAW / "train.csv")
    y = train_raw[TARGET].astype(float)
    core_raw = train_raw.drop(columns=[TARGET])
    core_features = make_core_features(core_raw)
    block_lookup = build_block_lookup(RAW / "auxiliary")
    spatial_features = add_spatial_features(core_features, block_lookup)
    folds = rolling_time_folds(
        train_raw,
        validation_months=args.validation_months,
        folds=args.folds,
    )
    rows: list[dict[str, object]] = []

    latest = folds[-1]
    if not args.skip_classical:
        train_core_raw = core_raw.loc[latest.train_mask].reset_index(drop=True)
        valid_core_raw = core_raw.loc[latest.validation_mask].reset_index(drop=True)
        y_train = y.loc[latest.train_mask].reset_index(drop=True)
        y_valid = y.loc[latest.validation_mask].reset_index(drop=True)

        started = time.perf_counter()
        prediction = np.full(len(y_valid), y_train.median())
        record(rows, "global_median", "raw", latest.name, y_valid, prediction, time.perf_counter() - started)
        started = time.perf_counter()
        prediction = group_median_predict(train_core_raw, valid_core_raw, y_train)
        record(rows, "town_flat_type_median", "raw", latest.name, y_valid, prediction, time.perf_counter() - started)

        X_train = core_features.loc[latest.train_mask].reset_index(drop=True)
        X_valid = core_features.loc[latest.validation_mask].reset_index(drop=True)
        classical_models = {
            "ridge": make_pipeline(classical_preprocessor(X_train, linear=True), Ridge(alpha=10.0)),
            "decision_tree": make_pipeline(
                classical_preprocessor(X_train, linear=False),
                DecisionTreeRegressor(max_depth=18, min_samples_leaf=20, random_state=args.seed),
            ),
            "random_forest": make_pipeline(
                classical_preprocessor(X_train, linear=False),
                RandomForestRegressor(
                    n_estimators=180,
                    max_features=0.7,
                    min_samples_leaf=3,
                    n_jobs=-1,
                    random_state=args.seed,
                ),
            ),
        }
        for name, model in classical_models.items():
            started = time.perf_counter()
            model.fit(X_train, y_train)
            prediction = model.predict(X_valid)
            record(rows, name, "core", latest.name, y_valid, prediction, time.perf_counter() - started)

    latest_spatial_prediction: np.ndarray | None = None
    for fold in folds:
        y_train = y.loc[fold.train_mask].reset_index(drop=True)
        y_valid = y.loc[fold.validation_mask].reset_index(drop=True)
        for feature_set, feature_frame in (("core", core_features), ("core_spatial", spatial_features)):
            X_train = feature_frame.loc[fold.train_mask].reset_index(drop=True)
            X_valid = feature_frame.loc[fold.validation_mask].reset_index(drop=True)
            started = time.perf_counter()
            prediction, best_iteration = fit_predict_catboost(
                X_train, y_train, X_valid, y_valid, args.iterations, args.seed
            )
            record(
                rows,
                "catboost",
                feature_set,
                fold.name,
                y_valid,
                prediction,
                time.perf_counter() - started,
                best_iteration,
            )
            if fold == latest and feature_set == "core_spatial":
                latest_spatial_prediction = prediction

    results = pd.DataFrame(rows)
    results.to_csv(reports / "stage2_experiment_results.csv", index=False)
    summary = (
        results.groupby(["model", "feature_set"], as_index=False)
        .agg(
            folds=("fold", "count"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            mean_bias=("bias", "mean"),
            total_seconds=("elapsed_seconds", "sum"),
        )
        .sort_values("mean_rmse")
    )
    summary.to_csv(reports / "stage2_experiment_summary.csv", index=False)
    if latest_spatial_prediction is not None:
        valid_meta = train_raw.loc[latest.validation_mask].reset_index(drop=True)
        errors = subgroup_metrics(
            valid_meta,
            y.loc[latest.validation_mask].reset_index(drop=True),
            latest_spatial_prediction,
            ["RENT_APPROVAL_DATE", "TOWN", "FLAT_TYPE"],
        )
        errors.to_csv(reports / "stage2_subgroup_errors.csv", index=False)
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
        "seed": args.seed,
    }
    (reports / "stage2_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
