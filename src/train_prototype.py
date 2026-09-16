"""Train the first reproducible HDB-rent prototype and create a Kaggle file.

The script deliberately starts with a chronological holdout, two transparent
baselines, and one task-suitable boosted-tree model. It uses only information
available in the supplied core train/test files; auxiliary features are added
later through controlled ablations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import root_mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TARGET = "MONTHLY_RENT"
DATE_COL = "RENT_APPROVAL_DATE"


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe features shared by train, validation, and test."""
    result = frame.copy()
    dates = pd.to_datetime(result.pop(DATE_COL), format="%Y-%m", errors="raise")
    result["RENT_YEAR"] = dates.dt.year.astype("int16")
    result["RENT_MONTH"] = dates.dt.month.astype("int8")
    result["MONTH_INDEX"] = (
        (dates.dt.year - 2021) * 12 + dates.dt.month - 1
    ).astype("int16")
    result["LEASE_AGE"] = (
        result["RENT_YEAR"] - pd.to_numeric(result["LEASE_COMMENCE_DATE"], errors="coerce")
    ).clip(lower=0)
    result["ADDRESS"] = (
        result["BLOCK"].astype("string").fillna("missing")
        + " "
        + result["STREET"].astype("string").fillna("missing")
    )

    # These fields are constant in the supplied files and cannot help a model.
    result = result.drop(columns=["FURNISHED", "FEE"], errors="ignore")

    for column in result.select_dtypes(include=["object", "string", "category"]):
        result[column] = result[column].astype("string").fillna("missing").astype(str)
    return result


def group_median_predict(
    train: pd.DataFrame, validation: pd.DataFrame, target: pd.Series
) -> np.ndarray:
    """Town/flat-type baseline with robust fallbacks for unseen groups."""
    keys = ["TOWN", "FLAT_TYPE"]
    lookup = train.assign(_target=target.to_numpy()).groupby(keys)["_target"].median()
    fallback = train.assign(_target=target.to_numpy()).groupby("FLAT_TYPE")["_target"].median()
    global_median = float(target.median())

    values = []
    for town, flat_type in validation[keys].itertuples(index=False, name=None):
        values.append(lookup.get((town, flat_type), fallback.get(flat_type, global_median)))
    return np.asarray(values, dtype=float)


def chronological_split(frame: pd.DataFrame, months: int) -> tuple[np.ndarray, np.ndarray, str]:
    dates = pd.to_datetime(frame[DATE_COL], format="%Y-%m", errors="raise")
    unique_months = np.sort(dates.unique())
    if len(unique_months) <= months:
        raise ValueError("Not enough unique months for the requested validation window")
    cutoff = pd.Timestamp(unique_months[-months])
    train_mask = (dates < cutoff).to_numpy()
    validation_mask = ~train_mask
    return train_mask, validation_mask, cutoff.strftime("%Y-%m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-months", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "submissions" / "prototype_catboost_v1.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    train_raw = pd.read_csv(RAW_DIR / "train.csv")
    test_raw = pd.read_csv(RAW_DIR / "test.csv")
    sample = pd.read_csv(RAW_DIR / "example-submission.csv")

    train_mask, validation_mask, cutoff = chronological_split(
        train_raw, args.validation_months
    )
    y = train_raw[TARGET].astype(float)
    core = train_raw.drop(columns=[TARGET])

    X_train_raw = core.loc[train_mask].reset_index(drop=True)
    X_valid_raw = core.loc[validation_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_valid = y.loc[validation_mask].reset_index(drop=True)

    global_prediction = np.full(len(y_valid), float(y_train.median()))
    group_prediction = group_median_predict(X_train_raw, X_valid_raw, y_train)

    X_train = make_features(X_train_raw)
    X_valid = make_features(X_valid_raw)
    categorical = X_train.select_dtypes(include=["object"]).columns.tolist()

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=args.iterations,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=args.seed,
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=categorical,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=80,
        use_best_model=True,
    )
    validation_prediction = model.predict(X_valid)
    best_iteration = int(model.get_best_iteration())
    final_iterations = max(best_iteration + 1, 100)

    metrics = {
        "validation_cutoff": cutoff,
        "validation_months": args.validation_months,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "metric": "RMSE",
        "global_median_rmse": float(
            root_mean_squared_error(y_valid, global_prediction)
        ),
        "town_flat_type_median_rmse": float(
            root_mean_squared_error(y_valid, group_prediction)
        ),
        "catboost_rmse": float(
            root_mean_squared_error(y_valid, validation_prediction)
        ),
        "catboost_best_iteration": best_iteration,
        "final_iterations": final_iterations,
        "seed": args.seed,
    }

    X_full = make_features(core)
    X_test = make_features(test_raw)
    categorical_full = X_full.select_dtypes(include=["object"]).columns.tolist()
    final_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=final_iterations,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=5.0,
        random_seed=args.seed,
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )
    final_model.fit(X_full, y, cat_features=categorical_full)
    test_prediction = final_model.predict(X_test)

    submission = sample.copy()
    if submission.columns.tolist() != ["Id", "Predicted"]:
        raise ValueError("Unexpected sample-submission schema")
    if len(submission) != len(test_prediction):
        raise ValueError("Prediction length does not match sample submission")
    submission["Predicted"] = test_prediction
    if not np.isfinite(submission["Predicted"]).all():
        raise ValueError("Submission contains non-finite predictions")

    args.submission.parent.mkdir(parents=True, exist_ok=True)
    args.submission.write_text(submission.to_csv(index=False), encoding="utf-8")

    model_path = ROOT / "artifacts" / "prototype_catboost_v1.cbm"
    final_model.save_model(model_path)
    metrics["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    metrics["submission"] = str(args.submission)
    metrics["model"] = str(model_path)

    metrics_path = ROOT / "reports" / "prototype_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
