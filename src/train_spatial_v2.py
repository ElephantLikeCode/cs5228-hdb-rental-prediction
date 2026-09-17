"""Train the validated spatial CatBoost model on all data and create submission v2."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from project_features import (
    add_spatial_features,
    build_block_lookup,
    make_core_features,
    prepare_for_catboost,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TARGET = "MONTHLY_RENT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=965)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "submissions" / "catboost_spatial_v2.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    sample = pd.read_csv(RAW / "example-submission.csv")
    lookup = build_block_lookup(RAW / "auxiliary")
    X_train = prepare_for_catboost(
        add_spatial_features(make_core_features(train.drop(columns=[TARGET])), lookup)
    )
    X_test = prepare_for_catboost(add_spatial_features(make_core_features(test), lookup))
    y = train[TARGET].astype(float)
    categorical = X_train.select_dtypes(include=["object"]).columns.tolist()
    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=args.iterations,
        learning_rate=0.04,
        depth=8,
        l2_leaf_reg=6.0,
        random_seed=args.seed,
        thread_count=-1,
        allow_writing_files=False,
        verbose=200,
    )
    model.fit(X_train, y, cat_features=categorical)
    prediction = model.predict(X_test)
    if sample.columns.tolist() != ["Id", "Predicted"]:
        raise ValueError("Unexpected sample-submission schema")
    if len(sample) != len(prediction):
        raise ValueError("Prediction length does not match sample submission")
    sample["Predicted"] = prediction
    if sample["Id"].tolist() != list(range(len(sample))):
        raise ValueError("Submission IDs are not contiguous and ordered")
    if not np.isfinite(prediction).all():
        raise ValueError("Submission contains non-finite predictions")
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.submission, index=False, lineterminator="\n")
    reloaded = pd.read_csv(args.submission)
    if reloaded.columns.tolist() != sample.columns.tolist():
        raise ValueError("Reloaded submission columns differ from generated data")
    if not reloaded["Id"].equals(sample["Id"]):
        raise ValueError("Reloaded submission IDs differ from generated data")
    if not np.allclose(
        reloaded["Predicted"].to_numpy(),
        sample["Predicted"].to_numpy(),
        rtol=1e-12,
        atol=1e-9,
    ):
        raise ValueError("Reloaded predictions differ beyond CSV precision")

    model_path = ROOT / "artifacts" / "catboost_spatial_v2.cbm"
    model.save_model(model_path)
    metadata = {
        "model": "CatBoostRegressor",
        "feature_set": "core_spatial",
        "iterations": args.iterations,
        "learning_rate": 0.04,
        "depth": 8,
        "l2_leaf_reg": 6.0,
        "seed": args.seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "submission": str(args.submission.relative_to(ROOT)),
        "model_path": str(model_path.relative_to(ROOT)),
    }
    (ROOT / "reports" / "spatial_v2_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
