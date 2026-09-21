"""Train the validated final CatBoost pipeline and create a submission file."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from modeling import (DEFAULT_CONFIG, CANDIDATES, load_config, build_features,
                      build_training_features, fit_model, predict_model)

from features import (
    build_block_lookup,
    suspected_low_rent_mask,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TARGET = "MONTHLY_RENT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate", choices=list(CANDIDATES))
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--augmentation-lags", type=int, nargs="+",
                        help="Optional experimental training views; does not modify configs/model.json.")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument(
        "--outlier-policy",
        choices=["keep", "remove"],
        default=None,
        help="Keep all labels, or remove only reproducibly flagged low-rent labels.",
    )
    parser.add_argument("--outlier-ratio", type=float, default=0.5)
    parser.add_argument("--model-prefix", default="final_model")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "reports" / "final_model_metadata.json",
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "submissions" / "final_submission.csv",
    )
    return parser.parse_args()


def validate_output_paths(args) -> None:
    """Keep experimental runs away from the mainline and raw data."""
    root = ROOT.resolve()
    for field, folder in [("submission", "submissions"), ("metadata", None)]:
        path = Path(getattr(args, field))
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        allowed = [root / folder] if folder else [root / "reports", root / "artifacts"]
        if not any(path.is_relative_to(base) and path != base for base in allowed):
            raise ValueError(f"{field} must stay inside its project output directory")
        setattr(args, field, path)
    prefix = (root / "artifacts" / args.model_prefix).resolve()
    if not prefix.is_relative_to(root / "artifacts") or prefix == root / "artifacts":
        raise ValueError("model-prefix must stay inside artifacts")
    custom = (args.candidate is not None or args.config.resolve() != DEFAULT_CONFIG.resolve()
              or args.augmentation_lags is not None or args.seeds is not None
              or args.iterations is not None or args.outlier_policy is not None
              or args.outlier_ratio != 0.5)
    if custom and (args.submission == root / "submissions/final_submission.csv"
                   or args.metadata == root / "reports/final_model_metadata.json"
                   or (prefix.parent == root / "artifacts" and prefix.name.startswith("final_model"))):
        raise ValueError("Experimental overrides require separate --submission, --metadata and --model-prefix")


def archive_previous_run(metadata_path: Path, submission_path: Path) -> Path | None:
    """Snapshot an existing complete run before writing any replacement models.

    The snapshot mirrors repository-relative paths; metadata remains byte-exact.
    Treat the snapshot directory as its root when resolving the saved paths.
    """
    if not metadata_path.exists() and not submission_path.exists():
        return None
    if not metadata_path.exists() or not submission_path.exists():
        raise ValueError("Incomplete previous run; use fresh output paths instead of overwriting")
    previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    digest = previous["submission_sha256"]
    archive = ROOT / "artifacts" / "submitted" / digest
    files = [(submission_path, digest), (metadata_path, None)]
    files.extend((ROOT / c["model_path"], c["sha256"]) for c in previous["components"])
    for path, expected in files:
        path = path.resolve()
        path.relative_to(ROOT.resolve())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and actual != expected:
            raise ValueError(f"Previous artifact hash mismatch: {path}")
        destination = archive / path.relative_to(ROOT.resolve())
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != actual:
            raise ValueError(f"Archive collision: {destination}")
    for path, _ in files:
        destination = archive / path.resolve().relative_to(ROOT.resolve())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)
        if destination.read_bytes() != path.read_bytes():
            raise ValueError(f"Archive verification failed: {destination}")
    return archive


def main() -> None:
    args = parse_args()
    validate_output_paths(args)
    config = dict(CANDIDATES[args.candidate]) if args.candidate else load_config(args.config)
    if args.augmentation_lags is not None:
        config.update(augmentation_lags=args.augmentation_lags, name="custom_history_views")
    args.seeds = args.seeds if args.seeds is not None else config["seeds"]
    args.outlier_policy = args.outlier_policy or config["outlier_policy"]
    config["seeds"] = args.seeds
    config["outlier_policy"] = args.outlier_policy
    if args.iterations is not None:
        config["iterations"] = args.iterations
    args.iterations = config["iterations"]
    target_modes = config.get("target_modes", [config["residual_target"]])
    if not target_modes or len(set(target_modes)) != len(target_modes):
        raise ValueError("target_modes must contain unique prediction modes")
    if any(mode not in (False, True) for mode in target_modes):
        raise ValueError("target_modes must be booleans")
    if any(target_modes) and config["history"] != "hierarchical":
        raise ValueError("Residual components require hierarchical features")
    history_lags = config.get("augmentation_lags", [config.get("augmentation_lag", 0)])
    if not history_lags or len(set(history_lags)) != len(history_lags):
        raise ValueError("augmentation_lags must contain unique history views")
    if any(not isinstance(lag, int) or lag < 0 for lag in history_lags):
        raise ValueError("augmentation_lags must be nonnegative integers")
    if not args.submission.is_absolute():
        args.submission = (ROOT / args.submission).resolve()
    if not args.metadata.is_absolute():
        args.metadata = (ROOT / args.metadata).resolve()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain unique seed values")
    archive_previous_run(args.metadata, args.submission)
    started = time.perf_counter()
    train = pd.read_csv(RAW / "train.csv")
    test = pd.read_csv(RAW / "test.csv")
    sample = pd.read_csv(RAW / "example-submission.csv")
    lookup = build_block_lookup(RAW / "auxiliary")
    raw_y = train[TARGET].astype(float)
    raw_train = train.drop(columns=[TARGET])
    suspected = suspected_low_rent_mask(
        raw_train, raw_y, ratio_threshold=args.outlier_ratio
    )
    if args.outlier_policy == "remove":
        model_raw_train = raw_train.loc[~suspected].reset_index(drop=True)
        model_y = raw_y.loc[~suspected].reset_index(drop=True)
    else:
        model_raw_train = raw_train.reset_index(drop=True)
        model_y = raw_y.reset_index(drop=True)
    X_test = build_features(model_raw_train, model_y, test, lookup, config["history"])
    history_columns = [c for c in X_test if c.startswith(("HIST_", "ADAPT_"))]
    predictions: list[np.ndarray] = []
    model_paths: list[Path] = []
    components: list[dict] = []
    training_views = {}
    for lag in history_lags:
        view_config = dict(config, augmentation_lag=lag)
        X_train, fit_y, fit_weight = build_training_features(model_raw_train, model_y, lookup, view_config)
        training_views[str(lag)] = dict(fit_rows=len(fit_y), weight_sum=float(fit_weight.sum()))
        for seed in args.seeds:
            for residual_mode in target_modes:
                component_config = dict(view_config, residual_target=residual_mode)
                component_started = time.perf_counter()
                print(f"FINAL FIT history_lag={lag} residual={residual_mode} seed={seed}", flush=True)
                model = fit_model(X_train, fit_y, component_config, seed, args.task_type,
                                  verbose=200, sample_weight=fit_weight)
                predictions.append(predict_model(model, X_test, component_config))
                suffix = (f"_lag{lag}" if lag else "") + ("_residual" if residual_mode else "")
                model_path = ROOT / "artifacts" / f"{args.model_prefix}{suffix}_seed_{seed}.cbm"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model.save_model(model_path)
                restored = model.__class__()
                restored.load_model(model_path)
                restored_prediction = predict_model(restored, X_test.iloc[:100], component_config)
                if not np.allclose(restored_prediction, predictions[-1][:100], rtol=1e-12, atol=1e-9):
                    raise ValueError("Saved model does not reproduce rent-unit predictions")
                del restored
                model_paths.append(model_path)
                components.append(dict(seed=seed, residual_target=residual_mode, augmentation_lag=lag,
                                       model_path=str(model_path.relative_to(ROOT)),
                                       actual_trees=model.tree_count_,
                                       elapsed_seconds=round(time.perf_counter() - component_started, 2),
                                       sha256=hashlib.sha256(model_path.read_bytes()).hexdigest()))
    prediction = np.mean(predictions, axis=0)
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

    metadata = {
        "model": "CatBoostRegressor equal-weight component ensemble",
        "feature_set": "all_spatial_" + config["history"],
        "config": config,
        "feature_count": X_test.shape[1],
        "code_sha256": {name: hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest()
                        for name in ["features.py", "modeling.py", "train_final_model.py"]},
        "history_features": history_columns,
        "iterations": args.iterations,
        "learning_rate": config["learning_rate"],
        "depth": config["depth"],
        "l2_leaf_reg": config["l2_leaf_reg"],
        "task_type": args.task_type,
        "outlier_policy": args.outlier_policy,
        "outlier_ratio": args.outlier_ratio,
        "suspected_low_rent_rows": int(suspected.sum()),
        "suspected_low_rent_fraction": float(suspected.mean()),
        "category_normalization": True,
        "recent_or_market_adjusted_features": config["history"] != "legacy",
        "seeds": args.seeds,
        "weights": [1.0 / len(predictions)] * len(predictions),
        "components": components,
        "raw_train_rows": len(train),
        "train_rows": len(model_raw_train),
        "training_views": training_views,
        "test_rows": len(test),
        "raw_sha256": {name: hashlib.sha256((RAW / name).read_bytes()).hexdigest()
                       for name in ["train.csv", "test.csv"]},
        "submission_sha256": hashlib.sha256(args.submission.read_bytes()).hexdigest(),
        "prediction_min": float(prediction.min()),
        "prediction_max": float(prediction.max()),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "submission": str(args.submission.relative_to(ROOT)),
        "model_paths": [str(path.relative_to(ROOT)) for path in model_paths],
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
