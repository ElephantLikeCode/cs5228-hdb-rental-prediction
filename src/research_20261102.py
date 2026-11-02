"""Isolated CPU experiments; never edit the selected config or upload predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

try:
    from .check_setup import RAW_SHA256
    from .features import build_block_lookup
    from .modeling import build_features, build_training_features, load_config
    from .validation import regression_metrics
except ImportError:
    from check_setup import RAW_SHA256
    from features import build_block_lookup
    from modeling import build_features, build_training_features, load_config
    from validation import regression_metrics

ROOT = Path(__file__).resolve().parents[1]
TARGET = "MONTHLY_RENT"
WINDOWS = {
    "short_recent": ("2025-01", "2025-03"),
    "long_recent": ("2024-04", "2025-03"),
    "long_older": ("2023-04", "2024-03"),
}
CANDIDATES = ("reference", "lgbm", "lgbm_relative", "cat_raw", "cat_log")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject unexpected data versions before consuming any labels."""
    required = [k for k in RAW_SHA256 if k != "example-submission.csv"]
    missing = [k for k in required if not (raw_dir / k).is_file()]
    if missing:
        raise FileNotFoundError("Missing official data: " + ", ".join(missing))
    for name in required:
        if digest(raw_dir / name) != RAW_SHA256[name]:
            raise ValueError(f"Unexpected dataset checksum: {name}")
    train = pd.read_csv(raw_dir / "train.csv")
    test = pd.read_csv(raw_dir / "test.csv")
    if TARGET not in train or TARGET in test:
        raise ValueError("Unexpected target schema")
    if train.drop(columns=TARGET).columns.tolist() != test.columns.tolist():
        raise ValueError("Train/test feature schema differs")
    return train, test


def raw_signatures(features: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Keep original spellings and area structure, using no target information."""
    result = features.copy()
    for column in ("TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL"):
        values = raw[column].astype("string").fillna("missing")
        result[f"RAW_{column}"] = values.astype(str).to_numpy()
        result[f"RAW_{column}_UPPER_COUNT"] = values.str.count("[A-Z]").to_numpy()
    area = pd.to_numeric(raw["FLOOR_AREA_SQM"], errors="coerce")
    result["AREA_MOD_5"] = (area % 5).to_numpy()
    result["AREA_MOD_10"] = (area % 10).to_numpy()
    result["AREA_BIN_5"] = (5 * np.floor(area / 5)).to_numpy()
    return result


def encode_lgbm(train: pd.DataFrame, query: pd.DataFrame):
    """Vocabulary is fit on training data; unknown query categories are missing."""
    left, right = train.copy(), query.copy()
    vocabulary = {}
    for column in left.select_dtypes(include=["object", "string", "category"]):
        categories = sorted(left[column].dropna().astype(str).unique().tolist())
        vocabulary[column] = categories
        left[column] = pd.Categorical(left[column], categories=categories)
        right[column] = pd.Categorical(right[column], categories=categories)
    return left, right, vocabulary


def fit_component(train, target, query, *, learner, mode, seed, config, threads,
                  sample_weight=None):
    """Fixed iterations; the scoring-window labels cannot enter fitting."""
    if mode in ("residual", "relative"):
        offset_train = train["ADAPT_ADDRESS_FLAT_AREA_LEVEL"].to_numpy(float)
        offset_query = query["ADAPT_ADDRESS_FLAT_AREA_LEVEL"].to_numpy(float)
    else:
        offset_train, offset_query = np.zeros(len(train)), np.zeros(len(query))
    if mode == "relative":
        scale_train = np.maximum(offset_train, 500) / 3000
        scale_query = np.maximum(offset_query, 500) / 3000
    else:
        scale_train, scale_query = np.ones(len(train)), np.ones(len(query))
    transformed = (np.asarray(target) - offset_train) / scale_train
    if mode == "log":   # 2026-11-02: log target, back-transformed with the training-set smearing factor
        transformed = np.log(np.asarray(target, float))
    weights = scale_train**2 * (1 if sample_weight is None else np.asarray(sample_weight))
    vocabulary = None
    if learner == "lgbm":
        train, query, vocabulary = encode_lgbm(train, query)
        model = LGBMRegressor(
            objective="regression", n_estimators=config["iterations"],
            learning_rate=config["learning_rate"], num_leaves=31,
            min_child_samples=50, reg_lambda=10, colsample_bytree=0.9,
            subsample=0.9, subsample_freq=1, max_bin=255,
            random_state=seed, n_jobs=threads, verbosity=-1,
            deterministic=True, force_col_wise=True,
        )
        model.fit(train, transformed, sample_weight=weights)
    else:
        model = CatBoostRegressor(
            loss_function="RMSE", iterations=config["iterations"],
            learning_rate=config["learning_rate"], depth=config["depth"],
            l2_leaf_reg=config["l2_leaf_reg"], random_seed=seed,
            task_type="CPU", thread_count=threads,
            allow_writing_files=False, verbose=False,
        )
        categorical = train.select_dtypes(include=["object", "string", "category"])
        model.fit(train, transformed, sample_weight=weights,
                  cat_features=categorical.columns.tolist())
    if mode == "log":
        smear = float(np.average(np.exp(transformed - model.predict(train)), weights=weights))
        prediction = np.exp(model.predict(query)) * smear
    else:
        prediction = model.predict(query) * scale_query + offset_query
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    return model, prediction, vocabulary


def data_audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    keys = train.columns.drop(TARGET).tolist()
    groups = train.groupby(keys, dropna=False)[TARGET].agg(["count", "nunique", "mean"])
    group_mean = train.groupby(keys, dropna=False)[TARGET].transform("mean")
    conflict = groups["nunique"].gt(1)
    return {
        "train_rows": len(train), "test_rows": len(test),
        "train_months": [train.RENT_APPROVAL_DATE.min(), train.RENT_APPROVAL_DATE.max()],
        "test_months": [test.RENT_APPROVAL_DATE.min(), test.RENT_APPROVAL_DATE.max()],
        "exact_input_groups": len(groups),
        "conflicting_input_groups": int(conflict.sum()),
        "rows_in_conflicting_groups": int(groups.loc[conflict, "count"].sum()),
        "in_sample_exact_input_group_mean_rmse": float(
            np.sqrt(np.mean((train[TARGET] - group_mean) ** 2))
        ),
        "warning": "Group-mean RMSE uses the row's own label. It is an in-sample diagnostic, not an OOS result or a Bayes error bound.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/research_20261102")
    parser.add_argument("--windows", nargs="+", choices=list(WINDOWS), default=["short_recent"])
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    source, test = checked_data(args.raw_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    audit = data_audit(source, test)
    (args.output / "data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return
    config = load_config()
    code_hash = hashlib.sha256(b"".join(
        (ROOT / "src" / name).read_bytes()
        for name in ("research_20261102.py", "features.py", "modeling.py")
    )).hexdigest()
    packages = {name: version(name) for name in
                ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm")}
    protocol = {
        "config": config, "windows": {w: WINDOWS[w] for w in args.windows},
        "seeds": args.seeds, "candidates": args.candidates,
        "packages": packages, "platform": platform.platform(), "code_hash": code_hash,
        "data_hash": digest(args.raw_dir / "train.csv"),
        "reference": "Selected recipe, CPU; one seed = four heads, two seeds = eight heads.",
        "selection": "Exploratory windows already used by teammates; no claim of untouched holdout. Confirm improvements on long_recent and long_older before promotion.",
        "stopping": "Four predeclared candidates. No scoring-label early stopping, no target removal, no leaderboard tuning, no automatic upload.",
    }
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
    folder = args.output / identity
    folder.mkdir(exist_ok=True)
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2))
    lookup = build_block_lookup(args.raw_dir / "auxiliary")
    raw, target = source.drop(columns=TARGET), source[TARGET]
    rows = []
    for window in args.windows:
        start, end = WINDOWS[window]
        train_mask = raw.RENT_APPROVAL_DATE.lt(start)
        valid_mask = raw.RENT_APPROVAL_DATE.between(start, end)
        train = raw.loc[train_mask].reset_index(drop=True)
        valid = raw.loc[valid_mask].reset_index(drop=True)
        y_train = target.loc[train_mask].reset_index(drop=True)
        y_valid = target.loc[valid_mask].reset_index(drop=True)
        assert train.RENT_APPROVAL_DATE.max() < valid.RENT_APPROVAL_DATE.min()
        pd.DataFrame({"source_row": np.flatnonzero(valid_mask), "target": y_valid}).to_csv(
            folder / f"{window}_scoring_rows.csv", index=False)
        median_prediction = np.full(len(valid), np.median(y_train))
        rows.append(dict(window=window, candidate="train_median_control",
                         **regression_metrics(y_valid, median_prediction)))
        cache = {}
        for view in config["history_views"]:
            recipe, lag = view["history"], view["lag"]
            cache_path = folder / f"{window}_{recipe}_{lag}.pkl"
            if cache_path.exists():
                cache[recipe, lag] = pd.read_pickle(cache_path)
            else:
                print(f"Building {window} {recipe} lag={lag}", flush=True)
                # lag is TRAINING augmentation, not a prediction-time lag.
                # Match train_final_model.py, including half-weight duplicates.
                fit_x, fit_y, fit_weight = build_training_features(
                    train, y_train, lookup, dict(config, history=recipe, augmentation_lag=lag))
                fit_raw = train.copy()
                if lag:
                    dates = pd.to_datetime(train.RENT_APPROVAL_DATE)
                    months = dates.dt.year * 12 + dates.dt.month
                    eligible = months.ge(months.min() + lag + 6)
                    fit_raw = pd.concat([train, train.loc[eligible]], ignore_index=True)
                assert len(fit_raw) == len(fit_y) == len(fit_x) == len(fit_weight)
                pair = (fit_x, build_features(train, y_train, valid, lookup, recipe),
                        fit_y, fit_weight, fit_raw)
                pd.to_pickle(pair, cache_path)
                cache[recipe, lag] = pair
        for candidate in args.candidates:
            started = time.perf_counter()
            predictions = []
            modes = ["relative"] if candidate == "lgbm_relative" else (["log"] if candidate == "cat_log" else ["direct", "residual"])
            for seed in args.seeds:
                for view in config["history_views"]:
                    recipe, lag = view["history"], view["lag"]
                    left, right, fit_y, fit_weight, fit_raw = cache[recipe, lag]
                    if candidate == "cat_raw":
                        left, right = raw_signatures(left, fit_raw), raw_signatures(right, valid)
                    for mode in modes:
                        stem = f"{window}_{candidate}_{recipe}_{lag}_{mode}_{seed}"
                        prediction_path = folder / f"{stem}.npy"
                        if prediction_path.exists():
                            prediction = np.load(prediction_path)
                        else:
                            print(f"Fitting {stem}", flush=True)
                            learner = "lgbm" if candidate.startswith("lgbm") else "catboost"
                            model, prediction, vocabulary = fit_component(
                                left, fit_y, right, learner=learner, mode=mode,
                                seed=seed, config=config, threads=args.threads,
                                sample_weight=fit_weight)
                            if learner == "lgbm":
                                model.booster_.save_model(str(folder / f"{stem}.txt"))
                                (folder / f"{stem}_categories.json").write_text(json.dumps(vocabulary))
                            else:
                                model.save_model(str(folder / f"{stem}.cbm"))
                            np.save(prediction_path, prediction)
                        predictions.append(prediction)
            prediction = np.mean(predictions, axis=0)
            np.save(folder / f"{window}_{candidate}_ensemble.npy", prediction)
            record = dict(window=window, candidate=candidate, heads=len(predictions),
                          seconds=time.perf_counter() - started,
                          **regression_metrics(y_valid, prediction))
            rows.append(record)
            pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
            print(json.dumps(record), flush=True)
            bins = pd.cut(y_valid, [-np.inf, 2500, 3000, 3500, 4000, np.inf])
            errors = pd.DataFrame({"target": y_valid, "prediction": prediction, "bin": bins})
            errors.to_csv(folder / f"{window}_{candidate}_errors.csv", index=False)
    print(f"Results: {folder}", flush=True)


if __name__ == "__main__":
    main()
