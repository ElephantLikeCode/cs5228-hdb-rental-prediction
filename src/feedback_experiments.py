"""Bounded, chronological experiments motivated by the progress-report feedback.

Run with --smoke for execution QA; formal runs compare against frozen mainline
OOF predictions. Never reads competition test labels or replaces final artifacts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold

try:
    from .features import build_block_lookup, normalize_text, make_core_features
    from .modeling import CANDIDATES, build_features, build_training_features, fit_model, predict_model
    from .validation import regression_metrics
    from .market_experiments import (monthly_market, market_predictions, independent_predictions,
                                     correction_predictions, guarded_ridge_predictions, method_budget, paired_gain_interval)
except ImportError:
    from features import build_block_lookup, normalize_text, make_core_features
    from modeling import CANDIDATES, build_features, build_training_features, fit_model, predict_model
    from validation import regression_metrics
    from market_experiments import (monthly_market, market_predictions, independent_predictions,
                                   correction_predictions, guarded_ridge_predictions, method_budget, paired_gain_interval)

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {
    "short_recent": ("2025-01", "2025-03"),
    "short_middle": ("2024-10", "2024-12"),
    "long_recent": ("2024-04", "2025-03"),
    "long_older": ("2023-04", "2024-03"),
    "horizon16": ("2023-12", "2025-03"),
}
HEADS = ("local_trend_direct", "local_trend_residual", "horizon_direct", "horizon_residual")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_baseline(window, raw, y, return_components=False):
    sources = pd.read_csv(ROOT / "reports/stacking_sources.csv")
    rows = sources.loc[sources.window.eq(window)]
    expected = {(head, seed) for head in HEADS for seed in (42, 2026)}
    if set(zip(rows.candidate, rows.seed)) != expected or len(rows) != 8:
        raise ValueError("Incomplete frozen baseline")
    estimates, ids, components = [], None, {}
    for row in rows.itertuples():
        path = ROOT / row.prediction_file
        if file_hash(path) != row.prediction_sha256:
            raise ValueError("Frozen prediction checksum changed")
        if file_hash(ROOT / "data/raw/train.csv") != row.data_sha256:
            raise ValueError("Frozen baseline uses different raw data")
        with np.load(path) as saved:
            if ids is None:
                ids = saved["row_id"].copy()
            if not np.array_equal(ids, saved["row_id"]):
                raise ValueError("Frozen baseline row mismatch")
            if not np.array_equal(y.iloc[ids].to_numpy(), saved["y"]):
                raise ValueError("Frozen baseline target mismatch")
            if not np.array_equal(raw.iloc[ids].RENT_APPROVAL_DATE.to_numpy(str), saved["month"].astype(str)):
                raise ValueError("Frozen baseline dates mismatch")
            estimates.append(saved["prediction"].copy())
            components[(row.candidate, row.seed)] = saved["prediction"].copy()
    result = ids, np.mean(estimates, axis=0), rows.run_id.tolist()
    return (*result, components) if return_components else result


PREMIUM_GROUPS = {
    "TOWN_FLAT": ["TOWN", "FLAT_TYPE"],
    "STREET_FLAT": ["TOWN", "STREET", "FLAT_TYPE"],
    "ADDRESS_FLAT": ["TOWN", "STREET", "BLOCK", "FLAT_TYPE"],
    "ADDRESS_FLAT_AREA": ["TOWN", "STREET", "BLOCK", "FLAT_TYPE", "AREA"],
}


def comparable_keys(raw):
    keys = pd.DataFrame({c: normalize_text(raw[c]).fillna("missing")
                         for c in ("TOWN", "STREET", "BLOCK", "FLAT_TYPE")})
    keys["AREA"] = pd.to_numeric(raw.FLOOR_AREA_SQM).round(0)
    keys["MONTH"] = raw.RENT_APPROVAL_DATE.astype(str)
    return keys.reset_index(drop=True)


def premium_statistics(reference, target, query, smoothing=20.):
    """All labels belong to the supplied reference; no query targets accepted.

    Remove time/town/flat market levels before aggregating comparable rents.
    Inner-fold encodings may use later months WITHIN outer training, as this is
    a cross-sectional estimator. Outer validation labels remain unavailable.
    """
    if len(reference) != len(target) or len(reference) == 0:
        raise ValueError("Invalid reference targets")
    a, b = comparable_keys(reference), comparable_keys(query)
    a["Y"] = np.asarray(target, float)
    flat_month = a.groupby(["MONTH", "FLAT_TYPE"]).Y.transform("mean")
    local = a.groupby(["MONTH", "TOWN", "FLAT_TYPE"]).Y
    market = (local.transform("sum") + smoothing * flat_month) / (local.transform("count") + smoothing)
    a["RATIO"] = a.Y / market
    a["SECOND"] = a.RATIO**2
    a["LOW"] = (a.RATIO < .65).astype(float)
    prior = a.groupby("FLAT_TYPE").agg(mean=("RATIO", "mean"),
        second=("SECOND", "mean"), low=("LOW", "mean"))
    fallback = a[["RATIO", "SECOND", "LOW"]].mean().to_numpy()
    priors = [b.FLAT_TYPE.map(prior[c]).fillna(fallback[i]).to_numpy()
              for i, c in enumerate(("mean", "second", "low"))]
    output = pd.DataFrame(index=b.index)
    for name, columns in PREMIUM_GROUPS.items():
        stats = a.groupby(columns, dropna=False).agg(count=("RATIO", "size"),
            total=("RATIO", "sum"), second=("SECOND", "sum"), low=("LOW", "sum"))
        merged = b[columns].join(stats, on=columns).fillna(0)
        mass = merged["count"].to_numpy()
        mean = (merged.total.to_numpy() + smoothing * priors[0]) / (mass + smoothing)
        second = (merged.second.to_numpy() + smoothing * priors[1]) / (mass + smoothing)
        output[f"CF_{name}_MEAN"] = mean
        output[f"CF_{name}_LOG_COUNT"] = np.log1p(mass)
        output[f"CF_{name}_STD"] = np.sqrt(np.maximum(second - mean**2, 0))
        output[f"CF_{name}_LOW_RATE"] = (merged.low.to_numpy() + smoothing * priors[2]) / (mass + smoothing)
    if not np.isfinite(output.to_numpy()).all():
        raise ValueError("Nonfinite comparable statistics")
    return output


def crossfit_statistics(train, target, valid, splits=5):
    """Keep identical canonical inputs in one fold; exclude the ENTIRE fold."""
    train = train.reset_index(drop=True)
    target = pd.Series(np.asarray(target))
    canonical = make_core_features(train)
    groups = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
    if len(np.unique(groups)) < splits:
        raise ValueError("Too few distinct input groups for cross-fitting")
    features = None
    fold_id = np.empty(len(train), dtype=np.int8)
    for fold, (reference, query) in enumerate(GroupKFold(splits).split(train, groups=groups)):
        encoded = premium_statistics(train.iloc[reference], target.iloc[reference], train.iloc[query])
        if features is None:
            features = pd.DataFrame(index=train.index, columns=encoded.columns, dtype=float)
        features.iloc[query] = encoded.to_numpy()
        fold_id[query] = fold
    return features, premium_statistics(train, target, valid), fold_id


def past_statistics(train, target):
    """Same statistic definitions, but only strictly earlier training months."""
    train = train.reset_index(drop=True)
    target = pd.Series(np.asarray(target))
    result = None
    for month in sorted(train.RENT_APPROVAL_DATE.unique()):
        query = train.RENT_APPROVAL_DATE.eq(month).to_numpy()
        reference = train.RENT_APPROVAL_DATE.lt(month).to_numpy()
        if reference.any():
            encoded = premium_statistics(train.loc[reference], target.loc[reference], train.loc[query])
        else:
            encoded = pd.DataFrame({f"CF_{group}_{stat}": np.full(query.sum(), 1. if stat == "MEAN" else 0.)
                                   for group in PREMIUM_GROUPS
                                   for stat in ("MEAN", "LOG_COUNT", "STD", "LOW_RATE")})
        if result is None:
            result = pd.DataFrame(index=train.index, columns=encoded.columns, dtype=float)
        result.loc[query] = encoded.to_numpy()
    return result


def crossfit_predictions(train, y, valid, X_train, X_valid, config, seed,
                         task_type, baseline, components):
    clock = time.perf_counter()
    train_extra, valid_extra, folds = crossfit_statistics(train, y, valid)
    prior_extra = past_statistics(train, y)
    statistics_seconds = time.perf_counter() - clock
    result, fit_seconds = {}, {}
    for mode in ("direct", "residual"):
        result[f"matched_{mode}"] = components[(f"local_trend_{mode}", seed)][:len(valid)]
        fit_seconds[f"matched_{mode}"] = 0.
    result["matched_normal_view"] = (result["matched_direct"] + result["matched_residual"]) / 2
    fit_seconds["matched_normal_view"] = 0.
    for view in ("means", "distribution", "past_distribution"):
        columns = [c for c in train_extra if view != "means" or c.endswith(("_MEAN", "_LOG_COUNT"))]
        extra = prior_extra if view == "past_distribution" else train_extra
        a, b = X_train.join(extra[columns]), X_valid.join(valid_extra[columns])
        for mode in ("direct", "residual"):
            cfg = dict(config, residual_target=(mode == "residual"))
            print(f"  crossfit {view}/{mode}: {len(a)} rows, {len(a.columns)} features", flush=True)
            clock = time.perf_counter()
            model = fit_model(a, y, cfg, seed, task_type)
            result[f"crossfit_{view}_{mode}"] = predict_model(model, b, cfg)
            fit_seconds[f"crossfit_{view}_{mode}"] = time.perf_counter() - clock
            del model
            gc.collect()
        result[f"crossfit_{view}_normal_view"] = (result[f"crossfit_{view}_direct"] + result[f"crossfit_{view}_residual"]) / 2
        fit_seconds[f"crossfit_{view}_normal_view"] = sum(fit_seconds[f"crossfit_{view}_{mode}"] for mode in ("direct", "residual"))
        # Change only this seed's TWO normal-history components in the frozen
        # eight-component ensemble. This is an actual 2/8 replacement, not an
        # estimate of what the untrained second seed would do.
        result[f"crossfit_{view}_replace_seed"] = baseline + (
            result[f"crossfit_{view}_direct"] - result["matched_direct"] +
            result[f"crossfit_{view}_residual"] - result["matched_residual"]) / 8
        fit_seconds[f"crossfit_{view}_replace_seed"] = fit_seconds[f"crossfit_{view}_normal_view"]
    return result, train_extra, valid_extra, folds, fit_seconds, statistics_seconds


def information_audit():
    """Aggregate input contracts only; preserve raw rows and all labels."""
    records = []
    input_frames = {}
    for dataset in ("train", "test"):
        frame = pd.read_csv(ROOT / f"data/raw/{dataset}.csv")
        input_frames[dataset] = frame
        for column in frame:
            records.append(dict(dataset=dataset, check="field", field=column,
                rows=len(frame), unique=frame[column].nunique(dropna=False),
                missing=int(frame[column].isna().sum()), value="",
                detail="target" if column == "MONTHLY_RENT" else "input"))
        for edge, value in (("min", frame.RENT_APPROVAL_DATE.min()), ("max", frame.RENT_APPROVAL_DATE.max())):
            records.append(dict(dataset=dataset, check="month_"+edge, field="RENT_APPROVAL_DATE",
                rows=len(frame), value=value, detail="month precision only"))
        inputs = frame.drop(columns=["MONTHLY_RENT", "ID"], errors="ignore")
        canonical = make_core_features(inputs)
        for column in ("TOWN", "FLAT_TYPE", "FLAT_MODEL", "STREET"):
            records.append(dict(dataset=dataset, check="canonical_category_count", field=column,
                rows=len(frame), value=canonical[column].nunique(), detail="production normalization"))
        records.append(dict(dataset=dataset, check="identical_input_rows", field="all_inputs",
            rows=len(frame), value=int(canonical.duplicated(keep=False).sum()),
            detail="not proof of repeated individual-flat transactions"))
    lookup = build_block_lookup(ROOT / "data/raw/auxiliary")
    overlap = set(input_frames["train"].RENT_APPROVAL_DATE) & set(input_frames["test"].RENT_APPROVAL_DATE)
    for dataset, frame in input_frames.items():
        records.append(dict(dataset=dataset, check="overlap_month_rows", field="RENT_APPROVAL_DATE",
            rows=len(frame), value=int(frame.RENT_APPROVAL_DATE.isin(overlap).sum()),
            detail="overlapping month bins: " + ",".join(sorted(overlap))))
    for column in ("TOWN", "FLAT_TYPE", "FLAT_MODEL", "STREET"):
        known = set(normalize_text(input_frames["train"][column]))
        unknown = ~normalize_text(input_frames["test"][column]).isin(known)
        records.append(dict(dataset="test", check="unseen_canonical_category_rows", field=column,
            rows=len(unknown), value=int(unknown.sum()), detail="relative to full provided training data"))
    for path in sorted((ROOT / "data/raw/auxiliary").glob("*.csv")):
        records.append(dict(dataset="auxiliary", check="usage", field=path.name,
            value="not used" if "coe" in path.name or "stock" in path.name else "spatial features",
            detail="current production feature construction"))
    for dataset in ("train", "test"):
        frame = pd.read_csv(ROOT / f"data/raw/{dataset}.csv").drop(columns=["MONTHLY_RENT", "ID"], errors="ignore")
        # Reuse the production spatial join; target-derived history is excluded.
        try:
            from .features import add_spatial_features
        except ImportError:
            from features import add_spatial_features
        spatial = add_spatial_features(make_core_features(frame), lookup, "all_spatial")
        for column in ("LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED"):
            records.append(dict(dataset=dataset, check="spatial_join_missing", field=column,
                rows=len(frame), value=int(spatial[column].isna().sum()),
                detail="before model missing-value preparation"))
    pd.DataFrame(records).to_csv(ROOT / "reports/information_audit.csv", index=False)
    print("AUDIT: reports/information_audit.csv", flush=True)


def flat_group(frame):
    flat = normalize_text(frame.FLAT_TYPE).to_numpy(str)
    return np.where(np.isin(flat, ["5 room", "executive"]), "large",
                    np.where(flat == "4 room", "4 room",
                             np.where(flat == "3 room", "3 room", "small")))


def rent_class(target):
    return np.digitize(np.asarray(target), [2500., 4000.], right=True)


def soft_prediction(probability, experts):
    if probability.shape != experts.shape or not np.allclose(probability.sum(axis=1), 1.):
        raise ValueError("Invalid expert probabilities")
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("Invalid probability values")
    return np.sum(probability * experts, axis=1)


def market_forecast(table, horizon, mode):
    """Only label aggregates before the validation origin enter this forecast."""
    table = table.sort_index().tail(12)
    last = table.tail(3)
    anchor = float(np.average(last["mean"], weights=last["count"]))
    slope = 0. if len(table) < 6 else float(np.polyfit(table.index, np.log(table["mean"]), 1)[0])
    slope = float(np.clip(slope, -.01, .01))
    distance = {"flat": 0., "damped": 6. * (1. - np.exp(-horizon / 6.)),
                "bounded_linear": min(horizon, 18.)}[mode]
    return anchor * np.exp(slope * distance)


def nearest_predictions(train, y, valid, X_train, X_valid):
    """Same flat type, geographical proximity, similar area/age and recency.

    Target ratios use training-only month/flat means. These are a transformation
    of observed reference rents, not query features; validation labels never enter.
    """
    types = normalize_text(train.FLAT_TYPE).to_numpy(str)
    query_types = normalize_text(valid.FLAT_TYPE).to_numpy(str)
    dates = pd.to_datetime(train.RENT_APPROVAL_DATE)
    months = (dates.dt.year * 12 + dates.dt.month).to_numpy()
    qdates = pd.to_datetime(valid.RENT_APPROVAL_DATE)
    query_months = (qdates.dt.year * 12 + qdates.dt.month).to_numpy()
    if months.max() >= query_months.min():
        raise ValueError("Nearest reference includes query/future months")
    monthly = pd.DataFrame({"flat": types, "month": months, "y": np.asarray(y)})
    denominator = monthly.groupby(["flat", "month"]).y.transform("mean").to_numpy()
    ratio = np.asarray(y) / denominator
    coords = ["LATITUDE", "LONGITUDE", "FLOOR_AREA_SQM", "LEASE_AGE"]
    median = X_train[coords].apply(pd.to_numeric).median()
    a = X_train[coords].apply(pd.to_numeric).fillna(median).to_numpy(float)
    b = X_valid[coords].apply(pd.to_numeric).fillna(median).to_numpy(float)
    # Approximate km scaling in Singapore; area 20 sqm and age 20y are comparable
    # to 1 km. Constants fixed before any new validation result.
    scale = np.array([111., 111., 1./20., 1./20.])
    a, b = a * scale, b * scale
    forecasts = {mode: np.full(len(valid), np.mean(y)) for mode in ("flat", "damped", "bounded_linear")}
    for flat in np.unique(query_types):
        fit = np.flatnonzero(types == flat)
        query = np.flatnonzero(query_types == flat)
        if len(fit) < 10:
            continue
        neighbors = NearestNeighbors(n_neighbors=min(64, len(fit)), algorithm="kd_tree").fit(a[fit])
        distance, local = neighbors.kneighbors(b[query])
        indices = fit[local]
        weights = np.exp2((months[indices] - months[fit].max()) / 18.) / (distance + .1)**2
        mass = weights.sum(axis=1)
        premium = (np.sum(weights * ratio[indices], axis=1) + 5.) / (mass + 5.)
        stats = monthly.loc[monthly.flat.eq(flat)].groupby("month").y.agg(["mean", "count"])
        anchor = int(stats.index.max())
        for mode in forecasts:
            for month in np.unique(query_months[query]):
                take = query_months[query] == month
                forecasts[mode][query[take]] = premium[take] * market_forecast(stats, month-anchor, mode)
    return forecasts


def specialist_predictions(train, y, valid, X_train, X_valid, config, seed, task_type, fallback):
    groups, query_groups = flat_group(train), flat_group(valid)
    prediction = fallback.copy()
    for group in np.unique(groups):
        fit, query = groups == group, query_groups == group
        if fit.sum() < 500 or not query.any():
            continue
        print(f"  specialist {group}: {fit.sum()} rows", flush=True)
        model = fit_model(X_train.loc[fit], y.loc[fit], config, seed, task_type)
        prediction[query] = predict_model(model, X_valid.loc[query], config)
        del model
        gc.collect()
    return prediction


def mixture_predictions(y, X_train, X_valid, config, seed, task_type):
    classes = rent_class(y)
    experts = np.empty((len(X_valid), 3))
    if set(np.unique(classes)) != {0, 1, 2}:
        raise ValueError("All rent bands required for mixture smoke/formal run")
    classifier = CatBoostClassifier(iterations=min(config["iterations"], 1000), depth=6,
        learning_rate=.04, l2_leaf_reg=6, random_seed=seed, loss_function="MultiClass",
        task_type=task_type, devices="0" if task_type == "GPU" else None,
        allow_writing_files=False, verbose=False)
    classifier.fit(X_train, classes, cat_features=X_train.select_dtypes(include="object").columns.tolist())
    probability = classifier.predict_proba(X_valid)
    if not np.array_equal(classifier.classes_, [0, 1, 2]):
        raise ValueError("Unexpected class order")
    del classifier
    for band in range(3):
        fit = classes == band
        print(f"  rent expert {band}: {fit.sum()} rows", flush=True)
        model = fit_model(X_train.loc[fit], y.loc[fit], config, seed, task_type)
        experts[:, band] = predict_model(model, X_valid, config)
        del model
        gc.collect()
    return soft_prediction(probability, experts)


def diagnostics(raw, y, ids, prediction, method, window):
    metadata = raw.iloc[ids].copy().reset_index(drop=True)
    metadata["FLAT_TYPE"] = normalize_text(metadata.FLAT_TYPE)
    metadata["RENT_BAND"] = pd.cut(np.asarray(y), [-np.inf, 2500, 4000, np.inf], labels=["low", "middle", "high"])
    metadata["TOWN_FLAT"] = normalize_text(metadata.TOWN) + "/" + metadata.FLAT_TYPE
    error = prediction - np.asarray(y)
    metadata["ERROR"] = error
    metadata["EXTREME_ERROR"] = np.abs(error) >= 1000
    metadata["SSE"] = error**2
    result = []
    for column in ("TOWN", "FLAT_TYPE", "TOWN_FLAT", "RENT_BAND", "RENT_APPROVAL_DATE"):
        for name, part in metadata.groupby(column, observed=True):
            result.append(dict(window=window, method=method, group_column=column,
                group_value=name, rows=len(part), rmse=float(np.sqrt(part.SSE.mean())),
                mae=float(part.ERROR.abs().mean()), bias=float(part.ERROR.mean()),
                extreme_error_rate=float(part.EXTREME_ERROR.mean()),
                sse_fraction=float(part.SSE.sum()/metadata.SSE.sum())))
    return result


def run(args):
    if args.summarize_runs:
        summarize_runs(args.summarize_runs)
        return
    information_audit()
    if args.audit_only:
        return
    raw_path = ROOT / "data/raw/train.csv"
    raw = pd.read_csv(raw_path)
    y = raw.pop("MONTHLY_RENT")
    config = dict(CANDIDATES["local_trend_direct"], iterations=40 if args.smoke else 1600)
    identity = dict(data_sha256=file_hash(raw_path), seed=args.seed,
        code_sha256=hashlib.sha256(b"".join((ROOT / "src" / name).read_bytes() for name in
            ("feedback_experiments.py", "market_experiments.py", "features.py", "modeling.py", "validation.py"))).hexdigest(),
        task_type=args.task_type, config=config, smoke=args.smoke,
        nearest_k=64, nearest_half_life=18, fixed_blend=.25, bands=[2500,4000],
        crossfit_splits=5, crossfit_smoothing=20, crossfit_low_ratio=.65,
        crossfit_groups=PREMIUM_GROUPS,
        market_lag_months=1, market_variants=["coe", "stock", "both", "origin_only"],
        independent_learners=["xgboost_depth8", "ridge_alpha100"],
        correction_oof_months=12, correction_low_ratio=.65,
        correction_trees=600, correction_depth=4, correction_weights=[.25,.5],
        auxiliary_sha256={p.name: file_hash(p) for p in sorted((ROOT / "data/raw/auxiliary").glob("*.csv"))},
        python_version=platform.python_version(),
        protocol="fixed_budget_chronological_frozen_mainline")
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    folder = ROOT / "artifacts/feedback_study" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Run identity collision")
    manifest.write_text(json.dumps(identity, indent=2))
    snapshot = folder / "source"
    snapshot.mkdir(exist_ok=True)
    for name in ("feedback_experiments.py", "market_experiments.py", "features.py", "modeling.py", "validation.py"):
        destination = snapshot / name
        if not destination.exists():
            shutil.copy2(ROOT / "src" / name, destination)
    lookup = build_block_lookup(ROOT / "data/raw/auxiliary")
    market_table = monthly_market(ROOT / "data/raw/auxiliary") if set(args.families) & {"market", "learners", "ridge_guarded"} else None
    if market_table is not None:
        market_table.to_csv(folder / "monthly_market_features.csv", index_label="month")
    all_rows, all_groups = [], []
    for window in args.windows:
        start, end = WINDOWS[window]
        ids, baseline, source_ids, components = frozen_baseline(window, raw, y, return_components=True)
        expected = raw.index[raw.RENT_APPROVAL_DATE.between(start, end)].to_numpy()
        if not np.array_equal(ids, expected):
            raise ValueError("Mainline does not cover exactly the requested window")
        train = raw.loc[raw.RENT_APPROVAL_DATE.lt(start)].copy()
        train_y = y.loc[train.index].copy()
        if args.smoke:
            train = train.sample(min(3000, len(train)), random_state=42).sort_index()
            train_y = y.loc[train.index].copy()
            ids, baseline = ids[:300], baseline[:300]
        training_ids = train.index.to_numpy().copy()
        valid, valid_y = raw.iloc[ids].reset_index(drop=True), y.iloc[ids].reset_index(drop=True)
        train, train_y = train.reset_index(drop=True), train_y.reset_index(drop=True)
        train.attrs["raw_ids"], valid.attrs["raw_ids"] = training_ids, ids
        predictions = {"mainline": baseline}
        start_clock = time.perf_counter()
        print(f"FEATURES {window}: train={len(train)} valid={len(valid)}", flush=True)
        X_train, fit_y, fit_weight = build_training_features(train, train_y, lookup, config)
        X_valid = build_features(train, train_y, valid, lookup, "local_trend")
        if len(X_train) != len(train) or not np.allclose(fit_weight, 1):
            raise ValueError("Screen requires the unaugmented normal-history view")
        feature_seconds = time.perf_counter() - start_clock
        durations = {"mainline": 0.}
        if set(args.families) & {"market", "learners", "ridge_guarded"}:
            for mode in ("direct", "residual"):
                predictions[f"matched_{mode}"] = components[(f"local_trend_{mode}", args.seed)][:len(valid)]
                durations[f"matched_{mode}"] = 0.
            predictions["matched_normal_view"] = (predictions["matched_direct"] + predictions["matched_residual"]) / 2
            durations["matched_normal_view"] = 0.
        crossfit_feature_seconds = 0.
        for family in args.families:
            method_seconds = None
            cache = folder / f"{window}_{family}.npz"
            if args.resume and cache.exists():
                with np.load(cache) as saved:
                    if not np.array_equal(saved["row_id"], ids) or not np.array_equal(saved["y"], valid_y):
                        raise ValueError("Cached rows/targets changed")
                    methods = saved["methods"].astype(str)
                    values = saved["prediction"]
                    seconds = float(saved["seconds"])
                    estimates = dict(zip(methods, values.T))
                    if "method_seconds" in saved:
                        method_seconds = dict(zip(methods, saved["method_seconds"]))
                        crossfit_feature_seconds = float(saved["statistics_seconds"])
                print(f"REUSE {window}/{family}", flush=True)
            else:
                clock = time.perf_counter()
                if family == "nearest":
                    estimates = {f"nearest_{key}": value for key,value in nearest_predictions(
                        train, train_y, valid, X_train, X_valid).items()}
                elif family == "crossfit":
                    estimates, train_extra, valid_extra, folds, method_seconds, crossfit_feature_seconds = crossfit_predictions(
                        train, fit_y, valid, X_train, X_valid, config, args.seed,
                        args.task_type, baseline, components)
                    np.savez_compressed(folder / f"{window}_crossfit_features.npz",
                        train_features=train_extra, valid_features=valid_extra,
                        columns=train_extra.columns.to_numpy(str), fold_id=folds,
                        train_month=train.RENT_APPROVAL_DATE.to_numpy(str), row_id=ids)
                elif family in {"market", "learners"}:
                    function = market_predictions if family == "market" else independent_predictions
                    estimates, method_seconds = function(train, valid, X_train, fit_y, X_valid,
                        config, args.seed, args.task_type, baseline, components, market_table)
                elif family == "correction":
                    estimates, method_seconds = correction_predictions(raw, y, train, valid,
                        X_train, X_valid, baseline, config, args.seed, args.task_type,
                        frozen_baseline, folder, window)
                elif family == "ridge_guarded":
                    estimates, method_seconds = guarded_ridge_predictions(train, valid,
                        X_train, fit_y, X_valid, config, baseline, market_table, folder, window)
                elif family == "specialists":
                    estimates = {"flat_specialists": specialist_predictions(train, fit_y, valid,
                        X_train, X_valid, config, args.seed, args.task_type, baseline)}
                else:
                    estimates = {"rent_mixture": mixture_predictions(fit_y, X_train, X_valid,
                        config, args.seed, args.task_type)}
                seconds = time.perf_counter() - clock
                extras = {} if method_seconds is None else dict(
                    method_seconds=[method_seconds[n] for n in estimates], statistics_seconds=crossfit_feature_seconds)
                np.savez_compressed(cache, row_id=ids, y=valid_y, methods=list(estimates),
                                    prediction=np.column_stack(list(estimates.values())), seconds=seconds, **extras)
            for name, prediction in estimates.items():
                if not np.isfinite(prediction).all():
                    raise ValueError("Nonfinite predictions")
                predictions[name] = prediction
                durations[name] = seconds if method_seconds is None else method_seconds[name]
                if family in {"nearest", "specialists", "mixture"}:
                    predictions[f"{name}_blend25"] = .75 * baseline + .25 * prediction
                    durations[f"{name}_blend25"] = seconds
        for name, prediction in predictions.items():
            scores = regression_metrics(valid_y, prediction)
            gain = regression_metrics(valid_y, baseline)["rmse"] - scores["rmse"]
            reference = "mainline"
            if name.startswith("crossfit_") and not name.endswith("replace_seed"):
                reference = "matched_normal_view" if name.endswith("normal_view") else "matched_" + name.rsplit("_", 1)[1]
            if name.startswith(("market_", "learner_")) and not name.endswith(("replace_seed", "blend25")):
                reference = "matched_normal_view" if name.endswith("normal_view") else "matched_residual" if "ridge" in name else "matched_" + name.rsplit("_", 1)[1]
            matched_gain = regression_metrics(valid_y, predictions[reference])["rmse"] - scores["rmse"]
            encoding_gain = np.nan
            if name.startswith("crossfit_distribution_"):
                counterpart = name.replace("crossfit_distribution_", "crossfit_past_distribution_")
                encoding_gain = regression_metrics(valid_y, predictions[counterpart])["rmse"] - scores["rmse"]
            feature_group = "all_spatial_local_trend"
            if name == "mainline":
                feature_group = "frozen_local_trend_and_stale_hierarchical"
            elif name.startswith("crossfit_"):
                feature_group += "+" + name.removeprefix("crossfit_").rsplit("_", 1)[0]
            elif name.startswith(("market_", "learner_", "correction_")):
                feature_group += "+" + name
            learner, trees = method_budget(name)
            all_rows.append(dict(run_id=run_id, window=window, method=name, **scores,
                gain=gain, train_end=train.RENT_APPROVAL_DATE.max(), validation_start=start,
                validation_end=end, train_rows=len(train), validation_rows=len(ids), seed=args.seed,
                task_type=args.task_type, feature_seconds=feature_seconds,
                model_seconds=durations[name], trees=40 if args.smoke and trees else trees,
                learner=learner, actual_device="CPU" if "ridge" in name else args.task_type,
                feature_group=feature_group, source_run_ids=json.dumps(source_ids),
                comparison_reference=reference, comparison_gain=matched_gain,
                crossfit_vs_past_gain=encoding_gain,
                crossfit_feature_seconds=crossfit_feature_seconds,
                data_sha256=identity["data_sha256"], code_sha256=identity["code_sha256"],
                protocol=identity["protocol"], smoke=args.smoke))
            all_rows[-1]["availability"] = ("origin_only" if "origin_only" in name else
                "supplied_snapshot_previous_month" if name.startswith("market_") or "xgb_market" in name or "ridge_market" in name or "guarded_market" in name else
                "past_training_labels_only")
            all_groups.extend(diagnostics(raw, valid_y, ids, prediction, name, window))
            print(f"RESULT {window} {name}: RMSE={scores['rmse']:.4f} gain={gain:+.4f}", flush=True)
        pd.DataFrame(all_rows).to_csv(folder / "results.csv", index=False)
        pd.DataFrame(all_groups).to_csv(folder / "error_groups.csv", index=False)
        if not args.smoke:
            path = ROOT / "reports/feedback_results.csv"
            previous = pd.read_csv(path, dtype={"run_id": str}) if path.exists() else pd.DataFrame()
            fresh = pd.DataFrame(all_rows)
            if not previous.empty:
                keys = set(zip(fresh.run_id, fresh.window, fresh.method))
                previous = previous.loc[[key not in keys for key in zip(previous.run_id, previous.window, previous.method)]]
            pd.concat([previous, fresh], ignore_index=True).to_csv(path, index=False)
        del X_train, X_valid
        gc.collect()
    print(f"FINISHED: {folder}", flush=True)


def summarize_runs(run_ids):
    """Audit saved predictions and append descriptive paired intervals.

    Sources are verified against each ORIGINAL fit snapshot, not today's code.
    No model fitting or submission generation occurs during this operation.
    """
    raw = pd.read_csv(ROOT / "data/raw/train.csv")
    y = raw.pop("MONTHLY_RENT")
    ledger_path = ROOT / "reports/feedback_results.csv"
    ledger = pd.read_csv(ledger_path, dtype={"run_id": str})
    for run_id in run_ids:
        if len(run_id) != 16 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("Invalid saved run identifier")
        folder = ROOT / "artifacts/feedback_study" / run_id
        config = json.loads((folder / "config.json").read_text())
        names = ("feedback_experiments.py", "market_experiments.py", "features.py", "modeling.py", "validation.py")
        actual = hashlib.sha256(b"".join((folder / "source" / n).read_bytes() for n in names)).hexdigest()
        if actual != config["code_sha256"] or file_hash(ROOT / "data/raw/train.csv") != config["data_sha256"]:
            raise ValueError("Saved source/data fingerprint changed")
        for name, expected in config["auxiliary_sha256"].items():
            if file_hash(ROOT / "data/raw/auxiliary" / name) != expected:
                raise ValueError("Saved auxiliary fingerprint changed")
        rows = ledger.loc[ledger.run_id.eq(run_id)]
        if config["smoke"] or rows.empty:
            raise ValueError("Only recorded formal runs can be summarized")
        for window, part in rows.groupby("window"):
            ids, baseline, _ = frozen_baseline(window, raw, y)
            estimates = {"mainline": baseline}
            for cache in folder.glob(window + "_*.npz"):
                with np.load(cache) as saved:
                    if "methods" not in saved:
                        continue
                    if not np.array_equal(saved["row_id"], ids) or not np.array_equal(saved["y"], y.iloc[ids]):
                        raise ValueError("Saved validation rows/targets changed")
                    estimates.update(zip(saved["methods"].astype(str), saved["prediction"].T))
            for index, row in part.iterrows():
                if row.method.startswith("matched_"):
                    continue
                prediction = estimates[row.method]
                if not np.isfinite(prediction).all():
                    raise ValueError("Saved nonfinite predictions")
                scores = regression_metrics(y.iloc[ids], prediction)
                if not np.isclose(scores["rmse"], row.rmse, atol=1.e-8, rtol=0):
                    raise ValueError("Ledger RMSE does not reproduce")
                if row.method.endswith(("replace_seed", "blend25")) or row.method.startswith("correction_"):
                    lower, upper = paired_gain_interval(raw, ids, y.iloc[ids].to_numpy(), baseline, prediction)
                    ledger.loc[index, ["paired_gain_lower", "paired_gain_upper"]] = [lower, upper]
            print(f"VERIFIED {run_id}/{window}: source, data, predictions and RMSE", flush=True)
    # Repair the generic base-budget field in older research ledgers. This is
    # metadata only; original run-local files and source snapshots are preserved.
    for index, row in ledger.iterrows():
        learner, trees = method_budget(row.method)
        ledger.loc[index, ["learner", "trees", "actual_device"]] = [learner, trees,
            "CPU" if learner in {"ridge_cpu", "nearest_neighbors"} else row.task_type]
    ledger.to_csv(ledger_path, index=False)
    print("SUMMARY: reports/feedback_results.csv (descriptive intervals, not independent tests)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", nargs="+", choices=list(WINDOWS), default=["short_recent", "horizon16"])
    parser.add_argument("--families", nargs="+", choices=["nearest", "specialists", "mixture", "crossfit", "market", "learners", "correction", "ridge_guarded"], default=["nearest", "specialists", "mixture"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--summarize-runs", nargs="+", help="Verify saved formal runs and append paired intervals without training")
    run(parser.parse_args())
