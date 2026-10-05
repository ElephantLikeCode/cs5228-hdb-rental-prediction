"""Research-only market covariates and honest OOF residual corrections.

Invoked by feedback_experiments.py; never modifies the production recipe.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .modeling import CANDIDATES, fit_model, predict_model, target_offset
except ImportError:
    from modeling import CANDIDATES, fit_model, predict_model, target_offset


def monthly_market(auxiliary: Path) -> pd.DataFrame:
    """No rent labels. Backward-only rolling summaries, not adjusted-close data.

    Use CLOSE to avoid ex-post dividend adjustment. Corporate actions may still
    affect returns; cross-sectional medians reduce, but do not eliminate, this.
    Stock membership reflects the supplied snapshot, not historical membership.
    """
    coe = pd.read_csv(auxiliary / "sg-coe-prices.csv")
    dates = pd.to_datetime(coe.YEAR.astype(str) + "-" + coe.MONTH, format="%Y-%b")
    coe["PERIOD"] = dates.dt.to_period("M")
    for column in ("QUOTA_PREMIUM", "PREVAILING_QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"):
        coe[column] = pd.to_numeric(coe[column].astype(str).str.replace(r"[$,]", "", regex=True))
    coe["DEMAND"] = coe.BIDS_RECEIVED / coe.QUOTA.clip(lower=1)
    premium = coe.pivot_table(index="PERIOD", columns="CATEGORY", values="QUOTA_PREMIUM", aggfunc="mean")
    demand = coe.pivot_table(index="PERIOD", columns="CATEGORY", values="DEMAND", aggfunc="mean")
    stock = pd.read_csv(auxiliary / "sg-stock-prices.csv")
    stock["DATE"] = pd.to_datetime(stock.DATE)
    stock["PERIOD"] = stock.DATE.dt.to_period("M")
    close = stock.sort_values("DATE").pivot_table(index="PERIOD", columns="SYMBOL", values="CLOSE", aggfunc="last")
    months = pd.period_range(min(premium.index.min(), close.index.min()),
                             max(premium.index.max(), close.index.max()), freq="M")
    premium, demand, close = premium.reindex(months), demand.reindex(months), close.reindex(months)
    result = pd.DataFrame(index=months)
    for category in premium:
        result[f"MACRO_COE_{category}_LEVEL"] = premium[category] / 100000.
        result[f"MACRO_COE_{category}_CHANGE3"] = premium[category].pct_change(3, fill_method=None)
        result[f"MACRO_COE_{category}_DEMAND"] = demand[category]
    returns = close.pct_change(fill_method=None)
    median = returns.median(axis=1).clip(-.5, .5)
    result["MACRO_STOCK_INDEX"] = (1 + median.fillna(0)).cumprod()
    result["MACRO_STOCK_RETURN1"] = median
    result["MACRO_STOCK_RETURN3"] = (1 + median).rolling(3, min_periods=3).apply(np.prod, raw=True) - 1
    result["MACRO_STOCK_VOL3"] = median.rolling(3, min_periods=2).std()
    result["MACRO_STOCK_BREADTH"] = returns.gt(0).sum(axis=1) / returns.notna().sum(axis=1).replace(0, np.nan)
    result["MACRO_STOCK_COVERAGE"] = close.notna().sum(axis=1) / len(close.columns)
    return result.replace([np.inf, -np.inf], np.nan)


def align_market(table, query, origin=None):
    """Each row uses the preceding month's information.

    origin freezes ALL future covariates to the last month strictly before the
    forecast origin. Missing history has fixed zero + indicator, never bfill.
    Lag-one is a coarse month-availability proxy, not verified release timing.
    """
    periods = pd.PeriodIndex(query.RENT_APPROVAL_DATE, freq="M") - 1
    if origin is not None:
        cutoff = pd.Period(origin, freq="M") - 1
        periods = pd.PeriodIndex([min(p, cutoff) for p in periods], freq="M")
    values = table.reindex(periods).reset_index(drop=True)
    values["MACRO_MISSING"] = values.isna().mean(axis=1)
    return values.fillna(0.)


def paired_methods(X_train, y, X_valid, config, seed, task_type, baseline, components,
                   variants, prefix):
    predictions, seconds = {}, {}
    for name, (a, b, learner) in variants.items():
        for mode in ("direct", "residual"):
            cfg = dict(config, residual_target=(mode == "residual"))
            if learner == "xgboost":
                cfg = dict(CANDIDATES[f"xgb_trend_depth8_{mode}"], iterations=config["iterations"])
            clock = time.perf_counter()
            print(f"  {prefix}/{name}/{mode}: {len(a)} rows, {len(a.columns)} features", flush=True)
            model = fit_model(a, y, cfg, seed, task_type)
            key = f"{prefix}_{name}_{mode}"
            predictions[key] = predict_model(model, b, cfg)
            seconds[key] = time.perf_counter() - clock
            del model
            gc.collect()
        direct, residual = (predictions[f"{prefix}_{name}_{m}"] for m in ("direct", "residual"))
        pair = (direct + residual) / 2
        cost = sum(seconds[f"{prefix}_{name}_{m}"] for m in ("direct", "residual"))
        predictions[f"{prefix}_{name}_normal_view"] = pair
        predictions[f"{prefix}_{name}_replace_seed"] = baseline + (
            direct - components[("local_trend_direct", seed)][:len(b)] +
            residual - components[("local_trend_residual", seed)][:len(b)]) / 8
        # Fixed blend checks complementarity, not an equal-capacity comparison.
        predictions[f"{prefix}_{name}_blend25"] = .75 * baseline + .25 * pair
        for suffix in ("normal_view", "replace_seed", "blend25"):
            seconds[f"{prefix}_{name}_{suffix}"] = cost
    return predictions, seconds


def market_predictions(train, valid, X_train, y, X_valid, config, seed, task_type,
                       baseline, components, table):
    a = align_market(table, train)
    b = align_market(table, valid)
    frozen = align_market(table, valid, valid.RENT_APPROVAL_DATE.min())
    variants = {}
    for name in ("coe", "stock", "both", "origin_only"):
        columns = [c for c in a if name in ("both", "origin_only") or name.upper() in c or c == "MACRO_MISSING"]
        variants[name] = (X_train.join(a[columns]), X_valid.join((frozen if name == "origin_only" else b)[columns]), "catboost")
    return paired_methods(X_train, y, X_valid, config, seed, task_type, baseline,
                          components, variants, "market")


def independent_predictions(train, valid, X_train, y, X_valid, config, seed,
                            task_type, baseline, components, table):
    a, b = X_train.join(align_market(table, train)), X_valid.join(align_market(table, valid))
    variants = {"xgb_original": (X_train, X_valid, "xgboost"), "xgb_market": (a, b, "xgboost")}
    predictions, seconds = paired_methods(X_train, y, X_valid, config, seed, task_type,
                                          baseline, components, variants, "learner")
    for name, fit, query in (("ridge_original", X_train, X_valid), ("ridge_market", a, b)):
        cats = fit.select_dtypes(include=["object"]).columns.tolist()
        nums = [c for c in fit if c not in cats]
        transform = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), cats)], sparse_threshold=1.)
        model = make_pipeline(transform, Ridge(alpha=100., solver="lsqr"))
        cfg = dict(config, residual_target=True)
        clock = time.perf_counter()
        model.fit(fit, np.asarray(y) - target_offset(fit, cfg))
        values = model.predict(query) + target_offset(query, cfg)
        predictions[f"learner_{name}"] = values
        predictions[f"learner_{name}_blend25"] = .75 * baseline + .25 * values
        cost = time.perf_counter() - clock
        seconds[f"learner_{name}"] = seconds[f"learner_{name}_blend25"] = cost
    return predictions, seconds


def clip_to_training_range(train, query):
    """A training-input-only extrapolation guard; never uses query labels."""
    result = query.copy()
    numeric = train.select_dtypes(include=["number"]).columns
    result[numeric] = result[numeric].clip(lower=train[numeric].min(),
        upper=train[numeric].max(), axis=1)
    return result


def guarded_ridge_predictions(train, valid, X_train, y, X_valid, config,
                              baseline, table, folder, window):
    """Follow-up to unbounded Ridge: same inputs, train-range-clipped inference.

    Training historical-horizon features have little variation; long future
    horizons can create linear extrapolation unsupported by training examples.
    The origin-only market control reuses the SAME fitted market Ridge.
    """
    predictions, seconds, ranges = {}, {}, []
    a = X_train.join(align_market(table, train))
    b = X_valid.join(align_market(table, valid))
    origin = X_valid.join(align_market(table, valid, valid.RENT_APPROVAL_DATE.min()))
    cfg = dict(config, residual_target=True)
    for name, fit, queries in (("original", X_train, {"original": X_valid}),
                               ("market", a, {"market": b, "origin_only": origin})):
        cats = fit.select_dtypes(include=["object"]).columns.tolist()
        nums = [c for c in fit if c not in cats]
        transform = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), nums),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), cats)], sparse_threshold=1.)
        model = make_pipeline(transform, Ridge(alpha=100., solver="lsqr"))
        clock = time.perf_counter()
        model.fit(fit, np.asarray(y) - target_offset(fit, cfg))
        cost = time.perf_counter() - clock
        for label, query in queries.items():
            guarded = clip_to_training_range(fit, query)
            # Keep the ORIGINAL permitted market offset, not its clipped copy.
            values = model.predict(guarded) + target_offset(query, cfg)
            key = f"learner_ridge_guarded_{label}"
            predictions[key] = values
            predictions[key + "_blend25"] = .75 * baseline + .25 * values
            seconds[key] = seconds[key + "_blend25"] = cost
            for column in nums:
                ranges.append(dict(window=window, variant=label, feature=column,
                    train_min=fit[column].min(), train_max=fit[column].max(),
                    query_min=query[column].min(), query_max=query[column].max(),
                    outside_fraction=float(((query[column] < fit[column].min()) |
                                           (query[column] > fit[column].max())).mean())))
    pd.DataFrame(ranges).to_csv(folder / f"{window}_ridge_extrapolation.csv", index=False)
    return predictions, seconds


def method_budget(name):
    """Per-fitted-model budgets; aggregate rows reuse fits, not new training."""
    if "ridge" in name:
        return "ridge_cpu", 0
    if name.startswith("correction_"):
        return "catboost", 600
    if "xgb" in name:
        return "xgboost", 1600
    if name.startswith("nearest_"):
        return "nearest_neighbors", 0
    return "catboost", 1600


def paired_gain_interval(raw, ids, y, baseline, prediction, seed=42, samples=1000):
    """Descriptive address-cluster bootstrap, batched to bound memory use."""
    groups = raw.iloc[ids][["TOWN", "BLOCK", "STREET"]].astype(str).agg("|".join, axis=1)
    codes, _ = pd.factorize(groups)
    counts = np.bincount(codes)
    original = np.bincount(codes, weights=(np.asarray(baseline)-np.asarray(y))**2)
    candidate = np.bincount(codes, weights=(np.asarray(prediction)-np.asarray(y))**2)
    rng, gains = np.random.default_rng(seed), []
    for start in range(0, samples, 100):
        draws = rng.integers(len(counts), size=(min(100, samples-start), len(counts)))
        mass = counts[draws].sum(axis=1)
        gains.extend(np.sqrt(original[draws].sum(axis=1)/mass) - np.sqrt(candidate[draws].sum(axis=1)/mass))
    return tuple(float(x) for x in np.quantile(gains, [.025, .975]))


def honest_residual_rows(raw, y, origin, load_baseline):
    """Use already-frozen predictions from origins before their target months.

    For overlaps choose the most recent origin, without looking at errors.
    The entire outer validation interval is excluded, not just its labels.
    """
    candidates = []
    for window in ("bridge_2022", "long_older", "horizon16", "long_recent", "short_middle", "short_recent"):
        ids, prediction, _ = load_baseline(window, raw, y)
        months = raw.iloc[ids].RENT_APPROVAL_DATE.to_numpy(str)
        first = min(months)
        mask = months < origin
        if mask.any() and first < origin:
            candidates.append(pd.DataFrame({"row_id": ids[mask], "prediction": prediction[mask], "source_origin": first}))
    if not candidates:
        raise ValueError("No past out-of-fold rows for correction")
    pool = pd.concat(candidates).sort_values(["source_origin", "row_id"]).drop_duplicates("row_id", keep="last")
    # Fixed latest twelve months; never select rows using validation errors.
    earliest = str(pd.Period(origin, freq="M") - 12)
    pool = pool.loc[raw.iloc[pool.row_id].RENT_APPROVAL_DATE.to_numpy(str) >= earliest]
    if len(pool) < 1000:
        raise ValueError("Insufficient honest residual-training rows")
    return pool.sort_values("row_id").reset_index(drop=True)


def correction_features(features, prediction):
    columns = [c for c in features if c in ("TOWN", "FLAT_TYPE", "FLAT_MODEL", "FLOOR_AREA_SQM",
        "LEASE_AGE", "REGION", "SUBZONE", "PLANNING_AREA", "LATITUDE", "LONGITUDE",
        "MAX_FLOOR", "YEAR_COMPLETED", "RENT_MONTH") or c.startswith(("DIST_", "COUNT_"))]
    result = features[columns].copy()
    result["BASE_PREDICTION"] = prediction
    return result


def correction_predictions(raw, y, train, valid, X_train, X_valid, baseline,
                           config, seed, task_type, load_baseline, folder, window):
    pool = honest_residual_rows(raw, y, valid.RENT_APPROVAL_DATE.min(), load_baseline)
    # X_train has original raw order; index mapping is explicit (smoke samples).
    # Raw training indices are carried in the runner, not inferred by row values.
    positions = pd.Series(np.arange(len(train)), index=train.attrs["raw_ids"])
    pool = pool.loc[pool.row_id.isin(positions.index)].reset_index(drop=True)
    if len(pool) < (20 if config["iterations"] < 100 else 1000):
        raise ValueError("Too few residual-training rows after sampling")
    a = correction_features(X_train.iloc[positions.loc[pool.row_id].to_numpy()].reset_index(drop=True), pool.prediction.to_numpy())
    b = correction_features(X_valid, baseline)
    target = y.iloc[pool.row_id].to_numpy()
    residual = target - pool.prediction.to_numpy()
    low = target < .65 * pool.prediction.to_numpy()
    cats = a.select_dtypes(include=["object"]).columns.tolist()
    params = dict(iterations=40 if config["iterations"] < 100 else 600, depth=4,
        learning_rate=.04, l2_leaf_reg=20., random_seed=seed,
        task_type=task_type, devices="0" if task_type == "GPU" else None,
        allow_writing_files=False, verbose=False)
    clock = time.perf_counter()
    reg = CatBoostRegressor(loss_function="RMSE", **params).fit(a, residual, cat_features=cats)
    adjustment = reg.predict(b)
    regression_seconds = time.perf_counter() - clock
    del reg
    print(f"  correction {window}: honest OOF rows={len(pool)}, low rate={low.mean():.4f}", flush=True)
    clock = time.perf_counter()
    classifier = CatBoostClassifier(loss_function="Logloss", **params).fit(a, low.astype(int), cat_features=cats)
    probability = classifier.predict_proba(b)[:, 1]
    experts = np.empty((len(b), 2))
    for value in (0, 1):
        mask = low == value
        if mask.sum() < 10:
            experts[:, value] = residual[mask].mean() if mask.any() else 0.
        else:
            expert = CatBoostRegressor(loss_function="RMSE", **params).fit(a.loc[mask], residual[mask], cat_features=cats)
            experts[:, value] = expert.predict(b)
            del expert
    soft_adjustment = (1-probability) * experts[:, 0] + probability * experts[:, 1]
    mixture_seconds = time.perf_counter() - clock
    # Labels below are used for diagnosis ONLY, after prediction is complete.
    query_low = y.loc[valid.attrs["raw_ids"]].to_numpy() < .65 * baseline
    classification = dict(window=window, train_rows=len(pool), train_low_rate=float(low.mean()),
        valid_low_rate=float(query_low.mean()), average_precision=average_precision_score(query_low, probability),
        prevalence_baseline=float(query_low.mean()), brier=brier_score_loss(query_low, probability),
        auc=roc_auc_score(query_low, probability) if np.unique(query_low).size == 2 else np.nan)
    pd.DataFrame([classification]).to_csv(folder / f"{window}_low_rent_detection.csv", index=False)
    pool.to_csv(folder / f"{window}_correction_training_rows.csv", index=False)
    np.savez_compressed(folder / f"{window}_low_rent_probabilities.npz", row_id=valid.attrs["raw_ids"], probability=probability)
    predictions, seconds = {}, {}
    for name, delta, cost in (("residual", adjustment, regression_seconds), ("relative_mixture", soft_adjustment, mixture_seconds)):
        for weight in (.25, .5):
            key = f"correction_{name}_shrink{int(weight*100)}"
            predictions[key] = baseline + weight * delta
            seconds[key] = cost
    return predictions, seconds
