"""Research-only market covariates and honest OOF residual corrections.

Invoked by feedback_experiments.py; never modifies the production recipe.
"""
from __future__ import annotations

import gc
import hashlib
import json
import re
import time
import urllib.request
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


PUBLIC_MACRO_URLS = {
    "cpi": "https://tablebuilder.singstat.gov.sg/api/table/tabledata/M213751?limit=100000",
    "ura": "https://data.gov.sg/api/action/datastore_search?resource_id=d_8e4c50283fb7052a391dfb746a05c853&limit=10000",
}


def download_public_macro(directory):
    """Cache only official aggregate indices, never individual rental records.

    Existing downloads are immutable. Delete/move them explicitly to refresh;
    experiments identify the original bytes, including the current data vintage.
    """
    directory.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for name, url in PUBLIC_MACRO_URLS.items():
        path = directory / f"{name}.json"
        if not path.exists():
            error = None
            for attempt in range(4):
                try:
                    request = urllib.request.Request(url, headers={
                        "User-Agent": "Mozilla/5.0", "Accept": "application/json"})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        payload = response.read()
                    value = json.loads(payload)
                    if name == "cpi":
                        assert any(row["rowText"] == "All Items" for row in value["Data"]["row"])
                    else:
                        assert value["success"] and len(value["result"]["records"]) == value["result"]["total"]
                    path.write_bytes(payload)
                    break
                except Exception as exc:
                    error = exc
                    if attempt == 3:
                        raise RuntimeError(f"Official {name} download failed") from error
                    time.sleep(1)
        receipt[name] = dict(url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (directory / "manifest.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def public_macro_tables(directory):
    """Latest-vintage data with conservative release lags; no future bfill.

    Month-only rentals use information available at the START of the month.
    CPI is released late the following month, so activate month m at m+2.
    URA full quarterly statistics are published the following month; activate
    quarter q at the start of the second month after quarter end. Revisions and
    2024-base CPI rebasing are NOT reconstructed into original historic vintages.
    """
    cpi = json.loads((directory / "cpi.json").read_text())
    row = next(row for row in cpi["Data"]["row"] if row["rowText"] == "All Items")
    values = pd.Series({pd.Period(pd.to_datetime(c["key"], format="%Y %b"), freq="M"):
                        float(c["value"]) for c in row["columns"]}).sort_index()
    a = pd.DataFrame({"PUBLIC_CPI_LEVEL": values / 100.,
        "PUBLIC_CPI_YOY": values.pct_change(12, fill_method=None),
        "PUBLIC_CPI_CHANGE3": values.pct_change(3, fill_method=None)})
    a.index = (a.index + 2).to_timestamp()
    ura = json.loads((directory / "ura.json").read_text())["result"]
    if len(ura["records"]) != ura["total"]:
        raise ValueError("Incomplete URA API pagination")
    rows = pd.DataFrame(ura["records"])
    # The official dataset's All Residential series stops in 2021-Q2.
    # Use one continuous Non-Landed Whole Island series; never splice indices.
    selected = rows.loc[rows.property_type.eq("Non-Landed") & rows.locality.eq("Whole Island")]
    if selected.quarter.duplicated().any() or selected.empty:
        raise ValueError("Unexpected URA non-landed series")
    q = pd.PeriodIndex(selected.quarter.str.replace("-", "", regex=False), freq="Q")
    v = pd.Series(selected["index"].to_numpy(float), index=q).sort_index()
    b = pd.DataFrame({"PUBLIC_URA_LEVEL": v / 100.,
        "PUBLIC_URA_YOY": v.pct_change(4, fill_method=None),
        "PUBLIC_URA_CHANGE1": v.pct_change(fill_method=None)})
    b.index = (b.index.asfreq("M", "end") + 2).to_timestamp()
    return {"cpi": a, "ura": b}


def align_public_macro(table, query, origin=None):
    """Backward as-of release join. Freeze forecasts at their shared origin."""
    dates = pd.to_datetime(query.RENT_APPROVAL_DATE, format="%Y-%m")
    if origin is not None:
        dates = dates.clip(upper=pd.Timestamp(origin))
    keys = pd.DataFrame({"date": dates.to_numpy(), "row": np.arange(len(query))})
    source = table.sort_index().rename_axis("available").reset_index()
    merged = pd.merge_asof(keys.sort_values("date"), source, left_on="date", right_on="available")
    merged = merged.sort_values("row").reset_index(drop=True)
    result = merged[table.columns].copy()
    prefix = table.columns[0].rsplit("_", 1)[0]
    result[prefix + "_MISSING"] = result.isna().mean(axis=1)
    result[prefix + "_STALE_MONTHS"] = (
        dates.reset_index(drop=True).dt.year * 12 + dates.reset_index(drop=True).dt.month -
        merged.available.dt.year * 12 - merged.available.dt.month).fillna(0)
    return result.fillna(0.)


def station_calendar(stations, schedule):
    """Preserve every supplied row, recording verified versus excluded service.

    The catalog is incomplete for LRT; only supplied legacy LRT rows are used.
    Duplicate interchange platforms are deduplicated later by physical NAME.
    Legacy dates are lower-bound flags, not invented historical service dates.
    """
    result = stations.copy().reset_index(drop=True)
    result["effective"] = pd.NaT
    result["announced"] = pd.NaT
    result["source"] = "unverified_or_future_excluded"
    for i, code in enumerate(result.CODE.str.lower()):
        match = re.fullmatch(r"(ew|ns|ne|cc|dt|cg|te)(\d+)", code)
        prefix, number = (match.group(1), int(match.group(2))) if match else ("", -1)
        legacy = (prefix == "ew" and 1 <= number <= 33 or
                  prefix == "ns" and 1 <= number <= 28 and number != 6 or
                  prefix == "ne" and 1 <= number <= 17 or
                  prefix == "cc" and (1 <= number <= 29 and number != 18 or number in (33, 34)) or
                  prefix == "dt" and 1 <= number <= 35 and number != 4 or
                  prefix == "cg" and number in (1, 2) or
                  prefix == "te" and number in (1, 2, 3) or
                  code in ("bp1", "bp6", "pe4", "ptc", "stc"))
        if legacy:
            result.loc[i, ["effective", "announced"]] = pd.Timestamp("2021-01-01")
            result.loc[i, "source"] = "operational_before_study"
    for event in schedule["events"]:
        mask = result.CODE.isin(event["codes"])
        if mask.sum() != len(event["codes"]):
            raise ValueError("Opening schedule does not match supplied station codes")
        result.loc[mask, "effective"] = (pd.Period(event["opening_date"], freq="M") + 1).to_timestamp()
        result.loc[mask, "announced"] = pd.Timestamp(event["announced_date"])
        result.loc[mask, "source"] = event["source"]
    return result


def rail_features(query, core, calendar, origin=None, static=False, events=False):
    """Rental-month service; new line events include already existing interchanges.

    Validation uses only openings with a dated announcement before its origin.
    Opening months are treated as pre-opening because the rental day is unknown.
    Fixed-origin static features are a snapshot control, not historic availability.
    """
    try:
        from .features import _facility_features, EARTH_RADIUS_KM
    except ImportError:
        from features import _facility_features, EARTH_RADIUS_KM
    result = pd.DataFrame(index=np.arange(len(query)))
    months = pd.to_datetime(query.RENT_APPROVAL_DATE, format="%Y-%m").reset_index(drop=True)
    known = calendar if origin is None else calendar.loc[
        calendar.source.eq("operational_before_study") | calendar.announced.lt(pd.Timestamp(origin))]
    for month in sorted(months.unique()):
        ids = np.flatnonzero(months.eq(month))
        when = pd.Timestamp(origin) if static else pd.Timestamp(month)
        open_lines = known.loc[known.effective.le(when)]
        physical = open_lines.drop_duplicates("NAME")
        if physical.empty:
            raise ValueError("No verified service at query date")
        block = core.iloc[ids][["LATITUDE", "LONGITUDE"]].reset_index(drop=True)
        features = _facility_features(block, physical.reset_index(drop=True), "MRT")
        if events:
            new = open_lines.loc[open_lines.source.ne("operational_before_study")]
            if len(new):
                lat = np.radians(block.LATITUDE.to_numpy())[:, None]
                lon = np.radians(block.LONGITUDE.to_numpy())[:, None]
                lat2 = np.radians(new.LATITUDE.to_numpy())[None, :]
                lon2 = np.radians(new.LONGITUDE.to_numpy())[None, :]
                d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(
                    np.sin((lat2-lat)/2)**2 + np.cos(lat)*np.cos(lat2)*np.sin((lon2-lon)/2)**2, 0, 1)))
                age = (when.year * 12 + when.month - new.effective.dt.year * 12 - new.effective.dt.month).to_numpy()
                recent = age < 12
                features["RAIL_NEW_LINE_ACCESS"] = np.exp(-d/.8).sum(axis=1)
                features["RAIL_RECENT_LINE_ACCESS"] = (np.exp(-d/.8) * recent).sum(axis=1)
                features["RAIL_MONTHS_SINCE_NEAREST_NEW_LINE"] = np.minimum(age[d.argmin(axis=1)], 60)
            else:
                features["RAIL_NEW_LINE_ACCESS"] = 0.
                features["RAIL_RECENT_LINE_ACCESS"] = 0.
                features["RAIL_MONTHS_SINCE_NEAREST_NEW_LINE"] = 60.
        for column in features:
            if column not in result:
                result[column] = "" if column == "NEAREST_MRT" else 0.
            result.loc[ids, column] = features[column].to_numpy()
    return result


def public_information_predictions(train, valid, X_train, y, X_valid, config, seed,
                                  task_type, baseline, components, tables, calendar,
                                  selected=None, monthly_controls=False):
    """Isolated macro, corrected static rail, dynamic rail and joint ablations."""
    origin = valid.RENT_APPROVAL_DATE.min()
    variants, controls = {}, {}
    extras = {}
    for name, table in tables.items():
        extras[name] = (align_public_macro(table, train), align_public_macro(table, valid, origin))
        variants[name] = (X_train.join(extras[name][0]), X_valid.join(extras[name][1]), "catboost")
        if monthly_controls:
            controls[name] = {"monthly": X_valid.join(align_public_macro(table, valid))}
    ma = pd.concat([pair[0] for pair in extras.values()], axis=1)
    mb = pd.concat([pair[1] for pair in extras.values()], axis=1)
    variants["macro_both"] = (X_train.join(ma), X_valid.join(mb), "catboost")
    if monthly_controls:
        released = pd.concat([align_public_macro(table, valid) for table in tables.values()], axis=1)
        controls["macro_both"] = {"monthly": X_valid.join(released)}
    rail_columns = [c for c in X_train if c.startswith(("DIST_MRT", "COUNT_MRT", "NEAREST_MRT"))]
    for name, static, events in (("rail_static", True, False), ("rail_dynamic", False, False),
                                 ("rail_events", False, True)):
        a = rail_features(train, X_train, calendar, origin if static else None, static, events)
        b = rail_features(valid, X_valid, calendar, origin, static, events)
        fit = X_train.drop(columns=rail_columns).join(a)
        query = X_valid.drop(columns=rail_columns).join(b)
        variants[name] = (fit, query, "catboost")
        if monthly_controls and not static:
            monthly_rail = X_valid.drop(columns=rail_columns).join(
                rail_features(valid, X_valid, calendar, events=events))
            controls[name] = {"monthly": monthly_rail}
        if events:
            variants["combined"] = (fit.join(ma), query.join(mb), "catboost")
            if monthly_controls:
                controls["combined"] = {"monthly": monthly_rail.join(released)}
    if selected:
        variants = {name: variant for name, variant in variants.items() if name in selected}
    return paired_methods(X_train, y, X_valid, config, seed, task_type, baseline,
                          components, variants, "public", controls)


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
                   variants, prefix, query_controls=None):
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
            for label, query in (query_controls or {}).get(name, {}).items():
                control_key = f"{prefix}_{name}_{label}_{mode}"
                predictions[control_key] = predict_model(model, query, cfg)
                seconds[control_key] = seconds[key]
            del model
            gc.collect()
        for label in [None, *(query_controls or {}).get(name, {})]:
            stem = f"{prefix}_{name}" + (f"_{label}" if label else "")
            direct, residual = (predictions[f"{stem}_{m}"] for m in ("direct", "residual"))
            pair = (direct + residual) / 2
            cost = sum(seconds[f"{stem}_{m}"] for m in ("direct", "residual"))
            predictions[f"{stem}_normal_view"] = pair
            predictions[f"{stem}_replace_seed"] = baseline + (
                direct - components[("local_trend_direct", seed)][:len(b)] +
                residual - components[("local_trend_residual", seed)][:len(b)]) / 8
            # Fixed blend checks complementarity, not an equal-capacity comparison.
            predictions[f"{stem}_blend25"] = .75 * baseline + .25 * pair
            for suffix in ("normal_view", "replace_seed", "blend25"):
                seconds[f"{stem}_{suffix}"] = cost
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
