"""Leakage-safe feature construction for the HDB rental project."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATE_COLUMN = "RENT_APPROVAL_DATE"
EARTH_RADIUS_KM = 6371.0088
JOIN_KEYS = ["_TOWN_KEY", "_BLOCK_KEY", "_STREET_KEY"]
BLOCK_FEATURES = [
    "POSTAL_CODE",
    "LATITUDE",
    "LONGITUDE",
    "MAX_FLOOR",
    "YEAR_COMPLETED",
    "SUBZONE",
    "PLANNING_AREA",
    "REGION",
]
FACILITY_FEATURE_SETS = {
    "block_mrt": ["MRT"],
    "block_mall": ["MALL"],
    "block_school": ["SCHOOL"],
    "all_spatial": ["MRT", "MALL", "SCHOOL"],
}
HISTORY_GROUPS = {
    "ADDRESS": ["_ADDRESS_KEY"],
    "TOWN_FLAT": ["_TOWN_KEY", "_FLAT_TYPE_KEY"],
    "TOWN": ["_TOWN_KEY"],
    "FLAT_TYPE": ["_FLAT_TYPE_KEY"],
    "FLAT_MODEL": ["_FLAT_MODEL_KEY"],
}
DETAILED_HISTORY_GROUPS = {
    **HISTORY_GROUPS,
    "ADDRESS_FLAT": ["_ADDRESS_KEY", "_FLAT_TYPE_KEY"],
    "ADDRESS_FLAT_AREA": ["_ADDRESS_KEY", "_FLAT_TYPE_KEY", "_AREA_KEY"],
    "STREET_FLAT": ["_STREET_KEY", "_FLAT_TYPE_KEY"],
    "TOWN_FLAT_MODEL": ["_TOWN_KEY", "_FLAT_TYPE_KEY", "_FLAT_MODEL_KEY"],
}
CATEGORY_COLUMNS = ["TOWN", "BLOCK", "STREET", "FLAT_TYPE", "FLAT_MODEL"]


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize join keys while preserving the raw input tables."""
    return (
        series.astype("string")
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )


def make_core_features(
    frame: pd.DataFrame, canonical_categories: bool = True
) -> pd.DataFrame:
    """Create date, age, and address features available at prediction time."""
    result = frame.copy()
    dates = pd.to_datetime(result.pop(DATE_COLUMN), format="%Y-%m", errors="raise")
    result["RENT_YEAR"] = dates.dt.year.astype("int16")
    result["RENT_MONTH"] = dates.dt.month.astype("int8")
    result["RENT_QUARTER"] = dates.dt.quarter.astype("int8")
    result["MONTH_INDEX"] = (
        (dates.dt.year - 2021) * 12 + dates.dt.month - 1
    ).astype("int16")
    result["LEASE_AGE"] = (
        result["RENT_YEAR"]
        - pd.to_numeric(result["LEASE_COMMENCE_DATE"], errors="coerce")
    ).clip(lower=0)
    result["ADDRESS"] = (
        normalize_text(result["BLOCK"]).fillna("missing")
        + " "
        + normalize_text(result["STREET"]).fillna("missing")
    )
    if canonical_categories:
        for column in CATEGORY_COLUMNS:
            if column in result:
                result[column] = normalize_text(result[column]).fillna("missing")
    result = result.drop(columns=["FURNISHED", "FEE"], errors="ignore")
    return result


def suspected_low_rent_mask(
    frame: pd.DataFrame,
    target: pd.Series,
    ratio_threshold: float = 0.5,
) -> np.ndarray:
    """Flag implausibly low labels without changing the source tables.

    A row is flagged only when it represents a 3-room-or-larger flat (or an
    executive flat) and its rent is below ``ratio_threshold`` times the median
    for the same approval month, town, and canonical flat type.  The function
    deliberately uses no future information outside the supplied frame; it is
    intended for an explicit training-label sensitivity analysis, not for
    altering the unlabeled competition test set.
    """
    if not 0.0 < ratio_threshold < 1.0:
        raise ValueError("ratio_threshold must be between zero and one")
    if len(frame) != len(target):
        raise ValueError("Features and target have different lengths")

    comparison = pd.DataFrame(
        {
            "month": frame[DATE_COLUMN].astype("string").reset_index(drop=True),
            "town": normalize_text(frame["TOWN"]).reset_index(drop=True),
            "flat_type": normalize_text(frame["FLAT_TYPE"]).reset_index(drop=True),
            "target": pd.Series(target, dtype="float64").reset_index(drop=True),
        }
    )
    group_median = comparison.groupby(
        ["month", "town", "flat_type"], dropna=False
    )["target"].transform("median")
    room_count = comparison["flat_type"].str.extract(r"(\d+)", expand=False)
    room_count = pd.to_numeric(room_count, errors="coerce")
    eligible = room_count.ge(3) | comparison["flat_type"].str.contains(
        "executive", na=False
    )
    return (
        eligible & comparison["target"].lt(ratio_threshold * group_median)
    ).to_numpy(dtype=bool)


def _history_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return normalized grouping keys used by past-only target statistics."""
    result = pd.DataFrame(index=frame.index)
    result["_TOWN_KEY"] = normalize_text(frame["TOWN"])
    result["_FLAT_TYPE_KEY"] = normalize_text(frame["FLAT_TYPE"])
    result["_FLAT_MODEL_KEY"] = normalize_text(frame["FLAT_MODEL"])
    result["_STREET_KEY"] = normalize_text(frame["STREET"])
    if "FLOOR_AREA_SQM" in frame:
        result["_AREA_KEY"] = (
            pd.to_numeric(frame["FLOOR_AREA_SQM"], errors="coerce")
            .round(1)
            .astype("string")
            .fillna("missing")
        )
    else:
        result["_AREA_KEY"] = "missing"
    result["_ADDRESS_KEY"] = (
        normalize_text(frame["BLOCK"]).fillna("missing")
        + " "
        + normalize_text(frame["STREET"]).fillna("missing")
    )
    return result


def add_past_rent_features(
    reference: pd.DataFrame,
    reference_target: pd.Series,
    query: pd.DataFrame,
    smoothing: float = 20.0,
    detailed: bool = False,
    recent_months: int = 24,
    market_features: bool | None = None,
    include_recent: bool | None = None,
    include_market_adjusted: bool | None = None,
    history_lag: int = 0,
) -> pd.DataFrame:
    """Create target statistics using only reference rows before each query month.

    The function is safe for both training and future prediction.  When
    ``reference`` and ``query`` contain the same rows, observations from the
    query month (including the row's own target) are excluded.
    """
    if history_lag < 0:
        raise ValueError("history_lag must be nonnegative")
    if len(reference) != len(reference_target):
        raise ValueError("Reference features and target have different lengths")
    reference = reference.reset_index(drop=True)
    target = pd.Series(reference_target, dtype="float64").reset_index(drop=True)
    query = query.reset_index(drop=True)
    reference_dates = pd.to_datetime(
        reference[DATE_COLUMN], format="%Y-%m", errors="raise"
    )
    query_dates = pd.to_datetime(query[DATE_COLUMN], format="%Y-%m", errors="raise")
    reference_keys = _history_keys(reference)
    query_keys = _history_keys(query)
    output = pd.DataFrame(index=query.index)

    groups = DETAILED_HISTORY_GROUPS if detailed else HISTORY_GROUPS
    if market_features is None:
        market_features = detailed
    if include_recent is None:
        include_recent = market_features
    if include_market_adjusted is None:
        include_market_adjusted = market_features
    feature_columns = ["HIST_GLOBAL_MEAN", "HIST_GLOBAL_TREND"]
    for name in groups:
        feature_columns.extend([f"HIST_{name}_MEAN", f"HIST_{name}_LOG_COUNT"])
        if include_recent:
            feature_columns.extend(
                [
                    f"HIST_{name}_RECENT_MEAN",
                    f"HIST_{name}_RECENT_MEDIAN",
                    f"HIST_{name}_RECENT_LOG_COUNT",
                ]
            )
        if include_market_adjusted:
            feature_columns.append(f"HIST_{name}_MARKET_ADJUSTED")
    for column in feature_columns:
        output[column] = np.nan

    for month in sorted(query_dates.unique()):
        query_mask = query_dates.eq(month).to_numpy()
        cutoff = month - pd.DateOffset(months=history_lag)
        prior_mask = reference_dates.lt(cutoff).to_numpy()
        if not prior_mask.any():
            for name in groups:
                output.loc[query_mask, f"HIST_{name}_LOG_COUNT"] = 0.0
                if include_recent:
                    output.loc[query_mask, f"HIST_{name}_RECENT_LOG_COUNT"] = 0.0
            continue

        prior_target = target.loc[prior_mask]
        global_mean = float(prior_target.mean())
        output.loc[query_mask, "HIST_GLOBAL_MEAN"] = global_mean

        monthly = (
            pd.DataFrame(
                {
                    "month": reference_dates.loc[prior_mask].to_numpy(),
                    "target": prior_target.to_numpy(),
                }
            )
            .groupby("month", as_index=False)["target"]
            .mean()
            .tail(24)
        )
        if len(monthly) >= 2:
            x = (
                monthly["month"].dt.year * 12 + monthly["month"].dt.month
            ).to_numpy(float)
            query_x = float(month.year * 12 + month.month)
            slope, intercept = np.polyfit(x, monthly["target"].to_numpy(float), 1)
            output.loc[query_mask, "HIST_GLOBAL_TREND"] = intercept + slope * query_x
        else:
            output.loc[query_mask, "HIST_GLOBAL_TREND"] = global_mean

        prior = reference_keys.loc[prior_mask].copy()
        prior["_TARGET"] = prior_target.to_numpy()
        if include_market_adjusted:
            reference_month_means = target.groupby(reference_dates).transform("mean")
            prior["_PREMIUM"] = (
                target.loc[prior_mask].to_numpy()
                - reference_month_means.loc[prior_mask].to_numpy()
            )
        if include_recent:
            recent_start = month - pd.DateOffset(months=recent_months)
            recent_mask = prior_mask & reference_dates.ge(recent_start).to_numpy()
            recent = reference_keys.loc[recent_mask].copy()
            recent["_TARGET"] = target.loc[recent_mask].to_numpy()
            recent_global = float(target.loc[recent_mask].mean())
        for name, keys in groups.items():
            stats = prior.groupby(keys, dropna=False)["_TARGET"].agg(["sum", "count"])
            smoothed = (stats["sum"] + smoothing * global_mean) / (
                stats["count"] + smoothing
            )
            current = query_keys.loc[query_mask, keys]
            if len(keys) == 1:
                lookup_key = current[keys[0]]
            else:
                lookup_key = pd.MultiIndex.from_frame(current)
            output.loc[query_mask, f"HIST_{name}_MEAN"] = smoothed.reindex(
                lookup_key
            ).to_numpy()
            counts = stats["count"].reindex(lookup_key).fillna(0).to_numpy(float)
            output.loc[query_mask, f"HIST_{name}_LOG_COUNT"] = np.log1p(counts)
            if include_recent:
                recent_stats = recent.groupby(keys, dropna=False)["_TARGET"].agg(
                    ["sum", "count", "median"]
                )
                recent_smoothed = (
                    recent_stats["sum"] + smoothing * recent_global
                ) / (recent_stats["count"] + smoothing)
                output.loc[query_mask, f"HIST_{name}_RECENT_MEAN"] = (
                    recent_smoothed.reindex(lookup_key).to_numpy()
                )
                output.loc[query_mask, f"HIST_{name}_RECENT_MEDIAN"] = (
                    recent_stats["median"].reindex(lookup_key).to_numpy()
                )
                recent_counts = (
                    recent_stats["count"].reindex(lookup_key).fillna(0).to_numpy(float)
                )
                output.loc[query_mask, f"HIST_{name}_RECENT_LOG_COUNT"] = np.log1p(
                    recent_counts
                )
            if include_market_adjusted:
                premium_stats = prior.groupby(keys, dropna=False)["_PREMIUM"].agg(
                    ["sum", "count"]
                )
                premium = premium_stats["sum"] / (premium_stats["count"] + smoothing)
                output.loc[query_mask, f"HIST_{name}_MARKET_ADJUSTED"] = (
                    output.loc[query_mask, "HIST_GLOBAL_TREND"].to_numpy()
                    + premium.reindex(lookup_key).to_numpy()
                )

    return pd.concat([query.copy(), output], axis=1)


def add_adaptive_rent_features(
    reference, target, query, half_life=12.0, history_lag=0,
    include_local_trend=False, composition_control=False, trend_only_control=False,
):
    """Past-only decayed comparables with hierarchical fallbacks and damped trend.

    Weights are anchored to the last available reference month, not the query
    month: forecasts 16 months ahead do not lose all their comparables. Ratios
    remove the historical flat-type market level before pooling local premiums.
    No query-month label contributes to either its market index or premium.
    """
    if len(reference) != len(target) or half_life <= 0 or history_lag < 0:
        raise ValueError("Invalid reference target length or half life")
    reference = reference.reset_index(drop=True)
    query = query.reset_index(drop=True)
    keys = _history_keys(reference)
    qkeys = _history_keys(query)
    dates = pd.to_datetime(reference[DATE_COLUMN])
    qdates = pd.to_datetime(query[DATE_COLUMN])
    keys["_MONTH"] = dates.dt.year * 12 + dates.dt.month
    keys["_Y"] = np.asarray(target, dtype=float)
    groups = [
        ("TOWN_FLAT", ["_TOWN_KEY", "_FLAT_TYPE_KEY"], None, 30.0),
        ("STREET_FLAT", ["_TOWN_KEY", "_STREET_KEY", "_FLAT_TYPE_KEY"], "TOWN_FLAT", 12.0),
        ("ADDRESS_FLAT", ["_ADDRESS_KEY", "_FLAT_TYPE_KEY"], "STREET_FLAT", 8.0),
        ("ADDRESS_FLAT_AREA", ["_ADDRESS_KEY", "_FLAT_TYPE_KEY", "_AREA_KEY"], "ADDRESS_FLAT", 5.0),
    ]
    out = pd.DataFrame(index=query.index, dtype=float)
    out["ADAPT_MARKET_LEVEL"] = 2000.0  # Fixed cold-start prior, not fitted globally.
    out["ADAPT_HORIZON"] = 0.0
    if include_local_trend:
        out["ADAPT_TOWN_FLAT_PREMIUM_RECENT"] = 1.0
        out["ADAPT_TOWN_FLAT_PREMIUM_PREVIOUS"] = 1.0
        out["ADAPT_TOWN_FLAT_PREMIUM_CHANGE"] = 0.0
        out["ADAPT_TOWN_FLAT_TREND_LOG_SUPPORT"] = 0.0
    for name, _, _, _ in groups:
        out[f"ADAPT_{name}_LEVEL"] = 2000.0
        out[f"ADAPT_{name}_PREMIUM"] = 1.0
        out[f"ADAPT_{name}_LOG_MASS"] = 0.0
        out[f"ADAPT_{name}_AGE"] = 60.0
    for month in sorted(qdates.unique()):
        mask = qdates.eq(month).to_numpy()
        prior = keys.loc[dates.lt(month - pd.DateOffset(months=history_lag))].copy()
        if prior.empty:
            continue
        query_month = month.year * 12 + month.month
        anchor = int(prior["_MONTH"].max())
        horizon = query_month - anchor
        out.loc[mask, "ADAPT_HORIZON"] = horizon
        monthly = prior.groupby(["_FLAT_TYPE_KEY", "_MONTH"])["_Y"].agg(["mean", "count"])
        level = {}
        for flat, table in monthly.groupby(level=0):
            table = table.droplevel(0).tail(12)
            last = table.tail(3)
            anchor_level = float(np.average(last["mean"], weights=last["count"]))
            slope = 0.0
            if len(table) >= 6:
                slope = float(np.polyfit(table.index.to_numpy(float), np.log(table["mean"]), 1)[0])
            # Forecast saturation limits extrapolation; no validation-tuned coefficients.
            slope = np.clip(slope, -0.01, 0.01)
            damped_horizon = 6.0 * (1.0 - np.exp(-horizon / 6.0))
            level[flat] = anchor_level * np.exp(slope * damped_horizon)
        flat_level = qkeys.loc[mask, "_FLAT_TYPE_KEY"].map(level).fillna(prior["_Y"].mean()).to_numpy()
        out.loc[mask, "ADAPT_MARKET_LEVEL"] = flat_level
        flat_month = prior.groupby(["_FLAT_TYPE_KEY", "_MONTH"])["_Y"].transform("mean")
        denom = flat_month
        if composition_control:
            source = reference.loc[prior.index]
            area = pd.to_numeric(source["FLOOR_AREA_SQM"], errors="coerce")
            lease = pd.to_numeric(source["LEASE_COMMENCE_DATE"], errors="coerce")
            transaction_year = dates.loc[prior.index].dt.year
            prior["_COMP_AREA"] = (np.floor(area / 10.0) * 10.0).fillna(-1)
            prior["_COMP_AGE"] = (
                np.floor((transaction_year - lease).clip(lower=0) / 10.0) * 10.0
            ).fillna(-1)
            comp_cols = ["_FLAT_TYPE_KEY", "_MONTH", "_COMP_AREA", "_COMP_AGE"]
            composition_groups = prior.groupby(comp_cols)["_Y"]
            comp_sum = composition_groups.transform("sum")
            comp_count = composition_groups.transform("count")
            denom = (comp_sum + 8.0 * flat_month) / (comp_count + 8.0)
        # The isolated ablation changes only the trend numerator/denominator.
        # Preserve all four historical premium/level pairs and target offsets.
        prior["_TREND_RATIO"] = prior["_Y"] / denom
        prior["_RATIO"] = prior["_Y"] / (flat_month if trend_only_control else denom)
        prior["_W"] = np.exp2((prior["_MONTH"] - anchor) / half_life)
        prior["_WR"] = prior["_W"] * prior["_RATIO"]
        if include_local_trend:
            # Compare the latest six months with the preceding six months after
            # removing each flat type's contemporaneous market level. Counts
            # shrink sparse town/flat cells toward no local premium (ratio 1).
            trend_cols = ["_TOWN_KEY", "_FLAT_TYPE_KEY"]
            current = qkeys.loc[mask, trend_cols]
            lookup_key = pd.MultiIndex.from_frame(current)

            def period_ratio(start, stop):
                period = prior.loc[prior["_MONTH"].between(start, stop)]
                stats = period.groupby(trend_cols)["_TREND_RATIO"].agg(["sum", "count"])
                count = stats["count"].reindex(lookup_key).fillna(0).to_numpy(float)
                total = stats["sum"].reindex(lookup_key).fillna(0).to_numpy(float)
                ratio = (total + 20.0) / (count + 20.0)
                return ratio, count

            recent_ratio, recent_count = period_ratio(anchor - 5, anchor)
            previous_ratio, previous_count = period_ratio(anchor - 11, anchor - 6)
            premium_change = np.log(
                np.maximum(recent_ratio, 1e-6) / np.maximum(previous_ratio, 1e-6)
            ).clip(-0.15, 0.15)
            out.loc[mask, "ADAPT_TOWN_FLAT_PREMIUM_RECENT"] = recent_ratio
            out.loc[mask, "ADAPT_TOWN_FLAT_PREMIUM_PREVIOUS"] = previous_ratio
            out.loc[mask, "ADAPT_TOWN_FLAT_PREMIUM_CHANGE"] = premium_change
            out.loc[mask, "ADAPT_TOWN_FLAT_TREND_LOG_SUPPORT"] = np.log1p(
                np.minimum(recent_count, previous_count)
            )
        estimates = {}
        for name, cols, parent, strength in groups:
            stats = prior.groupby(cols).agg(mass=("_W", "sum"), total=("_WR", "sum"), last=("_MONTH", "max"))
            matched = stats.reindex(pd.MultiIndex.from_frame(qkeys.loc[mask, cols]))
            mass = matched["mass"].fillna(0).to_numpy()
            total = matched["total"].fillna(0).to_numpy()
            parent_ratio = estimates[parent] if parent else np.ones(mask.sum())
            ratio = (total + strength * parent_ratio) / (mass + strength)
            estimates[name] = ratio
            out.loc[mask, f"ADAPT_{name}_PREMIUM"] = ratio
            out.loc[mask, f"ADAPT_{name}_LEVEL"] = flat_level * ratio
            out.loc[mask, f"ADAPT_{name}_LOG_MASS"] = np.log1p(mass)
            out.loc[mask, f"ADAPT_{name}_AGE"] = query_month - matched["last"].fillna(query_month - 60).to_numpy()
    return out


def _haversine_matrix(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
    """Return nearest distance/index and amenity counts within fixed radii."""
    target_lat_rad = np.radians(target_lat)
    target_lon_rad = np.radians(target_lon)
    nearest_distance = np.empty(len(source_lat), dtype="float64")
    nearest_index = np.empty(len(source_lat), dtype="int32")
    counts = {
        radius: np.empty(len(source_lat), dtype="int16")
        for radius in (0.5, 1.0, 2.0)
    }
    for start in range(0, len(source_lat), chunk_size):
        stop = min(start + chunk_size, len(source_lat))
        lat1 = np.radians(source_lat[start:stop])[:, None]
        lon1 = np.radians(source_lon[start:stop])[:, None]
        dlat = target_lat_rad[None, :] - lat1
        dlon = target_lon_rad[None, :] - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(target_lat_rad)[None, :] * np.sin(dlon / 2) ** 2
        distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        local_index = np.argmin(distance, axis=1)
        nearest_distance[start:stop] = distance[np.arange(stop - start), local_index]
        nearest_index[start:stop] = local_index
        for radius in counts:
            counts[radius][start:stop] = (distance <= radius).sum(axis=1)
    return nearest_distance, nearest_index, counts


def _facility_features(
    blocks: pd.DataFrame, facilities: pd.DataFrame, prefix: str
) -> pd.DataFrame:
    nearest_distance, nearest_index, counts = _haversine_matrix(
        blocks["LATITUDE"].to_numpy(float),
        blocks["LONGITUDE"].to_numpy(float),
        facilities["LATITUDE"].to_numpy(float),
        facilities["LONGITUDE"].to_numpy(float),
    )
    result = pd.DataFrame(index=blocks.index)
    result[f"DIST_{prefix}_KM"] = nearest_distance
    if "NAME" in facilities:
        result[f"NEAREST_{prefix}"] = facilities["NAME"].astype("string").iloc[nearest_index].to_numpy()
    for radius, values in counts.items():
        suffix = str(radius).replace(".", "P")
        result[f"COUNT_{prefix}_{suffix}KM"] = values
    return result


def build_block_lookup(auxiliary_dir: Path) -> pd.DataFrame:
    """Build one row per HDB block with block and nearest-facility features."""
    blocks = pd.read_csv(auxiliary_dir / "sg-hdb-block.csv")
    for column in ("TOWN", "BLOCK", "STREET"):
        blocks[f"_{column}_KEY"] = normalize_text(blocks[column])
    if blocks.duplicated(JOIN_KEYS).any():
        raise ValueError("HDB block lookup has duplicate normalized address keys")

    facilities = {
        "MRT": pd.read_csv(auxiliary_dir / "sg-mrt-stations.csv"),
        "MALL": pd.read_csv(auxiliary_dir / "sg-shopping-malls.csv"),
        "SCHOOL": pd.read_csv(auxiliary_dir / "sg-schools.csv"),
    }
    open_mrt = facilities["MRT"].loc[
        facilities["MRT"]["STATUS"].astype("string").str.lower().eq("open")
    ].reset_index(drop=True)
    facilities["MRT"] = open_mrt
    for prefix, table in facilities.items():
        additions = _facility_features(blocks, table, prefix)
        blocks = pd.concat([blocks, additions], axis=1)
    return blocks


def spatial_columns(block_lookup: pd.DataFrame, feature_set: str) -> list[str]:
    """Return an explicit, auditable column list for a spatial ablation."""
    if feature_set == "core":
        return []
    if feature_set == "block":
        return BLOCK_FEATURES.copy()
    if feature_set not in FACILITY_FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    columns = BLOCK_FEATURES.copy()
    for prefix in FACILITY_FEATURE_SETS[feature_set]:
        columns.extend(column for column in block_lookup if f"_{prefix}" in column)
    return columns


def add_spatial_features(
    frame: pd.DataFrame,
    block_lookup: pd.DataFrame,
    feature_set: str = "all_spatial",
) -> pd.DataFrame:
    """Join one selected block/amenity feature group without target information."""
    if feature_set == "core":
        return frame.copy()
    result = frame.copy()
    for column in ("TOWN", "BLOCK", "STREET"):
        result[f"_{column}_KEY"] = normalize_text(result[column])
    useful = spatial_columns(block_lookup, feature_set)
    result = result.merge(
        block_lookup[JOIN_KEYS + useful],
        on=JOIN_KEYS,
        how="left",
        validate="many_to_one",
    )
    if result["LATITUDE"].isna().any():
        raise ValueError("Some rental rows did not match the HDB block lookup")
    return result.drop(columns=JOIN_KEYS)


def prepare_for_catboost(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas categorical columns into CatBoost-safe strings."""
    result = frame.copy()
    for column in result.select_dtypes(include=["object", "string", "category"]):
        result[column] = result[column].astype("string").fillna("missing").astype(str)
    return result
