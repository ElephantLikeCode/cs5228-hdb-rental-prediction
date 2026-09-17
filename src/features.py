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


def normalize_text(series: pd.Series) -> pd.Series:
    """Normalize join keys while preserving the raw input tables."""
    return (
        series.astype("string")
        .str.lower()
        .str.strip()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )


def make_core_features(frame: pd.DataFrame) -> pd.DataFrame:
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
    result = result.drop(columns=["FURNISHED", "FEE"], errors="ignore")
    return result


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
