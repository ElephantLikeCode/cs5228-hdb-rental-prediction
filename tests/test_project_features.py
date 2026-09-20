from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_past_rent_features,
    add_spatial_features,
    build_block_lookup,
    make_core_features,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def test_core_and_spatial_features_have_full_coverage() -> None:
    train = pd.read_csv(RAW / "train.csv", nrows=200)
    core = make_core_features(train.drop(columns=["MONTHLY_RENT"]))
    lookup = build_block_lookup(RAW / "auxiliary")
    spatial = add_spatial_features(core, lookup)
    expected = {
        "RENT_YEAR", "RENT_MONTH", "RENT_QUARTER", "MONTH_INDEX", "LEASE_AGE",
        "ADDRESS", "LATITUDE", "LONGITUDE", "DIST_MRT_KM", "DIST_MALL_KM",
        "DIST_SCHOOL_KM", "COUNT_MRT_1P0KM", "COUNT_MALL_1P0KM",
        "COUNT_SCHOOL_1P0KM",
    }
    assert expected.issubset(spatial.columns)
    assert spatial[list(expected)].notna().all().all()
    assert len(spatial) == len(train)


def test_core_features_canonicalize_equivalent_flat_types() -> None:
    frame = pd.DataFrame(
        {
            "RENT_APPROVAL_DATE": ["2021-01", "2022-01"],
            "TOWN": ["ANG MO KIO", "ang mo kio"],
            "BLOCK": ["1", "1"],
            "STREET": ["TEST ROAD", "test road"],
            "FLAT_TYPE": ["3 room", "3-room"],
            "FLAT_MODEL": ["Model A", "model-a"],
            "FLOOR_AREA_SQM": [70, 70],
            "LEASE_COMMENCE_DATE": [1980, 1980],
            "FURNISHED": ["yes", "yes"],
            "FEE": [0, 0],
        }
    )
    features = make_core_features(frame)
    assert features["FLAT_TYPE"].nunique() == 1
    assert features["FLAT_MODEL"].nunique() == 1
    assert features["TOWN"].nunique() == 1


def test_spatial_feature_groups_are_strictly_additive() -> None:
    train = pd.read_csv(RAW / "train.csv", nrows=20)
    core = make_core_features(train.drop(columns=["MONTHLY_RENT"]))
    lookup = build_block_lookup(RAW / "auxiliary")
    block = add_spatial_features(core, lookup, "block")
    mrt = add_spatial_features(core, lookup, "block_mrt")
    all_spatial = add_spatial_features(core, lookup, "all_spatial")
    assert "LATITUDE" in block and "DIST_MRT_KM" not in block
    assert "DIST_MRT_KM" in mrt and "DIST_MALL_KM" not in mrt
    assert {"DIST_MRT_KM", "DIST_MALL_KM", "DIST_SCHOOL_KM"}.issubset(all_spatial)
    assert len(core) == len(block) == len(mrt) == len(all_spatial)


def test_past_rent_features_exclude_same_and_future_months() -> None:
    frame = pd.DataFrame(
        {
            "RENT_APPROVAL_DATE": ["2024-01", "2024-01", "2024-02", "2024-03"],
            "TOWN": ["a", "b", "a", "a"],
            "BLOCK": ["1", "2", "1", "1"],
            "STREET": ["road", "road", "road", "road"],
            "FLAT_TYPE": ["3 room"] * 4,
            "FLAT_MODEL": ["model a"] * 4,
        }
    )
    target = pd.Series([1000.0, 3000.0, 5000.0, 9000.0])
    features = add_past_rent_features(frame, target, frame, smoothing=0.0)
    assert pd.isna(features.loc[0, "HIST_GLOBAL_MEAN"])
    assert features.loc[2, "HIST_GLOBAL_MEAN"] == 2000.0
    assert features.loc[2, "HIST_ADDRESS_MEAN"] == 1000.0
    assert features.loc[2, "HIST_ADDRESS_LOG_COUNT"] == pytest.approx(
        np.log1p(1)
    )
    assert features.loc[3, "HIST_GLOBAL_MEAN"] == 3000.0
    assert features.loc[3, "HIST_ADDRESS_MEAN"] == 3000.0


def test_detailed_history_separates_flat_types_within_address() -> None:
    frame = pd.DataFrame(
        {
            "RENT_APPROVAL_DATE": ["2024-01", "2024-01", "2024-02"],
            "TOWN": ["a", "a", "a"],
            "BLOCK": ["1", "1", "1"],
            "STREET": ["road", "road", "road"],
            "FLAT_TYPE": ["3 room", "5-room", "3-room"],
            "FLAT_MODEL": ["model a"] * 3,
            "FLOOR_AREA_SQM": [70, 110, 70],
        }
    )
    target = pd.Series([2000.0, 4000.0, 2500.0])
    features = add_past_rent_features(
        frame, target, frame, smoothing=0.0, detailed=True
    )
    assert features.loc[2, "HIST_ADDRESS_MEAN"] == 3000.0
    assert features.loc[2, "HIST_ADDRESS_FLAT_MEAN"] == 2000.0
    assert features.loc[2, "HIST_ADDRESS_FLAT_AREA_MEAN"] == 2000.0
