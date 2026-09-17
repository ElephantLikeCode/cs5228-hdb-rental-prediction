from pathlib import Path

import pandas as pd

from src.features import (
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
