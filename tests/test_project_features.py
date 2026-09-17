from pathlib import Path

import pandas as pd

from src.project_features import add_spatial_features, build_block_lookup, make_core_features


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
        "DIST_SCHOOL_KM",
    }
    assert expected.issubset(spatial.columns)
    assert spatial[list(expected)].notna().all().all()
    assert len(spatial) == len(train)
